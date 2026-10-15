"""DPO 起点体检：在**训练之前**确认 `π_θ = π_ref`，即初始 loss 应为 log 2。

为什么需要这个脚本
------------------
DPO 有一个罕见的性质：**起点是数学确定的，不需要经验判断。**

    π_θ = π_ref  ⇒  log(π_θ(y_w)/π_ref(y_w)) − log(π_θ(y_l)/π_ref(y_l)) = 0
                 ⇒  z = 0
                 ⇒  loss = −log σ(0) = log 2 = 0.693147…

所以「第一个 optimizer step 之前的 loss」是一个**可以用理论值检验**的量。
而 TRL 的 `Trainer` 默认在 step `logging_steps` 才打印，那时模型已经更新过若干次，
**理论值已经不可用** —— 开发期就踩过这个坑：第一眼看到的 loss 是 4.4551，
远高于 log 2，但已经无法判断是「起点就错」还是「5 步内漂移」。

这个脚本在加载模型与 adapter 之后、**任何梯度更新之前**，直接在若干真实
偏好对上算一次 DPO 损失，并同时报出 chosen / rejected 的参考奖励差。

判读
----
============================  ==========================================
观测                          结论
============================  ==========================================
``loss ≈ 0.6931``             ✓ reference 配置正确，可以开始训练
``loss`` 明显 > 0.6931        ✗ reference 不是 policy 的起点 —— 查
                              ``ref_model`` / ``init_adapter`` / 
                              adapter 是否在算 reference 时被正确禁用
``mean_margin ≈ 0``           同上，且说明两侧在起点无法区分
``mean_margin`` 明显 ≠ 0      起点就偏好某一侧 → 数据或 reference 有问题
============================  ==========================================

用法::

    python scripts/train/check_dpo_start.py --experiment dpo_joint
    python scripts/train/check_dpo_start.py --experiment dpo_joint --n 32
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# --- 让脚本能直接运行（无需安装包）---
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import CFG, load_config_file  # noqa: E402
from src.training.dpo_loss import dpo_implicit_reward_margin, dpo_loss_sigmoid  # noqa: E402

LOG2 = math.log(2.0)


def _resolve_cfg(experiment: str, cfg_arg: str | None) -> tuple[dict, str]:
    """解析训练配置，返回 (train 段, 用到的文件路径描述)。

    与 ``dpo_train.py`` 的推导规则保持一致 —— 不传 ``--config`` 时
    按 ``--experiment`` 推导，避免「静默使用危险默认值」。
    """
    path = cfg_arg
    if path is None:
        guess = CFG.path("configs", "train", f"{experiment}.yaml")
        if guess.exists():
            path = str(guess)
            print(f"[配置] 未指定 --config，自动使用 {guess.relative_to(CFG.path())}")
        else:
            print(f"[配置] 未指定 --config 且 {guess} 不存在 → 使用全局默认值")
    if not path:
        return {}, "(全局默认值)"
    data = load_config_file(path)
    return data.get("train", {}), path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="在训练前确认 DPO 起点 loss 是否为 log 2（π_θ = π_ref）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--experiment", default="dpo_joint")
    ap.add_argument("--config", default=None, help="训练配置 yaml（默认按 experiment 推导）")
    ap.add_argument("--model", default=None, help="基座模型路径")
    ap.add_argument("--init-adapter", default=None, help="先挂载这个 adapter（SFT→DPO 续训）")
    ap.add_argument("--data", default=None, help="偏好数据 jsonl（默认按 experiment 推导）")
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--n", type=int, default=16, help="抽多少个偏好对做检验")
    ap.add_argument("--max-length", type=int, default=None)
    args = ap.parse_args(argv)

    tcfg, cfg_desc = _resolve_cfg(args.experiment, args.config)
    beta = args.beta if args.beta is not None else float(tcfg.get("beta", CFG.train.beta))
    max_len = args.max_length if args.max_length is not None else int(
        tcfg.get("max_length", CFG.train.max_length)
    )

    data_path = Path(args.data) if args.data else CFG.path(
        "data", "preferences", f"{args.experiment}.train.jsonl"
    )
    if not data_path.exists():
        print(f"错误：偏好数据不存在: {data_path}", file=sys.stderr)
        return 1

    model_name = args.model or CFG.model.resolved
    print()
    print("=" * 74)
    print("DPO 起点体检（训练前，无梯度更新）")
    print("=" * 74)
    print(f"  模型        : {model_name}")
    print(f"  初始 adapter: {args.init_adapter or '无（从 base 开始）'}")
    print(f"  配置        : {cfg_desc}")
    print(f"  数据        : {data_path}")
    print(f"  β           : {beta}")
    print(f"  max_length  : {max_len}")
    print(f"  理论起点    : loss = log 2 = {LOG2:.6f}")
    print()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        print("⚠️ 没有可用 CUDA —— 结果仍有效，但会比较慢", file=sys.stderr)

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.config.use_cache = False

    policy = model
    if args.init_adapter:
        from peft import PeftModel

        policy = PeftModel.from_pretrained(model, args.init_adapter)
        print("  已挂载初始 adapter（reference 应为 merge 后的基座）")

    # 读取样本
    rows = []
    with data_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("prompt") and r.get("chosen") and r.get("rejected"):
                rows.append(r)
            if len(rows) >= args.n:
                break
    if not rows:
        print("错误：数据里没有可用的偏好对", file=sys.stderr)
        return 1
    print(f"  抽检样本    : {len(rows)} 条")
    print()

    def _seq_logprob(model_obj, prompt: str, completion: str) -> tuple[float, int]:
        """返回 (completion 的平均 log-prob, token 数)。

        用**平均** log-prob（除以 token 数）而不是求和 —— 这是关键：
        chosen 与 rejected 的长度不同（实测约 100 vs 58 字符），
        用求和会让长的一侧天然得到更负的值，两个量不可比。
        """
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tok(completion, add_special_tokens=False)["input_ids"]
        ids = (p_ids + c_ids)[:max_len]
        if len(ids) <= len(p_ids):
            return float("nan"), 0
        input_ids = torch.tensor([ids], device=model_obj.device)
        with torch.no_grad():
            out = model_obj(input_ids=input_ids)
        logits = out.logits[0]
        # 只对 completion 部分算 log-prob
        start = len(p_ids) - 1
        logp = torch.log_softmax(logits[start:len(ids) - 1].float(), dim=-1)
        tgt = torch.tensor(ids[start + 1:len(ids)], device=model_obj.device)
        vals = logp.gather(1, tgt.unsqueeze(1)).squeeze(1)
        if vals.numel() == 0:
            return float("nan"), 0
        return float(vals.mean()), int(vals.numel())

    losses: list[float] = []
    margins: list[float] = []
    accs: list[float] = []
    n_tok_chosen: list[int] = []
    n_tok_rejected: list[int] = []
    sum_losses: list[float] = []

    # ★ 关键：算 reference 时必须禁用 adapter（与 TRL 的 ref_model=None + PEFT 一致）
    for i, r in enumerate(rows):
        lp_c, nc = _seq_logprob(policy, r["prompt"], r["chosen"])
        lp_r, nr = _seq_logprob(policy, r["prompt"], r["rejected"])
        if math.isnan(lp_c) or math.isnan(lp_r):
            continue

        # reference：有 adapter 时禁用之，无 adapter 时同一模型即 ref
        if args.init_adapter and hasattr(policy, "disable_adapter"):
            with policy.disable_adapter():
                ref_c, _ = _seq_logprob(policy, r["prompt"], r["chosen"])
                ref_r, _ = _seq_logprob(policy, r["prompt"], r["rejected"])
        else:
            ref_c, ref_r = lp_c, lp_r

        # 用项目自己的 DPO 损失实现（已通过 10 项解析梯度自检）
        l = dpo_loss_sigmoid(lp_c, lp_r, ref_c, ref_r, beta=beta)
        margin = dpo_implicit_reward_margin(lp_c, lp_r, ref_c, ref_r, beta=beta)
        acc = 1.0 if margin > 0 else 0.0
        losses.append(l)
        margins.append(margin)
        accs.append(acc)
        n_tok_chosen.append(nc)
        n_tok_rejected.append(nr)

        # 同一批数据用**求和**口径再算一次。TRL 的 DPO 用的是逐序列求和
        # log-prob（见其 dpo_loss 的 chosen_logps 形状说明），
        # 而我们上面用的是平均 —— 两种口径下 loss 的量级不同。
        sum_c = lp_c * nc
        sum_r = lp_r * nr
        sum_losses.append(dpo_loss_sigmoid(sum_c, sum_r, sum_c, sum_r, beta=beta))

    if not losses:
        print("错误：所有样本都算不出 logprob（可能是 prompt 太长）", file=sys.stderr)
        return 1

    mean_loss = sum(losses) / len(losses)
    mean_margin = sum(margins) / len(margins)
    mean_acc = sum(accs) / len(accs)
    mean_nc = sum(n_tok_chosen) / len(n_tok_chosen)
    mean_nr = sum(n_tok_rejected) / len(n_tok_rejected)

    print("观测结果")
    print("-" * 74)
    print(f"  loss（平均）        {mean_loss:.6f}")
    print(f"  理论值 log 2        {LOG2:.6f}")
    print(f"  偏差                {mean_loss - LOG2:+.6f}")
    print(f"  margin（平均）      {mean_margin:+.6f}")
    print(f"  rewards/accuracy    {mean_acc:.4f}")
    print(f"  chosen token 数     {mean_nc:.1f}")
    print(f"  rejected token 数   {mean_nr:.1f}")
    print()
    print("判读")
    print("-" * 74)
    ok = True
    if abs(mean_loss - LOG2) <= 0.05:
        print("  ✓ loss 与 log 2 相符 → reference 配置正确，可以开始训练")
    else:
        ok = False
        print(f"  ✗ loss={mean_loss:.4f} 偏离 log 2={LOG2:.4f} 达 {mean_loss - LOG2:+.4f}")
        print("     这说明 reference 不是 policy 的起点。排查方向：")
        print("       1) ref_model=None + PEFT 时，算 reference 是否真的禁用了 adapter")
        print("       2) 若用了 --init-adapter（SFT→DPO），reference 应是**merge 后的基座**，")
        print("          而不是原始 base —— 否则起点天然偏移")
        print("       3) β 与 loss_type 是否与预期一致")
    if abs(mean_margin) <= 0.5:
        print(f"  ✓ margin={mean_margin:+.4f} 接近 0 → 两侧在起点基本无法区分（符合预期）")
    else:
        ok = False
        print(f"  ⚠️ margin={mean_margin:+.4f} 明显偏离 0 → 起点就偏好某一侧")
    if abs(mean_acc - 0.5) <= 0.15:
        print(f"  ✓ rewards/accuracy={mean_acc:.4f} 接近 0.5 → 起点接近随机（符合预期）")
    else:
        print(f"  ⚠️ rewards/accuracy={mean_acc:.4f} 偏离 0.5")
    print()
    print("=" * 74)
    print("结论:", "✓ 起点正常" if ok else "✗ 起点异常 —— 先修 reference 配置再训练")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
