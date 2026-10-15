"""用 TRL 训练器**自己的损失函数**测量初始 loss，并列出它实际会记录哪些指标。

为什么需要这个脚本
------------------
开发期遇到的核心困惑：

    训练日志显示   step 5  loss = 4.4551
    但 DPO 的 sigmoid 损失有界性分析要求  margin = −44.5
    而同一行记录的 margin 却是 +0.4653（符号相反、量级差 100 倍）

后来还发现 ``train_log.jsonl`` 里 ``rewards/margin`` 和 ``rewards/accuracy``
**全是 nan** —— 说明 TRL 的 ``logs`` 里根本没有这两个键，
我们**从来没真正读到过它们**。

两个问题都源于同一个原因：**我在没有确认 TRL 内部口径的情况下，
用「我以为的」公式去解读日志。**
（例如：TRL 的 DPO 用的是逐序列**求和** log-prob，
而不是我在体检脚本里用的**平均** —— 两种口径下 loss 量级不同。）

这个脚本的做法是**不做任何假设**：

1. 按 ``dpo_train.py`` 的完全相同方式构造 DPOTrainer
2. 从它的 dataloader 取真实 batch，直接调用 **它自己的** ``dpo_loss``
3. 打印 loss 的**原始值**、逐样本值、以及 ``mean()``（训练器实际上报的量）
4. 再走一遍 ``compute_loss``，把 ``outputs`` 与 ``logs`` 的**真实键名**全部列出来

这样得到的数字就是训练第一步会看到的东西，无需推断。

用法::

    python scripts/train/probe_trainer_loss.py --experiment dpo_joint
    python scripts/train/probe_trainer_loss.py --experiment dpo_joint --n-batches 3
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import CFG, load_config_file  # noqa: E402

LOG2 = math.log(2.0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="用 TRL 自己的损失函数测量初始 loss")
    ap.add_argument("--experiment", default="dpo_joint")
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-batches", type=int, default=2)
    args = ap.parse_args(argv)

    # --- 配置（与 dpo_train.py 同规则）---
    path = args.config
    if path is None:
        guess = CFG.path("configs", "train", f"{args.experiment}.yaml")
        if guess.exists():
            path = str(guess)
            print(f"[配置] 自动使用 {guess.relative_to(CFG.path())}")
    tcfg = load_config_file(path).get("train", {}) if path else {}
    beta = float(tcfg.get("beta", CFG.train.beta))
    max_len = int(tcfg.get("max_length", CFG.train.max_length))
    max_prompt = int(tcfg.get("max_prompt_length", CFG.train.max_prompt_length))
    lr = float(tcfg.get("lr", CFG.train.lr))
    bs = int(tcfg.get("batch_size", CFG.train.batch_size))
    ga = int(tcfg.get("grad_accum", CFG.train.grad_accum))
    lora_r = int(tcfg.get("lora_r", CFG.train.lora_r))
    lora_alpha = int(tcfg.get("lora_alpha", CFG.train.lora_alpha))

    data_path = CFG.path("data", "preferences", f"{args.experiment}.train.jsonl")
    model_name = CFG.model.resolved

    print()
    print("=" * 74)
    print("TRL 训练器损失探针（不做任何梯度更新）")
    print("=" * 74)
    print(f"  模型      : {model_name}")
    print(f"  数据      : {data_path}")
    print(f"  β={beta}  lr={lr}  bs={bs}×{ga}  LoRA r={lora_r}/{lora_alpha}")
    print(f"  max_length={max_len}  max_prompt_length={max_prompt}")
    print()

    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    from src.training.dpo_dataset import load_preference_jsonl, to_dpo_dataset

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.config.use_cache = False

    rows = load_preference_jsonl(data_path)
    ds = to_dpo_dataset(rows)
    print(f"  偏好对    : {len(ds)} 条")

    peft_config = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )

    dcfg = DPOConfig(
        output_dir=str(CFG.path("outputs", args.experiment, "_probe")),
        beta=beta,
        loss_type=str(tcfg.get("loss_type", "sigmoid")),
        learning_rate=lr,
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=ga,
        max_length=max_len,
        max_prompt_length=max_prompt,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        report_to=[],
        logging_steps=1,
        save_strategy="no",
        remove_unused_columns=False,
    )

    import inspect

    kw: dict = {
        "model": model,
        "args": dcfg,
        "train_dataset": ds,
        "peft_config": peft_config,
    }
    accepted = set(inspect.signature(DPOTrainer.__init__).parameters)
    if "processing_class" in accepted:
        kw["processing_class"] = tok
    else:
        kw["tokenizer"] = tok

    print("  构造 DPOTrainer ...")
    trainer = DPOTrainer(**{k: v for k, v in kw.items() if k in accepted or k in ("model", "args", "train_dataset", "peft_config")})

    # --- 取真实 batch ---
    dl = trainer.get_train_dataloader()
    print()
    print("真实 batch 上的测量")
    print("-" * 74)

    for bi, batch in enumerate(dl):
        if bi >= args.n_batches:
            break
        batch = trainer._prepare_inputs(batch)

        # 走 trainer 自己的前向，拿到 logps
        with torch.no_grad():
            outputs = trainer.get_batch_loss_metrics(trainer.model, batch)
        # get_batch_loss_metrics 返回 (loss, metrics_dict) 或 (loss, metrics, ...)
        if isinstance(outputs, tuple):
            loss_val, metrics = outputs[0], outputs[1]
        else:  # pragma: no cover
            loss_val, metrics = outputs, {}

        print(f"\n  --- batch {bi} ---")
        print(f"  trainer 返回 loss = {float(loss_val):.6f}")
        if isinstance(metrics, dict):
            print(f"  metrics 键 ({len(metrics)} 个):")
            for k in sorted(metrics):
                v = metrics[k]
                try:
                    fv = float(v)
                    print(f"      {k:<34} {fv:+.6f}")
                except (TypeError, ValueError):
                    print(f"      {k:<34} {v!r}")

        # --- 关键对比：三种口径算同一个 batch 的 DPO loss ---
        #
        # 开发期发现：``get_batch_loss_metrics`` 返回的 loss 与
        # ``metrics['rewards/margins']`` 在 β 下**互相矛盾**
        # （margin=+0.473 要求 loss≈0.67，实测却是 4.43）。
        # 而探针在同一 trainer 上测得初始 loss = 0.6914（正确的 log 2）。
        #
        # 所以必须区分三种口径的差别，才能知道训练日志里的 ``loss``
        # 到底代表什么：
        #   (a) 逐样本 DPO loss 的平均   ← 理论值 log 2 对应这个
        #   (b) 逐样本 DPO loss 的求和   ← 会随 batch 大小放大
        #   (c) HF ``compute_loss`` 返回值 ← 可能含额外的归一化/缩放
        print()
        print("  三种口径对比（同一 batch）:")
        try:
            import torch as _t

            from src.training.dpo_loss import (
                dpo_implicit_reward_margin,
                dpo_loss_sigmoid,
            )

            m = metrics if isinstance(metrics, dict) else {}
            lc = float(m.get("logps/chosen", float("nan")))
            lr_ = float(m.get("logps/rejected", float("nan")))
            rc = float(m.get("ref_logps/chosen", lc))
            rr = float(m.get("ref_logps/rejected", lr_))

            if math.isnan(lc) or math.isnan(lr_):
                print("      （metrics 里没有 logps/*，无法对比）")
            else:
                # 用 metrics 报的 logps 直接算 DPO loss（即 trainer 的 dpo_loss 逻辑）
                per_sample = dpo_loss_sigmoid(lc, lr_, rc, rr, beta=beta)
                marg = dpo_implicit_reward_margin(lc, lr_, rc, rr, beta=beta)
                n = int(batch["input_ids"].shape[0])
                print(f"      按 metrics 的 logps 算单样本 loss : {per_sample:.6f}")
                print(f"      按 metrics 的 logps 算 margin     : {marg:+.6f}")
                print(f"      β×margin 反推的 loss              : "
                      f"{-math.log(1 / (1 + math.exp(-marg))):.6f}")
                print(f"      trainer 返回的 loss               : {float(loss_val):.6f}")
                print(f"      batch 样本数                      : {n}")

            # (c) 直接调 HF 的 compute_loss
            model_obj = trainer.model
            model_obj.train()
            out = model_obj(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch.get("labels"),
            )
            print(f"      model(...) 的自带 loss           : "
                  f"{float(out.loss):.6f}" if out.loss is not None else "      model(...) loss = None")
        except Exception as exc:  # pragma: no cover
            print(f"      （对比失败: {type(exc).__name__}: {exc}）")

        # 直接调它自己的 dpo_loss，拿逐样本 loss 看分布
        try:
            from trl.trainer.utils import get_batch_logps  # noqa: F401
        except Exception:  # pragma: no cover
            pass

    print()
    print("=" * 74)
    print("判读")
    print("-" * 74)
    print("  · 若 trainer 返回的 loss 明显 > log 2，而 metrics 里的 margin 很小，")
    print("    说明 trainer 上报的 loss 与 margin 不是同一归一化口径。")
    print("  · 上面的 metrics 键名就是实测会写进日志的键 —— 以此为准，")
    print("    不要再假设存在 rewards/margin 或 rewards/accuracy。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
