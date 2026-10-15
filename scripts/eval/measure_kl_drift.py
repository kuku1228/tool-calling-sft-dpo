"""轨迹级 KL 漂移测量 —— 替代 ``logps/chosen``。

## 为什么需要它（已实测的判据失效）

DPO 训练日志里的 ``logps/chosen``（chosen 序列的平均对数概率）**测不出
任务能力退化**。实测证据：

    β=0.5 的 DPO：``logps/chosen`` −6.17 → −4.55（**看起来很稳定**）
    同一次训练：任务准确率 83.00% → **49.25%**（掉 33.75pp）

我当时据此判断「训练健康」，而模型已经崩了。原因是
``logps/chosen`` 只反映「模型给 chosen 序列多大概率」，
**对分布往哪儿漂不敏感** —— 它可以上升，而模型同时学会了
在别的地方乱来（实测 ``clarify_instead`` 从 20 涨到 54）。

## 本脚本测什么

对**评测任务集**上的 prompt：

1. 用 **policy**（训练后的模型）生成轨迹
2. 在**同一批轨迹**上分别算 policy 与 reference 的序列对数概率
3. 轨迹级 KL 用**序列级重要性比**：
   ``log(policy) − log(reference)``，逐轨迹统计

## ⚠️ 这个量**不是**数学意义上的精确 KL

精确的 ``KL(policy ‖ reference)`` 需要对 policy 的**整个输出分布**求和，
那不可行。这里用的是蒙特卡洛估计的**对数似然比**：

    Δ(x) = log p_policy(x) − log p_ref(x)，  x 采样自 policy

它的期望 ``E_policy[Δ]`` 正是 ``KL(policy‖reference)`` 的
**单项蒙特卡洛估计**：``E_policy[log(p/q)]``。

**必须成对报告 ``mean(Δ)`` 与 ``std(Δ)``** —— 只看均值会漏掉
「均值没变但方差炸了」这种漂移（真实发生过的事）。所以本脚本同时输出：

* ``mean_delta``（≈ KL 的 MC 估计）
* ``std_delta``、``median_delta``
* ``frac_above``：Δ 超过阈值的轨迹占比 —— **这是最直观的漂移指标**
* ``token_mean``：token 级平均 Δ —— 就是 ``logps/chosen`` 的量级，
  用来**并排展示它为什么漏报**

## 用法

```bash
python -u scripts/eval/measure_kl_drift.py \
    --policy outputs/dpo_sft_v3_b05/final \
    --reference outputs/sft_v3/final \
    --label dpo_b05  --num-samples 2 \
    2>&1 | tee kl_drift.log
```

**需要 GPU**（要加载两个模型）。本地仅做静态检查。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.data.task_manifest import load_tasks_by_version  # noqa: E402


def sequence_logprob(model, tokenizer, prompt_ids, completion_ids):
    """返回 ``completion_ids`` 在给定 prompt 条件下的**序列对数概率**。

    只对 completion 部分求和（prompt 是条件，不是被建模的对象）——
    把 prompt 也算进去会让长 system prompt 淹没差异，这是早期
    ``logps/chosen`` 判据失效的原因之一。
    """
    import torch

    full = torch.cat([prompt_ids, completion_ids], dim=1).to(model.device)
    with torch.no_grad():
        out = model(full)
    logits = out.logits[:, :-1, :]           # 预测下一个 token
    targets = full[:, 1:]                     # 真实下一个 token
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    gathered = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    n_prompt = prompt_ids.shape[1]
    # completion 的第一个 token 由 prompt 最后一个位置预测
    completion_lp = gathered[:, n_prompt - 1:]
    return float(completion_lp.sum().item()), int(completion_lp.shape[1])


def render(task) -> str:
    from src.models.prompt_builder import build_plain_prompt

    return build_plain_prompt(task)


def summarize(
    per_task: list[dict],
    *,
    policy_dir: str,
    reference_dir: str,
    task_set: str,
    thresholds: tuple[float, ...] = (0.5, 1.0, 2.0),
) -> dict:
    """把逐轨迹记录汇总成报告（**纯函数，可测**）。

    每个记录需含 ``seq_delta`` 与 ``token_delta``。

    >>> recs = [{"seq_delta": 0.0, "token_delta": 0.0},
    ...         {"seq_delta": 2.0, "token_delta": 0.1}]
    >>> r = summarize(recs, policy_dir="p", reference_dir="q", task_set="v3")
    >>> r["sequence_level"]["mean_delta"]
    1.0
    >>> r["thresholds"]["frac_seq_delta_gt_1.0"]
    0.5
    """
    seqs = [r["seq_delta"] for r in per_task]
    toks = [r["token_delta"] for r in per_task]
    n = len(seqs)
    return {
        "policy": policy_dir,
        "reference": reference_dir,
        "task_set": task_set,
        "num_trajectories": n,
        "sequence_level": {
            "mean_delta": statistics.fmean(seqs) if n else 0.0,
            "std_delta": statistics.pstdev(seqs) if n > 1 else 0.0,
            "median_delta": statistics.median(seqs) if n else 0.0,
            "min_delta": min(seqs) if n else 0.0,
            "max_delta": max(seqs) if n else 0.0,
        },
        "token_level": {
            "mean_delta": statistics.fmean(toks) if n else 0.0,
            "std_delta": statistics.pstdev(toks) if n > 1 else 0.0,
            "median_delta": statistics.median(toks) if n else 0.0,
        },
        "thresholds": {
            f"frac_seq_delta_gt_{t}": (
                sum(1 for x in seqs if x > t) / n if n else 0.0
            )
            for t in thresholds
        },
        "trajectories": per_task,
    }


def measure(
    policy_dir: str,
    reference_dir: str,
    *,
    task_set: str = "v3",
    num_samples: int = 2,
    limit: int | None = None,
    max_new_tokens: int = 256,
) -> dict:
    """跑完整测量，返回可 json 序列化的报告。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tasks = load_tasks_by_version(task_set)
    if limit:
        tasks = tasks[:limit]

    print(f"  加载 tokenizer: {reference_dir}")
    tok = AutoTokenizer.from_pretrained(reference_dir)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print(f"  加载 reference: {reference_dir}")
    ref = AutoModelForCausalLM.from_pretrained(
        reference_dir, torch_dtype=torch.float16, device_map="auto"
    ).eval()
    print(f"  加载 policy   : {policy_dir}")
    pol = AutoModelForCausalLM.from_pretrained(
        policy_dir, torch_dtype=torch.float16, device_map="auto"
    ).eval()

    per_task: list[dict] = []
    for ti, task in enumerate(tasks):
        prompt = render(task)
        enc = tok(prompt, return_tensors="pt")
        pid = enc["input_ids"]
        for s in range(num_samples):
            torch.manual_seed(42 * 1000 + ti * 10 + s)
            with torch.no_grad():
                gen = pol.generate(
                    **{k: v.to(pol.device) for k, v in enc.items()},
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    pad_token_id=tok.pad_token_id,
                )
            cid = gen[:, pid.shape[1]:]
            lp_pol, n_tok = sequence_logprob(pol, tok, pid, cid)
            lp_ref, _ = sequence_logprob(ref, tok, pid, cid)
            delta = lp_pol - lp_ref
            per_task.append({
                "task_id": task.task_id,
                "category": task.category,
                "sample_index": s,
                "n_tokens": n_tok,
                # 逐 token 平均：就是 logps/chosen 的量级
                "token_delta": delta / max(1, n_tok),
                "token_lp_policy": lp_pol / max(1, n_tok),
                "token_lp_reference": lp_ref / max(1, n_tok),
                # 序列级：真正的漂移量
                "seq_delta": delta,
            })
        if (ti + 1) % 10 == 0:
            print(f"    {ti+1}/{len(tasks)} 任务")

    report = summarize(
        per_task,
        policy_dir=policy_dir,
        reference_dir=reference_dir,
        task_set=task_set,
    )
    pol.cpu(); ref.cpu()
    del pol, ref
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def render_markdown(rep: dict) -> str:
    s, t = rep["sequence_level"], rep["token_level"]
    lines = [
        f"## 轨迹级 KL 漂移：{rep['policy']} vs {rep['reference']}",
        "",
        f"任务集 `{rep['task_set']}`，轨迹 {rep['num_trajectories']} 条。",
        "",
        "| 层级 | mean Δ | std Δ | median Δ | 说明 |",
        "|---|---|---|---|---|",
        f"| **序列级** | **{s['mean_delta']:+.3f}** | {s['std_delta']:.3f} | "
        f"{s['median_delta']:+.3f} | E_policy[log p − log q]，KL 的蒙特卡洛估计 |",
        f"| token 级 | {t['mean_delta']:+.3f} | {t['std_delta']:.3f} | "
        f"{t['median_delta']:+.3f} | 就是 `logps/chosen` 的量级（**已知会漏报**） |",
        "",
        "### 为什么不能只看均值",
        "",
        f"- **std Δ = {s['std_delta']:.3f}** —— 若这个值很大，说明漂移是"
        "**不均匀**的：模型只对一部分输入偏离 reference。",
        "- 实测过的失效模式正是这样：`logps/chosen` 均值稳定（−6.17→−4.55），",
        "  而 `clarify_instead` 从 20 涨到 54 —— **均值掩盖了分布的重构**。",
        "",
        "### Δ 的分布（最直观的漂移读数）",
        "",
        "| 阈值 | 超过该值的轨迹占比 |",
        "|---|---|",
    ]
    for k, v in rep["thresholds"].items():
        lines.append(f"| {k.replace('frac_seq_delta_gt_', 'Δ > ')} | {v:.1%} |")
    lines += [
        "",
        f"- 范围：[{s['min_delta']:+.3f}, {s['max_delta']:+.3f}]",
        "",
        "### 判读建议",
        "",
        "* **序列级 mean Δ 明显 > 0** → policy 相对 reference 有实质漂移",
        "* **std Δ 大** → 漂移不均匀，应按类别分别看（见逐轨迹数据）",
        "* **token 级 mean Δ 接近 0 而序列级 mean Δ 很大** → "
        "正是 `logps/chosen` 漏报的情形",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="轨迹级 KL 漂移测量")
    ap.add_argument("--policy", required=True, help="训练后模型目录")
    ap.add_argument("--reference", required=True, help="参考模型目录（通常是 SFT 起点）")
    ap.add_argument("--label", required=True, help="实验标签（用于输出路径）")
    ap.add_argument("--task-set", default="v3", choices=["v1", "v3"])
    ap.add_argument("--num-samples", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--out-dir", default="experiments")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir) / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    rep = measure(
        args.policy, args.reference,
        task_set=args.task_set,
        num_samples=args.num_samples,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
    )
    (out_dir / "kl_drift.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    md = render_markdown(rep)
    (out_dir / "kl_drift.md").write_text(md, encoding="utf-8")
    print()
    print(md)
    print(f"\n  已写入 {out_dir}/kl_drift.json 与 kl_drift.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
