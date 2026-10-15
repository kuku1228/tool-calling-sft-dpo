#!/usr/bin/env python
"""离线重评分：只读 ``predictions.jsonl``，用当前评分器重算全部指标。

## 为什么需要它

评分器（``src/eval/grader.py`` + ``src/eval/equivalence.py``）是所有实验
**共用的口径**。一旦口径修正，**所有既有数字都会变** —— 但模型输出
（``predictions.jsonl``）没变，因此**不需要重跑 GPU**。

开发期真实发生过：``extra_fields`` 没有用工具 schema 做白名单，导致模型
给出 ``cancel_booking.reason``（**schema 必填**）被判成「多给了 schema
未定义的参数」。该误判占全部失败的 **58.8%**，把 ``extra_param`` 抬成
头号失败原因。修掉这个口径后必须重算，否则新旧数字不可比。

## 用法

```bash
# 单个实验
python scripts/eval/regrade.py --label sft_v3

# 批量（把所有实验统一到同一口径上比较）
python scripts/eval/regrade.py --labels baseline_airline,sft_v2,sft_v3,dpo_joint

# 只打印新数字、不覆盖原文件（默认会写 regraded_report.json）
python scripts/eval/regrade.py --label sft_v3 --dry-run
```

## 安全设计

* **绝不覆盖** ``eval_report.json`` / ``error_analysis.json`` ——
  重评分结果写到 ``regraded_report.json``，便于新旧对照。
* 逐条重放，因此**可审计**：每条预测的判据都能复现。
* 与在线评测共用 ``grade()``，不存在第二套口径。
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.task_manifest import load_tasks  # noqa: E402
from src.eval.grader import grade  # noqa: E402


def regrade_label(
    label: str, exp_root: Path, *, dry_run: bool = False, allow_mock: bool = False
) -> dict:
    """对单个实验的 predictions.jsonl 重评分。"""
    d = exp_root / label

    # ⚠️ 文件名探测：mocks 产物一律带 ``.mock`` 后缀（见 mock 隔离机制），
    # 真实产物不带。早期只找 ``predictions.jsonl``，于是对 mock 实验
    # 报「缺 predictions.jsonl」—— 而文件其实叫 ``predictions.mock.jsonl``。
    # 默认优先真实产物；只有 ``--allow-mock`` 时才回退到 mock。
    candidates = [d / "predictions.jsonl"]
    if allow_mock:
        candidates.append(d / "predictions.mock.jsonl")
    pred_path = next((c for c in candidates if c.exists()), None)
    if pred_path is None:
        found = [c.name for c in d.glob("predictions*.jsonl")] if d.exists() else []
        hint = f"（该目录下找到: {found}）" if found else "（目录不存在或为空）"
        return {"label": label, "error": f"缺 predictions.jsonl {hint}"}

    # 拒绝把 mock 数字当作真实结果汇报 —— 这是项目的硬约束
    if ".mock." in pred_path.name and not allow_mock:
        return {
            "label": label,
            "error": "这是 mock 产物，真实结果不得引用 mock 数字（加 --allow-mock 可强制）",
        }

    tasks = {t.task_id: t for t in load_tasks()}
    rows = [
        json.loads(line)
        for line in pred_path.open(encoding="utf-8")
        if line.strip()
    ]

    per_task: dict[str, list[bool]] = collections.defaultdict(list)
    counts: collections.Counter[str] = collections.Counter()
    subtypes: collections.Counter[str] = collections.Counter()
    fmt_ok = 0
    fmt_ok_value_wrong = 0
    missing_task = 0

    for r in rows:
        task = tasks.get(r.get("task_id", ""))
        if task is None:
            missing_task += 1
            continue
        g = grade(task, r.get("text", ""))
        ok = bool(g.passed)
        per_task[task.task_id].append(ok)
        counts[g.classification.category] += 1
        # 子类型只在失败时统计（成功时 sub_type 表示「未触发」，会误导）
        if not ok:
            st = getattr(g.classification.detail, "subtype", None)
            if st:
                subtypes[st] += 1
        if g.format_ok:
            fmt_ok += 1
            if not ok:
                fmt_ok_value_wrong += 1

    n_samples = sum(len(v) for v in per_task.values())
    n_task = len(per_task)
    n_success = sum(sum(v) for v in per_task.values())

    # pass@k / pass^k（k = 该任务实际采样数，通常为 4）
    pass_at = 0
    pass_hat = 0
    for oks in per_task.values():
        n, c = len(oks), sum(oks)
        if c > 0:
            pass_at += 1
        if c == n:
            pass_hat += 1

    out = {
        "label": label,
        "num_tasks": n_task,
        "num_samples": n_samples,
        "accuracy": round(n_success / n_samples, 4) if n_samples else 0.0,
        "pass_at_k": round(pass_at / n_task, 4) if n_task else 0.0,
        "pass_hat_k": round(pass_hat / n_task, 4) if n_task else 0.0,
        "format_accuracy": round(fmt_ok / n_samples, 4) if n_samples else 0.0,
        "format_ok_value_wrong": fmt_ok_value_wrong,
        "counts": dict(counts),
        "failure_subtypes": dict(subtypes.most_common()),
        "skipped_missing_task": missing_task,
    }

    if not dry_run:
        (d / "regraded_report.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="离线重评分（不重跑 GPU）")
    p.add_argument("--label", default=None, help="单个实验标签")
    p.add_argument("--labels", default=None, help="逗号分隔的多个标签")
    p.add_argument("--exp-root", default="experiments")
    p.add_argument("--dry-run", action="store_true", help="只打印，不写文件")
    p.add_argument(
        "--allow-mock",
        action="store_true",
        help="允许对 mock 产物重评分（默认拒绝 —— mock 数字不得作为结果汇报）",
    )
    args = p.parse_args(argv)

    labels: list[str] = []
    if args.labels:
        labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    elif args.label:
        labels = [args.label]
    else:
        p.error("需要 --label 或 --labels")

    exp_root = Path(args.exp_root)
    results = []
    print("=" * 78)
    print("离线重评分（评分器口径修正后重算，模型输出未变）")
    print("=" * 78)
    for lab in labels:
        r = regrade_label(
            lab, exp_root, dry_run=args.dry_run, allow_mock=args.allow_mock
        )
        results.append(r)
        if "error" in r:
            print(f"\n### {lab}: {r['error']}")
            continue
        print(f"\n### {r['label']}")
        print(f"  样本          : {r['num_samples']}（{r['num_tasks']} 任务）")
        print(f"  单次准确率     : {r['accuracy']:.2%}")
        print(f"  pass@k        : {r['pass_at_k']:.2%}")
        print(f"  pass^k        : {r['pass_hat_k']:.2%}")
        print(f"  格式正确率     : {r['format_accuracy']:.2%}")
        print(f"  格式对但值错   : {r['format_ok_value_wrong']} 条")
        print(f"  错误类别       : {r['counts']}")
        top = list(r["failure_subtypes"].items())[:6]
        if top:
            print("  失败子类型 Top :")
            for k, v in top:
                print(f"      {k:<26} {v}")

    print()
    print("=" * 78)
    if len(results) > 1:
        print("对照表")
        print("=" * 78)
        print(f"  {'实验':<22} {'accuracy':>10} {'pass^k':>10} {'格式正确率':>12}")
        for r in results:
            if "error" in r:
                continue
            print(
                f"  {r['label']:<22} {r['accuracy']:>9.2%} "
                f"{r['pass_hat_k']:>9.2%} {r['format_accuracy']:>11.2%}"
            )
    if not args.dry_run:
        print()
        print("  已写入各实验目录的 regraded_report.json（原文件未被覆盖）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
