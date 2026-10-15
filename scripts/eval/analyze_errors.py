"""错误分析：把评测结果变成诊断报告需要的统计、图表与案例。

用法
----
::

    python scripts/eval/analyze_errors.py --label baseline
    python scripts/eval/analyze_errors.py --label baseline --persist-dir experiments/baseline_airline
    python scripts/eval/analyze_errors.py --all        # 分析所有已有评测

产出
----
::

    experiments/<label>/
    ├── error_analysis.json        统计（分类占比、子类型、按难度/类别交叉）
    └── failure_cases/             各类错误的代表案例

    docs/figures/
    ├── error_distribution.png     错误分布饼图 + 子类型柱状图
    ├── success_by_difficulty.png  按难度分组
    └── success_by_visibility.png  按 seen/unseen 分组（泛化）

为什么「按难度/可见性交叉」是必须的
-----------------------------------
整体准确率提升可能完全来自简单任务的改善，而困难任务一点没变。
如果不分难度看，就会把「在简单任务上刷分」误读成「能力提升」。
同理，unseen 上的提升才是真正的泛化证据。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import CFG  # noqa: E402
from src.data.task_manifest import load_tasks  # noqa: E402
from src.eval.error_taxonomy import (  # noqa: E402
    CATEGORY_LABELS_ZH,
    CATEGORY_REMEDIES_ZH,
    FAILURE_CATEGORIES,
    ErrorDistribution,
)
from src.eval.grader import analyze_recovery, grade, read_predictions  # noqa: E402
from src.eval.metrics import EvalReport  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="错误分析")
    p.add_argument("--label", default=None, help="实验标签（对应 experiments/<label>/）")
    p.add_argument("--persist-dir", default=None, help="结果目录（默认按 label 推断）")
    p.add_argument("--all", action="store_true", help="分析所有含 predictions.jsonl 的目录")
    p.add_argument(
        "--task-set",
        default="v3",
        choices=("v1", "v3"),
        help="用哪个任务集解析 task_id（默认 v3）。"
             "⚠️ 选错会让**全部样本**被判成 UNKNOWN：v3 的 task_id 是 taskv2_*，"
             "在 v1 里查不到，症状是「样本数 N / 成功 0 / 错误类别为空」。",
    )
    p.add_argument("--figures-dir", default="docs/figures", help="图表输出目录")
    p.add_argument("--no-figures", action="store_true", help="不出图")
    p.add_argument("--max-exemplars", type=int, default=3, help="每类错误保留几个案例")
    return p.parse_args(argv)


def analyze_one(
    persist_dir: Path,
    *,
    figures_dir: Path,
    make_figures: bool,
    max_exemplars: int,
    task_set: str = "v3",
) -> dict[str, Any]:
    """分析单个结果目录，返回分析结果 dict。"""
    label = persist_dir.name
    pred_path = persist_dir / "predictions.jsonl"
    if not pred_path.exists():
        raise FileNotFoundError(f"找不到 {pred_path}")

    preds = read_predictions(pred_path)
    # 从报告里读 metadata（若有），否则按 label 猜是否 mock
    report_path = persist_dir / "eval_report.json"
    report: dict[str, Any] = {}
    if report_path.exists():
        report = EvalReport.read_json(str(report_path))
    metadata = report.get("metadata", {})
    is_mock = bool(metadata.get("is_mock"))

    from src.data.task_manifest import load_tasks_by_version

    tasks = load_tasks_by_version(task_set)
    by_id = {t.task_id: t for t in tasks}
    if not any(p.task_id in by_id for p in preds):
        # 全部对不上号 = 任务集选错了。直接报错，不要静默产出「成功 0」的假报告。
        raise SystemExit(
            f"predictions 里的 task_id 在任务集 {task_set!r} 里一个都找不到"
            f"（样例 {[p.task_id for p in preds[:3]]}）。\n"
            "  多半是 --task-set 选错了：v3 的 id 形如 taskv2_0001，v1 形如 task_0001。\n"
            "  正确用法："
        )

    dist = ErrorDistribution(max_exemplars=max_exemplars)
    # 交叉统计：难度 × 类别
    cross: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    cross_total: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    # 恢复分析
    recovery_rows: list[dict[str, Any]] = []

    for pred in preds:
        task = by_id.get(pred.task_id)
        if task is None:
            continue
        g = grade(task, pred.text, finish_reason=pred.finish_reason)
        dist.add(
            g.classification,
            task_id=task.task_id,
            meta={
                "category": task.category,
                "difficulty": task.difficulty,
                "visibility": task.visibility,
                "sample_index": pred.sample_index,
            },
        )
        for key in (task.difficulty, task.visibility):
            cross_total[key][g.category] += 1
            if g.category != "SUCCESS":
                cross[key][g.category] += 1
        if task.is_recovery:
            info = analyze_recovery(task, pred.text)
            recovery_rows.append({**info, "passed": g.passed, "category": g.category})

    # --- 恢复相关指标 ---
    #
    # ⚠️ 命名纪律：本节的比率**一律**用「占样本比例」的口径，键名统一带
    # ``_rate`` 后缀且用 ``recovered_rate`` / ``repeat_rate``。
    #
    # 不在这里产出权威的 ``recovery_rate``（=改对次数/犯错次数），因为
    # 本脚本站在**单轮评分**视角，只知道每一轮的对错，无法判断
    # 「这次错是不是上次那个错的延续」。权威定义只在
    # ``scripts/eval/eval_recovery.py`` 产出（它走真实多轮，数得出错误次数）。
    #
    # 历史上仓库里三个文件各有一个同名 ``recovery_rate``、含义互不相同，
    # 引用极易出错，所以现在明确分工：
    #   - recovery_metrics.json  → recovery_rate（权威，改对次数/犯错次数）+ repeat_rate
    #   - error_analysis.json    → recovered_rate + repeat_rate（都是占样本比例）
    #   - eval_report.json       → recovered_rate + repeat_rate（同 error_analysis）
    recovery: dict[str, Any] = {}
    if recovery_rows:
        n = len(recovery_rows)
        repeated = sum(r.get("repeated_first_mistake", False) for r in recovery_rows)
        recovery = {
            "_definition_note": (
                "本节的 recovered_rate / repeat_rate 均为「占纠错轮样本数的比例」。"
                "权威的 recovery_rate（改对次数/犯错次数）见 recovery_metrics.json。"
            ),
            "num_samples": n,
            "recovered_rate": round(sum(r["passed"] for r in recovery_rows) / n, 6),
            "repeat_first_mistake_rate": round(repeated / n, 6),
            "switched_tool_rate": round(
                sum(r.get("switched_tool", False) for r in recovery_rows) / n, 6
            ),
            "by_mistake_key": {
                k: {
                    "total": len(v),
                    "recovered_rate": round(sum(r["passed"] for r in v) / len(v), 4),
                    "repeat_rate": round(
                        sum(r.get("repeated_first_mistake", False) for r in v) / len(v), 4
                    ),
                }
                for k, v in _group_by(recovery_rows, "mistake_key").items()
            },
        }

    # --- 每类错误对应哪些任务类别（用于定位「哪类任务最需要修」）---
    remedies = {
        cat: CATEGORY_REMEDIES_ZH.get(cat, "")
        for cat in FAILURE_CATEGORIES
        if dist.counts.get(cat, 0) > 0
    }

    result: dict[str, Any] = {
        "label": label,
        "is_mock": is_mock,
        "num_predictions": len(preds),
        "distribution": dist.to_dict(),
        "cross_tab": {k: dict(v) for k, v in cross.items()},
        "cross_tab_totals": {k: dict(v) for k, v in cross_total.items()},
        "recovery": recovery,
        "remedies": remedies,
        "remediation_note": (
            "每一类错误对应一种偏好数据构造策略，见 docs/method/preference_data_design.md"
        ),
    }

    # --- 落盘 ---
    (persist_dir / "error_analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    # --- 出图 ---
    if make_figures:
        from src.utils.plots import plot_error_distribution, plot_success_by_group

        figures_dir.mkdir(parents=True, exist_ok=True)
        made: list[str] = []
        p = plot_error_distribution(
            dist.to_dict(), figures_dir / f"error_distribution_{label}.png", label=label
        )
        if p:
            made.append(p)
            # 同时写一份不带 label 的默认名，方便 README 直接引用基线图
            if label.startswith("baseline") and not is_mock:
                p2 = plot_error_distribution(
                    dist.to_dict(), figures_dir / "error_distribution.png", label=label
                )
                if p2:
                    made.append(p2)
        if report:
            for group_key, fname, title in (
                ("by_difficulty", "success_by_difficulty", "按难度分组的成功率"),
                ("by_visibility", "success_by_visibility", "按可见性分组的成功率（unseen = 泛化）"),
            ):
                p = plot_success_by_group(
                    report,
                    figures_dir / f"{fname}_{label}.png",
                    group_key=group_key,
                    primary_k=report.get("primary_k", 4),
                    title=title,
                )
                if p:
                    made.append(p)
                    if label.startswith("baseline") and not is_mock:
                        p2 = plot_success_by_group(
                            report,
                            figures_dir / f"{fname}.png",
                            group_key=group_key,
                            primary_k=report.get("primary_k", 4),
                            title=title,
                        )
                        if p2:
                            made.append(p2)
        result["figures"] = made

    return result


def _group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        out[str(r.get(key, "unknown"))].append(r)
    return out


def print_analysis(res: dict[str, Any]) -> None:
    label = res["label"]
    dist = res["distribution"]
    print("=" * 76)
    print(f"错误分析: {label}" + ("   [MOCK]" if res["is_mock"] else ""))
    print("=" * 76)
    print(f"  样本数 {res['num_predictions']}")
    print(
        f"  成功 {dist['success_count']}（{dist['success_rate']:.1%}）"
    )
    print()
    print(f"  {'错误类别':<14}{'数量':>6}{'占总样本':>10}{'占失败':>10}")
    print("  " + "-" * 46)
    for cat in FAILURE_CATEGORIES:
        n = dist["counts"].get(cat, 0)
        if n == 0:
            continue
        print(
            f"  {CATEGORY_LABELS_ZH[cat]:<14}{n:>6}"
            f"{dist['rates'][cat]:>10.1%}{dist['failure_share'][cat]:>10.1%}"
        )
    print()
    subs = dist.get("failure_subtype_counts", {})
    if subs:
        print("  错误子类型 Top:")
        for k, v in list(subs.items())[:10]:
            from src.eval.error_taxonomy import SUBTYPE_LABELS_ZH

            print(f"    {SUBTYPE_LABELS_ZH.get(k, k):<26}{v:>5}")
    print()
    print("  按难度/可见性交叉（每组的错误构成占比）:")
    for key, row in res["cross_tab"].items():
        total = sum(res["cross_tab_totals"].get(key, {}).values())
        if not total:
            continue
        top = sorted(row.items(), key=lambda kv: -kv[1])[:3]
        desc = ", ".join(f"{CATEGORY_LABELS_ZH.get(k, k)} {v / total:.0%}" for k, v in top)
        print(f"    {key:<10} n={total:<5} 主要错误: {desc}")

    rec = res.get("recovery") or {}
    if rec:
        print()
        print("  错误恢复（纠错轮任务，单轮评分视角）:")
        print(f"    样本数              {rec['num_samples']}")
        print(f"    修正通过率          {rec['recovered_rate']:.1%}   ← 占样本比例")
        print(f"    重复首轮错误率      {rec['repeat_first_mistake_rate']:.1%}   ← 占样本比例")
        print(f"    换工具率            {rec['switched_tool_rate']:.1%}")
        print(
            "    注：权威的 recovery_rate（改对次数/犯错次数）需跑\n"
            "        python scripts/eval/eval_recovery.py（真实多轮）才数得出来。"
        )
        if rec.get("by_mistake_key"):
            print("    逐错误模式:")
            for k, v in sorted(rec["by_mistake_key"].items(), key=lambda kv: kv[1]["recovered_rate"]):
                print(
                    f"      {k:<34} n={v['total']:<4} "
                    f"通过率={v['recovered_rate']:.0%}  重复率={v['repeat_rate']:.0%}"
                )

    if res.get("remedies"):
        print()
        print("  针对每类错误的改进手段:")
        for cat, remedy in res["remedies"].items():
            print(f"    {CATEGORY_LABELS_ZH.get(cat, cat)}: {remedy}")

    if res.get("figures"):
        print()
        print("  图表:")
        for f in res["figures"]:
            print(f"    {f}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    figures_dir = Path(args.figures_dir)
    if not figures_dir.is_absolute():
        figures_dir = CFG.path() / figures_dir

    targets: list[Path] = []
    if args.all:
        exp_root = CFG.path("experiments")
        if not exp_root.exists():
            print(f"错误：{exp_root} 不存在", file=sys.stderr)
            return 1
        targets = sorted(
            d for d in exp_root.iterdir() if d.is_dir() and (d / "predictions.jsonl").exists()
        )
        # mock 子目录也纳入
        mock_root = exp_root / "mock"
        if mock_root.exists():
            targets += sorted(
                d for d in mock_root.iterdir() if d.is_dir() and (d / "predictions.jsonl").exists()
            )
        if not targets:
            print("没有找到任何 predictions.jsonl。先跑 run_baseline.py。", file=sys.stderr)
            return 1
    else:
        if not args.label and not args.persist_dir:
            print("错误：需要 --label 或 --persist-dir 或 --all", file=sys.stderr)
            return 2
        if args.persist_dir:
            d = Path(args.persist_dir)
            targets = [d if d.is_absolute() else CFG.path() / d]
        else:
            targets = [CFG.path("experiments", args.label)]

    failures = 0
    for d in targets:
        try:
            res = analyze_one(
                d,
                figures_dir=figures_dir,
                make_figures=not args.no_figures,
                max_exemplars=args.max_exemplars,
                task_set=args.task_set,
            )
            print()
            print_analysis(res)
            print()
        except FileNotFoundError as exc:
            print(f"跳过 {d}: {exc}", file=sys.stderr)
            failures += 1
        except Exception as exc:
            print(f"分析 {d} 失败: {type(exc).__name__}: {exc}", file=sys.stderr)
            failures += 1

    return 1 if failures == len(targets) else 0


if __name__ == "__main__":
    raise SystemExit(main())
