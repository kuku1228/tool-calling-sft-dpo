"""按**实测准确率**重标任务难度，替代「按类别拍脑袋赋难度」。

## 为什么必须重标

现状是难度**完全由类别决定**（实测）：

    baggage           全部 easy
    weather_advisory  全部 easy
    booking_flow      全部 hard
    flight_search     全部 hard
    cancellation      全部 hard
    multi_tool        全部 hard
    multi_turn        全部 hard
    error_correction  全部 medium

这是专家明确指出、且我确认成立的问题：

* 同一类别内难度完全相同是不可能的 —— `baggage` 里既有 `check_baggage`
  （只读，简单）也有 `modify_baggage`（写操作 + 数值参数），难度不该一样
* `flight_search` 有 14 个任务，不可能全是 hard
* 难度标签**会影响所有按难度分组的结论**（如「hard 上 SFT 只有 73.8%」），
  标签错了那些结论就错了

## 做法

读 `experiments/<label>/predictions.jsonl` 的**逐任务准确率**
（每个任务有 num_samples 次采样），按分位数重标：

    accuracy >= 上分位       → easy
    下分位 < accuracy < 上分位 → medium
    accuracy <= 下分位       → hard

分位数默认取 1/3 与 2/3。也支持 `--abs-thresholds` 用绝对阈值
（更适合跨实验比较）。

## 输出

* 控制台：原标签 vs 实测标签的**对照表**（这是本脚本的主要价值 ——
  看差异有多大）
* ``difficulty_recalibration.json``：逐任务的实测准确率与新标签

## 用法

    python scripts/eval/assign_difficulty_by_accuracy.py --label baseline
    python scripts/eval/assign_difficulty_by_accuracy.py \
        --label sft  --abs-thresholds 0.5,0.85
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.data.task_manifest import load_tasks_by_version  # noqa: E402


def _load_predictions(exp_dir: Path) -> list[dict]:
    for name in ("predictions.jsonl", "predictions.mock.jsonl"):
        p = exp_dir / name
        if p.exists():
            return [json.loads(ln) for ln in p.open(encoding="utf-8") if ln.strip()]
    raise FileNotFoundError(
        f"{exp_dir} 下找不到 predictions.jsonl。"
        "本脚本需要**逐任务预测**，不接受只有 eval_report.json 的情况。"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="按实测准确率重标任务难度")
    ap.add_argument("--label", required=True, help="实验标签")
    ap.add_argument("--exp-root", default="experiments")
    ap.add_argument("--task-set", default="v3", choices=["v1", "v3"])
    ap.add_argument(
        "--abs-thresholds", default=None,
        help="绝对阈值 lo,hi（如 0.5,0.85）。不填则用分位数 1/3、2/3",
    )
    ap.add_argument("--out", default=None, help="输出 json 路径")
    args = ap.parse_args(argv)

    exp_dir = Path(args.exp_root) / args.label
    rows = _load_predictions(exp_dir)

    # 逐任务聚合：正确数 / 采样数
    per_task: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        tid = r.get("task_id", "")
        per_task[tid][1] += 1
        if r.get("passed") or r.get("correct"):
            per_task[tid][0] += 1

    tasks = load_tasks_by_version(args.task_set)
    by_id = {t.task_id: t for t in tasks}

    records = []
    for tid, (ok, n) in per_task.items():
        t = by_id.get(tid)
        if t is None:
            continue
        records.append({
            "task_id": tid,
            "category": t.category,
            "tool": t.expected.tool,
            "old_difficulty": t.difficulty,
            "accuracy": ok / n if n else 0.0,
            "n_samples": n,
            "n_correct": ok,
        })
    if not records:
        print("  ✗ 没有匹配到任务 —— 检查 --task-set 是否与预测一致")
        return 1

    # 分位数或绝对阈值
    if args.abs_thresholds:
        lo, hi = (float(x) for x in args.abs_thresholds.split(","))
        print(f"  用绝对阈值：easy >= {hi}  >  medium >  {lo}  >= hard")
    else:
        accs = sorted(r["accuracy"] for r in records)
        lo = accs[len(accs) // 3]
        hi = accs[2 * len(accs) // 3]
        print(f"  用分位数阈值：lo={lo:.3f}  hi={hi:.3f}（1/3、2/3 分位）")

    # ⚠️ 退化保护：当大量任务的准确率**相同**时（例如 mock 下全为 0.0，
    # 或真实模型下全部 1.0），分位数会退化成 ``lo == hi``。
    # 早期判据写的是 ``easy if a >= hi else ...``，于是 ``0 >= 0`` 成立，
    # **全错的任务被判成 easy** —— 方向完全反了。
    # 修法：显式按区间判定（easy 必须严格优于 hi），并在退化时给出提示。
    degraded = (lo == hi)
    if degraded:
        print(f"  ⚠️ 阈值退化（lo == hi == {lo:.3f}）：绝大多数任务准确率相同。")
        print("     此时分位数无法区分难度，建议改用 --abs-thresholds。")
        print("     本脚本仍会输出，但新标签的可信度有限。")

    for r in records:
        a = r["accuracy"]
        if hi > lo and a >= hi:
            r["new_difficulty"] = "easy"
        elif hi > lo and a <= lo:
            r["new_difficulty"] = "hard"
        else:
            # 退化情形：只有「全对」才算 easy、「全错」才算 hard，其余 medium
            r["new_difficulty"] = (
                "easy" if a >= 1.0 else ("hard" if a <= 0.0 else "medium")
            )
        r["threshold_degraded"] = degraded

    # ---------------- 报告 ----------------
    print()
    print("=" * 76)
    print(f"难度重标：{args.label}（{len(records)} 个任务）")
    print("=" * 76)

    # 1) 混淆矩阵：原标签 × 新标签
    print("\n### 原标签 × 实测标签")
    conf = collections.Counter((r["old_difficulty"], r["new_difficulty"]) for r in records)
    labels = ["easy", "medium", "hard"]
    print(f"  {'原↓ / 新→':<12}" + "".join(f"{x:>10}" for x in labels))
    for a in labels:
        row = [conf.get((a, b), 0) for b in labels]
        if sum(row):
            print(f"  {a:<12}" + "".join(f"{v:>10}" for v in row))
    same = sum(conf.get((x, x), 0) for x in labels)
    print(f"  一致率：{same}/{len(records)} = {same/len(records):.1%}")

    # 2) 按类别看：实测难度是否真的分散
    print("\n### 按类别看实测难度分布（原状态是每类只有一种）")
    by_cat: dict[str, list[str]] = collections.defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r["new_difficulty"])
    for c in sorted(by_cat):
        cnt = collections.Counter(by_cat[c])
        old = {r["old_difficulty"] for r in records if r["category"] == c}
        span = min(r["accuracy"] for r in records if r["category"] == c), \
               max(r["accuracy"] for r in records if r["category"] == c)
        print(f"  {c:<18} n={len(by_cat[c]):>3}  原={sorted(old)}  "
              f"实测={dict(sorted(cnt.items()))}  acc范围=[{span[0]:.2f},{span[1]:.2f}]")

    # 3) 最不一致的任务
    diff = [r for r in records if r["old_difficulty"] != r["new_difficulty"]]
    if diff:
        print(f"\n### 标签变化最大的 10 个任务（共 {len(diff)} 个变化）")
        diff.sort(key=lambda r: abs(
            labels.index(r["new_difficulty"]) - labels.index(r["old_difficulty"])
        ), reverse=True)
        for r in diff[:10]:
            print(f"  {r['task_id']:<14} {r['old_difficulty']:>7} → "
                  f"{r['new_difficulty']:<7} acc={r['accuracy']:.2f} "
                  f"({r['n_correct']}/{r['n_samples']})  {r['category']}")

    out = Path(args.out) if args.out else exp_dir / "difficulty_recalibration.json"
    out.write_text(json.dumps({
        "label": args.label,
        "task_set": args.task_set,
        "thresholds": {"lo": lo, "hi": hi, "mode": "abs" if args.abs_thresholds else "quantile"},
        "agreement": same / len(records),
        "tasks": sorted(records, key=lambda r: r["task_id"]),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
