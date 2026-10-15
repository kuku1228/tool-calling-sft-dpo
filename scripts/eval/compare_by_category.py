"""多实验分类别对照 —— 回答「这次改动有没有修好那个 0%」。

## 为什么需要它

单看一次评测的「按类别」表不够用。真正要回答的是：

> **补了训练语料之后，那三个 0% 的类别涨起来了吗？**

这需要把两个实验**并排**看。而 `collect_results.py` 给的是总体数字，
看不出「哪一类涨了、哪一类没动」。

实测背景（v3 基线）：

    类别            baseline  格式(旧口径)  错误子类型
    weather          90.0%      90.0%
    baggage          67.9%      83.9%
    cancellation     62.5%       0.0%      ← 格式是测量 bug，已修
    error_correction 48.3%      11.7%
    multi_turn       12.5%      41.7%
    flight_search     0.0%     100.0%     wrong_value=56（三字码编造）
    multi_tool        0.0%      95.0%     「调了航班没调天气」=38/40
    booking_flow      0.0%       0.0%     action_hallucination=58/60

补了训练语料（多步流程 0→28、multi_tool 0→12）后，
`booking_flow` / `multi_tool` 应当改善。本脚本量化这个改善。

## 用法

```bash
python scripts/eval/compare_by_category.py \
    --labels baseline,sft_v5 
```
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
from src.eval.tool_call_parser import expected_format_ok  # noqa: E402


def load(label: str, exp_root: Path) -> list[dict]:
    d = exp_root / label
    for name in ("predictions.jsonl", "predictions.mock.jsonl"):
        p = d / name
        if p.exists():
            return [json.loads(ln) for ln in p.open(encoding="utf-8") if ln.strip()]
    raise FileNotFoundError(f"{d} 下找不到 predictions.jsonl")


def per_category(
    rows: list[dict], tasks: dict, *, task_set: str = "v3"
) -> dict[str, dict]:
    """算每个类别的成功率与**按任务口径**的格式正确率。

    ⚠️ 格式正确率必须按任务判：对「应先追问」任务，正确输出是自然语言追问，
    旧口径会把它判成格式错误（实测 `cancellation` 因此显示 0.0%）。
    """
    from src.eval.grader import grade

    by: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        t = tasks.get(r.get("task_id", ""))
        if t is None:
            continue
        text = r.get("text", "")
        g = grade(t, text, finish_reason=r.get("finish_reason", ""))
        by[t.category].append({
            "passed": bool(g.passed),
            "fmt": expected_format_ok(
                is_clarify_task=t.expected.tool is None, text=text),
        })
    out = {}
    for cat, rs in by.items():
        n = len(rs)
        out[cat] = {
            "n": n,
            "acc": sum(1 for x in rs if x["passed"]) / n,
            "fmt": sum(1 for x in rs if x["fmt"]) / n,
        }
    return out


def render(all_cats: dict[str, dict[str, dict]], labels: list[str]) -> str:
    L = ["## 分类别对照", ""]
    L.append("格式正确率用**按任务口径**（对「应先追问」任务，"
             "自然语言追问算格式正确）。")
    L.append("")
    head = "| 类别 | n | " + " | ".join(
        f"`{x}` 成功率 | `{x}` 格式" for x in labels) + " | 变化 |"
    L.append(head)
    L.append("|---" * (2 + 2 * len(labels)) + "|")

    for cat in sorted(all_cats):
        first = all_cats[cat].get(labels[0])
        n = first["n"] if first else 0
        cells = []
        for lab in labels:
            d = all_cats[cat].get(lab)
            cells.append(f"{d['acc']:.1%} | {d['fmt']:.1%}" if d else "— | —")
        # 变化：最后一个 vs 第一个
        delta = ""
        if len(labels) >= 2:
            a = all_cats[cat].get(labels[0])
            b = all_cats[cat].get(labels[-1])
            if a and b:
                dv = b["acc"] - a["acc"]
                mark = "↑" if dv > 0.01 else ("↓" if dv < -0.01 else "=")
                delta = f"{mark} {dv:+.1%}"
        L.append(f"| {cat} | {n} | " + " | ".join(cells) + f" | {delta} |")
    L.append("")

    # 总体
    L.append("### 总体")
    L.append("")
    L.append("| 实验 | 样本 | 成功率 | 格式（按任务） |")
    L.append("|---|---|---|---|")
    for lab in labels:
        # ⚠️ 不要在推导式里用海象运算符**重绑迭代变量**（``d := ...`` 而 ``d``
        # 也是 ``for d in ...``）—— 那是 SyntaxError，而且报错信息指向
        # 一个看起来没问题的表达式。用普通循环更清楚。
        tot = acc_sum = fmt_sum = 0
        for cat in all_cats:
            entry = all_cats[cat].get(lab)
            if not entry:
                continue
            tot += entry["n"]
            acc_sum += entry["acc"] * entry["n"]
            fmt_sum += entry["fmt"] * entry["n"]
        if tot:
            L.append(
                f"| `{lab}` | {tot} | {acc_sum/tot:.2%} | {fmt_sum/tot:.1%} |"
            )
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="多实验分类别对照")
    ap.add_argument("--labels", required=True, help="逗号分隔，按时间顺序")
    ap.add_argument("--task-set", default="v3", choices=["v1", "v3"])
    ap.add_argument("--exp-root", default="experiments")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    root = Path(args.exp_root)
    tasks = {t.task_id: t for t in load_tasks_by_version(args.task_set)}

    all_cats: dict[str, dict[str, dict]] = {}
    for lab in labels:
        try:
            rows = load(lab, root)
        except FileNotFoundError as exc:
            print(f"  ✗ {exc}")
            continue
        for cat, d in per_category(rows, tasks, task_set=args.task_set).items():
            all_cats.setdefault(cat, {})[lab] = d

    if not all_cats:
        print("  ✗ 没有可对照的数据")
        return 1
    md = render(all_cats, labels)
    print(md)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"\n  已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
