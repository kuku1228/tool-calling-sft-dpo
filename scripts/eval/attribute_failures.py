"""失败归因：只看 **passed=False** 的样本，按真实错误类型分解。

## 为什么必须重写（我在这里错了两次）

**错误 1：把「正确追问」当成错误。**
`clarify_instead` 是「该轮期望不调用工具，模型也没调用」的标记 ——
它在「应先追问」任务上是**正确答案**，判 ``SUCCESS``。
我把它读成「过度追问」，于是以为有 62 条过度追问，
实际只有 **2 条**。

**错误 2：用了可重叠的错误类别做归因。**
``error_analysis.json`` 的「错误类别」表**不互斥** ——
实测 sft_v7：``格式错误 5 + 参数错误 23 + 工具选错 63 = 91``，
而总失败只有 ``424 - 336 = 88``。
一个样本可以同时记多个缺陷，所以那张表**不能用来算占比**。

**正确做法**：读 ``eval_report.json`` 的 ``per_task``，
只看 ``outcomes[i] is False`` 的样本，用它的 ``categories[i]`` 归因。

## 用法

```bash
python scripts/eval/attribute_failures.py --label sft_v7 
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


def load_report(label: str, exp_root: Path) -> dict:
    d = exp_root / label
    for name in ("eval_report.json", "eval_report.mock.json"):
        p = d / name
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"{d} 下找不到 eval_report*.json")


def attribute(rep: dict, tasks: dict) -> dict:
    """只统计 **passed=False** 的样本，按类别/错误类型/任务归因。

    ⚠️ **字段名（实测确认，别猜）**：``per_task`` 每项含

    * ``outcomes``          ``[bool, ...]`` 每次采样是否通过
    * ``error_categories``  ``["WRONG_TOOL", ...]`` 每次采样的**错误类别**
      （**不是** ``categories`` —— 我猜错过一次）
    * ``predictions``       ``[{"classification": {"detail": {"subtype": ...}}}]``
      **子类型**要从这里取（``error_categories`` 只有顶层类别）

    字段名猜错的后果：所有错误的 ``error`` 都变成 ``"?"``，
    于是「过度追问」与「该追问却调工具」两类都统计为 0 ——
    **看起来像「没有问题」，实际是脚本坏了。**
    """
    per_task = rep.get("per_task") or []
    fails: list[dict] = []
    n_total = 0

    for r in per_task:
        tid = r.get("task_id", "")
        outcomes = r.get("outcomes") or []
        cats = r.get("error_categories") or []
        preds = r.get("predictions") or []
        t = tasks.get(tid)
        for i, ok in enumerate(outcomes):
            n_total += 1
            if ok:
                continue
            # 子类型：从 predictions[i].classification.detail.subtype 取
            subtype = "?"
            if i < len(preds) and isinstance(preds[i], dict):
                cl = (preds[i].get("classification") or {})
                det = (cl.get("detail") or {})
                subtype = det.get("subtype") or det.get("category") or "?"
            fails.append({
                "task_id": tid,
                "sample_index": i,
                "category": (t.category if t else r.get("category", "")),
                "expected_tool": (t.expected.tool if t else None),
                "expected_is_clarify": bool(t and t.expected.tool is None),
                "cat": cats[i] if i < len(cats) else "?",
                "subtype": subtype,
                "user": (t.turns[0].content[:52] if t else ""),
            })

    return {
        "label": rep.get("label", ""),
        "n_samples": n_total,
        "n_fail": len(fails),
        "fail_rate": len(fails) / max(1, n_total),
        "by_cat": dict(collections.Counter(f["cat"] for f in fails).most_common()),
        "by_subtype": dict(collections.Counter(
            f["subtype"] for f in fails).most_common()),
        "by_category": dict(collections.Counter(
            f["category"] for f in fails).most_common()),
        # 关键：区分两类方向相反的「工具调用决策」错误
        "over_clarify": [f for f in fails
                         if not f["expected_is_clarify"]
                         and f["subtype"] == "clarify_instead"],
        "under_clarify": [f for f in fails
                          if f["expected_is_clarify"]
                          and f["cat"] == "WRONG_TOOL"],
        "fails": fails,
    }


def render(rep: dict, tasks: dict) -> str:
    L = [f"## 失败归因：{rep['label']}", ""]
    L.append(f"总样本 {rep['n_samples']}，**失败 {rep['n_fail']}**"
             f"（{rep['fail_rate']:.1%}）。只统计 ``passed=False`` 的样本。")
    L.append("")
    L.append("### 按错误**类别**（顶层，5 类互斥）")
    L.append("")
    L.append("| 类别 | 数量 | 占失败 |")
    L.append("|---|---|---|")
    for k, v in rep["by_cat"].items():
        L.append(f"| `{k}` | {v} | {v/max(1,rep['n_fail']):.1%} |")
    L.append("")
    L.append("### 按错误**子类型**（细分，用于定位）")
    L.append("")
    L.append("| 子类型 | 数量 | 占失败 |")
    L.append("|---|---|---|")
    for k, v in rep["by_subtype"].items():
        L.append(f"| `{k}` | {v} | {v/max(1,rep['n_fail']):.1%} |")
    L.append("")
    L.append("### 按任务类别")
    L.append("")
    L.append("| 类别 | 失败数 |")
    L.append("|---|---|")
    for k, v in rep["by_category"].items():
        L.append(f"| {k} | {v} |")
    L.append("")

    # 两类方向相反的决策错误，必须分开
    L.append("### 两类方向相反的「工具调用决策」错误（关键区分）")
    L.append("")
    oc, uc = rep["over_clarify"], rep["under_clarify"]
    L.append(f"- **过度追问**（信息够，却停下追问）：**{len(oc)} 条**")
    L.append(f"- **该追问却调了工具**：**{len(uc)} 条**")
    L.append("")
    for name, rows in (("过度追问", oc), ("该追问却调工具", uc)):
        if not rows:
            continue
        L.append(f"**{name}** 明细：")
        L.append("")
        for f in rows[:20]:
            L.append(f"- `{f['task_id']}` [{f['category']}] "
                     f"期望 `{f['expected_tool']}` — 用户：{f['user']!r}")
        if len(rows) > 20:
            L.append(f"- … 另有 {len(rows)-20} 条")
        L.append("")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="失败归因（只看失败样本）")
    ap.add_argument("--label", required=True)
    ap.add_argument("--task-set", default="v3", choices=["v1", "v3"])
    ap.add_argument("--exp-root", default="experiments")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    from src.data.task_manifest import load_tasks_by_version

    tasks = {t.task_id: t for t in load_tasks_by_version(args.task_set)}
    try:
        rep = load_report(args.label, Path(args.exp_root))
    except FileNotFoundError as exc:
        print(f"  ✗ {exc}")
        return 1

    res = attribute(rep, tasks)
    md = render(res, tasks)
    print(md)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"\n  已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
