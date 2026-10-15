"""评测集与训练语料的**重合层级**审计（评测前门禁）。

## 为什么需要它

我原先只检查了「prompt 是否逐字相同」，得出「0 重合、干净」的结论。
那只覆盖了**一层**。实测把层级拆开后发现另外几层都有问题：

| 层级 | 判据 | v3 实测 |
|---|---|---|
| 1 | prompt 逐字相同 | **0** ✅ |
| 2 | **完整期望相同**（工具+参数名+参数值） | **20 (18.9%)** ⚠️ |
| 3 | 同工具 + 同参数名（签名） | 85 (80.2%) |
| 4 | 同句式、实体不同 | 0 |
| 5 | 全新 | 1 |

**层级 2 是真正需要处理的**：模型可能靠**匹配记忆中的那条训练样本**答对，
而不是学会了规则。

## 层级 2 里两类必须分开看

**（a）实体确实相同的泄漏** —— 5 个（4 个天气 + 1 个 multi_tool）。
这些是**真的可能靠记忆答对**：训练里有 `get_weather(city='三亚')`，
评测又考 `get_weather(city='三亚')`。

**（b）「应先追问」任务的必然重合** —— 15 个 clarify 任务。
它们期望调用是**空的**（正确行为 = 不调用工具），因此
「完整期望相同」对它们**必然成立**，不是巧合。

但 (b) 暴露一个**独立的问题**：这 15 个任务的**正确答案是同一个字符串行为**，
却在评测里被当作 **15 个独立样本**计分。这会把一个行为重复计数 15 次。

## 输出

* 逐任务的重合层级
* **被污染的指标 vs 干净子集上的指标**
* 「同一期望调用」的任务分组 —— 用于判断有效独立样本数

## 用法

```bash
python scripts/eval/audit_eval_overlap.py 
python scripts/eval/audit_eval_overlap.py  --json out.json
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
from src.data.recovery_corpus import build_training_tasks  # noqa: E402


def expected_key(task) -> tuple:
    """完整期望：(工具, (参数名, 参数值)...) —— 层级 2 的判据。

    >>> class T:
    ...     class E: tool="f"; args={"a": 1}
    ...     expected = E()
    >>> expected_key(T())
    ('f', (('a', '1'),))
    """
    a = task.expected.args or {}
    return (task.expected.tool, tuple(sorted((k, str(v)) for k, v in a.items())))


def signature_key(task) -> tuple:
    """签名：(工具, 参数名元组) —— 不含参数值。

    >>> class T:
    ...     class E: tool="f"; args={"a": 1}
    ...     expected = E()
    >>> signature_key(T())
    ('f', ('a',))
    """
    a = task.expected.args or {}
    return (task.expected.tool, tuple(sorted(a.keys())))


def prompt_key(task) -> str:
    return task.turns[0].content


def overlap_level(task, *, syn_prompts, syn_expected, syn_sig, syn_skel, skeleton_fn) -> int:
    """返回 1..5，数字越小重合越强。"""
    if prompt_key(task) in syn_prompts:
        return 1
    if expected_key(task) in syn_expected:
        return 2
    if signature_key(task) in syn_sig:
        return 3
    if skeleton_fn(prompt_key(task)) in syn_skel:
        return 4
    return 5


LEVELS = {
    1: "prompt 逐字相同",
    2: "完整期望相同",
    3: "同工具同参数名",
    4: "同句式（实体不同）",
    5: "全新",
}

#: 实体重合度三档（专家建议，替代没有含义的 seen/unseen）
#:
#: 原 ``seen``/``unseen`` 标签的赋值**从不读取训练数据**
#: （v1 是 ``counter % 3``、v3 是 ``sha256(skeleton) % 3``），
#: 实测 ``unseen`` 组的签名也被训练覆盖 **38/38** —— 那个标签没有含义。
#:
#: 专家建议改成按「训练里见过多少实体」分三档。这里用**期望调用里的
#: 实参值**（城市 / 订单号 / 航线 / 日期）作为「实体」：
#:
#: * ``entity_new``   —— 期望调用的**实参值**在训练里完全没出现过（真 unseen）
#: * ``entity_partial`` —— 部分实参值出现过（半 unseen）
#: * ``entity_seen``  —— 全部实参值都出现过（seen）
OVERLAP_TIERS = ("entity_new", "entity_partial", "entity_seen")


def entity_tier(task, *, syn_values: set[str]) -> str:
    """按「期望调用的实参值有多少在训练里出现过」分档。

    >>> entity_tier(_mk({"city": "三亚"}), syn_values={"三亚"})
    'entity_seen'
    >>> entity_tier(_mk({"city": "三亚"}), syn_values=set())
    'entity_new'
    """
    a = task.expected.args or {}
    vals = [str(v) for v in a.values()]
    if not vals:
        # 期望调用为空（应先追问类）：没有实参可比较，归为 partial
        return "entity_partial"
    hit = sum(1 for v in vals if v in syn_values)
    if hit == 0:
        return "entity_new"
    if hit == len(vals):
        return "entity_seen"
    return "entity_partial"


def _mk(args: dict):
    class _E:
        tool = "t"
        pass

    class _T:
        expected = _E()

        class turns:  # noqa: N801
            pass

    _T.expected.args = args
    _T.turns = [type("Turn", (), {"content": "p"})()]
    _T.task_id = "x"
    _T.category = "c"
    _T.visibility = "seen"
    return _T()


def audit(task_set: str = "v3") -> dict:
    from src.data.task_manifest_v2 import skeleton

    syn = build_training_tasks(seed=42)
    tasks = load_tasks_by_version(task_set)

    syn_prompts = {prompt_key(t) for t in syn}
    syn_expected = {expected_key(t) for t in syn}
    syn_sig = {signature_key(t) for t in syn}
    syn_skel = {skeleton(prompt_key(t)) for t in syn}

    # 训练语料里出现过的**所有实参值**（城市/订单号/航线/日期…），
    # 用于「实体重合度」分档。
    syn_values: set[str] = set()
    for x in syn:
        for v in (x.expected.args or {}).values():
            syn_values.add(str(v))

    rows = []
    for t in tasks:
        lv = overlap_level(
            t,
            syn_prompts=syn_prompts, syn_expected=syn_expected,
            syn_sig=syn_sig, syn_skel=syn_skel, skeleton_fn=skeleton,
        )
        rows.append({
            "task_id": t.task_id,
            "category": t.category,
            "tool": t.expected.tool,
            "visibility_label": t.visibility,
            # ⚠️ 下面这个才是**有含义**的分层：按「训练里见过多少实体」。
            # 原来的 visibility 标签从不读训练数据，实测 unseen 组
            # 也被覆盖 38/38 —— 无含义（专家建议替换成实体重合度）。
            "entity_tier": entity_tier(t, syn_values=syn_values),
            "level": lv,
            "level_name": LEVELS[lv],
            "expected": str(t.expected.args),
            "is_clarify": t.expected.tool is None,
        })

    by_level = collections.Counter(r["level"] for r in rows)

    # 层级 2 分成两类：实体真泄漏 vs clarify 的必然重合
    lv2 = [r for r in rows if r["level"] == 2]
    entity_leak = [r for r in lv2 if not r["is_clarify"]]
    clarify_dup = [r for r in lv2 if r["is_clarify"]]

    # 同一期望调用的分组（看「一个行为被计了几次」）
    groups: dict[str, list[str]] = collections.defaultdict(list)
    for r in rows:
        groups[r["expected"]].append(r["task_id"])
    multi = {k: v for k, v in groups.items() if len(v) > 1}

    return {
        "task_set": task_set,
        "n_tasks": len(tasks),
        "n_train_tasks": len(syn),
        "by_level": {str(k): by_level.get(k, 0) for k in sorted(LEVELS)},
        "level_names": {str(k): v for k, v in LEVELS.items()},
        "entity_leaks": entity_leak,
        "clarify_duplicates": clarify_dup,
        # 污染判定：层级 1+2 且**非 clarify**
        "contaminated_ids": [r["task_id"] for r in entity_leak],
        "clean_ids": [r["task_id"] for r in rows
                      if r["level"] >= 3 or r["is_clarify"]],
        "duplicate_expected_groups": multi,
        "entity_tiers": dict(collections.Counter(r["entity_tier"] for r in rows)),
        "rows": rows,
    }


def render(rep: dict) -> str:
    L = [f"## 评测集 / 训练语料 重合审计（{rep['task_set']}）", ""]
    L.append(f"评测 {rep['n_tasks']} 个任务，训练语料 {rep['n_train_tasks']} 个任务。")
    L.append("")
    L.append("| 重合层级 | 任务数 | 占比 |")
    L.append("|---|---|---|")
    for k, name in rep["level_names"].items():
        n = rep["by_level"][str(k)]
        L.append(f"| {k}. {name} | {n} | {n/rep['n_tasks']:.1%} |")
    L.append("")

    L.append("### 有含义的分层：实体重合度（替代无含义的 seen/unseen）")
    L.append("")
    L.append("原 `visibility` 标签的赋值**从不读取训练数据**"
             "（v1 是 `counter % 3`、v3 是 `sha256(skeleton) % 3`），"
             "实测 `unseen` 组的签名也被训练覆盖 **38/38** —— 那个标签没有含义。"
             "改用「期望调用的实参值有多少在训练里出现过」分档：")
    L.append("")
    L.append("| 档位 | 含义 | 任务数 |")
    L.append("|---|---|---|")
    tier_desc = {
        "entity_new": "实参值训练里**完全没出现过**（真 unseen）",
        "entity_partial": "部分实参值出现过（半 unseen）",
        "entity_seen": "全部实参值都出现过（seen）",
    }
    for tier, n in sorted(rep["entity_tiers"].items()):
        L.append(f"| `{tier}` | {tier_desc.get(tier, '')} | {n} |")
    L.append("")

    el = rep["entity_leaks"]
    cd = rep["clarify_duplicates"]
    L.append("### 需要处理的两类")
    L.append("")
    L.append(f"**（a）实体真泄漏：{len(el)} 个** —— 完整期望（工具+参数名+参数值）"
             "与某条训练样本相同。")
    if el:
        L.append("")
        L.append("| 任务 | 类别 | 现有标签 | 期望 |")
        L.append("|---|---|---|---|")
        for r in el:
            L.append(f"| `{r['task_id']}` | {r['category']} | "
                     f"{r['visibility_label']} | `{r['expected']}` |")
        L.append("")
        L.append("> ⚠️ 这些任务上「答对」可能来自**匹配记忆中的训练样本**，"
                 "而不是学会了规则。报告里应给出**剔除它们之后**的指标。")
    L.append("")
    L.append(f"**（b）「应先追问」的必然重合：{len(cd)} 个** —— 它们的期望调用是"
             "**空的**（正确行为 = 不调用工具），所以「完整期望相同」是必然的。")
    L.append("")
    L.append("> ⚠️ 但这是一个**独立问题**：这 %d 个任务的**正确答案是同一个行为**，"
             "却在评测里被当作 %d 个独立样本计分。"
             "一个行为被重复计数，会把它的权重放大。"
             % (len(cd), len(cd)))

    dup = rep["duplicate_expected_groups"]
    if dup:
        L.append("")
        L.append("### 同一期望调用的任务分组（用于判断有效独立样本数）")
        L.append("")
        L.append("| 期望调用 | 任务数 | 任务 |")
        L.append("|---|---|---|")
        for k, v in sorted(dup.items(), key=lambda kv: -len(kv[1]))[:10]:
            L.append(f"| `{k[:52]}` | {len(v)} | {', '.join(v[:4])}"
                     + (" …" if len(v) > 4 else "") + " |")
    L.append("")
    L.append("### 判定")
    L.append("")
    L.append(f"- 层级 1（prompt 逐字相同）：**{rep['by_level']['1']}** "
             + ("✅ 无污染" if rep["by_level"]["1"] == 0 else "❌ 存在污染"))
    L.append(f"- 层级 2（完整期望相同）：**{rep['by_level']['2']}** —— "
             f"其中实体真泄漏 {len(el)} 个、clarify 必然重合 {len(cd)} 个")
    L.append(f"- 建议：报告里**同时给出**全量指标与"
             f"「剔除 {len(el)} 个实体泄漏后」的指标（干净子集 {len(rep['clean_ids'])} 个任务）")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="评测集/训练语料重合审计")
    ap.add_argument("--task-set", default="v3", choices=["v1", "v3"])
    ap.add_argument("--json", default=None, help="把完整报告写到 json")
    args = ap.parse_args(argv)

    rep = audit(args.task_set)
    md = render(rep)
    print(md)
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n  已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
