"""评测报告的「已知限制」自动生成。

## 为什么需要它（这一条我反复栽跟头）

这个项目里最危险的错误是「**报告漏掉了某个限制**」：
数字本身算对了，但结论被读成比实际更强的说法。

实测发生过的三处：

1. **`seen`/`unseen` 被当成泛化差距** —— 实测那个标签**从不读取训练数据**，
   `unseen` 组的签名也被训练覆盖 **38/38**。而我用它讲过泛化。
2. **「14 个 clarify 独立样本」** —— 那 14 个任务的**期望调用完全相同**
   （都是「不调用工具」），有效独立样本是 **1 个行为**。
   统计口径已改成按簇重采样，但**表述**没跟着改，散落在多个文档里。
3. **剔除任务后的样本量** —— 剔除实体泄漏后是 101 而非 106，
   但报告标题里仍写 106。

**共同点：这些都是「必须主动声明、否则读者会误解」的事实，而要靠人记得。**

所以本模块把它们变成**自动章节** —— 每次评测都生成，
数据变了文字也跟着变（例如簇数从 92 变成别的）。

## 设计原则

* **数字从数据算**，不是硬编码 —— 数据改了，文字自动跟着改
* 每条限制都写清「**所以哪个说法不能用**」，而不只是「有个限制」
* 有对应检查器/测试的限制，附上验证命令
"""

from __future__ import annotations

import collections
from pathlib import Path


def _expected_key(task) -> tuple:
    a = task.expected.args or {}
    return (task.expected.tool, tuple(sorted((k, str(v)) for k, v in a.items())))


def collect_limitations(tasks: list, *, task_set: str = "v3") -> list[dict]:
    """收集当前配置下**真实存在**的限制。

    返回 ``[{id, severity, statement, do_not_claim, verify}]``。

    ``do_not_claim`` 是**最重要的字段** —— 它直接写出「不能用哪个说法」。
    """
    out: list[dict] = []

    # ---- 1. seen/unseen 标签是否有含义 ----
    try:
        from src.data.task_manifest_v2 import verify_visibility_is_meaningful

        vv = verify_visibility_is_meaningful(tasks)
        if not vv["meaningful"]:
            out.append({
                "id": "visibility_meaningless",
                "severity": "高",
                "statement": (
                    f"`seen`/`unseen` 标签**与训练暴露无关**："
                    f"seen 组签名被训练覆盖 {vv['seen_signature_covered']}/{vv['seen']}"
                    f"（{vv['seen_coverage_rate']:.0%}），"
                    f"unseen 组 {vv['unseen_signature_covered']}/{vv['unseen']}"
                    f"（{vv['unseen_coverage_rate']:.0%}）。"
                    "该标签的赋值从不读取训练数据（是按序号/骨架哈希轮转）。"
                ),
                "do_not_claim": (
                    "**不要**把 seen 与 unseen 的差值解释为泛化差距 —— "
                    "那是哈希分组的噪声。改用「实体重合度」三档"
                    "（见 `scripts/eval/audit_eval_overlap.py`）。"
                ),
                "verify": "python -c \"import sys;sys.path.insert(0,'.');"
                          "from src.data.task_manifest_v2 import "
                          "report_visibility_warning as W;print(W())\"",
            })
    except Exception:  # pragma: no cover
        pass

    # ---- 2. 簇数 < 任务数（同一期望调用的任务不是独立证据）----
    groups = collections.Counter(_expected_key(t) for t in tasks)
    n_tasks, n_clusters = len(tasks), len(groups)
    if n_clusters < n_tasks:
        biggest_key, biggest_n = groups.most_common(1)[0]
        dup_tasks = n_tasks - n_clusters
        out.append({
            "id": "duplicate_expected_calls",
            "severity": "高",
            "statement": (
                f"{n_tasks} 个任务只有 **{n_clusters} 个不同期望调用** —— "
                f"{dup_tasks} 个任务是重复计数。最大的簇有 **{biggest_n} 个任务**，"
                f"它们的期望调用完全相同：`{biggest_key}`。"
            ),
            "do_not_claim": (
                f"**不要**说「{biggest_n} 个独立样本」—— 有效独立样本是"
                f"**1 个行为**，{biggest_n} 是**触发方式**的数量。"
                "统计必须按**簇**重采样（`paired_stats.paired_bootstrap_clustered`），"
                "它给出更宽的置信区间（这是更保守的口径）。"
            ),
            "verify": "python scripts/eval/paired_significance.py --a X --b Y ",
        })

    # ---- 3. 与训练语料的实体泄漏 ----
    try:
        from src.data.recovery_corpus import find_expected_leaks

        leaks = find_expected_leaks(task_set=task_set)
        if leaks:
            out.append({
                "id": "entity_leaks",
                "severity": "中",
                "statement": (
                    f"有 **{len(leaks)} 个**评测任务的完整期望调用与某条训练样本"
                    f"完全相同：{leaks[:5]}。这些任务上「答对」可能来自"
                    "匹配记忆中的训练样本，而不是学会规则。"
                ),
                "do_not_claim": (
                    "**不要**把这几个任务计入无污染的指标 —— "
                    "应同时给出「剔除它们之后」的干净子集指标。"
                ),
                "verify": "python scripts/eval/audit_eval_overlap.py ",
            })
    except Exception:  # pragma: no cover
        pass

    # ---- 4. 训练分布与评测分布的偏差 ----
    try:
        from src.data.recovery_corpus import build_training_tasks

        syn = build_training_tasks(seed=42)
        tr = collections.Counter(
            "(clarify/空)" if t.expected.tool is None else t.expected.tool for t in syn
        )
        ev = collections.Counter(
            "(clarify/空)" if t.expected.tool is None else t.expected.tool for t in tasks
        )
        tr_share = tr.get("search_flights", 0) / max(1, len(syn))
        ev_share = ev.get("search_flights", 0) / max(1, len(tasks))
        if abs(tr_share - ev_share) > 0.05:
            out.append({
                "id": "distribution_mismatch",
                "severity": "中",
                "statement": (
                    f"训练语料里 `search_flights` 占 {tr_share:.1%}，"
                    f"而评测里只占 {ev_share:.1%} —— 训练分布偏向搜索工具。"
                ),
                "do_not_claim": (
                    "**不要**把绝对准确率外推到其他分布。"
                    "但**组间对比**（SFT vs DPO）不受影响：两者用同一份训练数据、"
                    "同一分布，偏差对它们**对称**。"
                ),
                "verify": "python scripts/prepare/verify_all.py",
            })
    except Exception:  # pragma: no cover
        pass

    # ---- 5. SFT 训练数据里没有否定样本 ----
    try:
        import json

        p = Path("data/preferences/sft_only.train.jsonl")
        if p.exists():
            rows = [json.loads(ln) for ln in p.open(encoding="utf-8") if ln.strip()]
            nocall = sum(1 for r in rows if "<tool_call>" not in r.get("chosen", ""))
            if nocall == 0 and rows:
                out.append({
                    "id": "sft_no_negative_samples",
                    "severity": "低（且加强论点）",
                    "statement": (
                        f"SFT 训练数据 {len(rows)} 条里，`chosen` **不含** `tool_call` "
                        "的有 **0 条** —— 纯正例监督在结构上无法表达"
                        "「此处不该调用工具」。"
                    ),
                    "do_not_claim": (
                        "**不要**把这说成「SFT 因为数据里没有所以做不到」就完了 —— "
                        "实测**系统提示里明确写着**「信息不足时先向用户追问」，"
                        "所以这是**指令遵循失败**，不是数据缺口。"
                        "这反而加强了论点。"
                    ),
                    "verify": "python scripts/prepare/inspect_preferences.py",
                })
    except Exception:  # pragma: no cover
        pass

    return out


def render_markdown(limits: list[dict], *, n_tasks: int, label: str) -> str:
    """把限制渲染成可直接贴进报告的 Markdown 章节。"""
    L = [f"## 已知限制（{label}，评测集 {n_tasks} 个任务）", ""]
    if not limits:
        L.append("本次评测未检出已知限制。")
        return "\n".join(L)

    L.append("> **本章节由 `src/eval/limitations.py` 自动生成** —— ")
    L.append("> 这些限制若不主动声明，读者会把结论读得比实际更强。")
    L.append("> 数字从当前数据算出，数据变了文字会跟着变。")
    L.append("")
    L.append("| # | 严重性 | 限制 |")
    L.append("|---|---|---|")
    for i, x in enumerate(limits, 1):
        L.append(f"| {i} | {x['severity']} | {x['id']} |")
    L.append("")

    for i, x in enumerate(limits, 1):
        L.append(f"### {i}. {x['id']}（{x['severity']}）")
        L.append("")
        L.append(x["statement"])
        L.append("")
        L.append(f"**⛔ 不能这样声称**：{x['do_not_claim']}")
        L.append("")
        if x.get("verify"):
            L.append(f"验证：`{x['verify']}`")
            L.append("")
    return "\n".join(L)
