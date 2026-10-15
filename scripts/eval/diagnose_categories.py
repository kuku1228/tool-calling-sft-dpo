"""分类别失败诊断 —— 回答「为什么某一类准确率是 0%」。

## 为什么需要它

v3 基线的读数里有一个非常刺眼的模式：

    类别                 需要机场三字码?   accuracy
    weather_advisory     否（中文城市名）   90.0%
    baggage              否（订单号）       67.9%
    cancellation         否（订单号）       62.5%
    error_correction     部分               48.3%
    ------------------------------------------------
    flight_search        **是**              0.00%
    multi_tool           **是**              0.00%
    multi_turn           **部分**            12.5%
    booking_flow         否（flight_id 由观测给）  0.00%

**「0.00%」不是「差」，而是「一个都没对」** —— 这个量级说明是**结构性失败**，
不是能力不足。而按类别看不出原因（所有类别都只报一个数字）。

这个脚本把每个失败样本的**实际输出**与期望对照，回答三个具体问题：

1. `flight_search` 的 0% 是**格式错误**（发不出 tool_call）还是
   **参数值错**（三字码仍然编造）？
2. `booking_flow` 的 0% 是什么？它期望 `book_flight(flight_id=...)`，
   而 `flight_id` **在观测里给了** —— 模型是没读到、还是多调了一步？
3. `multi_tool` 的 0% 是**没调天气**还是**没调航班**？
   （v3 要求同时调两个工具，v1 只要求一个）

## 用法

```bash
python scripts/eval/diagnose_categories.py \
    --label baseline 
```
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.data.task_manifest import load_tasks_by_version  # noqa: E402
from src.eval.grader import grade  # noqa: E402
from src.eval.tool_call_parser import extract_tool_calls  # noqa: E402


def _calls(text: str) -> list:
    """抽出工具调用。

    ⚠️ ``extract_tool_calls`` 返回的是 **``ParseResult``**，
    不是 list —— 调用列表在 ``.calls`` 里，元素是 ``ToolCall``，
    字段是 ``name`` / ``arguments``（**不是 ``args``**）。

    我在这里连续猜错了三次 API（``classify`` 的签名、
    ``extract_tool_calls`` 的返回类型、``ToolCall.args``）。
    **这个项目里已经因为「猜 API」栽过两次**，所以这次改成
    「先运行 inspect 看清楚再写」，并把结论记在这里。

    >>> _calls('<tool_call>{"name": "f", "arguments": {"a": 1}}</tool_call>')[0].name
    'f'
    >>> _calls("没有调用") == []
    True
    """
    try:
        return list(extract_tool_calls(text).calls)
    except Exception:
        return []


def _load_predictions(label: str, exp_root: Path) -> list[dict]:
    d = exp_root / label
    for name in ("predictions.jsonl", "predictions.mock.jsonl"):
        p = d / name
        if p.exists():
            return [json.loads(ln) for ln in p.open(encoding="utf-8") if ln.strip()]
    raise FileNotFoundError(
        f"{d} 下找不到 predictions.jsonl。"
        f"（目录内容: {sorted(x.name for x in d.iterdir()) if d.exists() else '不存在'}）"
    )


def airport_code_report(
    task, text: str, task_set: str = "v3"
) -> dict:
    """检查模型给出的三字码是否**在训练语料里出现过**。

    这是 B0 修复（把机场池从 20 扩到 60）是否**真的生效**的直接检验：
    * 若模型编造的码**不在**训练池里 → 覆盖没生效
    * 若模型给出的码**在**训练池里但错了 → 映射学错了（不同的问题）

    >>> class T:
    ...     class E: tool = "search_flights"; args = {"origin": "PEK"}
    ...     expected = E()
    >>> airport_code_report(T(), '<tool_call>{"name":"search_flights",'
    ...     '"arguments":{"origin":"XYZ"}}</tool_call>')["codes"]
    ['XYZ']
    """
    expected = set()
    for v in (task.expected.args or {}).values():
        if isinstance(v, str) and re.fullmatch(r"[A-Z]{3}", v):
            expected.add(v)
    got: set[str] = set()
    for c in _calls(text):
        for v in (getattr(c, "arguments", None) or {}).values():
            if isinstance(v, str) and re.fullmatch(r"[A-Z]{3}", v):
                got.add(v)
    return {"expected_codes": sorted(expected), "codes": sorted(got)}


def diagnose(label: str, *, task_set: str = "v3", exp_root: str = "experiments") -> int:
    root = Path(exp_root)
    rows = _load_predictions(label, root)
    tasks = {t.task_id: t for t in load_tasks_by_version(task_set)}

    # 按类别聚合
    by_cat: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        t = tasks.get(r.get("task_id", ""))
        if t is None:
            continue
        text = r.get("text", "")
        # ⚠️ 用 `grade()` 而不是直接调 `classify()` ——
        # `classify` 需要 8 个参数（calls / arg_match / valid_tool_names …），
        # 而那些**正是 `grade()` 内部已经算好的**。直接调它等于把
        # 评分逻辑重写一遍，且必然与在线评测口径不一致。
        g = grade(t, text, finish_reason=r.get("finish_reason", ""))
        cls = g.classification
        by_cat[t.category].append({
            "task_id": t.task_id,
            "passed": bool(g.passed),
            "category_err": cls.category,
            "subtype": (cls.detail.subtype if cls.detail else "-") or "-",
            "fmt_ok": bool(g.format_ok),
            "text": text,
            "task": t,
        })

    print("=" * 78)
    print(f"分类别失败诊断：{label}（{len(rows)} 个样本）")
    print("=" * 78)

    # ---- 总览：每类的格式正确率 vs 任务成功率 ----
    print("\n### 每类：格式正确率 vs 任务成功率（区分「发不出调用」与「调错」）")
    print()
    # ⚠️ 格式正确率必须**按任务类型**判：
    # 「应先追问」任务的正确输出是自然语言追问，本来就没有 tool_call。
    # 旧口径一律要求 tool_call，会把它们的**正确答案**判成格式错误 ——
    # 实测 `cancellation` 因此显示「成功率 62.5% / 格式 0.0%」，
    # 若不分开报会误读成「模型连格式都不会」。
    from src.eval.tool_call_parser import expected_format_ok

    print(f"  {'类别':<18} {'样本':>4} {'成功率':>7} {'格式(按任务)':>12} "
          f"{'格式(旧口径)':>12}  {'→ 诊断':<26}")
    for cat in sorted(by_cat):
        rs = by_cat[cat]
        n = len(rs)
        acc = sum(1 for x in rs if x["passed"]) / n
        fmt = sum(
            1 for x in rs
            if expected_format_ok(
                is_clarify_task=x["task"].expected.tool is None, text=x["text"])
        ) / n
        fmt_old = sum(1 for x in rs if x["fmt_ok"]) / n
        if acc == 0.0 and fmt < 0.3:
            diag = "**发不出格式合格的调用**"
        elif acc == 0.0 and fmt >= 0.5:
            diag = "**能发调用但参数/工具错**"
        elif acc == 0.0:
            diag = "格式与参数都有问题"
        else:
            diag = ""
        print(f"  {cat:<18} {n:>4} {acc:>6.1%} {fmt:>11.1%} {fmt_old:>11.1%}  {diag}")

    # ---- 每类的错误子类型 ----
    print("\n### 每类的错误子类型（失败样本）")
    for cat in sorted(by_cat):
        rs = [x for x in by_cat[cat] if not x["passed"]]
        if not rs:
            continue
        sub = collections.Counter(x["subtype"] for x in rs)
        top = ", ".join(f"{k}={v}" for k, v in sub.most_common(4))
        print(f"  {cat:<18} {top}")

    # ---- 问题 1：三字码 ----
    print("\n### 问题 1：机场三字码 —— 模型给出的码是否在训练池里？")
    try:
        from src.data.recovery_corpus import build_training_tasks

        syn = build_training_tasks(seed=42)
        train_codes = set()
        for x in syn:
            for v in (x.expected.args or {}).values():
                if isinstance(v, str) and re.fullmatch(r"[A-Z]{3}", v):
                    train_codes.add(v)
        # 训练语料覆盖的码（含观测里出现的 flight_id 前缀不算）
        print(f"  训练语料用过的三字码: {len(train_codes)} 个")

        for cat in ("flight_search", "multi_tool", "multi_turn"):
            rs = by_cat.get(cat, [])
            if not rs:
                continue
            invented = collections.Counter()
            in_pool_wrong = 0
            for x in rs:
                rep = airport_code_report(x["task"], x["text"])
                for c in rep["codes"]:
                    if c in rep["expected_codes"]:
                        continue
                    if c not in train_codes:
                        invented[c] += 1
                    else:
                        in_pool_wrong += 1
            print(f"  {cat}:")
            print(f"     编造的码（**不在训练池**，共 {sum(invented.values())} 次）: "
                  f"{dict(invented.most_common(6))}")
            print(f"     在训练池里但用错的: {in_pool_wrong} 次")
            if invented:
                print(f"     → 说明 **B0 的覆盖修复没生效**，或模型没学会映射")
            elif in_pool_wrong:
                print(f"     → 训练池覆盖够，但**映射学错了**（是学习问题，不是数据问题）")
    except Exception as exc:
        print(f"  ⚠️ 无法检查三字码：{exc}")

    # ---- 问题 2：booking_flow ----
    print("\n### 问题 2：booking_flow 为什么 0%？")
    rs = by_cat.get("booking_flow", [])
    if rs:
        patterns = collections.Counter()
        for x in rs:
            t = x["text"]
            exp_tool = x["task"].expected.tool
            names = [getattr(c, "name", "?") for c in _calls(t)]
            if not names:
                patterns["没有任何工具调用"] += 1
            elif exp_tool in names:
                patterns["调了正确的工具但参数不对"] += 1
            elif "search_flights" in names:
                patterns["只调了 search_flights（没走到 book_flight）"] += 1
            else:
                patterns[f"调了其他工具: {names[0]}"] += 1
        for k, v in patterns.most_common():
            print(f"     {k}: {v} 次")
        print(f"  期望的工具: {rs[0]['task'].expected.tool}")
        print(f"  对照：这些任务在 turn1 的**观测里给了 flight_id**")
    else:
        print("  （无此类样本）")

    # ---- 问题 3：multi_tool ----
    print("\n### 问题 3：multi_tool 是没调天气还是没调航班？")
    rs = by_cat.get("multi_tool", [])
    if rs:
        seen = collections.Counter()
        for x in rs:
            names = {getattr(c, "name", "?") for c in _calls(x["text"])}
            has_w = "get_weather" in names
            has_f = "search_flights" in names
            key = ("调了天气" if has_w else "没调天气") + " + " + \
                  ("调了航班" if has_f else "没调航班")
            seen[key] += 1
        for k, v in seen.most_common():
            print(f"     {k}: {v} 次")
        print(f"  期望：**两个都要**（also_required=('get_weather',)）")
    else:
        print("  （无此类样本）")

    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="分类别失败诊断")
    ap.add_argument("--label", required=True)
    ap.add_argument("--task-set", default="v3", choices=["v1", "v3"])
    ap.add_argument("--exp-root", default="experiments")
    args = ap.parse_args(argv)
    try:
        return diagnose(args.label, task_set=args.task_set, exp_root=args.exp_root)
    except FileNotFoundError as exc:
        print(f"  ✗ {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
