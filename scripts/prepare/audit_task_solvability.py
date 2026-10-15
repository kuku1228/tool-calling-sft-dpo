"""审计：有没有**不可能完成**的任务 —— 期望值无法从给定的信息推出。

## 为什么需要它（实测查出 2 个，都是我的 bug）

v3 的 106 个任务里有两个的期望值**用户从未提供过**：

```
taskv2_0093  用户: 「后天从青岛到长沙的航班有哪些」
             BASE_DATE_V2 = 2026-10-06 → 「后天」应是 2026-10-08
             期望 date   = 2026-11-08        ← 差 33 天，用户从未说过

taskv2_0100  用户: 「查一下订单 bk8e51a3 的行李额度」
             期望 booking_id = BK3A5C18      ← 与用户原话完全无关
```

第二个的根因很具体：我做全局订单号替换时**只改了期望、漏改了用户话**。

**危害**：这两个任务**永远不可能答对**，而它们在报告里表现为
「模型参数值错」—— 也就是**把标注错误记成了模型缺陷**。
`taskv2_0093` 还进了「过度追问」的统计里。

## 判据

对每个任务，期望参数的每个值必须能在**该任务提供的全部文本**里找到依据：

* 用户各轮话（含纠正轮）
* 工具观测（JSON）
* 三字码：用户话里出现对应中文城市名也算
* 相对日期（后天/明天…）：只做**提示**，不算硬失败
  （那是模型的换算任务，不是标注错误；但会单列出来供人工看）

## 用法

```bash
python scripts/prepare/audit_task_solvability.py 
python scripts/prepare/audit_task_solvability.py  --strict   # 有硬失败则退出码非 0
```
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.env.tool_catalog import AIRPORT_CODES  # noqa: E402

#: 相对日期词 —— 出现它们时该参数标为「需换算」而不是「无依据」
RELATIVE_DATE_WORDS = (
    "后天", "明天", "今天", "昨天", "大后天",
    "本周", "下周", "上周", "这周", "周一", "周二", "周三", "周四",
    "周五", "周六", "周日", "周天",
)


def _task_texts(task) -> tuple[str, str, str]:
    """返回 ``(所有非工具轮文本, 观测 JSON, 纠正轮文本)``。"""
    parts, obs_parts, corrections = [], [], []
    for i, tn in enumerate(task.turns):
        if tn.role == "tool":
            obs_parts.append(json.dumps(tn.observation or {}, ensure_ascii=False))
        else:
            parts.append(tn.content or "")
            # 第 3 轮（index>=2）视为纠正轮
            if i >= 2 and tn.role == "user":
                corrections.append(tn.content or "")
    return " ".join(parts), " ".join(obs_parts), " ".join(corrections)


def audit_task(task) -> dict:
    """检查单个任务的期望值是否可推导。

    Returns:
        ``{"task_id", "hard_fails": [...], "needs_conversion": [...], "ok": bool}``

    ⚠️ **判据要允许「规范化」**，否则会误报。

    实测：`taskv2_0100` 的用户说 `bk8e51a3`、期望 `BK8E51A3` ——
    那只是**大写化**，是合理要求（该任务的纠正消息正是
    「订单号要写成大写 BK 开头的格式」）。把它们判为「无依据」
    是误报，而**误报会让人忽略这个检查**。

    所以比对用 ``casefold()``：大小写差异不算无依据。

    >>> class T:
    ...     task_id = "t"; category = "c"
    ...     class E: tool="f"; args={"a": "X"}
    ...     expected = E()
    ...     class turns: pass
    >>> T.turns = [type("Turn", (), {"role":"user","content":"给我 X","observation":None})()]
    >>> audit_task(T())["ok"]
    True
    >>> class T2(T):
    ...     class E: tool="f"; args={"a": "BK8E51A3"}
    ...     expected = E()
    >>> T2.turns = [type("Turn", (), {"role":"user","content":"订单 bk8e51a3","observation":None})()]
    >>> audit_task(T2())["ok"]      # 大写化不算无依据
    True
    """
    user_text, obs_text, corr_text = _task_texts(task)
    all_text = f"{user_text} {corr_text}"
    hay = f"{all_text} {obs_text}".casefold()
    hard_fails, needs_conv = [], []

    for k, v in (task.expected.args or {}).items():
        vs = str(v)
        # 大小写不敏感比对：允许「规范化」（如订单号大写化）
        if vs and vs.casefold() in hay:
            continue
        # 三字码：用户话里有对应中文城市名
        cities = [c for c, code in AIRPORT_CODES.items() if code == vs]
        if cities and any(c in all_text for c in cities):
            continue
        # 相对日期：需要模型换算，不算硬失败
        if k == "date" and any(w in all_text for w in RELATIVE_DATE_WORDS):
            needs_conv.append((k, vs))
            continue
        # 机场码三字码大写比对（用户可能写小写）
        if re.fullmatch(r"[A-Za-z]{3}", vs) and vs.upper() in obs_text.upper():
            continue
        hard_fails.append((k, vs))

    return {
        "task_id": task.task_id,
        "category": getattr(task, "category", ""),
        "hard_fails": hard_fails,
        "needs_conversion": needs_conv,
        "ok": not hard_fails,
        "user": (task.turns[0].content[:56] if task.turns else ""),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="审计任务是否可解")
    ap.add_argument("--task-set", default="v3", choices=["v1", "v3"])
    ap.add_argument("--strict", action="store_true",
                    help="有硬失败时退出码非 0")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    from src.data.task_manifest import load_tasks_by_version

    tasks = load_tasks_by_version(args.task_set)
    rows = [audit_task(t) for t in tasks]
    bad = [r for r in rows if not r["ok"]]
    conv = [r for r in rows if r["needs_conversion"]]

    print("=" * 74)
    print(f"任务可解性审计（{args.task_set}，{len(tasks)} 个任务）")
    print("=" * 74)
    print(f"  **期望值无法从给定信息推出的任务：{len(bad)}**")
    print(f"  需要日期换算（相对日期，不算错误）：{len(conv)}")
    print()

    if bad:
        print("❌ 这些任务**永远不可能答对**（标注错误，不是模型缺陷）：")
        print()
        for r in bad:
            print(f"  `{r['task_id']}` [{r['category']}]")
            print(f"      用户话：{r['user']!r}")
            for k, v in r["hard_fails"]:
                print(f"      ❌ 期望 `{k} = {v}` 在用户话/观测/纠正轮里都找不到")
        print()
        print("  这类任务会让报告把**标注错误**记成**模型缺陷**，必须修。")
    else:
        print("✅ 所有任务的期望值都能从给定信息推出。")

    if conv:
        print()
        print(f"ℹ️ 以下 {len(conv)} 个任务含相对日期，需要模型自己换算")
        print("   （这是**测试内容**，不是标注错误）：")
        for r in conv[:8]:
            ks = ", ".join(f"{k}={v}" for k, v in r["needs_conversion"])
            print(f"     `{r['task_id']}`  {r['user']!r}  → {ks}")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"bad": bad, "needs_conversion": conv}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        print(f"\n  已写入 {out}")

    return 1 if (bad and args.strict) else 0


if __name__ == "__main__":
    raise SystemExit(main())
