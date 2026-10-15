"""定位「过度追问」的根因：训练语料里有多少任务**信息其实不足**。

## 要回答的问题

`sft_v7` 的剩余失败里，最大一类是 **「该调用但只做了澄清」62 条**。
模型在信息**足够**时停下追问 —— 这可能是：

* **(a) 训练语料本身有噪声**：某些「要调工具」的合成任务，
  其期望参数值**并不能从用户话或观测里推出**。
  那么模型学会「这种问法该追问」是**合理的** —— 是标注问题，不是模型问题。
* **(b) 模型容量限制**：语料没问题，但 1.5B 分不清细微的信息充分性差别。

这两者的修法完全不同，所以必须先分清楚。

## 判据

对每个「要调工具」的合成任务，检查每个期望参数值能否在
**用户话**或**观测**里找到依据：

* 值直接出现 → 有依据
* 三字码，且用户话里出现了对应中文城市名 → 有依据
* 其余 → **无依据（标注噪声）**

## 用法

```bash
python scripts/prepare/audit_parameter_grounding.py
python scripts/prepare/audit_parameter_grounding.py --json out.json
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

from src.env.tool_catalog import AIRPORT_CODES  # noqa: E402


def value_grounded(value, user_text: str, obs_text: str) -> tuple[bool, str]:
    """参数值是否有依据。返回 ``(是否有依据, 依据来源)``。

    >>> value_grounded("张伟", "乘客是张伟", "")
    (True, 'user')
    >>> value_grounded("PEK", "从北京出发", "")
    (True, 'user(城市名)')
    >>> value_grounded("PEK", "帮我订张票", "")
    (False, '无依据')
    >>> value_grounded("CZ4978", "帮我订", '{"flight_id": "CZ4978"}')
    (True, 'observation')
    """
    vs = str(value)
    if vs and vs in user_text:
        return True, "user"
    if vs and vs in obs_text:
        return True, "observation"
    # 三字码：用户话里出现对应中文城市名也算有依据
    cities = [c for c, code in AIRPORT_CODES.items() if code == vs]
    if cities and any(c in user_text for c in cities):
        return True, "user(城市名)"
    return False, "无依据"


def audit() -> dict:
    from src.data.recovery_corpus import build_training_tasks

    syn = build_training_tasks(seed=42)
    tool_tasks = [t for t in syn if t.expected.tool is not None]

    rows = []
    for t in tool_tasks:
        user_text = " ".join(x.content or "" for x in t.turns if x.role == "user")
        obs_text = json.dumps(
            [x.observation for x in t.turns if x.role == "tool"], ensure_ascii=False
        )
        for k, v in (t.expected.args or {}).items():
            ok, src = value_grounded(v, user_text, obs_text)
            rows.append({
                "task_id": t.task_id,
                "category": t.category,
                "param": k,
                "value": str(v),
                "grounded": ok,
                "source": src,
                "user": user_text[:60],
            })

    ungrounded = [r for r in rows if not r["grounded"]]
    bad_tasks = sorted({r["task_id"] for r in ungrounded})
    return {
        "n_tool_tasks": len(tool_tasks),
        "n_params": len(rows),
        "n_ungrounded_params": len(ungrounded),
        "n_ungrounded_tasks": len(bad_tasks),
        "ungrounded_task_ids": bad_tasks,
        "by_category": dict(collections.Counter(
            r["category"] for r in ungrounded)),
        "by_param": dict(collections.Counter(r["param"] for r in ungrounded)),
        "samples": ungrounded[:25],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="审计训练语料的参数依据")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    rep = audit()
    print("=" * 74)
    print("训练语料「参数依据」审计")
    print("=" * 74)
    print(f"  要调工具的合成任务      : {rep['n_tool_tasks']}")
    print(f"  期望参数总数            : {rep['n_params']}")
    print(f"  **无依据的参数          : {rep['n_ungrounded_params']}**"
          f"（涉及 {rep['n_ungrounded_tasks']} 个任务）")
    if rep["by_category"]:
        print(f"  按类别: {rep['by_category']}")
    if rep["by_param"]:
        print(f"  按参数: {rep['by_param']}")
    print()
    if rep["samples"]:
        print("  样例（这些任务**信息其实不足**，模型追问是合理的）：")
        print()
        print(f"    {'任务':<26} {'类别':<22} {'参数':<20} {'用户话'}")
        for r in rep["samples"]:
            print(f"    {r['task_id']:<26} {r['category']:<22} "
                  f"{r['param']:<20} {r['user']!r}")
        print()
        print("  ⚠️ 若这些任务确实信息不足，那么「模型过度追问」的一部分是")
        print("     **标注噪声**而不是模型缺陷 —— 修法是给它们补上信息依据，")
        print("     或者在训练时把它们标成 clarify。")
    else:
        print("  ✅ 所有「要调工具」任务的参数都有依据 —— 语料无噪声。")
        print("     那么 62 条「过度追问」更可能是**容量限制**。")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        print(f"\n  已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
