"""生成冻结任务集的指纹文件 —— `experiments/_frozen_fingerprint.json`。

## 为什么需要这个脚本（这是一个真实的缺口）

`tests/test_methodology.py::test_frozen_eval_tasks_unchanged` 的断言消息写着
「缺少冻结指纹文件，无法验证（**应由 scripts 生成**）」——
**但项目里根本没有生成它的脚本**。那个文件是我当初手工造的，
于是：

* 它一旦丢失（例如部署脚本的保护机制跳过了它），**无法重建**
* 云端 `test_frozen_eval_tasks_unchanged` 因此挂掉（实测 159 个测试挂 1 个）
* 而错误的断言消息**指向一个不存在的脚本**，让人以为去跑一下就好

**这类「消息承诺了一个不存在的能力」是危险的** —— 人会相信它然后卡住。

## 它做什么

对 v1 任务集（100 个**已冻结**的历史任务）逐个算指纹：

    sha256(json.dumps(t.to_dict(), sort_keys=True))[:16]

## 为什么要冻结 v1

v1 是**历史结果的基准**（baseline 50.25% / sft_v2 / sft_v3 / dpo_* 都跑在它上面）。
那些数字虽然已作废（见 `HANDOFF_next_steps.md` §1.2），
但**如果 v1 本身被改动**，连「为什么作废」都无法复核了 ——
作废的理由正是「跑在 v1 上」，那 v1 必须是可证明未变的。

所以这个指纹的作用是：**证明 v1 从未被改动**。

## 用法

```bash
# 生成（首次，或确认 v1 合法改动后重新冻结）
python scripts/prepare/freeze_v1_tasks.py

# 只检查是否一致，不写文件（用于 CI）
python scripts/prepare/freeze_v1_tasks.py --check
```

⚠️ **正常情况下不要重新生成**。`--check` 报不一致说明 v1 被改动了，
那要先判断改动是否合法（历史结论的可比性依赖它）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

DEFAULT_OUT = "experiments/_frozen_fingerprint.json"


def fingerprint(task) -> str:
    """单个任务的指纹（16 位十六进制）。

    用 ``sort_keys=True`` 保证字段顺序不影响结果 ——
    否则改动 ``to_dict()`` 的字段顺序会**假报**任务被改。

    >>> class T:
    ...     def to_dict(self): return {"a": 1, "b": 2}
    >>> fingerprint(T()) == fingerprint(T())
    True
    >>> len(fingerprint(T()))
    16
    """
    payload = json.dumps(task.to_dict(), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def compute(version: str = "v1") -> dict[str, str]:
    """算出某个版本任务集的指纹表。

    >>> fp = compute("v1")
    >>> len(fp)
    100
    >>> all(len(v) == 16 for v in fp.values())
    True
    """
    from src.data.task_manifest import load_tasks_by_version

    return {t.task_id: fingerprint(t) for t in load_tasks_by_version(version)}


def compare(current: dict[str, str], frozen: dict[str, str]) -> dict:
    """比对并将差异分类。

    * ``changed`` —— 两边都有但指纹不同（**v1 被改动，历史结论不再可比**）
    * ``missing`` —— 冻结文件里有、当前没有（任务被删）
    * ``added``   —— 当前有、冻结文件里没有（任务被加）

    >>> compare({"a": "1"}, {"a": "1"})
    {'changed': [], 'missing': [], 'added': []}
    >>> compare({"a": "2"}, {"a": "1"})["changed"]
    ['a']
    >>> compare({}, {"a": "1"})["missing"]
    ['a']
    >>> compare({"b": "1"}, {})["added"]
    ['b']
    """
    return {
        "changed": sorted(k for k in current if k in frozen and current[k] != frozen[k]),
        "missing": sorted(k for k in frozen if k not in current),
        "added": sorted(k for k in current if k not in frozen),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="生成/校验冻结任务集指纹")
    ap.add_argument("--version", default="v1",
                    help="要冻结的任务集版本（默认 v1 —— 历史结果的基准）")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true",
                    help="只校验，不写文件（不一致时返回非 0）")
    args = ap.parse_args(argv)

    out = Path(args.out)
    current = compute(args.version)
    print(f"  {args.version} 任务数: {len(current)}")

    if args.check:
        if not out.exists():
            print(f"  ✗ 指纹文件不存在: {out}")
            print(f"     生成：python {Path(__file__).name}")
            return 1
        frozen = json.loads(out.read_text(encoding="utf-8"))
        diff = compare(current, frozen)
        bad = diff["changed"] + diff["missing"] + diff["added"]
        if not bad:
            print(f"  ✅ 与 {out} 完全一致 —— {args.version} 未被改动")
            return 0
        print(f"  ❌ {args.version} 与冻结指纹不一致：")
        for key in ("changed", "missing", "added"):
            if diff[key]:
                print(f"     {key}: {len(diff[key])} 个 {diff[key][:5]}")
        print("\n  ⚠️ 不要直接重新生成。先判断改动是否合法：")
        print(f"     {args.version} 是**历史结果的基准**，改动它会让那些结果")
        print("     无法复核（包括「为什么作废」这个判断本身）。")
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(current, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"  ✅ 已写入 {out}")
    print(f"     校验：python {Path(__file__).name} --check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
