"""逐条诊断 ``wrong_value`` 失败：期望 vs 实际。

## 为什么单独做

`sft_v7` 的失败里 **23/28 是 `wrong_value`**，其余都不是问题
（决策错误只有 2 条过度追问、0 条该追问却调工具）。
所以「剩下 6.6% 到底错在哪」= 「这 23 条的参数值为什么错」。

## 字段位置（实测确认，别猜）

``eval_report.json`` → ``per_task[i]``：

* ``outcomes``         ``[bool,...]``
* ``error_categories`` ``["PARAM_ERROR",...]``
* ``predictions[j]``   ``{passed, category, classification, predicted, ...}``
  - **模型实际输出在 ``predicted``**（``[{name, arguments}, ...]``）
  - 子类型在 ``classification.detail.subtype``

⚠️ 我在这上面猜错过两次（``categories`` 应为 ``error_categories``；
实际输出在 ``predicted`` 而不是 ``text``），每次都让结果**看起来没问题**。

## 用法

```bash
python scripts/eval/diagnose_wrong_values.py --label sft_v7 
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


def collect(rep: dict, tasks: dict, *, subtype: str = "wrong_value") -> list[dict]:
    rows: list[dict] = []
    for p in rep.get("per_task") or []:
        tid = p.get("task_id", "")
        t = tasks.get(tid)
        if t is None:
            continue
        preds = p.get("predictions") or []
        for i, ok in enumerate(p.get("outcomes") or []):
            if ok:
                continue
            pr = preds[i] if i < len(preds) else {}
            cl = (pr.get("classification") or {})
            det = (cl.get("detail") or {})
            if (det.get("subtype") or cl.get("category")) != subtype:
                continue
            got = pr.get("predicted") or []
            got_args = (got[0].get("arguments") if got and isinstance(got[0], dict)
                        else None)
            rows.append({
                "task_id": tid,
                "sample_index": i,
                "task_category": t.category,
                "tags": list(t.tags),
                "n_turns": len(t.turns),
                "expected_tool": t.expected.tool,
                "expected_args": t.expected.args or {},
                "got_tool": (got[0].get("name") if got and isinstance(got[0], dict)
                             else None),
                "got_args": got_args,
                "user": t.turns[0].content[:50],
            })
    return rows


def render(rows: list[dict], tasks: dict) -> str:
    L = [f"## `wrong_value` 逐条诊断（{len(rows)} 条）", ""]
    # 按任务聚合
    by_task: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_task[r["task_id"]].append(r)

    L.append(f"涉及 **{len(by_task)} 个任务**"
             f"（每个任务 4 采样全错 = 系统性错误，不是采样噪声）")
    L.append("")

    # 按参数名归因
    param_hist: collections.Counter = collections.Counter()
    for r in rows:
        for k in r["expected_args"]:
            exp = r["expected_args"].get(k)
            got = (r["got_args"] or {}).get(k)
            if str(exp) != str(got):
                param_hist[k] += 1
    L.append("### 按「出错的参数名」")
    L.append("")
    L.append("| 参数 | 出错次数 |")
    L.append("|---|---|")
    for k, v in param_hist.most_common():
        L.append(f"| `{k}` | {v} |")
    L.append("")

    L.append("### 逐任务：期望 vs 实际")
    L.append("")
    for tid in sorted(by_task):
        rs = by_task[tid]
        r = rs[0]
        t = tasks.get(tid)
        L.append(f"**`{tid}`** [{r['task_category']}] tags={r['tags']} "
                 f"轮数={r['n_turns']} — {len(rs)} 个采样全错")
        L.append("")
        L.append(f"- 用户话：{r['user']!r}")
        if r["n_turns"] > 2 and t is not None:
            L.append(f"- 纠正消息：{t.turns[2].content!r}")
        L.append(f"- 期望调用：`{r['expected_tool']} {r['expected_args']}`")
        L.append(f"- 实际调用：`{r['got_tool']} {r['got_args']}`")
        # 逐参数差异
        diffs = []
        for k, exp in r["expected_args"].items():
            got = (r["got_args"] or {}).get(k, "<缺失>")
            if str(exp) != str(got):
                diffs.append(f"`{k}`: 期望 `{exp}` → 实际 `{got}`")
        if diffs:
            L.append(f"- **差异**：{'；'.join(diffs)}")
        L.append("")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="wrong_value 逐条诊断")
    ap.add_argument("--label", required=True)
    ap.add_argument("--task-set", default="v3", choices=["v1", "v3"])
    ap.add_argument("--exp-root", default="experiments")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    from src.data.task_manifest import load_tasks_by_version

    tasks = {t.task_id: t for t in load_tasks_by_version(args.task_set)}
    d = Path(args.exp_root) / args.label
    rep = None
    for name in ("eval_report.json", "eval_report.mock.json"):
        p = d / name
        if p.exists():
            rep = json.loads(p.read_text(encoding="utf-8"))
            break
    if rep is None:
        print(f"  ✗ {d} 下找不到 eval_report*.json")
        return 1

    rows = collect(rep, tasks)
    md = render(rows, tasks)
    print(md)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"\n  已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
