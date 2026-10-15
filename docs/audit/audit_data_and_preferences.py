"""数据污染审计（修正版）：正确读取 Task.turns / ExpectedCall.args。

重点回答三个问题：
  1. 评测任务与训练语料在「话术 / 期望调用 / 实体值」三个层次上是否重合
  2. broken_json_* 这类非 trainsyn_ 前缀的 task_id 是什么，是否构成泄漏
  3. 训练语料的实体覆盖是否与评测分布匹配（B0 修的问题是否真的修好了）
"""
from __future__ import annotations

import collections
import json
import pathlib
import re
import sys

#: 仓库根目录 —— 本脚本位于 ``docs/audit/``，所以要上溯两层。
#: （原先在 ``_audit/`` 下只需一层；移动目录后必须同步改，否则
#: ``import src`` 会失败，而失败信息是 ModuleNotFoundError，容易误判成
#: 「依赖没装」。）
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "docs" / "audit" / "audit_report.txt"
lines: list[str] = []


def w(s: str = "") -> None:
    lines.append(s)


def load_rows(name: str) -> list[dict]:
    p = ROOT / "data" / "preferences" / name
    return [json.loads(ln) for ln in p.open(encoding="utf-8") if ln.strip()] if p.is_file() else []


def call_of(text: str):
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text or "", re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
    except Exception:
        return None
    args = obj.get("arguments") or obj.get("args") or {}
    return obj.get("name"), (tuple(sorted(args.items())) if isinstance(args, dict) else ())


def user_texts(prompt: str) -> set[str]:
    return {m.group(1).strip()
            for m in re.finditer(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", prompt or "", re.S)}


def main() -> None:
    from src.data.task_manifest import load_tasks_by_version
    from src.data.task_manifest_v2 import load_tasks_v3
    from src.data.recovery_corpus import build_training_tasks

    v3 = load_tasks_v3()
    v1 = load_tasks_by_version("v1")
    syn = build_training_tasks(seed=42)

    # ---- 评测侧提取 ----
    eval_user, eval_calls, eval_vals = set(), collections.Counter(), collections.defaultdict(set)
    eval_ids = set()
    for t in v3:
        eval_ids.add(t.task_id)
        for turn in t.turns:
            if turn.role == "user":
                eval_user.add(turn.content.strip())
        exp = t.expected
        if exp and exp.tool is not None:
            eval_calls[(exp.tool, tuple(sorted((exp.args or {}).items())))] += 1
            for k, v in (exp.args or {}).items():
                if isinstance(v, str):
                    eval_vals[k].add(v)

    # ---- 合成语料侧提取 ----
    syn_user, syn_calls, syn_vals = set(), set(), collections.defaultdict(set)
    syn_ids = set()
    for t in syn:
        syn_ids.add(t.task_id)
        for turn in t.turns:
            if turn.role == "user":
                syn_user.add(turn.content.strip())
        exp = t.expected
        if exp and exp.tool is not None:
            syn_calls.add((exp.tool, tuple(sorted((exp.args or {}).items()))))
            for k, v in (exp.args or {}).items():
                if isinstance(v, str):
                    syn_vals[k].add(v)

    w("=" * 80)
    w("数据污染审计（修正版）")
    w("=" * 80)

    w(f"\n[A] 规模")
    w(f"    v1 任务 {len(v1)}   v3 任务 {len(v3)}   合成语料任务 {len(syn)}")
    w(f"    v3 用户话术（逐条去重） {len(eval_user)}")
    w(f"    合成语料用户话术（逐条去重） {len(syn_user)}")

    w(f"\n[B] 话术层重合：合成语料 vs v3 评测")
    inter = syn_user & eval_user
    w(f"    交集 {len(inter)}  <-- 期望 0")
    for s in list(inter)[:5]:
        w(f"      {s[:70]}")

    w(f"\n[C] 期望调用层重合：合成语料 vs v3 评测（层级 2）")
    hit = syn_calls & set(eval_calls)
    w(f"    合成语料有期望调用 {len(syn_calls)} 种，v3 有 {len(eval_calls)} 种")
    w(f"    完全相同 {len(hit)} 种  <-- 期望 0")
    for k in list(hit)[:8]:
        w(f"      {k[0]}  {dict(k[1])}")

    w(f"\n[D] 偏好数据（真正喂进训练的） vs v3 评测")
    TRAIN = ["dpo_joint.train.jsonl", "sft_only.train.jsonl", "sft_clarify.train.jsonl",
             "dpo_clarify.train.jsonl", "dpo_format.train.jsonl", "dpo_recovery.train.jsonl"]
    for name in TRAIN:
        rows = load_rows(name)
        if not rows:
            continue
        pu, pc, pv = set(), set(), collections.defaultdict(set)
        for r in rows:
            pu |= user_texts(r["prompt"])
            c = call_of(r.get("chosen", ""))
            if c:
                pc.add(c)
                for k, v in c[1]:
                    if isinstance(v, str):
                        pv[k].add(v)
        w(f"    {name}")
        w(f"        话术交集           {len(pu & eval_user)}")
        w(f"        期望调用交集       {len(pc & set(eval_calls))}")
        missing = {k: len(eval_vals[k] - pv.get(k, set())) for k in eval_vals}
        w(f"        实体缺失（评测有/训练无）：{missing}")

    w(f"\n[E] task_id 命名空间")
    ids = set()
    for name in TRAIN:
        ids |= {r.get("task_id", "") for r in load_rows(name)}
    bad = sorted(i for i in ids if not i.startswith("trainsyn_"))
    w(f"    偏好数据唯一 task_id {len(ids)}，非 trainsyn_ 前缀 {len(bad)}")
    w(f"    前缀分布：{collections.Counter(i.split('_')[0] for i in ids)}")
    w(f"    越界样例：{bad[:8]}")
    w(f"    这些是否与 v3 评测 task_id 相交？{len(set(bad) & eval_ids)}")

    # broken_json 的 prompt 是否落进评测分布
    rows = load_rows("dpo_joint.train.jsonl")
    bj = [r for r in rows if r.get("task_id", "").startswith("broken_json")]
    w(f"\n    broken_json_* 条目 {len(bj)} 条；其话术与评测交集 "
      f"{len({s for r in bj for s in user_texts(r['prompt'])} & eval_user)}")
    if bj:
        w(f"    样例 task_id={bj[0]['task_id']} pair_type={bj[0]['pair_type']}")
        w(f"      chosen: {bj[0]['chosen'][:90]}")
        w(f"      rejected: {bj[0]['rejected'][:90]}")

    w(f"\n[F] 合成语料实体覆盖（B0 声称已修到 0 缺失）")
    missing_syn = {k: sorted(eval_vals[k] - syn_vals.get(k, set())) for k in eval_vals}
    for k, v in missing_syn.items():
        mark = "  <-- 缺失" if v else ""
        w(f"    {k:<18} 评测 {len(eval_vals[k]):>3}  合成语料 {len(syn_vals.get(k,set())):>3}  "
          f"缺失 {len(v):>3}{mark}  {v[:6]}")

    w(f"\n[G] 有效样本量")
    w(f"    v3 任务 {len(v3)}，不同期望调用 {len(eval_calls)}，"
      f"重复计数 {len(v3) - len(eval_calls) - sum(1 for t in v3 if not (t.expected and t.expected.tool))}")
    clarify = sum(1 for t in v3 if not (t.expected and t.expected.tool))
    w(f"    无期望调用的任务（应先追问类） {clarify} 个 —— 它们共享同一个「行为」")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
