"""检查偏好数据集：统计、长度、prompt 一致性、抽样人工检查。

用法
----
::

    python scripts/prepare/inspect_preferences.py
    python scripts/prepare/inspect_preferences.py --data-dir data/preferences --show 5
    python scripts/prepare/inspect_preferences.py --file data/preferences/dpo_joint.train.jsonl

为什么这个脚本很重要
--------------------
偏好数据的错误很难在训练 loss 上看出来 ——
如果 chosen 与 rejected 语义其实相同，loss 会稳定在 log 2 附近，
看起来「训练正常」，但模型什么也没学到。

本脚本把四类问题显式暴露出来：

1. **两侧语义等价**（样本对噪声）
2. **chosen 格式不合规**（教坏格式）
3. **prompt 与评测分布不一致**（训练学到的条件反射在评测时不被激活）
4. **长度超出 max_length**（截断后两侧可能变成一样）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import CFG  # noqa: E402
from src.data.preference_builder import PAIR_TYPE_LABELS_ZH, read_jsonl  # noqa: E402
from src.data.task_manifest import load_tasks  # noqa: E402
from src.env.tool_catalog import VALID_TOOL_NAMES  # noqa: E402
from src.eval.equivalence import equivalent  # noqa: E402
from src.eval.tool_call_parser import extract_tool_calls, is_canonical_format  # noqa: E402
from src.training.dpo_dataset import (  # noqa: E402
    preference_data_stats,
    verify_prompt_consistency,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="检查偏好数据集质量")
    p.add_argument("--data-dir", default="data/preferences", help="数据目录")
    p.add_argument("--file", default=None, help="只检查单个 jsonl 文件")
    p.add_argument("--show", type=int, default=3, help="展示多少条样本")
    p.add_argument("--max-samples", type=int, default=0, help="每条最多检查多少样本（0=全部）")
    p.add_argument(
        "--skip-consistency",
        action="store_true",
        help="跳过 prompt 一致性检查（它需要重新渲染全部任务，较慢）",
    )
    return p.parse_args(argv)


def audit_rows(rows: list[dict], limit: int = 0) -> dict:
    """对偏好数据做质量审计。

    返回各类问题的计数与示例。这是「数据是否可以拿去训练」的准入检查。
    """
    if limit:
        rows = rows[:limit]

    issues: dict[str, list[dict]] = {
        "sides_equivalent": [],
        "chosen_not_canonical": [],
        "chosen_unknown_tool": [],
        "rejected_unknown_tool_ok": [],  # 这类是**预期**的（幻觉负样本）
        "prompt_missing_assistant_head": [],
        "empty_side": [],
    }
    checked = 0

    for r in rows:
        checked += 1
        prompt, chosen, rejected = r["prompt"], r["chosen"], r["rejected"]
        c_ids = {"task_id": r.get("task_id"), "pair_type": r.get("pair_type")}

        if not chosen.strip() or not rejected.strip():
            issues["empty_side"].append(c_ids)
            continue
        if not prompt.rstrip().endswith("<|im_start|>assistant"):
            issues["prompt_missing_assistant_head"].append(c_ids)

        c = extract_tool_calls(chosen)
        j = extract_tool_calls(rejected)

        # 1. chosen 必须是 canonical 格式（澄清类除外）
        if c.calls and not is_canonical_format(chosen):
            issues["chosen_not_canonical"].append(
                {**c_ids, "chosen_head": chosen[:120]}
            )
        # 2. chosen 的工具名必须在白名单
        if c.first and c.first.name not in VALID_TOOL_NAMES:
            issues["chosen_unknown_tool"].append({**c_ids, "tool": c.first.name})
        # 3. rejected 是幻觉时是**预期行为**，单独统计
        if j.first and j.first.name not in VALID_TOOL_NAMES:
            issues["rejected_unknown_tool_ok"].append({**c_ids, "tool": j.first.name})
        # 4. 两侧语义等价 → 噪声样本
        if c.first and j.first and c.first.name == j.first.name:
            same = True
            for k, v in c.first.arguments.items():
                if k in j.first.arguments:
                    ok, _ = equivalent(v, j.first.arguments[k], param_name=k)
                    if not ok:
                        same = False
                        break
            if same and len(c.first.arguments) == len(j.first.arguments):
                issues["sides_equivalent"].append(
                    {**c_ids, "chosen": chosen[:100], "rejected": rejected[:100]}
                )

    # 真正的问题（rejected_unknown_tool 不算问题）
    real_problems = {k: v for k, v in issues.items() if k != "rejected_unknown_tool_ok" and v}
    return {
        "checked": checked,
        "num_issues": {k: len(v) for k, v in issues.items()},
        "problem_counts": {k: len(v) for k, v in real_problems.items()},
        "examples": {k: v[:3] for k, v in real_problems.items()},
        "passed": not real_problems,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    files: list[Path] = []
    if args.file:
        files = [Path(args.file)]
    else:
        data_dir = Path(args.data_dir)
        if not data_dir.is_absolute():
            data_dir = CFG.path() / data_dir
        if not data_dir.exists():
            print(f"错误：数据目录不存在 {data_dir}", file=sys.stderr)
            print("先运行：python scripts/prepare/build_preferences.py", file=sys.stderr)
            return 1
        files = sorted(data_dir.glob("*.jsonl"))

    if not files:
        print(f"错误：{args.data_dir} 下没有 jsonl 文件", file=sys.stderr)
        return 1

    print("=" * 76)
    print("偏好数据检查")
    print("=" * 76)
    print(f"  文件数: {len(files)}")
    print()

    overall_pass = True
    for f in files:
        try:
            rows = read_jsonl(f)
        except Exception as exc:
            print(f"  ✗ {f.name}: 读取失败 {exc}")
            overall_pass = False
            continue

        print(f"── {f.name} （{len(rows)} 条）")
        stats = preference_data_stats(rows)
        for line in stats.summary_lines():
            print("   " + line)

        # 质量审计
        audit = audit_rows(rows, limit=args.max_samples)
        print(f"   审计: 检查 {audit['checked']} 条")
        if audit["passed"]:
            print("     ✓ 无质量问题")
        else:
            overall_pass = False
            for k, n in sorted(audit["problem_counts"].items(), key=lambda kv: -kv[1]):
                print(f"     ✗ {k}: {n}")
                for ex in audit["examples"].get(k, [])[:2]:
                    print(f"        {json.dumps(ex, ensure_ascii=False)[:150]}")
        if audit["num_issues"].get("rejected_unknown_tool_ok"):
            print(
                f"     · rejected 含幻觉工具 {audit['num_issues']['rejected_unknown_tool_ok']} 条"
                "（预期行为：这是工具选择负样本）"
            )

        # 展示样本
        if args.show > 0:
            print(f"   样本预览（前 {args.show} 条）:")
            for r in rows[: args.show]:
                print(f"     [{PAIR_TYPE_LABELS_ZH.get(r.get('pair_type', ''), r.get('pair_type'))}] "
                      f"task={r.get('task_id')} neg={r.get('negative_kind')}")
                print(f"       intent  : {r.get('intent', '')}")
                tail = r["prompt"].split("<|im_start|>assistant")[-2][-140:] if "<|im_start|>assistant" in r["prompt"] else ""
                print(f"       prompt  : ...{tail.strip()[-110:]}")
                print(f"       chosen  : {r['chosen'][:110]}")
                print(f"       rejected: {r['rejected'][:110]}")
        print()

    # --- prompt 一致性 ---
    if not args.skip_consistency:
        print("── prompt 一致性检查（训练分布 vs 评测分布）")
        try:
            tasks = load_tasks()
            # 只对第一个文件做（检查逻辑与文件无关）
            rows = read_jsonl(files[0])
            cons = verify_prompt_consistency(rows, tasks)
            print(f"   检查 {cons['checked']} 条（有 task_id 且属于评测任务集的样本）")
            if cons["consistent"]:
                print("   ✓ 训练 prompt 与评测 prompt 前缀一致")
            else:
                overall_pass = False
                print(f"   ✗ 发现 {cons['num_mismatch']} 条不一致")
                print("     这会导致：训练 loss 下降但评测指标不动。")
                for ex in cons["examples"][:2]:
                    print(f"     任务 {ex['task_id']} ({ex['pair_type']})")
                    print(f"       期望: {ex['expected_head'][:100]}")
                    print(f"       实际: {ex['actual_head'][:100]}")
        except Exception as exc:
            print(f"   ⚠️ 一致性检查跳过: {type(exc).__name__}: {exc}")

        # --- token 长度（需要 tokenizer）---
        print()
        print("── token 长度检查")
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(CFG.model.resolved, trust_remote_code=True)
            from src.training.dpo_dataset import token_length_stats, truncation_report

            rows = read_jsonl(files[0])
            st = token_length_stats(rows, tok)
            print(f"   tokenizer: {st.tokenizer_name}（采样 {st.num_samples} 条）")
            print(f"   prompt  token: p50={st.prompt_p50} p90={st.prompt_p90} p99={st.prompt_p99} max={st.prompt_max}")
            print(f"   prompt+comp : p50={st.total_p50} p90={st.total_p90} p99={st.total_p99} max={st.total_max}")
            rec = st.recommend_max_length()
            print(f"   建议配置: max_length={rec['max_length']} max_prompt_length={rec['max_prompt_length']}")
            tr = truncation_report(rows, tok, rec["max_length"], rec["max_prompt_length"])
            print(f"   截断率: prompt={tr['prompt_truncation_rate']:.1%} completion={tr['completion_truncation_rate']:.1%}")
            print(f"   截断后退化(两侧相同)率: {tr['degenerate_after_truncation_rate']:.1%}")
            if tr["degenerate_after_truncation_rate"] > 0.05:
                print("   ⚠️ 退化率 > 5%：请调大 max_length 或过滤长样本")
        except ImportError:
            print("   （未安装 transformers，跳过。用字符数粗估：见上面的字符长度统计）")
        except Exception as exc:
            print(f"   ⚠️ 长度检查跳过: {type(exc).__name__}: {exc}")

    print()
    print("=" * 76)
    print("结论: " + ("✓ 数据可用" if overall_pass else "✗ 存在问题，建议先修复再训练"))
    print("=" * 76)
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
