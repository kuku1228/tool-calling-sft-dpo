"""构造偏好数据集（Week 2 的核心产物）。

用法
----
::

    # 默认：构造全部四类偏好对，写入 data/preferences/
    python scripts/prepare/build_preferences.py

    # 只构造联合方案需要的两类，并缩小规模（快速验证）
    python scripts/prepare/build_preferences.py --pair-types format,recovery --n-format 200 --n-recovery 200

    # 只做预算检查，不写文件
    python scripts/prepare/build_preferences.py --dry-run

产出
----
::

    data/preferences/
    ├── all.train.jsonl              全部类型的训练集
    ├── all.dev.jsonl                全部类型的验证集
    ├── dpo_format.train.jsonl       按实验切片
    ├── dpo_format.dev.jsonl
    ├── dpo_recovery.{train,dev}.jsonl
    ├── dpo_joint.{train,dev}.jsonl
    ├── dpo_tool.{train,dev}.jsonl
    ├── sft_only.{train,dev}.jsonl
    └── build_report.json            统计 + 质量门禁报告（可审计）

为什么按实验切片而不是训练时再筛
--------------------------------
切片的结果是「确定的数据文件」，训练脚本只读一个文件，
不会再引入筛选逻辑的不确定性。这也让「Exp1 用了哪些数据」
在文件系统层面就可查证 —— 复现实验时不依赖代码版本的细节。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 允许直接 `python scripts/prepare/xxx.py` 运行（把仓库根加进 sys.path）
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import CFG  # noqa: E402
from src.data.preference_builder import (  # noqa: E402
    EXPERIMENT_PAIR_TYPES,
    PAIR_RECOVERY,
    PAIR_FORMAT,
    PAIR_PARAM,
    PAIR_CLARIFY,
    PAIR_TOOL_SELECTION,
    PAIR_TYPE_LABELS_ZH,
    PreferenceBuilder,
    build_experiment_dataset,
    read_jsonl,
    summarize_pairs,
    write_jsonl,
)

#: 命令行里可用的类型短名
TYPE_ALIASES: dict[str, str] = {
    "format": PAIR_FORMAT,
    "recovery": PAIR_RECOVERY,
    "tool": PAIR_TOOL_SELECTION,
    "param": PAIR_PARAM,
    # ⚠️ 新增类型必须同时加进这里**和** --pair-types 的 default，
    # 否则会被静默跳过：脚本按 wanted 逐类构造，不在 wanted 里的类型
    # 既不构造也不报错。开发期踩过 —— clarify_boundary 加进了
    # PAIR_TYPES 与 EXPERIMENT_PAIR_TYPES，却漏了这一处，
    # 结果偏好数据里一条都没有，而所有输出看起来完全正常。
    "clarify": PAIR_CLARIFY,
}

#: ``--pair-types`` 的默认值。**从 TYPE_ALIASES 派生**而不是手写，
#: 这样以后新增类型时不会再漏掉这一处。
DEFAULT_PAIR_TYPES: str = ",".join(TYPE_ALIASES)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="构造 DPO 偏好数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--out-dir",
        default="data/preferences",
        help="输出目录（相对仓库根），默认 data/preferences",
    )
    p.add_argument(
        "--pair-types",
        default=DEFAULT_PAIR_TYPES,
        help=f"要构造的类型，逗号分隔。可选: {','.join(TYPE_ALIASES)}",
    )
    p.add_argument("--n-format", type=int, default=None, help="格式约束对目标条数")
    p.add_argument("--n-recovery", type=int, default=None, help="错误恢复对目标条数")
    p.add_argument("--n-tool", type=int, default=None, help="工具选择对目标条数")
    p.add_argument("--n-param", type=int, default=None, help="参数约束对目标条数")
    p.add_argument(
        "--n-clarify",
        type=int,
        default=None,
        help="该追问边界对条数（默认取 CFG.preference.n_clarify）",
    )
    p.add_argument("--seed", type=int, default=None, help="随机种子（默认取 CFG.seed）")
    p.add_argument(
        "--experiments",
        default=",".join(EXPERIMENT_PAIR_TYPES),
        help="要切出哪些实验的数据集",
    )
    p.add_argument("--dry-run", action="store_true", help="只打印统计，不写文件")
    p.add_argument("--quiet", action="store_true", help="只输出最终统计")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    seed = args.seed if args.seed is not None else CFG.seed

    # --- 解析类型选择 ---
    wanted: list[str] = []
    for raw in args.pair_types.split(","):
        key = raw.strip().lower()
        if not key:
            continue
        if key not in TYPE_ALIASES:
            print(f"错误：未知类型 {raw!r}。可选: {', '.join(TYPE_ALIASES)}", file=sys.stderr)
            return 2
        wanted.append(TYPE_ALIASES[key])

    print("=" * 72)
    print("构造偏好数据集")
    print("=" * 72)
    print(f"  类型      : {', '.join(PAIR_TYPE_LABELS_ZH[t] for t in wanted)}")
    print(f"  目标条数  : format={args.n_format or CFG.preference.n_format} "
          f"recovery={args.n_recovery or CFG.preference.n_recovery} "
          f"tool={args.n_tool or CFG.preference.n_tool_selection} "
          f"param={args.n_param or CFG.preference.n_format} "
          f"clarify={args.n_clarify or CFG.preference.n_clarify}")
    print(f"  随机种子  : {seed}")
    print(f"  硬负样本比: {CFG.preference.hard_negative_ratio}")
    print()

    builder = PreferenceBuilder(seed=seed)

    # 逐类构造（这样可以按 --pair-types 精确控制，并分别报告进度）
    from src.data.recovery_corpus import build_scenarios, build_training_tasks
    from src.data.task_manifest import load_tasks

    # ⚠️⚠️ 训练数据**只能**来自合成语料，绝不能吃评测任务集 ⚠️⚠️
    #
    # 开发期这里写的是 ``pool = list(load_tasks()) + syn_tasks``，也就是把
    # 100 个**评测任务**混进了训练池。后果实测：
    #
    #     sft_only 的 2509 条训练样本里，1292 条（51.5%）的 prompt
    #     与评测 prompt **逐字相同**，且 chosen 就是该任务的正确答案。
    #
    # 这等于在测试集上做梯度下降，SFT 的 95.0% 因此虚高、与泛化无关。
    # 现在 pool 只含合成任务；``build_all`` 里另有硬门禁兜底。
    scenarios = build_scenarios(seed=seed)
    syn_tasks = build_training_tasks(seed=seed)
    pool = list(syn_tasks)

    # ------------------------------------------------------------------
    # ⚠️⚠️ 硬门禁（与 PreferenceBuilder.build_all 里的那道**同一判据**）⚠️⚠️
    # ------------------------------------------------------------------
    #
    # 为什么这里也要有一道：**本脚本并不调用 build_all**，它逐个调
    # ``build_*_pairs``。也就是说，真正生成
    # ``data/preferences/*.jsonl`` 的这条生产路径，原先**完全绕过**了
    # 唯一的训练/评测隔离检查 —— 保护只存在于一条没人走的代码分支里。
    #
    # 历史事故：原实现此处写的是 ``pool = list(load_tasks()) + syn_tasks``，
    # 把 100 个评测任务混进了训练池，导致 sft_only 的 2509 条样本里
    # 1292 条（51.5%）prompt 与评测逐字相同、chosen 就是标准答案。
    #
    # 判据方向：**带有** SYNTHETIC_TASK_PREFIX 的才是合成语料（允许）。
    from src.data.recovery_corpus import SYNTHETIC_TASK_PREFIX

    _leaked = [t for t in pool if not t.task_id.startswith(SYNTHETIC_TASK_PREFIX)]
    if _leaked:
        raise ValueError(
            "训练池里混入了非合成语料的任务，这会让全部实验结论作废：\n"
            f"  越界任务数: {len(_leaked)}"
            f"（样例 {[t.task_id for t in _leaked[:5]]}）\n"
            "  pool 只能来自 build_training_tasks()"
            f"（task_id 前缀 {SYNTHETIC_TASK_PREFIX!r}）。"
        )

    print(f"  候选任务池: 合成训练任务 {len(syn_tasks)}（**不含评测任务**）")
    print(f"  纠错场景  : {len(scenarios)}")
    print(f"  评测任务  : {len(load_tasks())}（仅用于评测，不进训练）")
    print()

    pairs = []
    shortfalls: list[tuple[str, int, int]] = []
    if PAIR_FORMAT in wanted:
        n = args.n_format if args.n_format is not None else CFG.preference.n_format
        sub = builder.build_format_pairs(pool, n)
        print(f"  [格式约束对]   目标 {n:>5}  实得 {len(sub):>5}")
        if len(sub) < n:
            shortfalls.append(("格式约束对", n, len(sub)))
        pairs += sub
    if PAIR_RECOVERY in wanted:
        n = args.n_recovery if args.n_recovery is not None else CFG.preference.n_recovery
        sub = builder.build_recovery_pairs(pool, n, scenarios=scenarios)
        print(f"  [错误恢复对]   目标 {n:>5}  实得 {len(sub):>5}")
        if len(sub) < n:
            shortfalls.append(("错误恢复对", n, len(sub)))
        pairs += sub
    if PAIR_TOOL_SELECTION in wanted:
        n = args.n_tool if args.n_tool is not None else CFG.preference.n_tool_selection
        sub = builder.build_tool_selection_pairs(pool, n)
        print(f"  [工具选择对]   目标 {n:>5}  实得 {len(sub):>5}")
        if len(sub) < n:
            shortfalls.append(("工具选择对", n, len(sub)))
        pairs += sub
    if PAIR_PARAM in wanted:
        n = args.n_param if args.n_param is not None else CFG.preference.n_format
        sub = builder.build_param_pairs(pool, n)
        print(f"  [参数约束对]   目标 {n:>5}  实得 {len(sub):>5}")
        if len(sub) < n:
            shortfalls.append(("参数约束对", n, len(sub)))
        pairs += sub
    if PAIR_CLARIFY in wanted:
        n = args.n_clarify if args.n_clarify is not None else CFG.preference.n_clarify
        # ⚠️ 这一类用**评测任务集**（tasks）而非 pool：
        # 它要精确复现评测时的「信息不足」轮次结构，
        # 而合成纠错任务里没有这种结构。见 PAIR_CLARIFY 的说明。
        # 这一类从合成语料里取「该追问」任务（``expected.tool is None``）。
        #
        # ⚠️ 早期版本用的是**评测任务集**里那几个 clarify 任务，理由是
        # 「要精确复现评测时的轮次结构」。但那本身就是污染：拿评测任务
        # 的 prompt 去训「此处该追问」，再在同一个任务上评测。
        # 现在改用 ``_synthetic_clarify_tasks()`` 产出的合成任务，
        # 与评测集里的 clarify 任务彻底分离。
        clarify_pool = [t for t in syn_tasks if t.expected.tool is None]
        sub = builder.build_clarify_pairs(clarify_pool, n)
        print(f"  [该追问边界对] 目标 {n:>5}  实得 {len(sub):>5}")
        if len(sub) < n:
            shortfalls.append(("该追问边界对", n, len(sub)))
        pairs += sub

    # --- 未达目标时必须说明，不能静默少给 ---
    #
    # 每类偏好对的条数受**去重上限**约束：(prompt, chosen, rejected) 完全相同的
    # 对只保留一条。各类型的可达上限差异很大：
    #
    #   - 格式约束对：prompt 来自候选池（177），chosen 是该任务的规范调用（唯一），
    #     rejected 只有约 7 种变形 → 上限约 900+ 条。目标调得再高也拿不到更多。
    #   - 恢复 / 工具选择 / 参数约束对：rejected 侧通过「换相邻工具」
    #     「扰动参数值」等生成，取值空间大得多 → 基本可到任意目标。
    #
    # 静默少给会让「我明明配了 1200 条格式对」溯源困难，所以这里明确报出来。
    dedup_limited = {name: (req, got) for name, req, got in shortfalls}
    if shortfalls:
        print()
        print("  ⚠️ 以下类型未达目标（受去重上限约束，不是 bug）：")
        for name, req, got in shortfalls:
            print(f"     {name}: 目标 {req}，实际 {got}（去重后上限）")
        print("     说明：同一 (prompt, chosen, rejected) 只保留一条。")
        print("     格式约束对的 rejected 侧只有约 7 种变形，因此上限约 900 条；")
        print("     想提高上限需要增加 rejected 的构造方式（见 preference_builder）。")

    # --- train/dev 划分（按 task_id，不按样本随机）---
    import random

    task_ids = sorted({p.task_id for p in pairs})
    rng = random.Random(seed + 999)
    rng.shuffle(task_ids)
    n_dev = max(1, int(len(task_ids) * 0.1))
    dev_ids = set(task_ids[:n_dev])
    for p in pairs:
        p.split = "dev" if p.task_id in dev_ids else "train"

    print()
    print(summarize_pairs(pairs))
    print()

    # --- 质量门禁报告 ---
    print("质量门禁:")
    print(f"  通过 {len(pairs)} 条")
    for bucket, n in sorted(builder.stats["rejected_by_gate"].items(), key=lambda kv: -kv[1]):
        print(f"  拒绝[{bucket}] {n} 条")
    for reason, n in sorted(builder.stats["gate_reasons"].items(), key=lambda kv: -kv[1]):
        print(f"    原因 {reason}: {n}")
    print()

    if not pairs:
        print("错误：没有构造出任何偏好对。请检查错误率配置与质量门禁。", file=sys.stderr)
        return 1

    if args.dry_run:
        print("--dry-run：未写入任何文件。")
        return 0

    # --- 落盘 ---
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = CFG.path("data", "preferences") if args.out_dir == "data/preferences" else CFG.path() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, int] = {}
    train_all = [p for p in pairs if p.split == "train"]
    dev_all = [p for p in pairs if p.split == "dev"]
    written["all.train.jsonl"] = write_jsonl(train_all, out_dir / "all.train.jsonl")
    written["all.dev.jsonl"] = write_jsonl(dev_all, out_dir / "all.dev.jsonl")

    exp_names = [e.strip() for e in args.experiments.split(",") if e.strip()]
    for exp in exp_names:
        if exp not in EXPERIMENT_PAIR_TYPES:
            print(f"  警告：跳过未知实验 {exp!r}")
            continue
        tr = build_experiment_dataset(exp, pairs, split="train")
        dv = build_experiment_dataset(exp, pairs, split="dev")
        if tr:
            written[f"{exp}.train.jsonl"] = write_jsonl(tr, out_dir / f"{exp}.train.jsonl")
        if dv:
            written[f"{exp}.dev.jsonl"] = write_jsonl(dv, out_dir / f"{exp}.dev.jsonl")

    print("已写入:")
    for name, n in written.items():
        print(f"  {name:<34} {n:>6} 条")

    # --- 可审计报告 ---
    report = {
        "is_mock": False,
        "seed": seed,
        "pair_types_requested": wanted,
        "num_tasks_pool": len(pool),
        "num_scenarios": len(scenarios),
        "total_pairs": len(pairs),
        "train_pairs": len(train_all),
        "dev_pairs": len(dev_all),
        "stats": builder.stats,
        # 未达目标的类型：记录「去的重上限」，让数据规模可审计
        "dedup_limited_types": dedup_limited,
        "written": written,
        "experiment_slices": {
            exp: {
                "pair_types": list(EXPERIMENT_PAIR_TYPES.get(exp, ())),
                "train": len(build_experiment_dataset(exp, pairs, split="train")),
                "dev": len(build_experiment_dataset(exp, pairs, split="dev")),
            }
            for exp in exp_names
            if exp in EXPERIMENT_PAIR_TYPES
        },
        "config": {
            "n_format": args.n_format if args.n_format is not None else CFG.preference.n_format,
            "n_recovery": args.n_recovery if args.n_recovery is not None else CFG.preference.n_recovery,
            "n_tool_selection": args.n_tool if args.n_tool is not None else CFG.preference.n_tool_selection,
                "n_clarify": args.n_clarify if args.n_clarify is not None else CFG.preference.n_clarify,
            "hard_negative_ratio": CFG.preference.hard_negative_ratio,
        },
    }
    report_path = out_dir / "build_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n构建报告: {report_path}")
    print(
        "\n下一步：\n"
        f"  python scripts/prepare/inspect_preferences.py --data-dir {out_dir}\n"
        "  python scripts/train/dpo_train.py --experiment dpo_joint --config configs/train/dpo_joint.yaml"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
