"""一键跑完整个实验矩阵（Week 3 的执行入口）。

用法
----
::

    # 完整矩阵：基线 + 4 个训练实验 + 各自评测 + 汇总
    python scripts/eval/run_experiment.py --all

    # 只跑基线诊断（Week 1）
    python scripts/eval/run_experiment.py --baseline-only

    # 只跑联合方案（在已有基线的前提下）
    python scripts/eval/run_experiment.py --experiments dpo_joint

    # 加一组对照：SFT-only，验证「对比信号」是否必要
    python scripts/eval/run_experiment.py --experiments dpo_joint --with-sft

    # 先干跑看命令，不真的执行
    python scripts/eval/run_experiment.py --all --dry-run

为什么要有这个脚本
------------------
手工跑实验最容易出的错不是技术问题，而是**流程不一致**：
Exp1 用了 4 次采样、Exp2 用了 3 次；Exp1 跑在旧代码上、Exp2 跑在新代码上。
一旦评测协议不一致，消融表就没有意义。

本脚本把「每个实验必须做的事」固定成一条流水线：
构造数据 → 数据体检 → 训练 → 用同一套协议评测 → 汇总。
每个实验的元数据都会记录配置哈希与评测协议，便于事后审计。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import CFG, PROJECT_ROOT  # noqa: E402
from src.data.preference_builder import EXPERIMENT_PAIR_TYPES  # noqa: E402

#: 要跑的训练实验（按依赖顺序：单方案 → 联合方案）
TRAIN_EXPERIMENTS: tuple[str, ...] = ("dpo_format", "dpo_recovery", "dpo_joint")
OPTIONAL_EXPERIMENTS: tuple[str, ...] = ("dpo_tool", "dpo_all", "dpo_joint_param")

#: 走 SFT 训练脚本的实验（不是 DPO 路径）——见下面 §4 的说明
#: SFT 允许的实验（都是「只用 chosen」的对照组）
#:
#: ⚠️ **加新 SFT 变体时必须同时改这里**，否则 `--experiment X` 会报
#: 「未知实验」而无法训练。实测踩过：加了 `sft_clarify` 到
#: `EXPERIMENT_PAIR_TYPES` 却忘了这里，dry-run 通过、真正训练时才拦。
SFT_EXPERIMENTS: tuple[str, ...] = ("sft_only", "sft_all", "sft_clarify")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="一键跑实验矩阵",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--all", action="store_true", help="跑基线 + 三个主实验 + 汇总")
    p.add_argument("--baseline-only", action="store_true", help="只跑基线诊断")
    p.add_argument("--experiments", default=None, help="逗号分隔的实验名")
    p.add_argument("--with-sft", action="store_true", help="额外跑 SFT-only 对照组")
    p.add_argument("--skip-baseline", action="store_true", help="跳过基线（已有）")
    p.add_argument("--skip-prepare", action="store_true", help="跳过偏好数据构造（已有）")
    p.add_argument("--skip-train", action="store_true", help="跳过训练，只做评测")
    p.add_argument("--skip-eval", action="store_true", help="只训练不评测")
    p.add_argument("--mock", action="store_true", help="全程走 mock（CPU 冒烟，不需要 GPU）")
    p.add_argument("--num-samples", type=int, default=None, help="评测采样次数")
    p.add_argument("--task-limit", type=int, default=0, help="评测任务数上限（0=全部）")
    p.add_argument("--dry-run", action="store_true", help="只打印将要执行的命令")
    return p.parse_args(argv)


def run_cmd(cmd: list[str], *, dry_run: bool = False, label: str = "") -> int:
    """执行一条命令，实时打印。"""
    printable = " ".join(f'"{c}"' if " " in c else c for c in cmd)
    print(f"\n$ {printable}")
    if dry_run:
        return 0
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    dt = time.time() - t0
    if proc.returncode != 0:
        print(f"\n✗ 失败（exit={proc.returncode}）: {label or printable}")
    else:
        print(f"  ✓ 完成，用时 {dt:.1f}s")
    return proc.returncode


def py(*args: str) -> list[str]:
    return [sys.executable, *args]


def config_hash(paths: list[Path]) -> str:
    """把关键配置文件的哈希记进实验元数据。

    用途：事后核对「这份结果是在哪版配置下跑出来的」。
    只靠时间戳不够 —— 配置文件被改过但时间戳没变的情况很常见
    （例如 git checkout 回滚）。
    """
    h = hashlib.sha256()
    for p in sorted(paths):
        if p.exists():
            h.update(p.name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t_start = time.time()

    # --- 决定跑哪些实验 ---
    if args.baseline_only:
        experiments: tuple[str, ...] = ()
    elif args.experiments:
        experiments = tuple(e.strip() for e in args.experiments.split(",") if e.strip())
    elif args.all:
        experiments = TRAIN_EXPERIMENTS
    else:
        experiments = ("dpo_joint",)
    if args.with_sft and "sft_only" not in experiments:
        experiments = ("sft_only",) + experiments

    for e in experiments:
        if e not in EXPERIMENT_PAIR_TYPES:
            print(f"错误：未知实验 {e!r}，可选: {', '.join(EXPERIMENT_PAIR_TYPES)}", file=sys.stderr)
            return 2

    protocol = {
        "num_samples": args.num_samples or CFG.eval.num_samples_per_task,
        "primary_k": CFG.eval.primary_k,
        "task_limit": args.task_limit,
        "temperature": CFG.model.temperature,
        "model": CFG.model.resolved,
        "mock": args.mock,
    }

    print("=" * 78)
    print("实验矩阵执行计划" + ("   [MOCK 模式]" if args.mock else ""))
    print("=" * 78)
    print(f"  基线        : {'跳过' if args.skip_baseline else '跑'}")
    print(f"  训练实验    : {', '.join(experiments) if experiments else '（无）'}")
    print(f"  偏好数据    : {'跳过构造' if args.skip_prepare else '构造'}")
    print(f"  评测协议    : {protocol['num_samples']} 采样/任务, 主指标 pass^{protocol['primary_k']}"
          + (f", 任务数上限 {protocol['task_limit']}" if protocol['task_limit'] else ", 全部任务"))
    print(f"  配置哈希    : {config_hash([PROJECT_ROOT / 'configs' / 'default.yaml', PROJECT_ROOT / 'src' / 'config.py'])}")
    print()

    steps: list[tuple[str, int]] = []
    mock_flag = ["--mock"] if args.mock else []
    eval_extra: list[str] = []
    if args.num_samples:
        eval_extra += ["--num-samples", str(args.num_samples)]
    if args.task_limit:
        eval_extra += ["--task-limit", str(args.task_limit)]

    # ---- 1. 任务集 ----
    if not args.skip_prepare:
        steps.append(
            (
                "导出任务集",
                run_cmd(
                    py("scripts/prepare/dump_tasks.py", "--out", "experiments/tasks_airline_v1.json"),
                    dry_run=args.dry_run,
                    label="dump_tasks",
                ),
            )
        )
        # ---- 2. 偏好数据 ----
        steps.append(
            (
                "构造偏好数据",
                run_cmd(
                    py(
                        "scripts/prepare/build_preferences.py",
                        "--out-dir",
                        "outputs/mock_pipeline" if args.mock else "data/preferences",
                    )
                    + (["--quiet"] if not args.mock else []),
                    dry_run=args.dry_run,
                    label="build_preferences",
                ),
            )
        )

    # ---- 3. 基线 ----
    #
    # ⚠️ 目录名统一用 ``baseline_airline``（与项目方案、configs/eval/eval_baseline.yaml
    # 的 output.dir 一致）。早期版本这里硬编码 ``--out-dir experiments/baseline``，
    # 覆盖了配置里的 output.dir，导致同一份基线在不同入口下落到
    # experiments/baseline 与 experiments/baseline_airline 两个目录 ——
    # 汇总时会被当成两个实验。
    if not args.skip_baseline:
        label = "baseline_airline"
        cmd = py(
            "scripts/eval/run_baseline.py",
            "--config",
            "configs/eval/eval_baseline.yaml",
            "--label",
            label,
            *mock_flag,
            *eval_extra,
        )
        if not args.skip_eval:
            steps.append(("基线评测", run_cmd(cmd, dry_run=args.dry_run, label="baseline")))

    # ---- 4. 每个实验：训练 → 评测 ----
    for exp in experiments:
        if not args.skip_train:
            # ⚠️ sft_only 必须走 SFT 脚本，不能走 DPO 脚本。
            # 两者的训练目标不同（SFT 只用 chosen 算交叉熵；DPO 用
            # chosen/rejected 的对比损失）。让 sft_only 走 DPOTrainer
            # 得到的不是「SFT 基线」，这个对照组就失去意义 ——
            # 而它恰恰是验证「对比信号是否必要」的关键实验。
            if exp in SFT_EXPERIMENTS:
                cmd = py(
                    "scripts/train/sft_train.py",
                    "--experiment",
                    exp,
                    "--config",
                    f"configs/train/{exp}.yaml",
                    *mock_flag,
                )
            else:
                cmd = py(
                    "scripts/train/dpo_train.py",
                    "--experiment",
                    exp,
                    "--config",
                    f"configs/train/{exp}.yaml",
                    *mock_flag,
                )
            if args.mock:
                cmd += ["--max-steps", "60"]
            steps.append((f"训练 {exp}", run_cmd(cmd, dry_run=args.dry_run, label=f"train:{exp}")))
        if not args.skip_eval:
            cmd = py(
                "scripts/eval/run_baseline.py",
                "--label",
                exp,
                "--adapter",
                str(CFG.path("outputs", exp, "final")),
                "--out-dir",
                str(CFG.path("experiments", exp)),
                *mock_flag,
                *eval_extra,
            )
            steps.append((f"评测 {exp}", run_cmd(cmd, dry_run=args.dry_run, label=f"eval:{exp}")))

    # ---- 5. 汇总 ----
    if not args.skip_eval and (experiments or not args.skip_baseline):
        collect = py("scripts/eval/collect_results.py")
        if args.mock:
            collect += ["--include-mock"]
        steps.append(("汇总消融结果", run_cmd(collect, dry_run=args.dry_run, label="collect")))

    # ---- 结果 ----
    failed = [(name, rc) for name, rc in steps if rc != 0]
    print()
    print("=" * 78)
    print("执行结果")
    print("=" * 78)
    for name, rc in steps:
        print(f"  {'✓' if rc == 0 else '✗'} {name}")
    if failed:
        print(f"\n✗ {len(failed)} 个步骤失败，请检查上面的输出。")
        return 1

    # 写执行记录
    if not args.dry_run:
        record = {
            "protocol": protocol,
            "experiments": list(experiments),
            "steps": [name for name, _ in steps],
            "config_hash": config_hash(
                [PROJECT_ROOT / "configs" / "default.yaml", PROJECT_ROOT / "src" / "config.py"]
            ),
            "is_mock": args.mock,
            "wall_time_s": round(time.time() - t_start, 1),
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        rec_dir = CFG.path("experiments", "summary")
        rec_dir.mkdir(parents=True, exist_ok=True)
        suffix = ".mock" if args.mock else ""
        out = rec_dir / f"run_record{suffix}.json"
        out.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n执行记录: {out}")

    print(f"\n总用时 {(time.time() - t_start) / 60:.1f} 分钟")
    if args.mock:
        print("⚠️ MOCK 模式：所有结果仅用于验证流水线，不能作为项目结论。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
