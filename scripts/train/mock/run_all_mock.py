"""一键 mock 冒烟：在 CPU 上验证整条流水线。

用法
----
::

    python scripts/train/mock/run_all_mock.py                # 完整流水线
    python scripts/train/mock/run_all_mock.py --quick         # 缩小规模（约 10 秒）
    python scripts/train/mock/run_all_mock.py --no-train      # 只验证数据与评测
    python scripts/train/mock/run_all_mock.py --task-limit 20 # 只用前 20 个任务

这是**上 GPU 之前必须跑通的第一道关**。
它执行：评测 → 错误分类 → 偏好数据构造 → 训练（真实 DPO 损失）→ 再评测 → 出图。

跑通它只能说明流水线没坏，**不能**说明方案有效 ——
所有产物都带 ``is_mock`` 标记，汇总脚本默认拒绝读取。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.config import CFG, describe  # noqa: E402
from src.training.dpo_loss import report_self_check  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="mock 流水线冒烟测试")
    p.add_argument("--quick", action="store_true", help="缩小规模，约 10 秒跑完")
    p.add_argument("--no-train", action="store_true", help="跳过训练阶段")
    p.add_argument("--no-plots", action="store_true", help="不出图")
    p.add_argument("--task-limit", type=int, default=0, help="只跑前 N 个任务")
    p.add_argument("--train-steps", type=int, default=None, help="mock 训练步数")
    p.add_argument("--num-samples", type=int, default=None, help="每任务采样次数")
    p.add_argument("--out-root", default="outputs/mock_pipeline", help="输出根目录")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t_start = time.time()

    print(describe())
    print()
    print("=" * 78)
    print("MOCK 流水线冒烟测试")
    print("=" * 78)
    print("目的：在 CPU 上验证「评测 → 分类 → 偏好构造 → 训练 → 再评测 → 出图」全链路。")
    print("说明：策略是合成的，因此以下所有指标都是 SIMULATED，不能作为项目结论。")
    print()

    # --- 先做损失自检（最快能发现的数学错误）---
    print("[0/1] DPO 损失实现自检")
    chk = report_self_check()
    print(chk)
    if "总体: 通过" not in chk:
        print("\n✗ 损失实现自检失败，先修这个再跑流水线。", file=sys.stderr)
        return 1
    print()

    from src.mock.mock_pipeline import MockPipelineConfig, run_mock_pipeline

    if args.quick:
        cfg = MockPipelineConfig(
            num_samples_per_task=2,
            primary_k=2,
            task_limit=args.task_limit or 30,
            n_format=100,
            n_recovery=100,
            n_tool_selection=60,
            train_steps=30,
            train_epochs=1,
            eval_every=10,
            run_training=not args.no_train,
            make_plots=not args.no_plots,
            out_root=args.out_root,
        )
    else:
        cfg = MockPipelineConfig(
            num_samples_per_task=args.num_samples or 4,
            primary_k=min(4, args.num_samples or 4),
            task_limit=args.task_limit,
            train_steps=args.train_steps or 120,
            train_epochs=1,
            eval_every=(args.train_steps or 120) // 6,
            run_training=not args.no_train,
            make_plots=not args.no_plots,
            out_root=args.out_root,
        )

    result = run_mock_pipeline(cfg, verbose=True)

    print()
    print("=" * 78)
    print("冒烟测试结果")
    print("=" * 78)
    print(f"  任务数        : {result.num_tasks}")
    print(f"  实验数        : {len(result.experiments)}")
    print(f"  偏好对（各切片）: {json_dumps_compact(result.pair_counts)}")
    print(f"  注入一致性    : {result.consistency.get('consistency', 0):.1%}")
    # 出图清单已在流水线里逐张打印过；这里只给数量，
    # 并明确「走的是哪条路径」——两条路径的图名集合不同，
    # 数量差异（6 vs 7）是正常的，不是失败。
    engine = result.figures.get("_engine", "unknown")
    n_figs = len([k for k in result.figures if not k.startswith("_")])
    print(f"  图            : {n_figs} 张（引擎: {engine}）")
    print("                  注：matplotlib 路径出 PNG（含 pass_k_curve），")
    print("                      SVG 路径出 SVG + Mermaid（多两个 .mermaid.md，")
    print("                      无独立 pass@k 曲线）—— 数量不同属正常。")
    print(f"  用时          : {time.time() - t_start:.1f}s")
    print()

    # --- 完整性检查：产出的文件是否齐全 ---
    checks: list[tuple[str, bool, str]] = []
    out_root = Path(result.out_dir)
    for name in ("pipeline_summary.mock.json", "predictions_baseline.mock.jsonl"):
        p = out_root / name
        checks.append((name, p.exists(), str(p)))

    ok = all(c[1] for c in checks)
    for name, exists, path in checks:
        print(f"  {'✓' if exists else '✗'} {name}")

    print()
    if ok and result.consistency.get("consistency", 0) >= 0.85:
        print("✓ 流水线跑通。下一步（需要 GPU）：")
        print("    1. 把模型下到本地并设好环境变量:")
        print("         export AGENTOPT_MODEL_LOCAL=/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct")
        print("         export AGENTOPT_HF_ENDPOINT=https://hf-mirror.com")
        print("    2. 跑真实基线（Week 1）:")
        print("         python scripts/eval/run_baseline.py --config configs/eval/eval_baseline.yaml")
        print("    3. 分析基线错误:")
        print("         python scripts/eval/analyze_errors.py --label baseline")
        print("    4. 构造偏好数据（Week 2）:")
        print("         python scripts/prepare/build_preferences.py")
        print("    5. 数据体检（确认 max_length 与一致性）:")
        print("         python scripts/prepare/inspect_preferences.py")
        print("    6. 训练（Week 3）:")
        print("         python scripts/train/dpo_train.py --experiment dpo_joint --dry-run")
        print("         python scripts/train/dpo_train.py --experiment dpo_joint --config configs/train/dpo_joint.yaml")
        print("    7. 评测与汇总:")
        print("         python scripts/eval/run_experiment.py --all")
        print("\n⚠️ 再次强调：以上 mock 指标是 SIMULATED，项目结论必须来自真实 GPU 评测。")
        return 0

    print("✗ 有问题需要修复（见上面的输出）。", file=sys.stderr)
    if result.consistency.get("consistency", 0) < 0.85:
        print(
            f"  注入一致性偏低（{result.consistency.get('consistency', 0):.1%}）："
            "mock 生成器与评分器的口径可能已漂移。",
            file=sys.stderr,
        )
    return 1


def json_dumps_compact(d: dict) -> str:
    import json

    return json.dumps(dict(sorted(d.items(), key=lambda kv: -kv[1])), ensure_ascii=False)


if __name__ == "__main__":
    raise SystemExit(main())
