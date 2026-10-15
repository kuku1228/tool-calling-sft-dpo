"""在 agent 多轮循环上评测模型。

与 ``run_baseline.py`` 的分工
----------------------------
============================  ==========================================
脚本                           回答的问题
============================  ==========================================
``run_baseline.py``（单轮）     「**这一步**模型调对了吗」
``run_agent_eval.py``（本脚本）  「**整条轨迹**模型把事办成了吗」
============================  ==========================================

单轮评测把任务里预计算的 observation 直接注入 prompt，所以模型是「读现成答案」；
本脚本不给它 —— 多轮任务里后续轮次需要的信息必须由模型**自己调工具**取得。
这是「工具调用微调」与「agent」的实质差别。

用法::

    # 基线（原始基座）
    python scripts/eval/run_agent_eval.py --label baseline_agent

    # SFT 模型
    python scripts/eval/run_agent_eval.py --adapter outputs/sft_only/final --label sft_agent

    # 只跑多轮任务（单轮任务在 agent 下与单轮评测等价，跑它们没新信息）
    python scripts/eval/run_agent_eval.py --multi-turn-only --label sft_agent_multi

    # 工具鲁棒性：注入瞬时故障，测恢复率
    python scripts/eval/run_agent_eval.py --adapter outputs/sft_only/final \
        --fault-rate 1.0 --label sft_agent_fault
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agent.loop import AgentLoop, FaultSpec, Trajectory  # noqa: E402
from src.agent.metrics import aggregate  # noqa: E402
from src.config import CFG, describe, load_config_file  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="agent 多轮循环评测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--label", default="agent")
    p.add_argument("--adapter", default=None, help="LoRA 适配器目录（不填 = 原始基座）")
    p.add_argument("--model", default=None, help="基座模型路径")
    p.add_argument("--out-dir", default=None, help="输出目录（默认 experiments/<label>）")
    p.add_argument("--config", default=None, help="评测配置 yaml（读 max_new_tokens 等）")
    p.add_argument("--max-turns", type=int, default=5, help="每条轨迹的轮数上限")
    p.add_argument("--task-limit", type=int, default=0, help="只用前 N 个任务（0 = 全部）")
    p.add_argument("--multi-turn-only", action="store_true", help="只评测多轮任务")
    p.add_argument(
        "--task-set",
        default="v1",
        choices=["v1", "v3"],
        help="评测协议版本：v1 = 旧 100 任务，v3 = 新高质量 100 任务",
    )
    p.add_argument(
        "--chain-only",
        action="store_true",
        help="只评测**真需工具链**的任务（后续轮次所需信息只存在于前序工具返回里）。"
        "比 --multi-turn-only 更严格：后者会包含 recovery/cancellation 这类"
        "「turns 数组多轮但一轮即可完成」的任务，从而稀释 avg_turns",
    )
    p.add_argument(
        "--fault-rate",
        type=float,
        default=0.0,
        help="工具瞬时故障注入概率（0 = 不注入）。用于测 agent 鲁棒性",
    )
    p.add_argument(
        "--fault-tools",
        default="",
        help="只对这些工具注入故障，逗号分隔（默认全部）",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="只打印将要评测的任务，不加载模型")
    return p.parse_args(argv)


def select_tasks(args: argparse.Namespace) -> list:
    from src.agent.loop import task_needs_tool_chaining
    from src.data.task_manifest import load_tasks_by_version

    tasks = list(load_tasks_by_version(getattr(args, "task_set", "v1")))
    if args.chain_only:
        # 真需工具链：后续轮次所需信息只存在于前序工具返回里。
        # 用 len(turns) > 1 会混入 recovery/cancellation 这类「数组多轮但
        # 一轮即可完成」的任务，把 avg_turns 稀释掉（实测 1.72 vs 1.82）。
        tasks = [t for t in tasks if task_needs_tool_chaining(t)]
    elif args.multi_turn_only:
        tasks = [t for t in tasks if len(t.turns) > 1]
    if args.task_limit and args.task_limit > 0:
        tasks = tasks[: args.task_limit]
    return tasks


def build_policy_fn(args: argparse.Namespace, tasks: list):
    """构造 ``PolicyFn``：从任务的 ``build_messages`` 固定 system prompt 与 tools。

    ⚠️ 必须固定下来并在整条轨迹里复用 —— 否则 agent 推理与单轮推理的
    system prompt / 工具定义不一致，会引入训练/推理分布漂移
    （本项目最忌讳的那类问题）。
    """
    from src.models.hf_policy import GenerationConfig, HFPolicy
    from src.models.prompt_builder import build_messages

    model_name = args.model or CFG.model.resolved
    hp = HFPolicy(
        model_name=model_name,
        adapter_path=args.adapter,
        generation=GenerationConfig(),
    )
    print(f"加载模型 {model_name}" + (f" + adapter {args.adapter}" if args.adapter else " ..."))
    hp.load()
    print(f"  显存: {hp.vram_report()}")

    # 用第一个任务取 system prompt 与 tools 定义（同一任务集内它们一致）
    ref = build_messages(tasks[0])
    from src.agent.loop import make_hf_policy_fn

    return make_hf_policy_fn(hp, tools=ref.tools, system_prompt=ref.system_prompt), hp


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print(describe())
    print()
    print("=" * 74)
    print(f"agent 多轮评测: {args.label}")
    print("=" * 74)

    tasks = select_tasks(args)
    if not tasks:
        print("错误：没有可评测的任务", file=sys.stderr)
        return 1

    from src.agent.loop import task_needs_tool_chaining

    multi = sum(1 for t in tasks if len(t.turns) > 1)
    chain = sum(1 for t in tasks if task_needs_tool_chaining(t))
    print(f"  任务数        : {len(tasks)}（多轮 {multi}，其中真需工具链 {chain}）")
    print(f"  轮数上限      : {args.max_turns}")
    print(f"  故障注入      : {'关闭' if args.fault_rate <= 0 else f'{args.fault_rate:.0%}'}")
    if args.fault_tools:
        print(f"  注入工具      : {args.fault_tools}")
    print(f"  适配器        : {args.adapter or '无（原始基座）'}")

    out_dir = Path(args.out_dir) if args.out_dir else CFG.path("experiments", args.label)
    if args.dry_run:
        print()
        print("--dry-run：将评测以下任务")
        for t in tasks[:20]:
            kind = "多轮" if len(t.turns) > 1 else "单轮"
            print(f"    {t.task_id} [{kind}] {t.category:<18} expected={t.expected.tool}")
        if len(tasks) > 20:
            print(f"    ... 另有 {len(tasks) - 20} 个")
        return 0

    policy_fn, hp = build_policy_fn(args, tasks)

    fault_tools = tuple(s.strip() for s in args.fault_tools.split(",") if s.strip()) or None
    fault = (
        FaultSpec(
            at_turn=None,
            tools=fault_tools,
            probability=args.fault_rate,
            transient=True,
        )
        if args.fault_rate > 0
        else None
    )

    trajs: list[Trajectory] = []
    t_start = time.time()
    for i, task in enumerate(tasks, 1):
        loop = AgentLoop(policy_fn, max_turns=args.max_turns, fault=fault)
        traj = loop.run(task)
        trajs.append(traj)
        if not args.quiet and (i % 10 == 0 or i == len(tasks)):
            done = sum(1 for t in trajs if t.success)
            elapsed = time.time() - t_start
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(tasks) - i) / rate if rate > 0 else 0
            print(
                f"  [{args.label}] {i}/{len(tasks)}  "
                f"成功 {done}（{done / i:.1%}）  "
                f"{elapsed / 60:.1f} 分已用  预计剩余 {eta / 60:.1f} 分",
                flush=True,
            )

    metrics = aggregate(trajs, label=args.label)
    metrics.wall_time_s = time.time() - t_start

    print()
    for line in metrics.headline_lines():
        print(line)

    # --- 落盘 ---
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "label": args.label,
        "metadata": {
            "adapter_path": args.adapter,
            "model_name": args.model or CFG.model.resolved,
            "max_turns": args.max_turns,
            "fault_rate": args.fault_rate,
            "fault_tools": list(fault_tools) if fault_tools else None,
            "multi_turn_only": args.multi_turn_only,
            "chain_only": args.chain_only,
            "task_limit": args.task_limit,
            "num_tasks": len(tasks),
            "num_multi_turn": multi,
            "is_mock": False,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_start)),
        },
        "metrics": metrics.to_dict(),
    }
    (out_dir / "agent_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    with (out_dir / "trajectories.jsonl").open("w", encoding="utf-8") as f:
        for t in trajs:
            f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")

    print()
    print("=" * 74)
    print(f"用时 {time.time() - t_start:.1f}s")
    print(f"  报告      : {out_dir / 'agent_report.json'}")
    print(f"  轨迹      : {out_dir / 'trajectories.jsonl'}（{len(trajs)} 条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
