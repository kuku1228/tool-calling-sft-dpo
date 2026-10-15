"""把评测任务集固化到 JSON，供跨机器复用与人工检查。

用法
----
::

    python scripts/prepare/dump_tasks.py
    python scripts/prepare/dump_tasks.py --out experiments/tasks_airline_v1.json
    python scripts/prepare/dump_tasks.py --show-hard 3

为什么要把任务集落盘
--------------------
任务集由代码生成（``src/data/task_manifest.py``）。如果不同机器上
代码版本有微小差异，生成的任务就会不同，评测结果也就不可比。
把生成结果固化成 JSON 并在评测时读取它，可以保证：

- 两份实验结果**确定**是在同一套任务上得到的
- 出问题时能直接 diff 两个版本的任务集，而不是 diff 代码
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import CFG  # noqa: E402
from src.data.task_manifest import (  # noqa: E402
    BASE_DATE,
    _self_check,
    error_tasks,
    load_tasks,
    recovery_tasks,
    summary,
    write_manifest,
)
from src.env.tool_catalog import render_tool_manual  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="导出评测任务集")
    p.add_argument("--out", default="experiments/tasks_airline_v1.json", help="输出路径")
    p.add_argument("--seed", type=int, default=None, help="生成种子")
    p.add_argument("--show-hard", type=int, default=0, help="打印几个 hard 任务详情")
    p.add_argument("--no-check", action="store_true", help="跳出自检")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    seed = args.seed if args.seed is not None else CFG.seed

    print("=" * 74)
    print("导出评测任务集")
    print("=" * 74)

    if not args.no_check:
        print("运行任务集自检 ...")
        _self_check()
        print("  ✓ 自检通过（ID 唯一、目标轮合法、期望工具在白名单、相对日期已换算）")

    tasks = load_tasks(seed=seed)
    print()
    print(summary(tasks))
    print()
    print(f"基准日期 BASE_DATE = {BASE_DATE.isoformat()}")
    print(f"纠错轮任务: {len(recovery_tasks(tasks))} 个")
    print(f"易错任务（纠错轮 + 需澄清）: {len(error_tasks(tasks))} 个")

    out = Path(args.out)
    if not out.is_absolute():
        out = CFG.path() / out
    write_manifest(str(out), tasks)
    print(f"\n已写入: {out}  ({out.stat().st_size / 1024:.1f} KB)")

    # 同时导出工具目录，方便离线核对
    tools_path = out.parent / (out.stem + "_tools.json")
    from src.env.tool_catalog import TOOL_REGISTRY

    tools_path.write_text(
        json.dumps(
            {
                "num_tools": len(TOOL_REGISTRY),
                "tools": [
                    {
                        "name": s.name,
                        "kind": s.kind,
                        "description": s.description,
                        "required": list(s.required_params),
                        "optional": list(s.optional_params),
                        "schema": s.to_openai_schema(),
                    }
                    for s in TOOL_REGISTRY.values()
                ],
                "manual_rendered": render_tool_manual(),
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"已写入: {tools_path}")

    if args.show_hard:
        print()
        print("=" * 74)
        print(f"Hard 任务示例（前 {args.show_hard} 个）")
        print("=" * 74)
        hard = [t for t in tasks if t.difficulty == "hard"]
        for t in hard[: args.show_hard]:
            print(f"\n{t.task_id}  [{t.category}]  visibility={t.visibility}")
            print(f"  目标轮 {t.target_turn}（{t.target_turn_kind}）")
            for i, turn in enumerate(t.turns):
                mark = " ← 打分轮" if i == t.target_turn - 1 else ""
                obs = ""
                if turn.observation is not None:
                    obs = f"  obs={json.dumps(turn.observation, ensure_ascii=False)[:90]}"
                print(f"    [{i}] {turn.role}: {turn.content[:90]}{obs}{mark}")
            exp = t.expected
            print(f"  期望: tool={exp.tool} args={json.dumps(exp.args, ensure_ascii=False)}")
            if exp.args_subset:
                print(f"       宽松口径只比对: {json.dumps(exp.args_subset, ensure_ascii=False)}")
            print(f"  考察点: {exp.note}")

    print("\n下一步：python scripts/eval/run_baseline.py --config configs/eval/eval_baseline.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
