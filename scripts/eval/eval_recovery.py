"""错误恢复评测：用真实多轮对话测「被用户纠正后能否改对」。

用法
----
::

    # mock 驱动（CPU 冒烟）
    python scripts/eval/eval_recovery.py --mock --repeat 3

    # 真实模型
    python scripts/eval/eval_recovery.py --adapter outputs/dpo_joint/final --repeat 5

与主评测的区别
--------------
``run_baseline.py`` 评的是**目标轮**（给定固定的多轮前缀，评一轮），
主指标是 pass^k。

本脚本走**真实多轮**：模型先答 → 框架判错并注入纠错 → 模型再答。
只有这样才测得出「错误恢复」这个能力，因为第二步依赖第一步的实际输出。

指标定义
--------
====================  ==========================================================
指标                   定义
====================  ==========================================================
``recovery_rate``     被纠正后改对的比例 = recoveries / errors
``first_try_rate``    第一轮就做对的比例（不需要纠正）
``repeat_rate``       纠正后仍然原样重复同一错误的比例 ← 最直接的失败模式
``avg_turns``         平均消耗轮数（越少越好）
``corrections_used``  平均用掉几次纠错预算
====================  ==========================================================

为什么 ``repeat_rate`` 比 ``recovery_rate`` 更重要
--------------------------------------------------
``recovery_rate`` 的分母是「犯错的次数」，不同方案犯错次数不同，
所以这个比率不可直接跨方案比较。``repeat_rate`` 的分母是**样本数**，
含义是「在所有尝试中，有多大比例出现了『说了也不改』」——
这个量跨方案可比，也直接对应本项目要解决的问题。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import CFG, describe  # noqa: E402
from src.data.task_manifest import load_tasks, recovery_tasks  # noqa: E402
from src.models.agent_loop import run_dialogue, write_dialogues  # noqa: E402
from src.utils.hashing import stable_seed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="错误恢复率评测（真实多轮）")
    p.add_argument("--adapter", default=None, help="LoRA 适配器目录")
    p.add_argument("--model", default=None, help="基座模型")
    p.add_argument("--label", default=None, help="实验标签")
    p.add_argument("--repeat", type=int, default=4, help="每个任务重复几次（衡量稳定性）")
    p.add_argument("--max-turns", type=int, default=6, help="最多对话轮数")
    p.add_argument("--max-corrections", type=int, default=2, help="最多纠正几次")
    p.add_argument("--task-limit", type=int, default=0, help="只跑前 N 个纠错任务")
    p.add_argument("--mock", action="store_true", help="用 mock 策略（不需要 GPU）")
    p.add_argument("--mock-policy", default="baseline", help="mock 用哪个错误率假设")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def make_mock_generate_fn(policy: Any, task_id: str, attempt: int):
    """构造 mock 的多轮生成函数。

    逻辑：第一次调用返回「首轮错误」，被纠正后按恢复概率返回正确回复。
    这样多轮循环的每一步都是真实的 —— 解析、判错、注入纠错、再判。
    """
    import random

    from src.data.task_manifest import load_tasks as _load
    from src.eval.tool_call_parser import canonical_call

    tasks = {t.task_id: t for t in _load()}
    task = tasks[task_id]
    state = {"call_count": 0}
    rng = random.Random(stable_seed(policy.seed, task_id, attempt))
    rate = policy._recovery_rate(task)

    def _fn(messages: list[dict[str, Any]]) -> tuple[str, str, int]:
        state["call_count"] += 1
        if state["call_count"] == 1:
            # 首轮：给出任务预置的错误回复
            text = task.turns[1].content if len(task.turns) > 1 else ""
        else:
            # 后续：按恢复概率决定改对还是继续错
            if rng.random() < rate:
                if task.expected.tool is None:
                    text = "请问您能告诉我取消这个订单的原因吗？我需要记录理由才能处理。"
                else:
                    text = canonical_call(task.expected.tool, task.expected.graded_args)
            else:
                text = task.turns[1].content if len(task.turns) > 1 else ""
        return text, "stop", len(text) // 3

    return _fn


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t_start = time.time()
    seed = args.seed if args.seed is not None else CFG.seed
    label = args.label or (f"mock_recovery_{args.mock_policy}" if args.mock else (Path(args.adapter).parent.name if args.adapter else "baseline"))

    tasks = recovery_tasks(load_tasks())
    if args.task_limit:
        tasks = tasks[: args.task_limit]

    out_dir = Path(args.out_dir) if args.out_dir else CFG.path("experiments", label)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(describe())
    print()
    print("=" * 76)
    print(f"错误恢复评测: {label}")
    print("=" * 76)
    print(f"  纠错任务数 : {len(tasks)}")
    print(f"  每任务重复 : {args.repeat}")
    print(f"  最多轮数   : {args.max_turns}（纠错预算 {args.max_corrections} 次）")
    print(f"  模式       : {'MOCK（非真实结果）' if args.mock else '真实模型'}")
    print()

    # --- 构造生成函数 ---
    mock_policy = None
    hf = None
    if args.mock:
        from src.mock.mock_policy import MockPolicy

        mock_policy = MockPolicy(args.mock_policy, seed=seed)
    else:
        from src.models.hf_policy import GenerationConfig, HFPolicy

        hf = HFPolicy(
            model_name=args.model,
            adapter_path=args.adapter,
            generation=GenerationConfig.from_cfg(seed=None),
        )
        print("加载模型 ...")
        hf.load()
        print(f"  显存: {hf.vram_report()}")

    # --- 跑 ---
    all_results = []
    per_task: dict[str, list[dict[str, Any]]] = {}
    for ti, task in enumerate(tasks):
        rows = []
        for attempt in range(args.repeat):
            if args.mock:
                gen_fn = make_mock_generate_fn(mock_policy, task.task_id, attempt)
            else:
                bundle_cache: dict[str, Any] = {}

                def gen_fn(messages: list[dict[str, Any]], _hf=hf) -> tuple[str, str, int]:
                    # 把 messages 直接喂给模型（system prompt 已在其中）
                    from src.models.prompt_builder import PromptBundle

                    bundle = PromptBundle(messages=messages, tools=[], system_prompt="")
                    return _hf.generate(bundle, sample_index=attempt)

            try:
                res = run_dialogue(
                    task,
                    gen_fn,
                    max_turns=args.max_turns,
                    max_corrections=args.max_corrections,
                )
            except Exception as exc:
                print(f"  ⚠️ {task.task_id} 第 {attempt} 次失败: {type(exc).__name__}: {exc}")
                continue
            rows.append(res.to_dict(include_text=False))
            all_results.append(res)
        per_task[task.task_id] = rows
        if not args.quiet and (ti + 1) % 4 == 0:
            succ = sum(r["success"] for r in rows)
            print(f"  [{ti + 1}/{len(tasks)}] {task.task_id} 成功 {succ}/{len(rows)}")

    # --- 指标 ---
    n = len(all_results)
    if n == 0:
        print("没有成功执行任何对话。", file=sys.stderr)
        return 1

    success = sum(r.success for r in all_results)
    total_errors = sum(r.errors for r in all_results)
    total_recoveries = sum(r.recoveries for r in all_results)
    first_try = sum(1 for r in all_results if r.errors == 0 and r.success)
    # 「说了也不改」：模型在被纠正后仍然没改对
    repeat = sum(1 for r in all_results if r.corrections_used > 0 and not r.success)
    avg_turns = sum(r.num_turns for r in all_results) / n
    avg_corr = sum(r.corrections_used for r in all_results) / n

    metrics = {
        "label": label,
        "is_mock": bool(args.mock),
        "num_dialogues": n,
        "num_tasks": len(tasks),
        "repeat_per_task": args.repeat,
        "task_success_rate": round(success / n, 6),
        "first_try_rate": round(first_try / n, 6),
        "recovery_rate": round(total_recoveries / total_errors, 6) if total_errors else 0.0,
        "repeat_rate": round(repeat / n, 6),
        "total_errors": total_errors,
        "total_recoveries": total_recoveries,
        "avg_turns": round(avg_turns, 3),
        "avg_corrections_used": round(avg_corr, 3),
        "stop_reasons": _count(all_results, lambda r: r.stop_reason),
        "by_mistake_key": {},
        "metadata": {
            "model": args.model or CFG.model.resolved,
            "adapter_path": args.adapter,
            "max_turns": args.max_turns,
            "max_corrections": args.max_corrections,
            "mock_policy": args.mock_policy if args.mock else None,
            "seed": seed,
        },
    }

    # 逐错误模式
    by_key: dict[str, list[Any]] = {}
    for r in all_results:
        task = next((t for t in tasks if t.task_id == r.task_id), None)
        key = task.mistake_hint if task else "unknown"
        by_key.setdefault(key, []).append(r)
    metrics["by_mistake_key"] = {
        k: {
            "n": len(v),
            "task_success_rate": round(sum(r.success for r in v) / len(v), 4),
            "recovery_rate": round(
                sum(r.recoveries for r in v) / max(1, sum(r.errors for r in v)), 4
            ),
            "repeat_rate": round(
                sum(1 for r in v if r.corrections_used > 0 and not r.success) / len(v), 4
            ),
        }
        for k, v in sorted(by_key.items())
    }

    # --- 打印 ---
    print()
    print("=" * 76)
    print(f"恢复评测结果（{n} 次对话）")
    print("=" * 76)
    print(f"  任务成功率 task_success_rate   {metrics['task_success_rate']:>7.2%}")
    print(f"  一次就对 first_try_rate        {metrics['first_try_rate']:>7.2%}")
    print(f"  恢复率 recovery_rate           {metrics['recovery_rate']:>7.2%}  （= 改对次数/犯错次数）")
    print(f"  说了也不改 repeat_rate         {metrics['repeat_rate']:>7.2%}  （★ 跨方案可比的关键指标）")
    print(f"  平均轮数 avg_turns             {metrics['avg_turns']:>7.2f}")
    print(f"  平均用掉纠错 avg_corrections   {metrics['avg_corrections_used']:>7.2f}")
    print(f"  终止原因: {metrics['stop_reasons']}")
    print()
    print("  逐错误模式:")
    print(f"    {'模式':<36}{'n':>5}{'成功率':>9}{'恢复率':>9}{'重复率':>9}")
    for k, v in metrics["by_mistake_key"].items():
        print(
            f"    {k:<36}{v['n']:>5}{v['task_success_rate']:>9.1%}"
            f"{v['recovery_rate']:>9.1%}{v['repeat_rate']:>9.1%}"
        )

    # --- 落盘 ---
    suffix = ".mock" if args.mock else ""
    (out_dir / f"recovery_metrics{suffix}.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    write_dialogues(all_results, out_dir / f"dialogues{suffix}.jsonl")

    if hf is not None:
        hf.unload()

    print()
    print(f"用时 {time.time() - t_start:.1f}s")
    print(f"  指标: {out_dir / f'recovery_metrics{suffix}.json'}")
    print(f"  对话: {out_dir / f'dialogues{suffix}.jsonl'}")
    if args.mock:
        print("\n⚠️ MOCK 结果：不能作为项目结论。")
    return 0


def _count(items: list[Any], key_fn) -> dict[str, int]:
    out: dict[str, int] = {}
    for it in items:
        k = str(key_fn(it))
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


if __name__ == "__main__":
    raise SystemExit(main())
