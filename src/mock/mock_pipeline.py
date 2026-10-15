"""Mock 流水线：在没有 GPU 的机器上把整条研究链路真实跑一遍。

它执行的完整链路
----------------
::

    1. 基线评测          run_evaluation(baseline mock policy, 100 任务 × 4 采样)
    2. 错误分类          错误分布 + 子类型 + 失败案例
    3. 偏好数据构造      4 类偏好对 + 质量门禁 + train/dev 划分
    4. 训练（每个实验）  mock 训练器：真的做 DPO 损失优化 + 出曲线
    5. 再评测            同一套评测协议，得到各实验指标
    6. 消融汇总          pass@k / pass^k / Δstability 对比 + 出图
    7. 产物落盘          experiments/ 与 outputs/

为什么要专门做一条 mock 流水线（而不是「写个假数据脚本」）
----------------------------------------
这条链路里**每一个模块都是真实的**：真实的解析器、真实的评分器、
真实的语义等价判定、真实的错误分类、真实的偏好数据门禁、真实的 DPO 损失。
只有「策略」本身是合成的。

因此它能抓出的 bug 类型非常广 —— 本项目开发过程中，它实际抓到了：

- 错误类型键的大小写不一致（导致所有失败都被归成格式错误）
- 偏好对去重键过粗（丢掉 97% 的候选数据）
- 纠错话术与错误类型不匹配（生成自相矛盾的训练数据）
- 变量名遮蔽（``candidates`` 被内层循环覆盖，直接崩）
- 语义等价判定把「同义改写」误判成错误

这些 bug 在真实 GPU 训练里要浪费几小时才会暴露。

明确不做什么
------------
**不产出任何关于 Qwen2.5-1.5B 或 DPO 效果的结论。**
所有产物都带 ``is_mock: true``，文件名带 ``.mock.`` 后缀，
``scripts/eval/collect_results.py`` 默认拒绝读取它们。
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from src.config import CFG
from src.data.preference_builder import (
    EXPERIMENT_PAIR_TYPES,
    PreferenceBuilder,
    build_experiment_dataset,
    summarize_pairs,
    write_jsonl,
)
from src.data.task_manifest import load_tasks, summary as task_summary
from src.eval.error_taxonomy import ErrorDistribution
from src.eval.grader import (
    build_eval_report,
    grade,
    run_evaluation,
    write_predictions,
)
from src.mock.mock_policy import EXPERIMENT_LABELS_ZH, MockPolicy
from src.training.mock_trainer import MockTrainConfig, run_mock_training
from src.utils.plots import make_all_figures

#: mock 流水线覆盖的实验顺序（对齐 docs/ablation/ablation_plan.md 的实验矩阵）
PIPELINE_EXPERIMENTS: tuple[str, ...] = (
    "baseline",
    "sft_only",
    "dpo_format",
    "dpo_recovery",
    "dpo_joint",
)


@dataclass
class MockPipelineConfig:
    """流水线配置。"""

    num_samples_per_task: int = 4
    primary_k: int = 4
    #: 为了快速冒烟，可只跑前 N 个任务（0 表示全部）
    task_limit: int = 0
    #: 偏好数据条数（None 用 CFG.preference 的默认值）
    n_format: int | None = None
    n_recovery: int | None = None
    n_tool_selection: int | None = None
    #: mock 训练步数
    train_steps: int = 120
    train_epochs: int = 1
    eval_every: int = 20
    #: 是否跑训练阶段（False 时只做评测 + 数据构造）
    run_training: bool = True
    #: 是否出图
    make_plots: bool = True
    seed: int = 42
    #: 输出根目录（相对仓库根或绝对路径）
    out_root: str = "outputs/mock_pipeline"
    #: 实验产物目录前缀。mock 产物一律带 mock 标识，避免与真实结果混淆
    experiments_dirname: str = "mock"


@dataclass
class MockPipelineResult:
    """流水线执行结果。"""

    out_dir: str
    num_tasks: int
    experiments: list[str] = field(default_factory=list)
    reports: dict[str, dict[str, Any]] = field(default_factory=dict)
    baseline_distribution: dict[str, Any] = field(default_factory=dict)
    preference_stats: dict[str, Any] = field(default_factory=dict)
    pair_counts: dict[str, int] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    figures: dict[str, str] = field(default_factory=dict)
    consistency: dict[str, Any] = field(default_factory=dict)
    wall_time_s: float = 0.0
    loss_verification: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_mock": True,
            "warning": (
                "本文件由 mock 流水线产出。其中的指标来自合成策略的误差率假设，"
                "不是 Qwen2.5-1.5B 的真实评测结果，不能作为项目结论引用。"
            ),
            "out_dir": self.out_dir,
            "num_tasks": self.num_tasks,
            "experiments": self.experiments,
            "wall_time_s": round(self.wall_time_s, 2),
            "consistency": self.consistency,
            "loss_verification": self.loss_verification,
            "pair_counts": self.pair_counts,
            "preference_stats": self.preference_stats,
            "training": self.training,
            "figures": self.figures,
            "headline_by_experiment": {
                exp: rep.get("headline", {}) for exp, rep in self.reports.items()
            },
        }


# ============================================================================
# 流水线
# ============================================================================


def run_mock_pipeline(
    cfg: MockPipelineConfig | None = None,
    *,
    verbose: bool = True,
) -> MockPipelineResult:
    """执行完整 mock 流水线。"""
    cfg = cfg or MockPipelineConfig()
    t_start = time.time()

    out_root = Path(cfg.out_root).expanduser()
    if not out_root.is_absolute():
        # 相对路径直接相对仓库根解析。
        # 不要用 CFG.path("outputs", ...) —— 传入的 out_root 本身已含
        # "outputs/" 前缀，会被再拼一次产生 outputs/outputs/xxx。
        from src.config import PROJECT_ROOT

        out_root = PROJECT_ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    figures_dir = CFG.path("docs", "figures")
    experiments_root = CFG.path("experiments", cfg.experiments_dirname)

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    # ------------------------------------------------------------------
    # 阶段 0：任务集
    # ------------------------------------------------------------------
    tasks = load_tasks()
    if cfg.task_limit:
        tasks = tasks[: cfg.task_limit]
    log(f"\n{'=' * 74}\n[1/6] 任务集：{len(tasks)} 个任务\n{'=' * 74}")
    log(task_summary(tasks))

    result = MockPipelineResult(out_dir=str(out_root), num_tasks=len(tasks))

    # ------------------------------------------------------------------
    # 阶段 1：注入一致性自检（先确认 mock 与评分器口径一致）
    # ------------------------------------------------------------------
    log(f"\n{'=' * 74}\n[2/6] 注入一致性自检\n{'=' * 74}")
    probe = MockPolicy("baseline", seed=cfg.seed)
    consistency = probe.verify_injection(tasks, num_samples=2)
    result.consistency = consistency
    log(f"  一致性 = {consistency['consistency']:.1%}（检查 {consistency['total_checked']} 条）")
    if consistency["consistency"] < 0.85:
        log("  ⚠️ 一致性低于 85%：mock 生成器与评分器的口径可能已经漂移，请检查混淆矩阵")
        for inj, row in sorted(consistency["confusion"].items()):
            log(f"     注入 {inj:<16} → 判定 {dict(sorted(row.items(), key=lambda kv: -kv[1]))}")

    # ------------------------------------------------------------------
    # 阶段 2：基线评测
    # ------------------------------------------------------------------
    log(f"\n{'=' * 74}\n[3/6] 基线评测（{len(tasks)} 任务 × {cfg.num_samples_per_task} 采样）\n{'=' * 74}")
    baseline_policy = MockPolicy("baseline", seed=cfg.seed)
    base_preds, base_report = run_evaluation(
        baseline_policy.as_policy_fn(tasks),
        tasks,
        label="baseline",
        num_samples_per_task=cfg.num_samples_per_task,
        k_values=tuple(k for k in CFG.eval.pass_k_values if k <= cfg.num_samples_per_task),
        primary_k=min(cfg.primary_k, cfg.num_samples_per_task),
        seed=cfg.seed,
        metadata={"policy": "baseline", "engine": "mock", "is_mock": True},
        progress=verbose,
    )
    for line in base_report.headline_lines():
        log("  " + line)
    for line in base_report.detail_lines():
        log("  " + line)

    # 错误分布（用 grader 的分布统计，不是策略自报的统计）
    dist = ErrorDistribution()
    for pred in base_preds:
        task = next((t for t in tasks if t.task_id == pred.task_id), None)
        if task is None:
            continue
        g = grade(task, pred.text, finish_reason=pred.finish_reason)
        dist.add(
            g.classification,
            task_id=task.task_id,
            meta={"category": task.category, "difficulty": task.difficulty},
        )
    result.baseline_distribution = dist.to_dict()
    # 分布数据也标记，供 Mermaid 图与后续判断使用
    result.baseline_distribution["is_mock"] = True
    # 把 is_mock 顶到报告顶层 —— 出图函数（含 SVG 路径）靠它决定
    # 是否叠加 SIMULATED 横幅。只放在 metadata 里会让部分图漏标。
    base_report_dict = base_report.to_dict()
    base_report_dict["is_mock"] = True
    result.reports["baseline"] = base_report_dict
    log("")
    for line in dist.summary_lines():
        log("  " + line)

    write_predictions(base_preds, out_root / "predictions_baseline.mock.jsonl")
    base_report.write_json(str(out_root / "eval_report_baseline.mock.json"))

    # ------------------------------------------------------------------
    # 阶段 3：偏好数据构造
    # ------------------------------------------------------------------
    log(f"\n{'=' * 74}\n[4/6] 偏好数据构造\n{'=' * 74}")
    builder = PreferenceBuilder(seed=cfg.seed)
    # ⚠️ 不传评测任务集：训练数据只能来自合成语料。
    # 早期这里传了 ``tasks``（100 个评测任务），导致 51.5% 的训练样本与
    # 评测 prompt 逐字相同 —— 等于在测试集上训练。详见
    # ``src/data/recovery_corpus.EVAL_TASK_PREFIX`` 的说明。
    # ``build_all`` 里另有硬门禁，传错会直接抛错。
    pairs = builder.build_all(
        None,
        n_format=cfg.n_format,
        n_recovery=cfg.n_recovery,
        n_tool_selection=cfg.n_tool_selection,
        include_param=True,
    )
    log(summarize_pairs(pairs))
    log("")
    log("  质量门禁:")
    log(f"    通过 {len(pairs)} 条")
    for bucket, n in sorted(builder.stats["rejected_by_gate"].items(), key=lambda kv: -kv[1]):
        log(f"    拒绝[{bucket}] {n} 条")
    for reason, n in sorted(builder.stats["gate_reasons"].items(), key=lambda kv: -kv[1]):
        log(f"      原因 {reason}: {n}")

    result.preference_stats = builder.stats
    result.loss_verification = {}

    for exp in EXPERIMENT_PAIR_TYPES:
        train_pairs = build_experiment_dataset(exp, pairs, split="train")
        dev_pairs = build_experiment_dataset(exp, pairs, split="dev")
        if train_pairs:
            write_jsonl(train_pairs, out_root / f"preferences_{exp}.train.mock.jsonl")
        if dev_pairs:
            write_jsonl(dev_pairs, out_root / f"preferences_{exp}.dev.mock.jsonl")
        result.pair_counts[exp] = len(train_pairs)

    log("")
    log("  各实验可用条数:")
    for exp, n in sorted(result.pair_counts.items(), key=lambda kv: -kv[1]):
        log(f"    {exp:<22} {n}")

    # ------------------------------------------------------------------
    # 阶段 4 & 5：训练 + 再评测
    # ------------------------------------------------------------------
    log(f"\n{'=' * 74}\n[5/6] 训练与再评测（每个实验）\n{'=' * 74}")

    # --- 把各实验的训练数据等量化 ---
    # 不同实验的偏好对数量本来就不一样（sft_only 用三类共 227 条，
    # dpo_joint 用两类共 177 条）。如果直接这么比，sft_only 会同时
    # 多拿数据、又多拿「更好学的类型」，两个变量一起动，
    # 就没法判断差异到底来自「对比信号」还是「数据量」。
    #
    # 所以把各实验的训练集都降到最小规模，做一次等量对照。
    # 这牺牲了部分数据，换来的是**可比性** —— 消融实验里可比性优先。
    train_counts = {
        exp: len(build_experiment_dataset(exp, pairs, split="train"))
        for exp in PIPELINE_EXPERIMENTS
        if exp != "baseline"
    }
    nonzero = [c for c in train_counts.values() if c > 0]
    budget = min(nonzero) if nonzero else 0
    if budget and len(set(nonzero)) > 1:
        log(
            f"  各实验训练数据不等量 {train_counts}；"
            f"统一降采样到 {budget} 条以保证可比性"
        )

    # 每个实验用独立的随机种子做降采样，避免「恰好选到同一批简单样本」
    downsample_rng = random.Random(cfg.seed + 4242)

    for exp in PIPELINE_EXPERIMENTS:
        log(f"\n  ── {exp}  {EXPERIMENT_LABELS_ZH.get(exp, '')} ──")

        # ---- 训练（baseline 不训练）----
        if exp != "baseline" and cfg.run_training:
            exp_pairs = build_experiment_dataset(exp, pairs, split="train")
            if budget and len(exp_pairs) > budget:
                exp_pairs = downsample_rng.sample(exp_pairs, budget)
            if not exp_pairs:
                log("     ⚠️ 该实验没有可用的偏好数据，跳过训练")
            else:
                tcfg = MockTrainConfig(
                    experiment=exp,
                    output_dir=str(out_root / "train" / exp),
                    num_epochs=cfg.train_epochs,
                    steps_per_epoch=cfg.train_steps,
                    eval_every=cfg.eval_every,
                    seed=cfg.seed,
                )
                t0 = time.time()
                tres = run_mock_training(tcfg, exp_pairs, verbose=verbose)
                result.training[exp] = {
                    "initial_loss": round(tres.initial_loss, 6),
                    "final_loss": round(tres.final_loss, 6),
                    "num_steps": tres.num_steps,
                    "num_pairs_used": len(exp_pairs),
                    "wall_time_s": round(time.time() - t0, 2),
                    "curves_path": tres.curves_path,
                    "metrics": tres.metrics,
                }
                result.loss_verification = tres.loss_verification
                # 落盘到 experiments/ 供报告引用
                exp_dir = experiments_root / exp
                exp_dir.mkdir(parents=True, exist_ok=True)
                (exp_dir / "training_summary.mock.json").write_text(
                    json.dumps(result.training[exp], ensure_ascii=False, indent=1), encoding="utf-8"
                )
                log(
                    f"     数据 {len(exp_pairs)} 条  "
                    f"loss {tres.initial_loss:.4f} → {tres.final_loss:.4f}"
                    f"（{tres.num_steps} 步）"
                )

        # ---- 再评测 ----
        policy = MockPolicy(exp, seed=cfg.seed)
        preds, report = run_evaluation(
            policy.as_policy_fn(tasks),
            tasks,
            label=exp,
            num_samples_per_task=cfg.num_samples_per_task,
            k_values=tuple(k for k in CFG.eval.pass_k_values if k <= cfg.num_samples_per_task),
            primary_k=min(cfg.primary_k, cfg.num_samples_per_task),
            seed=cfg.seed,
            metadata={
                "policy": exp,
                "engine": "mock",
                "is_mock": True,
                "pair_types": list(EXPERIMENT_PAIR_TYPES.get(exp, ())),
            },
            progress=False,
        )
        result.reports[exp] = report.to_dict()
        write_predictions(preds, out_root / f"predictions_{exp}.mock.jsonl")

        exp_dir = experiments_root / exp
        exp_dir.mkdir(parents=True, exist_ok=True)
        report.write_json(str(exp_dir / "eval_report.mock.json"))

        h = result.reports[exp]["headline"]
        log(
            f"     accuracy={h['accuracy']:.1%}  pass@{cfg.primary_k}={h['pass_at_k']:.1%}  "
            f"pass^{cfg.primary_k}={h['pass_hat_k']:.1%}  Δstab={h['delta_stability']:+.1%}"
        )

    # ------------------------------------------------------------------
    # 阶段 6：汇总与出图
    # ------------------------------------------------------------------
    log(f"\n{'=' * 74}\n[6/6] 汇总与出图\n{'=' * 74}")

    # 对比表
    k = cfg.primary_k
    log(f"\n  {'实验':<16}{'accuracy':>10}{'pass@' + str(k):>9}{'pass^' + str(k):>9}{'Δstab':>10}")
    log("  " + "-" * 62)
    for exp in PIPELINE_EXPERIMENTS:
        h = result.reports[exp]["headline"]
        log(
            f"  {exp:<16}{h['accuracy']:>10.1%}{h['pass_at_k']:>9.1%}"
            f"{h['pass_hat_k']:>9.1%}{h['delta_stability']:>+10.1%}"
        )

    if cfg.make_plots:
        # 用最后一个训练实验的曲线出训练图
        curves = None
        for exp in reversed(PIPELINE_EXPERIMENTS):
            if exp in result.training and Path(result.training[exp]["curves_path"]).exists():
                curves = json.loads(
                    Path(result.training[exp]["curves_path"]).read_text(encoding="utf-8")
                )
                break
        result.figures = make_all_figures(
            figures_dir,
            baseline_report=result.reports["baseline"],
            baseline_distribution=result.baseline_distribution,
            reports=result.reports,
            curves=curves,
        )
        if result.figures:
            # 明确列出图名与引擎，而不是只报一个数字 ——
            # 「为什么这次 6 张、上次 7 张」必须当场可查。
            #
            # 注意用 .get 而不是 .pop：若在这里把 _engine 删掉，
            # 上层（run_all_mock.py 的摘要）就读不到引擎信息了，
            # 会显示成 unknown。元数据应当只读，不要顺手清掉。
            engine = result.figures.get("_engine", "unknown")
            names = sorted(k for k in result.figures if not k.startswith("_"))
            log(f"\n  出图引擎: {engine}")
            log(f"  生成 {len(names)} 张图：")
            for name in names:
                log(f"    · {name:<28} {result.figures[name]}")
        else:
            log("  ⚠️ 未生成任何图（matplotlib 与 SVG 两条路径都失败）")

    result.experiments = list(PIPELINE_EXPERIMENTS)
    result.wall_time_s = time.time() - t_start

    # 总报告
    summary_path = out_root / "pipeline_summary.mock.json"
    summary_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8"
    )
    log(f"\n总计用时 {result.wall_time_s:.1f}s")
    log(f"流水线摘要: {summary_path}")
    log(
        "\n⚠️ 以上全部为 SIMULATED 结果，仅用于验证流水线；"
        "项目结论必须引用 experiments/<exp>/eval_report.json（真实 GPU 产出）。"
    )
    return result


if __name__ == "__main__":  # pragma: no cover
    run_mock_pipeline()
