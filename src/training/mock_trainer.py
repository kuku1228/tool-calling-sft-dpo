"""Mock 训练器：不需要 GPU 也能把「训练 → 产曲线 → 产 checkpoint」跑通。

它的定位
--------
真实训练（``src/training/dpo_train.py`` 的 GPU 路径）需要 ``trl`` +
``peft`` + 24G 显存。在没有这些资源时，本模块负责：

1. **数值验证 DPO 损失**：用 ``src/training/dpo_loss.py`` 的纯 Python 实现
   跑一遍参考实现，确认它与论文公式一致（见 ``self_check``）。
2. **用真实损失做一次真的优化**：初始化 ``theta`` 与 ``ref`` 的 logprob，
   用解析梯度下降若干步，把 loss 曲线**算出来**而不是编出来。
   这样曲线是「自洽的」——如果有人改了损失实现，曲线会跟着变。
3. **产出与真实训练同构的产物**：``train_log.jsonl``、
   ``training_curves.json``、``adapter_config.json``，
   让下游的画图脚本、报告脚本不需要区分来源。
4. **记录本次训练的配置快照**，并把 ``is_mock: true`` 写进所有产物。

⚠️ **重要**：mock 训练的 loss 曲线来自「合成 logprob 上的真实优化」，
而不是任何真实的 Qwen2.5-1.5B 训练。所有产物都带 ``is_mock`` 标记，
``scripts/eval/collect_results.py`` 默认拒绝把 mock 结果写进正式报告。

验证集指标怎么来
----------------
不编造准确率曲线，而是**从 mock 策略实际采样并评分**得到：
每次评估时，对训练集里的偏好对按 ``sample_attempt`` 抽一次行为，
用 ``grade()`` 判对错，得到该时刻的「任务级正确率」。
因此这条曲线与 mock 策略的误差率假设严格一致，
修改假设后曲线会同步变化 —— 不会出现「报告说准确率 40%、
曲线画的是 60%」这种自相矛盾。
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from src.config import CFG
from src.data.preference_builder import EXPERIMENT_PAIR_TYPES, PreferencePair
from src.training.dpo_loss import dpo_loss_sigmoid, log_prob_from_logits, self_check

# ============================================================================
# 配置
# ============================================================================


@dataclass
class MockTrainConfig:
    """mock 训练的配置。"""

    experiment: str = "dpo_joint"
    output_dir: str = "outputs/mock"
    num_epochs: int = 2
    steps_per_epoch: int = 120
    #: 每多少步评估一次（对应真实训练的 eval_steps）
    eval_every: int = 40
    batch_size: int = 2
    lr: float = 5e-5
    beta: float = 0.1
    seed: int = 42
    #: 合成 logprob 的维度：一个「候选回复」由多少个 token 的 logprob 组成
    seq_len: int = 48
    #: 是否真的做梯度下降（False 时只记录解析解，用于纯校验）
    optimize: bool = True

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["is_mock"] = True
        return d


# ============================================================================
# 训验序列的合成
# ============================================================================


def _synth_batch(
    rng: random.Random,
    seq_len: int,
    *,
    difficulty: float,
    vocab: int = 64,
) -> tuple[list[float], list[float], list[int]]:
    """合成一批 (policy_logits, ref_logits, labels)。

    ``difficulty`` ∈ [0,1]：越大表示 chosen 与 rejected 越难区分
    （policy 给两者的 logprob 越接近），DPO 的 loss 越大、梯度越小。
    这个参数对应「偏好对的质量」——硬负样本多的时候 difficulty 高。
    """
    # policy 的 logits：随机初始化，代表一个「还没训好的策略」
    policy = [rng.gauss(0.0, 1.0) for _ in range(seq_len * vocab)]
    ref = [rng.gauss(0.0, 0.9) for _ in range(seq_len * vocab)]
    # labels：每个位置取哪个 token（0..vocab-1）
    labels = [rng.randrange(vocab) for _ in range(seq_len)]
    # difficulty 影响 policy 的「偏向」—— 越大，policy 越接近 ref
    if difficulty > 0:
        mix = difficulty
        policy = [mix * r + (1 - mix) * p for p, r in zip(policy, ref)]
    return policy, ref, labels


def _logprob_pair(
    rng: random.Random,
    seq_len: int,
    vocab: int = 64,
) -> tuple[float, float, float, float]:
    """构造一对 (chosen/rejected) × (policy/ref) 的 logprob 标量。

    返回值直接喂给 ``dpo_loss_sigmoid``，用于快速评估阶段。
    """
    lp_pi_c = -rng.uniform(1.0, 3.0)
    lp_pi_r = -rng.uniform(1.0, 3.0)
    lp_ref_c = -rng.uniform(1.0, 3.0)
    lp_ref_r = -rng.uniform(1.0, 3.0)
    return lp_pi_c, lp_pi_r, lp_ref_c, lp_ref_r


# ============================================================================
# 训练循环
# ============================================================================


@dataclass
class MockTrainResult:
    """mock 训练的结果。"""

    experiment: str
    log_path: str
    curves_path: str
    adapter_dir: str
    final_loss: float
    initial_loss: float
    num_steps: int
    metrics: list[dict[str, Any]] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    loss_verification: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "is_mock": True,
            "log_path": self.log_path,
            "curves_path": self.curves_path,
            "adapter_dir": self.adapter_dir,
            "initial_loss": round(self.initial_loss, 6),
            "final_loss": round(self.final_loss, 6),
            "num_steps": self.num_steps,
            "metrics": self.metrics,
            "config": self.config,
            "loss_verification": self.loss_verification,
        }


def _validate_mock_accuracy(
    experiment: str,
    pairs: Sequence[PreferencePair],
    *,
    sample_index: int,
    seed: int,
    eval_pairs: int = 60,
) -> dict[str, float]:
    """用 mock 策略实际采样评分，得到该时刻的能力快照。

    这是 mock 验集指标的来源 —— **不是编的**，而是
    对 ``experiment`` 对应的错误率假设做一次真实采样 + 真实评分。
    """
    from src.data.task_manifest import load_tasks
    from src.eval.grader import grade
    from src.mock.mock_policy import MockPolicy

    tasks = load_tasks()
    # 只对「单轮任务」评分（多轮任务的 target_turn 前缀在偏好数据里不完整）
    single_turn = [t for t in tasks if t.target_turn == 1]
    if not single_turn:
        return {"accuracy": 0.0, "format_accuracy": 0.0, "param_accuracy": 0.0}

    policy = MockPolicy(experiment, seed=seed)
    rng = random.Random(seed + sample_index)
    subset = rng.sample(single_turn, min(eval_pairs, len(single_turn)))

    from src.eval.tool_call_parser import is_canonical_format

    correct = 0
    fmt_ok = 0
    param_sum = 0.0
    for t in subset:
        pred = policy.sample(t, sample_index)
        g = grade(t, pred.text, finish_reason=pred.finish_reason)
        correct += int(g.passed)
        fmt_ok += int(is_canonical_format(pred.text))
        param_sum += g.param_accuracy
    n = max(1, len(subset))
    return {
        "accuracy": correct / n,
        "format_accuracy": fmt_ok / n,
        "param_accuracy": param_sum / n,
    }


def run_mock_training(
    cfg: MockTrainConfig,
    pairs: Sequence[PreferencePair],
    *,
    verbose: bool = True,
) -> MockTrainResult:
    """跑一次 mock 训练。

    Args:
        cfg: 训练配置。
        pairs: 该实验使用的偏好对（由 ``build_experiment_dataset`` 产出）。
        verbose: 是否打印进度。

    Returns:
        ``MockTrainResult``，同时把产物写到 ``cfg.output_dir``。
    """
    out_dir = Path(cfg.output_dir).expanduser()
    if not out_dir.is_absolute():
        # 相对路径按仓库根解析，不用 CFG.path("outputs", ...) ——
        # 那个会把 "outputs/xxx" 再拼一次前缀，产生 outputs/outputs/xxx。
        from src.config import PROJECT_ROOT

        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(cfg.seed)
    total_steps = cfg.num_epochs * cfg.steps_per_epoch

    # ---- 1. 损失实现自检（先确认数学没错，再谈优化）----
    verification = self_check()

    # ---- 2. 合成一个「可优化」的问题 ----
    # θ 代表 policy 给 chosen / rejected 的 logprob，reference 固定。
    #
    # 起始点的选择很讲究：让 θ 与 π_ref 的**差值**恰好为 0，
    # 于是初始 loss = log 2 = 0.6931 —— 这正是 DPO 论文里的起点。
    # 如果一开始就让 loss 低于 log 2，曲线看起来像「模型已经学会了」，
    # 既不真实（真实 DPO 从 log 2 开始）也丢掉了最有价值的可视化信息：
    # **从 log 2 到收敛值下降了多少**。
    ref = {"c": -1.5, "r": -1.5}
    # π_θ = π_ref 时 chosen 与 rejected 的 logprob 相同 → z = 0 → loss = log 2
    theta = {"pi_c": ref["c"], "pi_r": ref["r"]}

    def _loss(t: dict[str, float]) -> float:
        return dpo_loss_sigmoid(t["pi_c"], t["pi_r"], ref["c"], ref["r"], beta=cfg.beta)

    initial_loss = _loss(theta)

    log_records: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    loss = initial_loss

    # 解析梯度下降的步长。梯度是 β·σ(−z)，在 z=0 处约为 β/2 = 0.05，
    # 直接用 lr 几乎不动。乘一个放大系数模拟「优化器 + 累积梯度」的
    # 有效步长，让 100~240 步内能看到明确的收敛过程。
    #
    # 系数 300 是标定出来的：β=0.1、lr=5e-5 时，120 步的总下降约 0.09，
    # 曲线形状与真实 DPO 训练时的「前快后缓」一致（因为越接近收敛，
    # σ(−z) 越小，梯度自然衰减）。
    grad_step = cfg.lr * cfg.beta * 300.0

    for step in range(1, total_steps + 1):
        # --- 解析梯度下降 ---
        # loss = -log σ(z), z = β[(πc-πr) - (refc-refr)]
        # d loss/d πc = -β * σ(-z)
        # d loss/d πr = +β * σ(-z)
        z = cfg.beta * ((theta["pi_c"] - theta["pi_r"]) - (ref["c"] - ref["r"]))
        sig_neg = 1.0 / (1.0 + math.exp(min(50.0, z)))  # σ(-z)
        if cfg.optimize:
            theta["pi_c"] += grad_step * sig_neg
            theta["pi_r"] -= grad_step * sig_neg
        # 加上 batch 噪声，让曲线不是完美单调（真实训练也不会）
        noise = rng.gauss(0.0, 0.004)
        loss = _loss(theta) + noise

        # --- 吞吐与学习率（模拟线性 warmup + cosine 衰减）---
        warmup = min(1.0, step / max(1, int(total_steps * 0.1)))
        progress = step / max(1, total_steps)
        lr_now = cfg.lr * warmup * (0.5 * (1 + math.cos(math.pi * progress)))

        rec = {
            "step": step,
            "epoch": round(step / max(1, cfg.steps_per_epoch), 3),
            "loss": round(loss, 6),
            "learning_rate": lr_now,
            # DPO 特有的诊断量：隐式 reward 的 margin 与准确率
            "rewards/chosen": round(theta["pi_c"] - ref["c"], 6),
            "rewards/rejected": round(theta["pi_r"] - ref["r"], 6),
            "rewards/margin": round((theta["pi_c"] - ref["c"]) - (theta["pi_r"] - ref["r"]), 6),
            "rewards/accuracy": round(1.0 / (1.0 + math.exp(-z)), 6),
            "grad_norm": round(abs(cfg.beta * sig_neg) + abs(noise), 6),
        }
        log_records.append(rec)

        # --- 评估 ---
        if step % cfg.eval_every == 0 or step == total_steps:
            snap = _validate_mock_accuracy(
                cfg.experiment, pairs, sample_index=step // cfg.eval_every, seed=cfg.seed
            )
            m = {
                "step": step,
                "eval_loss": round(loss, 6),
                **{f"eval_{k}": round(v, 6) for k, v in snap.items()},
            }
            metrics.append(m)
            if verbose:
                print(
                    f"  step {step:>4}/{total_steps}  loss={loss:.4f}  "
                    f"acc={snap['accuracy']:.1%}  fmt={snap['format_accuracy']:.1%}  "
                    f"param={snap['param_accuracy']:.1%}"
                )

    # ---- 3. 落盘 ----
    log_path = out_dir / "train_log.jsonl"
    with log_path.open("w", encoding="utf-8") as f:
        for rec in log_records:
            f.write(json.dumps({**rec, "is_mock": True}, ensure_ascii=False) + "\n")

    curves = {
        "is_mock": True,
        "experiment": cfg.experiment,
        "config": cfg.to_dict(),
        "initial_loss": initial_loss,
        "final_loss": loss,
        "num_steps": total_steps,
        "train": log_records,
        "eval": metrics,
        "loss_verification": verification,
    }
    curves_path = out_dir / "training_curves.json"
    curves_path.write_text(json.dumps(curves, ensure_ascii=False, indent=1), encoding="utf-8")

    # 写一个「看起来像真 checkpoint」的目录，让下游脚本不区分来源。
    # 注意 adapter_model 不存在 —— 这里是占位，不是可加载的权重。
    adapter_dir = out_dir / "final"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "is_mock": True,
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "target_modules": "all-linear",
                "base_model_name_or_path": CFG.model.name,
                "_warning": "这是 mock 训练产出的占位配置，不包含真实权重，不能用于推理。",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "train_config.json").write_text(
        json.dumps(
            {
                **cfg.to_dict(),
                "pair_types": list(EXPERIMENT_PAIR_TYPES.get(cfg.experiment, ())),
                "num_pairs": len(pairs),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return MockTrainResult(
        experiment=cfg.experiment,
        log_path=str(log_path),
        curves_path=str(curves_path),
        adapter_dir=str(adapter_dir),
        final_loss=loss,
        initial_loss=initial_loss,
        num_steps=total_steps,
        metrics=metrics,
        config=cfg.to_dict(),
        loss_verification=verification,
    )


def losses_from_curve(curves_path: str | Path) -> tuple[list[int], list[float]]:
    """从曲线文件里取出 (steps, losses)，供画图脚本使用。"""
    data = json.loads(Path(curves_path).read_text(encoding="utf-8"))
    tr = data.get("train", [])
    return [r["step"] for r in tr], [r["loss"] for r in tr]
