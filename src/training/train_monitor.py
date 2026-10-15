"""训练监控回调：把 DPO 的关键健康信号实时打到日志。

为什么需要这个模块
------------------
真实训练走的是 TRL 的 ``Trainer``，它把指标交给 ``report_to``（tensorboard
等）而不是控制台。后果是：**训练过程中的健康信号在终端里完全看不见** ——
只有训练结束才读一次 ``train_result.metrics``。

而 DPO 恰恰是一个「看指标就能提前发现数据问题」的方法。要盯的是三个量：

======================  ============================  ==========================
指标                     健康表现                      异常时的含义
======================  ============================  ==========================
``loss``                 从 ``log 2 ≈ 0.6931`` 下降    明显偏离 log 2 → 偏好对被截断
                                                       成两侧相同，或 chosen/rejected
                                                       标反了
``rewards/margins``     单调上升                       不动 → 两侧语义等价，梯度为零
``rewards/accuracies``  从 ~0.5 上升                    贴着 0.5 → 模型没学到偏好
======================  ============================  ==========================

本项目开发期就踩过：手册里写「盯住 rewards/margin」，但真实训练路径
根本没输出这个量 —— 盯着一个不存在的日志等了半小时。这个模块就是为了
让那些信号**真的出现在终端里**。

用法::

    from src.training.train_monitor import DPOMonitorCallback, summarize_dpo_health

    trainer = DPOTrainer(..., callbacks=[DPOMonitorCallback(out_dir, verbose=True)])
    trainer.train()
    print(summarize_dpo_health(out_dir / "train_log.jsonl"))
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

#: 关注的核心指标（其余指标一并记录，但控制台只打印这些）。
#:
#: ⚠️ **键名必须与 TRL 实际输出一致 —— 这里曾经用错过，代价很大。**
#:
#: TRL 的 ``DPOTrainer`` 实际输出的键是（实测，见
#: ``scripts/train/probe_trainer_loss.py``）::
#:
#:     logits/chosen        logits/rejected
#:     logps/chosen         logps/rejected
#:     rewards/accuracies   rewards/chosen   rewards/rejected   rewards/margins
#:
#: 注意 ``rewards/accuracies`` 与 ``rewards/margins`` 都是**复数**。
#: 早期版本这里写的是 ``rewards/accuracy`` / ``rewards/margin``（单数），
#: 于是这两个键在日志里**永远是 nan** —— 看起来像「指标没记录」，
#: 实际是我查错了名字。而 DPO 恰恰是靠这两个量判断健康度的，
#: 等于训练全程失去了最关键的监控。
#:
#: 第三方库的指标键名不可假设，需打印实际返回的键。
_PRINT_KEYS: tuple[str, ...] = (
    "loss",
    "rewards/accuracies",
    "rewards/margins",
    "rewards/chosen",
    "rewards/rejected",
    "logps/chosen",
    "logps/rejected",
    "grad_norm",
    "learning_rate",
)

#: DPO 的理论起点：π_θ = π_ref 时 loss 恰为 log 2
LOG2 = math.log(2.0)


def _fmt(v: Any) -> str:
    """把指标格式化成定宽字符串。"""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return "nan/inf"
        if abs(f) < 1e-4 and f != 0:
            return f"{f:.2e}"
        return f"{f:.4f}"
    return str(v)[:14]


class DPOMonitorCallback:
    """把每个 logging step 的指标打印到控制台并落盘。

    继承 ``transformers.TrainerCallback`` 是**延迟**的：只有在真正需要
    transformers 时才要求它存在，这样没有装 transformers 的环境
    （mock 路径）也能 import 本模块。

    Args:
        out_dir: 日志目录，会写 ``train_log.jsonl``。
        verbose: 是否打印到控制台。
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        # 延迟导入：让本模块在无 transformers 环境下也能被 import
        # （mock 路径与单元测试需要）。
        try:
            from transformers import TrainerCallback
        except ImportError:  # pragma: no cover
            return _FallbackCallback(*args, **kwargs)

        class _Impl(TrainerCallback):  # type: ignore[misc, valid-type]
            def __init__(self, out_dir: str | Path, verbose: bool = True) -> None:
                self.out_dir = Path(out_dir)
                self.out_dir.mkdir(parents=True, exist_ok=True)
                self.log_path = self.out_dir / "train_log.jsonl"
                self.verbose = verbose
                self.records: list[dict[str, Any]] = []
                self._warned = False

            # --- 训练开始：先打一行基线说明 ---
            def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
                if not self.verbose:
                    return
                total = getattr(args, "max_steps", None)
                print("\n" + "-" * 74)
                print("DPO 训练监控（每 logging step 刷新）")
                print(f"  理论起点: loss = log 2 = {LOG2:.4f}（此时 π_θ = π_ref）")
                print("  健康信号: loss 下降 / rewards/margins 单调上升 / rewards/accuracies 从 0.5 上升")
                print(f"  总步数  : {total if total and total > 0 else '按 epochs 决定'}")
                print("-" * 74)

            # --- 每个 logging step ---
            def on_log(self, args: Any, state: Any, control: Any, logs: dict | None = None, **kwargs: Any) -> None:
                logs = logs or {}
                step = int(getattr(state, "global_step", 0))
                rec = {"step": step, "epoch": getattr(state, "epoch", None)}

                # 收集全部数值指标（含 TRL 的 rewards/*），落盘用全量
                for k, v in logs.items():
                    if isinstance(v, (int, float, bool)):
                        rec[k] = v

                # 首步就偏离 log 2 太多 → 数据几乎一定有问题，尽早告警
                loss = logs.get("loss")
                if (
                    not self._warned
                    and isinstance(loss, (int, float))
                    and step <= 10
                    and isinstance(loss, (int, float))
                ):
                    f = float(loss)
                    if f > LOG2 + 0.35 or f < LOG2 - 0.35:
                        self._warned = True
                        print(
                            f"\n  ⚠️ 早期 loss = {f:.4f}，明显偏离 log 2 = {LOG2:.4f}。\n"
                            f"     常见原因：\n"
                            f"       1) 偏好对里 chosen 与 rejected 被截断成相同内容（截断退化）\n"
                            f"       2) chosen / rejected 标反了\n"
                            f"       3) loss_type 或 β 配置与预期不符\n"
                            f"     排查：python scripts/prepare/inspect_preferences.py\n"
                        )

                self.records.append(rec)
                # 逐行追加落盘（即使训练崩溃，已记录的也不会丢）
                with self.log_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

                if self.verbose:
                    parts = [f"  step {step:>5}"]
                    for k in _PRINT_KEYS:
                        if k in logs and isinstance(logs[k], (int, float, bool)):
                            parts.append(f"{k}={_fmt(logs[k])}")
                    print("  ".join(parts))

            # --- 训练结束 ---
            def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
                if not self.verbose:
                    return
                print("-" * 74)
                print(f"训练监控日志: {self.log_path}（共 {len(self.records)} 条记录）")

        return _Impl(*args, **kwargs)


class _FallbackCallback:
    """没有 transformers 时的可用替身。

    它**必须实现与真实回调相同的接口**（``on_train_begin`` / ``on_log`` /
    ``on_train_end``），否则挂载时会 ``AttributeError`` 崩掉。

    这个替身的实际价值不只是「让 import 不报错」：

    - 在没有 transformers 的环境（mock 路径、单元测试、本地 CPU）里，
      可以**驱动同样的日志与落盘逻辑**，从而验证监控链路本身没坏；
    - 训练脚本在 CPU 上做数据体检时也能留下 ``train_log.jsonl``。

    实现刻意与真实回调保持一致的落盘格式，这样
    ``summarize_dpo_health`` 对两者都能解析。
    """

    #: 让外部能识别这是替身（真实回调没有这个属性）
    is_fallback: bool = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.out_dir = Path(args[0]) if args and isinstance(args[0], (str, Path)) else Path(".")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.out_dir / "train_log.jsonl"
        self.verbose = bool(kwargs.get("verbose", True))
        self.records: list[dict[str, Any]] = []
        self._warned = False

    def on_train_begin(self, args: Any = None, state: Any = None, control: Any = None, **kwargs: Any) -> None:
        if self.verbose:
            print("\n  （无 transformers 环境：使用日志替身回调，仅记录不训练）")

    def on_log(
        self,
        args: Any = None,
        state: Any = None,
        control: Any = None,
        logs: dict | None = None,
        **kwargs: Any,
    ) -> None:
        logs = logs or {}
        step = int(getattr(state, "global_step", 0))
        rec: dict[str, Any] = {"step": step, "epoch": getattr(state, "epoch", None)}
        for k, v in logs.items():
            if isinstance(v, (int, float, bool)):
                rec[k] = v
        self.records.append(rec)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if self.verbose:
            parts = [f"  step {step:>5}"]
            for k in _PRINT_KEYS:
                if k in logs and isinstance(logs[k], (int, float, bool)):
                    parts.append(f"{k}={_fmt(logs[k])}")
            print("  ".join(parts))

    def on_train_end(self, args: Any = None, state: Any = None, control: Any = None, **kwargs: Any) -> None:
        if self.verbose:
            print(f"  （替身回调记录 {len(self.records)} 条 → {self.log_path}）")


# ============================================================================
# 训练后体检：用日志判断这次训练是否健康
# ============================================================================


def summarize_dpo_health(log_path: str | Path) -> str:
    """读 ``train_log.jsonl``，给出训练健康度判读。

    这是「训练跑完了，但它真的学到东西了吗」的自动回答。判据全部来自
    DPO 的数学性质，而不是经验阈值拍脑袋：

    - 初始 loss 应接近 log 2（π_θ = π_ref 时 DPO 损失恰为 log 2）
    - ``rewards/margins`` 是隐式奖励差，应当单调上升（注意是复数）
    - ``rewards/accuracies`` 是「chosen 隐式奖励高于 rejected 的比例」（注意是复数），
      理论上应从 0.5 上升

    >>> # 空文件/不存在时给出明确提示而不是崩
    >>> summarize_dpo_health(".tmp/__nonexistent__.jsonl")
    '（没有找到训练监控日志，无法判断训练健康度）'
    """
    p = Path(log_path)
    if not p.exists():
        return "（没有找到训练监控日志，无法判断训练健康度）"

    rows: list[dict[str, Any]] = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        return "（训练监控日志为空）"

    def _series(key: str) -> list[tuple[int, float]]:
        out: list[tuple[int, float]] = []
        for r in rows:
            v = r.get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out.append((int(r.get("step", 0)), float(v)))
        return out

    losses = _series("loss")
    margins = _series("rewards/margins")
    accs = _series("rewards/accuracies")

    lines = ["DPO 训练健康度判读", "=" * 62, f"记录条数: {len(rows)}"]

    if losses:
        first_step, first = losses[0]
        _, last = losses[-1]
        lines.append(
            f"  loss            {first:.4f} → {last:.4f}   （log 2 = {LOG2:.4f}）"
        )
        # ⚠️ ``loss = log 2`` **只在初始化那一刻**成立（那时 π_θ = π_ref）。
        #
        # Trainer 在 ``logging_steps`` 之后才开始记录，第一条通常是 step 5 ——
        # 此时模型已更新过若干次，用 log 2 判「起点异常」**必然误报**。
        # 实测：两份 DPO 训练的健康度报告都给出无意义告警（差 +3.75 / +3.83），
        # 而我一开始真的据此怀疑过 reference 配置、中断过一次健康的训练。
        #
        # 正确做法：只有第一条记录确实落在 step <= 1 时才用 log 2 判起点；
        # 否则退化为「loss 是否下降」这个与时刻无关的判据。
        if first_step <= 1:
            if abs(first - LOG2) > 0.35:
                lines.append(
                    f"    ⚠️ 初始 loss 偏离 log 2 较多（差 {first - LOG2:+.4f}）："
                    "优先怀疑偏好对被截断成两侧相同，或 chosen/rejected 标反"
                )
            else:
                lines.append("    ✓ 初始 loss 接近 log 2，符合 DPO 的理论起点")
        else:
            lines.append(
                f"    （第一条记录在 step {first_step}，模型已更新过 —— "
                f"log 2 判据仅适用于 step<=1，故不据此判起点）"
            )
        if last >= first:
            lines.append("    ⚠️ loss 没有下降 —— 检查学习率与数据质量")
        else:
            lines.append(f"    ✓ loss 下降 {(first - last) / max(first, 1e-9):.1%}")

    if margins:
        m_first, m_last = margins[0][1], margins[-1][1]
        lines.append(f"  rewards/margins   {m_first:+.4f} → {m_last:+.4f}")
        if m_last > m_first:
            lines.append("    ✓ margin 上升，说明模型在拉大 chosen 与 rejected 的隐式奖励差")
        else:
            lines.append(
                "    ⚠️ margin 未上升 —— 常见原因是偏好对两侧语义等价（梯度为零）。"
                "跑 scripts/prepare/inspect_preferences.py 核对"
            )

    if accs:
        a_first, a_last = accs[0][1], accs[-1][1]
        lines.append(f"  rewards/accuracies {a_first:.4f} → {a_last:.4f}")
        if a_last > 0.55:
            lines.append("    ✓ accuracy 超过 0.5，模型能区分 chosen/rejected")
        else:
            lines.append("    ⚠️ accuracy 停留在 0.5 附近 —— 模型没学到有效偏好")

    lines.append("=" * 62)
    return "\n".join(lines)
