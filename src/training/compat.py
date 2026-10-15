"""TRL 版本兼容层。

为什么需要这一层
----------------
``DPOTrainer`` 的构造方式在 TRL 0.8 → 0.13 之间发生过多次破坏性变更，
而这些变更**不是简单的参数改名**，签名结构本身变了：

========================  ==================================================
版本                      变化
========================  ==================================================
≤ 0.8                    ``DPOTrainer(model, ref_model, beta, train_dataset, args)``
                         位置参数 + ``tokenizer=``
0.9 ~ 0.11               ``DPOConfig`` 引入，``beta`` 等移入 config；
                         位置参数仍可用但开始告警
0.12                     新增 ``processing_class``，``tokenizer=`` 弃用
0.13                     位置参数弃用到期，必须用关键字参数
0.14+                    ``tokenizer=`` 被移除；``max_prompt_length``
                         逐步被 ``max_length`` 取代
========================  ==================================================

本项目**推荐锁 ``trl==0.13.0``**（理由见
``docs/ops/trl_version_notes.md``），但为了让别人在自己环境里
也能直接跑，这里做运行时自适应：

1. 先 ``inspect.signature`` 看 ``DPOTrainer.__init__`` 到底接受什么；
2. 有 ``processing_class`` 就用它，否则退回 ``tokenizer``；
3. 有 ``DPOConfig`` 就把超参塞进 config，否则按旧式位置参数传；
4. 把实际使用的适配结果写进返回的 ``CompatReport``，
   训练脚本会把它写进 ``outputs/<exp>/compat_report.json``。

这样出问题时能立刻看出「是哪个分支生效了」，
而不是面对一个 TypeError 猜版本。

设计取舍：为什么不用 try/except 硬试
------------------------------------
早期写法是「先试新 API，TypeError 就退回旧 API」。问题是
TypeError 可能来自内部（例如 ``DPOConfig`` 少一个字段），
错误会被吞掉并误判成「版本不支持新 API」，最终报出一个
与真实原因无关的错。显式检查签名能给出准确的诊断。
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any

from src.config import CFG, TrainCfg


@dataclass
class CompatReport:
    """记录本次运行实际走了哪条兼容分支。"""

    trl_version: str = "unknown"
    transformers_version: str = "unknown"
    peft_version: str = "unknown"
    torch_version: str = "unknown"
    #: 是否使用 processing_class（新 API）
    uses_processing_class: bool = False
    #: 是否使用 DPOConfig
    uses_dpo_config: bool = False
    #: DPOConfig 实际接受的字段（用于排查「参数被静默忽略」）
    dpo_config_fields: list[str] = field(default_factory=list)
    #: 被本次适配丢弃的字段（重要！静默丢弃会让「我明明设了 β=0.5」溯源困难）
    dropped_fields: list[str] = field(default_factory=list)
    #: 是否有 ref_model 参数
    supports_ref_model: bool = True
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trl_version": self.trl_version,
            "transformers_version": self.transformers_version,
            "peft_version": self.peft_version,
            "torch_version": self.torch_version,
            "uses_processing_class": self.uses_processing_class,
            "uses_dpo_config": self.uses_dpo_config,
            "dpo_config_fields": self.dpo_config_fields,
            "dropped_fields": self.dropped_fields,
            "supports_ref_model": self.supports_ref_model,
            "notes": self.notes,
        }

    def summary(self) -> str:
        lines = [
            "TRL 兼容层报告",
            "-" * 58,
            f"  trl          {self.trl_version}",
            f"  transformers {self.transformers_version}",
            f"  peft         {self.peft_version}",
            f"  torch        {self.torch_version}",
            f"  processing_class : {self.uses_processing_class}",
            f"  DPOConfig        : {self.uses_dpo_config}",
        ]
        if self.dropped_fields:
            lines.append(f"  ⚠️ 被丢弃的字段: {self.dropped_fields}")
            lines.append("     （该版本 DPOConfig 不接受这些字段，训练中它们不生效）")
        for n in self.notes:
            lines.append(f"  · {n}")
        return "\n".join(lines)


def make_compat_report() -> CompatReport:
    """探测当前环境，生成兼容性报告。

    不导入 trl 也能调用（会返回 unknown 版本）。
    """
    rep = CompatReport()

    def _ver(mod_name: str, attr: str = "__version__") -> str:
        try:
            mod = __import__(mod_name)
            return str(getattr(mod, attr, "unknown"))
        except Exception:
            return "not-installed"

    rep.trl_version = _ver("trl")
    rep.transformers_version = _ver("transformers")
    rep.peft_version = _ver("peft")
    rep.torch_version = _ver("torch")

    try:
        from trl import DPOTrainer

        sig = inspect.signature(DPOTrainer.__init__)
        params = set(sig.parameters)
        rep.uses_processing_class = "processing_class" in params
        rep.supports_ref_model = "ref_model" in params
        if "tokenizer" in params and not rep.uses_processing_class:
            rep.notes.append("该版本仍用 tokenizer= 参数（< 0.12）")
        if "processing_class" in params and "tokenizer" in params:
            rep.notes.append("tokenizer= 与 processing_class= 同时存在（0.12.x），本层优先用 processing_class")
        elif "tokenizer" not in params:
            rep.notes.append("tokenizer= 已被移除（>= 0.14），本层用 processing_class")
        # 位置参数检查：0.13 起 beta 不再能被位置传入
        beta_param = sig.parameters.get("beta")
        rep.notes.append(
            "beta 仍可作为位置参数传入"
            if beta_param is not None
            else "beta 只能通过 DPOConfig 设置（>= 0.13）"
        )
    except Exception as exc:  # pragma: no cover
        rep.notes.append(f"无法检查 DPOTrainer 签名: {type(exc).__name__}: {exc}")

    try:
        from trl import DPOConfig

        rep.dpo_config_fields = sorted(DPOConfig.__dataclass_fields__.keys())  # type: ignore[attr-defined]
        rep.uses_dpo_config = True
    except Exception:
        pass

    return rep


# ============================================================================
# 参数映射
# ============================================================================

#: 本项目 TrainCfg 字段 → DPOConfig 字段名。
#: 大部分同名；这里列出**需要改名**的那些。
#: 注意 ``max_prompt_length`` 在 TRL 0.28+ 被移除，
#: 映射表里保留它，由 ``_filter_fields`` 决定是否真的传下去。
_CFG_TO_DPO_FIELD: dict[str, str] = {
    "lr": "learning_rate",
    "batch_size": "per_device_train_batch_size",
    "grad_accum": "gradient_accumulation_steps",
    "epochs": "num_train_epochs",
    "save_steps": "save_steps",
    "eval_steps": "eval_steps",
    "logging_steps": "logging_steps",
    "gradient_checkpointing": "gradient_checkpointing",
    "max_grad_norm": "max_grad_norm",
    "lr_scheduler": "lr_scheduler_type",
    "save_total_limit": "save_total_limit",
    "seed": "seed",
    "report_to": "report_to",
    "loss_type": "loss_type",
    "beta": "beta",
    "label_smoothing": "label_smoothing",
    "max_length": "max_length",
    "max_prompt_length": "max_prompt_length",
}


def build_dpo_config(train_cfg: TrainCfg, output_dir: str, rep: CompatReport, **extra: Any) -> Any:
    """构造 ``DPOConfig``（或等价的 ``TrainingArguments``）。

    Raises:
        ImportError: 环境里没有 trl。
    """
    try:
        from trl import DPOConfig
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "训练需要 trl。请先安装：\n"
            "    pip install -r requirements.txt\n"
            "若只想验证数据与流水线，用 mock 训练器：\n"
            "    python scripts/train/dpo_train.py --experiment dpo_joint --mock"
        ) from exc

    values: dict[str, Any] = {}
    for cfg_field, dpo_field in _CFG_TO_DPO_FIELD.items():
        values[dpo_field] = getattr(train_cfg, cfg_field)

    # --- 精度 ---
    if train_cfg.bf16:
        values["bf16"] = True
        values["fp16"] = False

    # --- 输出与日志 ---
    values.update(
        {
            "output_dir": output_dir,
            "logging_dir": f"{output_dir}/logs",
            "save_strategy": "steps",
            "eval_strategy": "steps",
            "load_best_model_at_end": False,
            "remove_unused_columns": False,
            "dataloader_num_workers": 0,
            "optim": "adamw_torch",
            # 偏好数据是长序列，进度条与日志的噪音较大，关掉更干净
            "disable_tqdm": False,
        }
    )

    # --- 评测显存（⚠️ 必须显式设定，否则会 OOM）---
    #
    # 实际遇到过的问题：训练跑到 step 50（第一个 eval_steps）时抛
    #
    #     torch.OutOfMemoryError: CUDA out of memory.
    #     Tried to allocate 10.22 GiB. GPU 0 has a total capacity of 23.56 GiB...
    #
    # 训练 batch 是 2，能跑；但 ``per_device_eval_batch_size``
    # **从来没设过**，于是用了 HF 的默认值 **8** —— 是训练 batch 的 4 倍。
    #
    # 为什么 DPO 的评测尤其吃显存：``concatenated_forward`` 会把 chosen 与
    # rejected **拼在 batch 维度**，所以有效 batch 还要再翻倍，然后
    # ``logits.log_softmax(-1)`` 要物化一个 ``(batch, seq, vocab)`` 张量：
    #
    #     8 × 2 × 1280 × 151936 × 4B ≈ 12.4 GB（单这一步就超了剩余显存）
    #
    # 设成 1 之后有效 batch 是 2，峰值降到约 1.5 GB。
    # ``eval_accumulation_steps`` 让累积的中间结果及时搬到 CPU，
    # 避免把整轮评测的 logits 都留在显存里。
    values.setdefault("per_device_eval_batch_size", 1)
    values.setdefault("eval_accumulation_steps", 1)
    values.update(extra)

    # --- 过滤掉该版本不接受的字段 ---
    accepted = set(rep.dpo_config_fields) if rep.dpo_config_fields else set(values)
    if rep.dpo_config_fields:
        dropped = sorted(k for k in values if k not in accepted)
        if dropped:
            rep.dropped_fields.extend(dropped)
            values = {k: v for k, v in values.items() if k in accepted}

    # --- max_prompt_length 已移除时，把它折算进 max_length 的提示 ---
    if rep.dpo_config_fields and "max_prompt_length" not in accepted:
        rep.notes.append(
            "该版本 DPOConfig 不接受 max_prompt_length（TRL >= 0.28 已移除）；"
            "max_length 现在同时约束 prompt+completion，请确认它足够大"
        )

    return DPOConfig(**values)


def build_trainer(
    model: Any,
    train_dataset: Any,
    eval_dataset: Any,
    tokenizer: Any,
    train_cfg: TrainCfg,
    output_dir: str,
    rep: CompatReport,
    *,
    peft_config: Any = None,
    ref_model: Any = None,
) -> Any:
    """按当前 TRL 版本构造 ``DPOTrainer``。

    ⚠️ ``ref_model=None`` + PEFT 时的语义陷阱（重要）：

    TRL 在 ``ref_model is None`` 且模型带 adapter 时，
    通过 ``model.disable_adapter()`` 来算 reference logprob。
    此时 reference **就是原始 base 模型**（未经过任何微调）。

    后果：如果你先在 base 上做 LoRA SFT，然后**继续用同一个 adapter**
    做 DPO，那么 reference 是 base，而 policy 是 base+SFT+DPO ——
    KL 锚点被拉得很远，DPO 会过度偏离，训练不稳。

    正确做法（本项目 scripts/train/dpo_train.py 采用）：
    先 ``merge_and_unload()`` 把 SFT adapter 合进 base，
    保存成一个新模型，再在那个模型上新建 LoRA 做 DPO。
    """
    try:
        from trl import DPOTrainer
    except ImportError as exc:  # pragma: no cover
        raise ImportError("训练需要 trl，见 requirements.txt 的 GPU 段") from exc

    dpo_config = build_dpo_config(train_cfg, output_dir, rep)

    kwargs: dict[str, Any] = {
        "model": model,
        "args": dpo_config,
        "train_dataset": train_dataset,
    }
    if eval_dataset is not None:
        kwargs["eval_dataset"] = eval_dataset
    if peft_config is not None:
        kwargs["peft_config"] = peft_config
    if ref_model is not None and rep.supports_ref_model:
        kwargs["ref_model"] = ref_model
    elif ref_model is not None:
        rep.notes.append("该版本 DPOTrainer 不接受 ref_model，已忽略（依赖 disable_adapter）")

    # tokenizer / processing_class 二选一
    if rep.uses_processing_class:
        kwargs["processing_class"] = tokenizer
    else:
        kwargs["tokenizer"] = tokenizer

    # --- 再次按签名过滤，防止个别版本缺少某些可选参数 ---
    sig = inspect.signature(DPOTrainer.__init__)
    accepted = set(sig.parameters)
    filtered = {k: v for k, v in kwargs.items() if k in accepted or k == "self"}
    dropped = sorted(set(kwargs) - set(filtered))
    if dropped:
        rep.dropped_fields.extend(dropped)
        rep.notes.append(f"DPOTrainer 不接受这些参数，已丢弃: {dropped}")

    try:
        return DPOTrainer(**filtered)
    except TypeError as exc:
        # 到这里说明签名探测与实际构造仍然不一致，把完整诊断抛给用户
        raise TypeError(
            "构造 DPOTrainer 失败 —— 这通常意味着 TRL 版本与本项目的适配层不匹配。\n"
            f"  原始错误: {exc}\n"
            f"  实际传入的参数: {sorted(filtered)}\n"
            f"  DPOTrainer 接受: {sorted(accepted)}\n"
            f"  {rep.summary()}\n"
            "请把这段信息连同 `pip show trl` 的输出一起反馈；\n"
            "或参考 docs/ops/trl_version_notes.md 锁定 trl==0.13.0。"
        ) from exc


def recommended_pins() -> dict[str, str]:
    """本项目推荐的版本组合（写进 README 与 docs/ops）。

    >>> recommended_pins()["trl"]
    '0.13.0'
    """
    return {
        "trl": "0.13.0",
        "transformers": ">=4.45.0,<4.50.0",
        "peft": ">=0.7.0",
        "torch": ">=2.1.0",
        "datasets": ">=2.16.0",
    }


def warn_if_versions_drift(rep: CompatReport) -> list[str]:
    """检查版本是否偏离推荐组合，返回警告列表。"""
    warns: list[str] = []
    trl = rep.trl_version
    if trl in ("not-installed", "unknown"):
        warns.append("未检测到 trl，无法做真实训练（mock 路径不受影响）")
        return warns
    try:
        major, minor = (int(x) for x in trl.split(".")[:2])
    except ValueError:
        return warns

    if (major, minor) >= (1, 0):
        warns.append(
            f"TRL {trl} 是 1.x：位置参数与 tokenizer= 都已移除，"
            "且需要 transformers>=4.56.2。本项目未在该版本上验证，建议锁 0.13.0。"
        )
    elif (major, minor) >= (0, 14):
        warns.append(
            f"TRL {trl}：tokenizer= 已移除；max_prompt_length 在 0.28+ 被删。"
            "本项目在 0.13.0 上验证，其他版本可能会出现参数被静默忽略。"
        )
    elif (major, minor) < (0, 9):
        warns.append(f"TRL {trl} 过旧：没有 DPOConfig，超参只能按位置参数传，"
                     "本项目未适配该路径。")
    return warns
