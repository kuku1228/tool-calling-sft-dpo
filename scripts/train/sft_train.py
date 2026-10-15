"""SFT 训练入口：为消融实验的 ``sft_only`` 对照组提供真正的 SFT 路径。

为什么需要独立脚本而不是复用 ``dpo_train.py``
---------------------------------------------
两者的**训练目标不同**，不是同一件事换个损失函数：

========================  ==========================================
DPO                        SFT
========================  ==========================================
用 (prompt, chosen, rejected) 只对 prompt + chosen 做因果语言建模，
三个字段，对比损失           按标准 next-token 预测训练
目标：让 chosen 的概率       目标：让 chosen 的**每个 token**
**高于** rejected             都更像目标输出
========================  ==========================================

具体差异：SFT 只用 chosen 一侧，**丢弃 rejected**；损失是在 chosen 的
token 上算的交叉熵。如果硬塞给 ``DPOTrainer``，它会去算 reference logprob、
算 margin，语义完全不对 —— 那得出的就不是「SFT 基线」，
这个对照组的结论也就没有意义。

本脚本存在的意义
----------------
``sft_only`` 是本项目**最重要的对照组**，也是主动设计的自证伪机制：
如果只用 chosen 做监督就能取得与 DPO 相当的提升，那「偏好对齐的对比信号」
这个卖点就不成立。要做这个判断，SFT 必须是**真正的 SFT**。

用法
----
::

    # 数据体检 + 损失自检（不训练，CPU 可跑）
    python scripts/train/sft_train.py --experiment sft_only --dry-run

    # CPU 冒烟
    python scripts/train/sft_train.py --experiment sft_only --mock

    # 真实训练（GPU）
    python scripts/train/sft_train.py --experiment sft_only --config configs/train/sft_only.yaml

关键实现细节：只在 completion 上算损失
--------------------------------------
标准做法是在 tokenize 后把 **prompt 部分的 label 置为 -100**，
让交叉熵只作用于模型自己该生成的那段（chosen）。

为什么不这么做会出问题：如果对 prompt 也算损失，模型会被训练去
「预测用户说的话」。在工具调用场景下这会稀释格式与参数上的监督信号，
而且让不同长度的 prompt 对损失的贡献不一致 —— 实验结论就不可比了。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import CFG, describe, load_config_file  # noqa: E402
from src.data.preference_builder import EXPERIMENT_PAIR_TYPES, read_jsonl  # noqa: E402
from src.training.dpo_dataset import (  # noqa: E402
    load_preference_jsonl,
    preference_data_stats,
)
from src.training.dpo_loss import report_self_check  # noqa: E402

#: SFT 允许的实验（都是「只用 chosen」的对照组）
#: SFT 允许的实验（都是「只用 chosen」的对照组）
#:
#: ⚠️ **加新 SFT 变体时必须同时改这里**，否则 `--experiment X` 会报
#: 「未知实验」而无法训练。实测踩过：加了 `sft_clarify` 到
#: `EXPERIMENT_PAIR_TYPES` 却忘了这里，dry-run 通过、真正训练时才拦。
SFT_EXPERIMENTS: tuple[str, ...] = ("sft_only", "sft_all", "sft_clarify")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SFT 训练（消融对照组的真实 SFT 路径）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--experiment", default="sft_only", help=f"实验名，可选: {', '.join(SFT_EXPERIMENTS)}")
    p.add_argument("--config", default="configs/train/sft_only.yaml", help="训练配置 yaml")
    p.add_argument("--data-file", default=None, help="偏好数据 jsonl（覆盖配置声明）")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--init-adapter", default=None, help="初始 LoRA 适配器（续训用）")

    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epochs", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--lora-r", type=int, default=None)
    p.add_argument("--lora-alpha", type=int, default=None)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)

    p.add_argument("--mock", action="store_true", help="mock 训练（无需 GPU）")
    p.add_argument("--dry-run", action="store_true", help="只做数据体检")
    p.add_argument("--max-steps", type=int, default=-1, help="限制步数（调试用）")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--eval-after", action="store_true", help="训练后立即评测")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


# ============================================================================
# 数据
# ============================================================================


def resolve_data_file(args: argparse.Namespace, file_cfg: dict) -> Path:
    """决定用哪个训练数据文件。

    ⚠️ **顺序很重要，这里栽过一次。**

    原来的顺序是「先看 yaml 里声明的 ``train_file``」。而 ``sft_only.yaml``
    里写死了 ``data/preferences/sft_only.train.jsonl`` ——
    于是**任何** SFT 变体（例如 ``--experiment sft_clarify``）
    都会静默地读到**对照组**的数据。

    实测后果：`sft_clarify` 的 dry-run 报「输入偏好对 2635 条」，
    而它自己的数据是 **2999** 条、且含 364 条 ``clarify_boundary``。
    若没发现，那次训练就是**白跑 30 分钟**，而且结果和 `sft_only`
    一模一样 —— 但看起来「实验失败了」。

    现在改为：**先按实验名找专属数据文件**，找到就用；只有找不到时才
    退回 yaml 声明。这样 ``--experiment X`` 会自然用上 ``X.train.jsonl``。
    """
    if args.data_file:
        p = Path(args.data_file)
        return p if p.is_absolute() else CFG.path() / p

    # 1) 优先：与实验名同名的数据文件（这才是「跑这个实验」的直觉含义）
    named = CFG.path("data", "preferences", f"{args.experiment}.train.jsonl")
    if named.exists():
        return named

    # 2) 退回：yaml 里声明的
    declared = file_cfg.get("pairs", {}).get("train_file")
    candidates: list[Path] = []
    if declared:
        candidates.append(CFG.path() / declared)
    candidates.append(CFG.path("data", "preferences", "all.train.jsonl"))
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "找不到训练数据。已尝试：\n  "
        + "\n  ".join(str(c) for c in [named, *candidates])
        + "\n请先运行：python scripts/prepare/build_preferences.py"
    )


def build_sft_records(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """把偏好对转成 SFT 样本：**只用 prompt + chosen，丢弃 rejected**。

    这一步是 SFT 对照组定义的核心。``rejected`` 一旦被保留，
    训练目标就会掺入对比信息，对照组的意义就没了。

    >>> rows = [{"prompt": "P", "chosen": "C", "rejected": "R"}]
    >>> build_sft_records(rows)
    [{'text': 'PC', 'prompt': 'P', 'completion': 'C'}]
    """
    out: list[dict[str, str]] = []
    for r in rows:
        prompt = str(r.get("prompt", ""))
        completion = str(r.get("chosen", ""))
        if not prompt or not completion:
            continue
        out.append({"text": prompt + completion, "prompt": prompt, "completion": completion})
    return out


# ============================================================================
# 数据集与 collator（模块级，便于独立测试）
# ============================================================================
#
# ⚠️ 这两个类原先定义在 ``run_real()`` **函数内部**，导致无法 import 来做验证。
# 而它们承载着 SFT 最关键的语义 ——「prompt 部分不算损失」。
# 一个不可测试的关键组件等于不可验证：当时看到初始 loss=0.66 偏低，
# 想验证标签是否 mask 正确，却发现类根本 import 不出来。
# 现在提到模块级，并把 tokenizer 作为参数传入（而不是闭包捕获），
# 这样没有 GPU、没有 transformers 的环境也能测。


class SFTPairDataset:
    """把偏好对转成 SFT 样本：**prompt 部分 label 置 -100**，只在 chosen 上算损失。

    Args:
        recs: ``build_sft_records`` 的输出。
        max_len: 序列长度上限（prompt + completion）。
        tokenizer: 任何提供 ``__call__(text, add_special_tokens=False)`` →
            含 ``input_ids`` 的可调用对象。**不要求是 HF tokenizer**，
            这样测试时可以用一个极简 stub。
    """

    def __init__(self, recs: list[dict[str, str]], max_len: int, tokenizer: Any) -> None:
        self.items: list[dict[str, list[int]]] = []
        self.skipped = 0
        pad_id = getattr(tokenizer, "eos_token_id", 0) or 0
        for r in recs:
            p_ids = list(tokenizer(r["prompt"], add_special_tokens=False)["input_ids"])
            c_ids = list(tokenizer(r["completion"], add_special_tokens=False)["input_ids"])
            c_ids = c_ids + [pad_id]
            # prompt 就占满预算 → 这条样本没有监督信号，跳过并计数
            if len(p_ids) >= max_len or not c_ids:
                self.skipped += 1
                continue
            budget = max_len - len(p_ids)
            c_ids = c_ids[:budget]
            input_ids = p_ids + c_ids
            # ★ 关键：prompt 部分不产生损失（-100 是 CrossEntropyLoss 的 ignore_index）
            labels = [-100] * len(p_ids) + list(c_ids)
            self.items.append({"input_ids": input_ids, "labels": labels})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict[str, list[int]]:
        return self.items[i]

    def masked_ratio(self, i: int = 0) -> float:
        """第 ``i`` 条样本里被 mask（不参与损失）的比例。

        健康值应 **> 0.8**：prompt 约 1000 tokens、chosen 只有几十 tokens。
        如果这个值很低，说明 prompt 参与了损失计算 —— 模型在抄 prompt
        而不是学输出，训练会「假成功」（loss 很低但能力没变）。

        >>> class T:
        ...     eos_token_id = 9
        ...     def __call__(self, s, add_special_tokens=False):
        ...         return {"input_ids": list(range(len(s.split())))}
        >>> ds = SFTPairDataset([{"prompt": "a b c d e f g h", "completion": "x y"}], 100, T())
        >>> ds.items[0]["input_ids"]
        [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 9]
        >>> ds.items[0]["labels"][:9]      # prompt 的 8 个位置被 mask 成 -100
        [-100, -100, -100, -100, -100, -100, -100, -100, 0]
        >>> round(ds.masked_ratio(0), 3)   # 8 / 11
        0.727
        """
        labels = self.items[i]["labels"]
        if not labels:
            return 0.0
        return sum(1 for lbl in labels if lbl == -100) / len(labels)


def make_pad_collator(pad_id: int, torch_module: Any) -> Any:
    """构造 pad collator（需要 torch，所以用工厂函数延迟绑定）。

    padding 位置的 label 也必须置 -100，否则模型会被训练去「预测 padding」。
    """

    class _PadCollator:
        def __call__(self, feats: list[dict[str, list[int]]]) -> dict[str, Any]:
            maxlen = max(len(f["input_ids"]) for f in feats)
            ids, labels, attn = [], [], []
            for f in feats:
                pad = maxlen - len(f["input_ids"])
                ids.append(f["input_ids"] + [pad_id] * pad)
                labels.append(f["labels"] + [-100] * pad)
                attn.append([1] * len(f["input_ids"]) + [0] * pad)
            return {
                "input_ids": torch_module.tensor(ids, dtype=torch_module.long),
                "labels": torch_module.tensor(labels, dtype=torch_module.long),
                "attention_mask": torch_module.tensor(attn, dtype=torch_module.long),
            }

    return _PadCollator()


def sft_health_check(
    rows: list[dict[str, Any]], *, max_length: int, max_samples: int = 0, verbose: bool = True
) -> dict[str, Any]:
    """SFT 数据体检：确认只用 chosen、长度够、prompt 尾部正确。"""
    if max_samples:
        rows = rows[:max_samples]
    records = build_sft_records(rows)

    report: dict[str, Any] = {
        "num_input_pairs": len(rows),
        "num_sft_records": len(records),
        "note": "SFT 样本 = prompt + chosen；rejected 被丢弃（这是对照组的定义）",
    }
    if verbose:
        print(f"  输入偏好对 {len(rows)} 条 → SFT 样本 {len(records)} 条（已丢弃 rejected）")
        print(f"  按类型: {json.dumps(preference_data_stats(rows).by_pair_type, ensure_ascii=False)}")

    # 结构检查
    problems = {"prompt_head_missing": 0, "not_ending_with_assistant": 0}
    for r in records:
        if not r["prompt"].rstrip().endswith("<|im_start|>assistant"):
            problems["not_ending_with_assistant"] += 1
        if not r["prompt"].startswith("<|im_start|>"):
            problems["prompt_head_missing"] += 1
    report["structure_problems"] = problems
    if verbose:
        total_bad = sum(problems.values())
        if total_bad == 0:
            print("  ✓ prompt 结构检查通过（以 <|im_start|>assistant 结尾）")
        else:
            print(f"  ✗ prompt 结构有问题: {problems}")

    # 长度
    try:
        from transformers import AutoTokenizer

        from src.eval.metrics import percentile

        tok = AutoTokenizer.from_pretrained(CFG.model.resolved, trust_remote_code=True)
        plens, tlens = [], []
        for r in records[:400]:
            plens.append(len(tok(r["prompt"], add_special_tokens=False)["input_ids"]))
            tlens.append(len(tok(r["text"], add_special_tokens=False)["input_ids"]))
        report["token_length"] = {
            "prompt_p50": int(percentile(plens, 50)),
            "prompt_p99": int(percentile(plens, 99)),
            "total_p50": int(percentile(tlens, 50)),
            "total_p99": int(percentile(tlens, 99)),
            "total_max": max(tlens) if tlens else 0,
            "max_length_configured": max_length,
            "truncation_rate": round(sum(1 for x in tlens if x > max_length) / max(1, len(tlens)), 4),
        }
        if verbose:
            t = report["token_length"]
            print(
                f"  长度: prompt p99={t['prompt_p99']}  prompt+chosen p99={t['total_p99']}  "
                f"截断率={t['truncation_rate']:.1%}（max_length={max_length}）"
            )
            if t["truncation_rate"] > 0.05:
                print("  ⚠️ 截断率 > 5%：建议调大 max_length")
    except ImportError:
        report["token_length"] = {"note": "未安装 transformers，跳过"}
        lens = [len(r["text"]) for r in records]
        if lens and verbose:
            print(f"  字符长度（粗估）: max={max(lens)}  p50={sorted(lens)[len(lens) // 2]}")
    except Exception as exc:
        report["token_length"] = {"error": f"{type(exc).__name__}: {exc}"}

    return report


# ============================================================================
# Mock 路径
# ============================================================================


def run_mock(args: argparse.Namespace, out_dir: Path, rows: list[dict[str, Any]], kwargs: dict[str, Any]) -> int:
    """mock SFT：不需要 GPU，产出同构的训练曲线与适配器占位目录。

    与 mock DPO 的区别：SFT 的损失是标准交叉熵，起点不是 log 2。
    这里用「模型对 chosen 的平均 logprob」作为等价量，
    从较低的值（模型还不会）上升（学会了）—— 这也是真实 SFT 曲线的形状。
    """
    import math
    import random

    from src.eval.tool_call_parser import is_canonical_format
    from src.utils.svg_plots import training_curve_svg

    print("\n" + "=" * 74)
    print("MOCK SFT 训练模式（不需要 GPU，产物带 is_mock 标记）")
    print("=" * 74)

    records = build_sft_records(rows)
    print(f"  SFT 样本 {len(records)} 条")
    print(f"  说明：本对照组**只用 chosen**，rejected 已被丢弃。\n")

    rng = random.Random(int(kwargs.get("seed", 42)))
    # ⚠️ max_steps 在 args 上，不在 kwargs 里 —— 从 kwargs 取会永远拿不到
    # 命令行传的值，静默退回默认的 120 步（开发期踩过）。
    total_steps = args.max_steps if args.max_steps and args.max_steps > 0 else 120
    eval_every = max(1, total_steps // 6)

    lr = float(kwargs.get("lr", 1e-4))
    # 起始平均 logprob：模型基本不会 （约 -3.0 nats/token）
    # 终值取决于 lr，收敛到约 -0.35（学会了格式）
    start_lp = -3.0
    target_lp = -0.35
    k = min(1.0, lr * 12000)  # 把 lr 映射成「学习强度」[0,1]

    log_records: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    for step in range(1, total_steps + 1):
        progress = step / total_steps
        # 指数衰减逼近（真实 SFT 的形状：前快后缓）
        lp = target_lp + (start_lp - target_lp) * math.exp(-4.0 * progress * k * 3)
        loss = -lp + rng.gauss(0.0, 0.01)
        warmup = min(1.0, step / max(1, int(total_steps * 0.1)))
        lr_now = lr * warmup * (0.5 * (1 + math.cos(math.pi * progress)))
        log_records.append(
            {
                "step": step,
                "loss": round(loss, 6),
                "learning_rate": lr_now,
                "mean_logprob": round(lp, 6),
                "perplexity": round(math.exp(min(50.0, -lp)), 4),
            }
        )
        if step % eval_every == 0 or step == total_steps:
            metrics.append(
                {
                    "step": step,
                    "eval_loss": round(loss, 6),
                    "eval_perplexity": round(math.exp(min(50.0, -lp)), 4),
                }
            )
            if not args.quiet:
                print(f"  step {step:>4}/{total_steps}  loss={loss:.4f}  ppl={math.exp(min(50.0,-lp)):.2f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    with log_path.open("w", encoding="utf-8") as f:
        for r in log_records:
            f.write(json.dumps({**r, "is_mock": True}, ensure_ascii=False) + "\n")

    curves = {
        "is_mock": True,
        "experiment": args.experiment,
        "objective": "sft_causal_lm",
        "note": "SFT 只用 chosen；损失是 completion 上的交叉熵",
        "num_sft_records": len(records),
        "initial_loss": log_records[0]["loss"],
        "final_loss": log_records[-1]["loss"],
        "num_steps": total_steps,
        "train": log_records,
        "eval": metrics,
    }
    curves_path = out_dir / "training_curves.json"
    curves_path.write_text(json.dumps(curves, ensure_ascii=False, indent=1), encoding="utf-8")

    # 训练曲线图（顺带验证出图链路）
    try:
        training_curve_svg(
            log_records, out_dir / "training_curves.svg", experiment=args.experiment, is_mock=True
        )
    except Exception:
        pass

    final_dir = out_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    (final_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "is_mock": True,
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": int(kwargs.get("lora_r", 16)),
                "lora_alpha": int(kwargs.get("lora_alpha", 32)),
                "base_model_name_or_path": CFG.model.resolved,
                "_warning": "mock SFT 的占位配置，不含真实权重，不能用于推理。",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "train_config.json").write_text(
        json.dumps(
            {
                "experiment": args.experiment,
                "objective": "sft_causal_lm",
                "is_mock": True,
                "num_pairs": len(rows),
                "num_sft_records": len(records),
                "hyperparameters": kwargs,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 74)
    print("MOCK SFT 完成")
    print("=" * 74)
    print(f"  loss  {log_records[0]['loss']:.4f} → {log_records[-1]['loss']:.4f}（{total_steps} 步）")
    print(f"  曲线  {curves_path}")
    print(f"  适配器目录（占位，不含真实权重）: {final_dir}")
    print("\n⚠️ 这是 MOCK 训练，不是真实 Qwen2.5-1.5B 的 SFT 结果。")
    return 0


# ============================================================================
# 真实 SFT
# ============================================================================


def run_real(
    args: argparse.Namespace,
    train_file: Path,
    dev_file: Path | None,
    out_dir: Path,
    rows: list[dict[str, Any]],
    kwargs: dict[str, Any],
) -> int:
    """真实 SFT。用 transformers.Trainer + 手写 collator。

    为什么不用 ``trl.SFTTrainer``：它的 API 在 0.9~1.x 之间也发生了多次
    破坏性变更（``tokenizer``→``processing_class``、``dataset_text_field``
    的行为调整、packing 默认值变化）。本项目的 SFT 是**对照组**，
    只需要「在 completion 上算交叉熵」这一件事，用原生 Trainer
    反而更透明、更少版本耦合。DPO 那边仍用 TRL（那里复用价值高）。
    """
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from torch.utils.data import Dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        print(
            f"错误：真实训练需要 torch / transformers / peft，当前缺失：{exc}\n"
            f"  或先用 mock 路径验证: python scripts/train/sft_train.py --experiment "
            f"{args.experiment} --mock",
            file=sys.stderr,
        )
        return 1

    from dataclasses import replace as _replace

    tcfg = CFG.train
    overrides = {k: v for k, v in kwargs.items() if k in tcfg.__dataclass_fields__ and v is not None}
    tcfg = _replace(tcfg, **overrides)

    model_name = args.model or CFG.model.resolved
    records = build_sft_records(rows)

    print(f"  输出目录  : {out_dir}")
    print(f"  训练数据  : {train_file}（{len(records)} 条 SFT 样本）")
    print(f"  lr={tcfg.lr}  epochs={tcfg.epochs}  bs={tcfg.batch_size}×{tcfg.grad_accum}  "
          f"LoRA r={tcfg.lora_r}/{tcfg.lora_alpha}  max_length={tcfg.max_length}")
    print()

    print(f"加载基座模型 {model_name} ...")
    dtype = torch.bfloat16 if tcfg.bf16 and torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if tcfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    if args.init_adapter:
        from peft import PeftModel

        print(f"合并初始 adapter {args.init_adapter} ...")
        model = PeftModel.from_pretrained(model, args.init_adapter).merge_and_unload()

    peft_config = LoraConfig(
        r=tcfg.lora_r,
        lora_alpha=tcfg.lora_alpha,
        lora_dropout=tcfg.lora_dropout,
        target_modules=(
            tcfg.lora_target_modules
            if tcfg.lora_target_modules == "all-linear"
            else [m.strip() for m in tcfg.lora_target_modules.split(",")]
        ),
        bias="none",
        task_type="CAUSAL_LM",
    )

    train_ds = SFTPairDataset(records, tcfg.max_length, tokenizer)
    print(f"  数据集: {len(train_ds)} 条可用（{train_ds.skipped} 条因 prompt 过长被跳过）")
    if len(train_ds) == 0:
        print("错误：没有可用样本。请调大 --max-length。", file=sys.stderr)
        return 1

    targs_kwargs: dict[str, Any] = dict(
        output_dir=str(out_dir),
        overwrite_output_dir=True,
        num_train_epochs=tcfg.epochs,
        per_device_train_batch_size=tcfg.batch_size,
        gradient_accumulation_steps=tcfg.grad_accum,
        learning_rate=tcfg.lr,
        warmup_ratio=tcfg.warmup_ratio,
        weight_decay=tcfg.weight_decay,
        max_grad_norm=tcfg.max_grad_norm,
        lr_scheduler_type=tcfg.lr_scheduler,
        logging_steps=tcfg.logging_steps,
        save_steps=tcfg.save_steps,
        save_total_limit=tcfg.save_total_limit,
        bf16=tcfg.bf16 and torch.cuda.is_available(),
        fp16=False,
        seed=tcfg.seed,
        report_to=tcfg.report_to,
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    if args.max_steps and args.max_steps > 0:
        targs_kwargs["max_steps"] = args.max_steps

    try:
        targs = TrainingArguments(**targs_kwargs)
    except TypeError:
        # 不同 transformers 版本对个别字段支持不同，退掉不认识的
        import inspect

        accepted = set(inspect.signature(TrainingArguments.__init__).parameters)
        dropped = sorted(k for k in targs_kwargs if k not in accepted)
        for k in dropped:
            targs_kwargs.pop(k)
        print(f"  （TrainingArguments 丢弃了不支持的字段: {dropped}）")
        targs = TrainingArguments(**targs_kwargs)

    model = get_peft_model(model, peft_config)
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=make_pad_collator(tokenizer.pad_token_id, torch),
    )

    print("\n开始 SFT 训练 ...")
    t0 = time.time()
    result = trainer.train()
    wall = time.time() - t0

    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    metrics = dict(getattr(result, "metrics", {}) or {})
    (out_dir / "train_config.json").write_text(
        json.dumps(
            {
                "experiment": args.experiment,
                "objective": "sft_causal_lm",
                "is_mock": False,
                "num_input_pairs": len(rows),
                "num_sft_records": len(train_ds),
                "skipped": train_ds.skipped,
                "wall_time_s": round(wall, 1),
                "hyperparameters": {k: getattr(tcfg, k) for k in tcfg.__dataclass_fields__},
                "final_metrics": metrics,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )

    print(f"\n已保存 LoRA 适配器: {final_dir}")
    print(f"训练用时 {wall / 60:.1f} 分钟")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v}")

    if args.eval_after:
        from scripts.eval.run_baseline import main as baseline_main

        return baseline_main(
            ["--adapter", str(final_dir), "--label", args.experiment,
             "--out-dir", str(CFG.path("experiments", args.experiment))]
        )
    print(f"\n下一步：\n  python scripts/eval/run_baseline.py --adapter {final_dir} --label {args.experiment}")
    return 0


# ============================================================================
# main
# ============================================================================


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t_start = time.time()

    if args.experiment not in SFT_EXPERIMENTS and args.experiment not in EXPERIMENT_PAIR_TYPES:
        print(f"错误：未知实验 {args.experiment!r}，可选: {', '.join(SFT_EXPERIMENTS)}", file=sys.stderr)
        return 2

    file_cfg: dict[str, Any] = {}
    if args.config:
        try:
            file_cfg = {k: v for k, v in load_config_file(args.config).items() if not k.startswith("_")}
        except FileNotFoundError:
            print(f"提示：配置文件 {args.config} 不存在，改用全局默认值")

    print(describe())
    print()
    print("=" * 74)
    print(f"SFT 训练（消融对照组）: {args.experiment}")
    print("=" * 74)
    print("本路径**只用 chosen**，丢弃 rejected —— 这是 SFT 对照组的定义。")
    print()

    train_file = resolve_data_file(args, file_cfg)
    dev_file = None
    dev_declared = file_cfg.get("pairs", {}).get("dev_file")
    for cand in ([CFG.path() / dev_declared] if dev_declared else []) + [
        CFG.path("data", "preferences", f"{args.experiment}.dev.jsonl")
    ]:
        if cand.exists():
            dev_file = cand
            break

    out_dir = Path(args.output_dir) if args.output_dir else CFG.path("outputs", args.experiment)
    out_dir.mkdir(parents=True, exist_ok=True)

    sft_cfg = file_cfg.get("sft", {})
    tr_cfg = file_cfg.get("train", {})
    merged = {**tr_cfg, **sft_cfg}  # sft 段优先
    kwargs: dict[str, Any] = {
        "lr": args.lr if args.lr is not None else merged.get("lr"),
        "epochs": args.epochs if args.epochs is not None else merged.get("epochs"),
        "batch_size": args.batch_size if args.batch_size is not None else merged.get("batch_size"),
        "grad_accum": args.grad_accum if args.grad_accum is not None else merged.get("grad_accum"),
        "lora_r": args.lora_r if args.lora_r is not None else merged.get("lora_r"),
        "lora_alpha": args.lora_alpha if args.lora_alpha is not None else merged.get("lora_alpha"),
        "max_length": args.max_length if args.max_length is not None else merged.get("max_length"),
        "seed": args.seed if args.seed is not None else merged.get("seed"),
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    print("[1/3] DPO 损失自检（顺带验证 src/training 的数学实现未被破坏）")
    print(report_self_check())

    print("\n[2/3] SFT 数据体检")
    rows = load_preference_jsonl(train_file)
    if args.max_samples:
        rows = rows[: args.max_samples]
    eff_max_length = int(kwargs.get("max_length", CFG.train.max_length))
    health = sft_health_check(rows, max_length=eff_max_length, verbose=not args.quiet)
    (out_dir / "data_health.json").write_text(
        json.dumps(health, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    if args.dry_run:
        print("\n--dry-run：体检完成，未开始训练。")
        print(f"  体检报告: {out_dir / 'data_health.json'}")
        return 0

    print("\n[3/3] 训练")
    if args.mock:
        rc = run_mock(args, out_dir, rows, kwargs)
    else:
        rc = run_real(args, train_file, dev_file, out_dir, rows, kwargs)

    print(f"\n总用时 {time.time() - t_start:.1f}s")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
