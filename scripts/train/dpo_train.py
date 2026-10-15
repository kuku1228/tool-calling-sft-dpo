"""DPO 训练入口（真实 GPU 训练 + mock 冒烟两条路径）。

用法
----
::

    # 真实训练（需要 GPU + trl + peft）
    python scripts/train/dpo_train.py --experiment dpo_joint --config configs/train/dpo_joint.yaml

    # CPU 冒烟：不加载模型，只验证数据/损失/产物（跑通后再上 GPU 省钱）
    python scripts/train/dpo_train.py --experiment dpo_joint --mock

    # 只检查数据与显存预算，不训练
    python scripts/train/dpo_train.py --experiment dpo_joint --dry-run

    # 调试：极小规模真跑（2 步），确认能跑通再上全量
    python scripts/train/dpo_train.py --experiment dpo_joint --max-steps 2

关键设计：为什么必须做数据长度体检
----------------------------------
DPO 的显存峰值约是普通 LM 训练的 2-3 倍（要同时算 policy 与 reference 的
logprob）。如果 ``max_length`` 设得过大，训练一开始就 OOM；
设得过小，偏好对会被截断成「两侧相同」的噪声。

所以本脚本在训练前**强制**做一次 token 长度体检并打印建议值，
让你在花 GPU 时间之前就知道该设多少。

关键设计：SFT → DPO 的 adapter 陷阱
-----------------------------------
``ref_model=None`` + LoRA 时，TRL 用 ``disable_adapter()`` 算 reference，
此时 reference 是**原始 base 模型**。如果你先做了 LoRA SFT，
又**继续用同一个 adapter** 做 DPO，KL 锚点就被拉远了，训练会不稳。

正确做法（本脚本 ``--init-adapter`` 参数实现）：把 SFT adapter
先 merge 进 base，保存成新模型，再在新模型上新建 LoRA 做 DPO。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import CFG, describe, load_config_file  # noqa: E402
from src.data.preference_builder import EXPERIMENT_PAIR_TYPES, PreferencePair, read_jsonl  # noqa: E402
from src.training.compat import (  # noqa: E402
    build_dpo_config,
    build_trainer,
    make_compat_report,
    recommended_pins,
    warn_if_versions_drift,
)
from src.training.dpo_dataset import (  # noqa: E402
    load_preference_jsonl,
    preference_data_stats,
    token_length_stats,
    truncation_report,
    verify_prompt_consistency,
)
from src.training.dpo_loss import report_self_check  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DPO 训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--experiment", required=True, help=f"实验名，可选: {', '.join(EXPERIMENT_PAIR_TYPES)}")
    p.add_argument("--config", default=None, help="训练配置 yaml")
    p.add_argument("--data-file", default=None, help="直接指定偏好数据 jsonl（覆盖 config 里的声明）")
    p.add_argument("--output-dir", default=None, help="输出目录（默认 outputs/<experiment>）")
    p.add_argument("--model", default=None, help="基座模型")
    p.add_argument("--init-adapter", default=None, help="初始 LoRA 适配器（用于 SFT→DPO 续训，见脚本说明）")

    # 超参覆盖
    p.add_argument("--beta", type=float, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epochs", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--lora-r", type=int, default=None)
    p.add_argument("--lora-alpha", type=int, default=None)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--max-prompt-length", type=int, default=None)
    p.add_argument("--loss-type", default=None, help="sigmoid / ipo / label_smoothing（本项目的实现；与 TRL 取值不同）")
    p.add_argument("--seed", type=int, default=None)

    # 运行模式
    p.add_argument("--mock", action="store_true", help="mock 训练（无需 GPU，验证数据与产物）")
    p.add_argument("--dry-run", action="store_true", help="只做数据体检，不训练")
    p.add_argument("--max-steps", type=int, default=-1, help="限制训练步数（调试用；-1 表示按 epochs）")
    p.add_argument("--max-samples", type=int, default=0, help="只用前 N 条训练数据（调试用）")
    p.add_argument("--eval-after", action="store_true", help="训练后立即用该 adapter 跑评测")
    p.add_argument("--no-eval-during-train", action="store_true", help="训练中不做验证（省时间）")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


# ============================================================================
# 数据准备
# ============================================================================


def resolve_data_file(args: argparse.Namespace, file_cfg: dict) -> Path:
    """确定训练数据文件路径。"""
    if args.data_file:
        p = Path(args.data_file)
        return p if p.is_absolute() else CFG.path() / p

    pairs_cfg = file_cfg.get("pairs", {})
    declared = pairs_cfg.get("train_file")
    candidates: list[Path] = []
    if declared:
        candidates.append(CFG.path() / declared)
    # 常规位置
    candidates += [
        CFG.path("data", "preferences", f"{args.experiment}.train.jsonl"),
        CFG.path("data", "preferences", "all.train.jsonl"),
        CFG.path("outputs", "mock_pipeline", f"preferences_{args.experiment}.train.mock.jsonl"),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "找不到训练数据。已尝试：\n  "
        + "\n  ".join(str(c) for c in candidates)
        + "\n请先运行：python scripts/prepare/build_preferences.py"
    )


def resolve_dev_file(args: argparse.Namespace, file_cfg: dict) -> Path | None:
    pairs_cfg = file_cfg.get("pairs", {})
    declared = pairs_cfg.get("dev_file")
    candidates: list[Path] = []
    if declared:
        candidates.append(CFG.path() / declared)
    candidates += [
        CFG.path("data", "preferences", f"{args.experiment}.dev.jsonl"),
        CFG.path("outputs", "mock_pipeline", f"preferences_{args.experiment}.dev.mock.jsonl"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def data_health_check(
    train_file: Path,
    dev_file: Path | None,
    *,
    max_length: int,
    max_prompt_length: int,
    max_samples: int = 0,
    verbose: bool = True,
) -> dict:
    """训练前的数据体检。返回体检报告。

    没有 tokenizer 时退化为字符长度粗估，不会失败。
    """
    rows = load_preference_jsonl(train_file)
    if max_samples:
        rows = rows[:max_samples]

    report: dict = {
        "train_file": str(train_file),
        "dev_file": str(dev_file) if dev_file else None,
        "num_train": len(rows),
    }
    stats = preference_data_stats(rows)
    report["stats"] = stats.to_dict()
    if verbose:
        for line in stats.summary_lines():
            print("  " + line)

    # 质量审计（复用 inspect 的逻辑）
    try:
        from scripts.prepare.inspect_preferences import audit_rows

        audit = audit_rows(rows)
        report["audit"] = audit
        if verbose:
            if audit["passed"]:
                print("  ✓ 数据审计通过")
            else:
                print(f"  ✗ 数据审计发现问题: {audit['problem_counts']}")
    except Exception as exc:  # pragma: no cover
        report["audit"] = {"error": f"{type(exc).__name__}: {exc}"}

    # prompt 一致性
    try:
        from src.data.task_manifest import load_tasks
        from src.training.dpo_dataset import verify_prompt_consistency

        cons = verify_prompt_consistency(rows, load_tasks())
        report["prompt_consistency"] = cons
        if verbose:
            if cons["num_mismatch"] == 0:
                print(f"  ✓ prompt 一致性通过（检查 {cons['checked']} 条）")
            else:
                print(f"  ✗ prompt 不一致 {cons['num_mismatch']} 条 —— 训练与推理分布不同！")
    except Exception as exc:  # pragma: no cover
        report["prompt_consistency"] = {"error": str(exc)}

    # token 长度
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(CFG.model.resolved, trust_remote_code=True)
        st = token_length_stats(rows, tok)
        report["token_length"] = st.to_dict()
        rec = st.recommend_max_length()
        report["recommended_max_length"] = rec
        tr = truncation_report(rows, tok, max_length, max_prompt_length)
        report["truncation"] = tr
        if verbose:
            print(
                f"  长度: prompt p99={st.prompt_p99}  prompt+comp p99={st.total_p99}  "
                f"max={st.total_max}"
            )
            print(
                f"  当前 max_length={max_length} / max_prompt_length={max_prompt_length} "
                f"→ 截断率 prompt={tr['prompt_truncation_rate']:.1%}, "
                f"completion={tr['completion_truncation_rate']:.1%}, "
                f"截断后退化={tr['degenerate_after_truncation_rate']:.1%}"
            )
            # ⚠️ prompt 截断率必须单独报警。
            #
            # 开发期踩过：这里原先只打印 completion 与 degenerate 两项，
            # 而 prompt 截断率虽然算出来了却没显示。于是当
            # max_prompt_length=1024 而 prompt p99=1097 / max=1136 时，
            # 报告显示「completion=0.0%, 退化=0.0%」看起来完全健康 ——
            # 但**最长的那批 prompt 已经被截掉了**。
            #
            # 截断 prompt 的后果比截断 completion 严重得多：system prompt
            # 里的工具定义会被切掉，模型看不到部分工具的参数说明，
            # 于是学到错误的工具语义，而 loss 曲线不会有任何异常。
            if tr["prompt_truncation_rate"] > 0:
                print(
                    f"  ⚠️ prompt 截断率 {tr['prompt_truncation_rate']:.1%} —— "
                    f"system prompt 的工具定义会被切掉，模型学不到完整参数说明。"
                    f"请把 max_prompt_length 提到 ≥ prompt max（{st.prompt_max}）"
                )
            if rec["max_length"] > max_length or rec["max_prompt_length"] > max_prompt_length:
                print(
                    f"  ⚠️ 建议把 max_length 提到 {rec['max_length']}、"
                    f"max_prompt_length 提到 {rec['max_prompt_length']}（按 p99）"
                )
            if tr["degenerate_after_truncation_rate"] > 0.05:
                print("  ⚠️ 截断后退化率 > 5%：这些样本对 DPO 是噪声，建议调大 max_length")
    except ImportError:
        report["token_length"] = {"note": "未安装 transformers，跳过（用字符长度粗估）"}
        lens = [len(r["prompt"]) + max(len(r["chosen"]), len(r["rejected"])) for r in rows]
        if lens:
            lens.sort()
            report["char_length"] = {
                "p50": lens[len(lens) // 2],
                "p99": lens[int(len(lens) * 0.99)],
                "max": lens[-1],
            }
            if verbose:
                print(
                    f"  字符长度（粗估）: p50={report['char_length']['p50']} "
                    f"p99={report['char_length']['p99']} max={report['char_length']['max']}"
                )
    except Exception as exc:
        report["token_length"] = {"error": f"{type(exc).__name__}: {exc}"}

    return report


# ============================================================================
# Mock 路径
# ============================================================================


def run_mock(args: argparse.Namespace, train_file: Path, out_dir: Path, train_kwargs: dict) -> int:
    """mock 训练：用真实 DPO 损失做优化，产出同构产物。"""
    from src.training.mock_trainer import MockTrainConfig, run_mock_training

    print("\n" + "=" * 74)
    print("MOCK 训练模式（不需要 GPU，产物带 is_mock 标记）")
    print("=" * 74)
    print(report_self_check())
    print()

    pairs = [PreferencePair(**{k: v for k, v in r.items() if k in PreferencePair.__dataclass_fields__})
             for r in read_jsonl(train_file)]
    # tools 字段不在 dataclass 的构造参数里（默认空），这里去掉以免冲突
    print(f"  训练数据: {len(pairs)} 条")

    cfg = MockTrainConfig(
        experiment=args.experiment,
        output_dir=str(out_dir),
        num_epochs=int(train_kwargs.get("epochs", 1)),
        # ⚠️ 显式 int()：train_kwargs 里的 epochs 可能是 float，
        # 直接整除会得到 float，而 MockTrainConfig.steps_per_epoch 标注为 int。
        # 虽然 range() 不直接用这个值，但类型标注与实参不符会给后续维护埋雷。
        steps_per_epoch=max(
            1,
            int(train_kwargs.get("max_steps", 120)) // max(1, int(train_kwargs.get("epochs", 1))),
        ),
        eval_every=max(1, int(train_kwargs.get("max_steps", 120)) // 6),
        lr=float(train_kwargs.get("lr", 5e-5)),
        beta=float(train_kwargs.get("beta", 0.1)),
        seed=int(train_kwargs.get("seed", 42)),
    )
    res = run_mock_training(cfg, pairs, verbose=not args.quiet)

    print()
    print("=" * 74)
    print("MOCK 训练完成")
    print("=" * 74)
    print(f"  loss  {res.initial_loss:.4f} → {res.final_loss:.4f}（{res.num_steps} 步）")
    print(f"  曲线  {res.curves_path}")
    print(f"  日志  {res.log_path}")
    print(f"  适配器目录（占位，不含真实权重）: {res.adapter_dir}")
    print("\n⚠️ 这是 MOCK 训练：loss 曲线来自合成 logprob 上的真实优化，")
    print("   不是 Qwen2.5-1.5B 的真实训练。产物不能作为项目结论。")
    return 0


# ============================================================================
# 真实训练路径
# ============================================================================


def run_real(args: argparse.Namespace, train_file: Path, dev_file: Path | None, out_dir: Path, train_kwargs: dict) -> int:
    """真实 DPO 训练。"""
    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        print(
            f"错误：真实训练需要 torch / transformers / peft / trl，当前缺失：{exc}\n"
            "  安装: pip install -r requirements.txt\n"
            "  或先用 mock 路径验证: python scripts/train/dpo_train.py --experiment "
            f"{args.experiment} --mock",
            file=sys.stderr,
        )
        return 1

    rep = make_compat_report()
    print(rep.summary())
    print()
    for w in warn_if_versions_drift(rep):
        print(f"  ⚠️ {w}")
    if not rep.dpo_config_fields:
        print("\n错误：当前环境没有 trl 的 DPOConfig，无法训练。", file=sys.stderr)
        print(f"  推荐版本: {json.dumps(recommended_pins(), ensure_ascii=False)}", file=sys.stderr)
        return 1

    # --- 构建 TrainCfg ---
    from dataclasses import replace

    tcfg = CFG.train
    overrides = {k: v for k, v in train_kwargs.items() if k in tcfg.__dataclass_fields__ and v is not None}
    tcfg = replace(tcfg, **overrides)

    print(f"  输出目录  : {out_dir}")
    print(f"  训练数据  : {train_file}")
    print(f"  验证数据  : {dev_file or '无'}")
    print(f"  β={tcfg.beta}  lr={tcfg.lr}  epochs={tcfg.epochs}  "
          f"bs={tcfg.batch_size}×{tcfg.grad_accum}  LoRA r={tcfg.lora_r}/{tcfg.lora_alpha}")
    print(f"  max_length={tcfg.max_length}  max_prompt_length={tcfg.max_prompt_length}")
    print()

    # --- 加载模型 ---
    model_name = args.model or CFG.model.resolved
    print(f"加载基座模型 {model_name} ...")
    dtype = torch.bfloat16 if tcfg.bf16 and torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.config.use_cache = False  # 与 gradient_checkpointing 冲突

    # --- SFT → DPO 续训：先 merge 掉已有 adapter ---
    if args.init_adapter:
        from peft import PeftModel

        print(f"合并初始 adapter {args.init_adapter} 到基座 ...")
        print("  理由：ref_model=None 时 reference 是 base；若继续沿用同一个")
        print("        SFT adapter，KL 锚点会偏离 base，导致 DPO 训练不稳。")
        model = PeftModel.from_pretrained(model, args.init_adapter)
        model = model.merge_and_unload()
        merged_dir = out_dir / "base_with_init_adapter"
        merged_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))
        print(f"  已保存合并后的基座: {merged_dir}")

    if tcfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

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

    # --- 数据集 ---
    from src.training.dpo_dataset import to_dpo_dataset

    rows = load_preference_jsonl(train_file)
    if args.max_samples:
        rows = rows[: args.max_samples]
    train_ds = to_dpo_dataset(rows)
    eval_ds = None
    if dev_file and not args.no_eval_during_train:
        eval_ds = to_dpo_dataset(load_preference_jsonl(dev_file))
    print(f"  数据集: train={len(train_ds)}  eval={len(eval_ds) if eval_ds is not None else 0}")

    # --- Trainer ---
    extra: dict = {}
    if args.max_steps and args.max_steps > 0:
        extra["max_steps"] = args.max_steps
    if eval_ds is None:
        extra["eval_strategy"] = "no"

    dpo_config = build_dpo_config(tcfg, str(out_dir), rep, **extra)
    # DPOTrainer 需要 DPOConfig；build_trainer 里会再构造一次，
    # 这里改成先构造 config 再直接调 DPOTrainer，避免重复。
    from trl import DPOTrainer

    trainer_kwargs: dict = {
        "model": model,
        "args": dpo_config,
        "train_dataset": train_ds,
        "peft_config": peft_config,
    }
    if eval_ds is not None:
        trainer_kwargs["eval_dataset"] = eval_ds
    if rep.uses_processing_class:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    import inspect

    accepted = set(inspect.signature(DPOTrainer.__init__).parameters)
    dropped = sorted(set(trainer_kwargs) - accepted)
    if dropped:
        rep.dropped_fields.extend(dropped)
        rep.notes.append(f"DPOTrainer 不接受: {dropped}")
        trainer_kwargs = {k: v for k, v in trainer_kwargs.items() if k in accepted}

    print("构造 DPOTrainer ...")
    trainer = DPOTrainer(**trainer_kwargs)

    # --- 挂上实时监控回调 ---
    #
    # ⚠️ 为什么必须挂：TRL 的 Trainer 把指标交给 report_to（tensorboard），
    # **不打印到控制台**。没有这个回调的话，训练过程中你只能看到进度条，
    # 完全看不到 loss / rewards/margin / rewards/accuracy —— 而 DPO 恰恰是
    # 「看这几个量就能提前发现数据问题」的方法。本项目开发期就因此
    # 盯着一个不存在的日志等了半小时。
    from src.training.train_monitor import DPOMonitorCallback, summarize_dpo_health

    monitor = DPOMonitorCallback(out_dir, verbose=not args.quiet)
    try:
        trainer.add_callback(monitor)
        print("  已挂载训练监控回调（指标会实时打印并写入 train_log.jsonl）")
    except Exception as exc:  # pragma: no cover
        print(f"  ⚠️ 监控回调挂载失败（不影响训练）: {type(exc).__name__}: {exc}")

    # --- 训练 ---
    print("\n开始训练 ...")
    t0 = time.time()
    train_result = trainer.train()
    wall = time.time() - t0

    # --- 训练后体检 ---
    health = summarize_dpo_health(out_dir / "train_log.jsonl")
    print("\n" + health)
    (out_dir / "train_health.txt").write_text(health + "\n", encoding="utf-8")

    # --- 保存 ---
    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\n已保存 LoRA 适配器: {final_dir}")

    # --- 产物 ---
    metrics = dict(train_result.metrics) if hasattr(train_result, "metrics") else {}
    (out_dir / "train_config.json").write_text(
        json.dumps(
            {
                "experiment": args.experiment,
                "model_name": model_name,
                "init_adapter": args.init_adapter,
                "train_file": str(train_file),
                "dev_file": str(dev_file) if dev_file else None,
                "num_train": len(train_ds),
                "num_eval": len(eval_ds) if eval_ds is not None else 0,
                "is_mock": False,
                "wall_time_s": round(wall, 1),
                "hyperparameters": {k: getattr(tcfg, k) for k in tcfg.__dataclass_fields__},
                "final_metrics": metrics,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    (out_dir / "compat_report.json").write_text(
        json.dumps(rep.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8"
    )

    print(f"\n训练用时 {wall / 60:.1f} 分钟")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v}")

    if args.eval_after:
        print("\n训练后评测 ...")
        from scripts.eval.run_baseline import main as baseline_main

        return baseline_main(
            [
                "--adapter", str(final_dir),
                "--label", args.experiment,
                "--out-dir", str(CFG.path("experiments", args.experiment)),
            ]
        )
    print(
        f"\n下一步：\n"
        f"  python scripts/eval/run_baseline.py --adapter {final_dir} --label {args.experiment}"
    )
    return 0


# ============================================================================
# main
# ============================================================================


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t_start = time.time()

    if args.experiment not in EXPERIMENT_PAIR_TYPES:
        print(
            f"错误：未知实验 {args.experiment!r}\n可选: {', '.join(EXPERIMENT_PAIR_TYPES)}",
            file=sys.stderr,
        )
        return 2

    file_cfg = {}
    cfg_path = args.config
    if cfg_path is None:
        # 自动推导配置文件：``--experiment dpo_joint`` 已经唯一确定了
        # ``configs/train/dpo_joint.yaml``，不该让用户再记一个参数。
        #
        # ⚠️ **这里有一个后果很隐蔽的坑。**
        #
        # 原实现是 ``if args.config: ...`` —— 不传就静默用全局默认值。
        # 实测中跑 ``--experiment dpo_joint --dry-run``（忘了加 --config）时，
        # 所有超参都退回 ``CFG.train`` 的默认值，于是：
        #
        #     max_length 用的是 1536 而不是配置里的 1280
        #     max_prompt_length 用的是 1024 而不是 1152
        #
        # 而 ``max_prompt_length=1024`` 会让 **23.2% 的 prompt 被截断**
        # （实测数字）—— system prompt 的工具定义被切掉，
        # 模型学不到完整参数说明，但 loss 曲线不会有任何异常。
        #
        # 这个 bug 的形态是「默认值恰好是个危险值」：不传参数不是报错，
        # 而是静默降级到一个会损坏训练数据的配置。所以必须自动推导 +
        # 明确告知用的是哪个文件。
        guess = CFG.path("configs", "train", f"{args.experiment}.yaml")
        if guess.exists():
            cfg_path = str(guess)
            print(f"[配置] 未指定 --config，自动使用 {guess.relative_to(CFG.path())}")
        else:
            print(
                f"[配置] 未指定 --config，且 {guess} 不存在 → 使用全局默认值"
                f"（max_length={CFG.train.max_length} / "
                f"max_prompt_length={CFG.train.max_prompt_length}）"
            )
    if cfg_path:
        file_cfg = {k: v for k, v in load_config_file(cfg_path).items() if not k.startswith("_")}

    print(describe())
    print()
    print("=" * 74)
    print(f"DPO 训练: {args.experiment}")
    print("=" * 74)

    # --- 数据 ---
    train_file = resolve_data_file(args, file_cfg)
    dev_file = resolve_dev_file(args, file_cfg)
    out_dir = Path(args.output_dir) if args.output_dir else CFG.path("outputs", args.experiment)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 超参（命令行 > 配置文件 > 全局）---
    tcfg_file = file_cfg.get("train", {})
    train_kwargs: dict = {
        "beta": args.beta if args.beta is not None else tcfg_file.get("beta"),
        "lr": args.lr if args.lr is not None else tcfg_file.get("lr"),
        "epochs": args.epochs if args.epochs is not None else tcfg_file.get("epochs"),
        "batch_size": args.batch_size if args.batch_size is not None else tcfg_file.get("batch_size"),
        "grad_accum": args.grad_accum if args.grad_accum is not None else tcfg_file.get("grad_accum"),
        "lora_r": args.lora_r if args.lora_r is not None else tcfg_file.get("lora_r"),
        "lora_alpha": args.lora_alpha if args.lora_alpha is not None else tcfg_file.get("lora_alpha"),
        "max_length": args.max_length if args.max_length is not None else tcfg_file.get("max_length"),
        "max_prompt_length": (
            args.max_prompt_length
            if args.max_prompt_length is not None
            else tcfg_file.get("max_prompt_length")
        ),
        "loss_type": args.loss_type if args.loss_type is not None else tcfg_file.get("loss_type"),
        "seed": args.seed if args.seed is not None else tcfg_file.get("seed"),
    }
    train_kwargs = {k: v for k, v in train_kwargs.items() if v is not None}

    eff_max_length = int(train_kwargs.get("max_length", CFG.train.max_length))
    eff_max_prompt = int(train_kwargs.get("max_prompt_length", CFG.train.max_prompt_length))

    # --- 损失实现自检（每次训练前都跑，成本可忽略）---
    print("\n[1/3] DPO 损失实现自检")
    print(report_self_check())

    # --- 数据体检 ---
    print("\n[2/3] 数据体检")
    health = data_health_check(
        train_file,
        dev_file,
        max_length=eff_max_length,
        max_prompt_length=eff_max_prompt,
        max_samples=args.max_samples,
        verbose=not args.quiet,
    )
    (out_dir / "data_health.json").write_text(
        json.dumps(health, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    if args.dry_run:
        print("\n--dry-run：体检完成，未开始训练。")
        print(f"  体检报告: {out_dir / 'data_health.json'}")
        rec = health.get("recommended_max_length")
        if rec:
            print(
                f"  建议命令: 加 --max-length {rec['max_length']} "
                f"--max-prompt-length {rec['max_prompt_length']}"
            )
        return 0

    # --- 训练 ---
    print("\n[3/3] 训练")
    if args.mock:
        rc = run_mock(args, train_file, out_dir, train_kwargs)
    else:
        rc = run_real(args, train_file, dev_file, out_dir, train_kwargs)

    print(f"\n总用时 {time.time() - t_start:.1f}s")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
