"""基线评测：跑未训练（或指定 checkpoint）的模型，得到诊断数据。

用法
----
::

    # CPU 冒烟（用 mock 策略，不加载模型）
    python scripts/eval/run_baseline.py --mock --task-limit 20

    # 真实基线（GPU，需要 ~10 分钟 / 100 任务 × 4 采样）
    python scripts/eval/run_baseline.py --config configs/eval/eval_baseline.yaml

    # 评测某个 DPO checkpoint
    python scripts/eval/run_baseline.py --adapter outputs/dpo_joint/final --label dpo_joint

    # 混淆因素排查：精简 system prompt（验证提升是否只是来自更啰嗦的提示词）
    python scripts/eval/run_baseline.py --lite-system --label baseline_lite

    # 限定工具集合（测试选择空间大小对幻觉率的影响）
    python scripts/eval/run_baseline.py --tools search_flights,get_weather --label baseline_2tools

产出
----
::

    experiments/<label>/
    ├── eval_report.json          完整评测报告（报告只引用这个文件）
    ├── predictions.jsonl         逐次采样的原始输出（所有分析的唯一来源）
    ├── error_analysis.json       错误分布与子类型
    └── failure_cases/            按错误类型抽出的失败案例（供人工检查）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import CFG, describe  # noqa: E402
from src.data.task_manifest import load_tasks, summary as task_summary  # noqa: E402
from src.eval.error_taxonomy import CATEGORY_LABELS_ZH, ErrorDistribution  # noqa: E402
from src.eval.grader import grade, run_evaluation, write_predictions  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="基线 / checkpoint 评测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", default=None, help="评测配置文件（yaml）")
    p.add_argument("--label", default=None, help="实验标签，决定输出目录名")
    p.add_argument("--adapter", default=None, help="LoRA 适配器目录（不填=原始基座）")
    p.add_argument("--model", default=None, help="基座模型名或路径")
    p.add_argument("--num-samples", type=int, default=None, help="每任务采样次数")
    p.add_argument("--primary-k", type=int, default=None, help="主指标用哪个 k")
    p.add_argument(
        "--k-values",
        default=None,
        help="要报告的 k 值，逗号分隔（如 1,2,4）。覆盖配置文件的 eval.k_values",
    )
    p.add_argument("--task-limit", type=int, default=0, help="只跑前 N 个任务（0=全部）")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--tools", default=None, help="限定工具集合，逗号分隔")
    p.add_argument("--lite-system", action="store_true", help="用精简版 system prompt")
    p.add_argument(
        "--no-lite-system",
        action="store_true",
        help="强制关闭精简版 system prompt（覆盖配置文件的 variants.system_prompt_lite）",
    )
    p.add_argument(
        "--task-set",
        default="v1",
        choices=["v1", "v3"],
        help="评测协议版本：v1 = 旧 100 任务（历史结果），v3 = 新高质量 100 任务",
    )
    p.add_argument(
        "--keep-contaminated",
        action="store_true",
        help="不排除「prompt 与训练语料逐字相同」的评测任务。"
        "默认排除（评测集 95 个任务）；仅在复现早期结果时使用",
    )
    p.add_argument("--no-observations", action="store_true", help="把工具返回结果替换成占位说明")
    p.add_argument("--mock", action="store_true", help="用 mock 策略而不是真实模型（CPU 冒烟）")
    p.add_argument("--mock-policy", default="baseline", help="mock 用哪个实验的错误率假设")
    p.add_argument("--out-dir", default=None, help="输出目录（默认 experiments/<label>）")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--quiet", action="store_true", help="不打印逐个任务的进度")
    return p.parse_args(argv)


def load_eval_config(path: str | None) -> dict:
    """读评测配置；没给就返回空字典（全部用命令行/全局默认）。"""
    if not path:
        return {}
    from src.config import load_config_file

    cfg = load_config_file(path)
    # 去掉以 _ 开头的注释键，避免污染后续逻辑
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t_start = time.time()

    file_cfg = load_eval_config(args.config)
    model_cfg = file_cfg.get("model", {})
    gen_cfg = file_cfg.get("generation", {})
    eval_cfg = file_cfg.get("eval", {})
    out_cfg = file_cfg.get("output", {})

    # --- 解析参数（命令行 > 配置文件 > 全局默认）---
    adapter = args.adapter if args.adapter is not None else model_cfg.get("adapter_path")
    # label 的优先级：显式 --label > adapter 目录名 > baseline
    if args.label:
        label = args.label
    elif adapter:
        label = Path(adapter).parent.name or Path(adapter).name
    elif args.mock:
        label = f"mock_{args.mock_policy}"
    else:
        label = "baseline"
    num_samples = args.num_samples or eval_cfg.get("num_samples_per_task") or CFG.eval.num_samples_per_task
    primary_k = args.primary_k or eval_cfg.get("primary_k") or CFG.eval.primary_k
    temperature = args.temperature if args.temperature is not None else gen_cfg.get("temperature", CFG.model.temperature)
    max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else gen_cfg.get("max_new_tokens", CFG.model.max_new_tokens)
    )
    tool_names = [t.strip() for t in args.tools.split(",") if t.strip()] if args.tools else None
    seed = args.seed if args.seed is not None else CFG.seed

    # --- 混淆因素开关：命令行 > 配置文件的 variants 段 > 默认 ---
    #
    # ⚠️ 这里必须真的读取 ``variants``，否则配置文件里的开关是**死的** ——
    # 改 yaml 不生效是最难排查的一类问题（表面看配置改了，实际没作用）。
    # 早期版本就是这样：variants 段只作为文档存在，代码完全忽略它。
    variants_cfg = file_cfg.get("variants", {})
    lite_system = bool(variants_cfg.get("system_prompt_lite", False))
    if args.lite_system:
        lite_system = True
    if args.no_lite_system:
        lite_system = False

    if tool_names is None:
        cfg_tools = variants_cfg.get("tool_names")
        if cfg_tools:
            tool_names = [str(t) for t in cfg_tools]

    include_observations = bool(variants_cfg.get("include_observations", True))
    if args.no_observations:
        include_observations = False

    # --- 采样次数与 k 值的一致性 ---
    # k 值优先取配置文件，缺省回退全局 CFG。
    # （早期版本只读 CFG，导致 eval_baseline.yaml 里的 k_values 静默失效 ——
    #   改配置文件不生效是最难排查的一类问题，所以这里显式读。）
    cfg_k = eval_cfg.get("k_values")
    if args.k_values:
        wanted_k = tuple(int(x) for x in args.k_values.split(",") if x.strip())
    elif cfg_k:
        wanted_k = tuple(int(x) for x in cfg_k)
    else:
        wanted_k = tuple(CFG.eval.pass_k_values)

    # 允许通过 --num-samples 临时降低采样数（快速验证常用），
    # 此时自动把 primary_k 与 k_values 收敛到合法范围，而不是直接报错。
    # 但必须在输出里**大声说明**——否则一次「快速验证」的结果
    # 会被误当成与全量协议可比的正式结果。
    k_values = tuple(k for k in wanted_k if k <= num_samples) or (1,)
    if primary_k > num_samples:
        print(
            f"⚠️ primary_k={primary_k} 超过 num_samples={num_samples}，"
            f"自动降为 primary_k={num_samples}。\n"
            f"   本次结果的 pass^{num_samples} 与全量协议（pass^{CFG.eval.primary_k}）"
            f"**不可直接比较**，请勿混入正式对比表。",
            file=sys.stderr,
        )
        primary_k = num_samples
    if set(k_values) != set(wanted_k):
        print(
            f"⚠️ 报告 k 值收敛为 {list(k_values)}（协议要求 {list(wanted_k)}，"
            f"采样数 {num_samples} 不足以估计较大的 k）",
            file=sys.stderr,
        )

    out_dir = (Path(args.out_dir) if args.out_dir else CFG.path() / out_cfg.get("dir", f"experiments/{label}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(describe())
    print()
    print("=" * 74)
    print(f"评测任务: {label}")
    print("=" * 74)
    print(f"  输出目录   : {out_dir}")
    print(f"  模型       : {args.model or CFG.model.resolved}")
    print(f"  适配器     : {adapter or '无（原始基座）'}")
    print(f"  采样       : T={temperature} max_new_tokens={max_new_tokens}")
    print(f"  协议       : {num_samples} 采样/任务，主指标 pass^{primary_k}")
    print(f"  工具集合   : {tool_names or '全部 6 个'}")
    print(f"  system     : {'精简版' if lite_system else '完整版'}")
    print(f"  观测       : {'占位' if not include_observations else '真实'}")
    print(f"  模式       : {'MOCK（CPU 冒烟，非真实结果）' if args.mock else '真实模型'}")
    print()

    # 评测协议版本：v1 = 100 任务（历史结果用），v2 = 400 任务（新协议）。
    # v2 的前 100 个与 v1 逐字相同，因此两套结果可逐任务配对比较。
    from src.data.task_manifest import load_tasks_by_version

    tasks = load_tasks_by_version(getattr(args, "task_set", "v1"), seed=seed)
    print(f"  协议版本   : {getattr(args, 'task_set', 'v1')}（{len(tasks)} 任务）")

    # ⚠️ 排除「prompt 与训练语料逐字相同」的评测任务。
    #
    # 合成训练语料与评测任务共用同一批领域短句，所以即便修掉全部污染，
    # 仍会有几处撞车（实测 5 个）。在这些任务上测出的成绩**不能代表泛化能力**。
    # 默认排除，评测集为 95 个任务。
    #
    # 判据是**整个渲染后 prompt 逐字相同** —— 用子串比对会把
    # 「查一下订单 X 的行李额度」这类通用短句误报为泄漏（实测误报 216 条）。
    if not getattr(args, "keep_contaminated", False):
        from src.data.recovery_corpus import (
            find_contaminated_eval_tasks,
            find_expected_leaks,
        )

        bad = set(find_contaminated_eval_tasks(tasks))
        # ⚠️ **第二层检查**：prompt 不同、但**完整期望调用**
        # （工具 + 参数名 + 参数值）与某条训练样本完全相同。
        # 这种情形下模型可能靠匹配记忆中的样本答对，而不是学会规则。
        # 原先只查 prompt 逐字相同，实测漏掉 5 个（4 个 get_weather + 1 个
        # search_flights）。
        leaked = set(find_expected_leaks(tasks, task_set=args.task_set))
        bad |= leaked
        if bad:
            before = len(tasks)
            tasks = [x for x in tasks if x.task_id not in bad]
            print(f"  已排除 {len(bad)} 个与训练语料重合的任务"
                  f"（{before} → {len(tasks)}）")
            if leaked:
                print(f"    · 其中 {len(leaked)} 个是「完整期望调用相同」的"
                      f"实体泄漏: {sorted(leaked)}")
            print()

    if args.task_limit:
        tasks = tasks[: args.task_limit]
    print(task_summary(tasks))

    # ⚠️ seen/unseen 标签在「合成语料覆盖整个评测分布」的配置下**没有含义**。
    # 实测：seen 组签名覆盖 67/68、unseen 组 38/38 —— unseen 也全部见过。
    # 把两者的差值解释成「泛化差距」是错的。每次评测都打印，避免我忘掉。
    try:
        from src.data.task_manifest_v2 import report_visibility_warning

        _w = report_visibility_warning()
        if _w:
            print()
            print(_w)
    except Exception:
        pass
    print()

    # --- 构造策略 ---
    metadata = {
        "label": label,
        "engine": "mock" if args.mock else "hf",
        "is_mock": bool(args.mock),
        "model_name": args.model or CFG.model.resolved,
        "adapter_path": adapter,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "tool_names": tool_names,
        "system_prompt_lite": lite_system,
        "include_observations": include_observations,
        "num_samples_per_task": num_samples,
        "primary_k": primary_k,
        "task_limit": args.task_limit,
        "seed": seed,
        "config_file": args.config,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if args.mock:
        from src.mock.mock_policy import MockPolicy

        policy = MockPolicy(args.mock_policy, seed=seed)
        policy_fn = policy.as_policy_fn(tasks)
        metadata["mock_policy"] = args.mock_policy
    else:
        from src.models.hf_policy import GenerationConfig, HFPolicy

        hp = HFPolicy(
            model_name=args.model,
            adapter_path=adapter,
            system_prompt_lite=lite_system,
            tool_names=tool_names,
            include_observations=include_observations,
            generation=GenerationConfig(
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                seed=None,
            ),
        )
        print("加载模型 ...")
        hp.load()
        print(f"  显存: {hp.vram_report()}")
        policy_fn = hp.as_policy_fn()

    # --- 跑评测 ---
    #
    # stream_path：边跑边把预测写盘。100 任务 × 4 采样在云上要跑约 30 分钟，
    # 不流式写的话（a）看不到进度、（b）中途崩溃会丢掉全部已完成结果。
    # 文件名的 mock 后缀沿用统一规则（真实与模拟严格分离）。
    #
    # ⚠️ suffix 必须在这里先算出来 —— 后面写报告时还会用同一个变量，
    # 早期版本把 stream_path 放在 suffix 定义之前，直接 UnboundLocalError。
    suffix = ".mock" if args.mock else ""
    report_name = f"eval_report{suffix}.json"
    pred_name = f"predictions{suffix}.jsonl"
    analysis_name = f"error_analysis{suffix}.json"
    pred_path = out_dir / pred_name
    report_path = out_dir / report_name

    print()
    preds, report = run_evaluation(
        policy_fn,
        tasks,
        label=label,
        num_samples_per_task=num_samples,
        k_values=k_values,
        primary_k=primary_k,
        seed=seed,
        metadata=metadata,
        max_new_tokens=max_new_tokens,
        progress=not args.quiet,
        stream_path=pred_path,
    )

    # --- 输出 ---
    print()
    for line in report.headline_lines():
        print(line)
    for line in report.detail_lines():
        print(line)

    # 错误分布
    dist = ErrorDistribution()
    by_id = {t.task_id: t for t in tasks}
    for pred in preds:
        task = by_id.get(pred.task_id)
        if task is None:
            continue
        g = grade(task, pred.text, finish_reason=pred.finish_reason, max_new_tokens=max_new_tokens)
        dist.add(
            g.classification,
            task_id=task.task_id,
            meta={"category": task.category, "difficulty": task.difficulty, "sample_index": pred.sample_index},
        )
    print()
    for line in dist.summary_lines():
        print(line)

    # 格式正确率与「格式对但值错」—— reward hacking 的监控指标
    from src.eval.tool_call_parser import (
        format_accuracy,
        format_accuracy_by_task,
    )

    # ⚠️ 两个口径都给：
    #   * `fmt_acc_taskaware`（**正确口径**）：按任务类型判 —— 「应先追问」
    #     的任务不应有 tool_call，它的正确输出是自然语言追问
    #   * `fmt_acc`（旧口径）：一律要求 canonical tool_call
    #
    # 实测踩过：v3 基线里 `cancellation` 任务成功率 **62.5%**
    # 而旧口径格式正确率 **0.0%** —— 那个 0.0% 是**测量产物**，
    # 因为该类别 22 个任务里 14 个是「应先追问」，它们的正确答案
    # 本来就没有 tool_call。**若不分开报，会把它误读成「模型连格式都不会」。**
    fmt_acc = format_accuracy([p.text for p in preds])
    fmt_acc_taskaware = format_accuracy_by_task([
        (by_id[p.task_id].expected.tool is None, p.text)
        for p in preds if p.task_id in by_id
    ])
    fovw = 0
    for pred in preds:
        task = by_id.get(pred.task_id)
        if task is None:
            continue
        g = grade(task, pred.text, finish_reason=pred.finish_reason)
        fovw += int(g.format_ok_value_wrong)
    print()
    print("退化模式监控:")
    print(f"  格式正确率（按任务）  {fmt_acc_taskaware:.1%}   ← 正确口径")
    print(f"  格式正确率（旧口径）  {fmt_acc:.1%}   ← 一律要求 tool_call，"
          "对「应先追问」任务会误判")
    print(f"  格式对但值错          {fovw} 条 ({fovw / max(1, len(preds)):.1%})")
    print("  ↑ 若格式正确率上升而任务成功率不动、且这一项上升，说明模型走了「格式捷径」")

    # 落盘
    #
    # 预测已在评测过程中边跑边写入 pred_path（见上面的 stream_path 参数）。
    # 这里只写报告与错误分析。
    #
    # 三重隔离保证 mock 数据不会被误当成结论：
    #   1. 文件名后缀：``--mock`` 时报告名固定为 ``eval_report.mock.json``，
    #      真实与模拟结果在文件系统层面天然不同名；
    #   2. 文件内容标记：所有 mock 报告内部带 ``is_mock: true``；
    #   3. 读取侧校验：``collect_results.py`` 默认拒绝任何
    #      ``metadata.is_mock == True`` 的报告（见该脚本的 find_reports）。
    #
    # 另外：如果目标目录**已存在真实结果**，明确告知本次 mock 不会覆盖它。
    real_report = out_dir / "eval_report.json"
    if args.mock and real_report.exists():
        try:
            existing = json.loads(real_report.read_text(encoding="utf-8"))
            if not existing.get("metadata", {}).get("is_mock"):
                print(
                    f"\nℹ️ 该目录已存在**真实**结果：{real_report}\n"
                    f"   本次 mock 结果写到 {report_path}，不会覆盖它。"
                )
        except (json.JSONDecodeError, OSError):
            pass

    # 预测已由 run_evaluation 边跑边写入 stream_path（= pred_path），
    # 这里不再重复写一遍，只写报告与分析。
    report.write_json(str(report_path))

    # ------------------------------------------------------------------
    # 已知限制：**自动生成**，不靠报告作者记得写
    # ------------------------------------------------------------------
    #
    # 这个项目里最危险的错误不是「算错」，而是「**报告漏掉了某个限制**」——
    # 数字本身对，但结论被读成比实际更强的说法。实测发生过的三处：
    #   * `seen`/`unseen` 被当成泛化差距（那个标签**从不读取训练数据**）
    #   * 「14 个 clarify 独立样本」（有效样本是 1 个行为，15 是触发方式数）
    #   * 剔除泄漏后样本量变了，但报告标题没跟着变
    #
    # 所以把它做成**每次评测都生成的产物**：数据变了，文字也跟着变。
    # ------------------------------------------------------------------
    # 簇级口径：同一「期望调用」的任务不是独立证据
    # ------------------------------------------------------------------
    #
    # 主指标用**任务级** bootstrap（`EvalReport` 的既定口径，不改它 ——
    # 改了会让所有历史报告不可比）。但这里**另给一套簇级数字**：
    # 实测 v3 的 106 个任务只有 92 个不同期望调用（15 个 clarify 任务
    # 共享同一个「不调用工具」的期望），按任务重采样会把一个行为重复计数。
    #
    # 两套并列，读者能看到口径差异；报告应以**簇级**为准（更保守）。
    clustered: dict = {}
    try:
        from src.eval import paired_stats as _PS

        _cluster_of = _PS.expected_keys(args.task_set)
        # ⚠️ 复用 `EvalReport` 已经算好的**逐任务准确率**，不要自己重算。
        # 我第一次写这段时用了 `pred.passed` —— **那个字段不存在**
        # （`TaskPrediction` 只有 `task_id/sample_index/text/finish_reason/...`）。
        # 而我这个项目里已经犯过一次「猜属性名」的错
        # （`ArgMatchResult.failures` 是方法不是属性）。
        # 正确做法：读 `SampledTaskResult.accuracy`，它与主指标同源，
        # 不可能与主指标口径不一致。
        _scores = {r.task_id: r.accuracy for r in report.results}
        clustered = _PS.cluster_ci(_scores, _cluster_of)
        if clustered["n_clusters"]:
            print()
            print("簇级口径（同一期望调用的任务归为一簇）:")
            print(f"  有效独立样本    {clustered['n_clusters']} 簇"
                  f"（任务 {clustered['n_tasks']} 个，"
                  f"收缩 {clustered['cluster_shrinkage']}）")
            print(f"  簇级准确率      {clustered['mean']:.2%}"
                  f"  95%CI [{clustered['ci_lo']:.2%}, {clustered['ci_hi']:.2%}]")
            if clustered["cluster_shrinkage"]:
                print(f"  ↑ 主指标按**任务**重采样，这里按**簇** —— "
                      f"后者更保守，报告应以它为准")
    except Exception as exc:  # pragma: no cover
        print(f"  ⚠️ 未能计算簇级口径：{exc}")

    limitations: list[dict] = []
    limits_md = ""
    try:
        from src.eval.limitations import collect_limitations, render_markdown

        limitations = collect_limitations(tasks, task_set=args.task_set)
        limits_md = render_markdown(limitations, n_tasks=len(tasks), label=label)
        (out_dir / "limitations.md").write_text(limits_md, encoding="utf-8")
        if limitations:
            print()
            print("已知限制（详见 limitations.md）：")
            for i, x in enumerate(limitations, 1):
                print(f"  {i}. [{x['severity']}] {x['id']}")
            print(f"  → 这些限制若不在报告里声明，读者会把结论读得比实际更强")
    except Exception as exc:  # pragma: no cover
        print(f"  ⚠️ 未能生成已知限制章节：{exc}")

    (out_dir / analysis_name).write_text(
        json.dumps(
            {
                "label": label,
                "is_mock": bool(args.mock),
                "n_tasks": len(tasks),
                "format_accuracy": round(fmt_acc, 6),
                # 按任务类型的正确口径（见 run_baseline 里的说明）
                "format_accuracy_task_aware": round(fmt_acc_taskaware, 6),
                "format_ok_value_wrong": fovw,
                "format_ok_value_wrong_rate": round(fovw / max(1, len(preds)), 6),
                "distribution": dist.to_dict(),
                # ⚠️ 限制随数据落盘 —— 报告若只引用 json 也能看到它们
                "limitations": limitations,
                # 簇级口径（有效独立样本数）—— 报告应优先引用它
                "clustered": clustered,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )

    # 失败案例：每类抽几个，供人工检查
    cases_dir = out_dir / "failure_cases"
    cases_dir.mkdir(exist_ok=True)
    idx = 0
    for pred in preds:
        task = by_id.get(pred.task_id)
        if task is None:
            continue
        g = grade(task, pred.text, finish_reason=pred.finish_reason, max_new_tokens=max_new_tokens)
        if g.passed:
            continue
        cat = g.category
        rec = {
            "case_id": f"case_{idx:03d}",
            "task_id": task.task_id,
            "category": task.category,
            "difficulty": task.difficulty,
            "visibility": task.visibility,
            "is_mock": bool(args.mock),
            "expected": {"tool": task.expected.tool, "args": task.expected.graded_args},
            "classification": g.classification.to_dict(),
            "defects": g.defects,
            "raw_output": pred.text[:2000],
            "finish_reason": pred.finish_reason,
        }
        (cases_dir / f"{rec['case_id']}_{task.task_id}_{cat}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        idx += 1
        if idx >= 40:  # 控制体积
            break

    if not args.mock:
        try:
            hp.unload()
        except Exception:
            pass

    print()
    print("=" * 74)
    print(f"用时 {time.time() - t_start:.1f}s")
    print(f"  报告        : {report_path}")
    print(f"  原始预测    : {pred_path}（{len(preds)} 条，边跑边写）")
    print(f"  错误分析    : {out_dir / analysis_name}")
    print(f"  失败案例    : {cases_dir}（{idx} 个）")
    if args.mock:
        print(
            "\n⚠️ 这是 MOCK 结果（--mock），不能作为项目结论。\n"
            "   真实评测请去掉 --mock 并确保有 GPU。"
        )
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
