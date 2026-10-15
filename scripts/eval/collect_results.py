"""汇总各实验的评测报告，生成消融对比表与图表。

用法
----
::

    # 汇总全部真实实验（默认拒绝 mock 结果）
    python scripts/eval/collect_results.py

    # 指定实验与顺序
    python scripts/eval/collect_results.py --experiments baseline,dpo_format,dpo_recovery,dpo_joint

    # 把 mock 结果也纳入（仅供流水线自检，产物会明确标注）
    python scripts/eval/collect_results.py --include-mock

产出
----
::

    experiments/summary/
    ├── ablation_table.md          可直接贴进 README 的 markdown 表
    ├── ablation_table.json        机器可读的对比数据（含效应量）
    ├── ablation_summary.txt       控制台友好的纯文本表
    └── （图表写到 docs/figures/）

关键设计：mock 与真实结果严格隔离
--------------------------------
``eval_report.json`` 是真实结果，``eval_report.mock.json`` 是 mock 结果。
默认只读前者。要纳入 mock 必须显式加 ``--include-mock``，
且输出文件会带 ``.mock`` 后缀、表格标题会标注 SIMULATED。

原因：mock 的指标是我们**喂进去的假设**。如果它们混进正式报告，
整个项目就变成了「自己写死结论再自己验证」。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import CFG  # noqa: E402
from src.eval.metrics import compare_reports, cohens_h, required_sample_size  # noqa: E402

#: 展示顺序（也是报告里的行顺序）。
#: ``baseline_airline`` 是基线的**规范目录名**（与 configs/eval/eval_baseline.yaml
#: 的 output.dir 一致）。``baseline`` 作为别名保留，兼容早期产物。
DEFAULT_ORDER: tuple[str, ...] = (
    "baseline_airline",
    "baseline",
    "sft_only",
    "dpo_format",
    "dpo_recovery",
    "dpo_joint",
)

#: 规范实验名集合。自动发现模式下**只接受这些名字**，避免
#: ``baseline`` 与 ``baseline_airline`` 这类同一实验的两种写法被当成两个实验，
#: 在对比表里各占一行（会让人误以为跑了 6 组而非 5 组）。
CANONICAL_EXPERIMENTS: tuple[str, ...] = (
    "baseline_airline",
    "sft_only",
    "dpo_format",
    "dpo_recovery",
    "dpo_joint",
    "dpo_tool",
    "dpo_all",
    "dpo_joint_param",
    "dpo_tool_param",
)

#: 中文标签
LABELS_ZH: dict[str, str] = {
    "baseline": "基线（未训练）",
    "sft_only": "对照：仅 SFT",
    "dpo_format": "Exp 1：格式约束偏好对",
    "dpo_recovery": "Exp 2：错误恢复偏好对",
    "dpo_joint": "Exp 3：联合方案",
    "dpo_tool": "扩展：工具选择偏好对",
    "dpo_all": "扩展：全部四类偏好对",
    "dpo_joint_param": "扩展：格式+恢复+参数",
}

#: 对比表里展示的指标
METRIC_ROWS: tuple[tuple[str, str], ...] = (
    ("accuracy", "单次准确率"),
    ("pass_at_k", "pass@k"),
    ("pass_hat_k", "pass^k"),
    ("delta_stability", "Δstability"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="汇总消融实验结果")
    p.add_argument("--experiments", default=None, help="逗号分隔的实验名（默认自动发现）")
    p.add_argument(
        "--baseline",
        default="baseline_airline",
        help="作为基线的实验名（默认 baseline_airline，与 eval_baseline.yaml 的 output.dir 一致）",
    )
    p.add_argument("--out-dir", default="experiments/summary", help="输出目录")
    p.add_argument("--figures-dir", default="docs/figures", help="图表目录")
    p.add_argument("--include-mock", action="store_true", help="把 mock 结果也纳入（会明确标注）")
    p.add_argument("--no-figures", action="store_true", help="不出图")
    return p.parse_args(argv)


def find_reports(
    experiments: list[str] | None, include_mock: bool
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """在 experiments/ 下发现报告文件。

    Returns:
        ``(reports, sources)``：``sources`` 记录每个实验的报告来自哪个文件，
        便于报告中标注数据来源。

    ⚠️ 只靠文件名区分 mock / 真实是**不够的**。这里在读取后再校验
    ``metadata.is_mock``：如果一份名为 ``eval_report.json`` 的文件内容
    自称是 mock（例如误用 ``--mock`` 跑到了默认目录），在没有
    ``--include-mock`` 时会被**拒绝并明确报错**，而不是静默纳入对比表。

    这是防「自欺欺人」的最后一道闸门：对比表里混进合成数据，
    整个项目的结论就失去意义了。
    """
    exp_root = CFG.path("experiments")
    reports: dict[str, dict[str, Any]] = {}
    sources: dict[str, str] = {}
    rejected: list[str] = []

    # 候选目录：experiments/<name>/ 和 experiments/mock/<name>/
    #
    # 自动发现模式下按 CANONICAL_EXPERIMENTS 过滤：
    # 只认规范名，避免同一实验的不同目录名（如 baseline 与 baseline_airline）
    # 各自占一行，让 5 组实验看起来像 6 组。
    search_dirs: list[tuple[str, Path]] = []
    if experiments:
        for name in experiments:
            search_dirs.append((name, exp_root / name))
            search_dirs.append((name, exp_root / "mock" / name))
    else:
        allowed = set(CANONICAL_EXPERIMENTS)
        for root in (exp_root, exp_root / "mock"):
            if not root.exists():
                continue
            for d in sorted(root.iterdir()):
                if not d.is_dir() or d.name in ("summary", "__pycache__"):
                    continue
                if d.name not in allowed:
                    continue
                search_dirs.append((d.name, d))

    for name, d in search_dirs:
        if name in reports:
            continue
        real = d / "eval_report.json"
        mock = d / "eval_report.mock.json"
        chosen: Path | None = None
        if include_mock:
            # mock 结果优先（流水线自检时它们往往是唯一存在的）
            chosen = mock if mock.exists() else (real if real.exists() else None)
        else:
            chosen = real if real.exists() else None
        if chosen is None:
            continue
        try:
            data = json.loads(chosen.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"  警告：{chosen} 解析失败: {exc}", file=sys.stderr)
            continue

        is_mock = bool(data.get("metadata", {}).get("is_mock"))
        if is_mock and not include_mock:
            rejected.append(str(chosen))
            continue

        reports[name] = data
        sources[name] = str(chosen)

    if rejected:
        print("  ⚠️ 以下文件自称是 mock 结果，已拒绝纳入对比表：", file=sys.stderr)
        for r in rejected:
            print(f"      {r}", file=sys.stderr)
        print(
            "     （若确实想纳入，请显式加 --include-mock，"
            "输出文件名与表格标题都会标注 SIMULATED）",
            file=sys.stderr,
        )

    return reports, sources


def build_markdown_table(
    reports: dict[str, dict[str, Any]],
    *,
    baseline: str,
    sources: dict[str, str],
    is_mock: bool,
) -> str:
    """生成可直接贴进 README 的 markdown 对比表。"""
    order = [e for e in DEFAULT_ORDER if e in reports] + [
        e for e in reports if e not in DEFAULT_ORDER
    ]
    k = reports[order[0]].get("primary_k", 4) if order else 4

    lines: list[str] = []
    title = "## 消融实验对比"
    if is_mock:
        title += "（⚠️ SIMULATED — 由 mock 流水线产出，不是真实模型结果）"
    lines.append(title)
    lines.append("")

    # 主表
    header = f"| 实验 | 单次准确率 | pass@{k} | pass^{k} | Δstability | 相对基线(pass^{k}) | 效应量 |"
    lines.append(header)
    lines.append("|---|---|---|---|---|---|---|")

    base_head = reports.get(baseline, {}).get("headline", {})
    for exp in order:
        h = reports[exp].get("headline", {})
        acc = h.get("accuracy", 0.0)
        p_at = h.get("pass_at_k", 0.0)
        p_hat = h.get("pass_hat_k", 0.0)
        ds = h.get("delta_stability", 0.0)
        label = LABELS_ZH.get(exp, exp)

        if exp == baseline:
            rel = "—"
            eff = "—"
        else:
            b = base_head.get("pass_hat_k", 0.0)
            rel_val = (p_hat - b) / b if b else float("nan")
            rel = f"{rel_val:+.1%}" if rel_val == rel_val else "n/a"
            h_val = cohens_h(b, p_hat)
            mag = "可忽略" if abs(h_val) < 0.2 else "小" if abs(h_val) < 0.5 else "中" if abs(h_val) < 0.8 else "大"
            eff = f"{mag} (h={h_val:+.2f})"

        lines.append(
            f"| {label} | {acc:.1%} | {p_at:.1%} | {p_hat:.1%} | {ds:+.1%} | {rel} | {eff} |"
        )

    # 分组明细
    lines.append("")
    lines.append(f"### 分组明细（pass^{k}）")
    lines.append("")
    for group_name, group_key in (("按难度", "by_difficulty"), ("按可见性", "by_visibility")):
        keys: list[str] = []
        for exp in order:
            for gk in reports[exp].get(group_key, {}):
                if gk not in keys:
                    keys.append(gk)
        if not keys:
            continue
        lines.append(f"**{group_name}**")
        lines.append("")
        lines.append("| 实验 | " + " | ".join(str(x) for x in keys) + " |")
        lines.append("|---" * (len(keys) + 1) + "|")
        for exp in order:
            grp = reports[exp].get(group_key, {})
            cells = []
            for gk in keys:
                v = grp.get(gk, {}).get(f"pass^{k}")
                cells.append(f"{v:.1%}" if isinstance(v, (int, float)) else "—")
            lines.append(f"| {LABELS_ZH.get(exp, exp)} | " + " | ".join(cells) + " |")
        lines.append("")

    # 数据来源
    lines.append("### 数据来源")
    lines.append("")
    for exp in order:
        lines.append(f"- `{exp}`: `{sources.get(exp, 'n/a')}`")
    lines.append("")
    return "\n".join(lines)


def build_plain_table(reports: dict[str, dict[str, Any]], baseline: str) -> str:
    """控制台友好的纯文本表。"""
    order = [e for e in DEFAULT_ORDER if e in reports] + [e for e in reports if e not in DEFAULT_ORDER]
    if not order:
        return "（没有可汇总的结果）"
    k = reports[order[0]].get("primary_k", 4)

    head = f"{'实验':<22}{'准确率':>9}{'pass@' + str(k):>10}{'pass^' + str(k):>10}{'Δstab':>9}"
    out = [head, "-" * len(head) + "------"]
    for exp in order:
        h = reports[exp].get("headline", {})
        out.append(
            f"{LABELS_ZH.get(exp, exp):<22}"
            f"{h.get('accuracy', 0):>9.1%}{h.get('pass_at_k', 0):>10.1%}"
            f"{h.get('pass_hat_k', 0):>10.1%}{h.get('delta_stability', 0):>+9.1%}"
        )
    return "\n".join(out)


def write_statistical_notes(reports: dict[str, dict[str, Any]], baseline: str) -> list[str]:
    """生成统计显著性说明 —— 这部分经常被省略，但没有它「提升」是不可信的。"""
    notes: list[str] = []
    order = [e for e in DEFAULT_ORDER if e in reports]
    if len(order) < 2:
        return notes

    # 用 pass^k 的置信区间是否重叠做粗略判断
    b = reports.get(baseline, {})
    b_ci = b.get("metrics", {}).get("pass_hat_k", {}).get("ci95", [0, 1])
    k = b.get("primary_k", 4)

    notes.append("### 统计说明")
    notes.append("")
    notes.append(
        "置信区间由 **任务级 bootstrap**（1000 次重采样任务）得到，"
        "而不是按采样样本重采样 —— 同一任务的多次采样高度相关，"
        "按样本算会严重低估区间宽度。"
    )
    notes.append("")
    for exp in order:
        if exp == baseline:
            continue
        e = reports[exp]
        e_ci = e.get("metrics", {}).get("pass_hat_k", {}).get("ci95", [0, 1])
        overlap = not (e_ci[1] < b_ci[0] or e_ci[0] > b_ci[1])
        verdict = (
            "置信区间与基线重叠 → **不能断言有真实提升**，需增加任务数或采样次数"
            if overlap
            else "置信区间与基线不重叠 → 有统计支持"
        )
        notes.append(
            f"- `{exp}`: pass^{k} = {e.get('headline', {}).get('pass_hat_k', 0):.1%} "
            f"(95%CI [{e_ci[0]:.1%}, {e_ci[1]:.1%}])，"
            f"基线 {b.get('headline', {}).get('pass_hat_k', 0):.1%} "
            f"(95%CI [{b_ci[0]:.1%}, {b_ci[1]:.1%}])；{verdict}"
        )

    # 样本量建议
    notes.append("")
    delta = reports.get("dpo_joint", reports.get(order[-1], {})).get("headline", {}).get("pass_hat_k", 0) - \
        b.get("headline", {}).get("pass_hat_k", 0)
    if delta > 0:
        need = required_sample_size(delta)
        n_tasks = b.get("num_tasks", 0)
        notes.append(
            f"以观测到的 pass^{k} 差值 {delta:+.1%} 计算，"
            f"要在 80% 功效下检出该效应，每组约需 **{need}** 个任务"
            f"（当前 {n_tasks} 个）。"
        )
        if need > n_tasks:
            notes.append(
                f"→ 当前任务数偏少，观测到的提升可能不稳定。"
                f"建议扩充任务集或提高采样次数（这会同步提高 pass^k 的估计精度）。"
            )
    notes.append("")
    return notes


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    experiments = [e.strip() for e in args.experiments.split(",")] if args.experiments else None

    print("=" * 78)
    print("汇总消融实验结果")
    print("=" * 78)

    reports, sources = find_reports(experiments, args.include_mock)
    if not reports:
        print("没有找到任何评测报告。", file=sys.stderr)
        print(
            "  真实结果需先运行: python scripts/eval/run_baseline.py --config configs/eval/eval_baseline.yaml",
            file=sys.stderr,
        )
        print(
            "  流水线自检: python scripts/train/mock/run_all_mock.py --include-mock",
            file=sys.stderr,
        )
        return 1

    # --- 基线别名解析 ---
    # 基线的规范目录名是 ``baseline_airline``，但早期产物 / 手工命令可能
    # 用 ``baseline``。这里做一次别名回退，避免因为目录名不同就让
    # 「相对提升」整列变成 n/a —— 那会让人误以为实验失败。
    baseline_key = args.baseline
    if baseline_key not in reports:
        for alias in ("baseline_airline", "baseline"):
            if alias in reports:
                if baseline_key != alias:
                    print(
                        f"  ℹ️ 基线 {baseline_key!r} 不在结果里；"
                        f"自动改用 {alias!r} 作为基线。"
                    )
                baseline_key = alias
                break

    # 判断这批结果是不是 mock
    is_mock = all(r.get("metadata", {}).get("is_mock") for r in reports.values())
    mixed = any(r.get("metadata", {}).get("is_mock") for r in reports.values()) and not is_mock

    print(f"  发现 {len(reports)} 个实验: {', '.join(sorted(reports))}")
    if is_mock:
        print("  ⚠️ 全部是 MOCK 结果 —— 仅用于验证流水线，不能作为项目结论")
    elif mixed:
        print("  ⚠️ 混有 MOCK 与真实结果 —— 对比表会失去意义，请检查数据来源")
    if baseline_key not in reports:
        print(f"  ⚠️ 基线实验 {baseline_key!r} 不在结果里，相对提升无法计算")
    print()

    print(build_plain_table(reports, baseline_key))
    print()

    # --- 对比数据 ---
    comparisons = compare_reports(reports, baseline_key=baseline_key)
    if comparisons:
        print("相对基线的变化:")
        for c in comparisons:
            if c.metric in ("pass_hat_k", "accuracy"):
                print("  " + c.pretty())
        print()

    # --- 落盘 ---
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = CFG.path() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".mock" if is_mock else ""

    md = build_markdown_table(reports, baseline=baseline_key, sources=sources, is_mock=is_mock)
    notes = write_statistical_notes(reports, baseline_key)
    md_full = md + "\n" + "\n".join(notes)
    (out_dir / f"ablation_table{suffix}.md").write_text(md_full, encoding="utf-8")
    (out_dir / f"ablation_summary{suffix}.txt").write_text(
        build_plain_table(reports, baseline_key)
        + "\n\n"
        + "\n".join(c.pretty() for c in comparisons),
        encoding="utf-8",
    )
    (out_dir / f"ablation_table{suffix}.json").write_text(
        json.dumps(
            {
                "is_mock": is_mock,
                "baseline": baseline_key,
                "experiments": sorted(reports),
                "sources": sources,
                "headlines": {e: r.get("headline", {}) for e, r in reports.items()},
                "by_difficulty": {e: r.get("by_difficulty", {}) for e, r in reports.items()},
                "by_visibility": {e: r.get("by_visibility", {}) for e, r in reports.items()},
                "comparisons": [c.to_dict() for c in comparisons],
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )

    # --- 出图 ---
    if not args.no_figures:
        figures_dir = Path(args.figures_dir)
        if not figures_dir.is_absolute():
            figures_dir = CFG.path() / figures_dir
        from src.utils.plots import plot_ablation_comparison, plot_pass_k_curve

        made: list[str] = []
        p = plot_ablation_comparison(
            reports, figures_dir / f"ablation_comparison{suffix.replace('.', '_')}.png"
        )
        if p:
            made.append(p)
        p = plot_pass_k_curve(reports, figures_dir / f"pass_k_curve{suffix.replace('.', '_')}.png")
        if p:
            made.append(p)
        if made:
            print("图表:")
            for m in made:
                print(f"  {m}")
        else:
            print("（未生成图：matplotlib 不可用）")

    print()
    print("=" * 78)
    print(f"  markdown 表 : {out_dir / f'ablation_table{suffix}.md'}")
    print(f"  JSON 数据   : {out_dir / f'ablation_table{suffix}.json'}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
