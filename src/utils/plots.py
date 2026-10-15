"""出图：把评测与训练结果画成报告里能直接引用的 PNG。

设计原则
--------
1. **matplotlib 可选**：没有装 matplotlib 时，所有函数返回 ``None``
   而不是抛异常。画图是报告的辅助，不该阻断整条流水线 ——
   尤其在只有 numpy 的干净环境里（本项目刻意支持这种环境）。
2. **中文字体兜底**：默认字体画不出中文（会显示成方框）。
   这里按平台依次尝试常见中文字体，找不到就自动退回英文标签，
   保证图至少可读。
3. **一图一信息**：不做「把所有指标塞进一张图」的复合图。
   报告里引用方便，读者也不用猜哪条线是什么。

产出的图
--------
=========================  ==========================================
文件                        内容
=========================  ==========================================
``error_distribution.png``  基线错误类型分布（饼图）
``success_by_difficulty.png`` 按难度的 accuracy / pass^k（分组柱状）
``ablation_comparison.png``  各实验主指标对比（分组柱状）
``training_curves.png``      训练 loss 与验证准确率双轴曲线
``pass_k_curve.png``         不同 k 下的 pass@k 与 pass^k 曲线
=========================  ==========================================
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

#: matplotlib 可用性探测（模块级，只做一次）
_MPL_AVAILABLE: bool | None = None
_MPL_FONT: str | None = None

#: 各平台常见中文字体，按优先级
_CJK_FONTS: tuple[str, ...] = (
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "WenQuanYi Zen Hei",
    "PingFang SC",
    "Heiti SC",
    "Arial Unicode MS",
)

#: 错误类别 → 固定颜色。固定颜色很重要：不同图里同一个类别必须同色，
#: 否则读者会把「颜色变了」误读成「性质变了」。
CATEGORY_COLORS: dict[str, str] = {
    "SUCCESS": "#4C9F70",
    "FORMAT_ERROR": "#D9534F",
    "HALLUCINATION": "#F0AD4E",
    "WRONG_TOOL": "#5BC0DE",
    "PARAM_ERROR": "#9B59B6",
    "REFUSED": "#7F8C8D",
    "UNKNOWN": "#BDC3C7",
}

EXPERIMENT_COLORS: dict[str, str] = {
    "baseline": "#7F8C8D",
    "sft_only": "#5BC0DE",
    "dpo_format": "#F0AD4E",
    "dpo_recovery": "#9B59B6",
    "dpo_joint": "#4C9F70",
}

#: 实验的英文短名（用于图上标签，避免中文字体缺失时全变方框）
EXPERIMENT_SHORT: dict[str, str] = {
    "baseline": "Baseline",
    "sft_only": "SFT only",
    "dpo_format": "DPO-format",
    "dpo_recovery": "DPO-recovery",
    "dpo_joint": "DPO-joint",
}


def matplotlib_available() -> bool:
    global _MPL_AVAILABLE
    if _MPL_AVAILABLE is None:
        try:
            import matplotlib  # noqa: F401

            _MPL_AVAILABLE = True
        except ImportError:
            _MPL_AVAILABLE = False
    return _MPL_AVAILABLE


def _use_cjk_font() -> bool:
    """尝试启用中文字体，返回是否成功。

    >>> # 结果依环境而定，这里只验证不抛异常
    >>> isinstance(_use_cjk_font(), bool)
    True
    """
    global _MPL_FONT
    if _MPL_FONT is not None:
        return _MPL_FONT != ""
    try:
        import matplotlib.font_manager as fm
        import matplotlib.pyplot as plt

        available = {f.name for f in fm.fontManager.ttflist}
        for name in _CJK_FONTS:
            if name in available:
                plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
                # 负号显示为方框是常见问题，显式关掉 Unicode 负号
                plt.rcParams["axes.unicode_minus"] = False
                _MPL_FONT = name
                return True
        _MPL_FONT = ""
        return False
    except Exception:  # pragma: no cover
        _MPL_FONT = ""
        return False


def _setup():
    """初始化 matplotlib，返回 (plt, has_cjk) 或 (None, False)。"""
    if not matplotlib_available():
        return None, False
    import matplotlib

    matplotlib.use("Agg")  # 无显示环境必须用 Agg
    import matplotlib.pyplot as plt

    has_cjk = _use_cjk_font()
    return plt, has_cjk


def _label(zh: str, en: str, has_cjk: bool) -> str:
    """中文字体缺失时退回英文标签。"""
    return zh if has_cjk else en


def _infer_is_mock(*objs: Any) -> bool:
    """判断这组出图输入是否来自模拟数据。

    判定依据（任一命中即为模拟）：

    - 顶层 ``is_mock is True``
    - ``label`` 以 ``mock`` 开头
    - ``metadata.is_mock is True``
    - 值为 dict 的 dict（例如 ``reports = {实验名: 报告}``）→ **递归查每个值**

    最后一条是必需的：``make_all_figures`` 收到的 ``reports`` 是
    ``{name: report}`` 映射，模拟标记在被嵌套的那一层里。只查顶层会漏判，
    表现就是「其他图都有 SIMULATED 横幅，消融对比图没有」。

    >>> _infer_is_mock({"is_mock": True})
    True
    >>> _infer_is_mock({"label": "mock_dpo_joint"})
    True
    >>> _infer_is_mock({"label": "baseline_airline"})
    False
    >>> _infer_is_mock({"metadata": {"is_mock": True}})
    True
    >>> _infer_is_mock({"dpo_joint": {"metadata": {"is_mock": True}}})
    True
    >>> _infer_is_mock(None, {"label": "baseline_airline"}, {"label": "dpo_joint"})
    False
    """
    for o in objs:
        if isinstance(o, dict):
            if o.get("is_mock") is True:
                return True
            if str(o.get("label", "")).lower().startswith("mock"):
                return True
            if o.get("metadata", {}).get("is_mock") is True:
                return True
            # 嵌套映射 {name: report}
            for v in o.values():
                if isinstance(v, dict):
                    if v.get("is_mock") is True:
                        return True
                    if str(v.get("label", "")).lower().startswith("mock"):
                        return True
                    if v.get("metadata", {}).get("is_mock") is True:
                        return True
    return False


def _watermark(fig: Any, is_mock: bool) -> None:
    """给图叠加 SIMULATED 横幅（模拟数据时）。

    ⚠️ 为什么必须画进图里，而不是只写在文档里：

    图一旦被单独截图、复制进 PPT、或贴到别处，就脱离了正文的免责说明。
    本项目曾有 5 张 PNG（matplotlib 路径）完全没有模拟标记，
    而 SVG 路径的同类图都有 —— 同一个仓库里两套行为不一致，
    读者无法从图本身判断数据真伪。

    现在两条路径一致：``is_mock=True`` 时都在右上角叠加横幅。
    """
    if not is_mock:
        return
    fig.text(
        0.995,
        0.985,
        "SIMULATED — 非真实结果",
        ha="right",
        va="top",
        fontsize=10,
        weight="bold",
        color="#8A6100",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFF3CD", edgecolor="#E0A800", linewidth=1.4),
    )


def _save(fig: Any, out_path: str | Path) -> str:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=140, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return str(p)


# ============================================================================
# 图 1：错误类型分布
# ============================================================================


def plot_error_distribution(
    distribution: dict[str, Any],
    out_path: str | Path,
    *,
    title: str = "基线错误类型分布",
    label: str = "baseline",
    is_mock: bool = False,
) -> str | None:
    """画错误分布饼图 + 子类型柱状图（一张画布两个子图）。

    Args:
        distribution: ``ErrorDistribution.to_dict()`` 的输出。
        out_path: 输出 png 路径。
        title: 图标题。
        label: 实验名，会写进图注（避免把某实验的分布误当成全局结论）。
    """
    plt, has_cjk = _setup()
    if plt is None:
        return None

    counts: dict[str, int] = distribution.get("counts", {})
    # 只画失败类型 + 成功，按固定顺序
    order = ["FORMAT_ERROR", "HALLUCINATION", "WRONG_TOOL", "PARAM_ERROR", "REFUSED", "UNKNOWN", "SUCCESS"]
    labels_en = {
        "FORMAT_ERROR": "Format error",
        "HALLUCINATION": "Hallucinated tool",
        "WRONG_TOOL": "Wrong tool",
        "PARAM_ERROR": "Param error",
        "REFUSED": "Refused",
        "UNKNOWN": "Unknown",
        "SUCCESS": "Success",
    }
    labels_zh = {
        "FORMAT_ERROR": "格式错误",
        "HALLUCINATION": "工具幻觉",
        "WRONG_TOOL": "工具选错",
        "PARAM_ERROR": "参数错误",
        "REFUSED": "拒绝作答",
        "UNKNOWN": "未知",
        "SUCCESS": "成功",
    }

    cats = [c for c in order if counts.get(c, 0) > 0]
    vals = [counts[c] for c in cats]
    colors = [CATEGORY_COLORS.get(c, "#BDC3C7") for c in cats]
    names = [_label(labels_zh[c], labels_en[c], has_cjk) for c in cats]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))

    # --- 左：饼图 ---
    ax = axes[0]
    total = sum(vals) or 1
    wedges, _, autotexts = ax.pie(
        vals,
        colors=colors,
        autopct=lambda p: f"{p:.1f}%" if p >= 3 else "",
        startangle=105,
        textprops={"fontsize": 9, "color": "white", "weight": "bold"},
        wedgeprops={"edgecolor": "white", "linewidth": 1.2},
    )
    ax.legend(
        wedges,
        [f"{n}  ({v}, {v / total:.1%})" for n, v in zip(names, vals)],
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        frameon=False,
    )
    ax.set_title(
        _label(f"{title}（{label}）", f"{title} ({label})", has_cjk), fontsize=12, weight="bold"
    )

    # --- 右：子类型柱状图 ---
    ax2 = axes[1]
    # 优先用 failure_subtype_counts（已剔除成功样本）
    subs: dict[str, int] = (
        distribution.get("failure_subtype_counts")
        or {
            k: v
            for k, v in distribution.get("subtype_counts", {}).items()
            if k not in ("-", "clarify_instead")
        }
    )
    subs_zh: dict[str, int] = distribution.get("subtype_counts_zh", {})
    top = sorted(subs.items(), key=lambda kv: -kv[1])[:8]
    if top:
        names2 = [k for k, _ in top]
        if has_cjk:
            # subs_zh 的键已经是中文，值集合与 subs 一致；
            # 按出现次数把中文名对齐回原始键
            zh_by_count: dict[int, list[str]] = {}
            for zh_name, cnt in subs_zh.items():
                zh_by_count.setdefault(cnt, []).append(zh_name)
            display = []
            for k in names2:
                cnt = subs[k]
                pool = zh_by_count.get(cnt, [])
                display.append(pool.pop(0) if pool else k)
        else:
            display = names2
        vals2 = [v for _, v in top]
        ypos = range(len(display))
        ax2.barh(list(ypos), vals2, color="#D9534F", alpha=0.82)
        ax2.set_yticks(list(ypos))
        ax2.set_yticklabels(display, fontsize=9)
        ax2.invert_yaxis()
        ax2.set_xlabel(_label("样本数", "count", has_cjk), fontsize=10)
        ax2.set_title(
            _label("错误子类型 Top 8", "Top-8 error subtypes", has_cjk), fontsize=12, weight="bold"
        )
        for i, v in enumerate(vals2):
            ax2.text(v + max(vals2) * 0.01, i, str(v), va="center", fontsize=8)
    else:
        ax2.text(0.5, 0.5, "no subtype data", ha="center", va="center")

    fig.suptitle(
        _label(
            f"样本总数 {distribution.get('total', 0)}，"
            f"成功 {distribution.get('success_count', 0)}（{distribution.get('success_rate', 0):.1%}）",
            f"n={distribution.get('total', 0)}, success={distribution.get('success_count', 0)} "
            f"({distribution.get('success_rate', 0):.1%})",
            has_cjk,
        ),
        fontsize=10,
        y=1.02,
    )
    _watermark(fig, is_mock)
    return _save(fig, out_path)


def _is_ascii(s: str) -> bool:
    try:
        s.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


# ============================================================================
# 图 2：按难度/可见性分组
# ============================================================================


def plot_success_by_group(
    report: dict[str, Any],
    out_path: str | Path,
    *,
    group_key: str = "by_difficulty",
    primary_k: int = 4,
    title: str = "按难度分组的成功率",
    is_mock: bool = False,
) -> str | None:
    """分组柱状图：每组画 accuracy 与 pass^k 两根柱。

    accuracy 高但 pass^k 低 = 「会做但不稳」，这是本图要一眼看出来的信息。
    """
    plt, has_cjk = _setup()
    if plt is None:
        return None

    groups: dict[str, dict[str, Any]] = report.get(group_key, {})
    if not groups:
        return None

    names = list(groups)
    accs = [groups[n].get("accuracy", 0.0) for n in names]
    pks = [groups[n].get(f"pass^{primary_k}", 0.0) for n in names]

    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(names) + 3), 5))
    x = range(len(names))
    w = 0.36
    b1 = ax.bar([i - w / 2 for i in x], accs, w, label=_label("单次准确率", "accuracy", has_cjk), color="#5B8FF9")
    b2 = ax.bar(
        [i + w / 2 for i in x], pks, w, label=f"pass^{primary_k}", color="#4C9F70"
    )
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 0.012, f"{h:.1%}", ha="center", fontsize=8.5)

    ax.set_xticks(list(x))
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(_label("比率", "rate", has_cjk), fontsize=10)
    ax.set_title(
        _label(f"{title}（{report.get('label', '')}）", f"{title} ({report.get('label', '')})", has_cjk),
        fontsize=12,
        weight="bold",
    )
    ax.legend(fontsize=9, frameon=False)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    _watermark(fig, is_mock)
    return _save(fig, out_path)


# ============================================================================
# 图 3：消融对比
# ============================================================================


def plot_ablation_comparison(
    reports: dict[str, dict[str, Any]],
    out_path: str | Path,
    *,
    metrics: Sequence[str] = ("accuracy", "pass_hat_k"),
    metric_labels: dict[str, str] | None = None,
    experiments: Sequence[str] | None = None,
    title: str = "消融实验对比",
    is_mock: bool = False,
) -> str | None:
    """各实验在不同指标上的分组柱状图。

    Args:
        reports: ``{实验名: report dict}``。
        metrics: ``headline`` 里要画的指标键。
        experiments: 指定顺序；None 表示用 ``EXPERIMENT_COLORS`` 的顺序。
    """
    plt, has_cjk = _setup()
    if plt is None:
        return None

    order = list(experiments) if experiments else [k for k in EXPERIMENT_COLORS if k in reports]
    order += [k for k in reports if k not in order]
    if not order:
        return None

    default_labels = {
        "accuracy": ("单次准确率", "accuracy"),
        "pass_at_k": ("pass@k（至少一次对）", "pass@k"),
        "pass_hat_k": ("pass^k（k 次全对）", "pass^k"),
        "delta_stability": ("Δstability", "Δstability"),
    }
    mlabels = metric_labels or {}

    n_m = len(metrics)
    fig, axes = plt.subplots(1, n_m, figsize=(5.2 * n_m, 5), squeeze=False)

    for mi, mk in enumerate(metrics):
        ax = axes[0][mi]
        vals = [float(reports[e].get("headline", {}).get(mk, 0.0)) for e in order]
        colors = [EXPERIMENT_COLORS.get(e, "#BDC3C7") for e in order]
        bars = ax.bar(range(len(order)), vals, color=colors, width=0.62)
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + (0.015 if v >= 0 else -0.045),
                f"{v:+.1%}" if mk == "delta_stability" else f"{v:.1%}",
                ha="center",
                fontsize=9,
                weight="bold",
            )
        zh, en = mlabels.get(mk, default_labels.get(mk, (mk, mk)))
        ax.set_title(_label(zh, en, has_cjk), fontsize=12, weight="bold")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([EXPERIMENT_SHORT.get(e, e) for e in order], rotation=18, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)
        if mk != "delta_stability":
            ax.set_ylim(0, 1.0)

    fig.suptitle(title, fontsize=13, weight="bold", y=1.02)
    _watermark(fig, is_mock)
    return _save(fig, out_path)


# ============================================================================
# 图 4：训练曲线
# ============================================================================


def plot_training_curves(
    curves: dict[str, Any],
    out_path: str | Path,
    *,
    title: str = "DPO 训练曲线",
    is_mock: bool = False,
) -> str | None:
    """双轴训练曲线：左轴 loss，右轴验证准确率。

    Args:
        curves: ``training_curves.json`` 的内容。
    """
    plt, has_cjk = _setup()
    if plt is None:
        return None

    train = curves.get("train", [])
    evals = curves.get("eval", [])
    if not train:
        return None

    fig, ax1 = plt.subplots(figsize=(9.5, 5.2))
    steps = [r["step"] for r in train]
    losses = [r["loss"] for r in train]
    ax1.plot(steps, losses, color="#D9534F", lw=1.5, alpha=0.85, label=_label("训练 loss", "train loss", has_cjk))
    ax1.set_xlabel(_label("训练步数", "step", has_cjk), fontsize=10)
    ax1.set_ylabel(_label("DPO loss", "DPO loss", has_cjk), color="#D9534F", fontsize=10)
    ax1.tick_params(axis="y", labelcolor="#D9534F")
    ax1.axhline(0.6931, color="#7F8C8D", ls=":", lw=1.2)
    ax1.text(
        steps[0],
        0.6931,
        _label("  log 2（π_θ = π_ref 的起点）", "  log 2 (start)", has_cjk),
        fontsize=8,
        color="#7F8C8D",
        va="bottom",
    )

    if evals:
        ax2 = ax1.twinx()
        est = [e["step"] for e in evals]
        accs = [e.get("eval_accuracy", 0.0) for e in evals]
        fmts = [e.get("eval_format_accuracy", 0.0) for e in evals]

        def _smooth(xs: list[float], w: int = 3) -> list[float]:
            """三点滑动平均。

            验证准确率是「用当前错误率假设做 N 个任务采样」的蒙特卡洛估计，
            逐点抖动是采样噪声而非真实波动。画原始点 + 平滑线，
            让读者既能看见噪声大小、又能看清趋势。
            """
            if len(xs) < w:
                return list(xs)
            out: list[float] = []
            for i in range(len(xs)):
                lo = max(0, i - w // 2)
                hi = min(len(xs), i + w // 2 + 1)
                out.append(sum(xs[lo:hi]) / (hi - lo))
            return out

        # 原始点用淡色，平滑线用实色
        ax2.plot(est, accs, "o", color="#4C9F70", ms=4, alpha=0.35)
        ax2.plot(est, _smooth(accs), "-", color="#4C9F70", lw=1.8, label=_label("验证准确率", "eval accuracy", has_cjk))
        ax2.plot(est, fmts, "s", color="#5B8FF9", ms=3, alpha=0.30)
        ax2.plot(
            est, _smooth(fmts), "--", color="#5B8FF9", lw=1.5, label=_label("格式正确率", "format acc", has_cjk)
        )
        ax2.set_ylabel(_label("比率", "rate", has_cjk), color="#4C9F70", fontsize=10)
        ax2.set_ylim(0, 1.0)
        ax2.tick_params(axis="y", labelcolor="#4C9F70")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="center right", frameon=False)
    else:
        ax1.legend(fontsize=9, frameon=False)

    is_mock = is_mock or bool(curves.get("is_mock"))
    suffix = "（SIMULATED — 非真实训练结果）" if is_mock else ""
    suffix_en = " (SIMULATED)" if is_mock and not has_cjk else ""
    ax1.set_title(
        _label(
            f"{title}：{curves.get('experiment', '')}{suffix}",
            f"{title}: {curves.get('experiment', '')}{suffix_en}",
            has_cjk,
        ),
        fontsize=12,
        weight="bold",
    )
    ax1.grid(alpha=0.22, linestyle="--")
    _watermark(fig, is_mock)
    return _save(fig, out_path)


# ============================================================================
# 图 5：pass@k / pass^k 曲线
# ============================================================================


def plot_pass_k_curve(
    reports: dict[str, dict[str, Any]],
    out_path: str | Path,
    *,
    k_values: Sequence[int] | None = None,
    title: str = "pass@k 与 pass^k",
    is_mock: bool = False,
) -> str | None:
    """每个实验一条 pass@k 曲线、一条 pass^k 曲线。

    两条线之间的**垂直距离**就是「稳定性缺口」：
    距离越大，说明模型「会做但做不稳」。
    DPO 的目标就是把实线抬上去、把缺口压窄。

    Args:
        k_values: 要画的 k。**None（默认）表示从报告里自适应推断**。

    ⚠️ 为什么默认自适应而不是硬编码 ``(1, 2, 4)``：

    报告的 k 值由**采样次数**决定 —— ``num_samples_per_task=2`` 时只有
    ``pass@1/pass@2``，**不存在** ``pass@4``。早期版本硬编码
    ``k_values=(1,2,4)`` 且要求每个 k 都存在才画线，于是：

    - 用 ``--num-samples 2`` 跑的流水线 → 一个 k 都凑不齐 → 一条线都不画
    - 函数静默 ``return None`` → 图凭空少一张，**日志里没有任何提示**

    表现就是「应该有 7 张图，实际只出 6 张」，且查不出原因。
    改成自适应后，有几个 k 就画几个 k，并把跳过原因打印出来。
    """
    plt, has_cjk = _setup()
    if plt is None:
        return None

    # --- 自适应推断可用的 k ---
    if k_values is None:
        available: set[int] = set()
        for rep in reports.values():
            for key in rep.get("metrics", {}):
                if key.startswith("pass@") and not key.startswith("pass@k"):
                    try:
                        available.add(int(key.split("@", 1)[1]))
                    except ValueError:
                        continue
        ks = sorted(available)
    else:
        ks = list(k_values)

    if not ks:
        print("  ⚠️ pass@k 曲线跳过：报告里没有任何 pass@k 指标（检查 metrics 字段）")
        return None

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    any_line = False
    skipped: list[str] = []

    for exp, rep in reports.items():
        metrics = rep.get("metrics", {})
        p_at = [metrics.get(f"pass@{k}", {}).get("value") for k in ks]
        p_hat = [metrics.get(f"pass^{k}", {}).get("value") for k in ks]
        color = EXPERIMENT_COLORS.get(exp, "#BDC3C7")
        drew = False
        if all(v is not None for v in p_at):
            ax.plot(
                ks, p_at, "o-", color=color, lw=1.9, ms=5,
                label=f"{EXPERIMENT_SHORT.get(exp, exp)} pass@k",
            )
            any_line = True
            drew = True
        if all(v is not None for v in p_hat):
            ax.plot(ks, p_hat, "s--", color=color, lw=1.5, ms=4, alpha=0.75, label=f"{EXPERIMENT_SHORT.get(exp, exp)} pass^k")
            any_line = True
            drew = True
        if not drew:
            missing = [k for k in ks if f"pass@{k}" not in metrics]
            skipped.append(f"{exp}(缺 k={missing})")

    if skipped:
        print(
            f"  ⚠️ pass@k 曲线：以下实验因缺少对应 k 的指标被跳过 → {', '.join(skipped)}\n"
            f"     常见原因：采样次数少于 k（pass^k 要求 k <= num_samples_per_task）"
        )

    if not any_line:
        plt.close(fig)
        print(f"  ⚠️ pass@k 曲线未生成：没有任何实验同时具备 k={ks} 的指标")
        return None

    ax.set_xticks(ks)
    ax.set_xlabel("k", fontsize=10)
    ax.set_ylabel(_label("比率", "rate", has_cjk), fontsize=10)
    ax.set_ylim(0, 1.02)
    ax.set_title(_label(title, title, has_cjk), fontsize=12, weight="bold")
    ax.legend(fontsize=8.5, ncol=2, frameon=False)
    ax.grid(alpha=0.22, linestyle="--")
    ax.set_axisbelow(True)
    _watermark(fig, is_mock)
    return _save(fig, out_path)


# ============================================================================
# 批量出图
# ============================================================================


def make_all_figures(
    out_dir: str | Path,
    *,
    baseline_report: dict[str, Any] | None = None,
    baseline_distribution: dict[str, Any] | None = None,
    reports: dict[str, dict[str, Any]] | None = None,
    curves: dict[str, Any] | None = None,
    prefer_svg: bool = False,
) -> dict[str, str]:
    """一次生成所有可用的图，返回 ``{图名: 路径}``。

    两条出图路径：

    - **matplotlib 可用** → 生成 PNG（通用，任何地方都能看）
    - **matplotlib 不可用** → 自动退回 `src/utils/svg_plots.py` 的纯标准库 SVG
      （矢量、更清晰，且不需要任何第三方依赖）

    ⚠️ **两条路径的图名集合不对称，数量本来就会不同（常见 6 vs 7）：**

    ========================================  ==========  ==========
    图                                          matplotlib  SVG
    ========================================  ==========  ==========
    ``error_distribution``                      PNG         SVG
    ``error_distribution``（Mermaid 源码）        —          ✅
    ``success_by_difficulty``                   PNG         SVG
    ``success_by_visibility``                   PNG         SVG
    ``ablation_comparison``                     PNG         SVG
    ``ablation_comparison``（Mermaid 源码）       —          ✅
    ``pass_k_curve``                            ✅          —（见下）
    ``training_curves``                         PNG         SVG
    ========================================  ==========  ==========

    SVG 分支多出两个 ``*.mermaid.md``（可直接贴进 GitHub 渲染），
    少一张独立的 pass@k 曲线图（该分支没实现 SVG 版）。
    所以**不要用数量判断是否出图失败** —— 请直接看返回的键名，
    调用方应当把它们逐条打印出来。

    任何一张图失败都不影响其他图 —— 出图不该阻断流水线。
    """
    out = Path(out_dir)
    made: dict[str, str] = {}

    use_mpl = matplotlib_available() and not prefer_svg

    if not use_mpl:
        # ---- 纯 SVG 路径 ----
        try:
            from src.utils.svg_plots import make_all_svg

            made = make_all_svg(
                out,
                baseline_report=baseline_report,
                baseline_distribution=baseline_distribution,
                reports=reports,
                curves=curves,
            )
            made["_engine"] = "svg（matplotlib 不可用，已自动降级）"
            return made
        except Exception as exc:  # pragma: no cover
            import sys

            print(f"  ⚠️ SVG 出图失败: {type(exc).__name__}: {exc}", file=sys.stderr)
            return {}

    # ---- matplotlib 路径 ----
    # 与 SVG 分支用同一套判断：只要输入里任何一处表明是模拟数据，
    # 所有图都打 SIMULATED 横幅（宁可给真数据多打一个标记，
    # 也不能让模拟图漏标）。
    is_mock = _infer_is_mock(baseline_distribution, baseline_report, reports, curves)
    if baseline_distribution:
        p = plot_error_distribution(
            baseline_distribution, out / "error_distribution.png", is_mock=is_mock
        )
        if p:
            made["error_distribution"] = p
    if baseline_report:
        p = plot_success_by_group(
            baseline_report,
            out / "success_by_difficulty.png",
            group_key="by_difficulty",
            is_mock=is_mock,
        )
        if p:
            made["success_by_difficulty"] = p
        p = plot_success_by_group(
            baseline_report,
            out / "success_by_visibility.png",
            group_key="by_visibility",
            title="按可见性分组的成功率（unseen = 泛化）",
            is_mock=is_mock,
        )
        if p:
            made["success_by_visibility"] = p
    if reports:
        p = plot_ablation_comparison(reports, out / "ablation_comparison.png", is_mock=is_mock)
        if p:
            made["ablation_comparison"] = p
        # k 值不传，让函数从报告里自适应推断（见该函数的文档说明）
        p = plot_pass_k_curve(reports, out / "pass_k_curve.png", is_mock=is_mock)
        if p:
            made["pass_k_curve"] = p
        else:
            print(
                "  ℹ️ pass_k_curve 未生成（通常是报告的 k 值不足）。"
                "缺这张图不影响其他图与结论。"
            )
    if curves:
        p = plot_training_curves(curves, out / "training_curves.png")
        if p:
            made["training_curves"] = p

    # 记录本次实际走的是哪条引擎，便于解释两次运行的图数差异
    made["_engine"] = f"matplotlib（中文字体: {_MPL_FONT or '未找到，已退回英文标签'}）"
    return made


if __name__ == "__main__":  # pragma: no cover
    print("matplotlib 可用:", matplotlib_available())
    print("中文字体:", _use_cjk_font() and _MPL_FONT or "未找到（会退回英文标签）")
