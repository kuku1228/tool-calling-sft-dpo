"""纯标准库出图：matplotlib 缺失时生成 SVG + Mermaid 图。

为什么要做这件事
----------------
本项目的核心设计目标之一是「在一台只有 numpy 的机器上也能跑通全流程」。
而 matplotlib 常常装不上（内网环境、PyPI 不可达、镜像没配好）。

如果没有图上不了报告，那这个目标就打了折扣。所以这里用**纯标准库**
（字符串拼 SVG）生成等价的图形：

- 矢量图（SVG），比 PNG 更清晰，可直接嵌入 markdown/HTML
- Mermaid 源码，可直接贴进 GitHub README（GitHub 原生渲染）

matplotlib 可用时优先用 matplotlib（PNG，更通用）；
不可用时自动退回这里。两条路径产出**同样的信息**。

手写 SVG 的注意事项
-------------------
1. **中文渲染**：SVG 里必须写 ``font-family`` 且不能用 ``text-anchor`` 误配，
   否则中文会挤在一起。这里统一用 ``middle`` 居中 + ``dominant-baseline``。
2. **XML 转义**：标签文本里的 ``&`` ``<`` ``>`` 必须转义，否则 SVG 解析失败。
   中文不需要转义，但用户可能从数据里带进特殊字符。
3. **坐标系统**：SVG 的 y 轴向下，所以数值要翻转（``height - margin - v*h``）。
"""

from __future__ import annotations

import json
import math
from html import escape
from pathlib import Path
from typing import Any, Sequence

#: 与 plots.py 保持一致的配色，否则同一类别在不同图里颜色不同，读者会误读
CATEGORY_COLORS: dict[str, str] = {
    "SUCCESS": "#4C9F70",
    "FORMAT_ERROR": "#D9534F",
    "HALLUCINATION": "#F0AD4E",
    "WRONG_TOOL": "#5BC0DE",
    "PARAM_ERROR": "#9B59B6",
    "REFUSED": "#7F8C8D",
    "UNKNOWN": "#BDC3C7",
}

CATEGORY_LABELS: dict[str, str] = {
    "SUCCESS": "成功 Success",
    "FORMAT_ERROR": "格式错误 Format",
    "HALLUCINATION": "工具幻觉 Hallucination",
    "WRONG_TOOL": "工具选错 Wrong tool",
    "PARAM_ERROR": "参数错误 Param",
    "REFUSED": "拒绝作答 Refused",
    "UNKNOWN": "未知 Unknown",
}

EXPERIMENT_COLORS: dict[str, str] = {
    "baseline": "#7F8C8D",
    "sft_only": "#5BC0DE",
    "dpo_format": "#F0AD4E",
    "dpo_recovery": "#9B59B6",
    "dpo_joint": "#4C9F70",
}

EXPERIMENT_SHORT: dict[str, str] = {
    "baseline": "Baseline",
    "sft_only": "SFT only",
    "dpo_format": "DPO-format",
    "dpo_recovery": "DPO-recovery",
    "dpo_joint": "DPO-joint",
}

#: 中文字体栈。SVG 由浏览器渲染，所以写一串候选让浏览器自己挑。
FONT_STACK = (
    "Microsoft YaHei, PingFang SC, Hiragino Sans GB, Noto Sans CJK SC, "
    "Source Han Sans SC, WenQuanYi Zen Hei, SimHei, sans-serif"
)


def _esc(s: Any) -> str:
    return escape(str(s))


def _svg_open(width: int, height: int, title: str, *, is_mock: bool = False) -> list[str]:
    """开始一个 SVG 文档。

    ``is_mock=True`` 时在图右上角叠加一个 SIMULATED 横幅。
    这不是装饰 —— 图一旦被单独截图引用，就脱离了正文的免责说明；
    把标记**画进图里**是唯一可靠的隔离手段。
    """
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        f"<title>{_esc(title)}</title>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
    ]
    if is_mock:
        parts.append(
            f'<rect x="{width - 172}" y="10" width="162" height="26" rx="4" '
            f'fill="#FFF3CD" stroke="#E0A800" stroke-width="1.5"/>'
        )
        parts.append(
            _text(
                width - 91,
                23,
                "SIMULATED — 非真实结果",
                size=12,
                color="#8A6100",
                anchor="middle",
                weight="bold",
            )
        )
    return parts


def _text(
    x: float,
    y: float,
    s: Any,
    *,
    size: int = 13,
    color: str = "#222222",
    anchor: str = "start",
    weight: str = "normal",
    baseline: str = "middle",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT_STACK}" font-size="{size}" '
        f'fill="{color}" text-anchor="{anchor}" dominant-baseline="{baseline}" '
        f'font-weight="{weight}">{_esc(s)}</text>'
    )


# ============================================================================
# 图 1：错误类型分布（横向条形图 + 数值标注）
# ============================================================================


def error_distribution_svg(
    distribution: dict[str, Any],
    out_path: str | Path,
    *,
    label: str = "baseline",
    is_mock: bool = False,
    width: int = 900,
) -> str:
    """错误类型分布横向条形图。

    为什么用条形图而不是饼图：条形图的长度可比、可标注精确数值，
    而且在 SVG 里不需要算圆弧。饼图在只有 3-4 个类别时也没有优势。
    """
    counts: dict[str, int] = distribution.get("counts", {})
    total = int(distribution.get("total", 0)) or sum(counts.values()) or 1
    success = int(distribution.get("success_count", 0))

    # 只画失败类别（按数量降序），成功单独写在标题里
    rows = [(k, v) for k, v in counts.items() if k != "SUCCESS" and v > 0]
    rows.sort(key=lambda kv: -kv[1])
    if not rows:
        rows = [("UNKNOWN", 0)]

    row_h = 34
    margin_top = 84
    margin_left = 210
    margin_right = 130
    bar_max_w = width - margin_left - margin_right
    height = margin_top + row_h * len(rows) + 46

    svg = _svg_open(width, height, f"错误类型分布 {label}", is_mock=is_mock)
    svg.append(
        _text(
            24,
            32,
            f"错误类型分布（{label}）",
            size=17,
            weight="bold",
        )
    )
    svg.append(
        _text(
            24,
            56,
            f"样本总数 {total}，成功 {success}（{success / total:.1%}），"
            f"失败 {total - success}（{1 - success / total:.1%}）",
            size=12,
            color="#666666",
        )
    )

    max_v = max(v for _, v in rows) or 1
    for i, (cat, v) in enumerate(rows):
        y = margin_top + i * row_h
        color = CATEGORY_COLORS.get(cat, "#BDC3C7")
        bar_w = max(2.0, v / max_v * bar_max_w)
        share = v / total

        # 左侧类别名
        svg.append(
            _text(margin_left - 12, y + row_h / 2 - 4, CATEGORY_LABELS.get(cat, cat), size=12, anchor="end")
        )
        # 条形
        svg.append(
            f'<rect x="{margin_left}" y="{y + 5}" width="{bar_w:.1f}" height="{row_h - 14}" '
            f'rx="3" fill="{color}" opacity="0.88"/>'
        )
        # 条形右侧数值
        svg.append(
            _text(
                margin_left + bar_w + 8,
                y + row_h / 2 - 4,
                f"{v}  ({share:.1%})",
                size=12,
                color="#444444",
            )
        )

    # 脚注
    svg.append(
        _text(
            24,
            height - 18,
            "占比以「样本总数」为分母；各类别互斥，总和为 100%。",
            size=11,
            color="#888888",
        )
    )
    svg.append("</svg>")

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(svg), encoding="utf-8")
    return str(p)


# ============================================================================
# 图 2：分组柱状图（按难度 / 可见性）
# ============================================================================


def grouped_bar_svg(
    groups: dict[str, dict[str, Any]],
    out_path: str | Path,
    *,
    primary_k: int = 4,
    title: str = "按难度分组的成功率",
    subtitle: str = "",
    series: Sequence[tuple[str, str]] = (("accuracy", "单次准确率"), (f"pass^k", f"pass^k")),
    is_mock: bool = False,
    width: int = 900,
) -> str:
    """分组柱状图：每个组画 accuracy 与 pass^k 两根柱。

    ``series`` 是 ``(指标键, 图例名)`` 列表。``pass^k`` 会被替换成真实的 k 值。
    """
    names = list(groups)
    if not names:
        names = ["(无数据)"]
        groups = {"(无数据)": {}}

    resolved: list[tuple[str, str]] = []
    for key, lab in series:
        resolved.append((key.replace("pass^k", f"pass^{primary_k}"), lab.replace("pass^k", f"pass^{primary_k}")))

    margin_top = 92
    margin_left = 68
    margin_right = 24
    margin_bottom = 62
    plot_h = 300
    height = margin_top + plot_h + margin_bottom
    plot_w = width - margin_left - margin_right

    svg = _svg_open(width, height, title, is_mock=is_mock)
    svg.append(_text(24, 30, title, size=17, weight="bold"))
    if subtitle:
        svg.append(_text(24, 54, subtitle, size=12, color="#666666"))

    # y 轴网格
    for t in range(0, 6):
        val = t / 5
        y = margin_top + plot_h - val * plot_h
        svg.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" '
            f'stroke="#E3E3E3" stroke-width="1"/>'
        )
        svg.append(_text(margin_left - 8, y, f"{val:.0%}", size=11, color="#888888", anchor="end"))

    n_groups = len(names)
    group_w = plot_w / n_groups
    n_series = len(resolved)
    bar_w = min(38.0, group_w / (n_series + 0.9))

    for gi, gname in enumerate(names):
        gx = margin_left + gi * group_w
        center = gx + group_w / 2
        vals = [groups[gname].get(key) for key, _ in resolved]

        for si, ((key, _lab), v) in enumerate(zip(resolved, vals)):
            if v is None:
                continue
            v = float(v)
            offset = (si - (n_series - 1) / 2) * (bar_w + 6)
            x = center + offset - bar_w / 2
            h = max(1.0, v * plot_h)
            y = margin_top + plot_h - h
            color = ["#5B8FF9", "#4C9F70", "#F0AD4E", "#9B59B6"][si % 4]
            svg.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" '
                f'rx="2" fill="{color}" opacity="0.9"/>'
            )
            svg.append(
                _text(x + bar_w / 2, y - 10, f"{v:.1%}", size=11, color="#333333", anchor="middle", weight="bold")
            )

        # x 轴标签
        svg.append(
            _text(center, margin_top + plot_h + 22, gname, size=12, anchor="middle", color="#333333")
        )
        n = groups[gname].get("num_tasks")
        if n:
            svg.append(
                _text(center, margin_top + plot_h + 40, f"n={n}", size=11, anchor="middle", color="#999999")
            )

    # 图例
    lx = margin_left
    for si, (key, lab) in enumerate(resolved):
        color = ["#5B8FF9", "#4C9F70", "#F0AD4E", "#9B59B6"][si % 4]
        svg.append(f'<rect x="{lx}" y="{height - 26}" width="13" height="13" rx="2" fill="{color}"/>')
        svg.append(_text(lx + 19, height - 19, lab, size=12, color="#444444"))
        lx += 24 + len(lab) * 12

    svg.append("</svg>")
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(svg), encoding="utf-8")
    return str(p)


# ============================================================================
# 图 3：实验对比柱状图
# ============================================================================


def comparison_svg(
    reports: dict[str, dict[str, Any]],
    out_path: str | Path,
    *,
    metric: str = "pass_hat_k",
    metric_label: str = "pass^4",
    is_mock: bool = False,
    width: int = 900,
) -> str:
    """各实验在单一指标上的对比柱状图。"""
    order = [e for e in EXPERIMENT_COLORS if e in reports]
    order += [e for e in reports if e not in order]
    if not order:
        return ""

    margin_top = 84
    margin_left = 68
    margin_right = 24
    plot_h = 300
    height = margin_top + plot_h + 66
    plot_w = width - margin_left - margin_right

    svg = _svg_open(width, height, f"消融实验对比：{metric_label}", is_mock=is_mock)
    svg.append(_text(24, 30, f"消融实验对比：{metric_label}", size=17, weight="bold"))
    svg.append(
        _text(
            24,
            54,
            "所有实验使用完全相同的评测协议（任务集 / 采样数 / 温度 / prompt）",
            size=12,
            color="#666666",
        )
    )

    for t in range(0, 6):
        val = t / 5
        y = margin_top + plot_h - val * plot_h
        svg.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" '
            f'stroke="#E3E3E3" stroke-width="1"/>'
        )
        svg.append(_text(margin_left - 8, y, f"{val:.0%}", size=11, color="#888888", anchor="end"))

    group_w = plot_w / len(order)
    bar_w = min(64.0, group_w * 0.6)

    for i, exp in enumerate(order):
        v = float(reports[exp].get("headline", {}).get(metric, 0.0))
        gx = margin_left + i * group_w + group_w / 2
        x = gx - bar_w / 2
        h = max(1.0, v * plot_h)
        y = margin_top + plot_h - h
        color = EXPERIMENT_COLORS.get(exp, "#BDC3C7")
        svg.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="3" '
            f'fill="{color}" opacity="0.9"/>'
        )
        svg.append(
            _text(gx, y - 11, f"{v:.1%}", size=13, color="#222222", anchor="middle", weight="bold")
        )
        svg.append(
            _text(gx, margin_top + plot_h + 24, EXPERIMENT_SHORT.get(exp, exp), size=12, anchor="middle")
        )

    svg.append(
        _text(
            24,
            height - 18,
            "基线为未训练模型；所有微调实验共用同一套评测协议与同一份任务集。",
            size=11,
            color="#888888",
        )
    )
    svg.append("</svg>")
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(svg), encoding="utf-8")
    return str(p)


# ============================================================================
# 训练曲线（折线图）
# ============================================================================


def training_curve_svg(
    train: Sequence[dict[str, Any]],
    out_path: str | Path,
    *,
    experiment: str = "",
    is_mock: bool = False,
    width: int = 900,
) -> str:
    """DPO 训练 loss 曲线。"""
    pts = [(int(r["step"]), float(r["loss"])) for r in train]
    if not pts:
        return ""

    margin_top = 84
    margin_left = 76
    margin_right = 30
    plot_h = 300
    height = margin_top + plot_h + 56
    plot_w = width - margin_left - margin_right

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    # y 轴范围：让曲线有呼吸空间，并保证 log2 参考线在视野内
    y_lo = min(y0, 0.6931) - 0.02
    y_hi = max(y1, 0.6931) + 0.02
    if y_hi - y_lo < 1e-6:
        y_hi = y_lo + 0.01

    def px(s: float) -> float:
        return margin_left + (s - x0) / max(1e-9, x1 - x0) * plot_w

    def py(v: float) -> float:
        return margin_top + plot_h - (v - y_lo) / (y_hi - y_lo) * plot_h

    svg = _svg_open(width, height, "训练曲线", is_mock=is_mock)
    title = f"DPO 训练 loss 曲线（{experiment}）" if experiment else "DPO 训练 loss 曲线"
    if is_mock:
        title += " — SIMULATED"
    svg.append(_text(24, 30, title, size=17, weight="bold"))
    svg.append(
        _text(
            24,
            54,
            f"起始 loss 应接近 log 2 ≈ 0.6931（π_θ = π_ref 的起点）；"
            f"当前 {ys[0]:.4f} → {ys[-1]:.4f}",
            size=12,
            color="#666666",
        )
    )

    # y 轴刻度
    for t in range(0, 6):
        v = y_lo + (y_hi - y_lo) * t / 5
        y = py(v)
        svg.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" '
            f'stroke="#EEEEEE" stroke-width="1"/>'
        )
        svg.append(_text(margin_left - 8, y, f"{v:.4f}", size=10, color="#888888", anchor="end"))

    # log 2 参考线
    y_log2 = py(0.6931)
    if margin_top <= y_log2 <= margin_top + plot_h:
        svg.append(
            f'<line x1="{margin_left}" y1="{y_log2:.1f}" x2="{width - margin_right}" y2="{y_log2:.1f}" '
            f'stroke="#999999" stroke-width="1.2" stroke-dasharray="5,4"/>'
        )
        svg.append(
            _text(width - margin_right - 4, y_log2 - 11, "log 2 = 0.6931", size=11, color="#777777", anchor="end")
        )

    # 曲线
    path = " ".join(f"{'M' if i == 0 else 'L'}{px(s):.1f},{py(v):.1f}" for i, (s, v) in enumerate(pts))
    svg.append(f'<path d="{path}" fill="none" stroke="#D9534F" stroke-width="2"/>')

    # x 轴
    svg.append(
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{width - margin_right}" '
        f'y2="{margin_top + plot_h}" stroke="#BBBBBB" stroke-width="1"/>'
    )
    svg.append(_text(margin_left, margin_top + plot_h + 22, f"step {x0}", size=11, color="#888888"))
    svg.append(
        _text(width - margin_right, margin_top + plot_h + 22, f"step {x1}", size=11, color="#888888", anchor="end")
    )
    svg.append(_text(24, margin_top + plot_h + 44, "训练步数 →", size=12, color="#555555"))

    svg.append("</svg>")
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(svg), encoding="utf-8")
    return str(p)


# ============================================================================
# Mermaid（可直接贴进 GitHub README）
# ============================================================================


def error_distribution_mermaid(distribution: dict[str, Any]) -> str:
    """错误分布饼图（Mermaid）。GitHub 原生渲染，无需图片。"""
    counts: dict[str, int] = distribution.get("counts", {})
    rows = [(k, v) for k, v in counts.items() if v > 0]
    rows.sort(key=lambda kv: -kv[1])
    lines = ["```mermaid", "pie showData", "    title 错误类型分布"]
    for cat, v in rows:
        lines.append(f'    "{CATEGORY_LABELS.get(cat, cat)}" : {v}')
    lines.append("```")
    return "\n".join(lines)


def comparison_mermaid(
    reports: dict[str, dict[str, Any]],
    *,
    metric: str = "pass_hat_k",
    metric_label: str = "pass^4",
) -> str:
    """实验对比柱状图（Mermaid）。"""
    order = [e for e in EXPERIMENT_COLORS if e in reports]
    order += [e for e in reports if e not in order]
    lines = [
        "```mermaid",
        "xychart-beta",
        f'    title "消融实验对比：{metric_label}"',
        "    x-axis [" + ", ".join(f'"{EXPERIMENT_SHORT.get(e, e)}"' for e in order) + "]",
        "    y-axis \"比率\" 0 --> 1",
        "    bar ["
        + ", ".join(f"{float(reports[e].get('headline', {}).get(metric, 0.0)):.4f}" for e in order)
        + "]",
        "```",
    ]
    return "\n".join(lines)


# ============================================================================
# 统一入口
# ============================================================================


def make_all_svg(
    out_dir: str | Path,
    *,
    baseline_report: dict[str, Any] | None = None,
    baseline_distribution: dict[str, Any] | None = None,
    reports: dict[str, dict[str, Any]] | None = None,
    curves: dict[str, Any] | None = None,
) -> dict[str, str]:
    """生成所有可用的 SVG 图，返回 ``{图名: 路径}``。

    ``is_mock`` 会自动推断：只要输入数据里有任何一处带 ``is_mock``
    或标签含 ``mock``，**所有**图都会打上 SIMULATED 横幅。
    宁可为真数据多打一个标记，也不能让模拟图漏标。
    """
    out = Path(out_dir)
    made: dict[str, str] = {}

    # --- 自动推断是否模拟数据 ---
    def _looks_mock(*objs: Any) -> bool:
        for o in objs:
            if not isinstance(o, dict):
                continue
            if o.get("is_mock") is True:
                return True
            if str(o.get("label", "")).lower().startswith("mock"):
                return True
            if o.get("metadata", {}).get("is_mock") is True:
                return True
        return False

    is_mock = _looks_mock(baseline_distribution, baseline_report, reports, curves)

    if baseline_distribution:
        made["error_distribution_svg"] = error_distribution_svg(
            baseline_distribution,
            out / "error_distribution.svg",
            label=baseline_report.get("label", "baseline") if baseline_report else "baseline",
            is_mock=is_mock,
        )
        made["error_distribution_mermaid"] = str(
            _write_text(out / "error_distribution.mermaid.md", error_distribution_mermaid(baseline_distribution))
        )

    if baseline_report:
        for group_key, fname, title in (
            ("by_difficulty", "success_by_difficulty.svg", "按难度分组的成功率"),
            ("by_visibility", "success_by_visibility.svg", "按可见性分组的成功率（unseen = 泛化）"),
        ):
            groups = baseline_report.get(group_key, {})
            if groups:
                made[fname] = grouped_bar_svg(
                    groups,
                    out / fname,
                    primary_k=baseline_report.get("primary_k", 4),
                    title=title,
                    subtitle=f"{baseline_report.get('label', '')} · "
                    f"{baseline_report.get('num_tasks', 0)} 任务 × "
                    f"{baseline_report.get('num_samples_per_task', 0)} 采样",
                    is_mock=is_mock,
                )

    if reports:
        made["ablation_comparison_svg"] = comparison_svg(
            reports,
            out / "ablation_comparison.svg",
            metric="pass_hat_k",
            metric_label="pass^4",
            is_mock=is_mock,
        )
        made["ablation_comparison_mermaid"] = str(
            _write_text(out / "ablation_comparison.mermaid.md", comparison_mermaid(reports))
        )

    if curves and curves.get("train"):
        made["training_curves_svg"] = training_curve_svg(
            curves["train"],
            out / "training_curves.svg",
            experiment=curves.get("experiment", ""),
            is_mock=is_mock or bool(curves.get("is_mock")),
        )

    return {k: v for k, v in made.items() if v}


def _write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    # 自检：用一份假数据验证 SVG 能生成且是合法 XML
    demo_dist = {
        "total": 400,
        "success_count": 200,
        "counts": {"SUCCESS": 200, "FORMAT_ERROR": 80, "PARAM_ERROR": 60, "HALLUCINATION": 35, "WRONG_TOOL": 25},
    }
    import xml.etree.ElementTree as ET

    p = error_distribution_svg(demo_dist, ".tmp/demo_dist.svg")
    ET.parse(p)  # 解析失败会抛异常
    print(f"✓ SVG 生成且 XML 合法: {p}")
    print(error_distribution_mermaid(demo_dist))
