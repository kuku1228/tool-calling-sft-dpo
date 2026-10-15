"""把各实验的评测结果汇总成一张可直接写进报告的对比表。

为什么需要这个脚本
------------------
``docs/ablation/ablation_report.md`` 里有 104 处 ``<待填>`` 与 18 处
``TODO(真实数据)``。**手工填这么多个位置必然出错**（漏填、串行、单位写错），
而且一旦某个实验重跑，手工填的表格就全部过期。

所以改成：从 ``experiments/<label>/eval_report.json`` 与
``error_analysis.json`` 读出真实数字，由脚本生成表格。重跑任何实验后
只需再执行一次本脚本。

数据源与优先级
--------------
**权威来源是 ``eval_report.json``**（由 ``run_baseline.py`` 产生）。
本脚本不读 ``predictions.jsonl`` 二次计算 —— 那会引入第二套口径，
开发期已经因此误判过一次错误分类。

用法::

    python scripts/eval/make_ablation_table.py
    python scripts/eval/make_ablation_table.py --labels baseline_airline,sft_only,dpo_joint,dpo_from_sft
    python scripts/eval/make_ablation_table.py --markdown > docs/ablation/_table.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import CFG  # noqa: E402

#: 报告里展示顺序（也是消融的逻辑顺序：起点 → 纯正例监督 → 从基座的对比 → 从 SFT 的对比）
DEFAULT_LABELS: tuple[str, ...] = (
    "baseline_airline",
    "sft_only",
    "dpo_joint",
    "dpo_from_sft",
    "dpo_format",
)

#: 展示用的中文名
DISPLAY: dict[str, str] = {
    "baseline_airline": "baseline（原始基座）",
    "sft_only": "SFT（纯正例监督）",
    "dpo_joint": "DPO-joint（从基座）",
    "dpo_from_sft": "DPO-joint（SFT 续训）",
    "dpo_format": "DPO-format（仅格式对）",
}


def load_one(label: str) -> dict | None:
    """读一个实验的全部结果。缺失时返回 None 并说明原因。"""
    d = CFG.path("experiments", label)

    rep_p = d / "eval_report.json"
    ea_p = d / "error_analysis.json"
    tr_p = CFG.path("outputs", label, "train_health.txt")
    # 从 SFT 续训的实验，产物目录名与 label 不同
    if not tr_p.exists() and label == "dpo_from_sft":
        tr_p = CFG.path("outputs", "dpo_from_sft", "train_health.txt")

    if not rep_p.exists():
        return {"label": label, "missing": f"{rep_p} 不存在"}

    rep = json.loads(rep_p.read_text(encoding="utf-8"))
    out: dict = {
        "label": label,
        "display": DISPLAY.get(label, label),
        "is_mock": rep.get("metadata", {}).get("is_mock"),
        "adapter": rep.get("metadata", {}).get("adapter_path"),
        "headline": rep.get("headline", {}),
        "num_tasks": rep.get("num_tasks"),
        "num_samples": rep.get("num_samples_per_task"),
        "by_difficulty": rep.get("by_difficulty"),
        "by_visibility": rep.get("by_visibility"),
        "by_category": rep.get("by_category"),
        "metadata": rep.get("metadata", {}),
    }

    if ea_p.exists():
        ea = json.loads(ea_p.read_text(encoding="utf-8"))
        dist = ea.get("distribution", {})
        out["counts"] = dist.get("counts", {})
        out["failure_subtype_counts"] = dist.get("failure_subtype_counts", {})
        out["success_rate"] = dist.get("success_rate")
        out["format_accuracy"] = ea.get("format_accuracy")
        out["format_ok_value_wrong"] = ea.get("format_ok_value_wrong")

    # ⚠️ 恢复分析从 ``eval_report.json`` 的 ``metadata.recovery_analysis`` 读，
    # **不要**读 ``error_analysis.json`` 的 ``recovery`` 段。原因：
    #   * 该段是后加功能，早期跑的 baseline / sft_only 里根本没有
    #   * 即使有，字段名也不同（那边是 recovered_rate 而非 recovered 计数）
    # 同一份报告混用两个 schema 的数据源是错的 —— 这里统一到 eval_report。
    out["recovery"] = (rep.get("metadata") or {}).get("recovery_analysis")
    # 训练健康度
    if tr_p.exists():
        out["train_health"] = tr_p.read_text(encoding="utf-8")
    return out


def _fmt_pct(v) -> str:
    try:
        return f"{float(v) * 100:.1f}%"
    except (TypeError, ValueError):
        return "—"


def _fmt_ci(hl: dict, key: str) -> str:
    """从 headline 里取点估计 + 95% CI。"""
    ci = hl.get(f"{key}_ci") or hl.get("ci", {}).get(key) if isinstance(hl.get("ci"), dict) else hl.get(f"{key}_ci")
    point = hl.get(key)
    if point is None:
        return "—"
    if isinstance(ci, (list, tuple)) and len(ci) == 2:
        return f"{float(point) * 100:.1f}% [{float(ci[0]) * 100:.1f}, {float(ci[1]) * 100:.1f}]"
    return f"{float(point) * 100:.1f}%"


def _group(rep: dict, key: str) -> dict:
    """把 by_* 分组统一成 {组名: {accuracy, pass_hat_k, n}} 的形式。

    eval_report 里这一项的形态在不同版本间不完全一致（dict of dict 或
    list of dict），这里两种都吃。
    """
    g = rep.get(key)
    if not g:
        return {}
    if isinstance(g, dict):
        items = g.items()
    else:
        items = ((x.get("group") or x.get("name"), x) for x in g)
    out: dict = {}
    for k, v in items:
        if not isinstance(v, dict):
            continue
        out[str(k)] = {
            "n": v.get("n", v.get("num_tasks", v.get("count"))),
            "accuracy": v.get("accuracy"),
            "pass_hat_k": v.get("pass_hat_k", v.get("pass_k")),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成消融实验对比表")
    ap.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    ap.add_argument("--markdown", action="store_true", help="输出 Markdown 表格")
    args = ap.parse_args(argv)

    labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    results = [load_one(lb) for lb in labels]

    print("=" * 78)
    print("消融实验对比表（数据源：experiments/<label>/eval_report.json）")
    print("=" * 78)

    ok = [r for r in results if r and "missing" not in r]
    missing = [r for r in results if r and "missing" in r]

    if missing:
        print("\n⚠️ 缺失的实验（表格中留空）：")
        for r in missing:
            print(f"    {r['label']:<18} {r['missing']}")

    if not ok:
        print("\n没有任何可用的实验结果。")
        return 1

    # --- 表格 1：总体指标 ---
    print("\n### 表 1 总体指标\n")
    hdr = f"| {'实验':<26} | accuracy | pass@4 | pass^4 | Δstability |"
    print(hdr)
    print("|" + "-" * 28 + "|" + "-" * 22 + "|" + "-" * 22 + "|" + "-" * 22 + "|" + "-" * 12 + "|")
    for r in results:
        name = r.get("display", r["label"])
        if "missing" in r:
            print(f"| {name:<26} | — | — | — | — |")
            continue
        hl = r["headline"]
        print(
            f"| {name:<26} | {_fmt_ci(hl,'accuracy'):<20} "
            f"| {_fmt_ci(hl,'pass_at_k'):<20} | {_fmt_ci(hl,'pass_hat_k'):<20} "
            f"| {_fmt_pct(hl.get('delta_stability')):>10} |"
        )

    # --- 表格 2：错误类型构成 ---
    print("\n### 表 2 错误类型构成（占总样本）\n")
    cats = ["SUCCESS", "FORMAT_ERROR", "PARAM_ERROR", "WRONG_TOOL", "HALLUCINATION", "REFUSED"]
    present = [c for c in cats if any(c in (r.get("counts") or {}) for r in ok)]
    print("| 实验 | " + " | ".join(present) + " |")
    print("|" + "-" * 26 + "|" + "|".join("-" * (len(c) + 2) for c in present) + "|")
    for r in results:
        if "missing" in r:
            continue
        counts = r.get("counts") or {}
        total = sum(counts.values()) or 1
        cells = []
        for c in present:
            v = counts.get(c)
            cells.append(f"{v} ({v / total * 100:.1f}%)" if v is not None else "0")
        print(f"| {r['display']:<24} | " + " | ".join(cells) + " |")

    # --- 表格 3：失败子类型 ---
    print("\n### 表 3 失败子类型（关键：看失败形态的结构性变化）\n")
    subs: set[str] = set()
    for r in ok:
        subs |= set((r.get("failure_subtype_counts") or {}).keys())
    subs_sorted = sorted(subs, key=lambda s: -sum((r.get("failure_subtype_counts") or {}).get(s, 0) for r in ok))
    if subs_sorted:
        print("| 子类型 | " + " | ".join(r["display"] for r in results if "missing" not in r) + " |")
        print("|" + "-" * 26 + "|" + "|".join("-" * (len(r["display"]) + 2) for r in results if "missing" not in r) + "|")
        for s in subs_sorted:
            cells = []
            for r in results:
                if "missing" in r:
                    continue
                cells.append(str((r.get("failure_subtype_counts") or {}).get(s, 0)))
            print(f"| {s:<24} | " + " | ".join(cells) + " |")
    else:
        print("（无子类型数据）")

    # --- 表格 4：分组（难度 / 可见性） ---
    for key, title in (("by_difficulty", "按难度"), ("by_visibility", "按可见性")):
        print(f"\n### 表 4  {title}\n")
        groups: set[str] = set()
        for r in ok:
            groups |= set(_group(r, key).keys())
        if not groups:
            print("（无数据）")
            continue
        print("| 组 | " + " | ".join(r["display"] for r in results if "missing" not in r) + " |")
        print("|" + "-" * 12 + "|" + "|".join("-" * (len(r["display"]) + 2) for r in results if "missing" not in r) + "|")
        for g in sorted(groups):
            cells = []
            for r in results:
                if "missing" in r:
                    continue
                v = _group(r, key).get(g)
                n = v.get("n") if v else None
                cells.append(f"{_fmt_pct(v.get('accuracy'))} (n={n})" if v else "—")
            print(f"| {g:<10} | " + " | ".join(cells) + " |")

    # --- 表格 5：错误恢复 ---
    print("\n### 表 5 错误恢复能力（纠错轮）\n")
    recs = [(r, (r.get("recovery") or {})) for r in results if "missing" not in r]
    if any(v for _, v in recs):
        print("| 实验 | recovered | recovered_rate | repeat_rate |")
        print("|" + "-" * 26 + "|" + "-" * 12 + "|" + "-" * 16 + "|" + "-" * 14 + "|")
        for r, rv in recs:
            rec_n = rv.get("recovered")
            tot_n = rv.get("num_recovery_samples")
            frac = f"{rec_n}/{tot_n}" if rec_n is not None and tot_n else "—"
            print(
                f"| {r['display']:<24} | {frac:>10} "
                f"| {_fmt_pct(rv.get('recovered_rate')):>14} "
                f"| {_fmt_pct(rv.get('repeat_rate')):>12} |"
            )
    else:
        print("（无恢复数据）")

    # --- 表 6：逐错误模式（最能说明问题的一节） ---
    print("\n### 表 6 逐错误模式 recovered / repeated\n")
    keys: set[str] = set()
    for _, rv in recs:
        keys |= set((rv.get("by_mistake_key") or {}).keys())
    if keys:
        names = [r["display"] for r, _ in recs]
        print("| 错误模式 | " + " | ".join(names) + " |")
        print("|" + "-" * 34 + "|" + "|".join("-" * (len(n) + 2) for n in names) + "|")
        for k in sorted(keys):
            cells = []
            for _, rv in recs:
                v = (rv.get("by_mistake_key") or {}).get(k)
                if not v:
                    cells.append("—")
                else:
                    # 字段是 recovered_rate（比率）而非 recovered（计数）
                    rr = v.get("recovered_rate")
                    rp = v.get("repeat_rate")
                    cells.append(
                        f"{_fmt_pct(rr)}/rpt {_fmt_pct(rp)}" if rr is not None else "—"
                    )
            print(f"| {k:<32} | " + " | ".join(cells) + " |")

    # --- 训练健康度 ---
    print("\n### 表 7 训练健康度\n")
    for r in results:
        if "missing" in r:
            continue
        th = r.get("train_health")
        if not th:
            print(f"  {r['display']}: （无 train_health.txt）")
            continue
        lines = [ln for ln in th.splitlines() if ln.strip()]
        print(f"  --- {r['display']}")
        for ln in lines:
            print(f"      {ln}")

    print("\n" + "=" * 78)
    print("提示：这些数字来自 eval_report.json（run_baseline.py 的产物）。")
    print("      不要用 predictions.jsonl 二次计算 —— 会引入第二套口径。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
