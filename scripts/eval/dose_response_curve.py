"""C1 判定：把三个剂量变体的结果画成曲线并检验单调性。

## 判定规则（**事先定好**，避免事后挑口径）

对剂量 ``0% / 30% / 100%``：

| 假设 | 判据 |
|---|---|
| **机制成立** | ``clarify_instead`` 错误数随剂量**单调上升** **且** accuracy 随剂量**单调下降** |
| **部分成立** | 只有其中之一单调 |
| **机制不成立** | 都不单调 |

**「事先定好」很重要**：如果先看结果再定判据，就总能找到一种说法
让数据「支持」假设。我把判据写在这里和测试里。

## 为什么用「单调」而不是「显著差异」

100 个任务、配对检验的 MDE ≈ 14pp（见 ``paired_stats.mde``）。
三个剂量点之间如果只差几个百分点，**任何单点对比都不可能显著**。
但**单调性**是一个更强的联合约束：三个点同向排列的概率
在无效应下是 1/3（对每个指标），两个指标同时成立是 1/9。
所以「单调」在小样本下比「单点显著」更有信息量 ——
但**仍不是显著性证明**，报告里必须这样表述。

## 用法

```bash
python scripts/eval/dose_response_curve.py \
    --labels dose000,dose030,dose100 \
    --doses 0,0.3,1.0 \
    --out experiments/dose_curve.md
```
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def monotonic(values: list[float], *, direction: str, tol: float = 0.0) -> bool:
    """判断序列是否单调。

    ``direction='up'`` 要求非严格递增（允许相等），``'down'`` 反之。
    ``tol`` 允许的容差（浮点噪声）。

    >>> monotonic([0, 1, 2], direction="up")
    True
    >>> monotonic([2, 1, 0], direction="down")
    True
    >>> monotonic([0, 2, 1], direction="up")
    False
    >>> monotonic([1, 1, 1], direction="up")
    True
    """
    for x, y in zip(values, values[1:]):
        if direction == "up" and y < x - tol:
            return False
        if direction == "down" and y > x + tol:
            return False
    return True


def strictly_monotonic(values: list[float], *, direction: str) -> bool:
    """严格单调（不允许相等）—— 用来区分「真的动了」与「纹丝不动」。

    >>> strictly_monotonic([1, 1, 1], direction="up")
    False
    >>> strictly_monotonic([0, 1, 2], direction="up")
    True
    """
    return monotonic(values, direction=direction) and (
        len(set(values)) == len(values)
    )


def load_report(label: str, exp_root: Path) -> tuple[dict | None, bool]:
    """读 ``eval_report*.json``，返回 ``(报告, 是否为 mock, 错误分析)``。

    ⚠️ mock 跑出来的报告名是 ``eval_report.mock.json``（带后缀）——
    早期只找 ``eval_report.json``，于是 mock 链路「评测成功但找不到报告」，
    而脚本只报「跳过」，看不出是命名问题。

    **是否为 mock 必须向上传递**：mock 数字不能作为结论，
    剂量曲线若基于 mock 数据必须显式标注。
    """
    d = exp_root / label
    for name, is_mock in (
        ("eval_report.json", False),
        ("report.json", False),
        ("eval_report.mock.json", True),
    ):
        p = d / name
        if p.exists():
            rep = json.loads(p.read_text(encoding="utf-8"))
            # 错误子类型在**另一份**报告里（见 extract_metrics 的说明）
            an = None
            for aname in ("error_analysis.json", "error_analysis.mock.json"):
                ap = d / aname
                if ap.exists():
                    an = json.loads(ap.read_text(encoding="utf-8"))
                    break
            return rep, is_mock, an
    return None, False, None


def extract_metrics(rep: dict, analysis: dict | None = None) -> dict:
    """从报告里抽出剂量曲线需要的量。

    ⚠️ **字段名的实际位置（实测确认，别再猜）**：

    * accuracy 在 ``eval_report.json`` 的 ``headline.accuracy``
      —— 不是 ``overall``，也不是顶层
    * **错误子类型**（``clarify_instead`` 等）在
      ``error_analysis.json`` 的 ``distribution.subtype_counts``；
      而 ``distribution.counts`` 只有 5 个顶层类别
      （SUCCESS / FORMAT_ERROR / HALLUCINATION / PARAM_ERROR / WRONG_TOOL），
      **不含子类型**

    早期我只从一份报告里按 ``error_distribution`` 找，结果**所有计数都是 0**
    而脚本不报错 —— 那会让剂量曲线看起来「三条线全是平的」，
    进而把「字段找错了」误读成「剂量没有效应」。**这是会直接误导结论的错误。**

    >>> r = {"headline": {"accuracy": 0.5}}
    >>> a = {"distribution": {"subtype_counts": {"clarify_instead": 3}}}
    >>> extract_metrics(r, a)["clarify_instead"]
    3
    >>> extract_metrics(r, a)["accuracy"]
    0.5
    >>> extract_metrics({})["accuracy"]
    0.0
    >>> extract_metrics({})["clarify_instead"]
    0
    """
    headline = rep.get("headline") or {}
    overall = rep.get("overall") or {}
    acc = headline.get("accuracy", overall.get("accuracy", rep.get("accuracy", 0.0)))

    dist: dict = {}
    if analysis:
        dist = analysis.get("distribution") or {}
    if not dist:
        dist = (rep.get("error_distribution") or rep.get("distribution")
                or rep.get("errors") or {})
    sub_counts = dist.get("subtype_counts") or {}
    plain_counts = dist.get("counts") or {}

    def _sub(name: str) -> int:
        """子类型计数；兼容「直接放在 dist 下」的旧报告。"""
        if name in sub_counts:
            return int(sub_counts[name])
        v = dist.get(name)
        return int(v) if isinstance(v, (int, float)) else 0

    return {
        "accuracy": float(acc),
        "clarify_instead": _sub("clarify_instead"),
        "multi_tool": _sub("multi_tool"),
        "wrong_value": _sub("wrong_value"),
        "adjacent_tool": _sub("adjacent_tool"),
        "should_clarify": _sub("should_clarify"),
        "format_error": int(plain_counts.get("FORMAT_ERROR", 0)),
    }


def analyze(doses: list[float], per_dose: list[dict]) -> dict:
    """判定机制是否成立（**判据事先定好**，见模块文档）。"""
    acc = [d["accuracy"] for d in per_dose]
    ci = [d["clarify_instead"] for d in per_dose]
    acc_down = monotonic(acc, direction="down")
    ci_up = monotonic(ci, direction="up")
    acc_down_strict = strictly_monotonic(acc, direction="down")
    ci_up_strict = strictly_monotonic(ci, direction="up")

    if acc_down and ci_up:
        verdict = "机制成立" if (acc_down_strict or ci_up_strict) else "机制成立（但变化幅度为 0）"
    elif acc_down or ci_up:
        verdict = "部分成立"
    else:
        verdict = "机制不成立"
    return {
        "doses": doses,
        "accuracy": acc,
        "clarify_instead": ci,
        "accuracy_monotone_down": acc_down,
        "clarify_instead_monotone_up": ci_up,
        "accuracy_strictly_down": acc_down_strict,
        "clarify_instead_strictly_up": ci_up_strict,
        "verdict": verdict,
    }


def render(res: dict, labels: list[str], per_dose: list[dict],
           *, is_mock: bool = False) -> str:
    L = ["## C1：clarify 偏好对剂量-反应", ""]
    if is_mock:
        L.append("> ⚠️ **本曲线基于 mock 报告，只能验证脚本链路，"
                 "不能作为项目结论。**")
        L.append("")
    L.append("| 剂量 | 实验 | accuracy | clarify_instead | multi_tool | wrong_value |")
    L.append("|---|---|---|---|---|---|")
    for dose, lab, d in zip(res["doses"], labels, per_dose):
        L.append(f"| {dose:.0%} | `{lab}` | {d['accuracy']:.3f} | "
                 f"{d['clarify_instead']} | {d['multi_tool']} | {d['wrong_value']} |")
    L.append("")
    L.append("### 判定（判据事先定好，见脚本模块文档）")
    L.append("")
    L.append(f"- accuracy 随剂量单调下降：**{'是' if res['accuracy_monotone_down'] else '否'}**"
             + ("（严格下降）" if res["accuracy_strictly_down"] else ""))
    L.append(f"- clarify_instead 随剂量单调上升："
             f"**{'是' if res['clarify_instead_monotone_up'] else '否'}**"
             + ("（严格上升）" if res["clarify_instead_strictly_up"] else ""))
    L.append(f"- **结论：{res['verdict']}**")
    L.append("")
    L.append("### 必须写进报告的限制")
    L.append("")
    L.append("- 「单调」是**联合约束**，不是显著性证明。三个点同向排列在无效应下"
             "概率为 1/3（单指标）、1/9（双指标），信息量大于单点对比，"
             "但**不能**表述为「统计显著」。")
    L.append("- 100 个任务、配对检验的 MDE ≈ 14pp：三个点之间若只差几个百分点，"
             "任何**单点对比**都不可能显著。")
    L.append("- 若 100% 变体用了**有放回重采样**（源数据 clarify 不足），"
             "必须在报告里写明 —— 同一数据被多次见到会放大该方向的梯度信号。")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="C1 剂量-反应曲线判定")
    ap.add_argument("--labels", required=True, help="逗号分隔的实验标签，按剂量递增")
    ap.add_argument("--doses", required=True, help="逗号分隔的剂量，如 0,0.3,1.0")
    ap.add_argument("--exp-root", default="experiments")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    doses = [float(x) for x in args.doses.split(",")]
    if len(labels) != len(doses):
        print(f"  ✗ labels({len(labels)}) 与 doses({len(doses)}) 数量不一致")
        return 1

    exp_root = Path(args.exp_root)
    per_dose, used_labels, used_doses = [], [], []
    any_mock = False
    for lab, dose in zip(labels, doses):
        rep, is_mock, an = load_report(lab, exp_root)
        if rep is None:
            print(f"  ✗ {lab}: 找不到 eval_report*.json，跳过")
            continue
        if is_mock:
            any_mock = True
        if an is None:
            print(f"  ⚠️ {lab}: 没有 error_analysis*.json —— "
                  "clarify_instead 等子类型计数将全为 0")
        per_dose.append(extract_metrics(rep, an))
        used_labels.append(lab)
        used_doses.append(dose)

    if any_mock:
        print("  ⚠️ 有剂量点用的是 **mock 报告**（``*.mock.json``）。")
        print("     下面这张曲线**只能用于验证脚本链路**，不能作为项目结论。")

    if len(per_dose) < 2:
        print("  ✗ 至少需要 2 个剂量点才能谈单调性")
        return 1

    res = analyze(used_doses, per_dose)
    md = render(res, used_labels, per_dose, is_mock=any_mock)
    print(md)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"\n  已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
