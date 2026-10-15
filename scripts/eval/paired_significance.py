"""配对显著性检验 CLI。

统计逻辑在 ``src/eval/paired_stats.py``（可被测试直接导入），
本脚本只负责读实验产物、渲染 Markdown。

## 为什么需要它（我之前的判据是错的）

我一直用「两个实验的 95% CI 是否重叠」判断显著性。**那是错的**：
两个实验跑在**同一批任务**上，是配对数据，必须用配对检验。
详见 ``src/eval/paired_stats`` 的模块文档。

## 用法

```bash
# 两组对比
python scripts/eval/paired_significance.py --a sft_v3 --b dpo_sft_v3_b05

# 基线 vs 多组（一次算出全部对比）
python scripts/eval/paired_significance.py \
    --baseline baseline \
    --against sft,dpo_dose_0,dpo_dose_30,dpo_dose_100

# 指定二值化阈值（默认 0.5 = 过半采样答对）
python scripts/eval/paired_significance.py --a X --b Y --thr 1.0

# 写到文件（Markdown，可直接粘进报告）
python scripts/eval/paired_significance.py --a X --b Y --out report/sig.md
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

from src.data.task_manifest import load_tasks_by_version  # noqa: E402
from src.eval import paired_stats as PS  # noqa: E402
from src.eval.grader import grade  # noqa: E402


def load_graded(label: str, exp_root: Path) -> dict[str, list[bool]]:
    """薄包装：真正的实现在 ``src/eval/paired_stats.load_graded``。

    放在 src 里是为了让测试能直接导入它 —— ``scripts`` 不是包，
    测试从 ``scripts.eval...`` 导入会依赖 ``sys.path`` 的偶然状态。
    """
    return PS.load_graded(label, exp_root, grade_fn=grade)



def render(results: list[dict], *, title: str) -> str:
    lines = [f"## {title}", ""]
    lines.append(
        "**差值 = A − B**（正数表示 A 更好）。"
        "「任务级」按**任务**重采样；「簇级」先把**同一期望调用**的任务取平均，"
        "再按**簇**重采样 —— 后者是更保守的口径，因为"
        "有些任务的正确答案完全相同（例如所有「应先追问」任务的期望调用都是空的），"
        "它们不是独立证据。"
    )
    lines.append("")
    lines.append(
        "| A | B | 口径 | 样本数 | A 均值 | B 均值 | **差值 [95% CI]** | 显著 | "
        "b/c | p | MDE |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        mc = r["mcnemar"]
        lines.append(
            f"| `{r['a']}` | `{r['b']}` | 任务级 | {r['n_common']} | "
            f"{r['mean_a']:.3f} | {r['mean_b']:.3f} | "
            f"**{r['diff']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]** | "
            f"{'✅' if r['ci_excludes_zero'] else '❌'} | "
            f"{mc['b']}/{mc['c']} | {mc['p_value']:.3f} | {r['mde']:.3f} |"
        )
        cl = r.get("clustered")
        if cl:
            cmc = cl["mcnemar"]
            lines.append(
                f"| `{r['a']}` | `{r['b']}` | **簇级** | **{cl['n_clusters']}** | "
                f"— | — | "
                f"**{cl['diff']:+.3f} [{cl['ci_lo']:+.3f}, {cl['ci_hi']:+.3f}]** | "
                f"{'✅' if cl['ci_excludes_zero'] else '❌'} | "
                f"{cmc['b']}/{cmc['c']} | {cmc['p_value']:.3f} | {cl['mde']:.3f} |"
            )
    lines.append("")
    lines.append("### 读数说明")
    lines.append("")
    for r in results:
        lines.append(f"**`{r['a']}` vs `{r['b']}`**")
        odd = r["n_only_a"] + r["n_only_b"]
        if odd:
            lines.append(
                f"- ⚠️ 两组任务集**不同**：共同 {r['n_common']} 个，"
                f"A 独有 {r['n_only_a']}、B 独有 {r['n_only_b']}"
                "（配对比较只用了交集）"
            )
        lines.append(
            f"- 任务级：差值 {r['diff']:+.3f}，CI [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] → "
            + ("**显著**" if r["ci_excludes_zero"]
               else "**不显著**（CI 含 0）")
        )
        cl = r.get("clustered")
        if cl:
            lines.append(
                f"- 簇级（有效样本 {cl['n_clusters']}）：差值 {cl['diff']:+.3f}，"
                f"CI [{cl['ci_lo']:+.3f}, {cl['ci_hi']:+.3f}] → "
                + ("**显著**" if cl["ci_excludes_zero"] else "**不显著**")
            )
            if cl["n_clusters"] < r["n_common"]:
                shrink = r["n_common"] - cl["n_clusters"]
                lines.append(
                    f"  → 有效样本从 {r['n_common']} 降到 **{cl['n_clusters']}**"
                    f"（{shrink} 个任务是重复计数）。**报告应采用簇级口径**，"
                    "它更保守。"
                )
            if (cl["ci_excludes_zero"] != r["ci_excludes_zero"]):
                lines.append(
                    "  → ⚠️ **两种口径的显著性结论不一致** —— "
                    "任务级说显著、簇级说不显著（或反之）。"
                    "此时应采信**簇级**，并在报告里写明这个差别。"
                )
        lines.append(
            f"- 任务级 MDE ≈ **{r['mde']:.3f}**（σ={r['sigma']:.3f}，n={r['n_common']}）"
        )
        if cl:
            lines.append(
                f"- 簇级 MDE ≈ **{cl['mde']:.3f}**（σ={cl['sigma']:.3f}，"
                f"n={cl['n_clusters']}）"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="配对显著性检验（任务级）")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--a", help="实验 A")
    g.add_argument("--baseline", help="基线实验（与 --against 配合）")
    ap.add_argument("--b", help="实验 B（与 --a 配合）")
    ap.add_argument("--against", help="逗号分隔的实验列表（与 --baseline 配合）")
    ap.add_argument("--exp-root", default="experiments")
    ap.add_argument("--thr", type=float, default=0.5,
                    help="任务级二值化阈值（默认 0.5 = 过半采样答对）")
    ap.add_argument("--task-set", default="v3", choices=["v1", "v3"],
                    help="用于构造簇定义的协议版本（默认 v3）")
    ap.add_argument("--no-cluster", action="store_true",
                    help="只出任务级口径（不做按簇重采样）")
    ap.add_argument("--out", default=None, help="输出 Markdown 路径")
    args = ap.parse_args(argv)

    exp_root = Path(args.exp_root)
    cache: dict[str, dict] = {}

    def _load(lab: str) -> dict:
        if lab not in cache:
            cache[lab] = load_graded(lab, exp_root)
        return cache[lab]

    if args.a:
        if not args.b:
            ap.error("--a 需要同时给 --b")
        pairs = [(args.a, args.b)]
    else:
        if not args.against:
            ap.error("--baseline 需要同时给 --against")
        pairs = [(args.baseline, x.strip())
                 for x in args.against.split(",") if x.strip()]

    # 簇定义：同一「期望调用」的任务归为一簇。有些任务的正确答案完全相同
    # （例如所有「应先追问」任务的期望调用都是空的），它们不是独立证据。
    # 用 --no-cluster 可以关掉（只出任务级口径）。
    cluster_of = None
    if not args.no_cluster:
        try:
            cluster_of = PS.expected_keys(args.task_set)
            n_uniq = len(set(cluster_of.values()))
            print(f"  簇定义：{len(cluster_of)} 个任务 → {n_uniq} 个不同期望调用"
                  f"（有效独立样本 {n_uniq}）")
            print()
        except Exception as exc:  # pragma: no cover
            print(f"  ⚠️ 无法构造簇定义（{exc}），只出任务级口径")

    results = []
    for a_lab, b_lab in pairs:
        try:
            A, B = _load(a_lab), _load(b_lab)
        except FileNotFoundError as exc:
            print(f"  ✗ {exc}")
            continue
        results.append(PS.compare(a_lab, b_lab, A, B, thr=args.thr,
                                 cluster_of=cluster_of))

    if not results:
        print("  ✗ 没有可比较的实验")
        return 1

    title = (f"配对显著性检验：{results[0]['a']} vs {results[0]['b']}"
             if len(results) == 1
             else f"配对显著性检验：{results[0]['a']} vs {len(results)} 组")
    md = render(results, title=title)
    print(md)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"\n  已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
