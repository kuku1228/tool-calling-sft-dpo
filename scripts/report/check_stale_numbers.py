"""检查文档里是否**无标注地**引用了已作废的数字。

## 为什么需要它

这个项目的性能数字被作废过**两轮**：

1. **污染数据**：`sft_only` 的 95.0% 来自训练数据含评测任务的版本
2. **旧任务集**：`50.25% / 83.00% / 49.25% / 68.5% / 10.0%` 等跑在 v1 上，
   而 v1 的骨架/任务比只有 0.52、日期全在过去、`multi_tool` 名不符实

每次作废之后，我都把「新数字」写进文档 —— 但**旧数字留在原地**，
因为文档里还有依赖它们的叙述。于是文档变成两套数字共存，
**而读者（包括面试官、包括几个月后的我）无法分辨哪个有效**。

这不是「数字旧了」，而是**文档会主动误导人**。

## 判据

对每个已知作废的数字，检查它出现的那一行**附近**（同一行 + 前后各 3 行）
是否有作废标注。标注词可配置，默认：

    作废 / 废弃 / 已作废 / 不要使用 / 不可引用 / ⚠️ / 等 v3 / 待重跑 / 暂缓

**豁免**：如果该数字出现在明确标记为「历史 / 附录 / 旧版」的小节里，
也算有标注 —— 那种场景下引用旧数字是**正确的**（例如「我推翻了 95% 那个结果」）。

## 用法

```bash
python scripts/report/check_stale_numbers.py
python scripts/report/check_stale_numbers.py --strict   # 有问题时退出码非 0
```

**退出码**：``--strict`` 时有未标注的作废数字则返回 1。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]

#: 已作废的数字（正则片段）→ 作废原因
#:
#: ⚠️ 维护这份列表是**有意的负担**：每次作废一批数字就在这里登记一条。
#: 不登记的话，检查器抓不到它，文档就会重新变得不可信。
STALE_NUMBERS: dict[str, str] = {
    r"50\.25%": "v1 任务集（骨架比 0.52、日期在过去）",
    r"83\.00%": "v1 任务集",
    r"78\.75%": "v1 任务集",
    r"49\.25%": "v1 任务集",
    r"68\.5%": "污染数据",
    r"95\.0%": "污染数据（1292/2509 训练样本与评测逐字相同）",
    r"95\.00%": "污染数据",
    r"10\.0%": "污染数据",
    r"89\.74%": "v1 任务集（agent 评测）",
    r"77\.78%": "v1 任务集（agent 评测）",
    r"100\.00%": "v1 任务集（agent 评测）",
    r"94\.87%": "v1 任务集（agent 评测）",
    r"53\.85%": "v1 任务集（agent 评测）",
    r"94\.0%": "污染数据的 seen 组",
    r"97\.0%": "污染数据的 unseen 组",
    r"±10pp": "统计判据错误（CI 重叠不是显著性判据）",
}

#: 作废标注词（出现在数字附近即视为已标注）
MARKERS = (
    "作废", "废弃", "已作废", "不要使用", "不可引用", "不可再引用",
    "等 v3", "待重跑", "待确认", "暂缓", "暂不要写", "⚠️",
    "历史", "附录", "旧版", "推翻", "已修正", "已弃用",
    # ⚠️ 下面这些是**同表行里的状态列**。
    # 实测踩过：表格写成 `| 模型 | 95.00% | 污染数据 |` ——
    # 「污染数据」就在同一行，本身就是标注，但早期判据不认它，
    # 于是报了 80 处假违规。判据应认「任何说明该数字不可用的词」。
    "污染", "已污染", "旧任务集", "v1 任务集", "跑在 v1",
    "不再有效", "superseded", "deprecated",
)

#: 默认扫描的文档
DOC_GLOBS = ("docs/**/*.md",)

#: 文档级作废声明的判据：文首若出现这些强标记，整份文档视为已标注。
#:
#: ⚠️ 为什么需要这一层：实测给 `agent_eval_report.md` 加了文首横幅
#: 「本报告的全部数字**已作废**」，但数字在 L111+，而附近的 3 行窗口里
#: 没有标记 —— 于是检查器仍报 13 处违规。
#: **一份文档声明了整体作废，就不该逐个数字再标一遍。**
DOC_LEVEL_MARKERS = (
    "本报告的全部数字", "本报告的数字**已作废", "本文件在",
    "全部数字已作废", "整体作废", "本稿的数字全部作废",
    "本报告的数字存在数据污染", "结论需重新验证",
    # ⚠️ 措辞不统一是实测踩过的坑：`resume_entries.md` 写的是
    # 「旧版的所有性能数字已作废」、`interview_prep.md` 写的是
    # 「本稿的**数字全部作废**」—— 两句都对，但早期列表只认后者，
    # 于是前者被漏掉、误报 17 处违规。
    # **不要靠单一措辞识别「整份作废」**，要把常见写法列全。
    "所有性能数字已作废", "数字全部作废", "的数字全部作废",
    "数字全部作废", "暂不要写", "等 v3 重跑", "待 v3 重跑",
    "判据错误", "有两处", "作废，分析框架",
)


def doc_is_deprecated(lines: list[str], *, head: int = 60) -> bool:
    """文首是否声明了「整份文档已作废」。

    >>> doc_is_deprecated(["# 标题", "> # ⚠️ 本报告的全部数字已作废"])
    True
    >>> doc_is_deprecated(["# 标题", "正文"])
    False
    """
    return any(
        m in ln for ln in lines[:head] for m in DOC_LEVEL_MARKERS
    )


def line_has_marker(lines: list[str], idx: int, *, window: int = 3) -> bool:
    """检查第 ``idx`` 行及其前后 ``window`` 行里是否有作废标注。

    ⚠️ **窗口包含本行** —— 表格里状态列与数字常在同一行
    （``| sft_only | 95.00% | 污染数据 |``），不认本行会大量误报。

    >>> line_has_marker(["正常一行", "**已作废** 50.25%", "另一行"], 1)
    True
    >>> line_has_marker(["| a | 50.25% | 污染数据 |"], 0)
    True
    >>> line_has_marker(["50.25%"], 0)
    False
    >>> line_has_marker(["50.25%", "", "此结果不可引用"], 0)
    True
    """
    lo = max(0, idx - window)
    hi = min(len(lines), idx + window + 1)
    for ln in lines[lo:hi]:
        if any(m in ln for m in MARKERS):
            return True
    return False


def check_file(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    # 文档级声明：文首说「整份作废」就不必逐个数字再标
    doc_dep = doc_is_deprecated(lines)
    hits: list[dict] = []
    for i, ln in enumerate(lines):
        for pat, why in STALE_NUMBERS.items():
            if re.search(pat, ln):
                hits.append({
                    "file": str(path),
                    "line": i + 1,
                    "number": pat.replace("\\", ""),
                    "why": why,
                    "annotated": doc_dep or line_has_marker(lines, i),
                    "annotated_by": ("文档级声明" if doc_dep else "就近标注"),
                    "text": ln.strip()[:96],
                })
    return hits


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="检查文档里未标注的作废数字")
    ap.add_argument("--root", default=None, help="仓库根（默认自动推断）")
    ap.add_argument("--strict", action="store_true",
                    help="有未标注的作废数字时返回非 0")
    ap.add_argument("--show-ok", action="store_true", help="也打印已标注的")
    args = ap.parse_args(argv)

    root = Path(args.root) if args.root else _REPO
    files: list[Path] = []
    for g in DOC_GLOBS:
        files.extend(sorted(root.glob(g)))
    files = [f for f in files if f.is_file()]

    if not files:
        print(f"  ✗ 在 {root} 下没找到文档")
        return 1

    all_hits: list[dict] = []
    for f in files:
        all_hits.extend(check_file(f))

    unannotated = [h for h in all_hits if not h["annotated"]]
    annotated = [h for h in all_hits if h["annotated"]]

    print(f"  扫描 {len(files)} 个文档，命中 {len(all_hits)} 处作废数字引用")
    print(f"    已标注 {len(annotated)} 处 / **未标注 {len(unannotated)} 处**")
    print()

    if unannotated:
        print("### ❌ 未标注的作废数字（会误导读者）")
        print()
        by_file: dict[str, list[dict]] = {}
        for h in unannotated:
            by_file.setdefault(Path(h["file"]).name, []).append(h)
        for name, hs in sorted(by_file.items()):
            print(f"  **{name}**（{len(hs)} 处）")
            for h in hs[:6]:
                print(f"    L{h['line']}: `{h['number']}` — {h['why']}")
                print(f"        {h['text']}")
            if len(hs) > 6:
                print(f"    … 另有 {len(hs)-6} 处")
            print()

    if args.show_ok and annotated:
        print("### ✅ 已标注（正确引用，例如「我推翻了 95% 那个结果」）")
        print()
        for h in annotated[:12]:
            print(f"  {Path(h['file']).name} L{h['line']}: {h['number']}")
        if len(annotated) > 12:
            print(f"  … 另有 {len(annotated)-12} 处")
        print()

    if not all_hits:
        print("  ✅ 文档里没有出现已知的作废数字")
        return 0

    if unannotated:
        print("  修法：在该数字附近加作废标注，或把该节移到「历史 / 附录」小节下。")
        print(f"  （标注词集合：{'、'.join(MARKERS[:8])} …）")
        return 1 if args.strict else 0
    print("  ✅ 所有作废数字都带了标注")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
