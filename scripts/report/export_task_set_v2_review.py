"""导出 v2 高质量任务集的抽查清单（Markdown + DOCX）。

与 ``export_task_set_review.py``（v1/400 任务用）的区别：
本脚本面向**显式定义的 100 个任务**，突出「每个任务的独立考查点」，
并把约束检查结果放在最前面 —— 因为 v2 的价值就在于这些约束成立。
"""

from __future__ import annotations

import collections
import datetime
import json
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

for _p in Path(__file__).resolve().parents:
    if _p.name == "项目2":
        DESKTOP = _p.parent
        break
else:
    DESKTOP = Path.home() / "Desktop"
OUT_DIR = Path(__file__).resolve().parent

CJK, LATIN, MONO = "Microsoft YaHei", "Microsoft YaHei", "Consolas"
ACCENT = RGBColor(0x1F, 0x3B, 0x73)
WARN = RGBColor(0xA3, 0x1D, 0x1D)
OK = RGBColor(0x14, 0x60, 0x2E)
MUTED = RGBColor(0x55, 0x55, 0x55)


def style_run(run, *, size=10, bold=False, color=None, mono=False):
    run.font.size = Pt(size)
    run.font.bold = bold
    if color is not None:
        run.font.color.rgb = color
    name = MONO if mono else LATIN
    run.font.name = name
    rpr = run._element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.append(rf)
    for k in ("w:ascii", "w:hAnsi"):
        rf.set(qn(k), name)
    rf.set(qn("w:eastAsia"), CJK)
    return run


def emit_bold(p, text, *, size=10, bold=False, color=None, mono=False):
    for i, seg in enumerate(re.split(r"(\*\*)", text)):
        if seg == "**" or not seg:
            continue
        style_run(p.add_run(seg), size=size, bold=bold or (i % 2 == 1),
                  color=color, mono=mono)


def para(doc, text="", *, size=10, bold=False, color=None, align=None,
         space_before=2, space_after=4, mono=False, indent=0):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    if indent:
        p.paragraph_format.left_indent = Cm(indent)
    if text:
        emit_bold(p, text, size=size, bold=bold, color=color, mono=mono)
    return p


def heading(doc, text, *, level=1):
    sizes = {1: 14, 2: 12, 3: 10.5}
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(13 if level == 1 else 9)
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.keep_with_next = True
    style_run(p.add_run(text), size=sizes.get(level, 11), bold=True, color=ACCENT)
    return p


def table(doc, headers, rows, *, widths=None, size=8):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        c = t.rows[0].cells[i]
        c.text = ""
        style_run(c.paragraphs[0].add_run(h), size=size, bold=True)
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            emit_bold(cells[i].paragraphs[0], str(val), size=size)
    if widths:
        for r in t.rows:
            for i, w in enumerate(widths):
                r.cells[i].width = Cm(w)
    for r in t.rows:
        r._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
    return t


def build():
    from src.data.task_manifest_v2 import (
        BASE_DATE_V2, PASSENGERS_V2, load_tasks_v3, skeleton, verify_task_set,
    )
    from src.data.task_set_v2_specs import build_specs

    specs = build_specs()
    ts = load_tasks_v3()
    rep = verify_task_set(specs)

    dates = sorted({d for s in specs for d in re.findall(r"\d{4}-\d{2}-\d{2}", s.user)})
    bks = [m for s in specs for m in re.findall(r"BK[0-9A-F]{6}", s.user)]
    nms = [n for s in specs for n in PASSENGERS_V2 if n in s.user]

    doc = Document()
    s = doc.sections[0]
    s.top_margin = Cm(1.8)
    s.bottom_margin = Cm(1.8)
    s.left_margin = Cm(1.7)
    s.right_margin = Cm(1.7)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    style_run(p.add_run("v2 高质量任务集（100 个）"), size=19, bold=True, color=ACCENT)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    style_run(p.add_run("每个任务独立设计：实体全新、日期未来、句式不重复"),
              size=11, color=MUTED)

    para(doc)
    heading(doc, "§1 与 v1 的对照（这是重做的理由）", level=1)
    table(doc, ["判据", "v1（旧 100）", "v2（新 100）", "为什么重要"], [
        ["骨架/任务比", "**0.52**", "**0.99**",
         "v1 是「52 种任务换 100 种说法」；v2 每个任务一个独立考查点"],
        ["multi_tool 骨架比", "**0.11**", "**1.00**",
         "v1 的 9 个 multi_tool 只有 1 种句式"],
        ["multi_tool 要求的工具数", "**1**（只有 search_flights）",
         "**2**（get_weather + search_flights）",
         "v1 名叫多工具却只测一个，观测给的天气从不参与判定"],
        ["订单号池", "8 个，`BK7F2A9C` 出现 **7 次**", "32 个，**全部唯一**",
         "否则模型答对可能只是记住了那个订单号"],
        ["乘客名池", "6 个", "18 个，**全部唯一**", "同上"],
        ["日期区间", "2025-07 ~ 2025-08（**全是过去**）",
         "2026-11 ~ 2027-01（**全部未来**）", "让模型查一年前的航班是无效请求"],
        ["seen/unseen 骨架交集", "**6 个（占 unseen 26%）**", "**0**",
         "v1 的「unseen 泛化」测的是同一批任务类型的另一种说法"],
    ], widths=[3.4, 3.6, 3.8, 5.6], size=8)

    para(doc)
    heading(doc, "§2 约束检查结果", level=1)
    if not rep["violations"]:
        para(doc, "✅ 全部结构性约束通过（由 verify_task_set 自动检查，"
                  "并已固化为 7 个单元测试）", color=OK, bold=True)
    else:
        para(doc, f"⚠️ {len(rep['violations'])} 条违规", color=WARN, bold=True)
        for x in rep["violations"]:
            para(doc, f"  - {x}", size=9)

    para(doc)
    table(doc, ["指标", "值"], [
        ["任务数", f"{len(ts)}"],
        ["骨架总数 / 骨架比", f"{rep['skeleton_total']} / **{rep['skeleton_ratio']}**"],
        ["订单号使用 / 唯一", f"{len(bks)} / {len(set(bks))}"],
        ["乘客名使用 / 唯一", f"{len(nms)} / {len(set(nms))}"],
        ["日期数 / 区间", f"{len(dates)} / {dates[0]} ~ {dates[-1]}"],
        ["基准日（相对日期换算起点）", str(BASE_DATE_V2)],
        ["seen / unseen", "%d / %d" % (
            sum(1 for t in ts if t.visibility == "seen"),
            sum(1 for t in ts if t.visibility == "unseen"))],
    ], widths=[5.6, 7.0], size=9)

    para(doc)
    heading(doc, "§3 分类别指标", level=1)
    table(doc, ["类别", "n", "骨架", "骨架比", "签名数", "轮次", "难度"], [
        [c, m["n"], m["skeletons"], f"**{m['ratio']}**", m["signatures"],
         "/".join(str(x) for x in sorted({len(t.turns) for t in ts
                                          if t.category == c})),
         next((t.difficulty for t in ts if t.category == c), "")]
        for c, m in rep["by_category"].items()
    ], widths=[3.8, 1.2, 1.4, 1.6, 1.6, 1.6, 1.6], size=8.5)

    doc.add_page_break()
    heading(doc, "§4 全部 100 个任务", level=1)
    para(doc, "按类别分组。这样列出的目的是让你能逐个检查「这个任务的考查点是否独立」。",
         color=MUTED, size=9)

    for cat in sorted({s.category for s in specs}):
        group = [t for t in ts if t.category == cat]
        heading(doc, f"{cat}（{len(group)} 个）", level=2)
        rows = []
        for t in group:
            exp = ("（应先追问，不调工具）" if t.expected.tool is None
                   else f"{t.expected.tool}({', '.join(f'{k}={v}' for k, v in (t.expected.args or {}).items())})")
            extra = [x for x in t.tags if x in ("get_weather", "tool_chaining")]
            rows.append([
                t.task_id.replace("taskv2_", ""),
                t.visibility,
                t.turns[0].content,
                exp,
                ("+" + ",".join(extra)) if extra else "",
            ])
        table(doc, ["ID", "可见性", "用户首轮", "期望调用", "额外要求"], rows,
              widths=[1.2, 1.4, 6.6, 5.2, 2.0], size=7.5)

    doc.save(OUT_DIR / "任务集v2抽查清单.docx")

    # 同时导一份 JSON 便于程序化核对
    (OUT_DIR / "任务集v2抽查清单.json").write_text(json.dumps({
        "num": len(ts),
        "report": rep,
        "dates": dates,
        "booking_ids_unique": len(set(bks)),
        "tasks": [
            {"task_id": t.task_id, "category": t.category,
             "visibility": t.visibility, "difficulty": t.difficulty,
             "user": t.turns[0].content,
             "tool": t.expected.tool, "args": t.expected.args,
             "num_turns": len(t.turns), "tags": list(t.tags)}
            for t in ts
        ],
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    return OUT_DIR / "任务集v2抽查清单.docx"


if __name__ == "__main__":
    print("已生成:", build())
