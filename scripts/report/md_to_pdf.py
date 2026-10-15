"""把 Markdown 转成带中文字体的 DOCX，再交给 LibreOffice 转 PDF。

## 为什么需要它

项目里有多份 Markdown 文档（交接文档、报告、面试稿）。每次要交付 PDF 时
重写一遍排版代码既浪费又容易不一致 —— 尤其是中文字体这块，
**"Noto Sans CJK SC" 这类字体名在本机不存在，会静默渲染成方块**。

## 支持的 Markdown 子集

* ``#`` / ``##`` / ``###`` 标题
* 段落（含 ``**粗体**`` 行内标记）
* ``-`` / ``*`` 无序列表，``1.`` 有序列表
* ```` ``` ```` 围栏代码块（等宽字体）
* ``| ... |`` 表格（含表头分隔行）
* ``>`` 引用块
* ``---`` 分隔线

## 用法

    python scripts/report/md_to_pdf.py docs/HANDOFF_next_steps.md
    python scripts/report/md_to_pdf.py docs/*.md --out-dir dist

不支持的语法会被当作普通段落，不会丢内容 —— 宁可排版差，不可少字。
"""

from __future__ import annotations

import argparse
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

# ⚠️ 必须用本机**真实存在**的字体名。早期写 "Noto Sans CJK SC" + "DejaVu Sans"，
# 两者本机都没有，LibreOffice 转换时报 missingFonts 并把中文渲染成方块。
CJK = "Microsoft YaHei"
LATIN = "Microsoft YaHei"
MONO = "Consolas"

ACCENT = RGBColor(0x1F, 0x3B, 0x73)
MUTED = RGBColor(0x55, 0x55, 0x55)
CODEBG = RGBColor(0xF4, 0xF4, 0xF4)


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


def emit_inline(p, text, *, size=10, bold=False, color=None, mono=False):
    """解析 ``**粗体**`` 与 `` `code` ``。

    ⚠️ 两个坑（都踩过）：

    1. ``re.split`` 带捕获组会把**分隔符本身保留**在结果里。必须显式丢弃，
       否则星号/反引号会被当成正文打印（实测文档里多出 88 处星号）。
    2. **粗体状态必须用 toggle 维护**，不能用「片段等于 ``**``」去判断 ——
       因为分隔符已经被丢弃了，那个判断永远为假，粗体就丢了。
       早期版本正是这样写的，结果 4 处 ``**`` 原样印在 PDF 上。
    """
    is_bold = bold
    is_mono = mono
    for seg in re.split(r"(\*\*|`)", text):
        if seg == "**":
            is_bold = not is_bold
            continue
        if seg == "`":
            is_mono = not is_mono
            continue
        if not seg:
            continue
        style_run(p.add_run(seg), size=size, bold=is_bold, color=color, mono=is_mono)


def para(doc, text="", *, size=10, bold=False, color=None, align=None,
         before=2, after=4, indent=0, mono=False):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(after)
    if indent:
        p.paragraph_format.left_indent = Cm(indent)
    if text:
        emit_inline(p, text, size=size, bold=bold, color=color, mono=mono)
    return p


def heading(doc, text, *, level=1):
    sizes = {1: 15, 2: 12.5, 3: 11}
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14 if level == 1 else 10)
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.keep_with_next = True
    # ⚠️ 标题也要走行内解析 —— 标题里常出现 ``**强调**``
    # （例如「阶段 B：让训练语料匹配 v3（**B0 是阻塞项**）」）。
    # 早期直接把原文交给 style_run，结果 4 个星号原样印在 PDF 上。
    emit_inline(p, text, size=sizes.get(level, 11), bold=True, color=ACCENT)
    return p


def code_block(doc, lines):
    """代码块：等宽 + 深色文字 + 左缩进。

    ⚠️ 不要用浅灰当**文字色** —— 我一度把 ``CODEBG = 0xF4F4F4``（本该是背景色）
    传给了文字色，结果代码块几乎看不见（渲染出来才发现的）。
    """
    if not lines:
        return
    for ln in lines:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.left_indent = Cm(0.4)
        style_run(p.add_run(ln if ln else " "), size=8.5, mono=True,
                  color=RGBColor(0x1A, 0x1A, 0x1A))
    para(doc, "", after=4)


def table(doc, rows, *, size=8.5):
    if not rows:
        return
    ncol = max(len(r) for r in rows)
    t = doc.add_table(rows=0, cols=ncol)
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for ci in range(ncol):
            val = row[ci] if ci < len(row) else ""
            cells[ci].text = ""
            emit_inline(
                cells[ci].paragraphs[0],
                val,
                size=size,
                bold=(ri == 0),
            )
    for r in t.rows:
        r._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
    return t


def _split_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def _is_table_sep(line: str) -> bool:
    s = line.strip()
    return bool(s) and set(s) <= set("|-: ") and "-" in s


def convert(md_path: Path, out_dir: Path) -> Path:
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = Cm(1.9)
    sec.bottom_margin = Cm(1.9)
    sec.left_margin = Cm(2.0)
    sec.right_margin = Cm(2.0)

    i = 0
    while i < len(lines):
        ln = lines[i]
        stripped = ln.strip()

        # 围栏代码块
        if stripped.startswith("```"):
            i += 1
            buf = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            code_block(doc, buf)
            continue

        # 表格
        if stripped.startswith("|") and i + 1 < len(lines) and _is_table_sep(lines[i + 1]):
            rows = [_split_row(stripped)]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(_split_row(lines[i]))
                i += 1
            table(doc, rows)
            para(doc, "", after=4)
            continue

        # 分隔线
        if stripped in ("---", "***", "___"):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(6)
            style_run(p.add_run("─" * 46), size=9, color=MUTED)
            i += 1
            continue

        # 标题
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            heading(doc, m.group(2), level=len(m.group(1)))
            i += 1
            continue

        # 引用块
        if stripped.startswith(">"):
            buf = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip().lstrip(">").strip())
                i += 1
            para(doc, " ".join(x for x in buf if x), size=9.5,
                 color=MUTED, indent=0.5)
            continue

        # 列表
        m = re.match(r"^(\s*)[-*]\s+(.*)$", ln)
        if m:
            p = doc.add_paragraph(style="List Bullet")
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(2)
            emit_inline(p, m.group(2), size=10)
            i += 1
            continue
        m = re.match(r"^(\s*)\d+\.\s+(.*)$", ln)
        if m:
            p = doc.add_paragraph(style="List Number")
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(2)
            emit_inline(p, m.group(2), size=10)
            i += 1
            continue

        # 空行
        if not stripped:
            i += 1
            continue

        # 普通段落
        para(doc, stripped)
        i += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    docx_path = out_dir / (md_path.stem + ".docx")
    doc.save(docx_path)
    return docx_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Markdown → DOCX（带中文字体）")
    ap.add_argument("inputs", nargs="+", help="Markdown 文件")
    ap.add_argument("--out-dir", default=None, help="输出目录（默认与源文件同目录）")
    args = ap.parse_args(argv)

    for src in args.inputs:
        md = Path(src)
        if not md.exists():
            print(f"  ✗ 不存在: {md}")
            continue
        out_dir = Path(args.out_dir) if args.out_dir else md.parent
        out = convert(md, out_dir)
        print(f"  ✓ {md} → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
