"""把专家两份 PDF 里**可恢复**的 ASCII 骨架整理成一份清单，供核对编号。

## 为什么需要这个

专家的两份 PDF 都由 ReportLab + Helvetica 生成，**未嵌入中文字体**，
中文在导出时就被替换成了占位符字符 —— 这是**导出期损坏**，
信息不在文件里，下游换任何字体重新解析都取不回来（实测：两份文件都
没有 `/ToUnicode` 映射，`/BaseFont` 只有 Helvetica/Symbol/ZapfDingbats）。

所以本文档做两件事：
1. 把我能可靠恢复的 ASCII 片段与结构列出来（含「n 串字符数」= 丢失的中文字数，
   它保留了原文的长度信息，可用于推断原词长度）
2. 把我**推断的内容**与**置信度**并排列出，请专家逐条核对 —— 这比让我猜更可靠

## 输出

一份 DOCX（校验后转 PDF），带「核对栏」空列，专家可直接勾选。
"""

from __future__ import annotations

import re
import zlib
import base64
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

# ⚠️ 桌面路径不能假设是 Path.home()/Desktop —— 本机用户目录被重定向到
# D:\HuaweiMoveData\Users\<user>，而 Path.home() 返回 C:\Users\<user>。
# 所以显式解析，并允许用 --desktop 覆盖。
_HERE = Path(__file__).resolve()


def _find_desktop() -> Path:
    """定位桌面：优先用本文件相对位置推断（本仓库就在桌面下）。"""
    for parent in _HERE.parents:
        if parent.name == "项目2":
            return parent.parent
    cand = Path.home() / "Desktop"
    return cand


DESKTOP = _find_desktop()
OUT_DIR = _HERE.parent
DOCX = OUT_DIR / "专家问题核对清单.docx"

CJK = "Microsoft YaHei"
LATIN = "Microsoft YaHei"
MONO = "Consolas"

ACCENT = RGBColor(0x1F, 0x3B, 0x73)
WARN = RGBColor(0xA3, 0x1D, 0x1D)
OK = RGBColor(0x14, 0x60, 0x2E)
MUTED = RGBColor(0x55, 0x55, 0x55)
TODO = RGBColor(0xB0, 0x50, 0x00)


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
    rf.set(qn("w:ascii"), name)
    rf.set(qn("w:hAnsi"), name)
    rf.set(qn("w:eastAsia"), CJK)
    return run


def emit_bold(p, text, *, size=10, bold=False, color=None, mono=False):
    """解析正文里的 **粗体**。注意 re.split 带捕获组会保留分隔符，必须丢弃。"""
    for i, seg in enumerate(re.split(r"(\*\*)", text)):
        if seg == "**" or not seg:
            continue
        style_run(p.add_run(seg), size=size, bold=bold or (i % 2 == 1),
                  color=color, mono=mono)


def para(doc, text="", *, size=10, bold=False, color=None, align=None,
         space_before=2, space_after=4, indent=0, mono=False):
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


def table(doc, headers, rows, *, widths=None, size=8.5):
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


def bullet(doc, text, *, size=10, color=None, bold_prefix=None):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(2)
    if bold_prefix:
        style_run(p.add_run(bold_prefix), size=size, bold=True)
    emit_bold(p, text, size=size, color=color)
    return p


# ---------------------------------------------------------------- PDF 解析
def _a85(d: bytes) -> bytes:
    s = d.strip()
    if s.startswith(b"<~"):
        s = s[2:]
    i = s.find(b"~>")
    if i >= 0:
        s = s[:i]
    return base64.a85decode(s, adobe=False, ignorechars=b" \t\r\n\v\f")


def extract_fragments(pdf: Path) -> list[str]:
    """取出所有 Tj 字面串，并把 n 串替换成「(N字)」以保留长度信息。"""
    raw = pdf.read_bytes()
    chunks, pos = [], 0
    while True:
        s = raw.find(b"stream", pos)
        if s < 0:
            break
        e = raw.find(b"endstream", s)
        if e < 0:
            break
        chunks.append(raw[s + 6:e].lstrip(b"\r\n"))
        pos = e + 9

    out: list[str] = []
    for c in chunks:
        tx = None
        for chain in (lambda d: zlib.decompress(_a85(d)), lambda d: zlib.decompress(d)):
            try:
                tx = chain(c).decode("latin-1", "replace")
                break
            except Exception:
                continue
        if not tx:
            continue
        for L in re.findall(r"\((?:\\.|[^()\\])*\)\s*Tj", tx):
            body = L[:L.rfind(")")][1:]
            body = body.replace("\\(", "(").replace("\\)", ")").replace("\\\\", "\\")
            out.append(body)
    return out


def recover_ascii(pdf: Path) -> str:
    frags = extract_fragments(pdf)
    line = "".join(frags)
    # n 串 → (N字)，保留原文长度信息
    return re.sub(r"n+", lambda m: "〔%d字〕" % len(m.group(0)), line)


def build() -> Path:
    doc = Document()
    s = doc.sections[0]
    s.top_margin = Cm(1.9)
    s.bottom_margin = Cm(1.9)
    s.left_margin = Cm(1.9)
    s.right_margin = Cm(1.9)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    style_run(p.add_run("专家问题核对清单"), size=19, bold=True, color=ACCENT)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    style_run(p.add_run("请核对「我的推断」列是否与您的原意一致"), size=11, color=MUTED)

    para(doc)

    heading(doc, "§0 关于字体问题（需要您在源头重出）", level=1)
    para(doc, "您发来的两份 PDF 存在同一个**导出期损坏**，我这边无法修复：")
    table(
        doc,
        ["文件", "生成方式", "中文字体", "ToUnicode 映射", "后果"],
        [
            ["项目问题清单.pdf", "ReportLab", "❌ 未嵌入", "❌ 无", "中文被替换为占位符"],
            ["后续工作计划分析.pdf", "ReportLab", "❌ 未嵌入", "❌ 无", "同上"],
            ["（我的交付物）回应文档_修订版v2.pdf", "LibreOffice", "✅ Microsoft YaHei",
             "✅ 有", "文本层正常，可提取可检索"],
        ],
        widths=[5.6, 2.6, 3.0, 2.6, 4.0],
    )
    para(doc)
    para(doc, "**为什么下游修不了**：占位符字符是**导出时写进内容流**的，"
              "原中文字形从来没有进入文件。所以换任何字体重新解析都取不回来 —— "
              "**只能从源头重出**（Word / Markdown / 邮件导出时选用带中文的字体）。")
    para(doc, "**好消息**：占位符的**个数等于原文的字数**，所以我保留了长度信息"
              "（下面用〔N字〕标注），它能帮助判断原词长度。", color=OK)

    doc.add_page_break()

    heading(doc, "§1 为什么不再附「原始 dump」", level=1)
    para(doc, "我试过把两份 PDF 的可恢复片段逐个列出，但结果**没有实用价值**：")
    bullet(doc, "占位符虽然保留了字数，但**词义完全丢失** —— 例如"
                "「〔6字〕DPO〔3字〕SFT〔11字〕」无法判断原句是在肯定还是否定")
    bullet(doc, "剩余字节大多是**双编码残留**（如 \\265\\266\\267 这类八进制转义），"
                "属于字体子集映射的噪声，不是可读文本")
    bullet(doc, "把它印成一张长表只会让读者花时间去猜，而不是核对")
    para(doc)
    para(doc, "**所以我把重心放在 §2 的逐条核对表** —— "
              "那是我真正需要的输入，也只需要您打勾。", color=TODO)

    doc.add_page_break()

    doc.add_page_break()

    heading(doc, "§2 逐条核对表（我需要的输入）", level=1)
    para(doc, "「核对」列请您填 ✓ / ✗ / 部分。置信度「中」的条目我可能读错了。",
         color=TODO, size=9.5)
    para(doc)

    rows = [
        ["1", "结论分层：哪些现在就能定，哪些需补实验", "高", ""],
        ["2", "样本量：100 → 400 任务，CI 从 ±10pp 压到 ±5pp", "高", ""],
        ["3", "剩余工作清单：轨迹级 KL 判据、β 扫描、LLM grader", "高", ""],
        ["4", "β 扫描区间应为 {0.5, 1.0, 2.0, 5.0}（我做的是 0.02~0.2）", "高", ""],
        ["5", "lr：SFT 是 DPO 的 10 倍；DPO 应取 5e-5 / 1e-4，"
              "因为 sigmoid loss 梯度远小于交叉熵", "高", ""],
        ["6", "7B 实验的必要性（且我写的「证明流水线可扩展」是错误理由）", "中", ""],
        ["7", "实测 vs 推断的方法论；prompt / task_id / 51.5%", "中", ""],
        ["8", "项目定位：Qwen2.5-1.5B + DPO + GitHub 作为作品集是否够", "中", ""],
        ["9", "剩余工作计划（paired_significance.py / measure_kl_drift.py）", "高", ""],
        ["10", "样本量 15-20 / 10 / MVP；recovery 扩到 n≥20", "高", ""],
        ["11", "领域定位：缺 MCP / ReAct / Plan-and-Execute / Tool Learning", "中", ""],
        ["12", "优先级重排；且我的文档里有「X pp」未填占位符", "高", ""],
    ]
    table(doc, ["#", "我的推断", "置信度", "核对（请填）"], rows,
          widths=[0.9, 10.4, 1.8, 3.2], size=8.5)

    para(doc)
    heading(doc, "§3 我特别不确定的两条", level=1)
    para(doc, "这两条会显著改变我的结论，所以单独列出：")
    bullet(doc, "我读到的是「DPO 与 SFT 相差 33pp、CI ≈ ±1pp，"
                "因此该差距是可靠的」，并指出我写「未显著」是错的。"
                "—— 若理解正确，我会把「DPO 使准确率可靠下降 33.75pp」"
                "写成成立的负面结论，而不是「待检验」。", bold_prefix="第 5 条相关：")
    bullet(doc, "我读到的是质疑「项目名叫 agent-optimization，"
                "但实现里没有 ReAct / Plan-and-Execute / MCP / Tool Learning」；"
                "我的实现是脚本驱动的固定编排，模型不自主决定控制流。"
                "—— 若理解正确，我倾向「改项目名 + 明确声明边界」而不是补架构。",
           bold_prefix="第 11 条相关：")

    para(doc)
    heading(doc, "§4 我的请求", level=1)
    para(doc, "请提供以下任一形式即可，我立刻校正并重出文档：")
    bullet(doc, "原始 Word / Markdown / 纯文本（最理想）")
    bullet(doc, "或：把 §3 表格的「核对」列填好回传")
    bullet(doc, "或：只针对 §4 的两条做一句话确认 / 否认")

    doc.save(DOCX)
    return DOCX


if __name__ == "__main__":
    print(f"已生成: {build()}")
