"""生成「项目后续工作计划」PDF 的源文档（DOCX），随后用 LibreOffice 转 PDF。

这个脚本本身也是可复现的交付物：计划里的数字全部来自 experiments/ 下的
真实报告，改数字只需改这里的常量并重跑。
"""

from __future__ import annotations

import re

from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

OUT_DIR = Path(__file__).resolve().parent
DOCX = OUT_DIR / "项目后续工作计划.docx"

# 中文字体：必须用**这台机器上真实存在**的字体名。
#
# ⚠️ 踩过的坑：最初写 "Noto Sans CJK SC" + "DejaVu Sans"，两者本机都没有，
# LibreOffice 转换时报 missingFonts 并把中文渲染成方块。
# 本机实际可用：微软雅黑(msyh.ttc) / 黑体(simhei.ttf) / 等线(Deng.ttf)。
# 另外必须同时设 ascii/hAnsi 与 eastAsia，否则中文仍会退化
# （这个项目在 matplotlib 上已经踩过一次同类问题）。
CJK = "Microsoft YaHei"
LATIN = "Microsoft YaHei"
MONO = "Consolas"

ACCENT = RGBColor(0x1F, 0x3B, 0x73)
WARN = RGBColor(0xA3, 0x1D, 0x1D)
OK = RGBColor(0x14, 0x60, 0x2E)
MUTED = RGBColor(0x55, 0x55, 0x55)


def style_run(run, *, size: int = 10, bold: bool = False, color=None, mono: bool = False):
    run.font.size = Pt(size)
    run.font.bold = bold
    if color is not None:
        run.font.color.rgb = color
    name = MONO if mono else LATIN
    run.font.name = name
    rpr = run._element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        from docx.oxml import OxmlElement

        rf = OxmlElement("w:rFonts")
        rpr.append(rf)
    rf.set(qn("w:ascii"), name)
    rf.set(qn("w:hAnsi"), name)
    rf.set(qn("w:eastAsia"), CJK)
    return run



def _emit_with_bold(p, text, *, size, bold, color):
    """把正文里的 ``**粗体**`` 解析成真正的粗体 run。

    早期版本只在 table() 里做了这个解析，正文里的 ``**…**`` 会**原样印出星号**
    （实测回应文档 v2 里有 88 处、工作计划里有 40 处）。这里统一处理。
    """
    import re as _re

    for i, seg in enumerate(_re.split(r"(\*\*)", text)):
        if not seg:
            continue
        style_run(p.add_run(seg), size=size, bold=bold or (i % 2 == 1), color=color)

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
        _emit_with_bold(p, text, size=size, bold=bold, color=color)
    return p


def rich(doc, segments, *, size=10, space_before=2, space_after=4, indent=0):
    """segments: list of (text, bold, color)"""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    if indent:
        p.paragraph_format.left_indent = Cm(indent)
    for seg in segments:
        text, bold = seg[0], seg[1]
        color = seg[2] if len(seg) > 2 else None
        _emit_with_bold(p, text, size=size, bold=bold, color=color)
    return p


def heading(doc, text, *, level=1):
    sizes = {1: 14, 2: 12, 3: 11}
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14 if level == 1 else 10)
    p.paragraph_format.space_after = Pt(5)
    # 标题不能孤立在页尾（与下一段同页）
    p.paragraph_format.keep_with_next = True
    style_run(p.add_run(text), size=sizes.get(level, 11), bold=True, color=ACCENT)
    return p


def table(doc, headers, rows, *, widths=None, size=9, footer=None):
    """footer: 可选的汇总行（加粗）。用它而不是在 rows 里塞空行 —— 空行会渲染成
    一个难看的空白行。"""
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ""
        style_run(hdr[i].paragraphs[0].add_run(h), size=size, bold=True)
    for row in list(rows) + ([footer] if footer else []):
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            txt = str(val)
            # 支持格内 **粗体**。注意 re.split 带捕获组会**保留分隔符**，
            # 必须显式丢弃，否则 '**' 会被当成正文打印出来（踩过这个坑）。
            for seg_i, seg in enumerate(re.split(r"(\*\*)", txt)):
                if seg == "**" or not seg:
                    continue
                style_run(
                    cells[i].paragraphs[0].add_run(seg),
                    size=size,
                    bold=(seg_i % 2 == 1),
                )
    if widths:
        for r in t.rows:
            for i, w in enumerate(widths):
                r.cells[i].width = Cm(w)

    # 行内不断页（避免一行文字被拆到两页）
    from docx.oxml import OxmlElement

    for r in t.rows:
        trPr = r._tr.get_or_add_trPr()
        cant = OxmlElement("w:cantSplit")
        trPr.append(cant)

    # 最后一行与紧随其后的段落同页 —— 否则表格尾行会孤零零留在下一页
    for c in t.rows[-1].cells:
        for pp in c.paragraphs:
            pp.paragraph_format.keep_with_next = True
    return t


def bullet(doc, text, *, size=10, color=None, bold_prefix=None):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(2)
    if bold_prefix:
        style_run(p.add_run(bold_prefix), size=size, bold=True)
    _emit_with_bold(p, text, size=size, bold=False, color=color)
    return p


def build() -> Path:
    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = Cm(2.0)
    sec.bottom_margin = Cm(2.0)
    sec.left_margin = Cm(2.2)
    sec.right_margin = Cm(2.2)

    # ---------------- 封面标题 ----------------
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    style_run(p.add_run("agent-optimization 项目：后续工作计划"), size=19, bold=True, color=ACCENT)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    style_run(p.add_run("针对问题清单的逐条回应与执行方案"), size=11, color=MUTED)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    style_run(p.add_run("2026-10-06"), size=9.5, color=MUTED)

    para(doc)

    # ---------------- §0 ----------------
    heading(doc, "§0 一页摘要", level=1)
    rich(doc, [
        ("这份文档回应专家提出的 20 条问题，并给出后续工作的优先级与执行方案。"
         "我的总判断：", False),
        ("清单里绝大多数是真问题，其中 3 条是致命的", True, WARN),
        ("，需要修正报告与补做实验；另有约 5 条属于「已发现但未清理干净」"
         "或「需要新增工作才能回答」。", False),
    ])

    rich(doc, [
        ("最重的一条是：", False),
        ("整个项目的叙事目前建立在一个已被我自己判定为污染的数字（SFT 95%）上", True, WARN),
        ("。虽然报告里标注了它不可用，但它仍留在主表中，"
         "导致任何读者第一眼看到的都是这个错误数字。修复方式是重构报告，"
         "主表只保留干净数据。", False),
    ])

    table(
        doc,
        ["优先级", "事项", "工作量", "GPU"],
        [
            ["**P0**", "重构报告：主表只用干净数据，污染数字下沉附录", "我 30 分", "否"],
            ["**P0**", "配对显著性检验（McNemar / 配对 bootstrap）+ 功效分析", "我 40 分", "否"],
            ["**P0**", "补真实 KL(π_θ ‖ π_ref) 漂移监控（替代 logps/chosen）", "我 60 分", "10 分"],
            ["**P0**", "SFT/DPO 超参对齐后重跑（当前 lr 不可比）", "—", "40 分"],
            ["**P1**", "任务集扩到 400（CI 从 ±10pp 压到 ±5pp）", "—", "60 分"],
            ["**P1**", "prompt engineering 对照（不改权重，只改 prompt）", "我 30 分", "20 分"],
            ["**P1**", "β 扫描 [0.02, 0.05, 0.1, 0.2]", "—", "80 分"],
            ["**P2**", "agent 评测补齐 flight_search 多轮变体", "我 60 分", "15 分"],
            ["**P2**", "LLM-based grader 对照 + 人工一致性", "我 60 分", "30 分"],
            ["**P2**", "7B 模型对照（证明流水线可扩展）", "—", "约 3 小时"],
            ["**P2**", "W&B 实验追踪接入", "我 30 分", "否"],
        ],
        footer=["合计", "我自己可完成约 5 小时（无 GPU）", "需租用 GPU 约 6 小时", "≈ ¥10"],
        widths=[1.5, 8.6, 2.9, 1.9],
        size=8,
    )

    # 合计并入表格最后一行：独立成段会被表格挤到下一页成为孤行
    # （试过 keep_with_next，对表格尾部不生效）。

    doc.add_page_break()

    # ---------------- §1 ----------------
    heading(doc, "§1 项目当前状态", level=1)

    heading(doc, "1.1 可信的数字", level=2)
    para(doc, "以下是当前唯一可对外引用的结果（训练数据与评测任务零重合，已脚本验证）：")
    table(
        doc,
        ["实验", "accuracy", "pass^4", "格式正确率", "数据状态"],
        [
            ["baseline（原始基座）", "50.25%", "45.00%", "62.25%", "—"],
            ["sft_v2（干净，但语料覆盖不足）", "78.75%", "77.00%", "98.00%", "✅ 干净"],
            ["**sft_v3（干净 + 覆盖补全）**", "**83.00%**", "**83.00%**", "**100%**", "**✅ 主结果**"],
            ["dpo_sft_v3_b05（β=0.5，从 sft_v3 出发）", "49.25%", "40.00%", "58.80%", "✅ 干净"],
        ],
        widths=[6.0, 2.3, 2.2, 2.3, 2.2],
    )

    heading(doc, "1.2 已作废的数字（必须明确声明）", level=2)
    rich(doc, [
        ("下面三组实验的训练数据里混入了评测任务（", False),
        ("1292 / 2509 = 51.5% 的训练样本，其 prompt 与评测 prompt 逐字相同", True, WARN),
        ("），因此它们的数字不可用于任何对比，包括「DPO 不如 SFT」这个结论本身。", False),
    ])
    table(
        doc,
        ["实验", "accuracy", "问题"],
        [
            ["sft_only", "95.00%", "训练数据含评测任务 → 虚高"],
            ["dpo_joint", "68.50%", "同上"],
            ["dpo_from_sft", "10.00%", "同上 + β=0.1 崩塌"],
        ],
        widths=[4.0, 3.0, 8.0],
    )

    heading(doc, "1.3 agent 多轮评测（sft_v3）", level=2)
    table(
        doc,
        ["指标", "39 个多轮任务", "19 个真需工具链", "50% 故障注入"],
        [
            ["轨迹成功率", "100.00%", "100.00%", "89.74%"],
            ["故障恢复率", "—", "—", "77.78%"],
            ["工具调用有效率", "98.31%", "100.00%", "100.00%"],
            ["成功轨迹平均轮数", "1.54", "2.00", "1.71"],
            ["循环异常率", "0.00%", "0.00%", "0.00%"],
        ],
        widths=[4.6, 3.4, 3.4, 3.4],
    )
    rich(doc, [
        ("⚠️ 覆盖度局限：", True, WARN),
        ("这 39 个任务不含 flight_search 与 weather_advisory，而 flight_search "
         "恰是单轮评测里最弱的类别（68.2%）。因此 100% 只说明"
         "「与 agent 循环相关的四个类别全部过关」，不代表模型整体能力。", False),
    ])

    doc.add_page_break()

    # ---------------- §2 ----------------
    heading(doc, "§2 对专家问题的逐条回应", level=1)
    rich(doc, [
        ("说明：问题清单 PDF 由 ReportLab 生成，中文字体未嵌入，"
         "原文的中文在导出时全部丢失（渲染为占位符），因此以下编号对应的"
         "问题描述是", False),
        ("我根据 PDF 中残留的 ASCII 片段推断的，可能有个别误读", True),
        ("。若专家能提供原文，我会修正这张表。", False),
    ])

    heading(doc, "2.1 致命问题（必须修，否则结论站不住）", level=2)

    heading(doc, "问题 1｜核心贡献是 DPO，但没有任何 DPO 配置达到统计显著", level=3)
    rich(doc, [
        ("我的判断：", True), ("成立，且我在报告里把结论写得比证据强。", False),
    ])
    bullet(doc, "我实际只检查过 baseline → sft_v3 这一对的 CI 不重叠；"
                "「DPO 不如 SFT」从未做过显著性检验。", bold_prefix="现状：")
    bullet(doc, "对所有组对做配对检验（同一批任务上的 McNemar / 配对 bootstrap），"
                "而不是只比较各自 CI 是否重叠——CI 重叠 ≠ 无差异，这一点我做错了。",
           bold_prefix="修复：")
    bullet(doc, "给出最小可检测效应（MDE）与当前样本量的功效分析，明确"
                "「哪些结论在 100 任务下根本无法确认」。", bold_prefix="补充：")

    heading(doc, "问题 2｜51.5% 污染是自己发现的，但它仍出现在主表里", level=3)
    rich(doc, [("我的判断：", True), ("成立。这是最严重的一条，属于我的报告结构问题。", False)])
    bullet(doc, "我发现污染、重训了干净模型、拿到了 83%，但把受污染的 95% "
                "继续留在消融报告的表 1 里「作对比」，结果它污染了读者的第一印象。",
           bold_prefix="现状：")
    bullet(doc, "报告重构——主表只保留干净数据（50.25% / 78.75% / 83.00% / 49.25%）；"
                "污染数字下沉到附录，并注明「不可用于任何对比」；"
                "报告开头第一段就写主结果，而不是让读者读到 §8 才知道 95% 作废。",
           bold_prefix="修复：")

    heading(doc, "问题 3｜三组实验用了同一份被污染的数据，所以「SFT +44.8pp」建立在污染上", level=3)
    rich(doc, [("我的判断：", True), ("成立。+44.8pp 是污染版的差值，真实值是 +32.75pp。", False)])
    bullet(doc, "sft_only 95.00% → 真实（干净）sft_v3 83.00%；"
                "提升幅度 44.8pp → 32.75pp。", bold_prefix="更正：")
    bullet(doc, "所有引用了 +44.8pp / 95% 的地方（消融报告、面试稿、简历条目）"
                "全部替换或明确标注。", bold_prefix="修复：")

    heading(doc, "问题 6｜用 logps/chosen 与 rewards/margins 判断健康，不能替代 KL", level=3)
    rich(doc, [("我的判断：", True),
               ("成立，这是原理性缺陷，而且我已经因此判断错过一次。", False)])
    bullet(doc, "DPO 的 KL 约束在 p_θ 与 p_ref 的整个分布之间；"
                "logps/chosen 只含 log π_θ(y_w) 单个序列，"
                "既不含 π_ref 项，也不覆盖整体分布。", bold_prefix="原因：")
    bullet(doc, "β=0.5 实验中 logps/chosen 稳定在 −4.55（我据此判断「健康」），"
                "而 accuracy 从 83% 崩到 49.25%。", bold_prefix="实测后果：")
    bullet(doc, "写 measure_kl_drift.py：加载 π_θ（DPO 模型）与 π_ref（SFT 模型），"
                "在同一批 prompt 上计算真实的 log-prob 差与逐 token KL，"
                "替代目前单序列的判据。", bold_prefix="修复：")

    heading(doc, "问题 17｜项目自己的结论是「DPO 是净负贡献」，但表述矛盾", level=3)
    rich(doc, [("我的判断：", True), ("成立。我的报告里同时写了正贡献与负贡献，却没给统一判断。", False)])
    table(
        doc,
        ["维度", "证据", "量级"],
        [
            ["正贡献", "missing_reason_corrected：SFT 0/4 → DPO 4/4", "+1 个错误模式、4 个样本"],
            ["负贡献", "总体 accuracy：83.00% → 49.25%", "−33.75pp（−135 个样本）"],
        ],
        widths=[3.0, 8.0, 4.5],
    )
    rich(doc, [("明确结论：", True),
               ("净负贡献。用 4 个样本的改善换 135 个样本的损失，不划算。", True, WARN)])

    heading(doc, "2.2 真问题，但属于「已发现未清理」或「需新增工作」", level=2)

    heading(doc, "问题 4｜chosen/rejected 构造可能有问题（拒答 / 澄清类）", level=3)
    bullet(doc, "部分成立，而且是本次 β=0.5 失效的直接原因。CLARIFY 偏好对"
                "教的是「信息不足时追问」，模型把这个模式泛化过头，"
                "在信息充足、本该动手的任务上也改成追问（clarify_instead 20 → 54；"
                "multi_tool 100% → 33.3%；booking_flow 72.2% → 16.7%）。",
           bold_prefix="现状：")
    bullet(doc, "偏好对只告诉模型「A 比 B 好」，没有告诉它「A 在什么条件下比 B 好」。"
                "修复方向是构造条件化偏好数据（在输入里显式标注信息是否充分），"
                "而不是继续增加同类偏好对。", bold_prefix="方案：")

    heading(doc, "问题 5｜β 从 0.5 试到 0.1，但 β 单因素未被隔离", level=3)
    bullet(doc, "我已做过否证实验，但不够受控。", bold_prefix="现状：")
    table(
        doc,
        ["实验", "β", "起点", "accuracy", "起点是否相同"],
        [
            ["dpo_from_sft", "0.1", "sft_only（污染模型）", "10.00%", "❌"],
            ["dpo_sft_v3_b05", "0.5", "sft_v3（干净模型）", "49.25%", "❌"],
        ],
        widths=[4.0, 1.8, 4.6, 2.6, 2.8],
    )
    bullet(doc, "仅 2 个点，且起点模型不同，不能构成 β 的受控比较。"
                "需补 β 扫描 [0.02, 0.05, 0.1, 0.2]，全部从同一个 sft_v3 出发。",
           bold_prefix="修复：")

    heading(doc, "问题 8｜SFT 与 DPO 的学习率不同，比较不公平", level=3)
    table(
        doc,
        ["实验", "lr", "epochs", "batch", "LoRA"],
        [
            ["SFT（sft_v3）", "1e-4", "2", "2 × 8", "r=16 / α=32"],
            ["DPO（dpo_sft_v3_b05）", "1e-5", "1", "2 × 8", "r=16 / α=32"],
        ],
        widths=[5.0, 2.4, 2.2, 2.4, 3.4],
    )
    bullet(doc, "成立。SFT 的 lr 是 DPO 的 10 倍。虽然两者的优化目标不同，"
                "lr 本来就不该完全相同，但当前差距过大，无法排除"
                "「DPO 崩是因为 lr 太低/太高」这一解释。", bold_prefix="判断：")
    bullet(doc, "把 DPO 的 lr 提到 5e-5 与 1e-4 各跑一次，与现有的 1e-5 形成三点对照。",
           bold_prefix="修复：")

    heading(doc, "问题 9｜任务集分布严重不均衡（easy 27 / medium 12 / hard 61）", level=3)
    bullet(doc, "成立。hard 占 61%，而 sft_v3 的 hard 准确率只有 73.8%，"
                "整体 accuracy 被 hard 类主导。按难度分组的结论"
                "（如「medium 比 hard 高」）受类别混杂影响。", bold_prefix="判断：")
    bullet(doc, "在报告里明确声明难度标签是按类别赋的、不是按任务实际难度测的；"
                "分组结论只作参考，不作为因果推断。", bold_prefix="修复：")

    heading(doc, "问题 10｜grader 是 rule-based，没有 LLM-based 对照与人工一致性", level=3)
    bullet(doc, "成立，这是评测效度的缺口。当前所有数字都来自一个规则评分器，"
                "没有第二判据。", bold_prefix="判断：")
    bullet(doc, "抽 100 条让 LLM 判一遍，算与规则 grader 的一致率（Cohen's κ）；"
                "再人工核 30 条，形成三方对照。若一致率低于 90%，"
                "需要重估所有历史数字。", bold_prefix="修复：")

    heading(doc, "问题 11｜recovery 分析只有 48 个样本", level=3)
    bullet(doc, "成立。missing_reason_corrected 只有 n=4，"
                "Fisher 精确检验 4/4 vs 0/4 得 p≈0.029，勉强显著，"
                "但样本量太小，不足以支撑强结论。", bold_prefix="判断：")
    bullet(doc, "把每个错误模式的样本量扩到 n≥20，再下结论。"
                "当前唯一让我对它有点信心的是：该结果在两个独立的 DPO 运行"
                "（β=0.1 与 β=0.5）中复现了。", bold_prefix="修复：")

    heading(doc, "问题 14｜README / baseline_diagnosis / clarify_boundary 与代码不一致", level=3)
    bullet(doc, "部分成立。我已给两份模板文档加了现状指引，"
                "但 README 的实验清单仍是旧的（4 个实验，而实际跑了 7 组）。",
           bold_prefix="现状：")
    bullet(doc, "统一文档与代码的实验清单；写一个脚本自动校验"
                "「文档里提到的实验名是否都存在于 experiments/」，"
                "把这个不一致变成测试。", bold_prefix="修复：")

    heading(doc, "问题 15｜缺少 W&B / MLflow 等实验追踪", level=3)
    bullet(doc, "成立，但优先级低。当前用 train_log.jsonl + train_monitor.py "
                "能出健康度报告，缺的是跨实验可比性与可视化。", bold_prefix="判断：")
    bullet(doc, "接入 W&B（约 30 分钟），或至少把 7 组实验的曲线汇总成一张对照图。",
           bold_prefix="修复：")

    heading(doc, "问题 16｜版本、随机种子、shuffle 的可复现性", level=3)
    bullet(doc, "部分成立。已固定随机种子、记录环境版本（torch 2.5.1+cu124、"
                "transformers 4.49.0、trl 0.13.0、peft 0.21.2），"
                "但从未做「同配置重跑两次、比对结果」的复现性验证。",
           bold_prefix="现状：")
    bullet(doc, "用同一配置重跑一次 sft_v3，比对 accuracy 与逐任务结果，"
                "给出复现性误差范围。", bold_prefix="修复：")

    heading(doc, "问题 19｜1.5B + 100 任务 + 3090 规模太小，没试 7B/13B", level=3)
    bullet(doc, "成立。当前结论都是在 1.5B 上得到的，"
                "「对比信号损害任务能力」这个结论能否外推到更大模型，未知。",
           bold_prefix="判断：")
    bullet(doc, "跑一组 7B 的 SFT + DPO 对照。这是性价比最高的一条扩展性验证"
                "（约 3 小时 GPU）。", bold_prefix="修复：")

    heading(doc, "问题 20｜缺基线与方法对比（Functionary / ToolLLM、IPO/KTO/ORPO）", level=3)
    bullet(doc, "成立。完全没有与外部方法对比，也只在 DPO 一个偏好优化算法上做过实验。",
           bold_prefix="判断：")
    bullet(doc, "至少补一组 ORPO 或 KTO（它们不需要 reference model，"
                "能验证「崩塌是否源于 DPO 的 reference 机制」），"
                "以及在报告里明确声明未与外部 tool-calling 模型对比。",
           bold_prefix="修复：")

    heading(doc, "2.3 需要更多信息才能判断的问题", level=2)
    para(doc, "以下四条我在 PDF 里只能读到部分 ASCII 片段，推断置信度较低，"
              "需要专家提供原文：")
    table(
        doc,
        ["编号", "我推断的内容", "置信度", "需要的原文"],
        [
            ["4", "chosen/rejected 构造（拒答类样本）", "中", "该条完整描述"],
            ["7", "数据里 35% / 65% 的比例、JSON 相关", "中", "该条完整描述"],
            ["12", "agent 评测的 token / 轮数口径", "中", "该条完整描述"],
            ["13", "dpo_from_sft 与 dpo_format/recovery 的实验，mock 混入", "中", "该条完整描述"],
        ],
        widths=[1.6, 6.4, 2.0, 4.5],
    )

    doc.add_page_break()

    # ---------------- §3 ----------------
    heading(doc, "§3 我自己认为还需要做的事（专家未提及）", level=1)

    heading(doc, "3.1 结论强度与证据强度不匹配", level=2)
    para(doc, "我在报告里把几条弱证据写成了强结论。逐条列出并降级：")
    table(
        doc,
        ["现有表述", "实际证据强度", "应改成"],
        [
            ["「对比信号能编码否定性约束」", "n=4 × 1 个错误模式（p≈0.029）",
             "「在两个独立 DPO 运行中复现的初步证据，样本量需扩大」"],
            ["「DPO 不如 SFT」", "从未做显著性检验", "「观测到下降，显著性待检验」"],
            ["「agent 成功率 100%」", "39 个任务，不含最弱的 flight_search",
             "「在与 agent 循环相关的四个类别上 100%」"],
            ["「seen/unseen 泛化指标」", "seen 反而比 unseen 低 16.3pp，构成均衡，原因未明",
             "「该维度不可用于泛化结论」"],
        ],
        widths=[5.0, 5.4, 6.0],
    )

    heading(doc, "3.2 评测效度：单一判据的风险", level=2)
    para(doc, "整个项目的所有数字都来自一个规则评分器。这个项目已经出现过两次"
              "「评分口径问题」：")
    bullet(doc, "错误分类器误标 100 条（json_broken 实为「不调工具」）")
    bullet(doc, "extra_fields 没有 schema 白名单，让报告冤枉模型 42 次"
                "（虽不影响打分，但误导了失败归因）")
    rich(doc, [("这两次都是我自己发现的，但", False),
               ("没有任何机制能在下次自动发现同类问题", True, WARN),
               ("。加 LLM-based grader 对照正是为了建立第二个独立判据。", False)])

    heading(doc, "3.3 我的错误模式（自我复盘）", level=2)
    para(doc, "回顾这个项目，我犯的错有一个共同模式：")
    table(
        doc,
        ["错误", "我的推断", "实际情况"],
        [
            ["Agent 播种逻辑", "「多给信息更公平」", "泄露答案，成功率差 41pp"],
            ["污染检查", "比对 task_id 即可", "字段为空导致假阴性，白跑一轮 GPU"],
            ["extra_fields", "它出现在错误报告里，所以影响打分", "只影响诊断标签，不改 passed"],
            ["logps/chosen", "它稳定，所以模型健康", "accuracy 崩了 33.75pp"],
            ["95% 主结果", "数字好看，先往下做", "污染，需推翻重做"],
        ],
        widths=[3.8, 6.2, 6.4],
    )
    rich(doc, [("共同点：", True),
               ("我把「看起来合理」当成了「已验证」。", True, WARN),
               ("专家的问题清单本质上是在同一个方向上追问——"
                "每一个结论，你有没有独立的判据？", False)])

    doc.add_page_break()

    # ---------------- §4 ----------------
    heading(doc, "§4 执行计划", level=1)

    heading(doc, "4.1 第一批：我自己可完成，不需要 GPU（约 5 小时）", level=2)
    table(
        doc,
        ["序", "任务", "产出", "预计"],
        [
            ["1", "重构消融报告：主表只用干净数据，污染数字下沉附录",
             "ablation_report.md v2", "30 分钟"],
            ["2", "写 paired_significance.py：McNemar + 配对 bootstrap + 功效分析",
             "统计报告（含 MDE）", "40 分钟"],
            ["3", "写 measure_kl_drift.py：真实 KL(π_θ‖π_ref) 监控",
             "KL 漂移脚本", "60 分钟"],
            ["4", "写 prompt_engineering.py：不改权重的 prompt 对照",
             "对照实验脚本", "30 分钟"],
            ["5", "统一文档与代码的实验清单，加自动校验测试",
             "一致性测试", "40 分钟"],
            ["6", "把所有弱结论按 §3.1 降级，补「净负贡献」的明确判断",
             "报告与面试稿修订", "40 分钟"],
            ["7", "接入 W&B（或汇总 7 组曲线为一张对照图）", "实验追踪", "30 分钟"],
        ],
        widths=[1.2, 7.2, 4.4, 2.2],
    )

    heading(doc, "4.2 第二批：需要租用 GPU（约 3.5 小时，≈ ¥6）", level=2)
    table(
        doc,
        ["序", "任务", "目的", "预计"],
        [
            ["8", "超参对齐的 DPO：并从 sft_v3 出发，lr = 5e-5 与 1e-4 各一次",
             "排除「lr 不可比」这一解释", "40 分钟"],
            ["9", "β 扫描 [0.02, 0.05, 0.1, 0.2]，全部从 sft_v3 出发",
             "β 的受控比较（替代当前不可比的 2 点）", "80 分钟"],
            ["10", "任务集扩到 400 任务/组：baseline + sft_v3",
             "CI 从 ±10pp 压到 ±5pp，让主结果更稳", "60 分钟"],
            ["11", "prompt engineering 对照（不改权重，只改 prompt）",
             "验证「DPO 想要的效果能否由 prompt 拿到」", "20 分钟"],
            ["12", "真实 KL 漂移测量（用第 3 项的脚本）",
             "给出 π_θ 与 π_ref 的真实分布距离", "10 分钟"],
        ],
        widths=[1.2, 7.6, 4.8, 2.0],
    )

    heading(doc, "4.3 第三批：时间允许（约 4.5 小时，≈ ¥8）", level=2)
    table(
        doc,
        ["序", "任务", "目的", "预计"],
        [
            ["13", "agent 评测补 flight_search / weather 的多轮变体",
             "消除「100% 只覆盖易类别」的质疑", "我 60 分钟 + GPU 15 分钟"],
            ["14", "LLM-based grader 对照 + 人工核 30 条",
             "建立第二个独立判据，给出一致率 κ", "我 60 分钟 + GPU 30 分钟"],
            ["15", "7B 模型 SFT + DPO 对照",
             "验证结论能否外推到更大模型", "约 3 小时"],
            ["16", "同一配置重跑 sft_v3 一次",
             "给出复现性误差范围", "30 分钟"],
            ["17", "补一组 ORPO 或 KTO",
             "验证崩塌是否源于 DPO 的 reference 机制", "60 分钟"],
        ],
        widths=[1.2, 6.8, 5.6, 3.0],
    )

    heading(doc, "4.4 时间与成本合计", level=2)
    table(
        doc,
        ["批次", "我的工时", "GPU 工时", "成本（3090 ¥1.68/h）"],
        [
            ["第一批（无 GPU）", "约 5 小时", "0", "¥0"],
            ["第二批", "0", "约 3.5 小时", "约 ¥6"],
            ["第三批", "约 2 小时", "约 5.5 小时", "约 ¥9"],
            ["**合计**", "**约 7 小时**", "**约 9 小时**", "**约 ¥15**"],
        ],
        widths=[4.2, 3.4, 3.4, 4.6],
    )

    doc.add_page_break()

    # ---------------- §5 ----------------
    heading(doc, "§5 需要确认的三件事", level=1)
    table(
        doc,
        ["#", "需要确认", "为什么重要"],
        [
            ["1", "问题清单的原文（Word / Markdown / 邮件皆可）",
             "PDF 的中文已丢失，我的问题描述是推断的；"
             "第 4、7、12、13 条置信度只有「中」，可能误读"],
            ["2", "第三批（7B 对照、ORPO/KTO、agent 覆盖补齐）是否要做",
             "会显著增加 GPU 成本与时间；不做的部分我会在报告里"
             "明确列为「已知局限」而不是省略"],
            ["3", "任务集扩到 400 任务是否接受「主结果数字会变」",
             "扩集后 baseline 与 sft_v3 都会重新测，"
             "现有 83.00% 这个数字会被新数字替代"],
        ],
        widths=[1.2, 6.6, 8.6],
    )

    para(doc)
    heading(doc, "§6 我不打算做的事（明确声明）", level=1)
    bullet(doc, "不为了好看而保留任何受污染的数字：三组污染实验（sft_only / "
                "dpo_joint / dpo_from_sft）的全部数字将从主表移出。")
    bullet(doc, "不把「DPO 有正面价值」写成结论：当前证据是净负贡献，"
                "除非 β 扫描 + prompt 对照 + 条件化偏好数据能翻盘。")
    bullet(doc, "不声称 agent 评测覆盖了模型整体能力："
                "它只覆盖 39 个多轮任务中的四类。")
    bullet(doc, "不把 seen/unseen 当作泛化指标：该维度已证明不可用。")

    para(doc)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    style_run(p.add_run(
        "一句话总结：这个项目现在最缺的不是新实验，"
        "而是把已有结论的证据强度与表述强度对齐——"
        "以及在每个结论上建立独立判据。"), size=10.5, bold=True, color=ACCENT)

    doc.save(DOCX)
    return DOCX


if __name__ == "__main__":
    path = build()
    print(f"已生成: {path}")
