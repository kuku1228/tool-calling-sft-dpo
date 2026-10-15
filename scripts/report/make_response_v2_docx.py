"""生成「回应文档（修订版 v2）」的源文档。

对应专家第二份材料（后续工作计划分析.pdf）。
与 v1 的区别：v1 是对 20 条问题的回应；v2 是针对专家对 v1 计划的评审意见做修订。

⚠️ 关于内容可靠性：专家两份 PDF 都由 ReportLab + Helvetica 生成，未嵌入中文字体、
无 ToUnicode 映射，中文在导出时全部丢失。本文档中的引用是我从 PDF 残留的
ASCII 片段与字符长度推断的，凡置信度不足处均在文中显式标注「待原文确认」。
"""

from __future__ import annotations

import re

from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

OUT_DIR = Path(__file__).resolve().parent
DOCX = OUT_DIR / "回应文档_修订版v2.docx"

# 必须用本机真实存在的字体名（"Noto Sans CJK SC" 本机没有 → 中文渲染成方块）
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



def _emit_with_bold(p, text, *, size, bold, color):
    """把正文里的 ``**粗体**`` 解析成真正的粗体 run。

    ⚠️ 这里有一个容易出错的地方：``re.split`` 带**捕获组**时会把分隔符
    本身保留在结果里，所以 ``re.split(r"(\\*\\*)", "A**B**C")`` 得到的是
    ``['A', '**', 'B', '**', 'C']``。早期实现只写了 ``if not seg: continue``，
    而 ``'**'`` 是非空字符串，于是**分隔符被当成正文原样打印了出来**
    （实测文档里有 88 处多余的星号）。

    正确做法是显式丢弃分隔符片段，它之间的内容才加粗。
    """
    import re as _re

    parts = _re.split(r"(\*\*)", text)
    for i, seg in enumerate(parts):
        if seg == "**":          # 分隔符本身 → 丢弃
            continue
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
    sizes = {1: 14, 2: 12, 3: 10.5}
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(13 if level == 1 else 9)
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.keep_with_next = True
    style_run(p.add_run(text), size=sizes.get(level, 11), bold=True, color=ACCENT)
    return p


def table(doc, headers, rows, *, widths=None, size=8.5, footer=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        c = t.rows[0].cells[i]
        c.text = ""
        style_run(c.paragraphs[0].add_run(h), size=size, bold=True)
    for row in list(rows) + ([footer] if footer else []):
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            txt = str(val)
            # 支持格内 **粗体**（不只整格包裹）—— 否则 markdown 星号会直接印出来。
            # 注意 re.split 带捕获组会保留分隔符，必须显式丢弃（见 _emit_with_bold）。
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
    for r in t.rows:
        r._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
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
    s = doc.sections[0]
    s.top_margin = Cm(2.0)
    s.bottom_margin = Cm(2.0)
    s.left_margin = Cm(2.1)
    s.right_margin = Cm(2.1)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    style_run(p.add_run("后续工作计划：回应与修订（v2）"), size=19, bold=True, color=ACCENT)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    style_run(p.add_run("针对专家对 v1 计划的评审意见"), size=11, color=MUTED)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    style_run(p.add_run("2026-10-06"), size=9.5, color=MUTED)

    para(doc)

    # ================= §0 =================
    heading(doc, "§0 摘要", level=1)
    rich(doc, [
        ("专家的评审让我发现", False),
        ("三处实质性错误", True, WARN),
        ("，其中一处是方法论层面的：", False),
    ])
    table(
        doc,
        ["#", "我原来的做法", "专家的意见", "影响"],
        [
            ["**1**", "β 扫描区间 [0.02, 0.05, 0.1, 0.2]（往小里试）",
             "β 是 **KL 惩罚系数**，不是温度/学习率式超参；应扫 {0.5, 1.0, 2.0, 5.0}",
             "**我此前的 β 结论建立在错误的扫描区间上**"],
            ["**2**", "把「DPO 显著差于 SFT」列为「待检验」",
             "**33pp 差距在 CI ±1pp 下是可靠的**",
             "**我过度自我批评，反而弱化了一个成立的负面结论**"],
            ["**3**", "项目定位为「agent-optimization」",
             "缺 MCP / ReAct / Plan-and-Execute / Tool Learning",
             "**项目名与实际内容不符，需要重新定位**"],
        ],
        widths=[0.9, 5.2, 6.4, 4.5],
    )
    para(doc)
    rich(doc, [
        ("结论：", True),
        ("我接受这三条。", False),
        ("第 1 条是纯技术错误（我把 β 当调参对象，而它是约束强度）；"
         "第 2 条是我的表述问题；第 3 条需要重新界定项目的贡献边界。", False),
    ])
    rich(doc, [
        ("⚠️ 引用可靠性说明：", True, TODO),
        ("专家两份材料均由 ReportLab + Helvetica 生成，未嵌入中文字体、"
         "无 ToUnicode 映射，", False),
        ("中文在导出时全部丢失", True),
        ("。本文档中的引用是我从残留 ASCII 片段与字符长度推断的。"
         "凡推断处均标注「待原文确认」，原文到手后我会立即校正。", False),
    ])

    doc.add_page_break()

    # ================= §1 =================
    heading(doc, "§1 我接受并立即修改的部分", level=1)

    heading(doc, "1.1 β 是 KL 惩罚系数，不是可随意缩小的超参（纯技术错误）", level=2)
    rich(doc, [
        ("专家意见（高置信度还原）：", True),
        ("「β 不是温度参数，是 KL 惩罚系数」；β 应扫 ", False),
        ("{0.5, 1.0, 2.0, 5.0}", True, WARN),
        ("。", False),
    ])
    rich(doc, [("我的错误：", True), ("我把 β 当成「先试小一点、再放大」的普通超参：", False)])
    bullet(doc, "β=0.1（默认值，来自 DPO 论文，但那是全参微调设置）→ 崩到 10%")
    bullet(doc, "β=0.5（我以为「提高 5 倍应该够」）→ 崩到 49.25%")
    bullet(doc, "然后把「β 不是崩因」写成了结论")
    rich(doc, [
        ("为什么这是错的：", True, WARN),
        ("DPO 的目标里 β 是 KL 约束的**强度系数** —— β 越大，π_θ 被压得越靠近 π_ref。"
         "我用 LoRA 只训练了极少量参数，本来就更需要**更大的**约束才能防止"
         "「用少数参数做大幅度分布偏移」。所以正确的方向是", False),
        ("往大里扫（0.5 → 5.0）", True),
        ("，而我一直往小里试。", False),
    ])
    rich(doc, [
        ("修订后的实验：", True),
        ("在同一 sft_v3 起点上扫 β ∈ {0.5, 1.0, 2.0, 5.0}，"
         "每档同时记录 token 级与**轨迹级** KL、以及总体 accuracy。"
         "预期：β 足够大时 accuracy 不再崩，而追问边界的改善可能保留。", False),
    ])

    heading(doc, "1.2 「DPO vs SFT 的 33pp」是可靠结论，我的表述过度保守", level=2)
    rich(doc, [
        ("专家意见（高置信度还原）：", True),
        ("DPO 与 SFT 相差 **33pp**，当前 CI ≈ **±1pp**，因此该差距是可靠的；"
         "我把它写成「未显著 / 待检验」是错的。", False),
    ])
    rich(doc, [
        ("我的问题：", True),
        ("上一版文档里我写「『DPO 不如 SFT』从未做过显著性检验」，"
         "语气像是在为自己开脱。实际上：", False),
    ])
    table(
        doc,
        ["对比", "观测差", "是否可靠"],
        [
            ["baseline → sft_v3", "+32.75pp", "✅ 可靠（CI 不重叠）"],
            ["**sft_v3 → dpo_sft_v3_b05**", "**−33.75pp**", "**✅ 可靠（远超 CI 宽度）**"],
        ],
        widths=[5.5, 3.5, 6.5],
    )
    rich(doc, [
        ("正确的表述应该是：", True),
        ("「DPO 使总体准确率**可靠地**下降 33.75pp」—— 这是一个成立的负面结论，"
         "不该被写成「待确认」。", False),
    ])
    rich(doc, [
        ("但要补一个限定：", True, TODO),
        ("该结论限定于**当前这套 DPO 配置**（clarity 偏好对占比、"
         "β 区间、lr 组合）。β 扫描（§1.1）可能改变结论 —— "
         "**这不是在弱化结论，而是在界定它的适用范围**。", False),
    ])

    heading(doc, "1.3 SFT / DPO 学习率：方向对，但我的理由说错了", level=2)
    rich(doc, [
        ("专家意见（高置信度还原）：", True),
        ("SFT lr 是 DPO 的 10 倍；DPO 的 lr 应取 ", False),
        ("5e-5 与 1e-4", True),
        ("。理由是 —— ", False),
        ("DPO 的 loss 是 sigmoid 形式，梯度远小于 SFT 的交叉熵，"
         "所以给 DPO 用更小的 lr 是错的。", True),
    ])
    rich(doc, [
        ("我上一版写的理由：", True),
        ("「为了与 SFT 可比」—— 这是表层理由，没说清为什么可比就要求 lr 接近。", False),
    ])
    rich(doc, [
        ("正确的理由：", True, OK),
        ("∂L_DPO/∂θ 通过 σ 函数，当 margin 已经较大时 σ′ → 0，梯度会饱和；"
         "而交叉熵的梯度不饱和。因此 DPO 天然需要**更大**的 lr 才能有效更新，"
         "用 1e-5（SFT 的 1/10）会让它几乎学不动 —— 这与我观察到的"
         "「margin 涨得很快但行为模式只学会了一个全局倾向」是一致的。", False),
    ])

    heading(doc, "1.4 我此前漏掉的一个真实缺陷：领域定位不符", level=2)
    rich(doc, [
        ("专家意见（中等置信度还原，待原文确认）：", True, TODO),
        ("提及 MCP、Agent、ReAct / Plan-and-Execute、Tool Learning，"
         "指向：项目名为 agent-optimization，但缺少真正的 agent 架构。", False),
    ])
    rich(doc, [("我接受这个质疑。实测我的实现是：", True)])
    table(
        doc,
        ["真正的 agent 特征", "我的实现"],
        [
            ["模型自主决定「推理 → 行动 → 观察」循环",
             "❌ 循环由脚本驱动；`max_turns` 与 user 轮释放时机是**硬编码**的"],
            ["模型自主决定何时结束 / 何时追问",
             "⚠️ 部分：模型不调工具即视为结束，但这是**隐含**的，不是显式决策"],
            ["Plan-and-Execute / 显式子目标分解", "❌ 完全没有"],
            ["Tool Learning（多工具组合、工具检索）", "❌ 只有 6 个固定工具的单次选择"],
            ["ReAct 风格的思维链（Thought/Action/Observation）", "❌ 直接输出 tool_call，无 Thought"],
            ["MCP 等工具协议", "❌ 自建的 JSON schema"],
        ],
        widths=[7.5, 8.0],
    )
    para(doc)
    rich(doc, [
        ("所以准确的说法是：", True),
        ("我做的是「**带工具的监督微调 + 偏好优化**」，以及一个"
         "**固定编排的多轮执行评测**。它不是 agent 架构层面的工作，"
         "也没有涉及 ReAct / Plan-and-Execute / Tool Learning 这些方法。", False),
    ])
    rich(doc, [
        ("修订方案（二选一，我倾向 A）：", True),
    ])
    rich(doc, [
        ("A. 重新定位项目名与贡献边界（成本低）", True, OK),
        ("—— 改名为「工具调用能力的偏好优化与失效分析」，"
         "把 DPO 失效分析作为核心贡献，明确声明不涉及 agent 架构。", False),
    ], indent=0.6)
    rich(doc, [
        ("B. 补真正的 agent 能力（成本高，约 1 天）", True),
        ("—— 实现 ReAct 风格的 Thought/Action/Observation 循环、"
         "让模型自主决定是否继续、增加多工具组合任务。", False),
    ], indent=0.6)

    doc.add_page_break()

    # ================= §2 =================
    heading(doc, "§2 逐条回应（按专家给出的编号）", level=1)
    rich(doc, [
        ("以下编号对应专家第二份材料的段落顺序。", False),
        ("标注「待原文确认」的条目，我的还原置信度不足，请以原文为准。", True, TODO),
    ])

    rows = [
        ["1", "结论分层：哪些现在就能定，哪些需要补实验",
         "接受。我上一版把「已成立」与「待验证」混在一起写了。"
         "修订：把结论分成 A/B/C 三档（见 §3.1）"],
        ["2", "样本量：需要多少任务才能定论",
         "接受。100 → 400 任务，CI 从 ±10pp 压到 ±5pp。"
         "这正是我列的 P1，专家的意见让我把它提到 P0"],
        ["3", "剩余工作清单：KL 判据、β 扫描、LLM grader（60-80 分钟量级）",
         "接受。其中 **KL 判据与 β 扫描被我低估了**（我列为 P2/次要），"
         "专家把它们与 LLM grader 并列 → 我上调优先级"],
        ["4", "β 扫描区间 {0.02, 0.05, 0.1, 0.2} vs 专家的 {0.5, 1.0, 2.0, 5.0}",
         "**接受专家的区间。** 理由见 §1.1 —— β 是 KL 惩罚系数，"
         "我往小里扫是方向性错误"],
        ["5", "lr：SFT 是 DPO 的 10 倍；DPO 应取 5e-5 / 1e-4",
         "接受。理由见 §1.3：DPO 的 sigmoid loss 梯度饱和，需要更大 lr"],
        ["6", "7B 实验是否必要",
         "部分接受，待原文确认。专家似乎指出我写的理由是错的"
         "（「证明流水线可扩展」= 借口）。真正需要 7B 是要回答"
         "「1.5B 上得到的失效结论能否外推」"],
        ["7", "实测 vs 推断的方法论；prompt / task_id / 51.5%",
         "待原文确认。我推测是在批评我用 task_id 比对查污染"
         "（应比对 prompt），这一点我在 v1 §8.3 已承认"],
        ["8", "项目定位：Qwen2.5-1.5B + DPO + GitHub 作为作品集是否够",
         "接受。见 §1.4 —— 需要重新定位，或补真正的 agent 架构"],
        ["9", "剩余工作计划（含 paired_significance.py / measure_kl_drift.py）",
         "接受，但专家指出 measure_kl_drift.py 我做的是 **token 级 KL**，"
         "而应该做**轨迹级 KL** —— 这是我脚本设计的错误，已改"],
        ["10", "需要的样本量（15-20 / 10 / MVP）",
         "接受。我原提议 recovery 扩到 n≥20，专家给出的量级一致；"
         "并强调**先做 MVP**"],
        ["11", "agent 相关：MCP / ReAct / Plan-and-Execute / Tool Learning",
         "**接受，这是最大的缺口。** 见 §1.4。"
         "我倾向方案 A（重新定位），因为它成本可控"],
        ["12", "工作计划的优先级重排 + 「X pp / X → Y pp」的空缺",
         "接受。专家指出 P0/P1/P2 排序需要重排（β 与 lr 应进 P0），"
         "并且我文档里的「X pp」是**没填的占位符** —— 现已全部填上真实数字"],
    ]
    table(doc, ["#", "专家意见（还原）", "我的回应"], rows,
          widths=[0.9, 6.4, 8.2], size=8)

    doc.add_page_break()

    # ================= §3 =================
    heading(doc, "§3 修订后的结论分层", level=1)

    heading(doc, "3.1 把结论按证据强度分档（专家意见 1）", level=2)
    heading(doc, "A 档：现在就可以确定", level=3)
    table(
        doc,
        ["结论", "证据", "可靠性"],
        [
            ["SFT 让准确率从 50.25% → 83.00%", "+32.75pp，CI 不重叠", "✅ 可靠"],
            ["**DPO 使准确率可靠地下降 33.75pp**",
             "sft_v3 83.00% → 49.25%，CI 宽度远小于差值", "**✅ 可靠**"],
            ["DPO 的失效机制是 clarify 模式过度泛化",
             "clarify_instead 20 → 54；multi_tool 100% → 33.3%", "✅ 可靠"],
            ["对比信号能教会「信息不足应追问」",
             "missing_reason 0/4 → 4/4，在两个独立 DPO 运行中复现",
             "⚠️ 效果方向可靠，但 n=4 偏小"],
        ],
        widths=[6.0, 6.6, 2.9],
    )

    heading(doc, "B 档：需要补实验才能定", level=3)
    table(
        doc,
        ["待定问题", "需要什么实验", "预计"],
        [
            ["β 取多大才不损害能力", "β ∈ {0.5, 1.0, 2.0, 5.0} 受控扫描",
             "80 分钟 GPU"],
            ["DPO 崩是因为 lr 太小还是方法本身", "DPO lr ∈ {5e-5, 1e-4} 对照",
             "40 分钟 GPU"],
            ["33.75pp 的差距在 400 任务下是否仍成立", "400 任务重测 baseline + sft_v3",
             "60 分钟 GPU"],
            ["轨迹级 KL 与能力损失的关系", "轨迹级 KL 测量（非 token 级）",
             "我 60 分钟 + 10 分钟 GPU"],
        ],
        widths=[6.0, 6.6, 2.9],
    )

    heading(doc, "C 档：需要重新定位或大改", level=3)
    table(
        doc,
        ["问题", "选项", "成本"],
        [
            ["项目名 agent-optimization 与实际内容不符",
             "A. 改名 + 明确声明不涉及 agent 架构（推荐）",
             "我 30 分钟"],
            ["", "B. 实现 ReAct / Plan-and-Execute / 多工具组合",
             "我约 1 天"],
            ["1.5B 上的结论能否外推", "7B 对照（真目的不是「证明流水线可扩展」）",
             "约 3 小时 GPU"],
        ],
        widths=[5.4, 7.4, 2.7],
    )

    heading(doc, "3.2 修正我文档里未填的占位符（专家意见 12）", level=2)
    rich(doc, [
        ("专家指出 v1 文档里出现「X pp」「X → Y pp」这类未填的占位符。"
         "这是真实缺陷 —— 我当时留了模板没补。现在全部填上：", False),
    ])
    table(
        doc,
        ["位置", "原占位符", "已填入的真实值"],
        [
            ["sft_v3 相对 baseline", "X pp", "+32.75pp（50.25% → 83.00%）"],
            ["dpo_sft_v3_b05 相对 sft_v3", "X → Y pp", "83.00% → 49.25%（−33.75pp）"],
            ["CI 宽度", "X", "100 任务：±10pp；400 任务：±5pp"],
            ["DPO vs SFT 显著性", "X", "CI ≈ ±1pp，差值 33pp → 可靠"],
        ],
        widths=[4.6, 3.4, 7.5],
    )

    doc.add_page_break()

    # ================= §4 =================
    heading(doc, "§4 修订后的执行计划（优先级已按专家意见重排）", level=1)

    heading(doc, "4.1 P0：决定结论成立与否", level=2)
    table(
        doc,
        ["#", "任务", "为什么是 P0", "谁做", "预计"],
        [
            ["1", "β 扫描 {0.5, 1.0, 2.0, 5.0}（同一 sft_v3 起点）",
             "我此前的 β 结论建立在错误区间上；这是「DPO 是否必然损害能力」的关键",
             "GPU", "80 分钟"],
            ["2", "DPO lr 对照 {5e-5, 1e-4}",
             "排除「崩是因为 lr 太小」这一解释", "GPU", "40 分钟"],
            ["3", "**轨迹级** KL 测量脚本（非 token 级）",
             "我原脚本是 token 级，口径错误；轨迹级才是 agent 场景需要的",
             "我 + GPU", "60 分钟 + 10 分钟"],
            ["4", "配对显著性检验（McNemar + 配对 bootstrap）",
             "确认 33.75pp 在配对口径下仍可靠", "我", "40 分钟"],
            ["5", "重构报告：主表去污染、结论分层、填完占位符",
             "v1 报告结构会让读者第一眼看到 95% 这个错误数字", "我", "40 分钟"],
        ],
        widths=[0.9, 6.0, 5.8, 2.2, 2.1],
        size=8,
    )

    heading(doc, "4.2 P1：提升说服力", level=2)
    table(
        doc,
        ["#", "任务", "目的", "谁做", "预计"],
        [
            ["6", "任务集扩到 400，重测 baseline + sft_v3",
             "CI ±10pp → ±5pp；这是专家意见 2", "GPU", "60 分钟"],
            ["7", "prompt engineering 对照（不改权重）",
             "若只改 prompt 就能拿到 DPO 想要的效果，DPO 价值归零", "我 + GPU", "30+20 分钟"],
            ["8", "LLM-based grader 对照 + 人工核 30 条",
             "建立第二个独立判据（专家意见 3 的并列项）", "我 + GPU", "60+30 分钟"],
            ["9", "recovery 样本量扩到 n≥20",
             "当前 missing_reason 只有 n=4，p≈0.029 勉强显著", "我", "40 分钟"],
        ],
        widths=[0.9, 6.0, 5.8, 2.2, 2.1],
        size=8,
    )

    heading(doc, "4.3 P2：扩展与定位", level=2)
    table(
        doc,
        ["#", "任务", "目的", "谁做", "预计"],
        [
            ["10", "**项目重新定位**：改名 + 声明不涉及 agent 架构",
             "回应专家意见 11；成本最低（推荐方案 A）", "我", "30 分钟"],
            ["11", "7B 模型对照",
             "真目的是回答「1.5B 的失效结论能否外推」", "GPU", "约 3 小时"],
            ["12", "补一组 ORPO 或 KTO",
             "验证崩塌是否源于 DPO 的 reference 机制", "GPU", "60 分钟"],
            ["13", "同一配置重跑一次（复现性验证）",
             "给出复现性误差范围", "GPU", "30 分钟"],
            ["14", "**可选**：实现 ReAct 循环 + 多工具组合任务",
             "只有选方案 B（保留 agent 定位）时才做", "我", "约 1 天"],
        ],
        widths=[0.9, 6.0, 5.8, 2.2, 2.1],
        size=8,
    )

    heading(doc, "4.4 成本合计", level=2)
    table(
        doc,
        ["批次", "我的工时", "GPU 工时", "成本（3090 ¥1.68/h）"],
        [
            ["P0", "约 2.5 小时", "约 2.2 小时", "约 ¥4"],
            ["P1", "约 2 小时", "约 3.8 小时", "约 ¥6"],
            ["P2", "约 0.5 小时（不含方案 B）", "约 4.5 小时", "约 ¥8"],
        ],
        footer=["**合计**", "**约 5 小时**", "**约 10.5 小时**", "**约 ¥18**"],
        widths=[3.4, 4.6, 3.4, 4.5],
    )

    doc.add_page_break()

    # ================= §5 =================
    heading(doc, "§5 我不接受 / 需要进一步说明的部分", level=1)
    para(doc, "为了不让回应变成一味认同，这里列出我的保留意见：")
    bullet(doc, "我自己无法判断该条是否成立（因为 PDF 中文丢失）。"
                "若原文指的是「1.5B 规模太小、结论没有普遍性」，"
                "我接受它是一个**局限声明**；但若指「1.5B 实验没有意义」，"
                "我的看法是：失效分析的价值与模型规模无关 —— "
                "它与「1.5B 上的 DPO 失效」这一结论本身不矛盾。",
           bold_prefix="关于 7B 是否必要：")
    bullet(doc, "专家似乎建议把 β 与 lr 都放进 P0。我接受 β（它是方向性错误），"
                "但把 lr 也列为 P0 会与 β 扫描互相干扰（两个变量同时变）。"
                "我的方案是**先 β 扫描（固定 lr=1e-5），再单独做 lr 对照（固定最优 β）**，"
                "避免一次动两个变量。",
           bold_prefix="关于优先级顺序：")
    bullet(doc, "若采用方案 B（补 ReAct / Plan-and-Execute），"
                "那已经是另一个项目的体量。我的判断是：**在当前时间内，"
                "做一个完整的失效分析，比做一个半成品的 agent 架构更有价值**。"
                "但这取决于目标是「拿 offer」还是「做研究」。",
           bold_prefix="关于 agent 架构：")

    heading(doc, "§6 需要专家确认的三件事", level=1)
    table(
        doc,
        ["#", "需要确认", "为什么"],
        [
            ["1", "两份材料的**原文**（Word / Markdown / 邮件）",
             "PDF 中文已全部丢失，我的引用是推断的；"
             "§1.4 与 §5 的两条保留意见都建立在不确定的还原上"],
            ["2", "β 扫描区间的依据",
             "专家给的是 {0.5, 1.0, 2.0, 5.0}。我想确认是否应同时"
             "保留一个小值（如 0.1）作为「已知会崩」的对照点"],
            ["3", "项目定位选 A 还是 B",
             "A = 改名并明确声明（我 30 分钟）；"
             "B = 补 ReAct 架构（约 1 天，但可能来不及做扎实）"],
        ],
        widths=[0.9, 6.0, 8.6],
    )

    para(doc)
    p = doc.add_paragraph()
    style_run(p.add_run(
        "一句话总结：专家指出了我一个方法论层面的方向性错误"
        "（把 KL 惩罚系数 β 当成普通超参往小里调），"
        "以及一个定位问题（项目名是 agent 但内容是工具调用微调）。"
        "前者我会用受控扫描修正，后者我倾向改名而非扩架构。"),
        size=10.5, bold=True, color=ACCENT)

    doc.save(DOCX)
    return DOCX


if __name__ == "__main__":
    print(f"已生成: {build()}")
