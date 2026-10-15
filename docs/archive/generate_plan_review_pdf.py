"""
生成后续工作计划分析报告 PDF
"""
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# 注册中文字体（Windows 黑体）
pdfmetrics.registerFont(TTFont('SimHei', r'C:\Windows\Fonts\simhei.ttf'))
CN_FONT = 'SimHei'

# 输出路径：桌面
output_path = r"D:\HuaweiMoveData\Users\马琬蕊\Desktop\后续工作计划分析.pdf"

# 创建文档
doc = SimpleDocTemplate(
    output_path,
    pagesize=A4,
    rightMargin=2*cm,
    leftMargin=2*cm,
    topMargin=2*cm,
    bottomMargin=2*cm
)

# 所有样式统一用中文字体
styles = getSampleStyleSheet()
styles['Title'].fontName = CN_FONT
styles['Title'].fontSize = 20
styles['Heading1'].fontName = CN_FONT
styles['Heading1'].fontSize = 16
styles['Heading2'].fontName = CN_FONT
styles['Heading2'].fontSize = 14
styles['Heading3'].fontName = CN_FONT
styles['Heading3'].fontSize = 12
styles['Normal'].fontName = CN_FONT
styles['Normal'].fontSize = 10
styles['Normal'].leading = 14

story = []

# ========== 封面 ==========
title = Paragraph("后续工作计划分析报告", styles['Title'])
story.append(title)
story.append(Spacer(1, 0.5*cm))

subtitle = Paragraph("agent-optimization 项目：以资深大模型 Agent 专家视角", styles['Heading2'])
story.append(subtitle)
story.append(Spacer(1, 0.3*cm))

date = Paragraph("生成时间：2026-10-06", styles['Normal'])
story.append(date)
story.append(Spacer(1, 1*cm))

# 执行摘要
summary_title = Paragraph("执行摘要", styles['Heading1'])
story.append(summary_title)
story.append(Spacer(1, 0.2*cm))

summary_text = """
先肯定：<b>这份计划的自我认知非常清醒</b>——能主动发现自己的问题、优先级划分合理、
还明确说了"不打算做的事"，这在学生项目里已经是 top 10% 的水平了。

但从资深专家 + 求职导向的视角，这个计划还有 <b>12 个深层问题</b>，按重要性分三类：<br/><br/>
• <b>战略级问题（3个）</b>：核心定位模糊、本末倒置、全是修复没有亮点<br/>
• <b>实验设计问题（3个）</b>：β 扫描方向错了、lr 对齐思路错了、7B 时机不对<br/>
• <b>简历表达问题（2个）</b>：数据污染的讲法、差异化价值<br/>
• <b>执行层面问题（2个）</b>：时间估算乐观、没有 MVP<br/>
• <b>更深层问题（2个）</b>：在修旧项目而非做新项目、没有正面结果<br/><br/>
<b>最关键的一句话</b>：你现在最缺的不是补实验，而是先想清楚这个项目到底要证明什么。
"""
story.append(Paragraph(summary_text, styles['Normal']))
story.append(Spacer(1, 0.5*cm))

# ========== 一、战略级问题 ==========
story.append(PageBreak())
h1 = Paragraph("一、战略级问题（最关键）", styles['Heading1'])
story.append(h1)
story.append(Spacer(1, 0.3*cm))

# 问题 1
p1 = Paragraph("<b>问题 1：核心定位模糊——你花这么多时间，到底要证明什么？</b>", styles['Heading3'])
story.append(p1)
story.append(Spacer(1, 0.2*cm))

detail1 = """
<b>现状</b>：计划是"补全所有实验，得到干净的数字"。<br/>
<b>但问题是</b>：如果补完之后，结论还是" DPO 是净负贡献"——用 4 个样本的改善换 135 个样本的损失——
那你花了这么多时间，最后得到了什么？<br/><br/>
<b>面试官会问</b>：那你这个项目的价值是什么？证明了 DPO 不好？<br/><br/>
<b>建议</b>：先想清楚核心叙事是什么，再决定补什么实验：<br/>
• A. "我发现了 DPO 在小模型工具调用上的失效模式"（负面结果，有价值但不够亮）<br/>
• B. "我建立了一套完整的工具调用评测与诊断流水线"（工程价值）<br/>
• C. "我提出了一种条件化偏好数据，解决了 DPO 的过度泛化问题"（正面贡献，最有亮点）<br/><br/>
<b>如果是 C</b>，那你的计划里根本没有"构造条件化偏好数据"这一项——这才是最该做的事。
"""
story.append(Paragraph(detail1, styles['Normal']))
story.append(Spacer(1, 0.4*cm))

# 问题 2
p2 = Paragraph("<b>问题 2：本末倒置——先补实验，还是先想清楚要解决什么问题？</b>", styles['Heading3'])
story.append(p2)
story.append(Spacer(1, 0.2*cm))

detail2 = """
<b>现状</b>：先把所有实验补全，得到"干净的数字"。<br/>
<b>但这是典型的"为了补实验而补实验"</b>。<br/><br/>
<b>举个例子</b>：<br/>
• 你说要把任务集从 100 扩到 400，把 CI 从 ±10pp 压到 ±5pp<br/>
• 但 DPO 比 SFT 低 33pp，就算 CI 压到 ±1pp，结论还是"DPO 不如 SFT"<br/>
• 这个投入的意义是什么？只是让数字"更精确"，但不改变结论<br/><br/>
<b>建议</b>：反过来想——你要回答什么问题？然后设计最小的实验去回答它。
"""
story.append(Paragraph(detail2, styles['Normal']))
story.append(Spacer(1, 0.4*cm))

# 问题 3
p3 = Paragraph('<b>问题 3：全是"修复问题"，没有"增强亮点"</b>', styles['Heading3'])
story.append(p3)
story.append(Spacer(1, 0.2*cm))

detail3 = """
<b>现状</b>：17 项任务里，全是：重构报告、补统计检验、补 KL 监控、补 β 扫描、补对照组、补 LLM grader。<br/>
<b>但这些都是"把项目从 60 分提到 80 分"，不是"做出一个能拿得出手的亮点"</b>。<br/><br/>
<b>面试官想看的是</b>：你能<b>解决问题</b>，而不是你能<b>发现问题</b>。<br/>
<b>你现在的项目</b>：发现了一堆问题，但没有解决任何一个。<br/><br/>
<b>建议</b>：至少加一项"正面贡献"的任务，比如：<br/>
• 构造条件化偏好数据，解决 DPO 的过度泛化问题<br/>
• 或者做一个简单的 agent 规划模块，展示你对 agent 的理解
"""
story.append(Paragraph(detail3, styles['Normal']))
story.append(Spacer(1, 0.4*cm))

# ========== 二、实验设计问题 ==========
story.append(PageBreak())
h2 = Paragraph("二、实验设计层面的问题", styles['Heading1'])
story.append(h2)
story.append(Spacer(1, 0.3*cm))

# 问题 4
p4 = Paragraph("<b>问题 4：β 扫描的方向错了</b>", styles['Heading3'])
story.append(p4)
story.append(Spacer(1, 0.2*cm))

detail4 = """
<b>现状</b>：计划扫描 β = [0.02, 0.05, 0.1, 0.2]。<br/>
<b>但之前的实验</b>：β=0.1 崩到 10%，β=0.5 崩到 49%。<br/><br/>
<b>问题</b>：β 更小 = KL 约束更弱 = 模型漂移更多 = <b>更崩</b>。<br/>
你往 β 更小的方向扫，是南辕北辙。<br/><br/>
<b>正确的方向</b>：应该往 <b>β 更大</b> 的方向扫（0.5, 1.0, 2.0, 5.0）。<br/>
因为你的问题是"模型漂移太多"，更大的 β = 更强的 KL 约束 = 更少的漂移。
"""
story.append(Paragraph(detail4, styles['Normal']))
story.append(Spacer(1, 0.4*cm))

# 问题 5
p5 = Paragraph('<b>问题 5："lr 对齐"的思路错了</b>', styles['Heading3'])
story.append(p5)
story.append(Spacer(1, 0.2*cm))

detail5 = """
<b>现状</b>：说"SFT 的 lr 是 DPO 的 10 倍，不公平"，计划把 DPO 的 lr 提到 5e-5 和 1e-4。<br/><br/>
<b>问题</b>：DPO 和 SFT 的优化目标<b>本来就不同</b>，loss 尺度完全不一样，lr 没有"对齐"的说法。<br/>
DPO 的 loss 是 sigmoid 形式，SFT 的 loss 是交叉熵，两者的梯度尺度差好几个数量级。<br/><br/>
<b>你应该问的是</b>：DPO 的最优 lr 是多少？<br/>
而不是"SFT 用了多少，DPO 也用多少"。
"""
story.append(Paragraph(detail5, styles['Normal']))
story.append(Spacer(1, 0.4*cm))

# 问题 6
p6 = Paragraph("<b>问题 6：7B 模型对照的时机不对</b>", styles['Heading3'])
story.append(p6)
story.append(Spacer(1, 0.2*cm))

detail6 = """
<b>现状</b>：计划最后跑 7B 模型，约 3 小时 GPU。<br/>
<b>但问题是</b>：如果 1.5B 上的结论是"DPO 不行"，那 7B 上大概率也是不行。<br/><br/>
<b>为什么不先在 1.5B 上把问题搞清楚，再往上扩？</b><br/>
这是典型的"用规模掩盖问题"。如果 1.5B 上都搞不定，7B 上只会更乱。<br/><br/>
<b>建议</b>：把 7B 对照放到最后，甚至可以砍掉。先把 1.5B 的问题搞透。
"""
story.append(Paragraph(detail6, styles['Normal']))
story.append(Spacer(1, 0.4*cm))

# ========== 三、简历表达问题 ==========
story.append(PageBreak())
h3 = Paragraph("三、简历表达层面的问题", styles['Heading1'])
story.append(h3)
story.append(Spacer(1, 0.3*cm))

# 问题 7
p7 = Paragraph("<b>问题 7：数据污染事件的风险，你还没讲好</b>", styles['Heading3'])
story.append(p7)
story.append(Spacer(1, 0.2*cm))

detail7 = """
<b>现状</b>：计划主动坦白数据污染的问题。<br/>
<b>这是对的，但怎么讲很重要</b>：<br/><br/>
• ❌ <b>减分讲法</b>："我犯了个错，训练数据里混入了评测集，后来发现了，修复了。"<br/><br/>
• ✅ <b>加分讲法</b>："我建立了一套<b>数据质量门禁系统</b>，通过 prompt 文本逐字比对
（而不是 task_id），自动检测训练数据与评测集的污染，发现并修复了 51.5% 的污染率。"<br/><br/>
<b>同样一件事，讲法不同，从减分变成加分。</b>
"""
story.append(Paragraph(detail7, styles['Normal']))
story.append(Spacer(1, 0.4*cm))

# 问题 8
p8 = Paragraph("<b>问题 8：你的差异化价值是什么？</b>", styles['Heading3'])
story.append(p8)
story.append(Spacer(1, 0.2*cm))

detail8 = """
<b>现状</b>：Qwen2.5-1.5B + DPO + 工具调用 + 航空客服。<br/>
<b>问题</b>：网上类似的项目一大堆，你的差异化在哪里？<br/><br/>
<b>面试官会想</b>：这和 GitHub 上的 "DPO 微调工具调用" 项目有什么区别？<br/><br/>
<b>你需要一个能让面试官记住你的点</b>，比如：<br/>
• "我发现了 DPO 在小模型上的<b>过度泛化失效模式</b>——模型把'该追问'的模式泛化到了所有任务上"<br/>
• "我建立了一套<b>错误分类体系 + 诊断流水线</b>，能从 baseline 自动定位失败模式并生成对应的偏好数据"
"""
story.append(Paragraph(detail8, styles['Normal']))
story.append(Spacer(1, 0.4*cm))

# ========== 四、执行层面问题 ==========
story.append(PageBreak())
h4 = Paragraph("四、执行层面的问题", styles['Heading1'])
story.append(h4)
story.append(Spacer(1, 0.3*cm))

# 问题 9
p9 = Paragraph("<b>问题 9：时间估算过于乐观</b>", styles['Heading3'])
story.append(p9)
story.append(Spacer(1, 0.2*cm))

detail9 = """
<b>现状</b>：说"约 5 小时我自己完成第一批"。<br/>
<b>但实际呢</b>：<br/>
• 写 paired_significance.py（McNemar + bootstrap + 功效分析）——至少 2 小时<br/>
• 写 measure_kl_drift.py（加载两个模型，计算逐 token KL）——至少 3 小时<br/>
• 重构消融报告——至少 2 小时<br/>
• 统一文档 + 自动校验测试——至少 2 小时<br/><br/>
<b>5 小时根本不够</b>，至少要 15-20 小时。
"""
story.append(Paragraph(detail9, styles['Normal']))
story.append(Spacer(1, 0.4*cm))

# 问题 10
p10 = Paragraph("<b>问题 10：没有 MVP（最低可交付版本）</b>", styles['Heading3'])
story.append(p10)
story.append(Spacer(1, 0.2*cm))

detail10 = """
<b>现状</b>：17 项任务，全是"要做的事"。<br/>
<b>但如果时间不够，哪几项是必须做的？</b><br/><br/>
<b>建议</b>：从 17 项里挑出 <b>3 项 MVP</b>，做完就能拿出手：<br/>
1. 重构报告：主表只用干净数据（30 分钟）<br/>
2. 配对显著性检验 + 功效分析（40 分钟）<br/>
3. 把"净负贡献"的结论写清楚，重新包装叙事（40 分钟）<br/><br/>
<b>这三项做完，你的项目就从"60 分"变成"75 分"了</b>。剩下的 14 项都是锦上添花。
"""
story.append(Paragraph(detail10, styles['Normal']))
story.append(Spacer(1, 0.4*cm))

# ========== 五、更深层问题 ==========
story.append(PageBreak())
h5 = Paragraph("五、更深层的问题（值得停下来想一想）", styles['Heading1'])
story.append(h5)
story.append(Spacer(1, 0.3*cm))

# 问题 11
p11 = Paragraph('<b>问题 11：你在"修复旧项目"，而不是"做新项目"</b>', styles['Heading3'])
story.append(p11)
story.append(Spacer(1, 0.2*cm))

detail11 = """
<b>现状</b>：整个计划都是在修复这个项目的问题。<br/>
<b>但你只有一个月的时间</b>。<br/><br/>
<b>换个角度想</b>：与其花一个月修复一个旧项目，不如花一周做一个<b>新的、更有亮点的项目</b>，比如：<br/>
• 基于 MCP 的工具调用 agent<br/>
• 多轮对话的 agent 规划（ReAct / Plan-and-Execute）<br/>
• 工具学习（Tool Learning）的小实验<br/><br/>
<b>现在的热点是 Agent，不是 DPO</b>。你还在做 DPO 微调，是不是有点过时了？
"""
story.append(Paragraph(detail11, styles['Normal']))
story.append(Spacer(1, 0.4*cm))

# 问题 12
p12 = Paragraph('<b>问题 12：你没有"拿得出手的正面结果"</b>', styles['Heading3'])
story.append(p12)
story.append(Spacer(1, 0.2*cm))

detail12 = """
<b>现状</b>：所有结果都是"负面的"：<br/>
• DPO 不如 SFT<br/>
• DPO 会导致模型退化<br/>
• 偏好数据构造有问题<br/>
• 统计功效不足<br/><br/>
<b>面试官会想</b>：那你做了这么久，到底做出了什么？<br/><br/>
<b>至少要有一个正面结果</b>，比如：<br/>
• "我的评测流水线能自动定位错误模式，效率提升了 X 倍"<br/>
• "我发现了某个失效模式，提出了修复方向"<br/>
• "我的某个改进，在 X 指标上提升了 Y pp"
"""
story.append(Paragraph(detail12, styles['Normal']))
story.append(Spacer(1, 0.5*cm))

# ========== 总结 ==========
h6 = Paragraph("总结与优先级建议", styles['Heading1'])
story.append(h6)
story.append(Spacer(1, 0.3*cm))

# 评分表
score_data = [
    ['维度', '评分', '说明'],
    ['自我认知', '⭐⭐⭐⭐⭐', '非常清醒，能发现自己的问题'],
    ['优先级划分', '⭐⭐⭐⭐', 'P0/P1/P2 合理'],
    ['实验设计', '⭐⭐⭐', 'β 扫描方向错了，lr 对齐思路错了'],
    ['简历导向', '⭐⭐', '全是修复问题，没有亮点'],
    ['时间估算', '⭐⭐', '过于乐观'],
    ['战略高度', '⭐⭐', '没有想清楚这个项目到底要证明什么'],
]
table_score = Table(score_data)
table_score.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('FONTNAME', (0, 0), (-1, 0), CN_FONT),
    ('FONTNAME', (0, 1), (-1, -1), CN_FONT),
    ('FONTSIZE', (0, 0), (-1, -1), 10),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ('TOPPADDING', (0, 0), (-1, -1), 8),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
]))
story.append(table_score)
story.append(Spacer(1, 0.5*cm))

# 建议
suggest_title = Paragraph("我的建议（按优先级）", styles['Heading2'])
story.append(suggest_title)
story.append(Spacer(1, 0.2*cm))

suggest_text = """
1. <b>先停下来，想清楚这个项目的核心叙事是什么</b>——是负面结果（发现失效模式），
   还是正面结果（提出解决方案）？<br/><br/>
2. <b>砍掉一半的任务</b>，只做 MVP 三项，先拿出一个能看的版本<br/><br/>
3. <b>把"数据污染"从"错误"包装成"数据质量门禁系统"</b>——这是加分项，不是减分项<br/><br/>
4. <b>β 扫描往大的方向扫</b>（0.5, 1.0, 2.0），不是往小的<br/><br/>
5. <b>最后再考虑 7B 扩展</b>，先把 1.5B 的问题搞透
"""
story.append(Paragraph(suggest_text, styles['Normal']))
story.append(Spacer(1, 0.5*cm))

final_text = """
<b>最关键的一句话</b>：你现在最缺的不是补实验，而是先想清楚这个项目到底要证明什么。<br/><br/>
如果补完所有实验，结论还是"DPO 不如 SFT"，那你花了这么多时间，
最后得到的只是一个"更精确的负面结果"——这在简历上的价值非常有限。
"""
story.append(Paragraph(final_text, styles['Normal']))

# 生成 PDF
doc.build(story)
print(f"PDF 已生成：{output_path}")
