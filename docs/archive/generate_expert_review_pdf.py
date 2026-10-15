"""
生成项目问题审查报告 PDF（专家版）
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

# 输出路径：桌面（覆盖之前的）
output_path = r"D:\HuaweiMoveData\Users\马琬蕊\Desktop\项目问题清单.pdf"

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
title = Paragraph("项目问题审查报告", styles['Title'])
story.append(title)
story.append(Spacer(1, 0.5*cm))

subtitle = Paragraph("小模型工具调用智能体的 DPO 偏好对齐优化", styles['Heading2'])
story.append(subtitle)
story.append(Spacer(1, 0.3*cm))

expert = Paragraph("审查视角：资深大模型 Agent 专家", styles['Heading3'])
story.append(expert)
story.append(Spacer(1, 0.3*cm))

date = Paragraph("生成时间：2026-10-05", styles['Normal'])
story.append(date)
story.append(Spacer(1, 1*cm))

# 执行摘要
summary_title = Paragraph("执行摘要", styles['Heading1'])
story.append(summary_title)
story.append(Spacer(1, 0.2*cm))

summary_text = """
本报告从实验设计、DPO 方法学、Agent 评测、工程可复现性、简历表达五个维度，
对项目进行了系统性审查。共发现 <b>20 个问题</b>，其中：<br/><br/>
• <b>严重问题（6个）</b>：直接影响核心结论的可信度<br/>
• <b>中等问题（8个）</b>：影响实验的严谨性与可复现性<br/>
• <b>小问题（6个）</b>：不影响结论，但不规范<br/><br/>
<b>最关键的三个发现：</b><br/>
1. <b>统计功效严重不足</b>：100任务×4采样的规模下，任何 DPO 的总体准确率提升都不可能获得统计支持<br/>
2. <b>数据污染事件</b>：51.5%的训练样本与评测集逐字相同，SFT 的95%准确率是记忆而非泛化<br/>
3. <b>核心结论是负面的</b>：DPO 不如 SFT，这在简历表达上需要重新包装
"""
story.append(Paragraph(summary_text, styles['Normal']))
story.append(Spacer(1, 0.5*cm))

# ========== 一、实验设计与方法学问题 ==========
story.append(PageBreak())
h1 = Paragraph("一、实验设计与方法学问题（严重）", styles['Heading1'])
story.append(h1)
story.append(Spacer(1, 0.3*cm))

# 问题 1
p1 = Paragraph("<b>问题 1：统计功效严重不足，核心结论不可靠</b>", styles['Heading3'])
story.append(p1)
story.append(Spacer(1, 0.2*cm))

detail1 = """
<b>现状</b>：100 任务 × 4 采样 = 400 样本，任务级 bootstrap CI 宽度约 ±10pp，
可检出的最小效应约 14pp。<br/><br/>
<b>问题</b>：<br/>
• SFT 已达 95% 准确率，天花板只剩 5pp<br/>
• 14pp > 5pp，意味着任何 DPO 配置的总体准确率提升都不可能获得统计支持<br/>
• 因此「DPO 不如 SFT」这个结论，可能只是噪声，不是真的效果差<br/><br/>
<b>影响</b>：所有关于总体准确率的对比结论都不可信，只能看错误类型的结构性变化。
"""
story.append(Paragraph(detail1, styles['Normal']))
story.append(Spacer(1, 0.4*cm))

# 问题 2
p2 = Paragraph("<b>问题 2：数据污染事件（51.5%训练样本与评测集逐字相同）</b>", styles['Heading3'])
story.append(p2)
story.append(Spacer(1, 0.2*cm))

detail2 = """
<b>现状</b>：sft_only 的 2509 条训练样本中，1292 条（51.5%）的 prompt 与评测 prompt 逐字相同，
且 chosen 就是该任务的正确答案。<br/><br/>
<b>问题</b>：<br/>
• 这等于在测试集上做梯度下降<br/>
• SFT 的 95% 准确率不是泛化能力，是记忆能力<br/>
• 所有用受污染数据训练的模型（sft_only、dpo_joint、dpo_from_sft）的结果都不可信<br/><br/>
<b>影响</b>：项目的主结果（SFT 带来 44.8pp 提升）完全站不住脚。
干净的 SFT 准确率预计会远低于 95%。
"""
story.append(Paragraph(detail2, styles['Normal']))
story.append(Spacer(1, 0.4*cm))

# 问题 3
p3 = Paragraph("<b>问题 3：缺乏关键对照组</b>", styles['Heading3'])
story.append(p3)
story.append(Spacer(1, 0.2*cm))

detail3 = """
<b>缺失的对照组</b>：<br/>
• <b>纯 prompt engineering 对照组</b>：只改 system prompt 不训练，能提升多少？<br/>
• <b>随机种子重复实验</b>：只跑了一次，无法判断稳定性<br/>
• <b>不同模型规模的消融</b>：只测了 1.5B，不知道 7B/13B 上是否同样成立<br/>
• <b>不同数据量的学习曲线</b>：不知道多少偏好数据是最优的<br/><br/>
<b>影响</b>：无法区分提升是来自 DPO 方法本身，还是来自其他因素。
"""
story.append(Paragraph(detail3, styles['Normal']))
story.append(Spacer(1, 0.4*cm))

# 问题 4
p4 = Paragraph("<b>问题 4：偏好数据全是人工合成的，缺乏真实人类偏好</b>", styles['Heading3'])
story.append(p4)
story.append(Spacer(1, 0.2*cm))

detail4 = """
<b>现状</b>：所有 chosen/rejected 都是代码生成的，没有真实人类标注的偏好数据。<br/><br/>
<b>问题</b>：<br/>
• 偏好数据的分布与真实用户偏好的分布可能不一致<br/>
• DPO 学到的是「代码定义的好」，不是「人类定义的好」<br/>
• 合成数据的多样性不足，可能导致模型过拟合到特定模式<br/><br/>
<b>影响</b>：项目结论的外部效度有限，不能推广到真实场景。
"""
story.append(Paragraph(detail4, styles['Normal']))
story.append(Spacer(1, 0.4*cm))

# ========== 二、DPO 方法学问题 ==========
story.append(PageBreak())
h2 = Paragraph("二、DPO 方法学问题", styles['Heading1'])
story.append(h2)
story.append(Spacer(1, 0.3*cm))

# 问题 5
p5 = Paragraph("<b>问题 5：β 选择缺乏依据，没有系统的敏感性分析</b>", styles['Heading3'])
story.append(p5)
story.append(Spacer(1, 0.2*cm))

detail5 = """
<b>现状</b>：直接用 DPO 论文的 β=0.1，后来实测 β=0.5 也崩了。<br/><br/>
<b>问题</b>：<br/>
• β=0.1 是全参微调的设置，不适合 LoRA（参数量少得多）<br/>
• 只测了两个点（0.1 和 0.5），没有系统的 β 扫描<br/>
• 不知道最优 β 是多少，也不知道 β 对最终效果的影响有多大<br/><br/>
<b>影响</b>：无法判断 DPO 失效是方法本身的问题，还是超参没调好。
"""
story.append(Paragraph(detail5, styles['Normal']))
story.append(Spacer(1, 0.4*cm))

# 问题 6
p6 = Paragraph("<b>问题 6：KL 散度监控缺失，无法量化分布偏移</b>", styles['Heading3'])
story.append(p6)
story.append(Spacer(1, 0.2*cm))

detail6 = """
<b>现状</b>：只监控了 logps/chosen 和 rewards/margins，没有直接监控 KL 散度。<br/><br/>
<b>问题</b>：<br/>
• KL 散度是 DPO 健康度的核心指标，直接反映 π_θ 与 π_ref 的分布偏移程度<br/>
• 没有 KL 监控，就无法判断过优化的程度<br/>
• logps/chosen 只能预测「分布是否漂移」，不能预测「任务能力是否保留」<br/><br/>
<b>影响</b>：训练监控不完整，无法早期发现模型退化。
"""
story.append(Paragraph(detail6, styles['Normal']))
story.append(Spacer(1, 0.4*cm))

# 问题 7
p7 = Paragraph("<b>问题 7：偏好对的难度设置不合理，简单负样本过多</b>", styles['Heading3'])
story.append(p7)
story.append(Spacer(1, 0.2*cm))

detail7 = """
<b>现状</b>：硬负样本比例只有 35%，其余 65% 是简单负样本（纯文本、坏JSON）。<br/><br/>
<b>问题</b>：<br/>
• 简单负样本模型很容易区分，梯度贡献小<br/>
• 可能导致训练 loss 下降快，但实际能力提升有限<br/>
• 硬负样本不足，模型学不到「看起来对但其实错」的精细判断<br/><br/>
<b>影响</b>：DPO 的学习效率可能不高。
"""
story.append(Paragraph(detail7, styles['Normal']))
story.append(Spacer(1, 0.4*cm))

# 问题 8
p8 = Paragraph("<b>问题 8：没有对比 SFT 和 DPO 的学习曲线</b>", styles['Heading3'])
story.append(p8)
story.append(Spacer(1, 0.2*cm))

detail8 = """
<b>现状</b>：只看了最终结果，没有画学习曲线。<br/><br/>
<b>问题</b>：<br/>
• 不知道 DPO 在什么阶段开始过优化<br/>
• 不知道 SFT 在什么阶段收敛<br/>
• 无法判断是数据问题还是训练步数问题<br/><br/>
<b>影响</b>：无法定位 DPO 失效的具体原因。
"""
story.append(Paragraph(detail8, styles['Normal']))
story.append(Spacer(1, 0.4*cm))

# ========== 三、Agent 能力评测问题 ==========
story.append(PageBreak())
h3 = Paragraph("三、Agent 能力评测问题", styles['Heading1'])
story.append(h3)
story.append(Spacer(1, 0.3*cm))

# 问题 9
p9 = Paragraph("<b>问题 9：评测任务集太小且领域单一</b>", styles['Heading3'])
story.append(p9)
story.append(Spacer(1, 0.2*cm))

detail9 = """
<b>现状</b>：只有 100 个任务，都是航空客服领域，7 类任务。<br/><br/>
<b>问题</b>：<br/>
• 任务量太小，统计意义有限<br/>
• 领域单一，无法泛化到其他场景<br/>
• 难度分布不均（easy 27、medium 12、hard 61）<br/><br/>
<b>影响</b>：结论的外部效度有限，不能推广到一般的 agent 场景。
"""
story.append(Paragraph(detail9, styles['Normal']))
story.append(Spacer(1, 0.4*cm))

# 问题 10
p10 = Paragraph("<b>问题 10：规则-based grader 可能存在偏差</b>", styles['Heading3'])
story.append(p10)
story.append(Spacer(1, 0.2*cm))

detail10 = """
<b>现状</b>：评分器是规则-based 的，不是 LLM-based 的。<br/><br/>
<b>问题</b>：<br/>
• 同义不同表达可能被判错（比如「明天」vs「2026-10-06」）<br/>
• 没有 human evaluation 验证 grader 的准确性<br/>
• 没有 inter-annotator agreement 测试<br/><br/>
<b>影响</b>：评测指标的准确性存疑。
"""
story.append(Paragraph(detail10, styles['Normal']))
story.append(Spacer(1, 0.4*cm))

# 问题 11
p11 = Paragraph("<b>问题 11：多轮交互评测不足</b>", styles['Heading3'])
story.append(p11)
story.append(Spacer(1, 0.2*cm))

detail11 = """
<b>现状</b>：主要是单轮评测，多轮评测只有 48 个纠错样本。<br/><br/>
<b>问题</b>：<br/>
• agent 的核心能力是多轮决策，但评测主要在单轮<br/>
• 48 个样本太少，统计意义有限<br/>
• 没有测「完成整个任务的成功率」<br/><br/>
<b>影响</b>：没有真正测到 agent 的核心能力。
"""
story.append(Paragraph(detail11, styles['Normal']))
story.append(Spacer(1, 0.4*cm))

# 问题 12
p12 = Paragraph("<b>问题 12：缺乏真实 agent 指标（效率、成本）</b>", styles['Heading3'])
story.append(p12)
story.append(Spacer(1, 0.2*cm))

detail12 = """
<b>缺失的指标</b>：<br/>
• <b>任务完成率</b>：整个任务链是否成功完成<br/>
• <b>步数效率</b>：完成任务用了多少步<br/>
• <b>token 成本</b>：完成任务消耗了多少 token<br/>
• <b>工具调用次数</b>：平均调用了多少次工具<br/><br/>
<b>影响</b>：只测了「单次调用对不对」，没有测真实 agent 的综合能力。
"""
story.append(Paragraph(detail12, styles['Normal']))
story.append(Spacer(1, 0.4*cm))

# ========== 四、工程与可复现性问题 ==========
story.append(PageBreak())
h4 = Paragraph("四、工程与可复现性问题", styles['Heading1'])
story.append(h4)
story.append(Spacer(1, 0.3*cm))

# 问题 13
p13 = Paragraph("<b>问题 13：实验结果目录与报告不一致，数据来源不透明</b>", styles['Heading3'])
story.append(p13)
story.append(Spacer(1, 0.2*cm))

detail13 = """
<b>现状</b>：<br/>
• 报告里说有 dpo_from_sft，但目录下没有<br/>
• 报告里说 dpo_format/dpo_recovery 没跑，但目录下有<br/>
• 训练日志都是 mock 的，但报告里有真实数字<br/><br/>
<b>影响</b>：无法复现实验结果，数据来源不透明。
"""
story.append(Paragraph(detail13, styles['Normal']))
story.append(Spacer(1, 0.4*cm))

# 问题 14
p14 = Paragraph("<b>问题 14：文档多处矛盾</b>", styles['Heading3'])
story.append(p14)
story.append(Spacer(1, 0.2*cm))

detail14 = """
<b>矛盾之处</b>：<br/>
• README 说「真实实验未运行」，但消融报告说「4组实验已跑」<br/>
• baseline_diagnosis 说「数字是占位符」，但消融报告里有真实数字<br/>
• 偏好数据设计文档说「四类偏好对」，但实际用了五类（加了 clarify_boundary）<br/><br/>
<b>影响</b>：读者无法判断哪个是对的，文档可信度下降。
"""
story.append(Paragraph(detail14, styles['Normal']))
story.append(Spacer(1, 0.4*cm))

# 问题 15
p15 = Paragraph("<b>问题 15：缺乏实验追踪系统</b>", styles['Heading3'])
story.append(p15)
story.append(Spacer(1, 0.2*cm))

detail15 = """
<b>现状</b>：没有用 Weights & Biases 或 MLflow。<br/><br/>
<b>问题</b>：<br/>
• 实验配置、结果、日志分散在不同目录<br/>
• 没有版本控制的实验记录<br/>
• 无法快速对比不同实验的配置差异<br/><br/>
<b>影响</b>：实验管理效率低，容易混乱。
"""
story.append(Paragraph(detail15, styles['Normal']))
story.append(Spacer(1, 0.4*cm))

# 问题 16
p16 = Paragraph("<b>问题 16：训练环境不透明，无法保证可复现</b>", styles['Heading3'])
story.append(p16)
story.append(Spacer(1, 0.2*cm))

detail16 = """
<b>缺失的信息</b>：<br/>
• GPU 型号、驱动版本、CUDA 版本<br/>
• 依赖版本（transformers、trl、peft 的具体版本）<br/>
• 随机种子的完整控制（数据 shuffle、模型初始化等）<br/><br/>
<b>影响</b>：别人无法复现你的实验结果。
"""
story.append(Paragraph(detail16, styles['Normal']))
story.append(Spacer(1, 0.4*cm))

# ========== 五、简历与求职表达问题 ==========
story.append(PageBreak())
h5 = Paragraph("五、简历与求职表达问题", styles['Heading1'])
story.append(h5)
story.append(Spacer(1, 0.3*cm))

# 问题 17
p17 = Paragraph("<b>问题 17：核心结论是负面的，需要重新包装</b>", styles['Heading3'])
story.append(p17)
story.append(Spacer(1, 0.2*cm))

detail17 = """
<b>现状</b>：主要发现是「DPO 不如 SFT」「DPO 会导致模型退化」。<br/><br/>
<b>问题</b>：<br/>
• 这是一个负面结果，直接写简历没有亮点<br/>
• 面试官可能会问「那你这个项目有什么价值？」<br/><br/>
<b>建议的包装方向</b>：<br/>
• 「发现了 DPO 在小模型工具调用上的失效模式」<br/>
• 「定位了 DPO 过优化的早期信号（logps/chosen 崩塌）」<br/>
• 「发现了纯正例监督学不会否定性约束的现象」<br/>
• 「建立了完整的工具调用评测与诊断流水线」
"""
story.append(Paragraph(detail17, styles['Normal']))
story.append(Spacer(1, 0.4*cm))

# 问题 18
p18 = Paragraph("<b>问题 18：数据污染事件的风险</b>", styles['Heading3'])
story.append(p18)
story.append(Spacer(1, 0.2*cm))

detail18 = """
<b>现状</b>：51.5% 的训练样本与评测集逐字相同。<br/><br/>
<b>风险</b>：<br/>
• 如果面试官发现，会质疑你的实验严谨性<br/>
• 这是一个严重的实验错误<br/><br/>
<b>建议的应对方式</b>：<br/>
• 主动坦白，并说明你如何发现、如何修复<br/>
• 强调你有「自我审计」的能力，能发现自己的错误<br/>
• 这反而可能成为一个加分项（如果你讲得好）
"""
story.append(Paragraph(detail18, styles['Normal']))
story.append(Spacer(1, 0.4*cm))

# 问题 19
p19 = Paragraph("<b>问题 19：项目规模偏小</b>", styles['Heading3'])
story.append(p19)
story.append(Spacer(1, 0.2*cm))

detail19 = """
<b>现状</b>：1.5B 模型、100 任务、单卡 3090。<br/><br/>
<b>风险</b>：<br/>
• 1.5B 不是主流的 7B/13B，可能被质疑「模型太小，结论不一般」<br/>
• 100 任务规模太小，可能被质疑「评测不充分」<br/>
• 单卡训练，可能被质疑「工程能力不够」<br/><br/>
<b>建议的应对</b>：<br/>
• 强调「单卡可复现」是优势，不是劣势<br/>
• 强调「1.5B 上的发现更有普适性」（大模型上可能不明显）
"""
story.append(Paragraph(detail19, styles['Normal']))
story.append(Spacer(1, 0.4*cm))

# 问题 20
p20 = Paragraph("<b>问题 20：缺乏与现有工作的对比</b>", styles['Heading3'])
story.append(p20)
story.append(Spacer(1, 0.2*cm))

detail20 = """
<b>缺失的对比</b>：<br/>
• 没有和主流的 tool calling 方法对比（比如 Functionary、ToolLLM）<br/>
• 没有和最新的 DPO 变体对比（比如 IPO、KTO、ORPO）<br/>
• 不知道你的方法在学术界处于什么水平<br/><br/>
<b>影响</b>：面试官可能会问「你这个工作和 X 有什么区别？」
"""
story.append(Paragraph(detail20, styles['Normal']))
story.append(Spacer(1, 0.5*cm))

# ========== 总结 ==========
h6 = Paragraph("总结与优先级建议", styles['Heading1'])
story.append(h6)
story.append(Spacer(1, 0.3*cm))

summary_table = [
    ['优先级', '问题', '建议动作'],
    ['P0（最紧急）', '数据污染', '用干净数据重训 SFT 和 DPO，得到真实数字'],
    ['P0（最紧急）', '统计功效不足', '把评测集扩到 200-300 任务，降低 CI 宽度'],
    ['P1（重要）', '缺乏对照组', '加一个纯 prompt engineering 对照组'],
    ['P1（重要）', 'β 敏感性分析', '系统扫描 β = [0.01, 0.05, 0.1, 0.5, 1.0]'],
    ['P1（重要）', '多轮 agent 评测', '补充真实的多轮任务成功率评测'],
    ['P2（改进）', '实验追踪', '用 W&B 记录所有实验'],
    ['P2（改进）', '文档一致性', '统一所有文档的说法'],
    ['P2（改进）', '简历包装', '把负面结果包装成「失效模式分析」'],
]
table_sum = Table(summary_table)
table_sum.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('FONTNAME', (0, 0), (-1, 0), CN_FONT),
    ('FONTNAME', (0, 1), (-1, -1), CN_FONT),
    ('FONTSIZE', (0, 0), (-1, -1), 10),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ('TOPPADDING', (0, 0), (-1, -1), 8),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('BACKGROUND', (0, 1), (0, 2), colors.lightcoral),
    ('BACKGROUND', (0, 3), (0, 5), colors.lightyellow),
    ('BACKGROUND', (0, 6), (0, 8), colors.lightgreen),
]))
story.append(table_sum)
story.append(Spacer(1, 0.5*cm))

final_text = """
<b>一句话评价</b>：这个项目的<b>工程框架和诊断思路是完整的</b>，
但<b>核心实验结果受数据污染影响不可信</b>，
且<b>统计功效不足以支撑「DPO 不如 SFT」的结论</b>。

<b>好消息是</b>：你已经发现了数据污染的问题，也有了修复方案。
只要用干净数据重训一遍，这个项目就能成为一个非常好的简历项目 ——
尤其是「发现 DPO 在小模型工具调用上的失效模式」这个角度，
在求职市场上是有差异化价值的。
"""
story.append(Paragraph(final_text, styles['Normal']))

# 生成 PDF
doc.build(story)
print(f"PDF 已生成：{output_path}")
