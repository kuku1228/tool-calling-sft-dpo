"""
生成项目问题清单 PDF
"""
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import cm

# 输出路径：桌面
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

styles = getSampleStyleSheet()
story = []

# 标题
title = Paragraph("项目问题清单与风险评估", styles['Title'])
story.append(title)
story.append(Spacer(1, 0.5*cm))

# 副标题
subtitle = Paragraph("项目：小模型工具调用智能体的 DPO 偏好对齐优化", styles['Heading3'])
story.append(subtitle)
story.append(Spacer(1, 0.3*cm))

date = Paragraph("生成时间：2026-10-05", styles['Normal'])
story.append(date)
story.append(Spacer(1, 0.8*cm))

# ========== 一、严重问题 ==========
h1 = Paragraph("一、严重问题（影响结论可信度）", styles['Heading1'])
story.append(h1)
story.append(Spacer(1, 0.3*cm))

# 问题 1
p1 = Paragraph("<b>问题 1：实验结果全是 mock 的，但报告里写了「真实数字」</b>", styles['Heading3'])
story.append(p1)
story.append(Spacer(1, 0.2*cm))

evidence1 = """
<b>发现的证据：</b><br/>
• experiments/*/eval_report*.json 全是 .mock.json 后缀<br/>
• outputs/*/train_log.jsonl 第一条就有 "is_mock": true<br/>
• baseline 的 mock 报告：30 任务 × 2 采样 = 60 样本，准确率 43.3%<br/><br/>
<b>但消融报告里写的是：</b><br/>
• 100 任务 × 4 采样 = 400 样本<br/>
• 基线准确率 50.2%、SFT 95%、DPO-joint 68.5%<br/><br/>
<b>问题：</b>这些「真实数字」是从哪里来的？目录里找不到对应的 eval_report.json（不带 mock 的）。
"""
story.append(Paragraph(evidence1, styles['Normal']))
story.append(Spacer(1, 0.5*cm))

# 问题 2
p2 = Paragraph("<b>问题 2：failure_cases 数量和报告里的准确率完全对不上</b>", styles['Heading3'])
story.append(p2)
story.append(Spacer(1, 0.2*cm))

# 表格
data2 = [
    ['实验', 'failure_cases 数量', '报告准确率', '按 400 样本算的错误数'],
    ['baseline_airline', '34 个', '50.2%', '~199 个错误'],
    ['sft_only', '167 个', '95.0%', '~20 个错误'],
    ['dpo_joint', '171 个', '68.5%', '~126 个错误'],
]
table2 = Table(data2)
table2.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 10),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
    ('TOPPADDING', (0, 0), (-1, -1), 12),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
]))
story.append(table2)
story.append(Spacer(1, 0.3*cm))

contradiction2 = """
<b>矛盾点：</b><br/>
• SFT 报告说只有 5% 错误（约 20 个），但实际有 167 个 failure_cases<br/>
• DPO-joint 的 failure_cases（171）比 SFT（167）还多，但报告说 DPO 准确率更低？<br/>
• 数字完全对不上
"""
story.append(Paragraph(contradiction2, styles['Normal']))
story.append(Spacer(1, 0.5*cm))

# 问题 3
p3 = Paragraph("<b>问题 3：dpo_from_sft 实验不存在，但报告里有它的结果</b>", styles['Heading3'])
story.append(p3)
story.append(Spacer(1, 0.2*cm))

evidence3 = """
<b>消融报告里写了：</b><br/>
• dpo_from_sft：准确率 10.0%，格式错误率 90%<br/>
• 还详细分析了它为什么崩（β=0.1 过优化）<br/><br/>
<b>但目录里：</b><br/>
• 没有 experiments/dpo_from_sft/ 文件夹<br/>
• 没有 outputs/dpo_from_sft/ 训练日志<br/><br/>
<b>问题：</b>这个实验的结果是从哪里来的？
"""
story.append(Paragraph(evidence3, styles['Normal']))
story.append(Spacer(1, 0.5*cm))

# ========== 二、中等问题 ==========
story.append(PageBreak())
h2 = Paragraph("二、中等问题（会让人困惑）", styles['Heading1'])
story.append(h2)
story.append(Spacer(1, 0.3*cm))

# 问题 4
p4 = Paragraph("<b>问题 4：dpo_format 和 dpo_recovery 跑了，但报告里说「没跑」</b>", styles['Heading3'])
story.append(p4)
story.append(Spacer(1, 0.2*cm))

evidence4 = """
<b>消融报告里说：</b><br/>
• dpo_format（仅格式约束）：❌ 未跑<br/>
• dpo_recovery（仅恢复对）：❌ 未跑<br/><br/>
<b>但目录里：</b><br/>
• experiments/dpo_format/ 有完整的 failure_cases<br/>
• experiments/dpo_recovery/ 有完整的 failure_cases<br/>
• outputs/dpo_format/ 有训练日志<br/>
• outputs/dpo_recovery/ 有训练日志<br/><br/>
<b>问题：</b>到底跑没跑？
"""
story.append(Paragraph(evidence4, styles['Normal']))
story.append(Spacer(1, 0.5*cm))

# 问题 5
p5 = Paragraph("<b>问题 5：README 状态过期，和实际情况不一致</b>", styles['Heading3'])
story.append(p5)
story.append(Spacer(1, 0.2*cm))

evidence5 = """
<b>README 第 3 节写的是：</b><br/>
• 真实 Qwen2.5-1.5B 基线评测：❌ 未运行<br/>
• 真实 DPO 训练（四组消融）：❌ 未运行<br/><br/>
<b>但消融报告里：</b><br/>
• 4 组实验已跑，但因数据污染需重跑<br/><br/>
<b>问题：</b>到底跑了还是没跑？两套说法完全矛盾。
"""
story.append(Paragraph(evidence5, styles['Normal']))
story.append(Spacer(1, 0.5*cm))

# ========== 三、小问题 ==========
h3 = Paragraph("三、小问题（不影响结论，但不规范）", styles['Heading1'])
story.append(h3)
story.append(Spacer(1, 0.3*cm))

# 问题 6
p6 = Paragraph("<b>问题 6：实验规模不一致</b>", styles['Heading3'])
story.append(p6)
story.append(Spacer(1, 0.2*cm))

evidence6 = """
• mock 的 baseline：30 任务 × 2 采样<br/>
• 报告里说的：100 任务 × 4 采样<br/>
• 两个规模完全不同，怎么对比？
"""
story.append(Paragraph(evidence6, styles['Normal']))
story.append(Spacer(1, 0.5*cm))

# 问题 7
p7 = Paragraph("<b>问题 7：数据污染修复了，但没重训</b>", styles['Heading3'])
story.append(p7)
story.append(Spacer(1, 0.2*cm))

evidence7 = """
消融报告里说：<br/>
• 已修复（硬门禁 + 合成语料独立 + 零撞车验证），但尚未重训<br/><br/>
这意味着：<br/>
• 所有现有结果都是用受污染的数据跑的<br/>
• SFT 的 95% 准确率不能作为结论引用<br/>
• 但报告里还是把这些数字写进去了
"""
story.append(Paragraph(evidence7, styles['Normal']))
story.append(Spacer(1, 0.8*cm))

# ========== 四、总结 ==========
h4 = Paragraph("四、总结：项目当前状态", styles['Heading1'])
story.append(h4)
story.append(Spacer(1, 0.3*cm))

summary_data = [
    ['模块', '状态'],
    ['代码框架', '✅ 完整'],
    ['单元测试', '✅ 通过'],
    ['Mock 流水线', '✅ 跑通'],
    ['真实实验', '❓ 不确定跑没跑'],
    ['数据污染', '⚠️ 已修复代码，但没重训'],
    ['文档一致性', '❌ 多处矛盾'],
]
table_sum = Table(summary_data)
table_sum.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 11),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
    ('TOPPADDING', (0, 0), (-1, -1), 10),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
]))
story.append(table_sum)
story.append(Spacer(1, 0.5*cm))

conclusion = """
<b>一句话总结：</b>你的代码和 mock 流程都没问题，但<b>真实实验的结果和文档对不上</b>——
要么是真实实验跑了但结果没保存到这个目录，要么是报告里的数字是从别的地方来的。
"""
story.append(Paragraph(conclusion, styles['Normal']))

# 生成 PDF
doc.build(story)
print(f"PDF 已生成：{output_path}")
