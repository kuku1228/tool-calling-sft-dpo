## 已知限制（dpo_joint，评测集 106 个任务）

> **本章节由 `src/eval/limitations.py` 自动生成** —— 
> 这些限制若不主动声明，读者会把结论读得比实际更强。
> 数字从当前数据算出，数据变了文字会跟着变。

| # | 严重性 | 限制 |
|---|---|---|
| 1 | 高 | visibility_meaningless |
| 2 | 高 | duplicate_expected_calls |
| 3 | 中 | distribution_mismatch |
| 4 | 低（且加强论点） | sft_no_negative_samples |

### 1. visibility_meaningless（高）

`seen`/`unseen` 标签**与训练暴露无关**：seen 组签名被训练覆盖 66/67（99%），unseen 组 39/39（100%）。该标签的赋值从不读取训练数据（是按序号/骨架哈希轮转）。

**⛔ 不能这样声称**：**不要**把 seen 与 unseen 的差值解释为泛化差距 —— 那是哈希分组的噪声。改用「实体重合度」三档（见 `scripts/eval/audit_eval_overlap.py`）。

验证：`python -c "import sys;sys.path.insert(0,'.');from src.data.task_manifest_v2 import report_visibility_warning as W;print(W())"`

### 2. duplicate_expected_calls（高）

106 个任务只有 **92 个不同期望调用** —— 14 个任务是重复计数。最大的簇有 **15 个任务**，它们的期望调用完全相同：`(None, ())`。

**⛔ 不能这样声称**：**不要**说「15 个独立样本」—— 有效独立样本是**1 个行为**，15 是**触发方式**的数量。统计必须按**簇**重采样（`paired_stats.paired_bootstrap_clustered`），它给出更宽的置信区间（这是更保守的口径）。

验证：`python scripts/eval/paired_significance.py --a X --b Y `

### 3. distribution_mismatch（中）

训练语料里 `search_flights` 占 40.5%，而评测里只占 30.2% —— 训练分布偏向搜索工具。

**⛔ 不能这样声称**：**不要**把绝对准确率外推到其他分布。但**组间对比**（SFT vs DPO）不受影响：两者用同一份训练数据、同一分布，偏差对它们**对称**。

验证：`python scripts/prepare/verify_all.py`

### 4. sft_no_negative_samples（低（且加强论点））

SFT 训练数据 2635 条里，`chosen` **不含** `tool_call` 的有 **0 条** —— 纯正例监督在结构上无法表达「此处不该调用工具」。

**⛔ 不能这样声称**：**不要**把这说成「SFT 因为数据里没有所以做不到」就完了 —— 实测**系统提示里明确写着**「信息不足时先向用户追问」，所以这是**指令遵循失败**，不是数据缺口。这反而加强了论点。

验证：`python scripts/prepare/inspect_preferences.py`
