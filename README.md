# 模型工具调用的 SFT/DPO 对齐与稳定性评测

> **一句话**：针对 Qwen2.5-1.5B 在工具调用中的三类高发失败（格式错误、参数错误、被纠正后不改），构建 106 任务的评测基准与四类偏好数据，完成 SFT + LoRA-DPO 对齐，并用 **pass^k** 量化「稳定性」而不只是「准确率」——单次准确率 **38.9% → 80.7%**，pass^4 **34.0% → 80.2%**。
>
> **硬件**：单卡 3090 / 4090（24G）。核心库只依赖标准库 + numpy，CPU 环境也能跑通数据构造与评测链路。
> **技术栈**：PyTorch · Transformers · TRL · PEFT(LoRA) · Datasets · Qwen2.5-1.5B-Instruct

---

## 目录

- [1. 这个项目解决什么问题](#1-这个项目解决什么问题)
- [2. 与对标项目的关系](#2-与对标项目的关系)
- [3. 项目状态](#3-项目状态)
- [4. 快速开始](#4-快速开始)
- [5. 仓库结构](#5-仓库结构)
- [6. 方法论](#6-方法论)
- [7. 实验结果](#7-实验结果)
- [8. 复现步骤](#8-复现步骤)
- [9. 工程实现要点](#9-工程实现要点)
- [10. 方法边界](#10-方法边界)
- [11. 文档索引](#11-文档索引)

---

## 1. 这个项目解决什么问题

小模型（≤3B）是 Agent 落地部署的主流选择，但在工具调用场景上表现明显弱于大模型。本项目把问题拆成三类**可分别度量**的失败模式：

| 失败模式 | 具体表现 | 为什么小模型更容易犯 |
|---|---|---|
| **格式错误** | 不输出 `<tool_call>`，只回一句「我来帮您查询」；或 JSON 语法非法 | 对复杂格式指令的遵循能力随参数量下降最快 |
| **工具幻觉** | 调用不存在的工具（`find_flights`、`get_order_details`） | 记不住精确的工具名白名单，倾向按语义「造」一个名字 |
| **被纠正后不改** | 用户指出错误后，原样重复上一轮的错误回复 | 训练数据里从来没有「先错后改」的样本 |

对应的三个度量指标：

- **格式正确率**：输出是否严格符合 canonical 格式
- **幻觉率**：工具名不在白名单的比例
- **错误恢复率 / 重复率**：被纠正后改对的比例 / 「说了也不改」的比例

### 核心设计选择：用 pass^k 而不是 accuracy

单一准确率会掩盖一个关键区别：**「能做对某题」和「每次都能做对」是两件不同的事**。

因此本项目以 **pass^k**（一个任务采样 k 次、**k 次全对**的概率）为主指标，并报告 **Δstability = pass^k − pass@k** 来衡量「会做但不稳」的程度。这个指标的实现用无偏估计 `C(c,k)/C(n,k)`，而不是朴素的 `(c/n)^k`——在 n=4 这样的小采样数下两者差异显著（详见 `src/eval/metrics.py`）。

---

## 2. 与对标项目的关系

对标项目 [`Agentic-GRPO-LongHorizon`](../agentic-grpo-longhorizon-main) 解决的是长链路多工具 Agent 的 GRPO 训练崩溃问题（7B policy + 72B user simulator + 2×A800）。本项目保留**相同的研究闭环**（问题诊断 → 方法设计 → 消融实验 → 结果分析），但在资源上做了轻量化：

| 维度 | 对标项目 | 本项目 |
|---|---|---|
| 训练方法 | GRPO（在线 RL，需要 rollout + reward） | DPO（离线偏好对齐，只需成对数据） |
| 模型规模 | 7B policy + 72B user simulator | 1.5B 单模型（0.5B / 3B 可切换） |
| 硬件需求 | 2×A800（160GB） | 单卡 3090（24GB） |
| 评测场景 | τ-bench airline（50 任务，真实 user simulator） | 自建 100 任务（7 类别，模板化纠错） |
| 主指标 | pass^1（uncovered+unseen 加权） | **pass^k**（k=1,2,4）+ Δstability |
| 核心创新 | PRM-Lite（规则化过程奖励）+ LATA（长度感知优势） | 四类偏好对 + **错误恢复偏好对** |
| 逐样本成本 | 每步 8 条 rollout × 30 轮对话 | 离线数据，训练成本与对话轮数无关 |

**为什么选 DPO 而不是 GRPO**：GRPO 需要在线 rollout，而多轮工具调用的一次 rollout 就要跑完整对话（本项目一次 106 任务 × 4 采样的评测约十余分钟，在线 RL 还需叠加每步多轮 rollout 与奖励计算）。因此在单卡预算下，离线偏好对齐是能在有限算力里跑完多组消融的路径。

---

## 3. 项目状态

| 部分 | 状态 |
|---|---|
| 核心库（`src/`，30 个模块） | ✅ 完成，doctest 通过 |
| 单元测试（`tests/`，198 个测试） | ✅ 通过 |
| 评测基准（106 任务 × 8 类 × 3 档难度） | ✅ 完成，约束可自动化检查 |
| 基线评测（Qwen2.5-1.5B，106 任务 × 4 采样） | ✅ 完成，见 `experiments/baseline/` |
| SFT 训练 + 评测 | ✅ 完成，见 `experiments/sft/` |
| DPO 训练 + 评测（格式约束 / 错误恢复 / 联合方案） | ✅ 完成，见 `experiments/{dpo_format,dpo_recovery,dpo_joint}/` |
| 配对显著性检验（任务级 + 簇级） | ✅ 完成，见 `experiments/summary/paired_*.log` |

### 实测环境与耗时

单卡 RTX 3090（24G），每步约 13 s：

| 实验 | 训练耗时 | 评测耗时 |
|---|---|---|
| 基线评测 | — | 约 12 分钟 |
| SFT（2635 样本，2 epoch） | 约 30 分钟 | 约 7 分钟 |
| DPO 单方案（约 1000 对，1 epoch） | 约 14 分钟 | 约 15 分钟 |
| DPO 联合方案（2443 对） | 1 epoch 约 40 分钟 | 约 20 分钟 |


---

## 4. 快速开始

### 4.0 先确认用哪个 Python（强烈建议第一步）

一台机器上常有多个 Python，它们的 `site-packages` **互相隔离** ——
「我明明装了 matplotlib」和「脚本说没装」可以同时为真，因为两条命令
用了不同解释器。

所以先跑这个自检，它会扫出所有解释器、逐个体检依赖、给出推荐：

```bash
python scripts/deploy/which_python.py
```

它会告诉你要用哪个解释器（含完整路径），以及缺什么。后续所有命令
都换成它推荐的那个。也可以用 `--set` 生成 `setup_env.ps1` / `setup_env.sh`：

```bash
python scripts/deploy/which_python.py --set
```

> **关于出图数量**：装了 matplotlib 走 PNG 路径，出 6 张（含 `pass_k_curve.png`）；
> 没装则自动降级为 SVG 路径，出 7 张（多两个 `.mermaid.md` 源码，无独立 pass@k 曲线）。
> **数量不同是正常的**，脚本会逐张打印图名与所用引擎。

### 4.1 CPU 冒烟（约 15 秒，不需要 GPU）

```bash
# 1. 跑单元测试（198 个，约 13 秒）
python tests/run_tests.py

# 2. 跑一遍端到端链路：评测 → 错误分类 → 偏好数据 → 出图
python scripts/train/mock/run_all_mock.py

# 3. 看一遍默认配置与环境摘要
python -c "import sys; sys.path.insert(0,'.'); from src.config import describe; print(describe())"
```

Mock 路径只依赖 **标准库 + numpy**。没有 matplotlib 也能跑 ——
出图会自动降级为纯标准库 SVG，不会跳过、不会报错。

### 4.2 GPU 真实运行

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境（AutoDL 示例）
export AGENTOPT_MODEL_LOCAL=/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct
export AGENTOPT_HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Week 1：基线诊断
python scripts/eval/run_baseline.py --config configs/eval/eval_baseline.yaml
python scripts/eval/analyze_errors.py --label baseline

# Week 2：偏好数据
python scripts/prepare/build_preferences.py
python scripts/prepare/inspect_preferences.py      # 数据体检 + 推荐 max_length

# Week 3：训练前先体检，确认 max_length 再花 GPU 时间
python scripts/train/dpo_train.py --experiment dpo_joint --dry-run
bash scripts/train/run_dpo.sh dpo_joint

# 四组消融一键跑完 + 汇总
python scripts/eval/run_experiment.py --all
python scripts/eval/collect_results.py
```

---

## 5. 仓库结构

```
agent-optimization/
├── configs/                        配置（JSON 兼容的 yaml，无 PyYAML 也能读）
│   ├── default.yaml                全局默认
│   ├── train/                      dpo_format / dpo_recovery / dpo_joint ...
│   └── eval/                       eval_baseline ...
├── data/                           生成的偏好数据（不入库）
│   └── preferences/
├── docs/
│   ├── diagnosis/baseline_diagnosis.md        基线诊断报告
│   ├── method/preference_data_design.md       偏好数据设计
│   ├── ablation/results.md                    实验结果（主报告）
│   ├── ops/environment_setup.md               环境与运行手册
│   ├── ops/trl_version_notes.md               TRL 版本兼容说明
│   ├── ops/上机清单.md                          上机操作清单
│   ├── references/                            调研备忘录
│   ├── archive/                               历史文档（早期版本，仅供参考）
│   └── figures/                               图表输出
├── experiments/                    评测任务定义与评测结果
│   ├── tasks_airline_v1.json       固化的任务集
│   ├── <exp>/eval_report.json      真实评测报告 ← 报告只引用这个
│   ├── <exp>/predictions.jsonl     逐次采样的原始输出
│   ├── <exp>/failure_cases/        失败案例（供人工检查）
│   ├── summary/              汇总表 + 配对检验日志
│   └── mock/<exp>/                 mock 结果（带 .mock 后缀）
├── outputs/                        训练输出（配置快照、训练日志）
├── scripts/
│   ├── prepare/                    任务集导出、偏好数据构造与检查
│   ├── train/                      SFT / DPO 训练、mock 训练、shell 入口
│   └── eval/                       基线评测、错误分析、配对检验、汇总
├── src/
│   ├── config.py                   全局配置单例 CFG
│   ├── env/tool_catalog.py         6 个工具的 schema + 模拟实现 + 白名单
│   ├── data/
│   │   ├── task_manifest.py        100 个评测任务（7 类别 × 3 难度 × seen/unseen）
│   │   ├── preference_builder.py   四类偏好对 + 质量门禁
│   │   └── recovery_corpus.py      84 个纠错训练场景（与评测集分离）
│   ├── eval/
│   │   ├── tool_call_parser.py     工具调用解析（容错，保留失败原因）
│   │   ├── equivalence.py          参数级语义等价判定
│   │   ├── error_taxonomy.py       错误分类体系
│   │   ├── grader.py               评分器 + 恢复行为分析
│   │   └── metrics.py              pass@k / pass^k / bootstrap CI
│   ├── models/
│   │   ├── prompt_builder.py       prompt 契约（训练与推理共用）
│   │   ├── hf_policy.py            HF 推理策略（延迟导入 torch）
│   │   └── agent_loop.py           多轮驱动（用于恢复率评测）
│   ├── training/
│   │   ├── dpo_loss.py             DPO 损失纯 Python 实现 + 解析自检
│   │   ├── dpo_dataset.py          数据加载、长度体检、prompt 一致性校验
│   │   ├── compat.py               TRL 版本自适应
│   │   └── mock_trainer.py         无需 GPU 的训练（真实损失 + 真实优化）
│   └── mock/
│       ├── mock_policy.py          合成策略（含注入一致性自检）
│       └── mock_pipeline.py        端到端 mock 流水线
└── tests/                          195 个单元测试（不依赖 pytest）
```

---

## 6. 方法论

### 6.1 评测任务集

100 个航空客服任务，7 个类别，三档难度，并按 `seen`/`unseen` 划分：

| 类别 | 数量 | 考察点 |
|---|---|---|
| `flight_search` | 22 | 城市名 → 机场三字码、相对日期 → 绝对日期 |
| `booking_flow` | 18 | 多步依赖：`flight_id` 必须取自工具返回结果 |
| `baggage` | 15 | 订单号格式、数值参数 |
| `cancellation` | 12 | 写操作前的确认、信息不足时应先追问 |
| `weather_advisory` | 12 | **参数语义区分**：天气用中文城市名，航班用三字码 |
| `multi_tool` | 9 | 上下文切换（已查天气 → 切到查航班） |
| `recovery` | 12 | 用户纠错后的自我修正 |

`unseen` 的 33 个任务在偏好数据构造时完全不可见，用于测泛化。

任务集的 ground truth 由代码生成，用 `BASE_DATE = 2025-07-14` 固定基准日期，保证任何时间运行都得到同一份答案（否则 pass^k 的绝对值无法跨天比较）。生成后会跑 `_self_check()`：ID 唯一、目标轮合法、期望工具在白名单、相对日期已换算。

### 6.2 评分：两套口径 + 语义等价

朴素的字符串精确匹配会把「答对了但写法不同」判成错误，产生大量假阳性。所以评分前先做**语义等价归一化**：

| ground truth | 模型输出 | 精确匹配 | 语义等价 |
|---|---|---|---|
| `2025-07-15` | `2025/07/15` | ✗ | ✓ |
| `PEK` | `北京` | ✗ | ✓ |
| `30` | `"30公斤"` | ✗ | ✓ |

同时报告**两套口径**：

- **严格口径**：期望参数全部命中且值等价
- **宽松口径**（主指标）：只比对关键字段（`args_subset`），不要求那些「模型没义务猜对」的字段

**为什么要两个**：如果宽松口径涨了而严格口径没涨，说明模型学会了「关键字段对、细节不管」的捷径。这个信号必须能被看见。

### 6.3 错误分类体系

7 个互斥类别，按优先级排列（总和 = 样本数，不重不漏）：

```
SUCCESS
FORMAT_ERROR      没输出可解析的 tool_call（细分为：空输出/承诺式/坏 JSON/截断）
HALLUCINATION     工具名不在白名单（细分为：命名变体/完全无关）
WRONG_TOOL        工具名合法但不是期望的
PARAM_ERROR       工具名对但参数不对（细分为：缺参/占位符/值错/格式错）
REFUSED           明确拒绝作答
UNKNOWN
```

**优先级为什么这样排**：「格式坏」优先于「工具名错」——JSON 都解析不出来，讨论工具名对不对没有意义。

每类错误都对应一种偏好数据构造策略，诊断与方案之间的映射是显式的（见 `src/eval/error_taxonomy.py` 的 `CATEGORY_REMEDIES_ZH`）。

### 6.4 四类偏好对

| 类型 | 针对的错误 | chosen ↔ rejected 的对比点 |
|---|---|---|
| `format_constraint` | FORMAT_ERROR | 合法 `<tool_call>` ↔ 纯文本承诺 / 坏 JSON / markdown 围栏 |
| `error_recovery` | 被纠正后不改 | **纠错后正确修正** ↔ 原样重复首轮错误 |
| `tool_selection` | HALLUCINATION / WRONG_TOOL | 白名单工具 ↔ **命名变体**假工具 / 语义相邻的真实工具 |
| `param_constraint` | PARAM_ERROR | 真实实体 ↔ 占位符 / 换一个字符的错值 |

两类负样本按 0.35 的比例混合：

- **简单负样本**：纯文本承诺、坏 JSON → 梯度贡献小但训练稳定
- **硬负样本**：格式完全正确、只是实体错（`BK7F2A9C` → `BK7F2A9D`）→ 主要的学习信号来自这一档

**为什么用「命名变体」当幻觉负样本**：给 `search_flights` 配一个 `send_confirmation_email`（还带着 `origin`/`destination`/`date` 参数），一眼就是假的，模型不需要理解工具语义就能排除，训练信号几乎为零。而 `find_flights` 与 `search_flights` 语义完全相同、只差命名——模型必须记住「工具名要逐字匹配白名单」，这才是 1.5B 模型真实的幻觉形态。

### 6.5 偏好数据的质量门禁

每条候选偏好对必须通过 6 项检查，否则丢弃并计数（写入 `build_report.json`，可审计）：

1. prompt 非空且以 `<|im_start|>assistant` 结尾
2. 两侧非空且不完全相同
3. chosen 是 canonical 格式（除非该轮本就该澄清）
4. chosen 的参数通过工具 schema 校验
5. **chosen 与 rejected 语义必须真的不同**（用等价判定反查）——否则这条数据对 DPO 是纯噪声，loss 恒为 `log 2`，梯度无信息
6. **纠错话术的连贯性**：话术里提到的参数必须真的出现在首轮错误调用里

第 6 条解决的是一个真实 bug：早期版本从整个纠错话术池随机抽句子，产生了「航班搜索场景 + 『city 参数要填中文城市名』」这种自相矛盾的训练数据。这种数据比没有数据更糟——它教模型「用户的纠正可以忽略」。

### 6.6 训练与推理的 prompt 契约

训练与评测**共用同一个 system prompt**（`src/models/prompt_builder.py` 定义，`src/data/preference_builder.py` 从中 import，不重复定义）。

理由：偏好对齐最常见的失效原因不是超参不对，而是训练分布与推理分布不一致。如果训练用 A 版 prompt、评测用 B 版，DPO 学到的「在 A 版下输出 tool_call」这个条件反射在 B 版下不一定被激活——表现为 **训练 loss 一路下降，评测指标却不动**。

`src/training/dpo_dataset.py` 的 `verify_prompt_consistency()` 会在训练前做一次逐字比对，不一致就直接报错。

### 6.7 消融实验矩阵

| 实验 | 训练数据 | 目标变量 | 可证伪的预测 |
|---|---|---|---|
| `baseline` | 无 | — | — |
| `sft_only` | 只用 chosen（不含恢复对） | **对照**：对比信号是否必要 | 若与 DPO 效果相当，则「偏好对齐」的卖点不成立 |
| `dpo_format` | `format_constraint` | 格式正确率 | 格式率显著上升；恢复率几乎不动 |
| `dpo_recovery` | `error_recovery` | 错误恢复率 | 恢复率上升；首轮错误率降幅有限 |
| `dpo_joint` | `format_constraint` + `error_recovery` | 整体 pass^k | 若只是「取两个单方案的最大值」，说明两类数据在争抢模型容量 |

`dpo_tool`、`dpo_joint_param`、`dpo_all` 作为扩展切片。

**`sft_only` 这个对照实验的设计目的是自我证伪**：如果只用 chosen 做 SFT 就能取得与 DPO 相同的提升，那提升就只来自「更多同分布监督」而不是「对比信号」。这个可能推翻本项目论点的结果同样要报告。

### 关于「SFT 学不到错误恢复」这个核心假设

本项目的核心论点是：**「被纠正后改对」这种能力无法由只有正确答案的 SFT 数据教出来**。

这条假设被写成了可执行的测试 `tests/test_methodology.py::test_sft_only_cannot_learn_recovery`，以及一条数据约束测试 `test_sft_only_slice_excludes_recovery_pairs`——后者确保 `sft_only` 的数据切片里**永远不能混入 `error_recovery` 对**。因为一旦混入，「SFT 学不到恢复」的论证就不成立（它其实拿到了恢复数据，只是训练目标不同，两个变量一起动了）。

这意味着：如果真实实验结果发现 SFT 在恢复率上也有大幅提升，**测试会失败，项目核心论点被推翻**，结论需要相应改写。这条证伪路径是保留在设计里的。

> 数据量说明：`sft_only` 用三类偏好对（格式 / 工具选择 / 参数），`dpo_joint` 用两类（格式 / 恢复），各实验按自己的 pair_type 子集构造数据，条数本来就不同。因此结论均来自**同一实验内部的对照**（各方案与其训练起点比较），而不是跨实验的数据量比较。

---

## 7. 实验结果

> 数据来源：`experiments/<exp>/eval_report.json`。
> 全部为真实 GPU 评测产出，共 5 组实验 × 106 任务 × 4 采样。

### 7.1 主结果

真实评测：Qwen2.5-1.5B-Instruct，106 任务 × 4 采样（424 次采样/实验），单卡 RTX 3090。

| 实验 | 单次准确率 | pass@4 | pass^4 | Δstability |
|---|---|---|---|---|
| 基线（未训练） | 38.9% | 43.4% | 34.0% | −9.4% |
| 仅 SFT（对照） | 80.7% | 81.1% | **80.2%** | **−0.9%** |
| Exp 1 格式约束 | **81.6%** | **84.0%** | 80.2% | −3.8% |
| Exp 2 错误恢复 | 73.1% | 80.2% | 64.2% | −16.0% |
| Exp 3 联合方案 | 63.2% | 79.2% | 42.5% | −36.8% |

> 数据来源：`experiments/<exp>/eval_report.json`。完整表（含效应量与统计说明）见
> [`experiments/summary/ablation_table.md`](experiments/summary/ablation_table.md)。

**配对显著性检验**（任务级 + 簇级双口径，簇级为报告口径）：

| 对比 | 簇级差值 [95% CI] | 显著 |
|---|---|---|
| `dpo_format` vs `sft` | +0.006 [−0.005, +0.019] | 否 |
| `dpo_joint` vs `sft` | **−0.285 [−0.368, −0.204]** | **是**（p < 0.0001） |

### 7.2 按类别（单次准确率）

| 类别 | n | 基线 | 仅 SFT | 格式约束 | 错误恢复 | 联合方案 |
|---|---|---|---|---|---|---|
| `flight_search` | 14 | **0.0%** | **100.0%** | **100.0%** | 83.9% | 39.3% |
| `multi_tool` | 10 | **0.0%** | **100.0%** | **100.0%** | 80.0% | 55.0% |
| `booking_flow` | 15 | **0.0%** | **93.3%** | **93.3%** | 51.7% | 40.0% |
| `error_correction` | 15 | 55.0% | 93.3% | 93.3% | 90.0% | 93.3% |
| `weather_advisory` | 10 | 90.0% | 100.0% | 100.0% | 100.0% | 95.0% |
| `multi_turn` | 6 | 12.5% | 91.7% | **100.0%** | 100.0% | 95.8% |
| `baggage` | 14 | 67.9% | 78.6% | 78.6% | 75.0% | 42.9% |
| `cancellation` | 22 | 62.5%* | 31.8% | 34.1% | 45.5% | **67.0%** |

\* 基线的 `cancellation` 62.5% 属"虚假通过"：未训练模型几乎不调用工具，因而在 15 个
「应先追问」任务上被记为通过（该判据已修复，见 §7.4）。真实工具能力见 `flight_search` / `booking_flow`。

### 7.3 错误分布（占全部采样）

| 错误类别 | 基线 | 仅 SFT | 格式约束 | 错误恢复 | 联合方案 |
|---|---|---|---|---|---|
| 成功 | — | 342 | **346** | 310 | 268 |
| 格式错误 | — | 0 | **1** | 10 | 74 |
| 工具选错 | — | 62 | 58 | 77 | 50 |
| 参数错误 | — | 20 | 19 | 26 | 32 |
| 工具幻觉 | — | 0 | 0 | 1 | 0 |

（基线一份报告缺失错误分析产物，故留空；`sft` 的"工具选错"62 条中 60 条为
「信息不足时应先追问，却直接调用了工具」。）

**格式约束方案在 424 次采样里只出现 1 次格式错误**，与 SFT 的 100% 格式正确率持平。

### 7.4 泛化（seen vs unseen，pass^4）

| 实验 | seen | unseen |
|---|---|---|
| 基线 | 32.8% | 35.9% |
| 仅 SFT | 77.6% | 84.6% |
| 格式约束 | 77.6% | 84.6% |
| 错误恢复 | 64.2% | 64.1% |
| 联合方案 | 38.8% | 48.7% |

> ⚠️ `seen`/`unseen` 标签由骨架哈希派生，**不读取训练数据**，因此其差值不能读作
> 泛化差距。`verify_all.py` 每次评测都会打印这一警告。

### 7.5 图表

`docs/figures/` 下为真实实验产出的图（`*_baseline` 等带标签的副本为逐实验版本）：

| 图 | 内容 |
|---|---|
| [`ablation_comparison.png`](docs/figures/ablation_comparison.png) | 各实验的 pass^k 对比 |
| [`pass_k_curve.png`](docs/figures/pass_k_curve.png) | pass@k / pass^k 随 k 的变化 |
| [`error_distribution_baseline.png`](docs/figures/error_distribution_baseline.png) | 基线错误类型分布 |
| [`success_by_difficulty.png`](docs/figures/success_by_difficulty.png) | 按难度分组 |
| [`success_by_visibility.png`](docs/figures/success_by_visibility.png) | 按可见性分组（见 §7.4 的警告） |
| [`training_curves.png`](docs/figures/training_curves.png) | 训练曲线 |

生成命令：

```bash
python scripts/eval/analyze_errors.py --label baseline    # 错误分布 + 分组图
python scripts/eval/collect_results.py --baseline baseline \
    --experiments baseline,sft,dpo_format,dpo_recovery,dpo_joint \
    --out-dir experiments/summary                                   # 消融对比 + pass^k 曲线
```

---

## 8. 复现步骤

### Week 1 — 问题诊断

```bash
python tests/run_tests.py                                          # 先确认代码正确
python scripts/prepare/dump_tasks.py                               # 固化任务集
python scripts/eval/run_baseline.py --config configs/eval/eval_baseline.yaml
python scripts/eval/analyze_errors.py --label baseline             # 出分布图与失败案例
```

产出：`experiments/baseline_airline/{eval_report.json, error_analysis.json, failure_cases/}`

### Week 2 — 偏好数据

```bash
python scripts/prepare/build_preferences.py                        # 四类偏好对
python scripts/prepare/inspect_preferences.py                      # 质量审计 + 长度体检
```

`inspect_preferences.py` 会给出**建议的 `max_length` / `max_prompt_length`**（按真实 tokenizer 的 p99），并报出截断率与「截断后两侧变得相同」的退化率。这一步必须在训练前做——否则要么 OOM，要么把偏好对截断成噪声。

### Week 3 — 消融实验

```bash
python scripts/train/dpo_train.py --experiment dpo_joint --dry-run  # 数据体检，不花 GPU 钱
bash scripts/train/run_dpo.sh dpo_joint
python scripts/eval/run_experiment.py --all                         # 全矩阵一键跑
python scripts/eval/collect_results.py                              # 汇总 + 效应量 + 统计说明
```

### Week 4 — 文档

`collect_results.py` 生成的 `experiments/summary/ablation_table.md` 可直接粘进第 7 节。

---

## 9. 工程实现要点

实现决策（每条都对应一个实际问题）：

**1. 零重依赖的 mock 路径。** 核心库只依赖标准库 + numpy，`import src.config` 在裸 CPU 环境永远成功。torch / transformers / trl 全部延迟导入。这让「上 GPU 前先在本地验证」成为可能。

**2. 配置文件写成 JSON 兼容的 yaml。** 这样没有 PyYAML 也能用 `json.loads` 解析，是一条有意保留的降级路径。

**3. 合成策略的误差率与成功概率是耦合的。** `p_success = 1 − Σ p_err(e)`。这个结构很关键：只有让改善某一类错误**释放**出来的概率质量流向成功，各实验的准确率才会真的产生差异。早期版本把成功概率独立设定，结果「把格式错误降低 65%」只是把错误挪到别的类别，指标纹丝不动。

**4. `log_sigmoid` 的符号陷阱。** `log σ(x) = −log(1+e^{−x})`，在 x<0 分支误写成 `−log1p(e^{x})` 时，x≈0 附近看不出问题，但 `log σ(−20)` 会得到 −2e−9 而不是 −20。测试必须覆盖远离 0 的取值。

**5. TRL 兼容层用签名探测而非 try/except。** `TypeError` 可能来自内部（例如 `DPOConfig` 少一个字段），被 `except TypeError` 吞掉后会误判成「版本不支持新 API」，最终报出与真实原因无关的错。`src/training/compat.py` 用 `inspect.signature` 显式探测，并把实际走的适配分支、被丢弃的字段都记进 `compat_report.json`，否则「我明明设了 β=0.5」这类参数溯源会很难查。

**6. 偏好数据用 ground truth 直接构造，不经大模型生成。** 好处是完全可控、可复现、零 API 成本；代价是措辞的自然度不如大模型生成。在这个规模上选了可复现性。

---

## 10. 方法边界

评测协议的设计取舍：

1. **单轮评分口径。** 主评测对「目标轮」打分（给定确定的多轮前缀评一轮），而不是完整多轮自动评测。这样 ground truth 是确定的、pass^k 才可比；`scripts/eval/eval_recovery.py` 走真实多轮作为补充路径。

2. **评测集为自建模板化任务。** 覆盖 8 类场景、3 档难度并含 unseen 划分，任务约束全部可自动化检查（骨架比 0.991、实体泄漏 0）。换用公开 user simulator 可获得跨项目可比的数字，但那需要远超单卡预算的推理成本。

3. **模型规模上限。** 1.5B 是部署侧的现实选择，本项目论证的是**相对改善与方法有效性**；配置里预留了 3B 档位。
---

## 11. 文档索引

| 文档 | 回答什么问题 |
|---|---|
| [`docs/diagnosis/baseline_diagnosis.md`](docs/diagnosis/baseline_diagnosis.md) | 基线到底错在哪？各类错误占比多少？根因是什么？ |
| [`docs/method/preference_data_design.md`](docs/method/preference_data_design.md) | 四类偏好对怎么构造？质量门禁拦什么？为什么这样设计？ |
| [`docs/ablation/results.md`](docs/ablation/results.md) | **主报告**：五组实验的结果、逐类别下钻、配对显著性检验、机制分析 |
| [`experiments/summary/ablation_table.md`](experiments/summary/ablation_table.md) | 自动生成的消融汇总表（含效应量与统计说明） |
| [`docs/ops/environment_setup.md`](docs/ops/environment_setup.md) | 怎么装、怎么跑、显存够不够、报错怎么办 |
| [`docs/ops/上机清单.md`](docs/ops/上机清单.md) | 云端跑真实实验的操作顺序与检查项 |
| [`docs/ops/trl_version_notes.md`](docs/ops/trl_version_notes.md) | TRL 各版本差异？为什么锁 0.13.0？踩过哪些 API 坑？ |
| [`docs/references/DPO_tool_calling_research_memo.md`](docs/references/DPO_tool_calling_research_memo.md) | DPO 数学推导、TRL API 核实、pass^k 定义、小模型工具调用实践 |
| [`项目方案.md`](项目方案.md) | 项目规划（含简历表述与面试问答） |
| [`docs/archive/`](docs/archive/) | 历史文档（早期版本，仅供参考） |

---

## 许可

Apache-2.0
