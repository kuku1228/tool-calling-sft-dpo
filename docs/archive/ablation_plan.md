# 消融实验计划：格式约束 / 错误恢复 / 参数约束偏好对各自贡献多少

> **文档定位**
>
> - **回答的问题**：本项目的 5 个核心实验 + 3 个扩展切片分别验证什么假设？为什么必须要有 `sft_only` 对照组？每个实验的可证伪预测是什么？评测协议如何保证可比？什么现象出现时必须停下检查？
> - **读者**：本项目作者（Week 3 按此执行的人）、代码评审者、面试官（想看「有没有实验设计能力，而不只是会跑脚本」）。
> - **在闭环中的位置**：`docs/diagnosis/baseline_diagnosis.md`（基线诊断）的下游。诊断给出「哪类错误占主要份额」，本文档给出「用什么数据、按什么顺序、如何证伪地去修」。
> - **配套文档**：`docs/diagnosis/baseline_diagnosis.md`（§8 的映射表是本文档 §1 的选题依据）、`../../项目方案.md`（项目规划；其实验编号与本文档不同，见 §10.1）。
>
> **本文档只写计划与设计原理，不含任何实际结果。** 所有需要真实数据的位置都是 `<!-- TODO(真实数据): ... -->` 占位，表格给完整表头 + 一行占位示例。
> 引用代码常量时标注来源；引用 mock 假设值时明确说明它是**假设值**。


> ⚠️ **执行结果请看 [`ablation_report.md`](ablation_report.md)。**
> 本文件是 **Week 3 的实验计划**（设计与预期），其中的数值位置保持为占位符。
>
> **计划与实际的偏离**（真实基线数据推翻了两条核心假设）：
>
> | 原计划 | 实测 | 后果 |
> |---|---|---|
> | 工具幻觉率约 25% | **0.0%**（400 样本 0 条编造工具名） | `dpo_tool`（工具选择对）**取消** |
> | 格式错误以 JSON 语法错为主 | **100% 是「不调工具」**（109 条中 0 条语法错） | `json_broken` 方向作废 |
> | — | 新发现 `action_hallucination` 41 条 | 原分类体系未覆盖，已补 |
>
> 因此实际执行的实验为 `baseline_airline` / `sft_only` / `sft_v2` /
> `sft_v3` / `dpo_joint` / `dpo_from_sft`，**不是**本计划里的
> `dpo_format` / `dpo_recovery` / `dpo_joint` 三组。
---

## 目录

- [§1 实验矩阵](#§1-实验矩阵)
- [§2 为什么必须要有 `sft_only` 对照组](#§2-为什么必须要有-sft_only-对照组)
- [§3 每个实验的详细设计](#§3-每个实验的详细设计)
- [§4 评测协议](#§4-评测协议)
- [§5 主指标与判据](#§5-主指标与判据)
- [§6 混淆因素排查](#§6-混淆因素排查)
- [§7 产物与目录约定](#§7-产物与目录约定)
- [§8 预期结果汇总表](#§8-预期结果汇总表)
- [§9 风险与触发条件表](#§9-风险与触发条件表)
- [§10 附录](#§10-附录)

---

## §1 实验矩阵

### 1.1 实验定义来自代码，不是本文档自造

代码事实：`src/data/preference_builder.py` 的 `EXPERIMENT_PAIR_TYPES`（第 889-900 行）是**消融矩阵的可执行定义**。训练脚本 `scripts/train/dpo_train.py --experiment <name>` 用 `--experiment` 的值去查这张表筛数据，所以「实验名」在仓库里只有这一处定义。

代码事实：四类偏好对的常量与中文名（同一文件第 80-92 行）：

| 常量 | 中文 | 针对的错误（模块文档原文） | chosen ↔ rejected 的对比点 |
|---|---|---|---|
| `format_constraint` | 格式约束对 | `FORMAT_ERROR`（占比最高） | 合法 `<tool_call>` ↔ 纯文本承诺 |
| `error_recovery` | 错误恢复对 | `NO_RECOVERY` / 纠错轮失败 | **纠错后正确修正** ↔ 原样重复错误 |
| `tool_selection` | 工具选择对 | `HALLUCINATION` / `WRONG_TOOL` | 白名单工具 ↔ 貌似合理的假工具 |
| `param_constraint` | 参数约束对 | `PARAM_ERROR` | 真实实体 ↔ 占位符 / 错格式 |

### 1.2 核心实验矩阵

下表「训练数据」列**逐字来自** `EXPERIMENT_PAIR_TYPES`；「目标变量」是本项目的实验设计意图；「预期改善指标」的取值来自 `CATEGORY_REMEDIES_ZH` 与 `MockPolicyCfg.mitigation_factors`（**后者是假设，不是结论**，见 §1.4）。

| 编号 | 实验名 | 训练数据（`pair_type`） | 目标变量 | 预期改善的指标 | **可证伪的预测** |
|---|---|---|---|---|---|
| Exp 0 | `baseline` | 无（未训练基座） | 参照系 | —（作为全部分母） | 无（不是实验，是基线） |
| Exp C | `sft_only` | `format_constraint` + `tool_selection` + `param_constraint`（**故意不含 `error_recovery`**） | 「更多同分布监督」vs「对比信号」 | `FORMAT_ERROR` / `HALLUCINATION` / `PARAM_ERROR` 占比下降；**`repeat_rate` 基本不变** | 若 `sft_only` 在全部指标上与 `dpo_format`/`dpo_joint` 无显著差异 → **本项目的核心卖点（对比信号）不成立**，结论需按此改写 |
| Exp 1 | `dpo_format` | `format_constraint` | 格式约束的**单独**贡献 | `format_accuracy`↑、`FORMAT_ERROR` 占比↓、accuracy / pass^k↑ | 若格式率上升但 accuracy / pass^4 不动 → 只教会了「加标签」，没教会「做对任务」（§3.2 的 R1 风险） |
| Exp 2 | `dpo_recovery` | `error_recovery` | 恢复能力的**单独**贡献 | `repeat_rate`↓、`first_try_rate` 不降、`task_success_rate`↑ | 若 `repeat_rate` 降了但 `first_try_rate` 也显著下降 → 模型学到的是「先错后改」的模式而不是「改对」（§3.3 的 R2 风险） |
| Exp 3 | `dpo_joint` | `format_constraint` + `error_recovery` | 两类效应是否**叠加**（超加性） | 格式率与恢复率同时上升；总提升 > 两个单方案之和 | 若联合提升只是「取两个单方案的最大值」→ 两类数据在争抢同一份模型容量，是**需要报告的负面结果**（`configs/train/dpo_joint.yaml` 的 `_falsifier` 字段原文） |

### 1.3 扩展切片

代码事实（`EXPERIMENT_PAIR_TYPES` 剩余四项）：

| 实验名 | 训练数据（`pair_type`） | 作用 | 与核心实验的关系 |
|---|---|---|---|
| `dpo_tool` | `tool_selection` | **单独定位幻觉/选错工具的贡献**。三类错误的第三类此前没有被单独消融过 | 补齐「四类偏好对各自贡献」的最后一块 |
| `dpo_tool_param` | `tool_selection` + `param_constraint` | 「工具名 + 参数」都对了，但缺少格式与恢复训练 | 用来分离「工具/参数类能力」与「格式/恢复类能力」两组效应 |
| `dpo_joint_param` | `format_constraint` + `error_recovery` + `param_constraint` | 在 Exp 3 上**只加一类**（参数约束） | **单变量增量实验**：`dpo_joint_param − dpo_joint` = 参数约束对的边际贡献 |
| `dpo_all` | 四类全部 | 上限参考 | 用来回答「全都加是不是最好」，以及「加得越多是否越可能出现互相干扰」 |

**为什么这些扩展切片是必要的**：核心矩阵只能回答「某几类的组合有没有用」，回答不了「某一类单独的边际贡献是多少」。`dpo_joint_param` 与 `dpo_joint` 的差就是参数约束对的**增量效应**，这是最干净的单变量对比 —— 它比「`dpo_all` vs `baseline`」有信息量得多。

> **⚠️ 扩展切片没有 mock 支持**（代码事实，见 `docs/diagnosis/baseline_diagnosis.md` 附录 A14）：
> `src/mock/mock_policy.py` `MockPolicy.__post_init__` 会校验 `experiment in cfg.mitigation_factors`，而 `mitigation_factors` 只有 5 个键（`baseline / dpo_format / dpo_recovery / dpo_joint / sft_only`）。
> 因此 `MockPolicy("dpo_tool")` 会直接抛 `ValueError`。**四个扩展切片必须走真实 GPU**，无法先用 mock 验证流水线。

### 1.4 关于「预期改善指标」中的 mock 假设值

`src/config.py` 的 `MockPolicyCfg.mitigation_factors` 与 `recovery_gain` 给出了一张「哪个实验改善哪一类错误」的**乘性系数表**。必须清楚它的性质：

| 项 | 性质 | 用途 |
|---|---|---|
| `mitigation_factors[exp][err]` | **配置假设值**（我们喂给 mock 的）；源码注释明确写着「这些数字是**待验证的假设值**，不是本项目声称的结论」 | 只在没有 GPU 时跑通「评测→分类→偏好构造→训练→再评测→画图」整条流水线，验证代码无 bug |
| `recovery_gain[exp]` | 同上（对纠错轮恢复概率的额外乘子） | 同上 |
| `EXPERIMENT_PAIR_TYPES` | **代码事实**（实验切片的定义） | 决定每个实验喂哪些数据 |

**本文档不抄录这些假设值的具体数字**，避免被误引用为「预期效果」。要看数值请直接读 `configs/default.yaml` 的 `mock` 段或 `src/config.py` 的 `MockPolicyCfg`，并同时阅读其 `⚠️` 警告。

**mock 的结论性产物一律不得作为实验结果引用**：`outputs/mock_pipeline/`、`outputs/_facts.json`、`experiments/mock/`、`experiments/mock_baseline/`。

### 1.5 依赖关系与执行顺序

```text
0. python -m src.data.task_manifest          # 任务集自检（100 个任务）
1. python tests/run_tests.py                 # 方法论测试（pass^k 公式 / DPO 损失 / 数据门禁 / mock 一致性）
2. python scripts/prepare/build_preferences.py   # 构造真实偏好数据 → data/preferences/*.jsonl
3. python scripts/eval/run_baseline.py --label baseline_airline      # Exp 0（必须先做，它是分母）
4. python scripts/train/dpo_train.py --experiment dpo_format  ...    # Exp 1
5. python scripts/eval/run_baseline.py --adapter outputs/dpo_format/final --label dpo_format
6. 同 4-5 做 dpo_recovery / dpo_joint
7. sft_only 对照组（见 §2.4，训练脚本不同）
8. （可选）4 个扩展切片
9. python scripts/eval/eval_recovery.py --adapter <每个实验的 final> --label <实验名>   # 逐实验的恢复评测
10. python scripts/eval/collect_results.py --experiments baseline,sft_only,dpo_format,dpo_recovery,dpo_joint
```

**为什么 baseline 必须第一个做**：`compare_reports()` 与 `collect_results.py` 的「相对提升 / 效应量」全部相对基线计算（`--baseline` 默认 `baseline`）。基线缺失时脚本会打印「基线实验 'baseline' 不在结果里，相对提升无法计算」。

**为什么每个实验训练完立刻评测**：DPO 的过拟合往往在第 2 个 epoch 就开始了（`CFG.train.epochs = 2`，`save_total_limit = 3`）。训练完立刻评测能拿到「哪个 checkpoint 最好」的信息；攒到最后一起评则会丢失中间 checkpoint 的对比机会。`dpo_train.py --eval-after` 提供了「训练后自动评测」的路径。

---

## §2 为什么必须要有 `sft_only` 对照组

> **这是全篇最重要的方法论论证。** 如果这一节说不清楚，本项目的「偏好对齐」卖点就是空的。

### 2.1 核心风险：提升可能只来自「更多同分布监督」

偏好对数据由三部分组成：`prompt` / `chosen` / `rejected`。

- **DPO**：同时使用 `chosen` 和 `rejected`，学习「**chosen 应该比 rejected 更可能**」这个**对比信号**；
- **SFT（只用 chosen）**：只使用 `chosen`，学习「在给定 prompt 下输出 chosen」这个**监督信号**。

现在考虑一个非常现实的可能性：

> 我们构造了 2400 条格式 + 恢复偏好对（`CFG.preference.n_format = 1200`、`n_recovery = 1200`，**这是配置目标值**）。这 2400 条数据的 prompt 全部来自同一批航空客服任务模板。如果把它们**只用 chosen 侧**做成 SFT 数据，模型也能获得 2400 个「在该任务上下文中输出合法 tool_call」的监督样本。

**如果只用 chosen 做 SFT 就能取得与 DPO 相同的提升，那么本项目的实验结论就只能是「加数据有用」，而不是「偏好对齐有用」。** 这个结论不算错，但它把项目从「DPO 偏好对齐研究」降级成「数据增强研究」——而 `rejected` 那一半（我们精心挑的硬负样本）成了纯浪费。

**所以必须主动设计并报告这个可能推翻自己论点的对照实验。** 一个不做对照组的消融实验，无论结果多好，都无法排除「提升来自数据量」这个平庸解释。

### 2.2 `sft_only` 的设计（代码事实）

代码事实（`EXPERIMENT_PAIR_TYPES["sft_only"]`）：

```python
"sft_only": (PAIR_FORMAT, PAIR_TOOL_SELECTION, PAIR_PARAM),
```

对照组的三个设计决策，每一个都有明确理由：

| 决策 | 做法 | 理由 |
|---|---|---|
| **保留哪些类型** | `format_constraint` + `tool_selection` + `param_constraint` | 这三类覆盖「**要学什么**」：格式、工具名、参数。它们是「正确答案本身就能教会」的东西 —— chosen 侧就包含了正确格式、正确工具名、正确参数 |
| **故意排除哪一类** | **不含** `error_recovery` | 见 §2.3，这是整个对照设计的关键 |
| **用什么训练器** | 标准 SFT（`chosen` 作为监督目标），不是 `DPOTrainer` | 若用 `DPOTrainer` 只喂 chosen 侧，训练目标就不是「模仿正确答案」而是「无条件提高 chosen 的 logprob 相对于 reference」——那已经不是 SFT 的语义了。对照组要干净，就必须用 SFT 训练器 |

**数据量必须对齐（否则对照失效）**：`sft_only` 的数据条数应尽量接近被对照的 DPO 实验。代码事实：`sft_only` 的三类合计目标条数（1200 + 600 + 1200 = 3000，按 `PreferenceCfg` 的目标值）**多于** `dpo_joint`（1200 + 1200 = 2400）。这意味着：

- 如果想让对比更严格，应当**下采样** `sft_only` 到与对照实验同样条数，或反过来给 `dpo_joint` 补齐同量的 `tool_selection`/`param_constraint`（也就是比较 `dpo_all`）。
- 报告里**必须明说**实际用了几条，不能只说「SFT 对照组」。

> ⚠️ 这一条是本计划里最容易执行出偏差的地方。建议在 `docs/ablation/` 下另存一份 `dpo_train` 的 `data_stats`（`dpo_train.py` 会把数据体检验血报告写到 `outputs/<exp>/data_health.json`，含 `num_train`、`by_pair_type`、`negative_kind` 分布），报告里直接引用它作为「实际条数」的权威来源。

### 2.3 为什么 `sft_only` **故意不含** `error_recovery`

**恢复能力无法由「只有正确答案」的数据教出来。**

这不是一句设计口号，它有明确的机理解释。看 `error_recovery` 偏好对的 prompt 结构（代码事实，`preference_builder._render_scenario_prompt()`）：

```text
<|im_start|>system
<完整 system prompt，含 6 工具手册与 5 条规则><|im_end|>
<|im_start|>user
<用户的第一次请求><|im_end|>
<|im_start|>assistant
<模型上一轮的错误回复（例如纯文本承诺 / 假工具 / 占位符参数）><|im_end|>
<|im_start|>user
<用户的纠错话术（例如「你没有真的调用工具，请用规定的 <tool_call> 格式重新输出」）><|im_end|>
<|im_start|>assistant
                                              ← 模型在这里生成
```

`chosen` = 「纠错后正确修正」，`rejected` = 「原样重复首轮错误」/「退化成纯文本」/「换个工具但还错」/「工具对了参数还错」。

关键在于：**这条 prompt 里包含了一个「模型曾经犯错」的上下文**。

- 对 DPO 而言，`rejected` 恰恰就是「在这个上下文里继续犯错」，`chosen` 是「在这个上下文里改对」。对比信号直接塑造了「看到用户的纠正 → 改变行为」这个映射。
- 对 SFT（只用 chosen）而言，模型看到的是「在包含一个错误回复 + 一句纠错的上下文之后，应当输出什么」。它学到的更像「**在长上下文里继续输出合理内容**」，而不是「**识别到自己被纠正了 → 必须改变行为**」。

更根本的一点（`preference_builder` 模块文档原文）：「SFT 的 chosen 里只会有『正确答案』，模型学不到『先错后改』这个模式。」如果 SFT 数据是从**正常任务**（prompt 里模型没犯错）构造的，那么 prompt 里根本没有「错误」这一状态，模型不可能学会从这个状态恢复。**这就是为什么 `sft_only` 不能包含恢复类样本 —— 不是「我们不想给」，而是「给了也不是同一件事」。**

顺带一个实现细节（代码事实）：`build_recovery_pairs()` 用的场景来自 `src/data/recovery_corpus.py` 的参数化展开（12 个模式 × 7 个变体 = 84 个场景），**不是评测集的 12 个 recovery 任务**。模块文档写明了这个边界：

```
task_manifest.build_tasks()       → 100 个任务，12 个 recovery，**用于评测**
recovery_corpus.build_scenarios() → ~84 个场景，**只用于构造训练偏好对**
```

两者的 `mistake_key` 用同一套 key（`format_ignore_correction` 等），所以「训练里教了什么」与「评测里考了什么」可以逐类对齐 —— 这是 §3.3 能做 per-mistake-type 归因分析的前提。

### 2.4 `sft_only` 的执行方式（待实现，需显式声明）

代码事实：`scripts/train/dpo_train.py` **只实现 DPO 路径**，`--experiment sft_only` 会走到 `run_real()` → `DPOTrainer`。因此当前仓库**没有**独立的 SFT 训练入口。

Week 3 执行 `sft_only` 时需要以下三种方式之一（按推荐度排序）：

| 方式 | 做法 | 优点 | 风险 |
|---|---|---|---|
| A（推荐） | 新增 `scripts/train/sft_train.py`，用 `trl.SFTTrainer`（或 transformers `Trainer`），数据来自 `build_experiment_dataset("sft_only", ...)` 的 **`chosen` 列**，超参与 DPO 尽量对齐（LoRA r/alpha、lr、epochs、batch、max_length） | 语义干净；产物可与其他实验同构 | 需要新写脚本（不在本次交付范围内，需自行实现） |
| B | 用一个 `sft_only` 专用 jsonl：把每条偏好对转成 `{"prompt": ..., "completion": chosen}` 再喂 SFTTrainer | 复用现有偏好数据 | 需要写转换脚本；`prompt` 已含 `<|im_start|>` 模板标记，需注意不要重复套 template |
| C（不推荐） | 用 DPO 但把 `rejected` 设成与 `chosen` 相同 | 几乎不用改代码 | 语义错误（DPO loss 恒为 log 2，等于没训练）；**不要用** |

**无论用哪种方式，报告里必须写清**：`sft_only` 用的训练器、条数、超参、以及它与被对照的 DPO 实验在数据量上的差异。

### 2.5 `sft_only` 的三种可能结果与解读（预注册）

在跑之前先把解读规则写下来，避免事后挑对自己有利的解释（这是「预注册」的核心价值）：

| 结果 | 现象 | 解读 | 报告怎么写 |
|---|---|---|---|
| **R-A** | `sft_only` 的提升**显著小于** `dpo_joint`（尤其在 pass^4 与 `repeat_rate` 上） | **支持本项目论点**：对比信号有独立贡献 | 报告主论点；给出 `sft_only` 与各 DPO 实验的 CI 重叠性判断 |
| **R-B** | `sft_only` 与 `dpo_joint` 的提升**无显著差异**（置信区间大幅重叠） | **推翻本题论点**：提升来自同分布监督的量，而非对比信号 | 需把结论改成「在 1.5B 规模 + 本数据规模下，DPO 的对比信号未带来可检出增益」。此时可讨论：是数据量不足、还是 β/epochs 未调好、还是任务太简单导致 DPO 与 SFT 等价 |
| **R-C** | `sft_only` 在格式/工具/参数上不错，但在**恢复相关指标**（`repeat_rate`、`task_success_rate` 于 recovery 子集）上显著更差 | **最理想的结果**：正好验证「恢复能力无法由只有正确答案的数据教出来」 | 主报告的核心证据。必须给出 `repeat_rate` 的绝对差值与 CI |

> **无论 R-A/R-B/R-C，`sft_only` 的结果都需一并给出。** 只报告有利实验的消融表不完整。R-B 也不等于项目失败：设计一个能推翻自身论点的对照组并给出结果，本身就是研究过程的一部分。

---

## §3 每个实验的详细设计

每个实验按统一模板展开：**假设 → 改了什么 → 为什么这样改 → 预期指标变化 → 风险与防御**。

### 3.1 Exp 0：`baseline`（未训练基座）

| 项 | 内容 |
|---|---|
| 假设 | 无（参照系） |
| 改了什么 | 什么都不改。`--adapter` 留空 |
| 为什么 | 全部效应量、相对提升都以它为分母。基线协议必须与后续实验完全一致，否则所有对比失效 |
| 预期指标 | 全部指标的参照值 |
| 数据来源 | `experiments/baseline_airline/eval_report.json`（**诊断报告 §2 与本表必须一致**） |
| 风险 | 只跑一次、或用了不同的 `--tools`/`--lite-system`。**基线一旦定了就不许改**；要改协议就换 label（见 §4.4） |

命令：

```bash
python scripts/eval/run_baseline.py --label baseline_airline
python scripts/eval/eval_recovery.py --label baseline_airline --repeat 4
```

### 3.2 Exp 1：`dpo_format`（只训格式约束对）

| 项 | 内容 |
|---|---|
| **假设** | 模型的 `FORMAT_ERROR` 主要来自「没把意图落成 `<tool_call>`」这一层；用「合法调用 ↔ 纯文本/坏 JSON」的对比对训练，能显著压低格式错误率 |
| **改了什么** | 只喂 `format_constraint` 一个类型的偏好对（`EXPERIMENT_PAIR_TYPES["dpo_format"]`） |
| **为什么这样改** | 单独一类才能把「格式约束的贡献」从其他三类里分离出来。混合训练无法回答「到底哪类数据起作用」 |
| **预期指标变化** | `format_accuracy` ↑；`FORMAT_ERROR` 占比 ↓（尤其 `promise_only` / `no_tool_call` 子类型）；accuracy 与 pass^4 同向上升；`HALLUCINATION` / `WRONG_TOOL` 占比**基本不动**（这不是它的目标） |
| **对照** | `baseline`（提升幅度）、`sft_only`（对比信号是否必要） |

**数据构造细节（代码事实，`PreferenceBuilder.build_format_pairs()`）**：`rejected` 有三种形态，按随机数分布：

| 形态 | 概率（代码常量） | `negative_kind` | 教学意图 |
|---|---|---|---|
| 简单负样本（纯文本承诺，`_SIMPLE_NEGATIVES` 池） | `mode < 0.45` | `simple` | 模型只是承诺要查，没有真的输出 tool_call |
| 坏 JSON（键缺引号） | `0.45 ≤ mode < 0.75` | `hard` | 有 `<tool_call>` 标签但 JSON 语法非法 |
| markdown 围栏（```` ```json ````）代替规定的标签 | `mode ≥ 0.75` | `hard` | 用 markdown 代码块代替规定的 `<tool_call>` 标签 |

> **这些概率是代码事实（可读源码复现），不是实验结果。** 引用时说「构造配方」而不是「效果」。

**风险与防御（R1：格式捷径 / reward hacking）**

| 项 | 内容 |
|---|---|
| 风险 | 格式约束对**天然最容易学**：chosen 与 rejected 的差异（有没有 `<tool_call>` 标签）非常显眼。模型可以在几百步内学会「加标签」，而**任务成功率不动**。它甚至可能学会「不管三七二十一先输出一个格式合法的调用」——包括在该澄清时也硬调 |
| 具体信号 | ① `format_accuracy` 上升但 `headline.accuracy` 不升；② `error_analysis.json` 的 `format_ok_value_wrong_rate` 上升；③ `WRONG_TOOL/should_clarify` 占比上升（该澄清却硬调） |
| **防御 1** | 报告里**并排**给出格式率与任务成功率（诊断报告 §5 的宽/窄两个口径）。任何只有格式率的结论都不允许写进报告 |
| **防御 2** | 监控 `param_accuracy` 字段级准确率（`ArgMatchResult.param_accuracy()`：命中字段数 / （比对字段数 + 缺失字段数））；它是一个比二元判定更细的量，能在成功率还没动的时候先暴露退化 |
| **防御 3** | 把 `dpo_joint_param` / `dpo_all` 作为「解毒版」：加入参数约束对直接对抗「结构完美但实体乱填」。若 `dpo_format` 出现 R1 而 `dpo_joint_param` 没有，这本身就是一个有价值的发现 |
| **防御 4（数据侧）** | 质量门禁已经拦掉一类污染：`PreferencePair.key` 把 `rejected` 纳入去重键（注释原文：只用 `prompt + chosen` 会「实测丢掉 97% 的候选」）；`_passes_quality_gate` 强制 chosen 与 rejected 语义不同 |

### 3.3 Exp 2：`dpo_recovery`（只训错误恢复对）

| 项 | 内容 |
|---|---|
| **假设** | 模型「被纠正后仍重复同一错误」的能力缺口，可以由「纠错后正确修正 ↔ 原样重复错误」的对比对直接补上。这是 SFT 数据无法提供的信号 |
| **改了什么** | 只喂 `error_recovery`（`EXPERIMENT_PAIR_TYPES["dpo_recovery"]`） |
| **为什么这样改** | 恢复能力是本项目的**原创点**。它必须在真实多轮评测（`eval_recovery.py`）上验证，而不是只看目标轮的通过率 |
| **预期指标变化** | `repeat_rate` ↓（**主指标**，跨方案可比）；`task_success_rate` ↑；**`first_try_rate` 不能下降**；目标轮评测里 `recovery` 类任务的 pass^4 ↑ |
| **数据来源（重要）** | 场景来自 `src/data/recovery_corpus.py`（12 模式 × 7 变体 = 84 场景），**不是**评测集的 12 个 recovery 任务。训练与评测因此完全分离 |
| **对照** | `baseline`（repeat_rate 的绝对差）、`sft_only`（R-C 的核心对比） |

**数据构造细节（代码事实，`build_recovery_pairs()`）**：`rejected` 四档：

| 形态 | 概率 | `negative_kind` | 教学意图 |
|---|---|---|---|
| 原样重复首轮错误 | `mode < 0.50` | `recovery` | 被指出错误后原样重复，没有听进用户的纠正（**真实观察到的最常见失败模式**，源码注释原文） |
| 退化成纯文本回复 | `0.50 ≤ mode < 0.75` | `simple` | 被指出错误后反而退化成纯文本，不再输出工具调用 |
| 换了工具但换错 | `0.75 ≤ mode < 0.90` | `hard` | 被指出错误后换了工具，但换到了另一个同样错误的工具 |
| 工具改对了但参数还错 | `mode ≥ 0.90` | `hard` | 被指出错误后工具改对了，但关键参数仍然是错的（最难的一档） |

**质量门禁（代码事实，这是本项目最独特的一道门）**：`_recovery_coherence()` 要求**纠错话术里点名的参数必须真的出现在首轮错误调用的参数里**。源码注释记录了这道门禁的由来：

> 「最初版本从整个纠错话术池里随机抽句子，结果生成了『航班搜索场景 + city 参数要填中文城市名』这样前后矛盾的训练数据。这种数据比没有数据更糟 —— 它教模型『用户的纠正可以忽略』。」

守护测试：`tests/test_methodology.py::test_recovery_pairs_have_coherent_corrections`。

**风险与防御（R2：「先错后改」被学成模式）**

| 项 | 内容 |
|---|---|
| 风险 | 恢复类 prompt 的形状是「assistant 输出了错误内容 → user 纠正 → assistant 改对」。模型可能学到的不是「识别纠正并改进」，而是「**在这种形状的上下文里，先输出一个错误再改**」。更坏的情况：模型把「被纠正」这个模式泛化到所有请求，导致**首轮就故意出错**（因为在训练数据里，首轮出错总是紧接着有纠正与正确答案） |
| 具体信号 | ① `first_try_rate` 显著下降（首轮就对的概率变低）；② `avg_corrections_used` 上升；③ 目标轮评测里 `initial` 类任务（88 个）的成功率下降，而 `correction` 类（12 个）上升；④ `stop_reasons` 里 `repeat_loop` 绝对数没降 |
| **防御 1** | **把 `first_try_rate` 提升为门禁指标**：任何 `dpo_recovery` / `dpo_joint` 的实验，若 `first_try_rate` 相对基线下降超过 §5.4 的判据阈值，**必须暂停并检查**（触发条件见 §9） |
| **防御 2** | 报告里把「目标轮评测」按 `target_turn_kind` 拆成 `initial`(88) 与 `correction`(12) 两组分别报。**只有 `correction` 组上升而 `initial` 组下降，就是 R2 的确诊证据** |
| **防御 3（数据侧）** | `recovery_corpus` 里的「正解是澄清」的场景（如 `_s_fake_reason`）的 `correct_reply` 是自然语言而非工具调用，`build_training_tasks()` **不把它们转成 Task**（注释：「因为它们的正解不是工具调用」）。这防止了「以工具调用为正解」的模式被过度泛化 |
| **防御 4** | 交叉验证：`dpo_format`（不含恢复数据）应当不出现 R2；如果两个实验都出现 R2，那问题在训练超参而非数据 |

**必须报告的两个数字（不可省）**：`total_errors` 与 `total_recoveries`。只看 `recovery_rate` 会误判（诊断报告 §6.2 有完整论证：一个方案犯错次数少，`recovery_rate` 会虚高）。

### 3.4 Exp 3：`dpo_joint`（格式 + 恢复，本项目主方案）

| 项 | 内容 |
|---|---|
| **假设** | 格式与恢复命中的是**两类不同的失败模式**（一个在「意图落地」，一个在「行为改变」），效应应当叠加（超加性） |
| **改了什么** | `format_constraint` + `error_recovery`（`EXPERIMENT_PAIR_TYPES["dpo_joint"]`） |
| **为什么这样改** | 这是「配置里声明的两个核心创新点的联合」。如果联合不成立，项目的主方案就必须被替换为单方案或 `dpo_all` |
| **预期指标变化** | 格式率与恢复率**同时**上升；总提升 > 两个单方案之和；`Δstability` 向 0 收敛 |
| **判据** | 见 §5.4：`dpo_joint` 的 pass^4 CI 下界 > 两个单方案 pass^4 的上界（超加性）；否则只能声明「与单方案无显著差异」 |

**`configs/train/dpo_joint.yaml` 里已经写好的可证伪声明（代码事实，配置原文）**：

```text
"_hypothesis": "格式与恢复命中的是两类不同的失败模式，效应应当叠加。"
"_expected":   "格式正确率与错误恢复率同时上升，整体准确率与 pass^k 的提升大于两个单方案之和（超加性）。"
"_falsifier":  "如果联合方案的提升只是「取两个单方案的最大值」，说明两类数据在争抢同一份模型容量，
                而不是解决正交问题。这是本实验最需要如实记录的负面结果。"
```

**风险与防御（R3：互相干扰，而非叠加）**

| 项 | 内容 |
|---|---|
| 风险 | ① 两类数据在**争抢同一份模型容量**：1.5B 的 LoRA（`lora_r = 16`、`lora_alpha = 32`、`target_modules = "all-linear"`）容量有限，两类梯度可能互相抵消；② 两类的**数据规模不同**会带来梯度占比偏斜（`CFG.preference.n_format = 1200`、`n_recovery = 1200`，按配置恰好 1:1）；③ **协议不一致**（格式类 prompt 是单轮，恢复类 prompt 是三段式）可能让模型在不同输入长度上表现分裂 |
| 具体信号 | ① `dpo_joint` 在格式上不如 `dpo_format`，在恢复上不如 `dpo_recovery`（**两头都退**，最坏情况）；② `dpo_joint` ≈ `max(单方案)`（**取最大值而非相加**，即配置里预言的负面结果）；③ 训练 loss 曲线出现「一类下降时另一类上升」的震荡 |
| **防御 1** | **预注册解读规则**（本节表格已给出），避免事后挑解释 |
| **防御 2** | 用 `data_health.json`（`dpo_train.py` 写盘）确认两类数据的实际条数与长度分布；若长度分布差异很大（恢复类 prompt 含三段式，明显更长），需检查截断率（`truncation.degenerate_after_truncation_rate`）——截断后 chosen 与 rejected 变得相同的样本对 DPO 是**纯噪声** |
| **防御 3** | 跑 `dpo_joint_param` 做**单变量增量**：如果 `dpo_joint` 与 `dpo_joint_param` 的差异巨大，说明「联合」这个词掩盖了「其实只有某一类在起作用」 |
| **防御 4** | 报告里同时给 `dpo_all`（四类全加）。若 `dpo_all` 明显差于 `dpo_joint`，是「加数据反而变差」的直接证据，属于高价值负面结果 |

### 3.5 扩展：`dpo_tool` / `dpo_tool_param` / `dpo_joint_param` / `dpo_all`

| 实验 | 假设 | 预期改善指标 | 特有风险与防御 |
|---|---|---|---|
| `dpo_tool` | 幻觉与选错工具是独立的失败维度，可以由「白名单工具 ↔ 貌似合理的假工具 / 相邻工具」单独修好 | `HALLUCINATION`（尤其 `plausible_fake_tool` 子类型）与 `WRONG_TOOL`（`adjacent_tool`）占比下降；格式与参数指标基本不动 | **风险**：`tool_selection` 的简单负样本用的是「正确工具命名的变体」（`FAKE_TOOL_VARIANTS`，见 `tool_catalog.py` 的注释：真正的区分度在于命名变体，而不是完全不相干的假工具）。如果模型只学会「不用完全不相干的名字」而仍然编造命名变体，幻觉率会降但降不到位。**防御**：子类型下钻必须分开报 `plausible_fake_tool` 与 `unknown_tool` |
| `dpo_tool_param` | 「工具名 + 参数」属于同一类能力（上下文实体引用），可以合并训练 | `HALLUCINATION` + `WRONG_TOOL` + `PARAM_ERROR` 三类占比同降 | **风险**：两类数据量不同（`n_tool_selection = 600` vs 参数类按 `n_format` 默认 1200）导致梯度偏斜。**防御**：用 `build_experiment_dataset(..., balance=True)` 做「各类型条数相等」的平衡版本，并作为一次额外的敏感性分析 |
| `dpo_joint_param` | 在 `dpo_joint` 上只加参数约束对，边际贡献可测 | `PARAM_ERROR` 占比↓、`format_ok_value_wrong_rate`↓，且不损害格式与恢复指标 | **风险**：与 `dpo_joint` 的差异可能落在噪声内（100 任务太小）。**防御**：用 §5.3 的 `required_sample_size()` 事先算「要检出这个差异需要多少任务」，若远超 100，则**事先声明此对比功效不足**，不当结论用 |
| `dpo_all` | 四类全加 = 上限 | 理论上最好 | **风险**：数据量最大（配置目标合计 1200+1200+600+1200 = 4200 条），`epochs=2` 下可能过拟合；`max_length=1536` 下长 prompt 截断率上升。**防御**：监控 `truncation_report()` 的 `degenerate_after_truncation_rate`，若 > 5% 必须调大 `max_length`（脚本自身会打印该警告） |

> **扩展切片的统一要求**：由于 mock 不支持它们（§1.3），**必须**真实 GPU 运行；每个都要有独立 label、独立 `experiments/<exp>/` 目录、独立 `recovery_metrics.json`。

### 3.6 训练超参：哪些必须固定，哪些允许扫描

代码事实：`CFG.train`（`src/config.py` `TrainCfg`）与 `configs/train/*.yaml`。

| 超参 | 默认值 | 消融实验中是否固定 | 说明 |
|---|---|---|---|
| `beta` | 0.1 | **固定** | 源码注释：「1.5B 小模型上 0.1 通常比 0.5 更有效」。β 扫描应作为**单独的敏感性分析**，不能混在核心消融里 |
| `loss_type` | `sigmoid` | **固定** | `dpo_loss.py` 支持 `sigmoid` / `ipo` / `label_smoothing`；`simpo` 会抛 `ValueError`（`tests/test_methodology.py` 有守护） |
| `lr` | 5e-5 | **固定** | 注释：DPO 论文用 1e-6 是全参微调，LoRA 下 5e-5 更常见 |
| `epochs` | 2 | **固定** | |
| `batch_size` × `grad_accum` | 2 × 8 = 16 有效批 | **固定** | |
| `lora_r` / `lora_alpha` / `lora_dropout` | 16 / 32 / 0.05 | **固定** | |
| `lora_target_modules` | `all-linear` | **固定** | |
| `max_length` / `max_prompt_length` | 1536 / 1024 | **固定，但可按体检报告上调** | 若 `data_health.json` 的 `recommended_max_length` 高于现值，**所有实验必须一起改**（否则引入协议差异，见 §4.4） |
| `seed` | 42 | **固定** | |
| `gradient_checkpointing` / `bf16` | True / True | **固定** | 显存约束下不可关 |

**唯一允许的例外**：如果某个实验触发了 §9 的触发条件（如 loss 停在 log 2、截断率过高），可以改超参重跑，但**必须**同时重跑基线以保持可比性，并在报告里标注「协议 v2」。

### 3.7 参数数量对训练过程的自检（每次训练前都会跑）

代码事实（`dpo_train.py` 的 `[1/3] DPO 损失实现自检` → `src/training/dpo_loss.py` `report_self_check()`）：

| 自检项 | 期望 | 意义 |
|---|---|---|
| 初始 loss 是否为 log 2 | 是 | `π_θ = π_ref` 时 DPO loss = log 2 = 0.6931，这是训练的起点 |
| loss 是否单调下降 | 是 | |
| loss 是否饱和到 0 | 是 | 正 margin 极大时 `-log σ(z) → 0` |
| `grad_chosen` 是否与解析梯度一致 / 为负 | 是 / 负 | chosen 的 logprob 上升 |
| `grad_rejected` 是否与解析梯度一致 / 为正 | 是 / 正 | rejected 的 logprob 下降 |
| `label_smoothing=0` 是否退化为标准 DPO | 是 | |
| IPO 是否在目标 margin 处取最小 | 是 | |

**这条自检是「loss 曲线能不能信」的前提。** 如果自检不通过，任何 loss 曲线都无意义 —— 但注意它**不能**验证训练数据或模型加载是否正确。

---

## §4 评测协议

### 4.1 统一协议表（所有实验必须逐项一致）

| 协议项 | 固定值 | 来源 | 为什么固定 |
|---|---|---|---|
| 任务集 | `build_tasks(seed=42)`，100 个任务 | `task_manifest.py`，`CFG.eval.num_tasks = 100` | 任务集变则 pass^k 绝对值不可比 |
| `BASE_DATE` | `2025-07-14`（周一） | `task_manifest.BASE_DATE` | 相对日期解释依赖它；用 `today()` 会导致跨天不可比 |
| 任务划分 | seen 67 / unseen 33，按类别轮转（`counter % 3 == 0`） | `_Builder.add()` | 保证 unseen 里每个类别都有分布 |
| 每任务采样数 n | 4 | `CFG.eval.num_samples_per_task` | pass^k 需 k ≤ n |
| 报告的 k | 1, 2, 4 | `CFG.eval.pass_k_values` | **注意**：`configs/eval/eval_baseline.yaml` 里的 `k_values` 不会被读取（诊断报告附录 A6） |
| 主指标 | pass^4 | `CFG.eval.primary_k = 4` | |
| 温度 T | 0.7 | `CFG.model.temperature`；`configs/eval/eval_baseline.yaml` | **必须 > 0**，否则同一任务多次采样结果相同，pass^k 恒等于 accuracy（`configs/default.yaml` 的 `_temperature_note`） |
| top_p | 0.9 | `CFG.model.top_p` | |
| max_new_tokens | 1536 | `CFG.model.max_new_tokens` | 太小会截断（`FORMAT_ERROR/truncated`），太大费时 |
| prompt | 完整版 `SYSTEM_PROMPT` + `render_tool_manual()` 全 6 工具 | `prompt_builder.py` | 训练与推理必须逐字一致 |
| observation 注入 | user 角色 `<observation>{json}</observation>` | `Task.to_messages()` / `build_messages()` | |
| 工具集合 | 全部 6 个 | `TOOL_REGISTRY` | |
| 评分口径 | 宽松为主指标，严格同时报告；`date_tolerance_days=0` | `grader.grade()` | |
| 每任务随机种子 | `CFG.seed * 1000 + sample_index` | `hf_policy.generate()` | 逐采样必须不同，否则稳定性维度被抹掉 |
| 恢复评测参数 | `--repeat 4 --max-turns 6 --max-corrections 2` | `eval_recovery.py` 默认值 | |
| 数据 `split` | `train`（`dev` 只用于早停，绝不用于报告） | `build_experiment_dataset(split="train")` | `dev` 是按 `task_id` 划分的 10%（`n_dev = max(1, int(len(task_ids) * 0.1))`），不是评测集 |

### 4.2 评测命令模板（逐实验复制）

```bash
EXP=dpo_joint        # 逐个替换：baseline / sft_only / dpo_format / dpo_recovery / dpo_joint / dpo_tool / ...

# 1) 目标轮主评测（100 任务 × 4 采样）
python scripts/eval/run_baseline.py \
    --adapter outputs/${EXP}/final \
    --label ${EXP}
# baseline 例外：不加 --adapter，label 用 baseline_airline

# 2) 真实多轮恢复评测（12 个 recovery 任务 × 4 次）
python scripts/eval/eval_recovery.py \
    --adapter outputs/${EXP}/final \
    --label ${EXP} --repeat 4 --max-turns 6 --max-corrections 2

# 3) 错误分析与出图
python scripts/eval/analyze_errors.py --label ${EXP}
```

### 4.3 为什么一致性是消融实验的生死线

消融实验的全部信息量来自**实验之间的差**。而这个差有两个来源：① 训练数据不同（这是我们想测的）；② 评测设置不同（这是我们想排除的）。

如果协议不一致：

- **差不可归因**：`dpo_joint` 比 `dpo_format` 高 5%，是因为数据更好，还是因为它这次跑的温度不同、任务数不同、工具集不同？无法回答。
- **置信区间不可比**：bootstrap CI 依赖样本量与重采样单位。任务数从 100 变成 30，CI 宽度会变成约 1.8 倍，两个宽度不同的 CI「是否重叠」这个判断本身就没有意义。
- **会出现「无中生有的提升」**：这是最危险的情形。例如：
  - 基线用 `--tools` 限定了 2 个工具（选择空间小、幻觉率天然低），实验用全 6 个工具 → 实验看起来更差，但这是设置差异；
  - 基线用 `--lite-system`（system prompt 短、格式要求弱），实验用完整 system prompt → 实验看起来更好，但提升来自提示词而不是 DPO。
- **相对提升的分母会变**：`collect_results.py` 用 `(p_hat - b) / b` 算相对提升。如果基线的 n 与实验的 n 不同，`p_hat` 的估计精度不同，相对提升的置信度不可评估。

**因此**：任何协议项的改变都必须**同时重跑全部实验**并新建 label 集（例如全部加 `_v2` 后缀），在报告里声明「v1 与 v2 不可比」。

### 4.4 协议变更的合规流程

| 步骤 | 动作 |
|---|---|
| 1 | 记录变更原因（例如「基线截断率 12%，必须提高 max_new_tokens」）与变更项 |
| 2 | 新建 label 集：`baseline_airline_v2` / `dpo_format_v2` / …（**禁止覆盖旧目录**） |
| 3 | 全部实验按新协议重跑（不允许只重跑一部分） |
| 4 | 报告里用一张小表列出 v1/v2 的协议差异，并声明两者不可比 |
| 5 | `collect_results.py --experiments ...v2` 只汇总同一版本的实验 |

> **常见违规**：为了省 GPU，只重跑基线，然后拿旧实验的数字与它对比。这会让整张消融表失去意义，且**很难在事后发现** —— 所以协议变更必须写在报告里的显眼位置。

---

## §5 主指标与判据

### 5.1 指标定义（逐字来自代码）

代码事实（`src/eval/metrics.py` 模块文档）：

- `accuracy`：单次采样正确率。
- `pass@k`：一个任务采 k 次，至少一次对的概率（无偏估计）。k=1 时等于 accuracy。
- `pass^k`：一个任务采 k 次，**k 次全对**的概率。τ-bench 的主指标。
- `pass^k_by_*`：按难度 / 类别 / seen-unseen 切分的 pass^k。
- `Δstability`：`pass^k − pass@k`。越接近 0 越稳定；负得越多说明「会做但飘」。

### 5.2 无偏估计公式

记 n = 每任务采样次数，c = 其中正确的次数，k ≤ n。

**pass@k（Chen et al. 2021, HumanEval）——「至少一次成功」**：

```
pass@k = 1 − C(n−c, k) / C(n, k)
```

**pass^k（τ-bench, Yao et al. 2024）——「k 次全部成功」**：

```
pass^k = C(c, k) / C(n, k)
```

代码实现用**对数域**计算以避免大数溢出（`log_comb()` 用 `math.lgamma`；`comb_ratio()` 在对数域相减后取指数，并把结果修约到 12 位以消除 ~1e-16 的浮点噪声）。

**为什么不用朴素的 `(c/n)^k`**：在 n 较小时（本项目 n=4）朴素估计是**有偏**的 —— 它把「从同一批样本里取 k 个」误当成了「独立重采 k 次」。代码事实：`n=4, c=2, k=2` 时无偏值是 `1/6 ≈ 0.1667`，朴素值是 `0.25`，相差约 8 个百分点（`tests/test_methodology.py::test_pass_hat_k_naive_differs_from_unbiased` 守护了这个差异）。

### 5.3 n == k 的退化与报告处理

**n = k = 4 时**：`C(n,k) = C(4,4) = 1`，两个估计都退化为「k 次采样的观测结果」，不再有估计方差：

- `pass^4 = C(c,4)/1`：`c = 4` 时为 1，否则为 0；
- `pass@4 = 1` 当且仅当 `c ≥ 1`，否则为 0。

这正是 τ-bench 报告 pass^4 时的行为（n 通常等于报告的 k）。

**报告处理方式（代码已实现）**：`EvalReport.pass_hat()` 在 `rs[0].n == k` 时，往 `MetricValue.note` 里追加一句：

```
k 次全对的无偏估计（n=4）；n==k 时退化为 0/1，参考值 (c/n)^k = <值>（有偏）
```

因此**报告里必须同时给出有偏的 `(c/n)^k` 作为参考值**，否则读者会以为 pass^4 恒为 0 是 bug。

`mean(per_task_naive)` 就是有偏参考值：它是逐个任务算 `(c_i/n)^k` 再取平均（代码事实：`per_task_naive = [pass_hat_k_naive(r.n, r.c, k) for r in rs]`）。

**报告模板**：

| 指标 | 无偏值（主） | 有偏参考 `(c/n)^4` | 说明 |
|---|---|---|---|
| pass^4 | `<待填>` | `<待填>` | 无偏值只有 0/1 两档；有偏值连续，用于观察趋势 |
| 示例行 | `—` | `—` | — |

<!-- TODO(真实数据): 从 experiments/<exp>/eval_report.json 的 metrics.pass^4.note 里摘录 (c/n)^k 参考值（每个实验各一条） -->

### 5.4 「什么算成功」的判据

#### 5.4.1 效果判据（三级，必须同时满足才算「有效提升」）

| 级别 | 判据 | 代码/方法 |
|---|---|---|
| **L1 点估计改善** | `pass^4` 与 `accuracy` 同时上升 | `compare_reports()` 的 `abs_delta > 0` |
| **L2 置信区间判据** | 实验的 pass^4 95% CI 与基线的 pass^4 95% CI **不重叠** | `collect_results.py::write_statistical_notes()` 的 `overlap = not (e_ci[1] < b_ci[0] or e_ci[0] > b_ci[1])`；脚本会直接给出「置信区间与基线重叠 → **不能断言有真实提升**，需增加任务数或采样次数」的判词 |
| **L3 效应量判据** | `|Cohen's h| ≥ 0.2`（小）才认为有实质意义；`< 0.2` 视为「可忽略」 | `metrics.cohens_h()`；`Comparison.effect_label` 的分档：`<0.2` 可忽略 / `<0.5` 小 / `<0.8` 中 / `≥0.8` 大 |

**CI 的构造方式（必须写进报告）**：**任务级 bootstrap**，1000 次重采样**任务**（不是采样样本），`alpha = 0.05`，固定 seed（`accuracy` 用 42，`pass@k` 用 `42+k`，`pass^k` 用 `142+k`，`Δstability` 用 7）。

为什么重采样任务是硬要求（代码注释原文）：「同一任务的多次采样高度相关，按样本重采样会严重低估置信区间宽度。重采样单位必须和分析单位一致，否则 CI 会被高估精度。」

**同一任务的 4 次采样不是独立观测** —— 这一点决定了：
- 不能用「400 个样本的二项分布」算 CI（会把 CI 缩窄约 √4 = 2 倍）；
- `per_task` 数组才是 bootstrap 的单位（`EvalReport.accuracy()` 用 `per_task = [r.accuracy for r in rs]`，每个任务一个值）。

#### 5.4.2 分组判据

分组表（`by_difficulty` / `by_category` / `by_visibility`）**只给 accuracy 的 CI**（代码事实：`_slice_summary()` 只对 accuracy 调 `_bootstrap_ci`）。所以：

- 分组里的 `pass@k` / `pass^k` 是**无 CI 的点值**，不能用来宣称「该组显著提升」；
- 若要分组显著性，必须**自行对子集重跑 bootstrap**（`EvalReport.accuracy(rs_subset)` 可以接受子集，`_bootstrap_ci` 也可直接调用）；
- 样本量小的组（`multi_tool` 9 个任务、`cancellation` 12 个、`recovery` 12 个）**默认不作为结论依据**。报告里可以列数字，但要写明「n 小、仅作观察」。

#### 5.4.3 样本量判据（**跑实验之前就要算**）

代码事实（`metrics.required_sample_size(effect, alpha=0.05, power=0.8)`）：

```
required_sample_size = ceil( 2 * (z_a + z_b)^2 * 0.25 / effect^2 )
其中 z_a = 1.959964（alpha=0.05 双侧）、z_b = 0.841621（power=0.8）、
方差取保守值 0.25（即 p=0.5 处的最大值）
```

模块文档说明了它的用途（原文）：「**在跑实验之前**判断 100 个任务够不够检测出目标提升。这个问题在开题阶段就应当回答，否则实验跑完才发现『提升 3% 但置信区间跨 0』会浪费一轮 GPU 预算。」

**必须填的两张表**：

**表 A：给定效应量所需的任务数**

| 目标效应（pass^4 绝对差） | `required_sample_size` | 现有 100 任务是否足够 | 备注 |
|---|---|---|---|
| 0.05（5 个百分点） | `<待填>` | `<是/否>` | 例：`required_sample_size(0.05)` 的结果请自行运行后填入 |
| 0.10 | `<待填>` | `<是/否>` | |
| 0.15 | `<待填>` | `<是/否>` | |
| 0.20 | `<待填>` | `<是/否>` | |
| 示例行 | `—` | `—` | — |

<!-- TODO(真实数据): 运行 python -c "from src.eval.metrics import required_sample_size as r; print([ (e, r(e)) for e in (0.05,0.10,0.15,0.20)])" 填入；这是纯计算，不依赖任何实验结果 -->

**表 B：以观测效应反推的建议任务数**（`collect_results.py` 会自动生成这段文字，抄进报告即可）

| 实验 | 观测 pass^4 差值（相对基线） | `required_sample_size` | 当前任务数 | `collect_results.py` 的判词 |
|---|---|---|---|---|
| `dpo_joint` | `<待填>` | `<待填>` | 100 | `<待填>`（脚本在 `need > n_tasks` 时会输出「当前任务数偏少，观测到的提升可能不稳定」） |
| 示例行 | `—` | `—` | 100 | — |

<!-- TODO(真实数据): 从 experiments/summary/ablation_table.md 的「统计说明」小节摘录「以观测到的 pass^4 差值 ... 计算，要在 80% 功效下检出该效应，每组约需 N 个任务（当前 100 个）」 -->

> **判据的正确用法**：`required_sample_size` 是两比例比较的**粗略**估计（源码注释原文），且用了保守方差 0.25。它的正确用法是**事前规划**（「本项目的 100 个任务能可靠检出多大的效应」），而不是**事后论证**（「我们的提升小是因为功效不足」）。事后论证只在「CI 与基线重叠」时作为可能原因之一提出。

#### 5.4.4 恢复能力的判据（同分母指标优先）

代码事实（`scripts/eval/eval_recovery.py` 模块文档，见诊断报告 §6.2 的完整论证）：

| 指标 | 分母 | 跨方案可比 | 判据用法 |
|---|---|---|---|
| `repeat_rate` | **样本数** | ✅ | **主判据**：相对基线显著下降（且 CI 不重叠）才算恢复能力提升 |
| `first_try_rate` | 样本数 | ✅ | **门禁判据**：不得显著下降（R2 风险的直接信号） |
| `task_success_rate` | 样本数 | ✅ | 辅助判据 |
| `recovery_rate` | **犯错次数** | ❌ | **只能同方案纵向比，不能跨方案横向比**；引用时必须附带 `total_errors` / `total_recoveries` |

#### 5.4.5 判据汇总表（每个实验都要填）

| 实验 | pass^4 点估计 | pass^4 95% CI | 与基线 CI 是否重叠 | Cohen's h | 效应量档位 | 判定 |
|---|---|---|---|---|---|---|
| `sft_only` | `<待填>` | `[<待填>, <待填>]` | `<是/否>` | `<待填>` | `<可忽略/小/中/大>` | `<有效/不可断言>` |
| `dpo_format` | `<待填>` | `[<待填>, <待填>]` | `<是/否>` | `<待填>` | `<...>` | `<...>` |
| `dpo_recovery` | `<待填>` | `[<待填>, <待填>]` | `<是/否>` | `<待填>` | `<...>` | `<...>` |
| `dpo_joint` | `<待填>` | `[<待填>, <待填>]` | `<是/否>` | `<待填>` | `<...>` | `<...>` |
| 示例行 | `—` | `—` | — | `—` | — | — |

<!-- TODO(真实数据): 从 experiments/summary/ablation_table.json 的 headlines 与 comparisons（含 cohens_h 与 effect_size）逐行填入 -->
<!-- TODO(真实数据): 从 experiments/summary/ablation_table.md 的「统计说明」抄录每个实验的 CI 重叠判词 -->

---

## §6 混淆因素排查

> **这一节的目的**：列出所有能产生「看起来像提升」的替代解释，并给出一条具体命令去排除它。
> 每条都要在报告里给出**结论**（排除 / 未排除 / 未做），不能只是列出来。

### 6.1 排查矩阵

| # | 替代解释 | 具体命令 | 观察什么 | 若成立则说明 |
|---|---|---|---|---|
| C1 | **提升只来自更啰嗦的 system prompt**（完整版有 6 工具手册 + 5 条规则 + 格式示例；精简版只有 2 行） | `python scripts/eval/run_baseline.py --adapter outputs/<exp>/final --lite-system --label <exp>_lite` | 与 `baseline_lite`（`--lite-system` 的基线）对比：如果 `<exp>_lite` 相对 `baseline_lite` **没有提升**，而完整 prompt 下有提升 → 提升依赖完整 prompt，**不是通用的对齐效果** | 说明 DPO 学到的能力与特定 prompt 强绑定（训练/推理 prompt 一致性太强，泛化性弱）。这是需要写清楚的局限 |
| C2 | **提升只来自「工具选择空间变小」**：只给 2-3 个工具时幻觉率天然更低 | `python scripts/eval/run_baseline.py --tools search_flights,get_weather --label <exp>_2tools`（以及对应的 `baseline_2tools`） | 与 `baseline_2tools` 对比。同时看 `HALLUCINATION` 占比：工具少了，可幻觉的名字池也小了 | 说明幻觉率下降是「选择空间」造成的，不是模型学会了遵守白名单 |
| C3 | **提升只来自「能看见工具返回结果」** | `python scripts/eval/run_baseline.py --no-observations --label <exp>_noobs` | ⚠️ **当前该开关不生效**（诊断报告附录 A9）：`include_observations` 只写进 metadata，`HFPolicy.predict()` 未传给 `build_messages()`。**必须先把代码改通**，否则这一条只能标「未做」 | 如果结论是「模型其实没在用 observation」，那么 `booking_flow` 的 `from_observation` 任务（10 个）的提升就无从解释，需要重新检查评分口径（例如 `flight_id` 是否被宽松口径放过了） |
| C4 | **提升只来自「更多监督数据量」而非对比信号** | 训练 `sft_only` 并评测（见 §2.4） | `sft_only` 与各 DPO 实验的 pass^4 CI 是否重叠、`repeat_rate` 是否差得开 | **这是最重要的一条**。若 `sft_only` ≈ `dpo_joint`，则本项目的「偏好对齐」卖点不成立（§2.5 的 R-B） |
| C5 | **提升来自评测设置差异**（温度、采样次数、k、任务数、随机种子） | 固定协议后**不做任何变体**；每次评测比对 `eval_report.json` 的 `metadata` 字段 | `metadata` 里的 `temperature` / `max_new_tokens` / `tool_names` / `system_prompt_lite` / `include_observations` / `num_samples_per_task` / `primary_k` / `task_limit` / `seed` 是否与基线逐项一致 | 任何一项不一致 → 该对比无效（§4.3） |
| C6 | **提升来自 checkpoint cherry-picking**（挑了最好的 checkpoint 报告） | 报告 `save_steps=50` 的**全部**中间 checkpoint 的评测结果，或明确声明「只报告 final」 | `outputs/<exp>/checkpoint-*/` 与 `final/` 的对比 | 若中间 checkpoint 明显更好而报告只用了 `final`，会被质疑挑选。**规范做法**：固定用 `final`（或固定用「dev loss 最低的那个」），并**在计划阶段就写死规则** |
| C7 | **提升来自评测集泄漏**（训练数据里混入了评测任务） | 检查 `data/preferences/<exp>.train.jsonl` 的 `task_id` 是否含 `task_0001`~`task_0100` | `src/data/recovery_corpus.py` 的 `build_training_tasks()` 把训练任务标为 `visibility="train_only"`、`task_id` 前缀 `trainsyn_`；`task_manifest` 的 100 个任务用 `task_XXXX` 格式 | 三类偏好对（format / tool_selection / param）的 prompt 池是 `评测任务 + trainsyn 任务`（`build_all()` 里 `pool = tasks + syn_tasks`），所以**训练数据里确实会出现评测任务的 prompt**。这是**有意的**（否则 prompt 复用次数太高、去重后数据量上不去，见 `build_training_tasks` 注释），但必须在报告里声明，并靠 `by_visibility` 的 **unseen** 结果作为泛化证据 |
| C8 | **评测集本身被改动** | `python -m src.data.task_manifest` + 比对 `experiments/tasks_airline_v1.json` | 任务集是否仍为 100 个、类别分布是否与 `docs/diagnosis/baseline_diagnosis.md` 一致 | 任务集在实验过程中被改过 → 所有历史结果作废 |

### 6.2 C1/C2 的正确执行方式（必须用同变体的基线做分母）

一个常见错误：拿「`--lite-system` 的实验」去比「完整 prompt 的基线」。这是**两个变量同时变**（prompt + 训练），得不出任何结论。

正确的对照关系：

| 对比 | 实验组 | 对照组 | 能回答的问题 |
|---|---|---|---|
| 主对比 | `<exp>`（完整 prompt） | `baseline_airline`（完整 prompt） | DPO 有没有用 |
| C1 | `<exp>_lite`（精简 prompt） | `baseline_lite`（精简 prompt） | 提升是否依赖完整 prompt |
| C2 | `<exp>_2tools`（2 工具） | `baseline_2tools`（2 工具） | 提升是否依赖工具选择空间 |
| C3 | `<exp>_noobs`（无 observation） | `baseline_noobs`（无 observation） | 提升是否依赖 observation |

**每一对都必须是一个完整的 2×1 对照**，两个臂同协议、同采样、同任务集。

### 6.3 C7 的严格处理：训练prompt复用评测任务是设计选择，不是缺陷，但必须声明

代码事实（`preference_builder.build_all()`）：

```python
pool = tasks + syn_tasks          # tasks = 100 个评测任务；syn_tasks = trainsyn_ 前缀的训练任务
pairs += self.build_format_pairs(pool, n_format)
pairs += self.build_recovery_pairs(pool, n_recovery, scenarios=scenarios)
pairs += self.build_tool_selection_pairs(pool, n_tool)
pairs += self.build_param_pairs(pool, n_param)
```

而**恢复类**偏好对**不**用评测任务：`build_recovery_pairs(..., scenarios=scenarios)` 显式使用 `recovery_corpus.build_scenarios()`（84 个训练场景），源码注释写明「不用评测集里的 12 个 recovery 任务（它们要保持固定、不被训练接触）」。

所以准确的表述是：

> 格式 / 工具选择 / 参数约束三类偏好对的 prompt **包含**评测任务的上下文（这是有意的，为的是 prompt 多样性；否则每个任务被复用约 12 次，去重后数据量上不去 —— `build_training_tasks` 的注释原文）。**恢复类偏好对完全不用评测任务。**
> 因此 **`recovery` 类的 pass^4 是最干净的泛化指标**；其余类别的提升必须靠 `by_visibility.unseen`（33 个任务，训练不接触）来支撑泛化性声明。

> **注意一个细节**：`visibility=unseen` 是**任务级**标记，而偏好数据是**prompt 级**复用这 100 个任务。所以严格来说，seen 任务的 prompt 被用作训练数据，unseen 任务的 prompt 没有被用作训练数据 —— 这正是 split 的意义。但 `task_manifest` 模块文档说 `seen` 是「5 个 epoch 训练时会用到」——**代码里并没有按 epoch 数划分的逻辑**，`seen/unseen` 只是标记，实际是否使用取决于 `preference_builder` 的 pool。这一点在报告里应表述为「`unseen` = 其 prompt 未进入任何偏好数据」，而不是「训练 5 个 epoch 时用到」。

---

## §7 产物与目录约定

### 7.1 一次实验的完整产物

```text
# —— 训练产物（outputs/）——
outputs/<exp>/
├── data_health.json           训练前数据体检（num_train / 分布 / token 长度 / 截断率 / prompt 一致性 / 审计）
├── train_config.json          超参、数据路径、is_mock=false、wall_time、final_metrics
├── compat_report.json         trl/peft/transformers 版本兼容性报告与被丢弃的字段
├── train_log.jsonl            逐 step 日志（真实训练由 Trainer 写）
├── training_curves.json       loss 与验证曲线（mock 训练由 mock_trainer 写）
├── checkpoint-<step>/         save_steps=50 的中间 checkpoint（save_total_limit=3）
└── final/                     **报告的固定评测对象**（LoRA adapter + tokenizer）

# —— 评测产物（experiments/）——
experiments/<exp>/
├── eval_report.json           目标轮主报告（headline / metrics / by_* / per_task）★报告只引用这个
├── predictions.jsonl          逐次采样原始输出（一切分析的唯一来源）
├── error_analysis.json        错误分布 + 子类型 + 交叉表 + 恢复分析 + remedies
├── failure_cases/             最多 40 个失败案例（case_NNN_task_XXXX_<错误类>.json）
├── recovery_metrics.json      **真实多轮**恢复指标（eval_recovery.py 产出）★§5.4.4 的判据来源
└── dialogues.jsonl            真实多轮对话全文（抽检用）

# —— 汇总产物 ——
experiments/summary/
├── ablation_table.md          可直接贴进 README 的 markdown 对比表（含统计说明）
├── ablation_table.json        机器可读（headlines / comparisons / cohens_h / effect_size）
├── ablation_summary.txt       控制台友好的纯文本表
└── （图表写到 docs/figures/）

# —— 图表 ——
docs/figures/                      ← 命名约定见 docs/figures/README.md
├── error_distribution_<label>.png        analyze_errors.py 始终生成这一份
├── error_distribution.png                仅当 label 以 "baseline" 开头且非 mock 时额外生成（README 引用这一份）
├── success_by_difficulty_<label>.png  +  success_by_difficulty.png（同上条件）
├── success_by_visibility_<label>.png  +  success_by_visibility.png（同上条件）
├── ablation_comparison.png               collect_results.py 产出
├── pass_k_curve.png                      collect_results.py 产出（两条线的垂直距离 = 稳定性缺口）
└── training_curves.png                   mock 训练 / 真实训练曲线
```

> **⚠️ 两处与图表相关的坑**
> 1. **带 label 与不带 label 是两份内容相同的 PNG**，且只有在 `label.startswith("baseline")` 且 `metadata.is_mock` 为假时才生成不带 label 的那份（代码事实：`analyze_errors.py::analyze_one()`）。因此 `docs/figures/README.md` 里列的文件名只对「label 以 baseline 开头」的基线实验成立；非基线实验的图永远带 `_<label>` 后缀。
> 2. `docs/figures/` 目录里现有的 `.svg` 与 `.mermaid.md` **不是** `src/utils/plots.py` 的产物（该模块只出 PNG）。引用前必须确认来源，非代码生成的图要在报告里标注「示意图，非真实数据生成」。

### 7.2 `eval_report.json` vs `eval_report.mock.json` 的严格区分

代码事实（`scripts/eval/run_baseline.py` 第 247-268 行 + `scripts/eval/collect_results.py` 的模块文档）：

| 后缀 | 来源 | 谁读它 |
|---|---|---|
| 无后缀（`eval_report.json`） | **真实 GPU 运行** | `collect_results.py` **默认只读它**；报告只引用它 |
| `.mock.json` | `--mock` 的 CPU 流水线 | 只有显式加 `--include-mock` 才会被纳入，且输出文件带 `.mock` 后缀、表格标题标注 **SIMULATED** |

**四道防线（代码已实现）**：

1. `run_baseline.py` 把 mock 结果写到 `eval_report.mock.json` / `predictions.mock.jsonl` / `error_analysis.mock.json`，**绝不覆盖**同名真实文件；若目录里已存在真实结果，会额外打印告警；
2. `collect_results.py` 默认只读 `eval_report.json`；
3. `collect_results.py` 检查 `metadata.is_mock`：全为 mock 时打印「⚠️ 全部是 MOCK 结果 —— 仅用于验证流水线，不能作为项目结论」；混合时打印「⚠️ 混有 MOCK 与真实结果 —— 对比表会失去意义，请检查数据来源」；
4. `analyze_errors.py` 出的图带 label，且只有「label 以 `baseline` 开头 + 非 mock」才写默认图名。

**为什么这么严**：`collect_results.py` 模块文档原文：「mock 的指标是我们**喂进去的假设**。如果它们混进正式报告，整个项目就变成了『自己写死结论再自己验证』。」

**⚠️ 已知的一个漏网情况（诊断报告附录 A15）**：`experiments/mock_baseline/eval_report.json` 的文件名**不带 `.mock`** 但内容是 mock（`is_mock: true`，且只有 30 个任务）。`collect_results.py` 的自动发现逻辑会把它当作 `baseline` 读进来。虽然第 3 道防线会打印「混有 MOCK 与真实结果」的告警，但**不会阻止它进入对比表**。执行消融前请先处理该文件（改名或移到 `experiments/mock/`），或**始终显式**用 `--experiments` 指定要汇总的实验名单。

### 7.3 每个实验必须产出的文件清单（自查表）

| 文件 | 是否必须 | 缺失时的后果 |
|---|---|---|
| `experiments/<exp>/eval_report.json` | ✅ 必须 | 无法进入消融表 |
| `experiments/<exp>/predictions.jsonl` | ✅ 必须 | 无法复核评分、无法重算宽口径指标 |
| `experiments/<exp>/error_analysis.json` | ✅ 必须 | 无法填错误结构相关列 |
| `experiments/<exp>/recovery_metrics.json` | ✅ 必须（`repeat_rate` 是本项目核心指标） | 恢复能力无法评估 |
| `experiments/<exp>/failure_cases/` | ⭕ 建议（至少 baseline 必须有） | 无法人工抽检 |
| `experiments/<exp>/dialogues.jsonl` | ⭕ 建议 | 无法抽检多轮过程 |
| `outputs/<exp>/data_health.json` | ✅ 必须 | 无法证明数据量与截断率，`sft_only` 对照的数据量对齐无法核实 |
| `outputs/<exp>/train_config.json` | ✅ 必须 | 无法核对超参一致性（§3.6） |
| `docs/figures/pass_k_curve.png` | ⭕ 建议 | 报告缺少最直观的稳定性图 |

---

## §8 预期结果汇总表

> **全部为占位。表格结构已完整，真实数据一到即可逐格填入。**
> 空单元格请填 `—`，不要用 0 或任何估计值占位。

### 8.1 主表（`headline` 四项 × 各实验）

| 实验 | 训练数据（`pair_type`） | 实际条数 | accuracy | pass@4 | pass^4 | Δstability | 相对基线（pass^4） | Cohen's h | 效应量档位 |
|---|---|---|---|---|---|---|---|---|---|
| `baseline` | 无 | — | `<待填>` | `<待填>` | `<待填>` | `<待填>` | — | — | — |
| `sft_only` | fmt+tool+param | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| `dpo_format` | fmt | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| `dpo_recovery` | rec | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| `dpo_joint` | fmt+rec | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| `dpo_tool` | tool | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| `dpo_tool_param` | tool+param | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| `dpo_joint_param` | fmt+rec+param | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| `dpo_all` | 四类 | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| 示例行（占位） | — | `—` | `—` | `—` | `—` | `—` | `—` | `—` | `—` |

<!-- TODO(真实数据): 从 experiments/summary/ablation_table.md 与 ablation_table.json 的 headlines / comparisons 逐行填入 -->
<!-- TODO(真实数据): 「实际条数」从 outputs/<exp>/data_health.json 的 num_train 填入（不要用配置里的目标条数） -->

> **脚本会自动生成这张表**：`python scripts/eval/collect_results.py --experiments baseline,sft_only,dpo_format,dpo_recovery,dpo_joint` 会写出 `experiments/summary/ablation_table.md`（含主表、分组明细、数据来源、统计说明）。
> **首选做法**：直接把这四节内容粘贴进本文档，而不是手工抄数字（减少转录错误），并保留「数据来源」小节里的文件路径。

### 8.2 分组明细（pass^4）

**按难度**（`collect_results.py` 的 `build_markdown_table()` 会自动生成）：

| 实验 | easy (43) | medium (32) | hard (25) |
|---|---|---|---|
| `baseline` | `<待填>` | `<待填>` | `<待填>` |
| `sft_only` | `<待填>` | `<待填>` | `<待填>` |
| `dpo_format` | `<待填>` | `<待填>` | `<待填>` |
| `dpo_recovery` | `<待填>` | `<待填>` | `<待填>` |
| `dpo_joint` | `<待填>` | `<待填>` | `<待填>` |
| 示例行 | `—` | `—` | `—` |

<!-- TODO(真实数据): 从 experiments/summary/ablation_table.md 的「分组明细（pass^4）→ 按难度」小节粘贴 -->

**按可见性（泛化的关键证据）**：

| 实验 | seen (67) | unseen (33) | unseen − seen |
|---|---|---|---|
| `baseline` | `<待填>` | `<待填>` | `<待填>` |
| `sft_only` | `<待填>` | `<待填>` | `<待填>` |
| `dpo_format` | `<待填>` | `<待填>` | `<待填>` |
| `dpo_recovery` | `<待填>` | `<待填>` | `<待填>` |
| `dpo_joint` | `<待填>` | `<待填>` | `<待填>` |
| 示例行 | `—` | `—` | `—` |

<!-- TODO(真实数据): 从 experiments/summary/ablation_table.md 的「分组明细（pass^4）→ 按可见性」小节粘贴 -->
<!-- 判据：若某实验的 seen 大涨而 unseen 基本不动，则该实验的「提升」高度可疑（疑似任务记忆） -->

**按任务类别**（7 类，`collect_results.py` 不生成这一节，需手工从各 `eval_report.json` 的 `by_category` 汇总）：

| 实验 | flight_search (22) | booking_flow (18) | baggage (15) | cancellation (12) | weather_advisory (12) | multi_tool (9) | recovery (12) |
|---|---|---|---|---|---|---|---|
| `baseline` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| `dpo_format` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| `dpo_recovery` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| `dpo_joint` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| 示例行 | `—` | `—` | `—` | `—` | `—` | `—` | `—` |

<!-- TODO(真实数据): 逐个实验从 experiments/<exp>/eval_report.json 的 by_category.<类别>.pass^4 填入 -->
<!-- 注意：括号里是任务数（代码事实）。multi_tool 只有 9 个、cancellation/recovery 各 12 个，默认不作为结论依据 -->

### 8.3 退化模式与恢复能力

**表：退化模式监控（所有实验必须都填，用于判断是否 reward hacking）**

| 实验 | 格式正确率 | 「格式对但值错」窄口径 | 「格式对但没做对」宽口径 | 严格 vs 宽松口径差 | 是否出现格式捷径 |
|---|---|---|---|---|---|
| `baseline` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<是/否>` |
| `dpo_format` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<是/否>` |
| `dpo_joint` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<是/否>` |
| 示例行 | `—` | `—` | `—` | `—` | — |

<!-- TODO(真实数据): 前两列从 experiments/<exp>/error_analysis.json 的 format_accuracy / format_ok_value_wrong_rate 填入 -->
<!-- TODO(真实数据): 宽口径用 docs/diagnosis/baseline_diagnosis.md §5.4 给出的只读脚本统计 per_task[*].predictions[*] 得到 -->

**表：恢复能力（真实多轮，`recovery_metrics.json`）**

| 实验 | task_success_rate | **repeat_rate（★主判据）** | first_try_rate | recovery_rate（附 total_errors / total_recoveries） | avg_turns | stop_reasons 里 repeat_loop 的次数 |
|---|---|---|---|---|---|---|
| `baseline` | `<待填>` | `<待填>` | `<待填>` | `<待填>` / `<待填>` / `<待填>` | `<待填>` | `<待填>` |
| `sft_only` | `<待填>` | `<待填>` | `<待填>` | `<待填>` / `<待填>` / `<待填>` | `<待填>` | `<待填>` |
| `dpo_recovery` | `<待填>` | `<待填>` | `<待填>` | `<待填>` / `<待填>` / `<待填>` | `<待填>` | `<待填>` |
| `dpo_joint` | `<待填>` | `<待填>` | `<待填>` | `<待填>` / `<待填>` / `<待填>` | `<待填>` | `<待填>` |
| 示例行 | `—` | `—` | `—` | `—` / `—` / `—` | `—` | `—` |

<!-- TODO(真实数据): 逐实验从 experiments/<exp>/recovery_metrics.json 填入 6 个字段；recovery_rate 三个占位分别是值 / total_errors / total_recoveries -->

**判读规则（预先写死，避免事后解释）**：

1. `repeat_rate` 必须**同时**低于基线且 CI 不重叠，才允许写「恢复能力提升」；
2. `first_try_rate` 若显著低于基线，**先按 §9 的 F2 触发检查**，在检查完成前不得宣称任何提升；
3. `recovery_rate` 上升但 `repeat_rate` 未变 → 只说明「改对的比例高了」，可能只是分母（犯错次数）变小，不构成恢复能力提升的证据。

### 8.4 假设验证总表（把 §1 的可证伪预测逐条判定）

| 编号 | 假设 | 判定所需数据 | 判定结果 | 若是负面结果，怎么写 |
|---|---|---|---|---|
| H-ExpC | 提升来自对比信号，而非同分布监督 | `sft_only` vs `dpo_joint` 的 pass^4 CI + `repeat_rate` | `<支持/推翻/不确定>` | 见 §2.5 的 R-B 文案 |
| H-Exp1 | 格式约束单独能降 `FORMAT_ERROR` | `dpo_format` 的 `FORMAT_ERROR` 占比与格式率 | `<...>` | 若格式率涨而成功率不动 → 报告「格式捷径」 |
| H-Exp2 | 恢复对能降 `repeat_rate` 且不伤 `first_try_rate` | `dpo_recovery` 的 `recovery_metrics.json` | `<...>` | 若 `first_try_rate` 降 → 报告 R2「先错后改」模式 |
| H-Exp3 | 格式 + 恢复**超加**（> 两个单方案之和） | 三个实验的 pass^4 与 CI | `<...>` | 若 ≈ `max(单方案)` → 报告「容量争抢」（配置里的 `_falsifier` 已预言） |
| H-ext | 参数约束对在 `dpo_joint` 上有正边际贡献 | `dpo_joint_param` − `dpo_joint` | `<...>` | 若差分落在噪声内 → 声明功效不足，不下结论 |
| 示例行 | — | — | `<...>` | — |

<!-- TODO(真实数据): 逐个实验的判定结果与依据填入 -->

---

## §9 风险与触发条件表

### 9.1 触发条件（出现即停下检查，不要继续跑）

| 编号 | 现象 | 可能的根因 | 立刻做什么 | 不要做什么 |
|---|---|---|---|---|
| **F1** | **训练 loss 停在 log 2（0.6931）附近不下降** | ① 偏好对的 chosen 与 rejected **实际等价**（DPO 会拟合噪声，梯度无信息）；② 加载的数据文件是 mock 的（`configs/train/*.yaml` 的 `train_file` 指向 `outputs/mock_pipeline/*.mock.jsonl`，见诊断报告附录 A16）；③ `label_smoothing` 或 `loss_type` 被误设；④ LoRA 没挂上（`target_modules` 不匹配）；⑤ `lr` 太小/训练步数不足 | ① 检查 `outputs/<exp>/data_health.json` 的 `audit` 与 `truncation.degenerate_after_truncation_rate`；② 确认 `train_file` 路径指向 `data/preferences/`；③ 跑 `report_self_check()` 确认损失实现自检通过；④ 打印几条偏好对的 `chosen` / `rejected` 肉眼确认不同 | 不要加大 lr 或加长训练——那只会让噪声拟合更充分 |
| **F2** | **pass^k 恒等于 accuracy**（`Δstability ≈ 0` 且两者数值相同） | ① 温度 = 0 或 `do_sample=False`（同一任务 4 次采样结果完全相同）；② 采样种子没有随 `sample_index` 变化；③ 模型退化成确定性输出（例如 LoRA 过大导致分布坍缩） | ① 核对 `configs/eval/*.yaml` 的 `generation.temperature` 与 `do_sample`；② 检查 `hf_policy.generate()` 的 `seed = CFG.seed * 1000 + sample_index` 是否生效；③ 从 `predictions.jsonl` 里抽一个任务，看 4 条 `text` 是否完全相同 | 不要把 `Δstability = 0` 解读成「模型非常稳定」——这是**度量失效**，不是好消息 |
| **F3** | **格式正确率上升，而任务成功率（accuracy / pass^4）不动** | reward hacking：模型学会了输出 `<tool_call>` 结构，但没学会填对实体（§3.2 的 R1） | ① 按诊断报告 §5.4 统计宽口径「格式对但没做对」；② 检查 `WRONG_TOOL/should_clarify` 是否上升；③ 加入 `param_constraint` 数据重训（`dpo_joint_param` / `dpo_all`） | 不要把格式率写进结论句——必须与成功率并排呈现 |
| **F4** | **`first_try_rate` 相对基线显著下降**（恢复类实验） | R2：「先错后改」被学成模式（§3.3） | ① 按 `target_turn_kind` 拆开目标轮评测（`initial` 88 vs `correction` 12），看是不是只有 correction 组上升；② 检查恢复类 prompt 的构造（是否所有样本都以「模型已犯错」开头）；③ 考虑下调恢复类数据的比例后重训 | 不要因为 `repeat_rate` 降了就宣布成功 |
| **F5** | **`unseen` 组 pass^4 显著低于 `seen` 组，且差值远大于基线的 seen−unseen 差** | 任务记忆：模型记住了训练 prompt 的表面模式 | ① 确认基线本身的 seen−unseen 差作为参照；② 检查训练数据的 prompt 是否过度重复（`build_all()` 的 pool 含 100 个评测任务，`PreferencePair.key` 去重后的复用次数）；③ 考虑只保留 `trainsyn_` 任务构造训练数据 | 不要把 seen 上的提升当作泛化证据 |
| **F6** | **`FORMAT_ERROR/truncated` 占比 > 某个阈值，或 `degenerate_after_truncation_rate > 5%`** | `max_length` / `max_new_tokens` 不足，偏好对被截断成「两侧相同」的噪声 | ① 跑 `dpo_train.py --dry-run` 读 `recommended_max_length`；② 按推荐值调整 `max_length` / `max_prompt_length`；③ **按 §4.4 的流程重跑全部实验并新建 label 集** | 不要只调实验、不重跑基线（会毁掉可比性） |
| **F7** | **两个实验的 `metadata` 有一项不一致** | 协议漂移（温度、工具集、prompt 版本、采样数） | 立刻停止汇总，按 §4.4 重跑 | 不要用「差不多」说服自己 |
| **F8** | **`collect_results.py` 打印「混有 MOCK 与真实结果」** | 目录里混入了 mock 产物（例如 `experiments/mock_baseline/eval_report.json`） | 定位并移走/改名 mock 文件；或始终用 `--experiments` 显式指定名单 | 不要忽视这条告警继续用生成的对比表 |
| **F9** | **`extension` 实验训练时抛 `ValueError: 未知实验 'dpo_tool'`** | `MockPolicy` 只支持 5 个实验；如果你在 mock 模式下跑扩展切片 | 扩展切片改用真实 GPU 路径（不带 `--mock`） | 不要往 `mitigation_factors` 里加臆造的系数来「让 mock 跑通」——那会把假设伪装成实验 |
| **F10** | **单次评测运行抛异常导致 `n` 小于 `--repeat`**（`eval_recovery.py` 会打印 `⚠️ <task_id> 第 N 次失败`） | 对话执行异常（例如生成返回空、JSON 结构异常） | 报告里必须注明缺失的样本；补齐后重跑该任务 | 不要静默丢掉异常样本——那会让不同实验之间的 `n` 不同，破坏跨方案可比性 |

### 9.2 触发后的处理流程（统一）

```text
1) 停下当前实验（不要继续消耗 GPU）
2) 定位根因：按上表的「立刻做什么」逐项排查，把结果写进 experiments/<exp>/notes.md
3) 判断是否需要改协议：
   需要 → 按 §4.4 新建 label 集，全部重跑
   不需要 → 只重跑当前实验，并说明为什么其他实验不需要重跑
4) 在报告里显式记录这次触发与处理（不是掩盖它）
```

> **第 4 条是过程记录的重点**：一次「发现 loss 不降 → 定位到数据里 chosen/rejected 语义等价率过高 → 修正质量门禁 → 重跑」的记录，比只给最终结果更能说明问题定位能力。`preference_builder` 的源码注释里已经保留了两次真实的失败记录（纠错话术不一致、去重键丢掉 97% 候选），报告里可以引用这两个例子说明本项目的自我纠错机制。

### 9.3 需要提前锁定的三个数值阈值（跑实验前填）

| 阈值 | 建议初值 | 依据 | 最终值 |
|---|---|---|---|
| `first_try_rate` 允许的最大下降（相对基线） | `<待填>`（建议先设 3 个百分点，或设为「CI 重叠即通过」） | 需要一个可操作的门禁线；建议用「CI 不重叠」作为唯一硬判据，避免引入任意阈值 | `<待填>` |
| `format_ok_value_wrong_rate` 的告警线 | `<待填>`（建议设为基线值 + 5 个百分点） | 一旦超过，说明格式捷径已经出现 | `<待填>` |
| `degenerate_after_truncation_rate` 的告警线 | **5%**（代码事实：`dpo_train.py` 在该值 > 5% 时打印「这些样本对 DPO 是噪声，建议调大 max_length」） | 代码里已经写死的告警线 | 5%（代码固定） |

<!-- TODO(真实数据): 用基线实测值回填前两行的「最终值」，并说明取该值的理由 -->

---

## §10 附录

### 10.1 与其他文档的分工

| 文档 | 定位 | 与本文档的关系 |
|---|---|---|
| `docs/diagnosis/baseline_diagnosis.md` | Week 1 基线诊断（填真实数据的地方） | 本文档 §1 的选题依据；§8 的映射表直接决定实验矩阵 |
| **`docs/ablation/ablation_plan.md`（本文档）** | Week 3 的**设计与计划**：实验矩阵、方法论论证、协议、判据、混淆因素、风险触发条件 | — |
| `docs/ablation/ablation_report.md` | Week 3/4 的**执行结果**：实际跑出来是什么、结论是什么 | 本文档的下游。**两者的实验矩阵与指标定义必须严格对应**；本文档只写计划与预期，结果一律写进那份 |
| `docs/method/preference_data_design.md` | 偏好数据构造的详细设计（四类 pair 的构造配方与质量门禁） | 本文档 §3 各实验的「数据构造细节」只给摘要，完整配方在那份 |
| `docs/ops/trl_version_notes.md` | TRL/PEFT/transformers 版本兼容性记录 | 本文档 §4 的协议表假定环境可用；具体版本约束见那份 |
| `docs/figures/README.md` | 图表的生成方式与命名约定 | 本文档 §7.1 的图表清单以它为准（但注意其「命名约定」未覆盖 `_<label>` 后缀，见诊断报告附录 A20） |

**引用纪律**：本文档**只写计划与预期**。任何真实数字一旦产生，写进 `ablation_report.md` 与 `baseline_diagnosis.md`，本文档只保留「预期」列并标注「见结果报告」。

### 10.2 与 `项目方案.md` 的实验编号对照

`项目方案.md` §四 Week 3 用的是 Exp 0-3，与本文档的编号**含义相同但名称不同**；它还漏了 `sft_only` 对照组与全部扩展切片。以代码与本文档为准。

| `项目方案.md` | 本文档 / 代码 | 差异 |
|---|---|---|
| Exp 0（无，基线） | `baseline` | 一致 |
| Exp 1（格式约束偏好对 500 条） | `dpo_format` | 条数不同（配置是 1200） |
| Exp 2（错误恢复偏好对 500 条） | `dpo_recovery` | 条数不同（配置是 1200） |
| Exp 3（格式 + 恢复 1000 条） | `dpo_joint` | 条数不同（配置是 2400） |
| —（未提） | **`sft_only`** | **方案文档完全没有对照组**，这是最大的缺口（见 §2） |
| —（未提） | `dpo_tool` / `dpo_tool_param` / `dpo_joint_param` / `dpo_all` | 方案文档只提了 3 类偏好对，代码实际有 4 类 |
| 指标写「格式正确率 / 任务完成率 / 错误恢复率 / 幻觉率」 | 本项目主指标是 **pass^k** 与 **Δstability** | 方案文档没有 pass^k，无法评估稳定性 |

其他方案文档与代码的不一致（任务数 30 vs 100、难度分布、工具可选参数、偏好对条数）已在 `docs/diagnosis/baseline_diagnosis.md` 附录 A 逐条列出。

### 10.3 与对标项目 `agentic-grpo-longhorizon` 的对照

| 维度 | 对标项目 | 本项目 |
|---|---|---|
| 训练方法 | GRPO（在线 RL） | DPO（离线偏好对齐） |
| 模型规模 | 7B 策略 + 72B 用户模拟器 | 1.5B 单模型 |
| 硬件 | 2×A800 | 单卡 24G（3090/4090） |
| 评测口径 | τ-bench airline（50 任务） | 自建 100 任务（7 类 / 3 档难度 / 67 seen + 33 unseen） |
| 核心方法 | PRM-Lite + LATA | 格式约束 + 错误恢复 + 工具选择 + 参数约束四类偏好对 |
| 稳定性指标 | pass^k | pass^k + **Δstability**（显式分离「会不会做」与「稳不稳」） |
| 研究闭环 | 问题诊断 → 方法设计 → 消融实验 → 结果分析 | **相同闭环**，并额外包含一个可能推翻自身论点的 `sft_only` 对照组 |

**为什么本项目没有 `pooled_stdev` / group advantage 那套分析**：`src/eval/metrics.py` 的 `pooled_stdev()` 保留了该函数，源码注释说明了它的用途是对照诊断「group saturation」（GRPO 的 advantage 在组内方差趋于 0 时消失）。本项目的 DPO **没有 group 结构**，因此不受此困扰 —— 这个对照本身可以写进报告（「为什么我们从 GRPO 换成 DPO」的一段论证）。

### 10.4 执行前检查清单（Week 3 开工前逐项打勾）

- [ ] `python -m src.data.task_manifest` 无断言失败，任务集是 100 个
- [ ] `python tests/run_tests.py` 全通过（尤其 `test_experiment_slices_are_disjoint_by_type` 与 `test_recovery_pairs_have_coherent_corrections`）
- [ ] `scripts/prepare/build_preferences.py` 已跑，`data/preferences/*.jsonl` 存在且**不是** `outputs/mock_pipeline/` 下的 mock 版本
- [ ] 各 `configs/train/*.yaml` 的 `pairs.train_file` 已确认指向真实数据（当前指向 mock 文件，见 A16）
- [ ] `experiments/mock_baseline/eval_report.json` 已处理（改名/移走），避免污染汇总
- [ ] 实现 `sft_only` 的训练路径（§2.4 的方式 A 或 B）
- [ ] 修复 `--no-observations` 的实际生效问题（A9），否则混淆因素 C3 只能标「未做」
- [ ] §5.4.3 的两张样本量表已填（事前算清楚 100 个任务能检出多大效应）
- [ ] §9.3 的两个阈值已锁定
- [ ] 基线的 `eval_report.json` + `recovery_metrics.json` 已产出，且已回填诊断报告
- [ ] 每个实验的 `data_health.json` 都保存了（用于证明数据量与截断率）
- [ ] 汇总命令固定使用显式实验名单：`--experiments baseline,sft_only,dpo_format,dpo_recovery,dpo_joint`
