# 基线诊断报告：Qwen2.5-1.5B 在航空客服工具调用任务上到底错在哪

> # ⚠️⚠️ 本报告的数字**已作废**（2026-10 复核）
>
> ## 为什么作废
>
> 本报告跑在**旧任务集（v1）**上，其缺陷见 `HANDOFF_next_steps.md` §1.2：
> 骨架/任务比 0.52、日期全在过去、`multi_tool` 名不符实、
> `seen`/`unseen` 标签与训练暴露无关。
>
> 另有一处**判据错误**：报告里用「CI 是否重叠」判断显著性 ——
> 配对数据必须用配对检验，CI 重叠**不构成**「无差异」的证据。
>
> ## 哪些还能用
>
> * **错误分类体系本身**（7 类互斥 + 12 子类型）—— 设计与实现，不受数字影响
> * **失败模式的定性描述**（模型会编造三字码、会只承诺不调用工具等）
>   —— 这些现象在新任务集上也复现了（见 `docs/ablation/ablation_report.md` §8）
> * 不能用：每一类的具体占比
>
> ## 下一步
>
> 需在 v3 任务集上重跑 baseline。**在那之前不要引用本报告的占比数字。**
>
> ---


> **文档定位**
>
> - **回答的问题**：未训练的 Qwen2.5-1.5B-Instruct 在本项目自建工具调用评测集上，失败模式是什么？每一类错误占多少、出现在什么难度/类别/可见性上、能不能被用户纠错挽回？
> - **读者**：本项目作者（填数据的人）、代码评审者、面试官（想看「会不会做诊断」）。
> - **在闭环中的位置**：Week 1 的产出，是 `docs/ablation/ablation_plan.md`（消融实验计划）的**输入**。本报告 §8 的「错误类别 → 偏好数据策略 → 预期改善指标」映射表，就是消融矩阵各实验的选题依据。
> - **配套文档**：`docs/ablation/ablation_plan.md`（Exp 设计）、`../../项目方案.md`（项目规划；注意它多处与代码不一致，见 §9.4）。
>
> **数据状态（已更新）**
>
> ⚠️ **README**：本文件是 Week 1 的**诊断框架**（指标定义、错误分类体系、
> 结论的推理结构），其中的数值位置**仍是占位符**。
> **真实实验数据请以以下两份为准，不要从本文件取数**：
>
> | 需要什么 | 看哪里 |
> |---|---|
> | baseline 真实错误分布 | `experiments/baseline_airline/eval_report.json` |
> | 全部实验的最终对照 | [`docs/ablation/ablation_report.md`](../ablation/ablation_report.md) |
> | agent 多轮指标 | [`docs/agent/agent_eval_report.md`](../agent/agent_eval_report.md) |
> | 面试用数字 | [`docs/interview/interview_prep.md`](../interview/interview_prep.md) |
>
> **baseline 真实结果（一句话）**：accuracy `50.25%`（CI [40.5%, 59.8%]）、
> pass^4 `45.0%`、格式正确率 `62.25%`；失败主因是 `wrong_value` 89 条、
> `action_hallucination` 41 条、`promise_only` 31 条。
>
> ⚠️ **另需注意**：仓库中 `experiments/sft_only/` 的 95.0% 结果
> **不可引用** —— 它的训练数据里混入了评测任务（prompt 逐字相同率 51.5%）。
> 正确的主结果是 `experiments/sft_v3/` 的 **83.00%**。
> 详见消融报告 §8。
>
> 本文件原有的说明保留如下（它是当时的真实状态）：
>
> 本仓库代码完整、测试通过，最初**只跑过 CPU 上的 mock 流水线**（合成策略，产物带 `is_mock: true`）。
> **本文件中所有指标值位置均为待填占位符，没有任何编造的数字。**
> 凡是引用具体数值的地方，都标注了它是「代码事实」「配置假设值」还是「实验结论」。
> 代码事实 = 从源码可直接读出、不依赖任何实验；配置假设值 = 我们喂给 mock 的假设，**不是观测**；实验结论 = 必须由真实 GPU 实验产出。
>
> **与其他文档的分工**：本文件是 Week 1 的**诊断报告模板**（结果填在这里）；`docs/ablation/ablation_plan.md` 是 Week 3 的**实验计划**（只写设计与预期，不含结果）；`docs/ablation/ablation_report.md` 是消融结果的**执行报告**（结果填在那里）。三份文档的实验矩阵与指标定义必须严格对应。

---

## 目录

- [§1 诊断目的与协议](#§1-诊断目的与协议)
- [§2 整体结果](#§2-整体结果)
- [§3 错误分布](#§3-错误分布)
- [§4 分组下钻](#§4-分组下钻)
- [§5 「格式对但值错」这一退化模式](#§5-格式对但值错这一退化模式)
- [§6 错误恢复能力](#§6-错误恢复能力)
- [§7 根因分析](#§7-根因分析)
- [§8 诊断结论 → 改进方向](#§8-诊断结论--改进方向)
- [§9 失败案例库](#§9-失败案例库)

---

## §1 诊断目的与协议

### 1.1 这一节要回答什么

「基线准确率 38%」这句话对本项目**没有任何指导意义**。它不告诉你：

- 38% 里有多少是 JSON 语法错误（→ 教格式就能修）
- 有多少是工具名编造（→ 教工具选择才能修）
- 有多少是工具选对了但参数填错（→ 教参数约束才能修）
- 有多少是「明明会做，但 4 次里只对 2 次」（→ 这是稳定性的问题，和会不会做是两件事）

所以诊断目的有三个，缺一不可：

1. **定位**：把失败拆成互斥的几类，给出各类占比。
2. **归因**：把「哪类错误集中在哪类任务/难度/可见性上」交叉出来，判断问题出在能力还是稳定性。
3. **选题**：让每一类高发错误都对应到一种可执行的偏好数据构造策略（见 §8）。

### 1.2 评了什么

| 项 | 取值 | 来源与性质 |
|---|---|---|
| 模型 | `Qwen/Qwen2.5-1.5B-Instruct` | 配置默认值（`src/config.py` `ModelCfg.name`）；代码事实 |
| 训练状态 | 无 adapter（原始基座） | `configs/eval/eval_baseline.yaml` 的 `model.adapter_path: null` |
| 任务集规模 | 100 | 代码事实：`src/data/task_manifest.py` `build_tasks()` 断言 `len == 100` |
| 类别数 | 7 | 代码事实：`baggage / booking_flow / cancellation / flight_search / multi_tool / recovery / weather_advisory` |
| 难度分布 | easy 43 / medium 32 / hard 25 | 代码事实（由生成配方决定，可由 `python -c "from src.data.task_manifest import summary; print(summary())"` 复现） |
| 可见性分布 | seen 67 / unseen 33 | 代码事实：`_Builder.add` 用 `counter % 3 == 0` 轮转划分，保证每个类别在 unseen 里都有分布 |
| 目标轮性质 | `initial` 88 / `correction` 12 | 代码事实：`target_turn_kind` 字段 |
| 每任务采样次数 n | 4 | 配置值（`CFG.eval.num_samples_per_task`，`configs/default.yaml`） |
| 报告的 k | 1, 2, 4 | 配置值（`CFG.eval.pass_k_values`） |
| 主指标 | pass^4 | 配置值（`CFG.eval.primary_k = 4`） |
| 采样温度 | 0.7 | 配置值（`CFG.model.temperature`；`configs/eval/eval_baseline.yaml` 同值） |
| top_p | 0.9 | 配置值（`CFG.model.top_p`） |
| max_new_tokens | 1536 | 配置值（`CFG.model.max_new_tokens`） |
| 工具集合 | 全部 6 个 | 代码事实：`src/env/tool_catalog.py` `TOOL_REGISTRY` |
| system prompt | 完整版 `SYSTEM_PROMPT` | 代码事实：`src/models/prompt_builder.py` |

**为什么要报告「任务集是 100 个」而不只是「一批任务」**：pass^k 的绝对值依赖任务集本身，任务集变了，所有历史数字都不可比。任务集是**代码确定性生成**的（固定 seed=42、固定基准日期 `BASE_DATE = 2025-07-14`），因此它可以被任何人在任何机器上逐字复现 —— 这是本项目敢把 pass^k 当作跨实验主指标的前提。

> **⚠️ 与规划文档的冲突（必须知道）**
> `项目方案.md` 多处写「30 个任务」「评测场景 τ-bench airline（50任务）」「3 类难度 easy 10 / medium 8 / hard 12」。
> **实际代码是 100 个任务、7 个类别、三档难度 43/32/25。**
> 以代码为准。规划文档是早期草稿，本报告不做任何调整，但引用时请勿再传播「30 任务」这个数字。

**目标轮评测口径（这是本项目评测设计的核心取舍）**

代码事实（`src/data/task_manifest.py` 模块文档 + `Task.target_prefix`）：

- 每次采样只对 `target_turn` 这一轮的 assistant 输出打分；
- `target_turn` 之前的所有轮次作为**固定前缀**喂给模型；
- 工具轮（`Turn.role == "tool"`）在 `to_messages()` 里被渲染成 **user 角色的 `<observation>{...}</observation>` 文本**，不是单独的 `tool` role。

为什么这么做：真正的多轮全自动评测需要用户模拟器。对标项目 `agentic-grpo-longhorizon` 用 72B 模型当用户模拟器 + 7B 策略，本项目只有 1.5B 单模型和单卡 24G，做不出可信的模拟器。把目标轮显式标注出来后，评测变成「给定确定的多轮前缀，评这一轮」——ground truth 确定，pass^k 才有意义。

代价（需在报告中说明）：**测不到「模型自己把多轮串起来」的能力**。多轮串联能力只能通过 `scripts/eval/eval_recovery.py` 的**真实多轮**路径部分覆盖（见 §6）。

### 1.3 怎么算「对」：两套口径 + 语义等价判定

严格字符串匹配会把「答对了但写法不同」判成错。代码事实（`src/eval/equivalence.py` 模块文档的对照表）：

| ground truth | 模型输出 | 精确匹配 | 语义等价（本项目用） |
|---|---|---|---|
| `2025-07-15` | `2025/07/15` | ✗ | ✓ |
| `PEK` | `pek` | ✗ | ✓ |
| `30` | `"30"` | ✗ | ✓ |
| `北京` | `北京市` | ✗ | ✓ |
| `张伟` | ` 张伟 ` | ✗ | ✓ |

**两套口径同时报告**（代码事实，`src/eval/grader.py` 模块文档）：

| 口径 | 定义 | 用途 | 代码位置 |
|---|---|---|---|
| **宽松口径**（lenient） | 只比对 `ExpectedCall.graded_args`（= `args_subset or args`）列出的关键字段 | **主指标**。对「模型没义务猜对」的字段（如需从 observation 取的 `flight_id`）不做要求 | `grade()` 里 `arg_match.lenient_matched` |
| **严格口径**（strict） | `expected.args` 里全部字段都要命中且值等价 | 「能不能一次做全」的天花板 | `grade()` 里 `arg_match.strict_matched` |

代码事实：只有 9 个任务写了 `args_subset`（8 个带原因的取消任务 + 1 个 overbook 恢复任务），因此**绝大多数任务上两套口径等价**。两套口径的差值本身是有信息的：如果宽松涨了而严格没涨，说明模型学会了「关键字段对、细节不管」的捷径。

**语义等价判定的规则**（代码事实，`src/eval/equivalence.py`）：

1. `normalize_value(value, param_name)` **按参数名推断语义**，而不是只看值：
   - 参数名是 `date` / `*_date` → 解析成日期后比对 ISO 字符串（支持 `%Y-%m-%d`、`%Y/%m/%d`、`%Y.%m.%d`、`%Y年%m月%d日`、`%Y%m%d` 等）；
   - 参数名是 `origin` / `destination` / `city` / `from_city` / `to_city` 等 → `normalize_city()`，中文城市名与三字码**互相映射**（`PEK` ≡ `北京`），并剥离「市/自治区/地区」后缀、展开别名（`帝都` → `北京`）；
   - 参数名以 `_kg` / `_price` / `_cny` / `_count` / `_min` / `_num` 结尾或等于 `new_baggage_kg` / `max_price` / `amount` → 数值化（`"30公斤"` ≡ `30`）；
   - 其余 → 全角转半角、压缩空白、小写。
2. 日期容差默认为 **0 天**（`grade()` 调 `match_arguments(..., date_tolerance_days=0)`）。即日期必须同一天。
3. 数值用相对容差 `1e-6`。
4. 文本相似度兜底阈值默认 **关闭**（`text_similarity_threshold=0.0`）。

> **⚠️ 注意一个反直觉的口径**：`origin` 参数填 `北京` 会被判**对**（因为 `normalize_city` 做双向映射），而 `get_weather` 的 `city` 填 `PEK` 也会被判**对**。
> 也就是说：**基于等价判定的评分不会惩罚「参数语义用反」**（城市名 ↔ 三字码互换）。
> 这是有意的设计（判分口径要宽容，否则 DPO 会去拟合噪声），但它意味着：
> **§5 的「格式对但值错」和 §8 的参数约束偏好对，必须用另一套更严格的检查来证明有效**——
> 也就是 `src/env/tool_catalog.py` 的 `validate_arguments()`（它会用 `origin` 的 `pattern: ^[A-Z]{3}$` 把 `北京` 判成 `pattern` 违规）。
> 两者是**不同用途**：`equivalent` 用于**评分**（要不要给分），`validate_arguments` 用于**训练数据门禁**（这条 chosen 合不合法）与 `agent_loop.evaluate_step` 的**过程判错**（要不要注入纠错）。
> 报告里引用「参数语义错误率」时必须说清用的是哪一套。

### 1.4 为什么主指标用 pass^k 而不是 accuracy

代码事实（`src/eval/metrics.py` 模块文档）：

| 指标 | 定义 | 为什么报告它 |
|---|---|---|
| `accuracy` | 单次采样平均正确率 | 最直观，但不反映稳定性 |
| `pass@k` | 一个任务采 k 次，**至少一次**对的概率 | k=1 时等于 accuracy，与 accuracy 不独立 |
| `pass^k` | 一个任务采 k 次，**k 次全对**的概率 | τ-bench 的主指标；同时惩罚「不会做」和「做得不稳」 |
| `Δstability` | `pass^k − pass@k` | 越接近 0 越稳定；负得越多说明「会做但飘」 |

**「能做出某题」和「每次都能做对」是两件不同的事。** 工具调用是生产环境里要反复执行的动作：一个 40% 单次成功率、但每次都靠采样运气才对的 agent，和一个 40% 单次成功率、但做对的任务永远做对的 agent，工程价值完全不同。accuracy 把这两者压成同一个数字，`Δstability` 把它们的差别显式暴露出来。

这也是 DPO 的**正确优化目标**：不是把 `pass@k` 拉高（那只需要「偶尔能蒙对」），而是**在保持 pass@k 的同时把 pass^k 拉上来**，即把 Δstability 推向 0。

### 1.5 协议冻结声明

> **本节是硬约束。评测协议一旦确定，不得修改。**

冻结项（改动任何一项都会让历史结果失去可比性）：

1. 任务集定义（`build_tasks()` 的配方、`BASE_DATE`、seed）；
2. 每任务采样次数 n=4、报告 k ∈ {1,2,4}、主指标 pass^4；
3. 温度 0.7 / top_p 0.9 / max_new_tokens 1536 / 无 repetition penalty 之外的采样参数改动；
4. system prompt 全文（完整版）+ 工具集合（全部 6 个）+ observation 注入方式（user 角色 `<observation>`）；
5. 评分口径（宽松为主指标，严格同时报告；`date_tolerance_days=0`）；
6. 错误分类体系与优先级（§3.1）。

例外通道：任何协议变更必须**新建一个 label**（例如 `baseline_airline_v2`），在报告里显式声明「与 v1 不可比」，**不得覆盖旧目录**。理由：`experiments/<label>/` 的名字就是「这份数字是在什么协议下得到的」的声明。

**同时冻结配置读取行为**（这是代码事实，不是我们选的）：

`configs/eval/eval_baseline.yaml` 里的 `eval.k_values` 与 `variants.*`、`output.dir` **目前不会被 `run_baseline.py` 读取**（脚本只读 `eval.num_samples_per_task`、`eval.primary_k`、`generation.temperature`、`generation.max_new_tokens`、`model.adapter_path`）。k 值实际来自 `CFG.eval.pass_k_values = (1,2,4)`。

因此：

- 报告里的 k 值**必须**从产出的 `eval_report.json` 的 `metrics` 键名（`pass@1` / `pass@2` / `pass@4` / `pass^1` / `pass^2` / `pass^4`）反查，**不要**引用 yaml 里写的值；
- `--lite-system` / `--tools` 必须**显式写在命令行上**（yaml 的 `variants` 段是摆设），见 `docs/ablation/ablation_plan.md` §6。

关于 `--no-observations` 的一个已知问题：`run_baseline.py` 把 `include_observations` 只写进了 `metadata`（第 155 行），而 `HFPolicy.predict()` → `build_messages()` **没有**把该参数传下去（`src/models/hf_policy.py` 第 242-246 行只传了 `tool_names` 和 `lite_system`）。也就是说该开关当前**对模型输入没有任何影响**（只在报告 metadata 里留一个可能失真的标记）。详见 §9.4 与消融计划 §6。

### 1.6 语义等价判定的自检

评分器的判定口径直接决定 DPO 的训练目标是否合理：如果把「同义但写法不同」判成错，偏好数据里就会出现「chosen 与 rejected 其实等价」的垃圾对，DPO 会去拟合噪声。

因此基线评测**必须**附带这两项前置检查（否则后面的错误分布不可信）：

| 检查项 | 命令 | 通过标准 | 记录位置 |
|---|---|---|---|
| 任务集自检 | `python -m src.data.task_manifest` | 无 `AssertionError`（唯一 ID、目标轮合法、期望工具在白名单、ground truth 无未换算相对日期、recovery 任务有明确纠错话术） | 控制台 |
| 评分器/解析器单测 | `python tests/run_tests.py` | 全通过 | 控制台 |
| mock 注入一致性（仅证明「解析器 + 分类器」链路自洽，**不是**真实结果） | `python scripts/eval/run_baseline.py --mock --label sanity_mock` | `pipeline` 一致性 ≥ 0.85（`MockPolicyCfg` 有 5 类错误，但一致性自检把纠错轮排除） | `experiments/sanity_mock/error_analysis.mock.json` |

> **一致性数字不是结论，只是链路自检。** 详见 `src/mock/mock_policy.py` `verify_injection()` 的说明。

### 1.7 诊断产物清单（真实运行后应存在）

| 文件 | 内容 | 谁来读 |
|---|---|---|
| `experiments/baseline_airline/eval_report.json` | 主报告：`headline` / `metrics`（含 95% CI）/ `by_difficulty` / `by_category` / `by_visibility` / `per_task` / `metadata` | 本报告 §2 §4；消融对比 |
| `experiments/baseline_airline/predictions.jsonl` | 逐次采样原始输出（`TaskPrediction`） | 一切分析的唯一来源 |
| `experiments/baseline_airline/error_analysis.json` | 错误分布、子类型、难度×类别交叉、恢复分析、remedies | 本报告 §3 §4 §6 §8 |
| `experiments/baseline_airline/failure_cases/` | 按类别抽出的失败案例（最多 40 个） | 本报告 §9 |
| `experiments/baseline_airline/recovery_metrics.json` | **真实多轮**恢复评测指标 | 本报告 §6 |
| `experiments/baseline_airline/dialogues.jsonl` | 真实多轮对话全文 | 本报告 §6 抽检 |
| `docs/figures/error_distribution.png` | 错误分布饼图 + 子类型 Top8 | README / 本报告 §3 |

`analysis.json` / `recovery_metrics.json` / `dialogues.jsonl` 的**无 `.mock` 后缀**版本才是真实结果；带 `.mock` 的是 CPU 流水线自检产物，**绝不可引用为结论**（`scripts/eval/run_baseline.py` 第 247-268 行有明确的隔离逻辑与告警）。

**关于 `docs/figures/` 目录的现状**（执行前请先确认）：`docs/figures/README.md` 定义了图表的**预期**文件名（`error_distribution.png` 等）。但该目录当前存在若干 `.svg` 与 `.mermaid.md` 文件，**它们不是 `src/utils/plots.py` 的产物**（该模块只出 PNG，且 matplotlib 缺失时静默返回 `None`）。引用任何图之前请先确认它是由哪条命令生成的；源码注释里没有出处、`plots.py` 也不会产出 SVG 的图，应在报告里标注为「示意图，非真实数据生成」。

### 1.8 待填数据清单（本报告的全部 TODO）

真实运行后需要回填的位置，集中列在这里，便于一次性核对：

| # | 位置 | 字段 | 来源 |
|---|---|---|---|
| 1 | §2 表 1 | accuracy / pass@4 / pass^4 / Δstability + 各自 CI | `experiments/baseline_airline/eval_report.json` → `headline.*`、`metrics.accuracy.ci95`、`metrics.pass@4.ci95`、`metrics.pass^4.ci95`、`metrics.delta_stability.ci95` |
| 2 | §2 表 2 | pass@1 / pass@2 / pass^1 / pass^2 | 同上 `metrics.pass@{1,2}` / `metrics.pass^{1,2}` |
| 3 | §3 表 1 | 各类错误 count / 占总样本 / 占失败 | `error_analysis.json` → `distribution.counts` / `rates` / `failure_share` |
| 4 | §3 表 2 | 子类型 count | `error_analysis.json` → `distribution.failure_subtype_counts` |
| 5 | §4 表 1 | 按难度 accuracy / pass^4 | `eval_report.json` → `by_difficulty.{easy,medium,hard}` |
| 6 | §4 表 2 | 按类别 accuracy / pass^4 | `eval_report.json` → `by_category.*` |
| 7 | §4 表 3 | 按可见性 accuracy / pass^4 | `eval_report.json` → `by_visibility.{seen,unseen}` |
| 8 | §5 表 | 格式正确率 / 格式对但值错条数与比率 | `error_analysis.json` → `format_accuracy`、`format_ok_value_wrong`、`format_ok_value_wrong_rate` |
| 9 | §5 表 | 严格 vs 宽松口径差值 | `eval_report.json` → `per_task[].predictions[].strict_ok` / `lenient_ok` 统计 |
| 10 | §6 表 1 | 5 个多轮恢复指标 | `experiments/baseline_airline/recovery_metrics.json` |
| 11 | §6 表 2 | 逐 mistake_key 的 n / 成功率 / 恢复率 / 重复率 | `recovery_metrics.json` → `by_mistake_key.*` |
| 12 | §6 表 3 | 停止原因分布 | `recovery_metrics.json` → `stop_reasons` |
| 13 | §3 / §4 图 | 3 张 PNG 的实际路径 | `docs/figures/` |

---

## §2 整体结果

### 2.1 headline 表

> 全部数值待真实运行后填入。示例行的 `<待填>` 是占位符，**不得**替换成任何估计值。

**表 1：主指标（主指标 pass^4，n=4）**

| 指标 | 值 | 95% CI | CI 单位数 | 读法 |
|---|---|---|---|---|
| accuracy（单次采样正确率） | `<待填>` | `[<待填>, <待填>]` | 100 任务 | 每次采样平均对多少 |
| pass@4（至少一次对） | `<待填>` | `[<待填>, <待填>]` | 100 任务 | 「会做这道题」的比例 |
| pass^4（4 次全对） | `<待填>` | `[<待填>, <待填>]` | 100 任务 | 「每次都做对」的比例 |
| **Δstability = pass^4 − pass@4** | `<待填>` | `[<待填>, <待填>]` | 100 任务 | 越接近 0 越稳 |
| 示例行（占位） | `—` | `[—, —]` | 100 | — |

<!-- TODO(真实数据): 从 experiments/baseline_airline/eval_report.json 的 headline.accuracy 填入 -->
<!-- TODO(真实数据): 从 experiments/baseline_airline/eval_report.json 的 headline.pass_at_k 填入 pass@4 -->
<!-- TODO(真实数据): 从 experiments/baseline_airline/eval_report.json 的 headline.pass_hat_k 填入 pass^4 -->
<!-- TODO(真实数据): 从 experiments/baseline_airline/eval_report.json 的 headline.delta_stability 填入 Δstability -->
<!-- TODO(真实数据): 从 experiments/baseline_airline/eval_report.json 的 metrics.accuracy.ci95 / metrics.pass@4.ci95 / metrics.pass^4.ci95 / metrics.delta_stability.ci95 填入 CI -->

**表 2：各 k 的 pass@k 与 pass^k（用于画 `docs/figures/pass_k_curve_baseline.png`）**

| k | pass@k | pass^k | 备注 |
|---|---|---|---|
| 1 | `<待填>` | `<待填>` | k=1 时两者都等于 accuracy（代码事实：`pass_at_k_unbiased(n,c,1)=c/n`，`pass_hat_k_unbiased(n,c,1)=c/n`） |
| 2 | `<待填>` | `<待填>` | n=4 > k=2，两者都是无偏估计，不是 0/1 |
| 4 | `<待填>` | `<待填>` | **n == k 退化情形**：`pass^4 = C(c,4)/C(4,4) ∈ {0,1}`，报告里同时给出有偏参考值 `(c/n)^4` |
| 示例行 | `—` | `—` | — |

<!-- TODO(真实数据): 从 experiments/baseline_airline/eval_report.json 的 metrics.pass@1 / pass@2 / pass@4 填入 -->
<!-- TODO(真实数据): 从 experiments/baseline_airline/eval_report.json 的 metrics.pass^1 / pass^2 / pass^4 填入 -->
<!-- TODO(真实数据): pass^4 的 note 字段里含 (c/n)^k 的有偏参考值，一并摘录 -->

**关于 n == k 的退化（必须写进报告，否则 0.00 会被误读）**

代码事实（`src/eval/metrics.py`）：

- `pass_hat_k_unbiased(4, c, 4) = C(c,4)/C(4,4)`，只有 `c=4` 时为 1，否则为 0；
- 因此当 pass^4 = `0.00` 时，它的含义是「**没有任何一个任务 4 次采样全部正确**」，而不是「所有任务都失败」；
- 代码在 `EvalReport.pass_hat()` 的 note 里同时给出有偏但连续的参考值 `mean((c/n)^k)`，并在文档里标注其为**有偏估计**（朴素估计把「从同一批样本里取 k 个」误当成「独立重采 k 次」）。
- 报告写法建议：主表用无偏值，紧跟一行括注「有偏参考 `(c/n)^4` = `<待填>`」，否则读者会认为 pass^4 恒为 0 是 bug。

### 2.2 Δstability 怎么读

`Δstability = pass^k − pass@k`（代码事实：`EvalReport.delta_stability()`，对**每个任务**分别算 `pass_hat_k_unbiased − pass_at_k_unbiased` 再做任务级 bootstrap，seed=7）。

三种典型形态：

| 形态 | 数值特征 | 含义 | 后续动作 |
|---|---|---|---|
| A. 稳且会做 | `pass@k` 高、`Δ` 接近 0 | 能力与稳定性都到位 | 无需针对稳定性做任何事 |
| B. 会做但飘 | `pass@k` 高、`Δ` 显著为负（例如 pass@4=1.0 而 pass^4=0.0，Δ=−1.0） | 模型**知道怎么做**，但输出分布太散、一次成功靠运气 | 这是 DPO 最该改善的量；也是本项目主指标选 pass^k 的直接理由 |
| C. 既不会也不稳 | `pass@k` 也低 | 能力缺失为主 | 先按 §3 定位错误类别，别指望靠「稳定性手段」解决 |

**形态 B 是基线最可能出现的形态，也是最容易被 accuracy 掩盖的形态。** 数学上的必然性：`pass^k ≤ pass@k` 恒成立（`tests/test_methodology.py` 有 `test_pass_hat_k_is_stricter_than_pass_at_k` 守护这条不等式），k 越大缺口越大；如果 baseline 的 `pass@4` 已经接近 `1.00`，`Δstability` 必然会显著负，这**不代表模型坏**，恰恰说明「题目本身会做，只是不稳」。

因此**报告里必须同时写出 pass@k 和 pass^k**，只写一个都会让读者误判。

> **一张图上最直观的呈现**：`docs/figures/pass_k_curve_*.png`（由 `src/utils/plots.py` `plot_pass_k_curve` 生成）把每个实验的 pass@k（实线）与 pass^k（虚线）画在一起，**两条线之间的垂直距离就是「稳定性缺口」**。DPO 的目标是「把虚线抬上去、把缺口压窄」，而不是「把实线抬得更高」。

### 2.3 与后续实验的可比性锚点

基线是全部消融实验的分母。填完 §2 后，请**立刻**把四个 headline 值抄进 `docs/ablation/ablation_plan.md` §8 的基线行，并在两处都标注数据来源文件路径 —— 消融表里出现「基线数字与诊断报告不一致」是常见的低级错误。

---

## §3 错误分布

### 3.1 分类体系与优先级（代码事实）

代码事实（`src/eval/error_taxonomy.py`）：

```
SUCCESS                     工具名与（宽松口径）参数全对
├─ 失败类型（互斥，优先级从高到低）
├─ FORMAT_ERROR            没输出可解析的 tool_call（含语法坏、承诺式回复）
├─ HALLUCINATION           工具名不在白名单
├─ WRONG_TOOL              工具名合法但不是期望的那个
├─ PARAM_ERROR             工具名对，但参数不对（缺失/占位符/值错）
├─ REFUSED                 明确拒绝或放弃任务
└─ UNKNOWN                 其他
```

`ALL_CATEGORIES = (SUCCESS, FORMAT_ERROR, HALLUCINATION, WRONG_TOOL, PARAM_ERROR, REFUSED, UNKNOWN)`，`FAILURE_CATEGORIES` = 去掉 SUCCESS。

中文标签（`CATEGORY_LABELS_ZH`，报告与图表必须用它，保证跨文件命名一致）：

| 常量 | 中文 |
|---|---|
| `SUCCESS` | 成功 |
| `FORMAT_ERROR` | 格式错误 |
| `HALLUCINATION` | 工具幻觉 |
| `WRONG_TOOL` | 工具选错 |
| `PARAM_ERROR` | 参数错误 |
| `REFUSED` | 拒绝作答 |
| `UNKNOWN` | 未知错误 |

**优先级规则：为什么格式错误优先于工具名错误**

代码事实（`classify()` 的执行顺序 + 模块文档）：

1. **先判「有没有可用的调用」**。`calls` 为空时直接进 FORMAT_ERROR / REFUSED / SUCCESS（后者仅当 `expected_tool is None`）。
2. 再取 `calls[0]`，检查 `arguments` 是不是对象。
3. 再判「该轮本不该调用工具」（`expected_tool is None`）→ `WRONG_TOOL`（子类型 `should_clarify`）。
4. 再判工具名是否在白名单 → 不在则 `HALLUCINATION`。
5. 再判工具名是否等于期望 → 不等则 `WRONG_TOOL`。
6. 最后才比参数 → `PARAM_ERROR`。

理由（模块文档原话）：「格式坏」优先于「工具名错」：**如果 JSON 根本解析不出来，就没法讨论工具名对不对**。这个顺序保证了各类别占比的**可加性**：所有类别之和 = 样本总数，不会重复计数。

> **⚠️ 读数据时必查的两处不一致（见 §9.4）**
> 1. `classify()` 在 `expected_tool is None` 且模型贸然调用时设置的子类型是 `should_clarify`，但 `SUBTYPE_LABELS_ZH` **没有这个键**，报告里会显示英文原文（其余子类型都有中文）。填表时手工补写中文「信息不足时应先追问」。
> 2. `REFUSED` 的检测（`detect_refusal`）在 `calls` 为空时**只覆盖「无法/不能」类话术**；如果模型既没调工具又命中拒绝正则，会被归到 `REFUSED` 而不是 `FORMAT_ERROR`。这是有意的（拒绝不是格式问题），但意味着 `REFUSED` 与 `FORMAT_ERROR` 的边界依赖正则表，跨版本不可直接比。

### 3.2 错误分布表

**表 1：按 7 个类别（SUCCESS + 6 个失败类）**

> `占总样本` = `ErrorDistribution.rate(cat)`；`占失败` = `failure_share(cat)`；两者之和分别 = 1.0。

| 错误类别（中文） | 常量 | 数量 | 占总样本 | 占失败 | 观测含义（这一类错误说明模型在哪一步掉了） |
|---|---|---|---|---|---|
| 格式错误 | `FORMAT_ERROR` | `<待填>` | `<待填>` | `<待填>` | 没能把「意图」落成 `<tool_call>`：可能是纯文本承诺、坏 JSON、markdown 围栏、输出为空、或被 `max_new_tokens` 截断 |
| 工具幻觉 | `HALLUCINATION` | `<待填>` | `<待填>` | `<待填>` | 格式合法，但工具名不在 6 工具白名单内（子类型区分「完全没见过」与「貌似合理的命名变体」） |
| 工具选错 | `WRONG_TOOL` | `<待填>` | `<待填>` | `<待填>` | 工具名合法但选错，或该澄清时贸然调用了写操作 |
| 参数错误 | `PARAM_ERROR` | `<待填>` | `<待填>` | `<待填>` | 工具选对，但参数字段缺失 / 填占位符 / 值错 / 格式不合 schema |
| 拒绝作答 | `REFUSED` | `<待填>` | `<待填>` | `<待填>` | 明确表示无法完成（命中 `_REFUSAL_PATTERNS`） |
| 未知错误 | `UNKNOWN` | `<待填>` | `<待填>` | `<待填>` | 兜底（例如工具名对但未提供参数比对结果） |
| 成功 | `SUCCESS` | `<待填>` | `<待填>` | — | 工具名 + 宽松口径参数全对 |
| — 合计样本 | — | `<待填>` | 100% | 100% | = 100 任务 × 4 采样 = 400（若 `task_limit=0`） |
| 示例行（占位） | — | `—` | `—` | `—` | — |

<!-- TODO(真实数据): 从 experiments/baseline_airline/error_analysis.json 的 distribution.counts 填入数量 -->
<!-- TODO(真实数据): 从 experiments/baseline_airline/error_analysis.json 的 distribution.rates 填入占总样本 -->
<!-- TODO(真实数据): 从 experiments/baseline_airline/error_analysis.json 的 distribution.failure_share 填入占失败 -->
<!-- TODO(真实数据): 从 experiments/baseline_airline/eval_report.json 的 metadata.num_predictions 填入合计样本数 -->

**表 2：子类型下钻（Top-N，只列失败子类型）**

`error_analysis.json` → `distribution.failure_subtype_counts` 已剔除 `-`（成功且无子类型）、`clarify_instead`、`extra_param` 三个键，避免「成功」混进错误排行榜（代码事实：`ErrorDistribution.to_dict()`）。

| 子类型常量 | 中文（`SUBTYPE_LABELS_ZH`） | 数量 | 属于哪个类别 | 观测含义 |
|---|---|---|---|---|
| `no_tool_call` | 完全没有 tool_call 标签 | `<待填>` | `FORMAT_ERROR` | 输出里既没有标签，也不像承诺、也不像澄清 |
| `promise_only` | 只承诺要查，没真的调用 | `<待填>` | `FORMAT_ERROR` | 命中 `PROMISE_MARKERS`（「我来帮您」「请稍等」…） |
| `json_broken` | JSON 语法非法 | `<待填>` | `FORMAT_ERROR` | 有 `<tool_call>` 标签但内部解析失败（键缺引号、中文键等） |
| `truncated` | 输出被 max_new_tokens 截断 | `<待填>` | `FORMAT_ERROR` | `finish_reason == "length"` 或未闭合标签 |
| `empty` | 输出为空 | `<待填>` | `FORMAT_ERROR` | 极端情况 |
| `unknown_tool` | 工具名不在白名单 | `<待填>` | `HALLUCINATION` | 与下面的「貌似合理」相对 |
| `plausible_fake_tool` | 工具名是「貌似合理但不存在」的名字 | `<待填>` | `HALLUCINATION` | 命中 `PLAUSIBLE_FAKE_TOOLS`，**最难修的一类** |
| `adjacent_tool` | 工具名属于同域相邻工具 | `<待填>` | `WRONG_TOOL` | 例如 `check_baggage` ↔ `modify_baggage` |
| `should_clarify` | （无中文标签，需手工补） | `<待填>` | `WRONG_TOOL` | 信息不足时应追问却直接调用写操作 |
| `placeholder_value` | 参数填了占位符 | `<待填>` | `PARAM_ERROR` | 命中 `PLACEHOLDER_KEYWORDS`（`previous_booking`、`未知`…） |
| `missing_required` | 缺少必填参数 | `<待填>` | `PARAM_ERROR` | |
| `wrong_value` | 参数值不对 | `<待填>` | `PARAM_ERROR` | 见 §5（含「格式对但值错」子集） |
| `wrong_format` | 参数格式不合 schema | `<待填>` | `PARAM_ERROR` | 注意：`classify()` 当前**不产出**这个子类型，见 §9.4 |
| `out_of_range` | 参数超出取值范围 | `<待填>` | `PARAM_ERROR` | 同上 |
| `refusal` / `cannot_help` | 模型明确拒绝 / 表示无法完成 | `<待填>` | `REFUSED` | |
| `clarify_instead` | 该调用但只做了澄清 | `<待填>` | `FORMAT_ERROR` | 注意：也出现在 `expected_tool is None` 的 **SUCCESS** 里（此时含义相反，是正确行为） |
| 示例行（占位） | — | `—` | — | — |

<!-- TODO(真实数据): 从 experiments/baseline_airline/error_analysis.json 的 distribution.failure_subtype_counts 填入各子类型数量，按数量降序排列 -->

**关于 `clarify_instead` 的双重含义（容易读错）**：同一个子类型字符串在两种上下文里含义相反 ——

- `expected_tool is None` 的任务（5 个：4 个需澄清的取消任务 + 1 个取消原因编造的恢复任务）里，输出纯文本追问是**正确行为**，记为 `SUCCESS`，子类型 `clarify_instead`；
- 其他任务里只追问不调用则是 **失败**，记为 `FORMAT_ERROR`，子类型同为 `clarify_instead`。

因此**不要**直接用 `subtype_counts["clarify_instead"]` 算「该调用不调用」的比例，必须配合任务侧的 `expected.tool is None` 过滤。

### 3.3 图表

| 图 | 路径 | 生成命令 | 内容 |
|---|---|---|---|
| 错误分布 | `docs/figures/error_distribution.png` | `python scripts/eval/analyze_errors.py --label baseline_airline` | 左：各类别饼图（固定颜色，`CATEGORY_COLORS`）；右：失败子类型 Top 8 横向柱状图 |
| 分组成功率（难度） | `docs/figures/success_by_difficulty.png` | 同上（仅当 label 以 `baseline` 开头且**非 mock**） | 每组两根柱：accuracy 与 pass^4 |
| 分组成功率（可见性） | `docs/figures/success_by_visibility.png` | 同上 | unseen 柱是泛化证据 |

> **⚠️ 图表命名是带 label 的**：`analyze_errors.py` 总是先写 `error_distribution_<label>.png`；只有在 `label.startswith("baseline")` **且** `metadata.is_mock` 为假时，才**额外**写一份不带 label 的 `error_distribution.png`（代码事实：`analyze_one()`）。
> 因此：
> 1. 请把基线目录命名为 `baseline_airline`（即以 `baseline` 开头），否则 README 引用的默认图名不会被生成；
> 2. 如果只跑过 mock，`docs/figures/error_distribution.png` **不存在**（mock 图会带 `.mock` 后缀或带 mock label）—— 不要在 README 里先引用一个不存在的图。

**每类错误对应的观测含义（写报告时要落到「模型在哪一步掉」这一层）**

| 类别 | 单看数量能得出什么 | 必须配合什么才能得出结论 |
|---|---|---|
| `FORMAT_ERROR` 高 | 模型的「意图 → 动作」映射弱 | 看子类型：`promise_only` 说明是**没落成调用**；`json_broken` 说明是**语法能力**问题；`truncated` 说明是**生成长度**问题。三者对应的偏好数据完全不同 |
| `HALLUCINATION` 高 | 模型在编工具名 | 看 `plausible_fake_tool` 占比：如果绝大多数是命名变体（`find_flights` vs `search_flights`），说明模型**不懂「逐字匹配白名单」**；如果是不相干的工具（`search_hotel`），说明它**没读 system prompt 的工具清单** |
| `WRONG_TOOL` 高 | 模型分不清 6 个工具的语义边界 | 看是不是集中在 `check_baggage`/`modify_baggage`、`search_flights`/`book_flight` 这两对相邻工具上（`_pick_adjacent_tool` 的 `prefer` 表就是这个先验） |
| `PARAM_ERROR` 高 | 参数填充是瓶颈 | 必须看 §5：这些错误里有多少是「格式对但值错」。这是判断「能不能靠格式约束对修好」的分水岭 |
| `REFUSED` 高 | 模型宁愿放弃也不冒险 | 可能是 system prompt 的「信息不足时先追问」被过度泛化成「拒绝」 |
| `UNKNOWN` 非零 | 评分器有未覆盖路径 | 必须人工抽检，并按 `CATEGORY_REMEDIES_ZH[UNKNOWN]`（「人工抽检，补充新类别」）行动 |

---

## §4 分组下钻

### 4.1 为什么必须分组看

**整体提升可能完全来自简单任务。** 这是本项目最需要在报告里主动暴露的偏差：

- 如果只看总准确率从 A 涨到 B，而分组后发现涨幅全部来自 `easy` 组、`hard` 组纹丝不动，那么正确的结论是「**在简单任务上刷分**」，不是「能力提升」；
- `unseen` 上的提升才是**泛化证据**。代码事实：67 个任务在 5 个 epoch 训练时会被用到（`visibility=seen`），33 个任务训练完全不接触（`unseen`），而且划分是按类别轮转的（`counter % 3 == 0`），所以**每个类别在 unseen 里都有分布**（baggage 5 / booking_flow 6 / cancellation 4 / flight_search 7 / multi_tool 3 / recovery 4 / weather_advisory 4）。这个划分设计本身就是为了让 unseen 结果有解释力。

因此本报告有 **3 张分组表**，且**每张都单独给 pass^4** —— 只看 accuracy 会在小分组上和 pass^k 给出相反的趋势。

### 4.2 按难度

**表 1：按难度分组**（来源：`eval_report.json` → `by_difficulty`）

| 难度 | 定义（代码事实，`task_manifest` 模块文档） | 任务数 | accuracy | accuracy 95% CI | pass@4 | pass^4 | Δstability |
|---|---|---|---|---|---|---|---|
| `easy` | 单工具、参数直接可提取（或已给全） | 43 | `<待填>` | `[<待填>, <待填>]` | `<待填>` | `<待填>` | `<待填>` |
| `medium` | 需要一次推理变换（相对日期换算、城市名→三字码、从 observation 取 flight_id） | 32 | `<待填>` | `[<待填>, <待填>]` | `<待填>` | `<待填>` | `<待填>` |
| `hard` | 多步依赖，或包含用户纠错轮 | 25 | `<待填>` | `[<待填>, <待填>]` | `<待填>` | `<待填>` | `<待填>` |
| 示例行（占位） | — | — | `—` | `—` | `—` | `—` | `—` |

<!-- TODO(真实数据): 从 experiments/baseline_airline/eval_report.json 的 by_difficulty.easy / .medium / .hard 依次填入 num_tasks / accuracy / accuracy_ci95 / pass@4 / pass^4 -->

> **注意**：`by_difficulty` 的切片摘要**只给 accuracy 的 CI**，`pass@4` / `pass^4` 是无 CI 的点值（代码事实：`EvalReport._slice_summary()`）。因此分组表里 `pass^4` 的 CI 列留空是**正常**的，不要为了凑格式编数字。若确实需要分组 CI，需在分析脚本里对子集重跑 bootstrap。

**这张表要支撑的判断**：

1. `easy` 组准确率是否已经接近饱和？如果 `easy` 的 pass@4 已接近 1.00 而 pass^4 明显低，说明简单任务上的失败是**纯稳定性问题**，靠扩数据量解决不了，需要靠偏好对齐（或降低采样温度）。
2. `hard` 组的失败是否集中在 `FORMAT_ERROR`？如果是，说明多步上下文（含 observation 的轮次）反而干扰了格式输出——这是一个具体的、可验证的假设。
3. 难度梯度是否单调？`easy > medium > hard` 若不成立（例如 medium 反而高于 easy），说明难度标注与真实难度不符，需要回到 `task_manifest` 复核这类任务的构造（例如 `baggage` 的 check/modify 难度标注）。

### 4.3 按类别（7 类）

**表 2：按类别分组**（来源：`eval_report.json` → `by_category`；「考察点」列来自 `task_manifest` 模块文档的类别分布表，属**代码事实**）

| 类别 | 数量 | 考察点（代码文档原文） | accuracy | pass@4 | pass^4 | 该组的头号错误类别 |
|---|---|---|---|---|---|---|
| `flight_search` | 22 | 城市名→机场码、相对日期→绝对日期 | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| `booking_flow` | 18 | 多步：先搜再订，`flight_id` 必须来自 observation | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| `baggage` | 15 | 订单号格式、数值参数 | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| `cancellation` | 12 | 写操作前的确认、原因参数 | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| `weather_advisory` | 12 | 参数语义区分（天气用中文城市名，航班用三字码） | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| `multi_tool` | 9 | 一轮内串起多个工具 | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| `recovery` | 12 | 用户纠错后的自我修正（Week 2 错误恢复偏好对的数据源） | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| 示例行（占位） | — | — | `—` | `—` | `—` | — |

<!-- TODO(真实数据): 从 experiments/baseline_airline/eval_report.json 的 by_category.<类别> 填入 num_tasks / accuracy / pass@4 / pass^4 -->
<!-- TODO(真实数据): 「头号错误类别」从 experiments/baseline_airline/error_analysis.json 的 cross_tab 交叉表按类别聚合后取最大值 -->

**这张表要支撑的判断**：

1. **`weather_advisory` 是本项目有意设计的「参数语义陷阱」**：`get_weather.city` 要中文城市名，而 `search_flights.origin/destination` 要三字码 —— 同一个模型、同一类实体、要求相反。如果这一组准确率显著低于其他 easy 组，就直接证明了「小模型对参数语义的区分依赖逐条 description，而不是全局规则」这一根因假设（§7）。
2. **`booking_flow` 的 `from_observation` 子任务**（10 个，`tags=("from_observation",)`）要求 `flight_id` 从 observation 里抄。如果这一组错得比同类其他任务多，说明模型不看 observation 或抄错实体 —— 这正是 `--no-observations` 混淆因素要排除的方向（消融计划 §6）。
3. **`cancellation` 的 4 个 `clarify_first` 任务**（`expected.tool is None`）是「该不调用」的否定样本。这一组的失败形态是 `WRONG_TOOL/should_clarify`，与「该调用却不调用」正好相反，必须分开讨论。
4. `multi_tool` 只有 9 个任务，**任何差异都在噪声量级**。代码事实：任务级 bootstrap 在 9 个单元上给出的 CI 会非常宽。报告里要给数字，但**不要**基于 9 个任务下结论。

### 4.4 按可见性（seen / unseen）

**表 3：按可见性分组**（来源：`eval_report.json` → `by_visibility`）

| 可见性 | 定义 | 任务数 | accuracy | accuracy 95% CI | pass@4 | pass^4 |
|---|---|---|---|---|---|---|
| `seen` | 5 个 epoch 训练时会用到（偏好数据构造源） | 67 | `<待填>` | `[<待填>, <待填>]` | `<待填>` | `<待填>` |
| `unseen` | 训练完全不接触，只用于评测 | 33 | `<待填>` | `[<待填>, <待填>]` | `<待填>` | `<待填>` |
| 差值（unseen − seen） | — | — | `<待填>` | — | `<待填>` | `<待填>` |
| 示例行（占位） | — | — | `—` | `—` | `—` | `—` |

<!-- TODO(真实数据): 从 experiments/baseline_airline/eval_report.json 的 by_visibility.seen / .unseen 填入 -->

> **⚠️ 读这张表前必须澄清一件事**：`baseline` 是**未训练**模型，所以 seen/unseen 对它**在统计上应当没有系统差异**。基线里的 seen−unseen 差值应当落在噪声范围内。
> 这个差值的真正用途是**给后续实验当对照**：如果 `dpo_*` 实验在 seen 上涨了 20% 而 unseen 只涨 1%，那「提升」很可能只是记住了训练任务的表面模式，不是泛化能力。
> 所以这张表的基线版本应该在报告里明确写成「未训练模型上的 seen/unseen 基线差异，用于后续实验的泛化性对照」，而**不是**「泛化能力评估」。

### 4.5 分组下钻的正确结论写法（模板）

避免这种写法：「整体准确率 <A>，其中简单任务 <B>，困难任务 <C>。」（只罗列数字）

应该这样写：

```
整体 accuracy <A>、pass^4 <P>。拆开后：
- 涨幅/失败集中在 <难度或类别> 组：<该组数字>，其头号错误是 <类别>（占该组失败 <x>%）。
- <另一组> 基本未变：<数字>。
- 因此本轮的主要瓶颈是 <具体能力>，而不是 <被排除的能力>。
- 佐证：unseen 组 pass^4 为 <数字>，与 seen 组差 <Δ>；<Δ 在噪声内 → 泛化成立 / Δ 显著为负 → 存在任务记忆>。
```

---

## §5 「格式对但值错」这一退化模式

### 5.1 为什么必须把「格式正确率」和「任务成功率」分开报告

这是 DPO 最典型的退化路径，也是本项目要**主动暴露**的风险：

1. 偏好数据里最容易构造的就是格式约束对（合法 `<tool_call>` vs 纯文本承诺）；
2. 模型很快学会「输出合法 JSON」这件事；
3. **格式正确率飙升，任务成功率不动**；
4. 如果报告只写「格式正确率从 X 提升到 Y」，读者会以为模型变强了，实际上模型学会了**格式捷径**：输出一个结构完美、但实体乱填的调用（例如订单号 `BK000000`、航班号 `CA0000`、日期 `2025-01-01`）。

所以两个指标必须**并排报告**，并且专门监控「格式对但值错」这个交叉量。

代码事实：`src/eval/tool_call_parser.py` `format_accuracy()` 的文档原话就是「DPO 常见的退化模式就是格式率飙升而任务成功率不动（reward hacking），所以两个指标必须分开报告」。

### 5.2 这两个量的精确定义

| 量 | 定义 | 代码位置 | 分母 |
|---|---|---|---|
| 格式正确率 `format_accuracy` | `is_canonical_format(text)` 为真的比例 | `src/eval/tool_call_parser.py` | **全部采样**（= 任务数 × n） |
| `GradeResult.format_ok` | 单条采样是否满足 canonical 格式 | `src/eval/grader.py` `grade()` | 单条 |
| `format_ok_value_wrong` | 是否「格式对但值错」 | `GradeResult` 字段；置值逻辑在 `error_taxonomy.classify()` | 单条 |
| `format_ok_value_wrong_rate` | 上一项的比例 | `scripts/eval/run_baseline.py` 写进 `error_analysis.json` | **全部采样**（不是「参数错误的采样」） |

**表：退化模式监控指标**

| 指标 | 值 | 来源 | 健康区间（判据在消融计划 §9） |
|---|---|---|---|
| 任务成功率（主指标 pass^4） | `<待填>` | `eval_report.json` → `headline.pass_hat_k` | 越高越好，且**必须与格式率同向变化** |
| 单次准确率 accuracy | `<待填>` | `eval_report.json` → `headline.accuracy` | — |
| 格式正确率 | `<待填>` | `error_analysis.json` → `format_accuracy` | 高本身不是好事，是**必要不充分**条件 |
| 「格式对但值错」条数 | `<待填>` | `error_analysis.json` → `format_ok_value_wrong` | 越低越好 |
| 「格式对但值错」比率 | `<待填>` | `error_analysis.json` → `format_ok_value_wrong_rate` | 上升即为退化信号 |
| 严格口径 vs 宽松口径的差值（严格更低多少） | `<待填>` | `eval_report.json` → `per_task[].predictions[].strict_ok` 与 `.lenient_ok` 的计数差 | 差值扩大 = 学会「关键字段对、细节不管」的捷径 |
| 示例行（占位） | `—` | — | — |

<!-- TODO(真实数据): 从 experiments/baseline_airline/error_analysis.json 的 format_accuracy 填入 -->
<!-- TODO(真实数据): 从 experiments/baseline_airline/error_analysis.json 的 format_ok_value_wrong 与 format_ok_value_wrong_rate 填入 -->
<!-- TODO(真实数据): 从 experiments/baseline_airline/eval_report.json 的 per_task[*].predictions[*].strict_ok / lenient_ok 统计严格与宽松口径的命中数，填差值 -->

### 5.3 ⚠️ 一个必须写进报告的判定口径陷阱

`is_canonical_format()` 用的是**正则结构匹配**，不是真的 JSON 解析。代码事实（`tool_call_parser._STANDARD_CALL_RE` + 实测）：

```text
<tool_call>{"name": "get_weather", "arguments": {"city": "北京"}}</tool_call>          → format_ok = True   （正确）
<tool_call>{"name": "check_baggage", "arguments": {booking_id: BK7F2A9C}}</tool_call>  → format_ok = True   ← ⚠️ JSON 其实非法
<tool_call>{"name": "search_flights", "arguments": {city: "SHA"}}</tool_call>          → format_ok = True   ← ⚠️ 键缺引号
<tool_call>{"name": "get_weather", "arguments": {"city": "北京"}}                       → format_ok = False  （未闭合，正确判负）
好的，我来帮您查询一下，请稍等。                                                          → format_ok = False  （正确判负）
```

也就是说：**`format_accuracy` 的真实含义是「输出看起来像一条 `<tool_call>`，而不是合法 JSON」**。

后果（必须在报告里说清，否则会得出错误结论）：

- 一个**键缺引号**的输出会同时满足「格式正确率 +1」和「判为 `FORMAT_ERROR/json_broken`」。所以 `format_accuracy` 与 `1 − FORMAT_ERROR 占比` **不相等**，两者相差的正是这类「有标签但 JSON 坏」的样本；
- 因此「格式率飙升」可能只是模型学会了**加标签**，而不是学会了**合法 JSON**。要区分这两种情况，必须看 `FORMAT_ERROR` 的子类型分布（`json_broken` vs `promise_only` vs `no_tool_call`）。

报告建议写法：把格式率拆成三项报告 —— `promise_only / no_tool_call`（没尝试）、`json_broken`（尝试了但语法坏）、canonical（结构合法）。三项之和 ≈ 全部样本。

### 5.4 `format_ok_value_wrong` 的置值规则与已知漏报

代码事实（`error_taxonomy.classify()` 第 393-427 行）：

```text
只有在 PARAM_ERROR 的 wrong_value 分支里，才会计算：
    format_ok_value_wrong = all(_same_shape(c.expected, c.actual) for c in failures)
```

`_same_shape()` 的判定：两个值都能解析成日期 → True；都能转成数值 → True；两个字符串**长度相同** → True；否则 False。

由此推出四条**已知漏报**（写报告时必须声明，否则会低估退化风险）：

| 情形 | 是否被计入 `format_ok_value_wrong` | 原因 |
|---|---|---|
| `date` 填了 `"后天"`（相对日期未换算） | ✗ **不计入** | `normalize_date("后天")` 返回 None，形状不同 → 但这是最典型的「值错」 |
| 参数是占位符（如 `previous_booking`） | ✗ 不计入 | 走 `placeholder_value` 分支，该分支不设置此字段 |
| 缺少必填参数 | ✗ 不计入 | 走 `missing_required` 分支 |
| `origin` 填 `"北京"`（语义用反） | ✗ 不计入 | 报告 `PARAM_ERROR/wrong_value`，但 `_same_shape("PEK","北京")` 长度为 3 vs 2 → False |
| 订单号 `BK7F2A9C` → `BK7F2A9D` | ✓ 计入 | 长度相同 → 形状一致 |
| 行李 30 → 35 | ✓ 计入 | 都是数值 |

**结论**：`format_ok_value_wrong_rate` 是退化信号的**下界**，不是全貌。报告里必须补一个更宽的统计口径，建议同时给：

- 窄口径（现成字段）：`format_ok_value_wrong_rate`；
- 宽口径（需自行从 `predictions.jsonl` 统计）：`format_ok == True` 且 `category in {PARAM_ERROR, WRONG_TOOL, HALLUCINATION}` 的比例 —— 即「格式全对但没做对任务」。

> **宽口径统计命令（建议直接落成脚本，避免手工算错）**
> ```bash
> python - <<'PY'
> import json, pathlib
> p = pathlib.Path("experiments/baseline_airline/eval_report.json")
> d = json.loads(p.read_text(encoding="utf-8"))
> tot = fmt = wide = narrow = 0
> for t in d["per_task"]:
>     for pr in t["predictions"]:
>         tot += 1
>         fmt += int(pr["format_ok"])
>         narrow += int(pr["format_ok_value_wrong"])
>         wide += int(pr["format_ok"] and pr["category"] != "SUCCESS")
> print(f"n={tot} format_ok={fmt/tot:.4f} narrow={narrow/tot:.4f} wide={wide/tot:.4f}")
> PY
> ```
> 这条命令**只读** `eval_report.json` 的 `per_task`，不做任何推断。

### 5.5 这一节要得出的判断

填完数据后，报告里必须明确回答：

1. 基线的格式率与成功率**是否已经脱钩**？（两者都低 = 还没到退化阶段；格式率明显高于成功率 = 已经存在捷径倾向）
2. 若脱钩，是「加了标签但 JSON 坏」（看 `json_broken` 占比）还是「JSON 合法但实体乱填」（看宽口径 + 窄口径）？
3. 这直接决定 §8 里参数约束对（`param_constraint`）的**优先级**：如果窄口径 + 宽口径都不低，参数约束对就不是「锦上添花」而是**必备项**——这正是 `EXPERIMENT_PAIR_TYPES` 里额外提出 `dpo_joint_param` / `dpo_all` 的原因，也是 `configs/train/dpo_joint.yaml` 的 `_falsifier` 所警告的风险。

---

## §6 错误恢复能力

### 6.1 为什么恢复能力必须用「真实多轮」测

代码事实（`src/models/agent_loop.py` 模块文档）：`grader.py` 评的是**目标轮**（给定固定前缀评一轮），但「错误恢复」这个能力的第二步**依赖第一步的实际输出**——模型先错、框架判错、注入纠错、模型再答。前缀不能预先固定，必须真跑多轮。

所以基线诊断有**两条独立的评测线**：

| 评测线 | 脚本 | 评什么 | 主指标 |
|---|---|---|---|
| 目标轮评测 | `scripts/eval/run_baseline.py` | 给定固定多轮前缀，评目标轮 | pass^k（§2） |
| 真实多轮评测 | `scripts/eval/eval_recovery.py` | 只跑 12 个 `recovery` 任务，真实多轮 + 框架注入纠错 | `task_success_rate` / `first_try_rate` / `recovery_rate` / `repeat_rate` / `avg_turns` |

命令：

```bash
# 真实多轮恢复评测（GPU）
python scripts/eval/eval_recovery.py --label baseline_airline --repeat 4 --max-turns 6 --max-corrections 2
# 产物：experiments/baseline_airline/recovery_metrics.json + dialogues.jsonl
```

代码事实（`eval_recovery.py` 模块文档的指标表，这是**权威定义**）：

| 指标 | 定义 | 代码 |
|---|---|---|
| `task_success_rate` | 对话最终成功的比例 = `success / n` | `metrics["task_success_rate"]` |
| `first_try_rate` | 第一轮就做对、**不需要纠正**的比例 = `count(errors == 0 and success) / n` | `metrics["first_try_rate"]` |
| `recovery_rate` | 被纠正后改对的比例 = `Σrecoveries / Σerrors` | `metrics["recovery_rate"]` |
| `repeat_rate` | **纠正后仍原样重复同一错误**的比例 = `count(corrections_used > 0 and not success) / n` | `metrics["repeat_rate"]` |
| `avg_turns` | 平均消耗轮数（越少越好） | `metrics["avg_turns"]` |
| `avg_corrections_used` | 平均用掉几次纠错预算（预算 `max_corrections`，默认 2） | `metrics["avg_corrections_used"]` |

终止条件（代码事实，`run_dialogue()`）：达到 `max_turns`（默认 6）｜输出成功｜连续两轮输出完全相同（`repeat_loop`）｜工具调用失败超过 `max_tool_errors`（默认 3）｜生成被截断（`finish_reason == "length"`）。

### 6.2 ★ `recovery_rate` 与 `repeat_rate` 的区别（本项目的关键指标设计决策）

**这是本节最重要的一段，也是最容易被写错的一段。**

**`recovery_rate` 的分母是「犯错的次数」，因此不可跨方案直接比较。**

`recovery_rate = Σrecoveries / Σerrors`。假设两个方案：

| 方案 | 犯错次数 | 改对次数 | `recovery_rate` | 绝对改对量 |
|---|---|---|---|---|
| A | 10 | 8 | 0.80 | 8 |
| B | 2 | 2 | 1.00 | 2 |

按 `recovery_rate` 排序，B 更好（100% > 80%）。但 B **本来就只错了 2 次**，它「恢复率 100%」是因为分母小；真正会改对的绝对数量 A 远多于 B。也就是说：**一个方案只要能减少首轮错误，`recovery_rate` 就会自动变得不稳定（甚至虚高或虚低）**——分母变了，指标的含义就变了。

**`repeat_rate` 的分母是「样本数」，含义是「说了也不改」的比例，因此跨方案可比。**

`repeat_rate = count(用过纠错 且 最终仍未成功) / n`。分母固定为对话次数（= 12 任务 × `--repeat`），所以「A 方案 15%、B 方案 30%」可以直接读成「B 方案里，被纠正后仍不改的比例是 A 的两倍」。

**为什么 `repeat_rate` 更重要**：它**直接对应本项目要解决的问题**，而不是一个派生比率。`eval_recovery.py` 的模块文档把它的重要性写成了一行注释：

```
repeat_rate   纠正后仍然原样重复同一错误的比例 ← 最直接的失败模式
```

而且它有直接的因果解释：**小模型在纠错轮最常见的失败不是「犯一个新错」，而是「原样重复上一轮的错误」** —— 这一点在 `src/mock/mock_policy.py` 的纠错轮注入逻辑里也被显式建模（以 `1 − recovery_rate` 的概率**重放首轮那段错误输出**）。

报告写法要求：

1. **主表里 `repeat_rate` 放在 `recovery_rate` 前面**，并加粗/加星标；
2. `recovery_rate` 必须**同时给出 `total_errors` 与 `total_recoveries` 两个绝对量**（这两个字段 `eval_recovery.py` 都写了），否则读者无法判断分母大小；
3. 跨方案比较时**只用 `repeat_rate`、`first_try_rate`、`task_success_rate`** 这三个同分母指标。

### 6.3 指标表（真实多轮）

**表 1：恢复能力总览**（来源：`experiments/baseline_airline/recovery_metrics.json`）

| 指标 | 值 | 分子/分母（必须一起填） | 读法 |
|---|---|---|---|
| `num_dialogues` | `<待填>` | — | = 12 任务 × `repeat` |
| `task_success_rate` | `<待填>` | `<成功次数>` / `<num_dialogues>` | 最终能完成的整体能力 |
| **`repeat_rate`（★ 跨方案可比）** | `<待填>` | `<说了也不改的次数>` / `<num_dialogues>` | 越低越好；**这是主指标** |
| `first_try_rate` | `<待填>` | `<errors==0 且成功>` / `<num_dialogues>` | 首轮就对的概率；**恢复类训练若让它下降，说明模型学会了「先错后改」** |
| `recovery_rate`（不可跨方案比） | `<待填>` | `<total_recoveries>` / `<total_errors>` | 必须附带下面两个绝对量 |
| `total_errors` | `<待填>` | — | 犯错总次数（`recovery_rate` 的分母） |
| `total_recoveries` | `<待填>` | — | 改对总次数 |
| `avg_turns` | `<待填>` | — | 越少越好 |
| `avg_corrections_used` | `<待填>` | 上限 = `max_corrections` | 接近上限说明模型基本不肯自己改对 |
| 示例行（占位） | `—` | `—` / `—` | — |

<!-- TODO(真实数据): 从 experiments/baseline_airline/recovery_metrics.json 的 task_success_rate / repeat_rate / first_try_rate / recovery_rate / total_errors / total_recoveries / avg_turns / avg_corrections_used / num_dialogues 依次填入 -->

**表 2：逐错误模式（`by_mistake_key`）**

12 个模式名来自 `src/data/task_manifest.py` `_recovery_patterns()` 的 `key` 字段（**代码事实**，每个模式在评测集里恰好 1 个任务；`recovery_corpus` 里用同一套 key 展开出 84 个训练场景，所以训练/评测可以逐类对齐）。

| `mistake_key` | 考察什么（代码事实，来自 `_RecoveryPattern.note`） | n | `task_success_rate` | `recovery_rate` | `repeat_rate` |
|---|---|---|---|---|---|
| `format_ignore_correction` | 格式错误 → 纠错后修正 | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| `hallucinated_tool_corrected` | 工具幻觉 → 纠错后修正 | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| `wrong_tool_corrected` | 工具选错 → 纠错后修正 | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| `param_placeholder_corrected` | 参数占位符 → 纠错后修正 | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| `city_code_corrected` | 参数格式错误（中文名当三字码）→ 纠错后修正 | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| `repeated_same_error` | 编造实体 → 纠错后回到正确前置步骤 | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| `missing_reason_corrected` | 参数编造 → 纠错后改为澄清（正解**不是**工具调用） | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| `wrong_date_corrected` | 相对日期未换算 → 纠错后修正 | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| `overbook_corrected` | 步骤不完整 → 纠错后推进到写操作 | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| `wrong_city_param_corrected` | 参数语义错误（三字码当城市名）→ 纠错后修正 | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| `hallucinated_weather_tool_corrected` | 工具幻觉（第二次）→ 纠错后修正 | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| `format_json_broken_corrected` | JSON 语法错误 → 纠错后修正 | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| 示例行（占位） | — | — | `—` | `—` | `—` |

<!-- TODO(真实数据): 从 experiments/baseline_airline/recovery_metrics.json 的 by_mistake_key.<key> 填入 n / task_success_rate / recovery_rate / repeat_rate -->

> **⚠️ `n` 列可能不等于 4**：`eval_recovery.py` 在单次对话抛异常时会 `continue` 跳过（只在控制台打印 `⚠️ ...`），跳过的那次不会进 `all_results`。所以每个 key 的 n 应当 = `--repeat`（默认 4），若小于 4 说明该任务有执行失败，**必须在报告里说明**，否则跨 key 比较失真。

**表 3：停止原因分布**（`recovery_metrics.json` → `stop_reasons`，按次数降序）

| `stop_reason` | 含义（代码事实） | 次数 |
|---|---|---|
| `success` | 某轮输出正确，正常结束 | `<待填>` |
| `repeat_loop` | 连续两轮输出完全相同（死循环，`stop_on_repeat`） | `<待填>` |
| `too_many_tool_errors` | 工具执行失败累计达 `max_tool_errors`（默认 3） | `<待填>` |
| `truncated` | 生成被 `max_new_tokens` 截断 | `<待填>` |
| `max_turns` | 用尽轮数预算 | `<待填>` |
| 示例行（占位） | — | `—` |

<!-- TODO(真实数据): 从 experiments/baseline_airline/recovery_metrics.json 的 stop_reasons 填入 -->

**`repeat_loop` 的次数是一个重要信号**：它说明模型在收到纠错消息后**逐字重复同一输出**。这与 `repeat_rate` 高度相关但**不是同一个量**（`repeat_rate` 判的是「最终不成功」，`repeat_loop` 判的是「相邻两轮逐字相同」），两者一起报能区分「不改」和「改了一下但还是错」。

### 6.4 ⚠️ 同名指标的两种不同定义（必须避免引用错文件）

**这是本项目最危险的一处命名冲突。** 两个脚本都产出名为 `recovery_rate` / `repeat_rate` 的字段，但**定义不同**：

| 来源文件 | 字段 | 实际定义 | 分母 | 可否跨方案比较 |
|---|---|---|---|---|
| `experiments/<label>/recovery_metrics.json`（`scripts/eval/eval_recovery.py`，**真实多轮**） | `recovery_rate` | `Σrecoveries / Σerrors` | 犯错次数 | **不可** |
| 同上 | `repeat_rate` | `count(corrections_used > 0 and not success) / n` | 样本数 | **可** |
| `experiments/<label>/error_analysis.json`（`scripts/eval/analyze_errors.py`，**目标轮**） | `recovery.recovery_rate` | `count(passed) / n`（= 纠错轮样本的**通过率**） | 样本数 | 可（但它不是恢复率，是成功率） |
| 同上 | `recovery.repeat_first_mistake_rate` | `count(repeated_first_mistake) / n` | 样本数 | 可 |
| 同上 | `recovery.by_mistake_key[k].recovery_rate` | `count(passed) / len(v)` | 样本数 | 可 |
| 同上 | `recovery.by_mistake_key[k].repeat_rate` | `count(repeated_first_mistake) / len(v)` | 样本数 | 可 |
| `eval_report.json` → `metadata.recovery_summary`（`src/eval/grader.py` `recovery_summary()`） | `recovery_rate` | `recovered / total`（= 通过率） | 样本数 | 可 |

**取值建议（本报告的规则）**：

1. §6 的三张表**只用 `recovery_metrics.json`**（真实多轮），因为只有它能测到「第二步依赖第一步输出」的因果链；
2. 如果引用 `error_analysis.json` 的 `recovery.*`，**必须改名引用**为「纠错轮通过率」与「重复首轮错误率」，不得写成 `recovery_rate` / `repeat_rate`；
3. `repeated_first_mistake` 的判据是 `text_similarity(首轮错误文本, 本轮输出) >= 0.75`（代码事实：`src/eval/grader.py` `analyze_recovery()`；阈值注释说明「实证上重复同一句错误回复的相似度通常在 0.9 以上，而正确修正后的回复相似度一般低于 0.5」）。**注意这个 0.75 是代码里的经验常量，不是本项目实测标定值**，报告里引用时要标注这一点。

### 6.5 抽检要求

`dialogues.jsonl`（`DialogueResult.to_dict(include_text=True)`）里含每轮的 `assistant_text` / `call` / `observation` / `problem` / `correction`。**必须人工抽检至少 5 条 `repeat_loop` 结局的对话**，确认：

- 纠错消息是否真的被注入（`step.correction` 非空）；
- 模型是「没听进去」还是「听进去了但换了个错法」；
- `evaluate_step` 的判错是否与人工判断一致（防止框架自己判错导致「恢复失败」的假象）。

---

## §7 根因分析

> # ⚠️ 本节在小模型真实数据到位前只能写「预期假设」，不能写成结论
>
> 这一节的结构、假设清单、以及**每条假设要用哪个字段证伪**都已经写好，方便真实数据一到就直接判定。
> 但在真实 `eval_report.json` / `error_analysis.json` 存在之前，下面的每一条都必须标注为【预期假设】。
> **禁止**把 mock 的假设值（`MockPolicyCfg.baseline_error_rates`）当成观测到的错误分布来支持这些假设。

### 7.1 假设清单与证伪方式

| # | 假设（【预期假设】） | 支持它的代码事实 | 用什么数据证伪 |
|---|---|---|---|
| H1 | **模型规模是主约束**：1.5B 的参数容量不足以同时装下「6 个工具的 schema 记忆」「<tool_call> 格式约束」「多步上下文跟踪」三件事，彼此竞争 | 无限定（属于模型能力先验） | 与 3B 同协议对比（若 3B 的错误结构不同，说明是容量问题；若结构相同，说明是任务/格式设计问题） |
| H2 | **指令跟随能力弱 → 规则条目数越多越容易漏** | `SYSTEM_PROMPT` 含 5 条编号规则 + 工具手册（每个参数都有 desc/pattern/range）；`SYSTEM_PROMPT_LITE` 只有 2 行 | `--lite-system` 变体：若精简 prompt 反而更好，说明完整 prompt 的信息量已超出模型的处理预算（消融计划 §6） |
| H3 | **格式约束复杂度是主要瓶颈**：`<tool_call>` 标签 + 嵌套 JSON（`name`/`arguments` 两层）+ 中文值（`ensure_ascii` 关闭）三者叠加 | 代码事实：`is_canonical_format` 要求「只有一处 `<tool_call>` 且整段就是它」；`classify` 把「标签未闭合」单列为 `unclosed_tag` | `FORMAT_ERROR` 的子类型分布：若 `json_broken` / `unclosed_tag` 占多数 → 格式复杂度主导；若 `promise_only` 占多数 → 意图落地主导 |
| H4 | **参数语义依赖逐条 description，而非全局规则**：同一个「城市」概念在 `search_flights` 要三字码、在 `get_weather` 要中文名 | 代码事实：`tool_catalog` 里两者 description 相反；`task_manifest` 把 `weather_advisory` 单列一类，`note` 写着「注意：这个工具要中文城市名，不是三字码 —— 与其他工具相反」 | `by_category` 的 `weather_advisory` vs `flight_search` 对比；以及 `PARAM_ERROR` 是否集中在 `city` / `origin` / `destination` 三个字段（`error_analysis.json` → `distribution.exemplars` / `classification.detail.fields`） |
| H5 | **相对日期换算是硬伤**：模型会照抄「后天」而不换算成 `YYYY-MM-DD` | 代码事实：`search_flights.date` 的 `pattern: ^\d{4}-\d{2}-\d{2}$`，desc 里写着「相对日期（明天/下周五）必须先换算成绝对日期」；任务集有 8 个 `relative_date` 中难任务；`task_manifest._self_check()` 专门断言 ground truth 里不得出现相对词 | `by_difficulty.medium` 与 `flight_search` 中 `tags=("relative_date",)` 的子集准确率；以及是否出现 `date` 参数 = 中文相对词的证据（`error_analysis.json` → `exemplars`） |
| H6 | **多步上下文中的实体引用失败**：`flight_id` 必须从 `<observation>` 里抄，模型会编造 | 代码事实：`booking_flow` 的 10 个 `from_observation` 任务；`book_flight.flight_id` 的 desc 写着「必须来自 search_flights 的返回结果，不能自己编造」；`looks_like_placeholder` 把 `previous_*` / `my` / `first` 判为占位符 | `booking_flow` 分组准确率；`--no-observations` 变体（**注意当前不生效，见 §9.4**）；`exemplars` 里 `flight_id` 的失败值 |
| H7 | **「不调用」比「调用」更难**：`expected.tool is None` 的 5 个任务要求模型在信息不足时追问而不是硬调 | 代码事实：`task_manifest` 的 cancellation 4 个 `clarify_first` 任务 + 1 个恢复任务正解为 `None`；`classify()` 为此单列 `should_clarify` | 这 5 个任务的准确率（需按 `expected.tool is None` 过滤，不能按类别过滤，因为 cancellation 类里还有 8 个**该调用**的任务） |

### 7.2 每条假设必须写出的三段式

若要把某条假设升级为**结论**（只在真实数据到位后），必须写全三段：

```text
【观测】<具体字段与数值>（来源：experiments/baseline_airline/error_analysis.json → ...）
【推论】<从观测到根因的推理链>
【替代解释与排除】<这个观测还有哪些别的解释；本项目的哪个变体实验排除了它们>
```

示例（**仅在真实数据填好后替换为真实值**）：

```text
【观测】PARAM_ERROR 占失败的 <x>%，其中 <y>% 的失败字段是 city/origin/destination。
【推论】同一实体类别在不同工具里要求相反的写法，而模型没有建立全局规则，只依赖逐条 description。
【替代解释与排除】替代解释是「模型根本不认识机场代码表」——若是这样，flight_search 的 origin
  也会大面积错。数据：flight_search 的 origin 失败率 <a>%，weather_advisory 的 city 失败率 <b>%。
  <a> 与 <b> 的对比排除了（或没有排除）该解释。
```

### 7.3 关于 mock 假设值的正确引用方式

如果要在报告里说明「我们**预期**的错误结构长什么样」，可以引用 `src/config.py` `MockPolicyCfg.baseline_error_rates`，但**必须**同时写明四点：

1. 这是 **mock 策略的输入假设**，不是观测结果；
2. 它的用途是「在没有 GPU 的机器上把评测→分类→偏好构造→训练→再评测→画图这条流水线跑通」，见 `src/config.py` 的模块级警告；
3. 数值本身（本报告**特意不抄录具体数字**，避免被误引用）在 `configs/default.yaml` 的 `mock.baseline_error_rates` 里，可自行查阅；
4. mock 的**结论性产物**（`outputs/mock_pipeline/`、`experiments/mock*`、`outputs/_facts.json`）**一律不得引用为项目结果**。`outputs/_facts.json` 的 `warning` 字段已经写明：「其中的指标来自合成策略的误差率假设，不是 Qwen2.5-1.5B 的真实评测结果，不能作为项目结论引用。」

**报告里唯一允许出现的 mock 相关表述**是下面这种形式：

> mock 流水线（`scripts/train/mock/run_all_mock.py`）用于验证代码链路的自洽性。其数值来自合成误差率假设，与 Qwen2.5-1.5B 的真实表现无关，本报告不引用任何 mock 数值。

---

## §8 诊断结论 → 改进方向

### 8.1 映射表（`CATEGORY_REMEDIES_ZH` 原文 + 本项目对应实验）

下表第一列与第三列是**代码事实**（`src/eval/error_taxonomy.py` `CATEGORY_REMEDIES_ZH` 的原文），第四列来自 `src/data/preference_builder.py` `EXPERIMENT_PAIR_TYPES`（也是代码事实）。中间一列是本项目的偏好对类型常量。

| 观测到的错误类别 | `CATEGORY_REMEDIES_ZH[cat]`（代码原文，即偏好数据构造策略） | 对应 pair_type / 实验切片 | 该实验**预期**改善的指标 | 证伪条件 |
|---|---|---|---|---|
| `FORMAT_ERROR` 格式错误 | 格式约束偏好对：chosen 是合法 tool_call，rejected 是纯文本/坏 JSON | `format_constraint` → `dpo_format`、`dpo_joint` | `format_accuracy`、`FORMAT_ERROR` 占比、accuracy | 格式率上升但 accuracy / pass^4 不动 → 只在教「加标签」 |
| `HALLUCINATION` 工具幻觉 | 工具选择偏好对：chosen 用白名单工具，rejected 用貌似合理的假工具 | `tool_selection` → `dpo_tool`、`dpo_tool_param`、`dpo_all`、`sft_only` | `HALLUCINATION` 占比（尤其 `plausible_fake_tool` 子类型） | 幻觉率不降，或降到「不相干假工具」减少但「命名变体」不减 |
| `WRONG_TOOL` 工具选错 | 工具选择偏好对：同一请求下正确工具 vs 相邻语义工具 | 同上（`negative_kind=hard` 那一档） | `WRONG_TOOL` 占比（`adjacent_tool`） | 相邻工具错误不减，说明模型没学到语义边界 |
| `PARAM_ERROR` 参数错误 | 参数约束偏好对：chosen 是真实实体，rejected 是占位符/错格式/编造值 | `param_constraint` → `dpo_joint_param`、`dpo_tool_param`、`dpo_all`、`sft_only` | `PARAM_ERROR` 占比、§5 的「格式对但值错」率 | 参数错误不减，或只是从「占位符」变成「形状对但值错」（更糟） |
| `REFUSED` 拒绝作答 | 错误恢复偏好对 + 澄清类样本：示范「信息不足时追问」而非放弃 | `error_recovery` → `dpo_recovery`、`dpo_joint` | `REFUSED` 占比、`first_try_rate` | 拒绝率降了但 `WRONG_TOOL/should_clarify` 上升 → 从「不干」变成「乱干」，**更糟** |
| `UNKNOWN` 未知错误 | 人工抽检，补充新类别 | 无（流程性动作） | 下一轮诊断里 `UNKNOWN` 占比下降 | 连续两轮 `UNKNOWN` 占比 > 5% 且未新增子类型 → 分类体系需要重构 |

`REFUSED` 那一行的 `error_recovery` 映射需要单独说明：代码事实是 `PAIR_RECOVERY` 针对的 `target_error` 来自 `recovery_corpus` 的 `error_family`（`FORMAT_ERROR` / `HALLUCINATION` / `WRONG_TOOL` / `PARAM_ERROR` / `NO_RECOVERY`），`preference_builder.CORRECTION_TEMPLATES` 里还有一组独立的 `NO_RECOVERY` 纠错话术。**恢复类偏好对的真正目标不是某一类错误，而是「被纠正后能不能改」这个行为本身**，所以在映射表里它跨类别生效。

### 8.2 因果链：为什么这个映射不是「拍脑袋配对」

诊断与方案之间的因果链必须写清，否则方案看起来像「我们造了一堆数据碰巧有用」：

```text
① 诊断给出「哪一类错误占主要份额」
        ↓（只对占比高、且集中度高的类别投入）
② 该类错误的**发生机制**决定 chosen/rejected 的对比点应该放在哪
   FORMAT_ERROR  → 对比点在「有没有把意图落成调用」→ chosen 合法调用 vs rejected 纯文本
   HALLUCINATION → 对比点在「工具名是否逐字匹配白名单」→ chosen 真工具 vs rejected 命名变体
   WRONG_TOOL    → 对比点在「语义边界」→ chosen 正确工具 vs rejected 相邻工具（都用合法格式）
   PARAM_ERROR   → 对比点在「实体是否来自上下文」→ chosen 真实实体 vs rejected 占位符/同形错值
        ↓
③ 对比点的**难度分层**决定数据是否有效
   全用简单负样本（纯文本承诺）→ 模型 3 步就能分开，梯度贡献趋近 0
   全用硬负样本（同形错值）    → 两侧差异过小，训练不稳
   → 所以 preference_builder 用 hard_negative_ratio（配置值 0.35）做配比
        ↓
④ 每类数据都要有**质量门禁**，否则 DPO 会去拟合噪声
   - chosen 必须 canonical 格式
   - chosen 与 rejected 语义必须真的不同（equivalent 判定）
   - chosen 的参数必须过 validate_arguments
   - 恢复类还要过「纠错话术与首轮错误是否一致」的门禁
        ↓
⑤ 实验切片按 pair_type 组合，形成可证伪的消融矩阵（见 ablation_plan.md §1）
```

**第 ④ 步在本仓库有真实的失败记录可引用**（代码事实，写在源码注释里）：

- `preference_builder._recovery_coherence()` 的注释记录了早期版本「从整个纠错话术池里随机抽句子」，生成了「航班搜索场景 + 『city 参数要填中文城市名』」这种前后矛盾的数据。结论：「这种数据比没有数据更糟 —— 它教模型『用户的纠正可以忽略』。」
- `tests/test_methodology.py` 的 `test_recovery_pairs_have_coherent_corrections` 就是为这个 bug 加的守护测试。
- `preference_builder.PreferencePair.key` 的注释记录了另一个 bug：去重键不含 `rejected` 时会「丢掉 97% 的候选」（注释原文），因为同一任务的不同错误类型负样本会互相顶掉。

报告里引用这些能让「数据构造」这一节显得可信：**我们不是一次就写对了，而是有具体的失败与修补记录。**

### 8.3 诊断结论的收尾模板

```text
基线诊断结论（Week 1）：
1. 整体：accuracy <A>、pass@4 <P4>、pass^4 <PH4>，Δstability <D>。
   主要瓶颈是 <会做但不稳 / 某类能力缺失>（依据：§2 表 1、§4）。
2. 错误结构：占失败份额前两位是 <类别 X>（<x>%）与 <类别 Y>（<y>%）。
   其中 <子类型 s> 占 <X> 的 <p>%，说明机制是 <...>（依据：§3 表 1/表 2）。
3. 分组：差距主要出现在 <难度/类别/可见性>；unseen 与 seen 差 <Δ>，
   <落在噪声内 / 显著为负>（依据：§4）。
4. 退化风险：格式正确率 <f>%，宽口径「格式对但没做对」<w>%，
   <已存在 / 尚未出现> 格式捷径（依据：§5）。
5. 恢复能力：repeat_rate <r>%、first_try_rate <ft>%、recovery_rate <rr>%
   （附带 total_errors=<e>、total_recoveries=<g>），
   最差的三个错误模式是 <k1/k2/k3>（依据：§6）。
6. → 因此 Week 2 的偏好数据以 <pair_type 列表> 为主，以 <pair_type> 为对照；
   消融实验的最小完备集是 baseline / sft_only / dpo_format / dpo_recovery / dpo_joint
   （依据：§8；实验细节见 docs/ablation/ablation_plan.md）。
```

---

## §9 失败案例库

### 9.1 用途

`experiments/<label>/failure_cases/` 是**人工检查的入口**，回答三个问题：

1. **分类器判对了吗？** 自动分类可能把「其实答对了但写法怪」判成错（假阳性）。人工看几条就能发现系统性问题。
2. **失败长什么样？** 报告里写「格式错误」是抽象的；一条 `promise_only` 的原始输出是具体的。面试与博客里要有具体例子。
3. **能提炼出新的偏好数据吗？** 真实失败样本是 `rejected` 的最佳来源（比合成负样本更真实）。

### 9.2 格式（代码事实）

由 `scripts/eval/run_baseline.py` 第 289-319 行生成：对每条**未通过**的采样写一个文件，**最多 40 个**（`if idx >= 40: break`）。

文件名：`case_<三位序号>_<task_id>_<错误类别>.json`

单文件字段（**逐字来自代码**）：

| 字段 | 类型 | 内容 |
|---|---|---|
| `case_id` | str | `case_000` 起递增，**跨 task 连续编号**（不是每任务重新计数） |
| `task_id` | str | 如 `task_0023` |
| `category` | str | **任务类别**（`flight_search` / `baggage` / …），注意与下面的 `classification.category` 不是一回事 |
| `difficulty` | str | `easy` / `medium` / `hard` |
| `visibility` | str | `seen` / `unseen` |
| `is_mock` | bool | **必须为 `false`** 才可作为证据 |
| `expected` | obj | `{"tool": <期望工具名或 null>, "args": <graded_args，即宽松口径比对的字段>}` |
| `classification` | obj | `Classification.to_dict()`：`category` / `category_zh` / `detail{category, category_zh, subtype, subtype_zh, message, fields, evidence}` / `recovered_from` / `format_ok_value_wrong` |
| `defects` | list[str] | 格式缺陷清单（`format_defects()` 输出，如 `missing_tool_call_tag`、`single_quoted_keys`、`trailing_comma`、`markdown_fence_instead_of_tag`、`unquoted_chinese_keys`） |
| `raw_output` | str | 模型原始输出，**截断到 2000 字符** |
| `finish_reason` | str | `stop` / `length` / `error` |

> **两个容易踩的点**：
> 1. `category` 是**任务类别**，`classification.category` 是**错误类别**。写报告引用时务必写全路径，别写「category 是 PARAM_ERROR」这种含混表述。
> 2. `expected.args` 是 `task.expected.graded_args`（= `args_subset or args`），**不是** `expected.args` 全字段。也就是说案例文件里看不到严格口径的完整期望参数，需要严格口径证据时要回查 `experiments/tasks_airline_v1.json` 或 `src/data/task_manifest.py`。
> 3. 40 个上限是**先到先得**（按 `predictions.jsonl` 顺序，也就是 `task_id` 升序 × `sample_index` 升序）。因此**前 10 个任务的失败案例可能占满全部配额**，后期任务的失败模式可能完全没被采样到。若需要覆盖全部类别，必须自己按 `classification.category` 分层重抽（改造 `analyze_errors.py` 或另写脚本）。

### 9.3 结构示例（内容全部为占位符）

> 下面是**结构示例**，用于说明文件长什么样。所有具体文本/任务号/证据都是 `<占位>`，**不得**当作真实案例引用。

**示例 1：格式错误 — 承诺式回复（`FORMAT_ERROR` / `promise_only`）**

```json
{
 "case_id": "case_<序号>",
 "task_id": "task_<四位编号>",
 "category": "<任务类别，如 flight_search>",
 "difficulty": "<easy|medium|hard>",
 "visibility": "<seen|unseen>",
 "is_mock": false,
 "expected": {
  "tool": "<期望工具名，如 search_flights>",
  "args": {"<参数名>": "<期望值>", "...": "..."}
 },
 "classification": {
  "category": "FORMAT_ERROR",
  "category_zh": "格式错误",
  "detail": {
   "category": "FORMAT_ERROR",
   "category_zh": "格式错误",
   "subtype": "promise_only",
   "subtype_zh": "只承诺要查，没真的调用",
   "message": "只承诺要查询，没有真的输出 tool_call",
   "fields": [],
   "evidence": "<模型原始输出片段>"
  },
  "recovered_from": null,
  "format_ok_value_wrong": false
 },
 "defects": ["missing_tool_call_tag"],
 "raw_output": "<模型原始输出，最多 2000 字符>",
 "finish_reason": "stop"
}
```

**读法**：`subtype=promise_only` 说明模型**意图正确**（它确实理解要查航班），失败点只在「没把意图落成调用」。这类样本是 `format_constraint` 偏好对最直接的 rejected 来源。

**示例 2：工具幻觉 — 命名变体（`HALLUCINATION` / `plausible_fake_tool`）**

```json
{
 "case_id": "case_<序号>",
 "task_id": "task_<四位编号>",
 "category": "<任务类别>",
 "difficulty": "<难度>",
 "visibility": "<seen|unseen>",
 "is_mock": false,
 "expected": {"tool": "search_flights", "args": {"<参数名>": "<期望值>"}},
 "classification": {
  "category": "HALLUCINATION",
  "category_zh": "工具幻觉",
  "detail": {
   "category": "HALLUCINATION",
   "category_zh": "工具幻觉",
   "subtype": "plausible_fake_tool",
   "subtype_zh": "工具名是「貌似合理但不存在」的名字",
   "message": "调用了貌似合理但不存在于工具列表的工具 '<假工具名，如 find_flights>'",
   "fields": ["tool"],
   "evidence": "<模型原始输出片段>"
  },
  "recovered_from": null,
  "format_ok_value_wrong": false
 },
 "defects": [],
 "raw_output": "<模型原始输出>",
 "finish_reason": "stop"
}
```

**读法**：`subtype=plausible_fake_tool` 说明模型**格式完全合法、参数也抄对了**，只是把 `search_flights` 写成了它的同义变体（`FAKE_TOOL_VARIANTS` 里的候选）。这类负样本比「随机假工具」有诊断价值得多，因为「`send_confirmation_email` 配 `origin/destination/date`」一眼就是假的，模型不需要理解语义就能排除（`tool_catalog.py` 的注释原文）。

**示例 3：参数错误 — 格式对但值错（`PARAM_ERROR` / `wrong_value`）**

```json
{
 "case_id": "case_<序号>",
 "task_id": "task_<四位编号>",
 "category": "<任务类别>",
 "difficulty": "<难度>",
 "visibility": "<seen|unseen>",
 "is_mock": false,
 "expected": {"tool": "<期望工具>", "args": {"<字段>": "<期望值>"}},
 "classification": {
  "category": "PARAM_ERROR",
  "category_zh": "参数错误",
  "detail": {
   "category": "PARAM_ERROR",
   "category_zh": "参数错误",
   "subtype": "wrong_value",
   "subtype_zh": "参数值不对",
   "message": "参数值不对：<字段> 期望 '<期望值>' 实际 '<实际值>'",
   "fields": ["<字段名>"],
   "evidence": "<模型原始输出片段>"
  },
  "recovered_from": null,
  "format_ok_value_wrong": true
 },
 "defects": [],
 "raw_output": "<模型原始输出>",
 "finish_reason": "stop"
}
```

**读法**：`format_ok_value_wrong=true` 就是 §5 要监控的那个信号 —— 结构完美、实体错。**注意**：这里为 `true` 要求所有失败字段都通过 `_same_shape()`（例如订单号错一个字符、数值差几公斤）。像「`date` 填了『后天』」这种不会被置为 `true`（见 §5.4），因此**典型案例库里可能有相当比例的退化样本没有被标记**，抽检时要按 `subtype=wrong_value` + 人工看 `evidence` 补判。

### 9.4 ⚠️ 抽检清单与「文档/代码不一致」记录

抽检失败案例时请按这份清单核对（每一条都是一次「自动判分是否可信」的检查）：

| # | 检查项 | 期望 | 发现异常时说明什么 |
|---|---|---|---|
| 1 | `is_mock` 是否全为 `false` | 必须 | 若出现 `true`，说明该目录混入了 mock 结果，**整份诊断不可用** |
| 2 | `classification.detail.subtype` 是否有中文标签 | 大部分有 | `should_clarify` 会显示英文（`SUBTYPE_LABELS_ZH` 缺该键），需手工补中文 |
| 3 | `defects` 与 `subtype` 是否一致 | `missing_tool_call_tag` ↔ `promise_only`/`no_tool_call` | 不一致说明解析器与格式缺陷检测口径漂移 |
| 4 | `format_ok` 为 `true` 但 `subtype=json_broken` | 会出现（见 §5.3） | 这是 `is_canonical_format` 用正则而非 JSON 解析的后果，需在报告里注明 |
| 5 | `finish_reason == "length"` 的比例 | 应较低 | 若偏高，说明 `max_new_tokens=1536` 不够，评测协议可能被截断污染 → 需要提高上限并**重跑全部实验**（协议变更！） |
| 6 | `raw_output` 是否被截断到 2000 字符导致证据不完整 | 极端长输出会 | 需要完整输出时回查 `predictions.jsonl` |

---

## 附录 A：本仓库「文档 / 代码不一致」清单（写报告与改代码时都要注意）

以下每一条都已定位到具体文件与行，供后续修正。

| # | 不一致 | 位置 | 影响 | 建议 |
|---|---|---|---|---|
| A1 | 规划文档写「30 个任务」，代码是 **100** 个 | `项目方案.md` §1.2 / §五 / §6.1；`src/data/task_manifest.py` | 报告里引用任务集会写错规模 | 以代码为准；规划文档需过期标注 |
| A2 | 规划文档写难度 easy 10 / medium 8 / hard 12，实际 **43 / 32 / 25** | `项目方案.md` §5.2「tasks.py」；实测 `summary()` | 同上 | 同上 |
| A3 | 规划文档写偏好对「格式 500 + 恢复 500 = 1000 条」，配置是 **1200 / 1200 / 600**（第三类工具选择） | `项目方案.md` §四 Week 2 / §6.1；`src/config.py` `PreferenceCfg` | 数据规模表述不一致 | 以 `CFG.preference` 与 `build_report.json` 为准 |
| A4 | 规划文档的四类偏好对只提三类（格式/恢复/工具选择），代码有 **四类**（多 `param_constraint`） | `项目方案.md` §5.2；`src/data/preference_builder.py` `PAIR_TYPES` | 漏掉了针对 `PARAM_ERROR` 的一整类 | 报告必须写四类（`preference_builder` 模块文档也是「三类/四类」并存，需统一为四类） |
| A5 | 规划文档 §3.3 工具表的「必填参数」与代码一致，但 `search_flights` 还有可选 `max_price`、`book_flight` 还有可选 `cabin` | `项目方案.md` §3.3；`src/env/tool_catalog.py` | 无实质影响 | 补全即可 |
| A6 | `configs/eval/eval_baseline.yaml` 的 `eval.k_values` **从不被读取**（脚本读 `CFG.eval.pass_k_values`） | `scripts/eval/run_baseline.py` 第 196 行；`configs/eval/eval_baseline.yaml` 第 25 行 | **改 yaml 的 k 值不会生效**，可能误以为已经限制到某个 k | 报告里的 k 值必须从 `eval_report.json.metrics` 的键名反查；或在脚本里补上读取 |
| A7 | `configs/eval/eval_baseline.yaml` 的 `variants.*`（`system_prompt_lite` / `tool_names` / `include_observations`）**从不被读取** | 同上，脚本只认 CLI 开关 | 以为改了 yaml 就换成了精简 prompt，实际没有 | 混淆因素实验**必须显式写 CLI**（`--lite-system` / `--tools`） |
| A8 | `configs/eval/eval_baseline.yaml` 的 `output.dir`（`experiments/baseline_airline`）**从不被读取** | 同上（`out_dir` 只由 `--out-dir` / `CFG.path("experiments", label)` 决定） | 不加 `--label baseline_airline` 就会写到 `experiments/baseline/`，而 `analyze_errors.py` 生成默认图名的条件是 `label.startswith("baseline")` —— 名字对不上会导致 README 引用的图不存在 | 跑基线固定用 `--label baseline_airline` |
| A9 | `--no-observations` **实际不生效**：`include_observations` 只写进 metadata，`HFPolicy.predict()` 未把它传给 `build_messages()` | `scripts/eval/run_baseline.py` 第 155 行；`src/models/hf_policy.py` 第 242-246 行；`src/models/prompt_builder.py` 第 113 行 | 「是否依赖 observation」这个混淆因素**无法用当前代码排除** | 必须改造代码（给 `HFPolicy.predict` 加参数并在 `run_evaluation` 侧透传），或在报告里声明该消融未完成 |
| A10 | `classify()` 会产生子类型 `should_clarify`，但 `SUBTYPE_LABELS_ZH` 无此键 | `src/eval/error_taxonomy.py` 第 330 行 vs 第 90-112 行 | 报告里出现英文子类型 | 补一个中文标签 |
| A11 | `SUBTYPE_LABELS_ZH` 里的 `wrong_format` / `out_of_range` / `extra_param` 等子类型，`classify()` 目前**不产出**（参数错误统一走 `wrong_value`，因为 `grade()` 传的 `arg_match` 已经把 schema 校验吞掉了） | `src/eval/error_taxonomy.py` `classify()` 第 393-440 行 | 子类型下钻表里这几行会是 0，容易被误读为「模型不犯这类错」 | 在报告里标注「本版本分类器不产出该子类型」；或让 `grade()` 把 `validate_arguments` 的结果传进 `classify()` |
| A12 | `format_ok_value_wrong` **只在** `PARAM_ERROR/wrong_value` 分支计算，且要求所有失败字段 `_same_shape` | `src/eval/error_taxonomy.py` 第 413-427 行 | 系统性漏报（`date="后天"`、占位符、缺字段、城市名/三字码互换都不计入） | §5.4 已给出宽口径替代统计，报告必须两个口径都给 |
| A13 | `NO_RECOVERY` 是 mock 的错误类型键与偏好对的 `target_error`，但**不在** `ALL_CATEGORIES` / `FAILURE_CATEGORIES` 里 | `src/config.py` `MockPolicyCfg.baseline_error_rates`；`src/mock/mock_policy.py` 第 98 行；`src/eval/error_taxonomy.py` 第 55-66 行 | 真实评测**永远不会**出现 `NO_RECOVERY` 这个类别；「不恢复」只能通过 §6 的 `repeat_rate` 度量 | 报告里不要写「NO_RECOVERY 占比 x%」 |
| A14 | `MockPolicy` 只支持 5 个实验（`mitigation_factors` 的键），而 `EXPERIMENT_PAIR_TYPES` 有 8 个切片 | `src/mock/mock_policy.py` 第 186-189 行；`src/data/preference_builder.py` 第 889-900 行 | `dpo_tool` / `dpo_all` / `dpo_joint_param` / `dpo_tool_param` **无法用 mock 验证**（`MockPolicy("dpo_tool")` 直接抛 `ValueError`） | 扩展切片必须走真实 GPU；mock 自检只覆盖 5 个主实验 |
| A15 | `experiments/mock_baseline/eval_report.json` 文件名**不带 `.mock`**，但内容是 mock | `experiments/mock_baseline/`（`is_mock: true`、`num_tasks: 30`） | `collect_results.py` 默认只读 `eval_report.json`，会把这份 mock 当成 `baseline` 纳入对比表；若其余实验是真实结果，`is_mock` 全真判断失败 → 只打印「混有 MOCK 与真实结果」告警而**不会**排除它 | 建议删除或改名该文件（改为 `eval_report.mock.json` / 移到 `experiments/mock/`）；在 `collect_results.py` 里增加「按 `metadata.is_mock` 过滤」而不是只看文件名 |
| A16 | `configs/train/dpo_joint.yaml` 等配置的 `pairs.train_file` 指向 `outputs/mock_pipeline/preferences_*.mock.jsonl` | `configs/train/*.yaml`；`scripts/train/dpo_train.py` `resolve_data_file()` 会**优先使用配置里声明的路径** | 真实训练若直接 `--config configs/train/dpo_joint.yaml`，会**误用 mock 偏好数据**（mock 流水线另存的一份），而不是 `data/preferences/` 下的真实构造结果 | 真实训练前先跑 `scripts/prepare/build_preferences.py` 生成 `data/preferences/*.jsonl`，并用 `--data-file` 显式指定；或把配置默认指向改掉 |
| A17 | `preference_builder` 模块文档写「三类偏好对的设计依据」，但实际实现了四类 | `src/data/preference_builder.py` 第 3 行 vs `PAIR_TYPES` | 文档描述与代码不符 | 统一为四类 |
| A18 | `analyze_errors.py` 与 `eval_recovery.py` 产出**同名不同义**的 `recovery_rate` / `repeat_rate` | `scripts/eval/analyze_errors.py` 第 114-137 行；`scripts/eval/eval_recovery.py` 第 203-247 行 | 极易引用错文件导致结论错误 | 见 §6.4 的取值规则；长期应改名（如 `correction_turn_pass_rate`） |
| A19 | 代码注释引用的文档中，`docs/method/preference_data_design.md`、`docs/ablation/ablation_plan.md`、`docs/diagnosis/baseline_diagnosis.md` 均已存在；但 `src/training/dpo_dataset.py` 第 31 行引用的 **`docs/ops/prompt_contract.md` 不存在**（`docs/ops/` 下只有 `trl_version_notes.md`） | `src/training/dpo_dataset.py` 第 31 行；`docs/ops/` | 读者按注释去翻会找不到 | 补一份 `docs/ops/prompt_contract.md`（把 `prompt_builder` 的「训练/推理 prompt 必须逐字一致」契约写下来），或改注释 |
| A20 | `docs/figures/README.md` 的「命名约定」只描述了不带 label 的默认名与 `_mock_` 中缀，**没有说明 `analyze_errors.py` 实际写出的文件名带 `_<label>` 后缀**；它给出的命令是 `--label baseline`，而本报告与消融计划用的 label 是 `baseline_airline` | `docs/figures/README.md` 第 26-39 行；`scripts/eval/analyze_errors.py` 第 171-206 行 | 按 README 的命令跑不出 README 表里列的文件名（会得到 `error_distribution_baseline_airline.png`），README 引用的图可能不存在 | 二者取一：① 基线统一用 `--label baseline`（则 `experiments/baseline/` 而非 `baseline_airline/`）；② 或把 README 的文件名表改成带 `_<label>` 的形式 |
| A21 | `docs/figures/` 里现有的 5 个 `.svg` + 2 个 `.mermaid.md`（`error_distribution` / `success_by_difficulty` / `success_by_visibility` / `ablation_comparison` / `training_curves`）**不是** `src/utils/plots.py` 的产物（该模块只出 PNG，且 matplotlib 缺失时返回 `None`） | `docs/figures/*.svg` / `*.mermaid.md`；`src/utils/plots.py` | 存在两套互不相干的图表链路，报告若混引会让人分不清哪些是代码重跑得到的、哪些是手工/示意产物 | 在报告里明确标注每张图的生成方式；示意性 SVG 必须标注「示意图，非真实数据生成」 |

## 附录 B：填表前的最小操作序列

```bash
# 0) 确保目录与任务集自检通过
python -m src.data.task_manifest                 # 应无 AssertionError，并打印类别分布
python tests/run_tests.py                        # 应全通过

# 1) 真实基线评测（GPU；约 100 任务 × 4 采样）
python scripts/eval/run_baseline.py \
    --label baseline_airline \
    --model Qwen/Qwen2.5-1.5B-Instruct
# 产物：experiments/baseline_airline/{eval_report.json,predictions.jsonl,error_analysis.json,failure_cases/}

# 2) 真实多轮恢复评测（GPU）
python scripts/eval/eval_recovery.py \
    --label baseline_airline --repeat 4 --max-turns 6 --max-corrections 2
# 产物：experiments/baseline_airline/{recovery_metrics.json,dialogues.jsonl}

# 3) 错误分析与出图
python scripts/eval/analyze_errors.py --label baseline_airline
# 产物：experiments/baseline_airline/error_analysis.json
#       docs/figures/error_distribution_baseline_airline.png        ← 始终生成（<图名>_<label>.png）
#       docs/figures/error_distribution.png                         ← 额外生成（label 以 "baseline" 开头 且 非 mock）
#       docs/figures/success_by_difficulty_baseline_airline.png  +  success_by_difficulty.png
#       docs/figures/success_by_visibility_baseline_airline.png  +  success_by_visibility.png
#   注意：带 label 与不带 label 是两份内容相同的图；README 引用的是不带 label 的那份。
#   详见 docs/figures/README.md（其「命名约定」小节未覆盖 _<label> 后缀这一行为，见附录 A20）。

# 4) 回填本文件的全部 TODO，然后核对 §1.8 的清单是否 13 项都已填
```

**回填后必须做的四项自检**：

1. 表 1 的三个 CI 是否都包含点估计？（否则 bootstrap 实现有问题）
2. `pass^4 ≤ pass@4 ≤ 1` 是否成立？（`tests/test_methodology.py` 有这条不等式的守护测试）
3. §3 表 1 的「占总样本」列之和是否 = 100%？（分类互斥性的检验）
4. `error_analysis.json` 的 `format_accuracy` 是否 ≥（1 − `FORMAT_ERROR` 占比）？（§5.3 的性质，不满足说明有解析器层面的问题）
