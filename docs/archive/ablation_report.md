# 消融实验结果报告

> ⚠️⚠️ **本报告的数字存在数据污染，结论需重新验证** ⚠️⚠️
>
> **已发现的污染**：训练偏好数据里混入了评测任务。实测
> `sft_only` 的 2509 条训练样本中，**1292 条（51.5%）的 prompt 与评测
> prompt 逐字相同**，且 `chosen` 就是该任务的正确答案。
>
> **根因**：`build_format_pairs` / `build_param_pairs` /
> `build_tool_selection_pairs` 都直接吃 `pool = 评测任务 + 合成任务`，
> 而 `build_recovery_pairs` 的注释却写着「评测与训练因此完全分离」——
> **只有它做到了**。注释与代码不一致，是这类错误长期潜伏的原因。
>
> **影响**：`sft_only` 的 95.0% 准确率**不能解释为泛化能力** ——
> 它是在测试集上做梯度下降的结果。表中所有涉及 `sft_only` /
> `dpo_joint` / `dpo_from_sft` 的数字都受同样影响（它们都用同一份
> 受污染的数据训练）。
>
> **修复状态**：✅ 已修复（硬门禁 + 合成语料独立 + 零撞车验证），
> 但**尚未重训**。修复后的正确数字见后续版本。

> **文档定位**：本文件是真实 GPU 实验的**执行结果**。
> 每个数字都可追溯到 `experiments/<exp>/eval_report.json`，并可由
> `scripts/eval/run_baseline.py` + `scripts/eval/make_ablation_table.py` 复现。
>
> **状态**：⚠️ 4 组实验已跑，但**因上述数据污染需重跑**。
>
> ⚠️ **与原计划的差异**：原计划含 `dpo_format` / `dpo_recovery` 两组，
> 但**真实基线数据推翻了两条关键假设**（见 §5），因此改为实测支持的组合。
> 详见 §0.2「计划与实测的偏离」。

---

## §0 数据来源与方法

### 0.1 数据来源映射

| 报告章节 | 数据来源 | 关键字段 |
|---|---|---|
| §1 主结果 | `experiments/<exp>/eval_report.json` | `headline.*` |
| §2 错误分布 | `experiments/<exp>/error_analysis.json` | `distribution.counts`, `distribution.failure_subtype_counts` |
| §3 分组下钻 | `experiments/<exp>/eval_report.json` | `by_difficulty`, `by_visibility`, `by_category` |
| §4 恢复能力 | `experiments/<exp>/eval_report.json` | `metadata.recovery_analysis.*` |
| §5 训练过程 | `outputs/<exp>/train_log.jsonl`, `train_health.txt` | `loss`, `rewards/margins`, `logps/chosen` |
| 汇总 | 由 `scripts/eval/make_ablation_table.py` 生成 |

> ⚠️ **恢复分析的数据源必须是 `eval_report.json` 的 metadata，不能是
> `error_analysis.json` 的 `recovery` 段。** 后者是后加功能，
> `baseline`/`sft_only` 里根本没有，且字段名与其他实验不同。
> 同一份报告混用两个 schema 会产生静默错误（开发期已踩过）。

> ⚠️ `eval_report.mock.json` 不是数据来源。任何 mock 数字都不得进入本报告。

### 0.2 计划与实测的偏离（重要）

原实验矩阵基于 **mock 假设值**设计，真实基线数据推翻了两条核心假设：

| 原假设 | 实测结果 | 后果 |
|---|---|---|
| 工具幻觉率约 25% | **0.0%**（400 样本中 0 条编造工具名） | `tool_selection` 偏好对（543 条）预期收益接近零 → **该组实验取消** |
| 格式错误以 JSON 语法错为主 | **100% 是「不调工具」**（109 条中 0 条 JSON 语法错） | `json_broken` 方向作废 |
| — | **新发现**：`action_hallucination`（声称已完成操作）41 条 | 原分类体系完全未覆盖 → 新增该子类型与对应偏好对 |

因此实际执行的实验为：

| 实验 | 训练数据 | 设计意图 |
|---|---|---|
| `baseline_airline` | 无 | 起点 |
| `sft_only` | 格式 + 工具选择 + 参数（纯正例） | **对照组**：验证提升是否只来自同分布监督 |
| `dpo_joint` | 格式 + 恢复 + 追问边界，**从基座开始** | 对比信号的价值 |
| `dpo_from_sft` | 同数据，**从 SFT 续训** | 能否保住 SFT 能力同时获得对比收益 |

---

## §1 主结果

### 1.1 实验配置

| 实验 | 训练数据 | 条数 | β | lr | epochs | LoRA r/α | max_length |
|---|---|---|---|---|---|---|---|
| baseline | 无 | — | — | — | — | — | — |
| sft_only | 格式+工具选择+参数 | 2435 | — | 1e-4 | 2 | 16/32 | 1280 |
| dpo_joint | 格式+恢复+追问边界 | 1994 | 0.1 | 5e-5 | 2 | 16/32 | 1280 |
| dpo_from_sft | 同上 | 1994 | 0.1 | 5e-5 | 1 | 16/32 | 1280 |

> `dpo_from_sft` 用 `--init-adapter outputs/sft_only/final`：
> SFT 权重先 `merge_and_unload()` 烘进基座，再挂**全新** LoRA 做 DPO。
> 因此 `π_ref` = merge 后的 SFT 模型，起点满足 `π_θ = π_ref`。

所有实验共用同一评测协议：100 任务 × 4 采样，`primary_k=4`，贪心/温度 0.7。

### 1.2 总体指标

| 实验 | 单次准确率 | pass@4 | pass^4 | Δstability |
|---|---|---|---|---|
| baseline（原始基座） | 50.2% | 57.0% | 45.0% | −12.0% |
| **sft_only（纯正例监督）** | **95.0%** | **95.0%** | **95.0%** | 0.0% |
| dpo_joint（从基座） | 68.5% | 71.0% | 67.0% | −4.0% |
| dpo_from_sft（β=0.1） | 10.0% | 13.0% | 6.0% | −7.0% |

**baseline 的 95% CI**：`accuracy [40.5%, 59.8%]`、`pass^4 [35%, 55%]`。

### 1.3 统计判据

**任务级 bootstrap**（1000 次重采样**任务**，不是重采样样本）——
同一任务的多次采样高度相关，按样本重采样会严重低估区间宽度。

| 对比 | 差值 | 判定 |
|---|---|---|
| baseline → sft_only | **+44.8pp** | ✅ **CI 不重叠，远超检测门槛** |
| baseline → dpo_joint | +18.3pp | ✅ CI 不重叠 |
| sft_only → dpo_joint | **−26.5pp** | ✅ SFT 显著更好 |
| sft_only → dpo_from_sft | **−85.0pp** | ✅ SFT 显著更好 |

**样本量有效性（关键的方法学边界）**：

```
实测 CI 宽度 ≈ ±10pp  →  100 任务可检出的最小效应 ≈ 14pp
而 sft_only 已达 95.0%  →  天花板仅剩 5.0pp

14pp > 5pp
```

> **结论：在这 100 任务规模上，任何 DPO 配置的总体准确率提升都
> 不可能获得统计支持。** 这不是实验失败，而是任务集规模决定的上限。
> 因此本报告的主证据放在**错误类型的结构性变化**（§4），
> 那是相对敏感的指标（20 条基数上的降幅，不受 CI 宽度压制）。

### 1.4 结论

1. **SFT 带来 44.8pp 的显著提升**（50.2% → 95.0%，CI 不重叠），是本项目的主结果。
2. **两组 DPO 都未能超越 SFT**：从基座做是 68.5%（少了 SFT 教会的参数映射能力），从 SFT 续训是 10.0%（β=0.1 过优化崩塌，见 §5）。
3. **DPO 的价值不在总体准确率**，而在特定错误类型 —— 见 §4 的 `missing_reason_corrected`：**SFT 是唯一失败的模型（0/4，且 4/4 重复同一错误）**，而 DPO 是 4/4 全对。
4. **SFT 在 `unseen` 上反而更好**（97.0% vs seen 94.0%）—— 泛化零衰减，说明它学的是通用格式与参数映射，而非记住任务。

---

## §2 错误分布的变化

| 实验 | SUCCESS | FORMAT_ERROR | PARAM_ERROR | WRONG_TOOL |
|---|---|---|---|---|
| baseline | 201 (50.2%) | 109 (27.3%) | 89 (22.2%) | 1 (0.2%) |
| **sft_only** | 380 (95.0%) | **0** | **0** | 20 (5.0%) |
| dpo_joint（基座） | 274 (68.5%) | 34 (8.5%) | 81 (20.2%) | 11 (2.8%) |
| dpo_from_sft | 40 (10.0%) | 360 (90.0%) | 0 | 0 |

**SFT 把 `FORMAT_ERROR` 与 `PARAM_ERROR` 完全清零**，但**新引入 20 条 `WRONG_TOOL`**。
这是整个项目的关键张力：**它修好了两类问题，却引入了一类新的**（该追问时硬调工具）。

### 2.1 失败子类型下钻

| 子类型 | baseline | sft_only | dpo_joint（基座） | dpo_from_sft |
|---|---|---|---|---|
| `wrong_value` | 89 | **0** | 69 | 0 |
| `no_tool_call` | 9 | 0 | 19 | **161** |
| `action_hallucination` | 41 | **0** | 1 | **55** |
| `promise_only` | 31 | **0** | 0 | 3 |
| `should_clarify` | 0 | **20** | 0 | 0 |
| `missing_required` | 0 | 0 | 12 | 0 |
| `adjacent_tool` | 1 | 0 | 11 | 0 |
| `empty` | 9 | 0 | 0 | 0 |

**`action_hallucination` 的转移链完整解释了两种 DPO 失败模式**：

```
baseline 41  →  sft_only 0  →  dpo_joint（基座）1  →  dpo_from_sft 55
```

* **SFT 把它清零了**（41 → 0），说明纯正例监督**能**修好「编造完成动作」
* **从 SFT 续训反而退化到 55 条** —— 与 §5 的 `logps/chosen` 崩塌完全对应

---

## §3 分组下钻

### 3.1 按难度

| 组 | baseline | sft_only | dpo_joint | dpo_from_sft |
|---|---|---|---|---|
| easy (n=27) | 98.2% | **100.0%** | 88.9% | 14.8% |
| medium (n=12) | 62.5% | **91.7%** | 77.1% | 16.7% |
| hard (n=61) | 26.6% | **93.4%** | 57.8% | 6.6% |

**难度标签已按基线实测重标**（原标签有误，见 `CATEGORY_DIFFICULTY` 的说明：
原 `flight_search` 被标成 easy，实测仅 26.1%，导致「medium 比 hard 还难」的自相矛盾）。

### 3.2 按可见性

| 组 | baseline | sft_only | dpo_joint | dpo_from_sft |
|---|---|---|---|---|
| seen (n=67) | 52.6% | 94.0% | 65.3% | 9.3% |
| **unseen (n=33)** | 45.5% | **97.0%** | 75.0% | 11.4% |

**SFT 的 unseen 高于 seen（97.0% vs 94.0%）** —— 泛化不但没衰减，反而更好。
这是「学通用能力而非背任务」的直接证据。

---

## §4 错误恢复能力（核心证据）

| 实验 | recovered | recovered_rate | repeat_rate |
|---|---|---|---|
| baseline | 30/48 | 62.5% | 33.3% |
| **sft_only** | 44/48 | **91.7%** | **75.0%** |
| dpo_joint（基座） | 37/48 | 77.1% | 27.1% |
| dpo_from_sft | 8/48 | 16.7% | 0.0% |

**SFT 的 `repeat_rate = 75.0%`** —— 被用户纠正后，**有 3/4 的概率再犯同一个错**。
它 91.7% 的 `recovered_rate` 主要来自「形式上改正了」，而不是真的不犯了。

### 4.1 逐错误模式（最关键的一张表）

格式：`recovered_rate / rpt repeat_rate`

| 错误模式 | baseline | **sft_only** | dpo_joint（基座） | dpo_from_sft |
|---|---|---|---|---|
| **`missing_reason_corrected`** | 100%/0% | **0% / rpt 100%** ❌ | **100% / rpt 0%** ✅ | 100%/0% |
| `repeated_same_error` | 25%/0% | **100%**/0% | 0%/0% | 0%/0% |
| `overbook_corrected` | 25%/0% | 100%/0% | 25%/0% | 0%/0% |
| `wrong_date_corrected` | 0%/0% | 100%/100% | 0%/**25%** | 0%/0% |
| `hallucinated_weather_tool_corrected` | 100%/100% | 100%/100% | **0%**/0% | 0%/0% |
| `wrong_city_param_corrected` | 0%/0% | 100%/100% | 100%/100% | 0%/0% |
| `city_code_corrected` | 0%/0% | 100%/100% | 100%/100% | 0%/0% |
| `format_json_broken_corrected` | 100%/100% | 100%/100% | 100%/100% | 0%/0% |
| `format_ignore_correction` | 100%/0% | 100%/0% | 100%/0% | 0%/0% |
| `hallucinated_tool_corrected` | 100%/0% | 100%/100% | 100%/0% | 100%/0% |
| `param_placeholder_corrected` | 100%/100% | 100%/100% | 100%/0% | 0%/0% |
| `wrong_tool_corrected` | 100%/100% | 100%/100% | 100%/0% | 0%/0% |

### 4.2 核心发现

**`missing_reason_corrected` 这一行是整个项目最强的证据**：

| 模型 | 总体准确率 | 该错误模式的纠错 |
|---|---|---|
| baseline | 50.2% | 4/4（100%） |
| **sft_only** | **95.0%（最好）** | **0/4（0%）+ 4/4 重复同一错误** |
| dpo_joint（基座） | 68.5% | **4/4（100%）** |

场景：用户说「我不想要 BK9A4C6F 这个订单了」但没说原因，`cancel_booking`
需要 `reason`。正确行为是**追问**而不是编造一个原因。

* **SFT（总体最好的模型）100% 会编造 reason 并重复同一错误**
* 两个 DPO 模型都 100% 改对

**为什么 SFT 学不会**：它的训练数据全是**正例**（chosen = 正确调用）。
**纯正例无法表达「此处不该调用」这个否定性约束** ——
数据里只有「该怎么做」，没有「不该怎么做」。

**对比信号能编码这个否定信息**，这正是 DPO 的不可替代之处。

**而这一项不受 CI 宽度限制**（0/4 → 4/4 是能力的从无到有，不是边际准确率提升），
所以即使总体 accuracy 不显著，结论依然成立。

---

## §5 训练过程与失效分析

### 5.1 DPO 过优化（`dpo_from_sft` 崩塌的原因）

`logps/chosen`（模型对**正确输出**的 log 概率）的轨迹：

```
step   5    logps/chosen =  -6.00    rewards/margins = +0.34
step  15                   -2.78                    +4.55
step  25                  -13.74                    +7.80
step  55                  -28.42                   +10.29
step 120                  -25.09                   +12.19   ← 崩塌
```

| 观测 | 含义 |
|---|---|
| `logps/chosen` 从 −6.00 → **−25.09** | 模型对**正确输出**的把握**越来越差** |
| `logps/rejected` 从 −35 → **−177** | rejected 崩得更狠，所以 margin 在涨 |
| `rewards/margins` 终值 **12.19** | DPO 常见目标 ≈ 2，这是**过优化 6 倍** |
| 训练 `loss` 从 4.52 → **0.0208** | ⚠️ **训练指标看起来完美** |
| 评测 `format_accuracy` = **0.0%** | 模型退化成纯对话，完全不输出 `<tool_call>` |

**关键结论：训练指标健康 ≠ 模型没崩。**

`loss` 降到 0.02、`rewards/accuracies` 达到 1.0 —— 这两个指标全都在说「训练很成功」。
但 `logps/chosen` 的**绝对值**在恶化，它才是崩塌的早期信号。

**根因**：`β = 0.1` 来自 DPO 论文，但那是**全参微调**的设置。LoRA 的参数量少得多，
同样的 β 意味着「用更少的参数做更大的分布偏移」→ 更容易过优化。

**修复方向**（~~未验证~~ → **已被 §9 否证**）：`β` 提到 0.5、`lr` 降到 1e-5、`epochs` 减到 1。

> ⚠️ §9 执行了这个修复：β=0.5 后准确率 **49.25%**（β=0.1 是 10.0%），
> **仍然崩塌**。因此「崩塌源于 β 过大」这个归因是**错的** ——
> 见 §9.5 的训练监控失效分析与 §9.4 的真实失效机制（clarify 模式过度泛化）。

### 5.2 训练监控的指标口径陷阱

TRL 的 `DPOTrainer` 输出的键是**复数**形：

```
rewards/accuracies   rewards/margins   rewards/chosen   rewards/rejected
logits/chosen        logits/rejected   logps/chosen     logps/rejected
```

早期监控回调查的是 `rewards/accuracy` / `rewards/margin`（单数）——
**这两个键根本不存在**，于是在 `train_log.jsonl` 里永远是 `nan`。
表面上像「指标没记录」，实际是查错了名字。而 DPO 恰恰靠这两个量判断健康度。

**这些键名是通过 `scripts/train/probe_trainer_loss.py` 实测得到的**
（它直接打印 trainer 返回的 metrics 字典的全部键），不是猜的。

### 5.3 `loss = log 2` 判据的适用边界

`π_θ = π_ref` 时 DPO 损失恰为 `log 2 = 0.6931`。但这个判据**只在初始化那一刻成立**。

`Trainer` 在 `logging_steps`（本项目为 5）之后才开始记录，第一条已是 step 5 ——
模型已更新过。用 log 2 判「起点异常」**必然误报**：

```
⚠️ 初始 loss 偏离 log 2 较多（差 +3.7473）
⚠️ 初始 loss 偏离 log 2 较多（差 +3.8290）
```

这两条告警都是无意义的。**开发期真的据此怀疑过 reference 配置、
中断过一次本来健康的训练，并在错误假设上排查了数轮。**

实测的正确做法：用探针在**任何梯度更新之前**测量（`probe_trainer_loss.py`
实测得 `loss = 0.691406`，与 log 2 相符到 bf16 精度）。

---

## §6 未做的实验与遗留问题

| 项 | 状态 | 原因 |
|---|---|---|
| `dpo_format`（仅格式约束） | ❌ 未跑 | 真实数据显示格式问题已由 SFT 完全解决（FORMAT_ERROR 109→0），单独跑无信息量 |
| `dpo_recovery`（仅恢复对） | ❌ 未跑 | 差异已被 `dpo_joint` vs `dpo_from_sft` 覆盖 |
| `dpo_tool`（工具选择对） | ❌ 取消 | **实测工具幻觉率 0%** —— 假设不成立 |
| β 敏感性分析 | ❌ 未跑 | 总体准确率上限仅 5pp（§1.3），β 差异必然落在噪声里 |
| `dpo_from_sft` 用 β=0.5 重跑 | ❌ 未跑 | 最该补的一组：可验证「修正超参后能否既保住 95% 又获得追问边界改善」 |
| 400 任务规模重跑 | ❌ 未跑 | 可把检测门槛从 14pp 降到约 7pp |

### 6.1 最该补的实验

**`dpo_from_sft` + β=0.5 + lr=1e-5 + 1 epoch**（约 30 分钟 GPU）。

它能回答本项目唯一还没回答的关键问题：

> **修正超参后，能否在保住 SFT 95% 总体能力的同时，
> 获得对比信号带来的追问边界改善？**

三种可能结果与对应结论：

| 结果 | 结论 |
|---|---|
| accuracy ≥95% 且 `missing_reason` 仍 4/4 | ✅ **DPO 价值确立** —— 两全其美 |
| accuracy ≥95% 但追问边界不改善 | DPO 无增量 → 结论记为「SFT 已足够」 |
| accuracy 仍 <95% | 对比信号在此规模下会损害基础能力 → 重要负面结论 |

---

## §7 结论汇总

### 7.1 各实验的实际贡献

| 实验 | 贡献 |
|---|---|
| `baseline_airline` | 建立起点与**统计功效边界**（CI ±10pp、可检出最小效应 14pp） |
| `sft_only` | **主结果 +44.8pp**；证明泛化不衰减；暴露「纯正例学不会否定约束」 |
| `dpo_joint`（基座） | 证明对比信号**能**修好追问边界（`missing_reason` 4/4）；但代价是丢失 SFT 的参数映射能力（33 个任务退化） |
| `dpo_from_sft` | **负面结果**：β=0.1 过优化导致崩塌；定位到 `logps/chosen` 这一早期信号 |

### 7.2 核心论点是否成立

**部分成立，但边界明确。**

✅ **成立的部分**：对比信号能做到纯正例监督做不到的事 ——
`missing_reason_corrected` 上 SFT 0/4（最优模型）而 DPO 4/4。
这是「否定性约束无法由正例学习」的直接证据。

❌ **不成立的部分**：「DPO 能提升总体准确率」。
在 100 任务规模、SFT 已达 95% 的条件下，这在统计上不可能被检出。

### 7.3 负面结果

1. **两组 DPO 都低于 SFT**（68.5% / 10.0% vs 95.0%）
2. **从基座直接做 DPO 会让 33 个 SFT 做对的任务退化**（全是 `flight_search` 的参数映射类）
3. **从 SFT 续训在 β=0.1 下会崩塌**（`format_accuracy` 0%），且**训练指标完全健康**
4. **原方案的两条核心假设被实测推翻**（工具幻觉 25% → 实测 0%；格式错以 JSON 语法为主 → 实测 100% 是「不调工具」）

### 7.4 项目的方法学价值

除了以上结论，本项目在工程上定位并修复了 5 个**会静默污染实验结论**的缺陷：

| 缺陷 | 症状 | 危害 |
|---|---|---|
| 偏好数据用内建 `hash()` 做种子 | 每次运行数据不同（条数相同、内容不同） | 消融实验失去可比性 |
| 错误分类器误标 | `json_broken` 100 条实为「不调工具」 | 误导偏好数据设计方向 |
| prompt 截断未告警 | 23.2% 的 prompt 被截断，报告显示「健康」 | system prompt 的工具定义被切掉 |
| PyYAML vs JSON 类型差异 | 本地正常、云端 `lr` 变成字符串而崩 | 环境相关的偶发失败 |
| 一致性检查器假通过 | 报告「✓ 一致」但实际一条没检查 | 训练/推理分布不一致被漏掉 |

**这类缺陷的共同特点：不报错、数字看起来正常、但结论是错的。**
它们比崩溃更难发现，也是本项目在工程上最有价值的部分。

---

## §8 数据污染事件

### 8.1 污染是什么

训练偏好数据里混入了评测任务。实测：

```
sft_only 训练样本总数              2509
其中 prompt 与评测 prompt 逐字相同  1292   （51.5%）
且 chosen 正是该任务的正确调用
```

**这等于在测试集上做梯度下降。** `sft_only` 的 95.0% 因此不能解释为泛化能力。

### 8.2 根因：注释与代码不一致

```python
# build_recovery_pairs 的文档字符串写着：
#   「不用评测集里的 12 个 recovery 任务…评测与训练因此完全分离。」

# 但实际代码：
pool = tasks + syn_tasks          # ← tasks 就是 100 个评测任务
pairs += self.build_format_pairs(pool, n_format)            # 吃评测任务
pairs += self.build_param_pairs(pool, n_param)              # 吃评测任务
pairs += self.build_tool_selection_pairs(pool, n_tool)      # 吃评测任务
```

**只有 `build_recovery_pairs` 真的做到了分离**（它用 `recovery_corpus`
的场景语料，不用 `pool`）。另外三处没有 —— 而注释让人以为都做到了。

这是这类错误能长期潜伏的根本原因：**读代码的人相信注释，就不再去核对**。

### 8.3 为什么早期的检查没发现

我曾做过一次「污染检查」，报告 `重叠 = 0`。但那个检查是**无效的**：

```python
ids = collections.Counter(r.task_id for r in with_tid)
overlap = sorted(set(ids) & set(tasks))    # → 0
```

它比对的是 `task_id`，而**偏好数据里的 `task_id` 字段是空的** ——
`task_0041` 的偏好对确实带 `task_id="task_0041"`，但我当时的加载路径
没有读到它，于是 `with_tid` 是空的、`overlap` 自然是 0。

**结论：检查「有没有污染」必须比对 prompt 文本，不能比对 ID 字段。**
ID 是元数据，可以缺失；prompt 是实际喂给模型的东西。

### 8.4 修复

| 措施 | 内容 |
|---|---|
| **硬门禁** | `build_all` 收到非 `trainsyn_` 前缀的任务时**直接抛 ValueError** |
| **数据源分离** | `build_preferences.py` / `mock_pipeline.py` 的 pool 只含合成任务 |
| **合成语料补齐** | 新增 `cancel_booking`（原语料完全缺失该工具）与「该追问」任务（否定性约束的唯一载体） |
| **撞车剔除** | 合成语料与评测任务共用同一批订单号/城市名，必然有几处首轮文本相同；在**合成侧**剔除（评测集保持 100 任务） |
| **回归测试** | `test_training_data_never_contains_eval_tasks` 断言：门禁生效、数据零撞车、合成语料覆盖全部评测所需工具 |

修复后验证：

```
sft_only   2118 条   与评测 prompt 逐字相同: 0
dpo_joint  1563 条   与评测 prompt 逐字相同: 0
all        3237 条   与评测 prompt 逐字相同: 0
```

### 8.5 未完成的部分

**修复后尚未重训。** 正确流程是：

1. `python scripts/prepare/build_preferences.py`（已执行，数据已干净）
2. 重训 SFT：`python scripts/train/sft_train.py --experiment sft_only`
3. 重评 SFT：`python scripts/eval/run_baseline.py --adapter outputs/sft_only/final --label sft_only_v2`
4. 重跑 DPO 两组
5. 用新数字更新本报告

**预期**：干净的 SFT 准确率会**低于 95.0%** —— 因为它不再见过测试集。
真实数字未知，这才是项目该报告的泛化能力。

### 8.6 这件事本身的价值

这个污染是**独立发现并修复**的，而且是**在项目声称 95% 之后就主动怀疑**的结果。

对报告而言，它改变了一件事：**「SFT 95%」不能作为结论引用**，
但它带来的另两条结论**不受影响**（因为它们是相对比较）：

* `missing_reason_corrected`：SFT 0/4 vs DPO 4/4 —— 两者都用同样的受污染数据，
  所以「SFT 学不会否定约束」这个**相对**结论依然成立
* agent 多轮评测（89.47% / 94.87% / 故障恢复 78.95%）用的是**另一条数据路径**，
  且 agent 循环不播种预计算 observation —— 但它同样受训练污染影响，需重测

---

## §8.7 逐切片语料覆盖审计（后补，直接影响各实验的解读）

### 8.7.1 为什么整体覆盖检查不够

`assert_training_covers_eval_entities()` 检查的是**合成语料整体**（247 个任务）
的机场码 / 工具 / 技能覆盖，它在所有实验上都是绿的。但真正喂进训练的是
`data/preferences/<exp>.train.jsonl` —— **每个实验用自己的 pair_type 子集
重建池子**，所以单个切片可能整块缺一个工具或一批机场码，
而整体断言看不到。

### 8.7.2 实测（v3）

用 `audit_preference_entity_coverage()` 逐切片审计，结果：

| 切片 | 缺机场码 | 缺工具 | 含观测轮 | chosen 调用数 |
|---|---|---|---|---|
| `all` / `sft_all` | 0 | — | ✓ | 3634 |
| `sft_only` | 0 | — | ✓ | 2635 |
| `sft_clarify` | 0 | — | ✓ | 2635 |
| `dpo_joint` | 0 | — | ✓ | 2013 |
| `dpo_joint_param` | 0 | — | ✓ | 3093 |
| `dpo_format` | 0 | — | ✓ | 1014 |
| `dpo_tool` / `dpo_tool_param` | 0 | — | ✓ | 541 / 1621 |
| **`dpo_recovery`** | **40** | **`cancel_booking`** | ✗ | 999 |
| `dpo_clarify`（设计上无工具调用） | 52 | 全部 | ✓ | 0 |

### 8.7.3 影响与解读约束

**`dpo_recovery` 的切片里 `cancel_booking` 一条都没有。** 根因在纠错语料：
84 个场景的正解工具分布是

```
search_flights 28 · check_baggage 21 · get_weather 14 · modify_baggage 7
book_flight 7 · （澄清，无工具调用）7 · cancel_booking 0
```

`cancel_booking` 的正解数为 **0** —— 与取消相关的场景（`fake_reason_*`）
正解是「**先追问原因**」，转不成工具调用任务。因此这个缺口是**结构性**的，
不是抽样运气。

**解读约束（必须遵守）**：v3 有 **22 个 cancellation 任务**，它们在
`dpo_recovery` 上**必然 0%**。所以

* 不要用 `dpo_recovery` 的**总体准确率**论证恢复能力或整体能力；
* 要看它的效果，请用**与恢复直接相关**的切片（`error_recovery` 命中率、
  `repeat_rate`），或只看不含 cancellation 的类别；
* 若要让它成为公平对照，需要在纠错语料里补一个「取消且带原因」的场景
  （正解为 `cancel_booking`），但这会改变该实验语料的**构成**，
  按 §「加语料会稀释占比」的既有教训（见 `HANDOFF_next_steps.md`：
  补多步语料把追问占比从 11.8% 稀释到 9.6%、`cancellation` 随之
  62.5% → 31.8%），需要成对补、不能只加一类。

该约束已被 `tests/test_overlap.py::test_no_slice_silently_drops_a_whole_tool`
锁住：任何切片静默丢掉一整个工具都会测试失败，已知例外必须写进
`_SLICE_TOOL_EXCEPTIONS` 并附原因。

### 8.7.4 复现

```bash
python -c "
import sys; sys.path.insert(0,'.')
from src.data.recovery_corpus import audit_preference_entity_coverage
for k,v in sorted(audit_preference_entity_coverage().items()):
    print(k, len(v['missing_origin_destination']), v['missing_tools'])
"
```

---

## §9 关键补充实验：修正 β 后 DPO 仍然失败（决定性负面结果）

### 9.1 实验目的

§5.1 把 `dpo_from_sft` 的崩塌归因于 **β=0.1 不适合 LoRA**，
并预期「β 提到 0.5、lr 降到 1e-5、1 epoch」能既保住 SFT 能力、
又获得对比信号带来的追问边界改善。

**本实验就是去验证这个假设。结果是：假设不成立。**

### 9.2 配置

```bash
python scripts/train/dpo_train.py \
    --experiment dpo_joint --init-adapter outputs/sft_v3/final \
    --epochs 1 --beta 0.5 --lr 1e-5 \
    --output-dir outputs/dpo_sft_v3_b05
```

起点是**干净的** `sft_v3`（accuracy 83.00%），`π_θ = π_ref`（merge 后挂新 LoRA）。

### 9.3 结果

| 指标 | **sft_v3**（起点） | **dpo_sft_v3_b05** | 变化 |
|---|---|---|---|
| accuracy | **83.00%** | **49.25%** | **−33.75pp** |
| pass@4 | 83.00% | 57.00% | −26pp |
| pass^4 | 83.00% | 40.00% | −43pp |
| Δstability | 0.00% | **−17.00%** | 严重不稳定 |
| **格式正确率** | **100%** | **58.8%** | **−41.2pp** |

按类别：

| 类别 | sft_v3 | dpo_sft_v3_b05 | 变化 |
|---|---|---|---|
| `multi_tool` | **100%** | **33.3%** | **−66.7pp** |
| `booking_flow` | 72.2% | **16.7%** | **−55.5pp** |
| `flight_search` | 68.2% | 26.1% | −42.1pp |
| `cancellation` | 66.7% | 33.3% | −33.4pp |
| `baggage` | 100% | 85.0% | −15.0pp |
| `recovery` | 91.7% | 81.2% | −10.5pp |
| `weather_advisory` | 100% | 91.7% | −8.3pp |

**注意 `multi_tool` 与 `booking_flow` 掉得最狠** —— 它们正是
「必须真正执行动作」的多轮依赖链任务。

### 9.4 失败模式：模型学会了**过度澄清**

```
子类型                              数量    sft_v3 对比
该调用但只做了澄清 (clarify_instead)   54   ← sft_v3 只有 20
声称已完成操作（编造执行结果）           44   ← sft_v3 是 0
只承诺要查，没真的调用                 16
输出为空                             11
完全没有 tool_call 标签               10
```

**这是本实验最重要的发现**：

DPO 的 clarify 偏好对教的是「信息不足时应追问」（`chosen=追问` / `rejected=硬调工具`）。
模型**把这个模式泛化过头了** —— 在**信息充足、本该动手**的任务上也改成追问。

`booking_flow` 从 72.2% 掉到 16.7%、`multi_tool` 从 100% 掉到 33.3% 都是这个原因：
**那些任务需要真正执行工具调用，而模型现在只追问不执行。**

### 9.5 训练监控失效

训练结束时的曲线**看起来是健康的**：

| 指标 | β=0.1（崩塌） | **β=0.5（本次）** | 表面判断 |
|---|---|---|---|
| 起点 `logps/chosen` | −6.00 | **−6.17** | 与探针预测一致 ✅ |
| 终点 `logps/chosen` | **−25.09** | **−4.55** | **没有漂移** ✅ |
| 终点 `loss` | 0.0208 | 0.1899 | 合理 ✅ |
| 终点 `rewards/margins` | +12.19 | +9.69 | （β 不同，不可直接比） |
| **实际 accuracy** | **10.0%** | **49.25%** | ❌ **都崩了** |

**我基于 `logps/chosen` 稳定而判断「本次明显更健康」—— 但任务能力依然崩掉 33.75pp。**

**结论：`logps/chosen` 只能预测「分布是否漂移」，不能预测「任务能力是否保留」。**

两者**可以独立失效**：一个模型完全可以在「隐式奖励差没有爆掉」的同时，
把**行为模式**改坏（从「执行动作」变成「先追问」）。
这类失效**不会**在 `loss`、`rewards/margins`、`rewards/accuracies`、
`logps/chosen` 任何一个指标上留下痕迹。

**也因此，§5.1 把崩塌归因于 β 的结论是错的** —— β 从 0.1 提到 0.5（5 倍），
准确率从 10.0% 到 49.25%，**都在崩**。真正的问题在**数据**：
clarify 偏好对的比例与它携带的行为模式强度。

### 9.6 一个明确的正面证据仍然成立

尽管总体崩塌，**这一组在追问边界上做对了**：

| 错误模式 | sft_v3 | dpo_sft_v3_b05 |
|---|---|---|
| **`missing_reason_corrected`** | **0% 通过 / 100% 重复** ❌ | **100% 通过 / 0% 重复** ✅ |

**这与 β=0.1 那版一致（那次也是 4/4）。** 说明对比信号**确实**能教会
「信息不足时应追问」这个否定性约束 —— 这是纯正例监督做不到的（见 §4.2）。

**但这正好揭示了问题的本质**：

> DPO 学会了一个**全局的行为倾向**（「遇到不确定就追问」），
> 而不是**有条件的判断规则**（「信息不足时追问，信息充足时执行」）。
>
> 偏好对只告诉它「A 比 B 好」，没有告诉它「A 在什么条件下比 B 好」。

### 9.7 对本项目核心论点的最终影响

| 论点 | 判定 |
|---|---|
| 「对比信号能做到纯正例做不到的事」 | ✅ **成立**（`missing_reason` 0%→100%，两组 DPO 一致复现） |
| 「DPO 能提升总体准确率」 | ❌ **不成立**（β=0.1 崩到 10.0%，β=0.5 崩到 49.25%） |
| 「修正 β 就能两全」 | ❌ **不成立**（本实验直接否定） |

**最终结论**：在本项目的任务集与数据构成下，**DPO 的对比信号能改善
特定的行为边界，但会以损害任务执行能力为代价**。
要让两者兼得，需要的是**条件化的偏好数据**
（明确标注「在什么输入特征下 A 优于 B」），而不是更多的同类偏好对。

### 9.8 这是一个有价值的负面结果

直接给出它，比修饰成「DPO 部分有效」更有意义：

1. **它否证了一个具体的、可检验的假设**（β 是崩因），而不是含糊的「效果不好」
2. **它给出了失效机制**（clarify 模式的过度泛化），而不只是「分数掉了」
3. **它指出了一个可操作的下一步**（条件化偏好数据），而不只是「需要更多数据」
4. **它暴露了训练监控的盲区**（`logps/chosen` 无法预测任务能力失效），
   这对任何做 DPO 的人都有直接参考价值
