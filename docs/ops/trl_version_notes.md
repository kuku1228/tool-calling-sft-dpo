# TRL 版本兼容说明

## 文档定位

这份文档回答的问题是：**为什么这个项目把 `trl` 锁在 0.13.0，以及换版本时具体会坏在哪一行。**

读者是：（1）在自己机器上 `pip install -r requirements.txt` 之后跑不起来训练、拿到一个 `TypeError` 的人；（2）要升级 TRL 或换 transformers 的人；（3）面试里被问「你踩过哪些训练框架的兼容性坑」时想要具体答案的人。

三条阅读前提：

1. **本文所有版本事实的来源是 [`docs/references/DPO_tool_calling_research_memo.md`](../references/DPO_tool_calling_research_memo.md)**。那份备忘录逐条核实了 TRL 各版本 tag 的源码与 release notes，并**显式标注了 7 项未核实条目**。本文不引入备忘录之外的新版本断言；凡属工程估算的地方都会标出来。
2. 备忘录的核实时间基准是 **TRL 1.14.1**（GitHub 最新 release tag `v1.14.1`）。
3. 本项目**没有**在真机上验证过 0.13.0 以外的任何 TRL 版本。所有「其他版本可能怎样」的说法都来自源码阅读，不是实测。见 §9。

代码位置：

| 关注点 | 代码 |
|---|---|
| 版本自适应层 | [`src/training/compat.py`](../../src/training/compat.py) |
| 训练入口（构造 `DPOConfig` / `DPOTrainer`） | [`scripts/train/dpo_train.py`](../../scripts/train/dpo_train.py) |
| 纯 Python DPO 损失（与 TRL 无关，可独立验证） | [`src/training/dpo_loss.py`](../../src/training/dpo_loss.py) |
| 依赖声明 | [`requirements.txt`](../../requirements.txt)、[`pyproject.toml`](../../pyproject.toml) |

---

## 1. `DPOTrainer` 的破坏性变更时间线

### 1.1 总览

| TRL 版本 | `DPOTrainer.__init__` 的位置参数 | tokenizer 参数名 | 超参入口 | `max_prompt_length` |
|---|---|---|---|---|
| **≤ 0.8** | `(model, ref_model, beta, train_dataset, args)` 全部可用 | `tokenizer=` | 位置参数为主，`args` 可选 | — |
| **0.9 ~ 0.11** | 仍可用；`beta` 是第 3 个位置参数 | `tokenizer=` | `DPOConfig` 引入，`beta` 等移入 config，但位置参数仍接受 | **存在，默认 512** |
| **0.12.0** | 仍在，但被 `_deprecate_arguments(version="0.13.0", ...)` 标记弃用 | **`processing_class=` 新增，`tokenizer=` 成为弃用别名**（`deprecate_kwarg(..., version="0.14.0")`） | 两条路都通 | 存在，默认 512 |
| **0.13.0** | **弃用到期**，必须用关键字参数 | 两者都在，本层优先用 `processing_class` | `DPOConfig` 是唯一推荐路径 | 存在，默认 512 |
| **0.14.0** | 位置参数已从签名移除 | **`tokenizer=` 被移除**，写它就报错 | `DPOConfig` | 存在；同版本 `max_length` 默认从 `None` 变 **1024**，并新增 `DPOConfig.tools` |
| **0.28.0** | — | — | — | **弃用**：默认值改为哨兵 `-1`，显式设置时发 `FutureWarning` |
| **0.29.0** | — | — | — | **从 `DPOConfig` 移除** |
| **1.x**（1.0.0 起） | — | — | `DPOConfig` 已改继承 `_BaseConfig` 而非 `TrainingArguments` | 已消失；只剩 `max_length` 单旋钮，默认 1024 |

### 1.2 逐段展开

#### ≤ 0.8：位置参数时代

```python
# 这个写法在 v0.10.x 及以前是合法的
trainer = DPOTrainer(model, ref_model, beta, train_dataset, args=...)
```

`beta` 是 `__init__` 的**第 3 个位置参数**。这一版还有 `DPODataCollatorWithPadding`，且 `loss_type` 里仍含 `kto_pair`。

#### 0.9 ~ 0.11：`DPOConfig` 引入，但位置参数还活着

v0.10.1 的签名是 `(model=None, ref_model=None, beta=0.1, label_smoothing=0, loss_type=None, args=None, ..., tokenizer=None, ..., max_length=None, max_prompt_length=None, max_target_length=None, peft_config=None, ...)`。`max_prompt_length` 从 **v0.9.6** 起存在，默认 `512`。

#### 0.12.0：两个弃用同时挂上，且**到期版本不同**

```python
@_deprecate_arguments(version="0.13.0", deprecated_args=["beta", ..., "max_prompt_length", ...],
    custom_message="Deprecated positional argument(s) used in DPOTrainer, please use the "
                   "DPOConfig to set these arguments instead.")
@deprecate_kwarg("tokenizer", new_name="processing_class", version="0.14.0",
                 raise_if_both_names=True)
def __init__(self, model=None, ref_model=None, beta: float = 0.1, ..., processing_class=None, ...)
```

| 装饰器 | `version=` | 含义 | 到期版本 |
|---|---|---|---|
| `_deprecate_arguments` | `"0.13.0"` | 位置参数从 0.13.0 起不可用 | **0.13.0** |
| `deprecate_kwarg("tokenizer", ...)` | `"0.14.0"` | `tokenizer=` 从 0.14.0 起不可用 | **0.14.0** |

v0.12.0 release notes 写的是 *"`tokenizer` is still supported … but deprecated and will be removed in the next release."* —— **这句话不精确**。以源码装饰器的 `version="0.14.0"` 为准：`tokenizer=` 在 0.13.0 上仍可用。同版本还引入 `use_weighting`（WPO，默认 `False`）与**会话式数据集直通**（`prompt`/`chosen`/`rejected` 可为 message list，DPO 对应 PR #2131）。

#### 0.13.0：位置参数弃用到期

`beta` / `loss_type` / `max_length` / `max_prompt_length` 等只能通过 `DPOConfig` 传。同时 `discopop` 加入 `loss_type`，会话式数据集支持完成。

#### 0.14.0：`tokenizer=` 消失，`max_length` 默认值改变

`tokenizer=` 别名移除（写就报错）。`max_length` 默认值从 `None` 变成 **`1024`**——这个变更很隐蔽：没显式设 `max_length` 时，0.13.0 上「不截断」，0.14.0 上「截到 1024」。多轮工具调用的 prompt 很容易超过 1024，于是**偏好对会被从末尾砍掉，chosen 与 rejected 被砍成相同 → 变成噪声**。新增 `DPOConfig.tools` 与 `PreferenceCollator`。

#### 0.28.0 / 0.29.0 / 1.x

0.28.0 同时弃用 `max_prompt_length`、`max_completion_length`、`reference_free`、`rpo_alpha`、`tools`、`use_logits_to_keep`、`label_pad_token_id`、`force_use_ref_model`、`model_adapter_name`、`ref_adapter_name`、`ref_model_init_kwargs`、`base_model_attribute_name`、`generate_during_eval`（PR #4969）；0.29.0 全部移除。1.x 的 `truncation_mode` 默认从 `keep_end` 改为 `keep_start`，且 `keep_end` 被弃用、计划 v2.0.0 移除；PyPI `requires_dist` 显示需要 `transformers>=4.56.2`。

---

## 2. `max_prompt_length` 的完整生命线

这是本项目最需要讲清楚的一个参数，因为它是「多轮 prompt 长度」与「completion 长度」两类预算的分界。

| 阶段 | 版本 | 行为 |
|---|---|---|
| **出生** | v0.9.6 | 参数存在，`field(default=512)`。正常可用 |
| **双旋钮窗口** | **v0.14.0 ~ v0.27.0** | `max_length`（默认 1024，管 prompt+completion 总长）与 `max_prompt_length`（默认 512，单独管 prompt）**都可用**。这是显式控制两侧预算的最实用区间 |
| **弃用** | **v0.28.0** | 参数保留，但默认值改为哨兵 **`-1`**；`__post_init__` 里若用户显式设置，发 `FutureWarning`：*"deprecated and will be removed in version 0.29.0. We recommend filtering out overlong prompts from your dataset before passing it to the trainer instead of using this parameter."* 未设置时回填旧默认 512 |
| **移除** | **v0.29.0** | 从 `DPOConfig` 完全消失。截断只剩 `max_length` 一个旋钮，作用于整条序列 |
| **之后** | main / 1.x | 维持 `max_length` 单旋钮（默认 1024） |

### 2.1 关于「0.13.0 的双旋钮」要说清一件事

备忘录 §E.2 指出 0.13.0 下 `max_length` 的**生效路径**：

> 注意 0.13.0 下 `max_length` 只在 prompt 超限时起截断作用（见 `_truncate_tokens`：先按 `max_prompt_length` 截 prompt，若仍超长再截 response 到 `max_length - max_prompt_length`）。

也就是说 0.13.0 上**两个旋钮有分工**，不是「谁更小谁生效」。这与 0.29.0 之后「`max_length` 同时约束 prompt+completion」的语义**不同**——同一份配置在 0.13.0 与 1.x 上会得到不同的截断行为。这是锁版本最实际的理由之一。

### 2.2 本项目的取值由数据体检给出，不是拍出来的

`TrainCfg` / `configs/default.yaml`：`max_length=1536`、`max_prompt_length=1024`。取值链条（全在 [`src/training/dpo_dataset.py`](../../src/training/dpo_dataset.py)）：

1. `token_length_stats(rows, tokenizer)` 用真实 tokenizer 量出 prompt 与 prompt+completion 的 p50/p90/p99/max。
2. `TokenLengthStats.recommend_max_length(safety=1.02, round_to=128)` 按 **p99** 推荐并向 128 取整。用 p99 而不是 max，是为了不为了 1 条异常长样本把显存翻倍。
3. `truncation_report(rows, tokenizer, max_length, max_prompt_length)` 报三个比例，最关键的是 `degenerate_after_truncation_rate`——**截断后 chosen 与 rejected 变成完全相同的比例**（这些样本对 DPO 是纯噪声，loss 恒为 `log 2`）。报告注明「若 > 5% 请调大 max_length 或过滤长样本」。

`scripts/train/dpo_train.py --dry-run` 打印这一切并写进 `outputs/<exp>/data_health.json`：

```
长度: prompt p99=...  prompt+comp p99=...  max=...
当前 max_length=1536 / max_prompt_length=1024 → 截断率 completion=..., 截断后退化=...
⚠️ 建议把 max_length 提到 N、max_prompt_length 提到 M（按 p99）
```

实测（`outputs/dpo_format/data_health.json`，采样 400 条，Qwen2.5-1.5B tokenizer）：

```
长度: prompt p99=1086  prompt+comp p99=1107  max=1108
当前 max_length=1280 / max_prompt_length=1152 →
      截断率 prompt=0.0%  completion=0.0%
      截断后退化（两侧变相同）= 0.0%
建议配置: max_length=1152  max_prompt_length=1152
```

`dpo_recovery` 的体检结果同量级（prompt p99=1055）。两者截断率均为 0，
说明 `max_length=1280` 有余量；退化率为 0 意味着没有偏好对因截断而变成噪声。

> 注意：`dpo_joint` 那次体检记录的是「未安装 transformers，跳过」——它是在没有
> tokenizer 的环境里跑的，因此**没有** token 长度数据，只写了字符长度。这不是 bug，
> 但说明「体检是否真的生效」需要看 `data_health.json` 里是 `token_length` 还是
> `char_length`。

---

## 3. `tokenizer` → `processing_class` 的精确边界

| 版本 | 状态 |
|---|---|
| **≤ v0.11.x** | 参数名是 `tokenizer` |
| **v0.12.0** | **新增 `processing_class`，`tokenizer` 成为弃用别名**（`@deprecate_kwarg("tokenizer", new_name="processing_class", version="0.14.0", raise_if_both_names=True)`）。对应 PR [#2162](https://github.com/huggingface/trl/pull/2162) |
| **v0.13.0** | **两者都在**。文档与 quick start 已统一改用 `processing_class`，但写 `tokenizer=` 仍然工作 |
| **v0.14.0** | **别名被移除**，写 `tokenizer=` 直接报错 |

两个容易忽略的细节：

1. **`raise_if_both_names=True`**：0.12.x / 0.13.0 上**同时**传 `tokenizer=` 和 `processing_class=` 会抛错，而不是「后者覆盖前者」。所以兼容层必须二选一：

   ```python
   if rep.uses_processing_class:
       kwargs["processing_class"] = tokenizer
   else:
       kwargs["tokenizer"] = tokenizer
   ```

2. **release notes 与源码措辞冲突**。v0.12.0 release notes 说会在 "the next release" 移除，源码装饰器写的是 `version="0.14.0"`。**以源码为准。** 这不是咬文嚼字——按 release notes 理解会以为 0.13.0 上不能用 `tokenizer=`，从而在「0.13.0 兼容性最宽」这个判断上得出相反结论。

---

## 4. 本项目为什么锁 `trl==0.13.0`

备忘录 §E.1 给出逐条判据，本项目采用同一结论：

| 判据 | 0.13.0 的情况 |
|---|---|
| **API 稳定性** | `DPOConfig` + `processing_class` 已是唯一推荐路径，且 `tokenizer=` 旧名仍兼容 → **兼容性最宽**：既能用新写法，也容得下旧环境 |
| **弃用警告噪声** | **位置参数弃用恰好到期于 0.13.0**。也就是说用 0.13.0 时写 `DPOConfig(beta=...)` 是「目标写法」，不会再看到 "please use the DPOConfig to set these arguments instead" 的警告刷屏 |
| **长度控制（双旋钮）** | `max_length`（默认 `None`）与 `max_prompt_length`（默认 512）**都正常可用**。多轮 prompt 与 completion 可以分别设上限 —— 这是 0.14.0 之后到 0.27.0 之间最实用的窗口，也是**本项目训练配置直接依赖的能力**（`max_length=1536` 与 `max_prompt_length=1024` 同时生效） |
| **不需要 `transformers>=4.56.2`** | 0.13.0 未把 transformers 的上界写进 PyPI metadata（备忘录标注为**未核实**），而 v1.x **确实**要求 `transformers>=4.56.2`。本项目 `requirements.txt` 写的是 `transformers>=4.45.0,<4.50.0`（理由：Qwen2.5 从 4.37 支持、`apply_chat_template(tools=...)` 的稳定行为在 4.45+ 才完善）。**选 0.13.0 才可以继续用 4.45~4.49 这条已验证的窗口** |
| **会话式数据集直通** | v0.12.0 起支持（PR #2131） |
| **`loss_type` 覆盖面** | 13 种，足够做消融（见 §5） |
| **工具调用相关 API** | ⚠️ **没有** `DPOConfig.tools`（0.14.0 才引入）。但本项目把 tools 写进 **prompt 文本**（见 [`preference_data_design.md`](../method/preference_data_design.md) §8.1），本就不依赖该参数 → 不是阻塞项 |
| **文档可得性** | `docs/source/dpo_trainer.mdx` 与 `dataset_formats.mdx` 在 tag 上可读，且 `tokenizer`/`processing_class` 双写法有明确迁移注解 |

**为什么不选 1.x**：需要 `transformers>=4.56.2`；参数面已收敛（`max_prompt_length` / `max_completion_length` / `reference_free` / `rpo_alpha` / `tools` 全移除）；`truncation_mode` 默认改为 `keep_start` 且 `keep_end` 本身被弃用。对**复现性优先**的项目，锁定一个中期的、文档完整的版本比追最新更有价值。

**为什么不选 0.14.0（唯一的次选）**：唯一好处是拿到 `DPOConfig.tools` 和新 `PreferenceCollator`；代价是 `tokenizer=` 报错（必须写 `processing_class=`）且 `max_length` 默认变 1024。既然 tools 写进 prompt 文本，这个特性用不上。

### 4.1 ⚠️ 依赖声明与本结论不一致（待修）

这是本次审计发现的一个**真实矛盾**，会影响复现：

| 位置 | 声明的 TRL 版本 |
|---|---|
| [`requirements.txt`](../../requirements.txt) L50 | `trl>=0.9.0,<0.13.0` |
| [`pyproject.toml`](../../pyproject.toml) L30（`gpu` extra） | `trl>=0.9.0,<0.13.0` |
| [`src/training/compat.py`](../../src/training/compat.py) 的 `recommended_pins()` | `"trl": "0.13.0"` |
| 本文与 README | 锁 0.13.0 |

`<0.13.0` **排除了 0.13.0**。按 `pip install -r requirements.txt` 装会得到 0.12.x（或更早），而不是本项目验证的目标版本。安装方式：

```bash
# 与文档一致（推荐）
pip install "trl==0.13.0"

# requirements.txt 当前会给你什么
pip install -r requirements.txt     # → trl 0.12.x，与 recommended_pins() 不一致
```

**影响评估**：0.12.x 上 `compat.py` 仍能工作（它会探测到 `processing_class` 存在，走新 API 分支），也不会触发 `tokenizer=` 弃用错误。但 `warn_if_versions_drift` 不会报警（0.12 < 0.14 且 ≥ 0.9），`compat_report.json` 里记的会是 `0.12.x`。所以**不会崩，但「锁 0.13.0」这个说法在按 requirements 安装的环境里是假的**。

**已修复**：`requirements.txt` 与 `pyproject.toml` 原先写的是区间 `>=0.9.0,<=0.13.0`，
实测会解析到 **0.12.x**，与 README / `compat.recommended_pins()` 声称的「锁 0.13.0」不一致。
现已改成精确锁定：

```
trl==0.13.0        # requirements.txt L56 与 pyproject.toml L30 同步
```

0.12.x 上 `compat.py` 本来也能工作（它探测到 `processing_class` 存在就走新 API 分支），
但 `warn_if_versions_drift` 不会报警（0.12 < 0.14 且 ≥ 0.9），
`compat_report.json` 里记的版本会与文档不符——所以这个不一致值得修掉而不是容忍。

---

## 5. `loss_type` 的正确取值

### 5.1 被流传最广的错误说法

> ❌ 「TRL 的 `DPOTrainer` 支持 `loss_type="simpo"` 或 `loss_type="orpo"`。」

**这是错的。** TRL 的 `DPOTrainer` **从未支持**这两个取值，任何版本都不支持，传进去会抛 `ValueError: Unknown loss type`。

原因：它们在 TRL 里是**独立的 trainer 实现**，不是 DPO 的一个 loss 分支。

- **SimPO** → `CPOTrainer` 家族（CPO/SimPO 是 reference-free 的 CPO 变体）
- **ORPO** → 独立的 `ORPOTrainer` / `ORPOConfig`（v0.13.0 仓库内已存在 `trl/trainer/orpo_config.py` 与 `orpo_trainer.py`）

术语上也不该混：SimPO 去掉了 reference model、隐式 reward 改为**长度归一化的平均 log-prob** 并引入 target margin γ；ORPO 把 SFT NLL 与 odds-ratio 项合并成一个单体目标。这两者的数学形式与 DPO 的 `−log σ(β·(log-ratio 差))` 都不是同一个东西，塞进 `loss_type` 一个字符串里也表达不了。

### 5.2 TRL 实际支持的取值（逐版本核实）

| TRL 版本 | `loss_type` 支持值 |
|---|---|
| v0.9.6 | `sigmoid`, `hinge`, `ipo`, `bco_pair`, `sppo_hard`, `nca_pair`, `robust`, `aot`, `aot_pair`, `exo_pair` |
| v0.10.1 / v0.11.0 / v0.12.0 | 同上 + `apo_zero`, `apo_down`（v0.12.0 新增的 `use_weighting` 是布尔开关，**不是**取值） |
| **v0.13.0**（本项目目标版本） | 同 v0.10.1 + `discopop` → **13 种** |
| v0.25.0 / v0.27.0 | 新增 `sft`；类型从 `Literal[...]` 变为 `str or list[str]`（MPO 式多损失加权，配 `loss_weights`） |
| v0.28.0 | 同 v0.27.0，但 `aot_pair` 更名为 `aot_unpaired` |
| main / 1.x | `sigmoid`, `hinge`, `ipo`, `exo_pair`, `nca_pair`, `robust`, `bco_pair`, `sppo_hard`, `aot`, `aot_unpaired`, `apo_zero`, `apo_down`, `discopop`, `sft`, `sigmoid_norm` |

**另一个已删除项**：`kto_pair` 曾存在于 v0.8.x 的 `loss_type` 中；v0.9.6 起在 `__post_init__` 里直接抛错：*"Support for kto_pair has been removed in DPOTrainer. Please use KTOTrainer."*

### 5.3 本项目自己的 `loss_type` 与 TRL **不是同一套取值**

```python
# src/training/dpo_loss.py
#: ⚠️ 与 TRL 的 DPOConfig.loss_type **不是**同一套取值！
SUPPORTED_LOSSES: tuple[str, ...] = ("sigmoid", "ipo", "label_smoothing")
```

| | 取值 |
|---|---|
| 本项目 `compute_dpo_loss` | `sigmoid`, `ipo`, `label_smoothing` |
| TRL `DPOConfig.loss_type` | `sigmoid`, `hinge`, `ipo`, `bco_pair`, …（13 种，见 §5.2） |

`label_smoothing` 在本项目里是一个**独立的 `loss_type`**，在 TRL 里是一个独立的**数值参数** `label_smoothing: float = 0.0`（区间 `[0.0, 0.5)`），与 `loss_type` 正交。所以 `dpo_train.py` 的帮助文本特意写了提示：

```python
p.add_argument("--loss-type", default=None,
               help="sigmoid / ipo / label_smoothing（本项目的实现；与 TRL 取值不同）")
```

**传 `--loss-type hinge` 会发生什么**：`dpo_train.py` 写进 `TrainCfg.loss_type`，`compat.build_dpo_config` 原样传给 `DPOConfig(loss_type="hinge")` —— TRL 会**接受**它。但 `dpo_loss.py` 的 `compute_dpo_loss`（只在 mock 路径与自检里用）遇到 `hinge` 会抛 `ValueError`。**真实训练路径与 mock 路径对 `loss_type` 的接受集合不同。** 这是已知裂缝，不是 bug，但排错时要意识到。默认值是 `"sigmoid"`。

---

## 6. `ref_model=None` + PEFT 的语义陷阱

### 6.1 TRL 到底怎么算 reference

**结论：是「临时关闭 adapter」，不是复制模型。** TRL 源码（v0.10.1 / v0.12.0）的核心分支：

```python
if ref_model:
    self.ref_model = ref_model
elif self.is_peft_model or args.precompute_ref_log_probs:
    # The `model` with adapters turned off will be used as the reference model
    self.ref_model = None
else:
    self.ref_model = create_reference_model(model)
```

以及真正取 reference 的地方：

```python
@contextmanager
def null_ref_context(self):
    """Context manager for handling null reference model (that is, peft adapter manipulation)."""
    with self.accelerator.unwrap_model(
        self.model
    ).disable_adapter() if self.is_peft_model and not self.ref_adapter_name else nullcontext():
        ...
```

所以 `ref_model=None` + LoRA 时，`compute_ref_log_probs` 会走 `with self.null_ref_context()`，而 `disable_adapter()` 把 LoRA 关掉后剩下的就是**原始 base 模型**。

### 6.2 陷阱：KL 锚点被拉远

TRL 的这套机制在语义上**假设「SFT 阶段已经 merge 进 base 权重」**。如果你在 base 上做 LoRA SFT（教会模型输出 `<tool_call>`），然后**继续用同一个 adapter** 做 DPO 且 `ref_model=None`，那么 $\pi_{\text{ref}}$ 是**未经 SFT 的原始 base**，而 $\pi_\theta$ 是 base + SFT + DPO。后果：

- 隐式 reward $\beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$ 的分子分母差距里混进了「SFT 学到的格式能力」，不只是「偏好学习带来的增量」。
- KL 约束（β 的物理意义）实际在惩罚「你做过 SFT」，而不是「你偏离偏好数据」。
- 表现是训练不稳、`rewards/margins` 异常大或异常抖。

量级直觉：SFT 把「输出合法 `<tool_call>`」的概率从 50% 抬到 95%，这部分 log-prob 提升约 **+0.64 nats/序列**，会**全额**进入 implicit reward；而 DPO 想学的那点偏好差异（工具名选对 vs 选错）通常远小于 1 nat。**信噪比被 SFT 项淹没了。**

### 6.3 三种正确做法

备忘录引用 TRL v0.13.0 文档 `dpo_trainer.mdx` 的 "Reference model considerations with PEFT" 一节，官方给出三条路：

| # | 做法 | 优点 | 缺点 |
|---|---|---|---|
| **1** | 创建**两个模型实例**，各自加载同一个 adapter | 语义最直白，无隐患 | **非常低效**（显存翻倍） |
| **2** | 把 adapter **merge** 进 base，保存成新模型，再在它上面**新建 adapter** 做 DPO，`ref_model` 留 `None` | 高效：`disable_adapter()` 得到的正好是「SFT 后的模型」，语义正确 | QLoRA 场景需先反量化再 merge，之后要么重新量化要么吃显存 |
| **3** | 同一 adapter 加载**两次**、起不同名字，训练中用 `set_adapter` 切换 | 只多一个 adapter 的显存开销，避开方案 2 的反量化问题 | 略复杂；需设 `model_adapter_name` / `ref_adapter_name`（0.28.0 弃用、0.29.0 移除） |

### 6.4 本项目采用的方案 2

`scripts/train/dpo_train.py --init-adapter <SFT adapter 路径>`：

```python
if args.init_adapter:
    from peft import PeftModel
    print("  理由：ref_model=None 时 reference 是 base；若继续沿用同一个")
    print("        SFT adapter，KL 锚点会偏离 base，导致 DPO 训练不稳。")
    model = PeftModel.from_pretrained(model, args.init_adapter)
    model = model.merge_and_unload()                    # ← 先 merge
    merged_dir = out_dir / "base_with_init_adapter"
    merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))
```

merge 之后 `model` 是纯 base 权重（SFT 已被吃进去），随后会创建一个**全新的** `LoraConfig`（`r=tcfg.lora_r`、`lora_alpha=tcfg.lora_alpha`、`bias="none"`、`task_type="CAUSAL_LM"`）——**先 merge 再新建**。此时 `disable_adapter()` 关掉的是**新** adapter，剩下的正是 merge 过的「SFT 后模型」，KL 锚点落在正确位置。中间产物 `outputs/<exp>/base_with_init_adapter/` 保留下来，让「reference 到底是哪个模型」在文件系统上可查证。

**不**传 `--init-adapter`（直接在 base 上做 DPO）时，reference 是 base、policy 是 base+DPO，语义同样正确——这是 `run_dpo.sh` 的默认行为。**唯一危险的情形是「SFT adapter 与 DPO adapter 是同一个」，`--init-adapter` 正是为它准备的。**

### 6.5 另外几个边界

- **`model` 与 `ref_model` 不能是同一个对象**。TRL 会抛错：*"`model` and `ref_model` cannot be the same object. If you want `ref_model` to be the same as `model`, you must make a copy of it, or `None` if you use peft."*
- **同时传 `ref_model` 与 `peft_config`** 时，除非 `force_use_ref_model=True`，TRL 直接抛错：*"You passed both a ref_model and a peft_config. For training PEFT adapters with DPO there is no need to pass a reference model..."*。本项目不传 `ref_model`，不会触发。
- **`precompute_ref_log_probs=True`** 把 reference log-prob 训练前算好写回数据集，训练时 reference 不常驻显存。本项目保持默认 `False`——LoRA + `disable_adapter()` 已经不需要第二个模型常驻，没必要预先多跑一遍 forward。
- **v1.x 修正了这个语义**：main 分支改用 `use_adapter(model, adapter_name="ref" if "ref" in model.peft_config else None)`，注释明确区分两种情况——*"New adapter: disabling adapters yields the base model. Re-training an existing adapter: an initial copy is loaded under the name 'ref'."*。本项目锁 0.13.0，**不能依赖这个修正**，所以必须自己 merge。

---

## 7. `src/training/compat.py` 的适配策略

### 7.1 为什么用 `inspect.signature` 而不是 `try/except` 硬试

模块 docstring 的原话：

> 早期写法是「先试新 API，`TypeError` 就退回旧 API」。问题是 `TypeError` 可能来自内部（例如 `DPOConfig` 少一个字段），错误会被吞掉并误判成「版本不支持新 API」，最终报出一个与真实原因无关的错。显式检查签名能给出准确的诊断。

失败链条是这样形成的：

| 步 | 发生什么 |
|---|---|
| 1 | 你写了 `processing_class=tokenizer` |
| 2 | 环境是 0.11.x，`__init__` 不接受该参数 → `TypeError` |
| 3 | `except TypeError` 捕获，判定「版本太老，不支持 `processing_class`」 |
| 4 | 退回 `tokenizer=tokenizer` |
| 5 | 仍然失败，但这次的真实原因是**别的**（`peft_config` 与 `ref_model` 冲突、或 `DPOConfig` 字段名写错） |
| 6 | 你看到的报错是「`tokenizer` 参数不被接受」——**与真实原因无关** |

显式探测把「版本的形状」变成一次可打印的查询结果，而不是一次猜测：

```python
sig = inspect.signature(DPOTrainer.__init__)
params = set(sig.parameters)
rep.uses_processing_class = "processing_class" in params
rep.supports_ref_model = "ref_model" in params
```

它也**不能免除**最终的 `try/except`——`build_trainer` 真正构造时仍捕获 `TypeError`，但把完整诊断（原始错误 + 实际传入的参数 + 签名接受集 + `CompatReport` 摘要 + 建议）一起抛出：

```python
raise TypeError(
    "构造 DPOTrainer 失败 —— 这通常意味着 TRL 版本与本项目的适配层不匹配。\n"
    f"  原始错误: {exc}\n"
    f"  实际传入的参数: {sorted(filtered)}\n"
    f"  DPOTrainer 接受: {sorted(accepted)}\n"
    f"  {rep.summary()}\n"
    "请把这段信息连同 `pip show trl` 的输出一起反馈；\n"
    "或参考 docs/ops/trl_version_notes.md 锁定 trl==0.13.0。"
) from exc
```

差别在于：**显式探测让「正常路径」不产生歧义，`try/except` 只负责兜住「探测与实际仍然不一致」这个异常情形，并把证据留全。** 这比「先猜再试」在排错上的效率差一个数量级。

### 7.2 `CompatReport` 记录哪些字段

```python
@dataclass
class CompatReport:
    trl_version: str = "unknown"
    transformers_version: str = "unknown"
    peft_version: str = "unknown"
    torch_version: str = "unknown"
    uses_processing_class: bool = False        # 是否走了新 API
    uses_dpo_config: bool = False              # 是否拿到 DPOConfig
    dpo_config_fields: list[str] = ...         # DPOConfig 实际接受的字段全集
    dropped_fields: list[str] = ...            # ★ 被本次适配丢弃的字段
    supports_ref_model: bool = True
    notes: list[str] = ...
```

`make_compat_report()` 用 `_ver(mod_name)` 容错地取版本（取不到返回 `not-installed`），然后从签名推导 `uses_processing_class` / `supports_ref_model`、追加人类可读的 `notes`（例如「tokenizer= 与 processing_class= 同时存在（0.12.x）」）、从 `DPOConfig.__dataclass_fields__` 取全部字段名。整份报告落盘到 `outputs/<exp>/compat_report.json`。

### 7.3 为什么 `dropped_fields` 是最重要的一个字段

`build_dpo_config` 的过滤逻辑（`values` 是从 `TrainCfg` 映射来的参数字典）：

```python
accepted = set(rep.dpo_config_fields) if rep.dpo_config_fields else set(values)
if rep.dpo_config_fields:
    dropped = sorted(k for k in values if k not in accepted)
    if dropped:
        rep.dropped_fields.extend(dropped)
        values = {k: v for k, v in values.items() if k in accepted}
```

**没有这个字段会发生什么**：你写 `--beta 0.5`，脚本一路跑到 `trainer.train()`，训练正常完成，loss 曲线正常——但 β 实际是默认的 0.1，因为该版本的 `DPOConfig` 不认识那个字段名，`**values` 展开时被静默丢掉。

这不是「报错难查」，而是**「根本没有报错」**。你会得到一份标注 β=0.5 的 `train_config.json`（`dpo_train.py` 写的是 `getattr(tcfg, k)`，即配置里的值）、一份 β 实际为 0.1 的权重、以及一条对不上的曲线。事后复现时，「我明明设了 β=0.5」会很难溯源。

`dropped_fields` 把这件事变成训练前就能看见的一行警告：

```
  ⚠️ 被丢弃的字段: ['max_prompt_length']
     （该版本 DPOConfig 不接受这些字段，训练中它们不生效）
  · 该版本 DPOConfig 不接受 max_prompt_length（TRL >= 0.28 已移除）；...
```

`build_trainer` 里还有**第二道**过滤，针对 `DPOTrainer.__init__` 的签名（`filtered = {k: v for k, v in kwargs.items() if k in accepted or k == "self"}`），丢弃的键同样累加进 `dropped_fields`。所以这个列表是**累加**的：config 层丢的与 trainer 层丢的合并在一起，报告里一次看完。

### 7.4 两处实现瑕疵

1. **`beta` 的位置参数探测意义有限**。`make_compat_report` 里有：

   ```python
   beta_param = sig.parameters.get("beta")
   rep.notes.append("beta 仍可作为位置参数传入" if beta_param is not None
                    else "beta 只能通过 DPOConfig 设置（>= 0.13）")
   ```

   语法正确，而且恰好在 0.10.1（`beta` 在签名里）与 0.13.0（已移除）上给出不同结果。但它得到的信息不如直接核对 `max_prompt_length` 是否在 `dpo_config_fields` 里可靠。**不要把它当版本判定的主要依据。**

2. **`build_trainer` 已实现但 `dpo_train.py` 未使用**。后者的注释说明了原因：「`DPOTrainer` 需要 `DPOConfig`；`build_trainer` 里会再构造一次，这里改成先构造 config 再直接调 `DPOTrainer`，避免重复。」代价是 **`tokenizer`/`processing_class` 二选一、签名过滤、错误诊断这三段逻辑在两个文件里各写了一遍，改一处不改另一处不会报错**。

**已知冗余（保留，未合并）**：`compat.build_trainer` 已实现，但 `dpo_train.py` **没有调用它**。
后者先构造 `DPOConfig` 再直接调 `DPOTrainer`，以避免 `build_trainer` 内部重复构造 config。

代价是 **`tokenizer`/`processing_class` 二选一、签名过滤、错误诊断这三段逻辑在两个文件里各写了一遍**，
改一处不改另一处不会报错。合并的方式很明确（让 `dpo_train.py` 把已构造的 config 传给
`build_trainer`），但属于重构而非缺陷修复，因此保持现状并在本节登记。

---

## 8. β 的论文原始取值

### 8.1 DPO 论文（Rafailov et al. 2023, Appendix B）

论文原文（Appendix B "DPO Implementation Details and Hyperparameters"）：

> *"Unless noted otherwise, we use a β = 0.1, batch size of 64 and the RMSprop optimizer with a learning rate of 1e-6 by default. We linearly warmup the learning rate from 0 to 1e-6 over 150 steps. For TL;DR summarization, we use β = 0.5, while rest of the parameters remain the same."*

| 项 | 主实验 | TL;DR 摘要任务 |
|---|---|---|
| **β** | **0.1** | **0.5** |
| batch size | 64 | 64 |
| 优化器 | RMSprop | RMSprop |
| learning rate | 1e-6 | 1e-6 |
| warmup | 从 0 线性升到 1e-6，**150 步** | 同 |

三点必须说清的限定：

1. 论文的 **RMSprop + lr 1e-6** 是**全参数微调**设定。本项目 LoRA 场景参数少、梯度尺度不同，`lr` 用 `5e-5`（`TrainCfg.lr`，`configs/default.yaml` 注明「DPO 论文用 1e-6，但那是全参微调；LoRA 下 5e-5 是更常见的量级」）。**这不是对论文的偏离，因为优化对象不同。**
2. 论文**并未报告**「0.1 / 0.5 / 1.0 的 grid search」。网上流传的三值区间是社区惯例，不是该论文的 ablation 结论。
3. TRL 的 `DPOConfig.beta` 默认值也是 **0.1**，与论文缺省一致（多版本确认）。

### 8.2 β 的梯度效应（为什么它必须和 lr 一起调）

[`src/training/dpo_loss.py`](../../src/training/dpo_loss.py) 的模块 docstring 记录了这个推导：

```
z = β · [ (logπ_θ(y_w) − logπ_ref(y_w)) − (logπ_θ(y_l) − logπ_ref(y_l)) ]
L = −log σ(z)

∂L/∂ logπ_θ(y_w) = −β · σ(−z)
∂L/∂ logπ_θ(y_l) = +β · σ(−z)
```

**梯度里有一个显式的 β 因子。** 所以：

- β 大 → 对参考模型偏离的惩罚大 → 更保守，`z` 被压小，loss 更大；
- β 小 → 允许大幅偏离 → 但**梯度幅度也被 β 线性缩小**。

这就是为什么 β 和 lr 必须一起调：把 β 从 0.1 降到 0.01 会让梯度也小 10 倍，不相应放大 lr 的话训练看起来「几乎不动」。反过来只调 lr 不调 β，会改变「保守程度」与「更新速度」的相对关系。

`dpo_loss.py` 的 `self_check()` 机械验证这套公式（10 项检查）。与 β 直接相关的几项：`initial_loss_is_log2`（π_θ = π_ref 时 loss 恰为 `log 2`）、`monotone_decreasing`、`saturates_to_zero`、`grad_chosen_matches_analytic` / `grad_rejected_matches_analytic`（有限差分验证 −β·σ(−z) 与 +β·σ(−z)）、`ipo_minimum_at_target`（IPO 目标 margin `1/(2β)`）。

这套自检**不需要 torch**，CPU 上随时能跑（`python -m src.training.dpo_loss`），而且 `dpo_train.py` 每次训练前都跑一遍。它的价值在于：**改了 β 相关的公式，曲线会跟着变**，而不是「模型不动也不知道为什么」。

一个已知瑕疵：`self_check()` 的返回值里混了 `checks["beta"] = "see individual checks"` —— 一个**字符串而不是布尔值**。`report_self_check()` 专门把它和 `passed` 一起排除在 ✓/✗ 打印之外：

```python
for k, v in res.items():
    if k in ("passed", "beta"):
        continue
    lines.append(f"  {'✓' if v else '✗'} {k}")
```

从「报告不撒谎」的角度保留这个键是合理的（β 的影响分散在各检查项里），但往 `dict[str, object]` 里混一个字符串会让人写 `all(v for v in res.values())` 时踩坑。

---

## 9. 版本漂移的自检：`warn_if_versions_drift`

```python
def warn_if_versions_drift(rep: CompatReport) -> list[str]:
```

它在 `dpo_train.py` 的真实训练路径里被调用，逐条打印 `⚠️`。判据：

| 条件 | 警告内容 |
|---|---|
| `trl in ("not-installed", "unknown")` | 「未检测到 trl，无法做真实训练（mock 路径不受影响）」并**立即返回** |
| 版本号无法解析成 `(major, minor)` | 返回已收集的警告（通常为空），静默通过 |
| `(major, minor) >= (1, 0)` | 「TRL {ver} 是 1.x：位置参数与 `tokenizer=` 都已移除，且需要 `transformers>=4.56.2`。本项目未在该版本上验证，建议锁 0.13.0。」 |
| `(0, 14) <= (major, minor) < (1, 0)` | 「TRL {ver}：`tokenizer=` 已移除；`max_prompt_length` 在 0.28+ 被删。本项目在 0.13.0 上验证，其他版本可能会出现参数被静默忽略。」 |
| `(major, minor) < (0, 9)` | 「TRL {ver} 过旧：没有 `DPOConfig`，超参只能按位置参数传，本项目未适配该路径。」 |
| `(0, 9) <= (major, minor) <= (0, 13)` | **不警告** |

**最后一格是这份自检的设计意图**：0.9 ~ 0.13 是本项目兼容层明确支持的区间，所以不打扰。但注意由此产生的一个后果——按 `requirements.txt` 装到的 **0.12.x 不会触发任何警告**，尽管它与文档声称的 0.13.0 不同（见 §4.1）。**`warn_if_versions_drift` 不能替代「核对实际装了什么版本」**，它只拦明显越界的版本。

另外 `dpo_train.py` 在警告之后还有一道硬门禁：

```python
if not rep.dpo_config_fields:
    print("\n错误：当前环境没有 trl 的 DPOConfig，无法训练。", file=sys.stderr)
    print(f"  推荐版本: {json.dumps(recommended_pins(), ensure_ascii=False)}", file=sys.stderr)
    return 1
```

`dpo_config_fields` 为空 = `from trl import DPOConfig` 失败。所以「没有 TRL」或「TRL 太老」会在**加载模型之前**就被挡住，不会白烧 GPU 时间去下载 3GB 权重。

---

## 10. 推荐版本组合

`compat.recommended_pins()` 的内容（唯一事实来源）：

```python
{
    "trl": "0.13.0",
    "transformers": ">=4.45.0,<4.50.0",
    "peft": ">=0.7.0",
    "torch": ">=2.1.0",
    "datasets": ">=2.16.0",
}
```

被 `print(recommended_pins())` 嵌进「没有 DPOConfig」的错误信息里，所以用户不需要翻文档就能拿到正确的安装命令。

各条的来源：

| 依赖 | 下界理由 | 上界理由 |
|---|---|---|
| `trl` | 0.9.6 起 `kto_pair` 已被移除并抛错（说明重构完成）；0.9 起有 `DPOConfig` | **锁 0.13.0**，见 §4。上界 `<0.13.0` 出现在 requirements.txt 里是 bug |
| `transformers` | 4.45：Qwen2.5 系列从 4.37 支持，但 `apply_chat_template(tools=...)` 的稳定行为 4.45+ 才完善 | `<4.50.0` 是实测窗口，不是理论边界。备忘录**标注「TRL 0.13.0 的 transformers 上界未核实」**（PyPI JSON 只保留最新版的 `requires_dist`）；0.12.0 release notes 提到 "Bump the minimum transformers version to v4.46"（PR #2245），可作下界参考 |
| `peft` | 0.7+ 才支持 `target_modules="all-linear"`（本项目 `TrainCfg.lora_target_modules` 用的就是它） | 无 |
| `torch` | 2.1+；3090/4090 是 sm_86，用 cu121 最稳。AutoDL 镜像通常已预装 | 无 |
| `datasets` | 2.16+；本项目用 `Dataset.from_list` 且只要 `prompt`/`chosen`/`rejected` 三列，对版本要求不高 | 无 |

备忘录 §E.1 还建议了更激进的 pin（`transformers>=4.46,<4.50`、`peft>=0.13`、`datasets>=3.0`、`accelerate>=0.34`），并明确说「请以本地实测为准，并在 requirements.txt 中写死」。

**这份仓库没有在真机上验证过任何 pin**。所有 pin 都是「按 API 需求推出的可工作区间」，不是实测最优区间。

---

## 11. 遇到版本问题怎么排查

按顺序做，每一步都留下证据。

### 第 1 步：确认实际装了什么

```bash
pip show trl transformers peft torch datasets | Select-String -Pattern "^(Name|Version)"
# 或者一行拿到全部
python -c "import trl,transformers,peft,torch,datasets;print('trl',trl.__version__);print('transformers',transformers.__version__);print('peft',peft.__version__);print('torch',torch.__version__);print('datasets',datasets.__version__)"
```

**不要跳过这一步。** §4.1 那个矛盾意味着「按文档装的」和「按 requirements 装的」可能不是同一个版本。

### 第 2 步：拿到兼容层报告

```bash
python -c "import sys;sys.path.insert(0,'.');from src.training.compat import make_compat_report,warn_if_versions_drift,recommended_pins;r=make_compat_report();print(r.summary());print();print('推荐:',recommended_pins());print('警告:',warn_if_versions_drift(r))"
```

`summary()` 会打印四个版本号 + `processing_class` / `DPOConfig` 两个布尔 + `dropped_fields`（如果有）+ notes。**如果这一行出现了 `⚠️ 被丢弃的字段`，问题就在那里，不用往下查了。**

### 第 3 步：看 `DPOTrainer` 与 `DPOConfig` 的形状

```bash
python -c "import inspect;from trl import DPOTrainer;print(sorted(inspect.signature(DPOTrainer.__init__).parameters))"
python -c "from trl import DPOConfig;fs=sorted(DPOConfig.__dataclass_fields__);print(len(fs),'fields');print([f for f in fs if 'max' in f or 'prompt' in f or 'completion' in f])"
```

对照 §1.1 与 §2 的表判断版本区间：

| 你在列表里看到 | 说明 |
|---|---|
| `tokenizer` **和** `processing_class` 都在 | 0.12.x / 0.13.x |
| 只有 `processing_class` | 0.14+ |
| 有 `beta` | ≤ 0.11.x |
| 同时有 `max_length` 和 `max_prompt_length` | 0.9 ~ 0.27（**本项目要的就是这个**） |
| `max_prompt_length` 默认是 `-1` | 0.28.x，且会收到 `FutureWarning` |
| 没有 `max_prompt_length` | 0.29+，`max_length` 是唯一旋钮，**必须确认它足够大** |

### 第 4 步：用 mock 路径隔离「框架问题」与「代码问题」

```bash
python scripts/train/dpo_train.py --experiment dpo_joint --mock
```

mock 路径**完全不导入 trl / torch / peft / transformers**，它用 `src/training/dpo_loss.py` 的纯 Python 实现做真实的解析梯度下降。所以：

- mock 跑通 + 真实跑挂 → **问题在框架层**（版本、显存、模型加载），不在数据或损失实现。
- mock 也挂 → **问题在本项目代码或数据**，先别动 TRL 版本。

这是最重要的一条分流手段，也是整个 mock 路径存在的首要理由。

### 第 5 步：收集这五样东西

出问题时一起贴出来，别人能立刻定位：

1. `python -c "from trl import DPOTrainer; import inspect; print(inspect.signature(DPOTrainer.__init__))"`
2. `outputs/<exp>/compat_report.json`（含 `dropped_fields`）
3. 完整的 traceback。本项目在版本不匹配时会抛出带「实际传入的参数 / `DPOTrainer` 接受 / `CompatReport` 摘要」的 `TypeError`——**这段本身就是诊断报告**
4. `pip list | Select-String -Pattern "trl|transformers|peft|torch|accelerate|datasets"`
5. `outputs/<exp>/data_health.json`（如果 `--dry-run` 已经跑过）

如果报错信息里**同时**出现 `tokenizer` 与 `processing_class`，检查是不是两条构造路径都被触发了（例如 `dpo_train.py` 内联构造 + 某个包装函数）。0.12.x / 0.13.0 上 `raise_if_both_names=True` 会直接抛错，不会静默选一个。

---

## 12. 未核实事项汇总

以下条目在[备忘录](../references/DPO_tool_calling_research_memo.md)中被显式标注为**未核实**，本文沿用同样的标注：

| 未核实项 | 说明 |
|---|---|
| TRL 0.13.0 的 `transformers` 上界 | PyPI JSON 只保留最新版的 `requires_dist`。§10 的 `<4.50.0` 是本项目实测窗口，不是官方边界 |
| TRL v1.x 中 DPO 从数据集的哪个列读取 `tools` | `DPOConfig.tools` 在 main 已移除、tools 必须由数据集提供，但具体列名未找到源码级证据 |
| SimPO 论文 Appendix B 的 β、γ 具体推荐值 | arxiv HTML 正文在 §3 之后被截断，无法核实「β=2.0, γ=0.5」这一常见说法 |
| Qwen2.5-0.5B/1.5B/3B 是否原生支持 function calling | Qwen 官方博客的 tool calling 示例**只用 7B**，未找到明确的尺寸限制原文。这恰好是本项目做偏好对齐的价值所在，但**不能**断言 1.5B 不支持 |
| τ-bench user simulator 的具体采样温度 | `RunConfig` 只给出 agent 的 `temperature=0.0` |
| Functionary 在 vLLM 中的 parser 名称 | — |
| Qwen2.5-1.5B 工具调用 LoRA 的推荐 rank / lr 的一手文献 | **未找到**。`TrainCfg` 的 `lora_r=16` / `lora_alpha=32` / `lr=5e-5` 是社区惯例 + DPO 论文锚点的工程推断 |
| **（本文新增）** 本项目在 0.13.0 以外的 TRL 版本上的实际行为 | `compat.py` 有 0.9 ~ 0.13+ 的多条分支，但**没有一条在真机上跑过**。本文所有「0.14+ 会怎样」的说法都来自源码阅读。首次上 GPU 时的首要动作应是核对 `compat_report.json` |
