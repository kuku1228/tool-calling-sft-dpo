# 技术备忘录：小模型工具调用 DPO 偏好对齐

> 面向"1.5B + LoRA + 单卡 3090"工具调用偏好对齐项目的代码实现与文档撰写。
> 所有 API 签名与版本结论均标注来源 URL。**未核实的条目已显式标注。**
>
> 核实时间基准：本文撰写时 TRL 最新版本为 **1.14.1**（PyPI `info.version`），GitHub 最新 release tag 为 `v1.14.1`（发布时间 2026-09-29）。

---

## A. DPO 损失函数

### A.1 原始论文的完整损失函数

Rafailov et al. 2023, *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, [arXiv:2305.18290v2](https://arxiv.org/html/2305.18290v2) 式 (7)：

```latex
\mathcal{L}_{\text{DPO}}(\pi_{\theta};\pi_{\text{ref}})
= -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}
\left[
\log \sigma\!\left(
\beta \log \frac{\pi_{\theta}(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}
-
\beta \log \frac{\pi_{\theta}(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}
\right)
\right]
```

引入简写 $\hat r_\theta(x,y) = \beta \log \frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}$（论文中称为 implicit reward），损失可写成等价的两行形式：

```latex
\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{\mathcal{D}}\Big[\log\sigma\big(\hat r_\theta(x,y_w) - \hat r_\theta(x,y_l)\big)\Big]
```

**逐项解释：**

| 符号 | 含义 | 工程对应 |
|---|---|---|
| $x$ | prompt（上下文），**不含** completion | 数据集 `prompt` 列；工具调用场景中即 system+历史多轮+本轮 user 消息 |
| $y_w$ (chosen) | 被偏好的完整回答 | 数据集 `chosen` 列，仅 completion 部分 |
| $y_l$ (rejected) | 被拒绝的回答 | 数据集 `rejected` 列 |
| $\mathcal{D}$ | 静态离线偏好数据集 $\{x^{(i)},y_w^{(i)},y_l^{(i)}\}_{i=1}^N$ | 无需在训练中采样 |
| $\pi_\theta$ | 正在训练的策略模型（含 LoRA adapter） | `model` |
| $\pi_{\text{ref}}$ | 参考策略，论文中即 SFT 模型 $\pi^{\text{SFT}}$ | `ref_model`（LoRA 场景下为关闭 adapter 的同一模型，见 §B.8） |
| $\sigma$ | logistic 函数 $\sigma(z)=1/(1+e^{-z})$ | `F.logsigmoid` |
| $\beta$ | 控制偏离 $\pi_{\text{ref}}$ 程度的温度/正则系数，**β 越大偏离越小** | `DPOConfig.beta`，默认 0.1 |
| $\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}$ | 序列级 log-prob 比（log-ratio），即 implicit reward / β | 实现为 completion 部分 token log-prob 的**求和** |

**关键实现细节（易踩坑）**：式中的 $\log\pi_\theta(y\mid x)$ 是 **序列级 log 概率**，即对 completion 的每个 token 求和（sum），而**不是**长度归一化的平均。TRL 源码中 `logps = per_token_logps.sum(dim=1)`（`sum over sequence length`），且 prompt 部分被 `completion_mask` 屏蔽。这一点在 IPO 与 SimPO 中被显式改动（见 §A.5）。

来源：[arXiv:2305.18290v2 §4](https://arxiv.org/html/2305.18290v2)

---

### A.2 从 Bradley-Terry + KL 约束推导到 DPO 闭式解

**Step 1 — Bradley-Terry 偏好模型**（式 1）：

```latex
p^*(y_1 \succ y_2 \mid x)=\frac{\exp\left(r^*(x,y_1)\right)}{\exp\left(r^*(x,y_1)\right)+\exp\left(r^*(x,y_2)\right)}=\sigma\big(r^*(x,y_1)-r^*(x,y_2)\big)
```

**Step 2 — KL 约束的 RLHF 目标**（式 3）：

```latex
\max_{\pi_{\theta}}\;
\mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_{\theta}(y\mid x)}\bigl[r_{\phi}(x,y)\bigr]
-\beta\,\mathbb{D}_{\mathrm{KL}}\bigl[\pi_{\theta}(y\mid x)\,\|\,\pi_{\text{ref}}(y\mid x)\bigr]
```

论文指出附加的 KL 约束很重要：它防止模型偏离到 reward model 不再准确的分布区域，并保持生成多样性、避免 mode collapse 到单一高 reward 回答。

**Step 3 — 闭式最优策略**（式 4）。该目标的最优解为：

```latex
\pi_r(y\mid x)=\frac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\,\exp\!\left(\frac{1}{\beta}r(x,y)\right),
\qquad
Z(x)=\sum_{y}\pi_{\text{ref}}(y\mid x)\exp\!\left(\frac{1}{\beta}r(x,y)\right)
```

其中 $Z(x)$ 是配分函数（partition function）。

**Step 4 — 反解出隐式 reward**（式 5）。对式 (4) 两边取对数并整理：

```latex
r(x,y)=\beta\log\frac{\pi_r(y\mid x)}{\pi_{\text{ref}}(y\mid x)}+\beta\log Z(x)
```

**Step 5 — 配分函数消去**。把式 (5) 代入 Bradley-Terry 的**差分**形式。因为 BT 只依赖两个 completion 的 reward **差值** $p^*(y_1\succ y_2\mid x)=\sigma(r^*(x,y_1)-r^*(x,y_2))$，而 $\beta\log Z(x)$ 对同一个 $x$ 是常数，**相减时自动消去**，于是人工偏好概率可以纯粹用 $\pi^*$ 与 $\pi_{\text{ref}}$ 表达（式 6）：

```latex
p^*(y_1\succ y_2\mid x)
=\frac{1}{1+\exp\left(\beta\log\frac{\pi^*(y_2\mid x)}{\pi_{\text{ref}}(y_2\mid x)}-\beta\log\frac{\pi^*(y_1\mid x)}{\pi_{\text{ref}}(y_1\mid x)}\right)}
```

**Step 6 — 最大似然 → DPO 损失**。对参数化策略 $\pi_\theta$ 做与式 (2) reward modeling 完全同构的负对数似然，即得式 (7)。

> **直觉总结**：DPO 是一种"变量替换"。它没有消除配分函数 $Z(x)$，而是证明 $Z(x)$ 在 BT 差分中**必然约掉**，因此可以完全绕过配分函数的估计、也绕过显式 reward model。策略网络同时扮演语言模型和（隐式）reward。

来源：[arXiv:2305.18290v2 §3–§4](https://arxiv.org/html/2305.18290v2)；推导细节见该文 Appendix A.1 / A.2

---

### A.3 DPO 的梯度形式

论文给出的梯度（§4 "What does the DPO update do?"）：

```latex
\nabla_{\theta}\mathcal{L}_{\text{DPO}}(\pi_{\theta};\pi_{\text{ref}})
=
-\beta\,\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}
\bigg[
\underbrace{\sigma\!\big(\hat{r}_{\theta}(x,y_l)-\hat{r}_{\theta}(x,y_w)\big)}_{\text{higher weight when reward estimate is wrong}}
\Big[
\underbrace{\nabla_{\theta}\log\pi(y_w\mid x)}_{\text{increase likelihood of } y_w}
-
\underbrace{\nabla_{\theta}\log\pi(y_l\mid x)}_{\text{decrease likelihood of } y_l}
\Big]
\bigg]
```

**三个要点：**

1. **括号内是"升 chosen、降 rejected"的对比项**，与朴素概率比目标形式相同。真正不同的是外面那个动态权重。

2. **权重项 $\sigma(\hat r_\theta(x,y_l)-\hat r_\theta(x,y_w))$ 是"隐式 reward 判断错了多少"的度量**：
   - 当模型已经把 chosen 排得远高于 rejected（$\hat r_\theta(x,y_w)\gg \hat r_\theta(x,y_l)$），括号内 $= \sigma(\text{大负数}) \to 0$，**该样本的梯度被自动衰减**；
   - 当模型排序错误或只是略微分对时，权重趋近 $\sigma(0)=0.5$ 甚至更大，梯度被放大。

   这正是论文所说的 "dynamic, per-example importance weight that prevents the model degeneration that we find occurs with a naive probability ratio objective"。论文指出去掉这个权重会使语言模型退化（Appendix Table 3）。

3. **为什么"隐式 reward 估计错误时梯度会衰减"的准确表述**：权重反映的是*当前估计已经对了多少*——已经分对的样本权重小（梯度衰减），分错/未拉开差距的样本权重大。所以 DPO 自动把优化预算集中在"还没学会"的样本上。这也是 DPO 的**梯度饱和**来源：一旦 chosen/rejected 的 log-ratio 差被拉得很开，该样本几乎不再贡献梯度，因此**β 过小或训练过久时训练信号会枯竭**，模型可能在少数样本上过拟合（IPO 正是针对这一点，见 §A.5）。

   权重还**被 β 缩放**：整体梯度前面的 $-\beta$ 加上权重内部隐含的 β 依赖，共同构成长度/尺度敏感项，因此 β 不能随意跨数据集迁移（见 §A.4）。

来源：[arXiv:2305.18290v2 §4](https://arxiv.org/html/2305.18290v2)

---

### A.4 β 的物理含义与取值经验

**物理含义**：β 是 RLHF 目标中 KL 惩罚的系数，因此控制策略允许偏离 $\pi_{\text{ref}}$ 的程度。

**极限行为**（TRL 源码注释直接给出了一部分结论）：

- **β → 0**：KL 约束几乎消失，式 (4) 中 $\exp(r/\beta)$ 对 reward 极度敏感，最优策略趋向把所有概率质量放到 argmax reward 上（mode collapse），生成多样性丧失。TRL 的 `dpo_loss` 注释原话：*"We ignore the reference model as beta → 0."*（在 TRL v0.10.1 / v0.12.0 的 `dpo_loss` 中，logits 计算 `pi_logratios - ref_logratios` 在 β→0 时趋向被忽略）。同时 §A.3 的权重项也会退化，训练退化为朴素的"升 chosen 降 rejected"，论文明确说这会导致退化。
- **β → ∞**：KL 约束无限强，$\pi_r \to \pi_{\text{ref}}$，模型完全不动，DPO 退化为空操作。
- 论文对 TRL 文档的表述（v0.13.0 `dpo_trainer.mdx`）：*"beta is the reciprocal of the margin"*（对 hinge loss）以及 *"beta is the reciprocal of the gap between the log-likelihood ratios of the chosen vs the rejected completion pair"*（对 IPO）。因此 β 小 ⇔ 目标 margin/gap 大。

**论文的实际取值**（Appendix B "DPO Implementation Details and Hyperparameters"，原文）：*"Unless noted otherwise, we use a β = 0.1, batch size of 64 and the RMSprop optimizer with a learning rate of 1e-6 by default. We linearly warmup the learning rate from 0 to 1e-6 over 150 steps. For TL;DR summarization, we use β = 0.5, while rest of the parameters remain the same."*
→ 即**主实验 β = 0.1，TL;DR 摘要任务 β = 0.5**。注意原文并未报告一个 "0.1 / 0.5 / 1.0 的 grid search"，网上流传的三值区间是社区惯例而非该论文的 ablation 结论。

来源：[HackerNoon 对 DPO 论文 Appendix B 的全文转载](https://hackernoon.com/lite/dpo-hyperparameters-and-implementation-details)（原文可交叉验证于 arXiv PDF Appendix B）

**TRL 的默认值**（多个版本一致）：

- `DPOConfig.beta` 默认 **0.1**；`DPOConfig.learning_rate` 默认 **1e-6**（覆盖 `TrainingArguments` 的 5e-5 默认值），见 [v0.13.0 dpo_config.py](https://raw.githubusercontent.com/huggingface/trl/v0.13.0/trl/trainer/dpo_config.py) 与 [main 分支 dpo_config.py](https://raw.githubusercontent.com/huggingface/trl/main/trl/trainer/dpo_config.py)。
- TRL 文档对 `beta` 的说明：*"Parameter controlling the deviation from the reference model. Higher β means less deviation from the reference model."*

**工程经验区间**：0.1 是 TRL 与论文的共同缺省，也是绝大多数开源 DPO 配方（Zephyr、UltraFeedback 系列）的取值；0.5 常用于摘要类长文本任务；1.0 及以上在社区中用于抑制长度膨胀/过度偏离，但**未在本调研的原始文献中找到系统性 ablation 支撑**（标记为社区经验，未核实到一手来源）。

---

### A.5 DPO 的已知问题与变体对比

| 方法 | 相对于 DPO 的损失差异 | 解决什么问题 | 论文 |
|---|---|---|---|
| **IPO** | 把 "$-\log\sigma(\cdot)$" 换成平方损失 $\big(\hat r_w-\hat r_l-\frac{1}{2\beta}\big)^2$；β 语义变为 $\tau$，且损失对 completion 的 log-likelihood **取平均**（非求和） | DPO 在确定性偏好下会**过拟合**（把偏好概率推向 0/1 的硬饱和），IPO 用有界目标回归"偏好是概率性的"这一事实 | Azar et al., *A General Theoretical Paradigm to Understand Learning from Human Preferences*, [arXiv:2310.12036](https://arxiv.org/abs/2310.12036) |
| **KTO** | Half-UPML：只对单个 completion 分类（desirable / undesirable），引入前景理论式的效用函数与非对称损失权重 $\lambda_D,\lambda_U$，并保留一个 KL 项（以 $z_0$ 为参考点） | **不需要成对偏好数据**，只需二元好坏标签，标注成本大幅降低 | Ethayarajh et al., *KTO: Model Alignment as Prospect Theoretic Optimization*, [arXiv:2402.01306](https://arxiv.org/abs/2402.01306) |
| **ORPO** | 单体目标：监督 NLL 项 + **odds ratio** 项 $\mathcal{L}_{OR}$，无 reference model、无 β | 把 SFT 与偏好对齐**合并为一个阶段**，省掉 SFT 之后的第二个 reference model，节省显存与流程 | Hong et al., *ORPO: Monolithic Preference Optimization without Reference Model*, [arXiv:2403.07691](https://arxiv.org/abs/2403.07691) |
| **SimPO** | 隐式 reward 改为**长度归一化的平均 log-prob** $r_{\text{SimPO}}(x,y)=\frac{\beta}{|y|}\log\pi_\theta(y\mid x)$，无 $\pi_{\text{ref}}$；并在 BT 中引入 **target reward margin** $\gamma$：$p(y_w\succ y_l\mid x)=\sigma(r_w-r_l-\gamma)$ | ①去掉 reference model（省显存/算力）；②**对齐训练目标与推理时使用的指标**（平均 log-likelihood），并抑制长度膨胀 | Meng et al., *SimPO: Simple Preference Optimization with a Reference-Free Reward*, [arXiv:2405.14734](https://arxiv.org/html/2405.14734v3) |

**SimPO 的核心动机（论文原文要点）**：DPO 的隐式 reward 需要 reference model 且 *"creates a mismatch between the reward optimized in training and the log-likelihood optimized during inference"*；论文实测 **仅约 50% 的训练三元组满足** $r(x,y_w)>r(x,y_l)$ 蕴含 $p_\theta(y_w\mid x)>p_\theta(y_l\mid x)$。SimPO 的损失为：

```latex
\mathcal{L}_{\text{SimPO}}(\pi_{\theta})
=-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}
\left[
\log\sigma\!\left(
\frac{\beta}{|y_w|}\log\pi_{\theta}(y_w\mid x)
-\frac{\beta}{|y_l|}\log\pi_{\theta}(y_l\mid x)
-\gamma
\right)
\right]
```

**关于 SimPO 推荐超参的说明**：论文 §2.3 明确说明 $\gamma>0$ 是 target reward margin、β 是"控制 reward 差缩放"的常数；论文正文宣称 β 在 DPO / SimPO 中含义不同（SimPO 的 β 只是 sigmoid 的缩放因子）。**我在本次调研中未能取到 Appendix B "Method-specific training hyperparameters" 的具体数值表**（arxiv HTML 正文在 §3 之后被截断），因此**"β=2.0, γ=0.5"这一常见说法标记为未核实**。

**关键版本事实（与很多教程的说法相反）**：**TRL 的 `DPOTrainer` 从未支持 `loss_type="simpo"`，也从未支持 `loss_type="orpo"`。** 逐版本核实 `DPOConfig.loss_type` 的 `Literal`/choices 列表：

| TRL 版本 | `loss_type` 支持值 | 证据 |
|---|---|---|
| v0.9.6 | sigmoid, hinge, ipo, bco_pair, sppo_hard, nca_pair, robust, aot, aot_pair, exo_pair | [源码](https://raw.githubusercontent.com/huggingface/trl/v0.9.6/trl/trainer/dpo_config.py) |
| v0.10.1 | 同上 + apo_zero, apo_down（新增） | [源码](https://raw.githubusercontent.com/huggingface/trl/v0.10.1/trl/trainer/dpo_config.py) |
| v0.11.0 | 同 v0.10.1 | [源码](https://raw.githubusercontent.com/huggingface/trl/v0.11.0/trl/trainer/dpo_config.py) |
| v0.12.0 | 同 v0.10.1（新增 `use_weighting`，但 WPO 不是 `loss_type` 取值） | [源码](https://raw.githubusercontent.com/huggingface/trl/v0.12.0/trl/trainer/dpo_config.py) |
| v0.13.0 | 同上 + discopop（新增） | [源码](https://raw.githubusercontent.com/huggingface/trl/v0.13.0/trl/trainer/dpo_config.py) |
| v0.25.0 / v0.27.0 | 新增 `sft`，且 `loss_type` 从 `Literal` 变为 `str or list[str]`（支持 MPO 式多损失加权，配 `loss_weights`）；`aot_pair` 语义在 v0.28 更名为 `aot_unpaired` | [v0.25.0](https://raw.githubusercontent.com/huggingface/trl/v0.25.0/trl/trainer/dpo_config.py)、[v0.27.0](https://raw.githubusercontent.com/huggingface/trl/v0.27.0/trl/trainer/dpo_config.py) |
| v0.28.0 | 同上（`aot_pair` 弃用→`aot_unpaired`；一堆参数弃用，见 §B.7） | [源码](https://raw.githubusercontent.com/huggingface/trl/v0.28.0/trl/trainer/dpo_config.py) |
| main / 1.x | `['sigmoid','hinge','ipo','exo_pair','nca_pair','robust','bco_pair','sppo_hard','aot','aot_unpaired','apo_zero','apo_down','discopop','sft','sigmoid_norm']` | [main dpo_config.py](https://raw.githubusercontent.com/huggingface/trl/main/trl/trainer/dpo_config.py) |

→ **SimPO 在 TRL 中的实现入口是 `CPOTrainer` 家族的独立实现，而非 `DPOTrainer.loss_type`；ORPO 在 TRL 中有独立的 `ORPOTrainer`/`ORPOConfig`（v0.13.0 仓库内已存在 `trl/trainer/orpo_config.py` 与 `orpo_trainer.py`，见 [v0.13.0 仓库树](https://api.github.com/repos/huggingface/trl/git/trees/v0.13.0?recursive=1)）。若文档中写"TRL 支持 DPO loss_type=simpo/orpo"，该说法错误。**

另外确认一个已删除项：`kto_pair` 曾存在于 v0.8.x 的 `loss_type` 中，v0.9.6 起在 `__post_init__` 中直接抛错：*"Support for kto_pair has been removed in DPOTrainer. Please use KTOTrainer."*（见 [v0.9.6 源码](https://raw.githubusercontent.com/huggingface/trl/v0.9.6/trl/trainer/dpo_config.py)）。

---

### A.6 reference model 的作用

**为什么必须有 $\pi_{\text{ref}}$**：

1. **数值基准**：DPO 学的是 *相对* log-prob 变化 $\log\frac{\pi_\theta}{\pi_{\text{ref}}}$。没有 $\pi_{\text{ref}}$ 就没有基准，无法区分"chosen 的绝对概率高是因为它本来就容易生成"还是"因为模型真的更偏好它"。
2. **KL 正则的载体**：β 的物理意义（偏离参考策略的代价）只有存在 $\pi_{\text{ref}}$ 时才成立。所有 reward 都是相对于 $\pi_{\text{ref}}$ 的增量。
3. **防止分布漂移与 mode collapse**：如 §A.2 Step 2 所述，KL 项把策略约束在 reward 估计可靠的分布邻域内。
4. **抑制长度/风格漂移**：去掉 reference 后，模型可以直接通过改变整体 log-prob 尺度（例如系统性拉长输出）来获得 reward，这正是 SimPO 论文用长度归一化 reward 要解决的问题。

**不加会怎样**：把 $\pi_{\text{ref}}$ 置为均匀分布（TRL 的 `reference_free=True` 就是"隐式使用给所有回答等概率的 reference"）等价于删掉 KL 项，DPO 退化为朴素的"提高 chosen 绝对似然、降低 rejected 绝对似然"，论文明确说这会造成模型退化。TRL 在 v0.28.0 已经**弃用 `reference_free`**，并建议"若你想要无 reference 的目标，请改用 `CPOTrainer`"（即 CPO/SimPO 家族）。

**LoRA 场景如何省掉第二个模型**：见 §B.8。

来源：[arXiv:2305.18290v2 §3–§4](https://arxiv.org/html/2305.18290v2)；[TRL v0.28.0 dpo_config.py](https://raw.githubusercontent.com/huggingface/trl/v0.28.0/trl/trainer/dpo_config.py)

---

## B. TRL 的 DPOTrainer 实际 API 与版本兼容性

### B.7 版本演进与破坏性变更（逐条核实）

#### (a) 旧位置参数 vs 新版关键参数

**两个时代的分水岭：**

**旧式（≤ v0.11.x）** —— `beta` 等参数直接是 `DPOTrainer.__init__` 的位置参数。v0.10.1 的签名为（实测源码）：

```python
def __init__(
    self,
    model=None,
    ref_model=None,
    beta: float = 0.1,
    label_smoothing: float = 0,
    loss_type: Optional[str] = None,
    args: Optional[DPOConfig] = None,
    data_collator: Optional[DataCollator] = None,
    label_pad_token_id: int = -100,
    padding_value: Optional[int] = None,
    truncation_mode: str = "keep_end",
    train_dataset: Optional[Dataset] = None,
    eval_dataset=None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,   # ← 旧名
    ...
    max_length=None, max_prompt_length=None, max_target_length=None,
    peft_config=None, ...
)
```
来源：[v0.10.1 dpo_trainer.py](https://raw.githubusercontent.com/huggingface/trl/v0.10.1/trl/trainer/dpo_trainer.py)

所以"`DPOTrainer(model, ref_model, beta, train_dataset, args=...)`"这种**位置参数写法在 v0.10.x 及以前是合法的**；`beta` 是第 3 个位置参数。

**v0.12.0** —— 位置参数仍在，但被打上弃用装饰器，并给出明确迁移目标：

```python
@_deprecate_arguments(
    version="0.13.0",
    deprecated_args=["beta", "label_smoothing", "loss_type", "label_pad_token_id",
                     "padding_value", "truncation_mode", "max_length", "max_prompt_length",
                     "max_target_length", "is_encoder_decoder", "disable_dropout",
                     "generate_during_eval", "precompute_ref_log_probs", "dataset_num_proc",
                     "model_init_kwargs", "ref_model_init_kwargs", "model_adapter_name",
                     "ref_adapter_name", "reference_free", "force_use_ref_model"],
    custom_message="Deprecated positional argument(s) used in DPOTrainer, please use the DPOConfig to set these arguments instead.",
)
@deprecate_kwarg("tokenizer", new_name="processing_class", version="0.14.0", raise_if_both_names=True)
def __init__(self, model=None, ref_model=None, beta: float = 0.1, ..., processing_class=None, ...)
```
来源：[v0.12.0 dpo_trainer.py](https://raw.githubusercontent.com/huggingface/trl/v0.12.0/trl/trainer/dpo_trainer.py)

**新版（v1.x / main）** —— 这些参数**已从 `__init__` 彻底移除**，只能通过 `DPOConfig` 传入。main 分支签名：

```python
def __init__(self, model=None, ref_model=None, args=None, data_collator=None,
             train_dataset=None, eval_dataset=None,
             processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
             ...)
```
来源：[main dpo_trainer.py](https://raw.githubusercontent.com/huggingface/trl/main/trl/trainer/dpo_trainer.py)

**结论**：推荐写法从 `DPOTrainer(model, ref_model, beta, train_dataset, args=...)` 变为 `DPOTrainer(model=model, args=DPOConfig(beta=...), train_dataset=..., processing_class=tokenizer)`。

#### (b) `tokenizer` → `processing_class`

**精确时间线（已核实）**：

| 版本 | 状态 |
|---|---|
| ≤ v0.11.x | 参数名为 `tokenizer` |
| **v0.12.0** | **新增 `processing_class`，`tokenizer` 成为弃用别名**（`@deprecate_kwarg("tokenizer", new_name="processing_class", version="0.14.0")`）。Release notes 原文：*"Rename trainer arg `tokenizer` to `processing_class` … `tokenizer` is still supported for `SFTTrainer` and `DPOTrainer` but deprecated and will be removed in the next release."* PR [#2162](https://github.com/huggingface/trl/pull/2162) |
| v0.13.0 | 文档与快速开始示例已统一使用 `processing_class` |
| **v0.14.0** | **`tokenizer` 别名被移除**（即在 v0.14.0 上写 `tokenizer=` 会直接报错） |

来源：[TRL v0.12.0 release notes](https://api.github.com/repos/huggingface/trl/releases/tags/v0.12.0)、[v0.12.0 源码](https://raw.githubusercontent.com/huggingface/trl/v0.12.0/trl/trainer/dpo_trainer.py)

> 注意版本语义：`deprecate_kwarg(..., version="0.14.0")` 表示"在 0.14.0 移除"。v0.12.0 release notes 中"will be removed in the next release"的措辞不精确；以源码装饰器为准。

#### (c) `DPOConfig` 各参数的存亡与改名

| 参数 | 状态 | 关键版本 |
|---|---|---|
| `beta` | 一直存在，默认 **0.1** | v0.9.6 → main 均确认 |
| `max_length` | 存在。v0.9.6–v0.13.0 默认 `None`；**v0.14.0 起默认 `1024`**，语义为"整条序列 prompt+completion 的最大长度" | [v0.14.0](https://raw.githubusercontent.com/huggingface/trl/v0.14.0/trl/trainer/dpo_config.py) |
| `max_prompt_length` | **v0.9.6 起存在，默认 `512`；v0.28.0 弃用；v0.29.0 起从 `DPOConfig` 中移除。** 弃用文案原文：*"We recommend filtering overlong prompts from your dataset before passing it to the trainer instead of using this parameter."* | 见下方专项 |
| `max_target_length` | v0.9.6 存在；**v0.11.0 起弃用**，注释 `# deprecated in favor of max_completion_length`，`__post_init__` 中发出 `FutureWarning` 并回填 `max_completion_length`；v0.14.0 起该参数在 dataclass 中已消失 | [v0.11.0](https://raw.githubusercontent.com/huggingface/trl/v0.11.0/trl/trainer/dpo_config.py)、[v0.14.0](https://raw.githubusercontent.com/huggingface/trl/v0.14.0/trl/trainer/dpo_config.py) |
| `max_completion_length` | v0.11.0 引入（承接 `max_target_length`）；v0.28.0 弃用；v0.29.0 移除 | [v0.14.0](https://raw.githubusercontent.com/huggingface/trl/v0.14.0/trl/trainer/dpo_config.py)、[v0.28.0](https://raw.githubusercontent.com/huggingface/trl/v0.28.0/trl/trainer/dpo_config.py) |
| `loss_type` | 一直存在；类型从 `Literal[...]`（单值）变为 `str or list[str]`（多损失加权，v0.25.0/0.27.0）再到 **v1.x 的 `list[str]`（含 `loss_weights`）** | [v0.25.0](https://raw.githubusercontent.com/huggingface/trl/v0.25.0/trl/trainer/dpo_config.py)、[main](https://raw.githubusercontent.com/huggingface/trl/main/trl/trainer/dpo_config.py) |
| `label_smoothing` | 一直存在，默认 `0.0`，取值区间 `[0.0, 0.5)` | 各版本一致 |
| `use_weighting` | **v0.12.0 新增**（WPO，PR [#2141](https://github.com/huggingface/trl/pull/2141)），默认 `False` | [v0.12.0 release notes](https://api.github.com/repos/huggingface/trl/releases/tags/v0.12.0) |
| `tools` | **v0.14.0 新增**，默认 `None`；v0.28.0 弃用并说明"0.29 起本参数将被忽略，tools 应通过数据集提供"；main 分支已移除 | [v0.14.0](https://raw.githubusercontent.com/huggingface/trl/v0.14.0/trl/trainer/dpo_config.py)、[v0.28.0](https://raw.githubusercontent.com/huggingface/trl/v0.28.0/trl/trainer/dpo_config.py) |
| `truncation_mode` | 一直存在。默认 `"keep_end"`（v0.9.6–v0.29.x）→ **main/v1.x 默认 `"keep_start"`，且 `"keep_end"` 本身在 v1.x 被弃用、计划 v2.0.0 移除** | [main](https://raw.githubusercontent.com/huggingface/trl/main/trl/trainer/dpo_config.py) |
| `reference_free` | v0.28.0 弃用（建议改用 `CPOTrainer`），v0.29.0 移除 | [v0.28.0](https://raw.githubusercontent.com/huggingface/trl/v0.28.0/trl/trainer/dpo_config.py) |
| `rpo_alpha` | v0.28.0 弃用（等价于在 `loss_type` 里加 `"sft"` 并在 `loss_weights` 设权重），v0.29.0 移除 | [v0.28.0](https://raw.githubusercontent.com/huggingface/trl/v0.28.0/trl/trainer/dpo_config.py) |

#### (d) `max_prompt_length` 弃用专项（v0.28.0 → v0.29.0）

已验证的完整链路：

- **v0.27.0**：`max_prompt_length: int | None = field(default=512)`，正常可用。[源码](https://raw.githubusercontent.com/huggingface/trl/v0.27.0/trl/trainer/dpo_config.py)
- **v0.28.0**：参数保留但**默认值改为哨兵 `-1`**，`__post_init__` 中若用户显式设置则发 `FutureWarning`（"deprecated and will be removed in version 0.29.0. We recommend filtering out overlong prompts from your dataset before passing it to the trainer instead of using this parameter."），否则回填旧默认 512。同一 PR 还同时弃用了 `max_completion_length`、`reference_free`、`rpo_alpha`、`tools`、`use_logits_to_keep`、`label_pad_token_id`、`force_use_ref_model`、`model_adapter_name`、`ref_adapter_name`、`ref_model_init_kwargs`、`base_model_attribute_name`、`generate_during_eval`。[源码](https://raw.githubusercontent.com/huggingface/trl/v0.28.0/trl/trainer/dpo_config.py)、弃用 PR [#4969](https://github.com/huggingface/trl/pull/4969)（merged 2026-02-05，标题 "Deprecate parameters in `DPOConfig`"，PR 描述：*"Most importantly, `max_prompt_length` is now None by default, and `force_use_ref_model` is enabled by default"*）
- **v0.29.0**：`max_prompt_length` **已从 `DPOConfig` 完全消失**。截断只剩 `max_length` 一个旋钮，作用于整条序列（prompt+completion）。[源码](https://raw.githubusercontent.com/huggingface/trl/v0.29.0/trl/trainer/dpo_config.py)
- **main / 1.x**：维持 `max_length`（默认 `1024`）单旋钮；`DPOConfig` 已改继承 `_BaseConfig` 而非 `TrainingArguments`。

> **所以问题"`max_prompt_length` 是否被弃用、何时换成 `max_length` 同时管 prompt+completion"的答案是**：`max_prompt_length` 于 **TRL v0.28.0 弃用、v0.29.0 移除**；从 **v0.14.0** 起 `max_length` 的文档语义就已经是 "Maximum length of the full sequence (prompt + completion)"，而 `max_prompt_length` 一直是"单独的 prompt 上限"。二者的合流发生在 v0.28/v0.29。

#### (e) 版本一览（关键节点）

| TRL | 关键变化 |
|---|---|
| 0.8.x | `DPOTrainer` 内部有 `DPODataCollatorWithPadding`；`loss_type` 含 `kto_pair` |
| 0.9.6 | `kto_pair` 移除并抛错；`KTOTrainer` 独立 |
| 0.10.x | 位置参数仍可用；`DPOConfig` 已是 `args` 入口 |
| 0.11.x | `max_target_length` 弃用 → `max_completion_length` |
| 0.12.0 | **`tokenizer` → `processing_class`**；`use_weighting`（WPO）；positional args 标记弃用（指向 0.13.0）；会话式数据集直通支持 |
| 0.13.0 | 位置参数弃用到期；会话式数据集支持完成；`discopop` 加入 `loss_type` |
| 0.14.0 | `tokenizer` 别名移除；`max_length` 默认变 1024、`max_prompt_length` 默认 512；**新增 `DPOConfig.tools`**；引入 `PreferenceCollator` 与新 tokenize 流程（[v0.14.0 源码](https://raw.githubusercontent.com/huggingface/trl/v0.14.0/trl/trainer/dpo_config.py)） |
| 0.25–0.27 | `loss_type` 支持 `list[str]` + `loss_weights`（MPO 风格）；新增 `sft` 损失；`ld_alpha`（LD-DPO）；`use_logits_to_keep` |
| **0.28.0** | **大批参数弃用**（含 `max_prompt_length`）；`aot_pair` → `aot_unpaired`；`f_divergence_type` 改字符串 |
| 0.29.0 | 弃用项全部移除；参数面收敛为 `max_length` 单旋钮 |
| **1.0.0**（2026-03-31） | v0→v1 迁移指南发布（[MIGRATION.md](https://raw.githubusercontent.com/huggingface/trl/main/MIGRATION.md)）；DPO/KTO/GRPO 明确列为稳定 trainer。迁移影响很小：v0.29 之后的变更主要是 `SFTConfig.packing` 的 `"bfd-requeue"`→`"bfd_split"`、GRPO/RLOO 的 `vllm_mode` 默认改 `"colocate"`、以及**训练器不再自动剥离嵌套列中的 `None` 值**（需手动 `dataset.with_transform(remove_none_values)`） |
| 1.14.x | 最新版 1.14.1；DPO 改走 chunked log-prob 路径 |

---

### B.8 `ref_model=None` + PEFT/LoRA 时 TRL 如何构造 reference

**结论：是"临时关闭 adapter"，不是复制模型。**

v0.12.0 / v0.10.1 源码中的核心逻辑（原文引用）：

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
        if self.ref_adapter_name:
            self.model.set_adapter(self.ref_adapter_name)
        yield
        if self.ref_adapter_name:
            self.model.set_adapter(self.model_adapter_name or "default")

def compute_ref_log_probs(self, batch):
    """Computes log probabilities of the reference model for a single padded batch of a DPO specific dataset."""
    with torch.no_grad(), ...:
        if self.ref_model is None:
            with self.null_ref_context():
                ref_model_output = self.concatenated_forward(self.model, batch)
        else:
            ref_model_output = self.concatenated_forward(self.ref_model, batch)
```

来源：[v0.12.0 dpo_trainer.py](https://raw.githubusercontent.com/huggingface/trl/v0.12.0/trl/trainer/dpo_trainer.py)（`null_ref_context` / `compute_ref_log_probs`）；[v0.10.1 dpo_trainer.py](https://raw.githubusercontent.com/huggingface/trl/v0.10.1/trl/trainer/dpo_trainer.py)（`get_batch_loss_metrics` 内联同逻辑）

**要点与注意事项**：

1. **两种模式**：
   - 未传 `ref_model` + 是 PEFT 模型 → `self.ref_model = None`，用 `disable_adapter()` 关掉 LoRA，此时模型退化为基础模型（base model），也就是 $\pi_{\text{ref}}$。
   - 传了 `ref_model` 且 PEFT → 除非 `force_use_ref_model=True`，否则**直接抛错**：*"You passed both a ref_model and a peft_config. For training PEFT adapters with DPO there is no need to pass a reference model. Please pass `ref_model=None` ... or pass a ref_model with `force_use_ref_model=True`."*
2. **`model` 与 `ref_model` 不能是同一对象**：*"`model` and `ref_model` cannot be the same object. If you want `ref_model` to be the same as `model`, you must make a copy of it, or `None` if you use peft."*
3. **`force_use_ref_model`**（v0.28 弃用）：当你想用**不同的模型**作 reference 时才需要它。
4. **语义陷阱（最重要的一条）**：`disable_adapter()` 给出的是**基础模型**。因此这套做法在语义上假设"SFT 阶段已经 merge 进 base 权重"。如果 LoRA 是在**未经 SFT 的 base 模型**上直接训练，则 $\pi_{\text{ref}}$ = 原始 base 模型，DPO 的 KL 锚点是 base 而不是"你 SFT 过的模型"。TRL 官方文档对此给出了**三种选项**（v0.13.0 `dpo_trainer.mdx` 的 "Reference model considerations with PEFT" 一节，原文）：
   > 1. Simply create two instances of the model, each loading your adapter - works fine but is very inefficient.
   > 2. Merge the adapter into the base model, create another adapter on top, then leave the `ref_model` param null, in which case DPOTrainer will unload the adapter for reference inference - efficient, but has potential downsides discussed below.
   > 3. Load the adapter twice with different names, then use `set_adapter` during training to swap between the adapter being DPO'd and the reference adapter - slightly less efficient compared to 2 (~adapter size VRAM overhead), but avoids the pitfalls.

   官方还给出了选项 3 的代码骨架：用 `PeftModel.from_pretrained(..., adapter_name="train")` + `model.load_adapter(..., adapter_name="reference")`，然后 `DPOConfig(model_adapter_name="train", ref_adapter_name="reference")`。选项 2 的缺点是 QLoRA 场景下需要先反量化再 merge，之后要么重新量化要么吃显存。
5. **`precompute_ref_log_probs`**：把 reference log-prob 在训练前一次性算好并写回数据集，训练时 `ref_model` 无需常驻显存。注意在 v0.12.0 的实现里，只有 `self.is_peft_model or args.precompute_ref_log_probs` 为真时 `ref_model` 才会被置 `None`，且 `get_batch_loss_metrics` 中优先从 batch 里读 `ref_chosen_logps`/`ref_rejected_logps`。
6. **v1.x 的变化**：main 分支改用 `use_adapter(model, adapter_name="ref" if "ref" in model.peft_config else None)`，注释明确区分了两种情况——*"New adapter: disabling adapters yields the base model. Re-training an existing adapter: an initial copy is loaded under the name 'ref'."* 同时 `sync_ref_model=True` 官方注明**与 PEFT 或 `precompute_ref_log_probs=True` 不兼容**。

来源：[TRL v0.13.0 dpo_trainer.mdx](https://raw.githubusercontent.com/huggingface/trl/v0.13.0/docs/source/dpo_trainer.mdx)、[main dpo_trainer.py](https://raw.githubusercontent.com/huggingface/trl/main/trl/trainer/dpo_trainer.py)、[main dpo_config.py](https://raw.githubusercontent.com/huggingface/trl/main/trl/trainer/dpo_config.py)

---

### B.9 数据集格式

TRL 的 `DPOTrainer` 期望 **[Preference 类型](https://raw.githubusercontent.com/huggingface/trl/v0.13.0/docs/source/dataset_formats.mdx) 的数据集**，支持 **standard（纯字符串）** 与 **conversational（消息列表）** 两种 format。TRL 官方文档明确推荐**显式 prompt**（explicit prompt）；若省略 `prompt` 列，trainer 会自动从 `chosen`/`rejected` 中抽取 prompt（implicit prompt，不推荐）。

**标准格式：**

```python
# 显式 prompt（推荐）
{"prompt": "The sky is", "chosen": " blue.", "rejected": " green."}
# 隐式 prompt
{"chosen": "The sky is blue.", "rejected": "The sky is green."}
```

**会话式格式（conversational）：**

```python
{
  "prompt":   [{"role": "user", "content": "What color is the sky?"}],
  "chosen":   [{"role": "assistant", "content": "It is blue."}],
  "rejected": [{"role": "assistant", "content": "It is green."}],
}
```

**`messages` 格式支持（重要澄清）**：`messages` 是 **[Language modeling 类型](https://raw.githubusercontent.com/huggingface/trl/v0.13.0/docs/source/dataset_formats.mdx) 的字段名（对应 `SFTTrainer`），不是 Preference 类型的字段名。** `DPOTrainer` 用的是 `prompt` + `chosen` + `rejected` 三列，其中每列的值可以是"消息列表"（即 conversational 形式）。**把 `{"messages":[...]}` 直接喂给 `DPOTrainer` 不是受支持的用法。**

**会话式数据集的支持时间**：**v0.12.0** 引入 —— release notes 原文：*"You can now use conversational datasets directly, without needing to apply a chat template beforehand, for the following trainers: BCOTrainer (#2107), CPOTrainer (#2144), **DPOTrainer (#2131)**, KTOTrainer (#2248), ORPOTrainer (#2184)"*。在此之前必须自己用 `apply_chat_template` 预处理。v0.12.0 同时重构了 DPO 数据处理（PR #2209）。来源：[v0.12.0 release notes](https://api.github.com/repos/huggingface/trl/releases/tags/v0.12.0)

**关于 `tools` 列**：`DPOConfig.tools` 是 v0.14.0 引入的**全局**工具列表（应用于整个数据集），v0.28.0 弃用并在 main 中移除。**在 v1.x / main 中，tools 必须通过数据集提供**（每行自带 tools schema）。这与 GRPO 的 agent training（PR [#4300](https://github.com/huggingface/trl/pull/4300)，v0.26.0）路径一致。**具体到 DPO 在 v1.x 中从数据集的哪个列名读取 tools，我未在本次调研中核实到源码级证据，标记为"未核实"。**

> **版本现状提醒**：TRL 官方文档在 `main` 分支上引用的是本项目**无法访问**的 `huggingface.co/docs/trl/...`；上文的 `dataset_formats.mdx` 内容来自 **v0.13.0 tag 的原始文件**（`raw.githubusercontent.com`），该文件在 v0.13.0 时是权威且与本项目目标版本一致的。

**最小可运行示例（TRL v0.13.0 风格，含 config 与 dataset）**

```python
# train_dpo.py
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="bfloat16")

# conversational preference 数据集（推荐：prompt / chosen / rejected 三列，值为 message list）
dataset = Dataset.from_list([
    {
        "prompt":   [{"role": "user", "content": "What is 2+2? Answer with a tool call."}],
        "chosen":   [{"role": "assistant", "content": "4"}],
        "rejected": [{"role": "assistant", "content": "The answer might be 5."}],
    },
])

args = DPOConfig(
    output_dir="out-dpo",
    beta=0.1,
    loss_type="sigmoid",
    learning_rate=5e-6,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    max_length=2048,
    max_prompt_length=1024,
    truncation_mode="keep_end",
    gradient_checkpointing=True,
    bf16=True,
    logging_steps=10,
    save_steps=200,
    report_to="none",
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,                    # PEFT → 用 disable_adapter 构造 reference
    args=args,
    train_dataset=dataset,
    processing_class=tokenizer,        # v0.12.0+ 的新名（v0.14.0 起 tokenizer= 不可用）
    peft_config=LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    ),
)
trainer.train()
```

（该示例的骨架与超参名逐项对照 [v0.13.0 dpo_trainer.mdx 的 Quick start 与 Example script](https://raw.githubusercontent.com/huggingface/trl/v0.13.0/docs/source/dpo_trainer.mdx) 以及 [v0.13.0 dpo_config.py](https://raw.githubusercontent.com/huggingface/trl/v0.13.0/trl/trainer/dpo_config.py)。LoRA 超参值来自 §D.14 的社区经验，非文档默认。）

---

### B.10 多轮对话 / 工具调用偏好训练的数据组织方式

**推荐结构（每行一条偏好样本）：**

| 列 | 内容 | 说明 |
|---|---|---|
| `prompt` | `[{"role":"system","content": <含 tools schema 的说明>}, {"role":"user",...}, {"role":"assistant","content":"<tool_call>...</tool_call>"}, {"role":"tool"/"user", "content": <工具返回>}, {"role":"user", ...}]` | 完整历史多轮。**chosen 与 rejected 必须共享同一个 prompt 前缀**，只有最后的 assistant 回复不同 |
| `chosen` | `[{"role":"assistant","content":"<tool_call>{...}</tool_call>"}]` | **单轮 assistant 回复**（只要这一条） |
| `rejected` | `[{"role":"assistant","content": <错误工具名/错误参数/幻觉参数> }]` | 同上，单轮 |

**为什么 chosen/rejected 只放单轮**：DPO 的 loss 是在 `prompt + completion` 拼接后、只对 completion 段求 log-prob。若把整段多轮对话都塞进 `chosen`/`rejected`，被优化的就不再是"这一步的工具调用决策"。**构造步骤**：先用真实或合成的 agent 轨迹跑出一整条对话；对"本轮该调用什么工具"这一步采样 N 个候选回复；用规则（工具名是否匹配、参数 JSON schema 是否合法、沙箱执行是否成功）或更强模型打分，取最优为 chosen、最差为 rejected；其余轮次（历史 + 环境返回值）原样冻结进 `prompt`。

**工程注意事项**：

1. **`prompt` 内必须自带 tools schema**。在 v1.x 中 `DPOConfig.tools` 已移除，若你的 chat template 依赖 `tools=` 参数注入工具定义，则必须在数据集层面把工具定义显式写进 system 消息（或使用数据集自带的 tools 列，见 §B.9 的未核实标注）。
2. **chosen 与 rejected 的 prompt 前缀必须逐 token 对齐**。TRL v0.10.x 的实现会主动校验：*"Chosen and rejected prompt_input_ids might only differ on the last token due to tokenizer merge ops."*（`_adjust_prompt_length`）。在 v0.12.0+ 的新 tokenize 流程中若前缀不一致，会抛错：*"This may be due to unexpected tokenizer behavior, whitespace issues, or special token handling. Verify that the tokenizer is processing text consistently."*（见 [v0.12.0 `_prepare_dataset`](https://raw.githubusercontent.com/huggingface/trl/v0.12.0/trl/trainer/dpo_trainer.py)）。所以**不要**把整个多轮对话复制进 chosen/rejected 再各自裁剪。
3. **避免把"同一语义的不同历史"当成 chosen/rejected 的差异来源**——那会让模型学到"历史风格"而非"工具选择"。
4. **长度控制**：多轮 + tools schema 很容易超长。用 `max_length`（整序列）+ `truncation_mode`。注意 v0.28.0 的官方建议是**在进 trainer 前先过滤掉超长 prompt**，而不是依赖 `max_prompt_length`——因为 `max_prompt_length` 在 v0.29.0 已不存在。
5. **`prompt` 的最后一轮应结束于 assistant 起始标记**。若用 conversational 格式，TRL 的 `apply_chat_template` 会处理这件事（官方建议用 `trl.apply_chat_template` 而不是直接调 `tokenizer.apply_chat_template`，因为前者会正确处理非 language-modeling 任务的模板细节，例如避免把 system prompt 放到对话中间）。

来源：[TRL v0.13.0 dataset_formats.mdx](https://raw.githubusercontent.com/huggingface/trl/v0.13.0/docs/source/dataset_formats.mdx)、[v0.12.0 dpo_trainer.py](https://raw.githubusercontent.com/huggingface/trl/v0.12.0/trl/trainer/dpo_trainer.py)

---

## C. pass^k 指标（τ-bench）

### C.11 pass^k 的精确定义

Yao et al. 2024, *τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains*, [arXiv:2406.12045](https://arxiv.org/html/2406.12045v1) 式：

```latex
\text{pass}^k=\mathbb{E}_{\text{task}}\left[\binom{c}{k}\middle/\binom{n}{k}\right],
\qquad
\text{pass@k}=1-\mathbb{E}_{\text{task}}\left[\binom{n-c}{k}\middle/\binom{n}{k}\right]
```

其中 $n$ = 每个任务重复运行的试验次数（i.i.d. trials），$c$ = 其中成功的次数（reward $r=1$）。

**文字定义（论文原文）**：

- **pass@k**：*"the chance that **at least one** out of k i.i.d. task trials is successful, which captures the trend of agents enabling **discovery** of solutions with scaling of inference-time compute"*。
- **pass^k**：*"defined as the chance that **all k** i.i.d. task trials are successful, averaged across tasks"*。

**与无偏 pass@k 的本质区别**：

| | pass@k | pass^k |
|---|---|---|
| 语义 | 至少成功 1 次 | **k 次全部成功** |
| 关心 | 能力上限 / 探索能力（"能不能做到"） | **稳定性与可靠性**（"能不能一直做到"） |
| 随 k 变化 | **单调递增**，k 越大越高 | **单调递减**，k 越大越低 |
| 适用场景 | 有可靠验证器的任务（代码单元测试、数学） | 真实世界 agent 任务（客服、订票），要求一致性与规则遵循 |

**为什么用 pass^k 衡量多轮 agent 的稳定性**（论文原文）：*"In our case, for the same task, the user prompt and database transitions are the same, with just the LM sampling of the user and agent messages generating sufficient stochasticity. Thus, pass^k can capture the reliability of the agent at handling variations in conversations with the same underlying semantics while adhering to the domain policies and rules."*

即：τ-bench 中数据库与任务标注是**确定**的，唯一的随机性来自 user simulator 与 agent 自身的采样。因此同一个任务重跑 k 次，语义不变但对话路径不同——pass^k 恰好度量"语义等价但表达不同"的对话扰动下，agent 是否稳定地给出唯一正确的终态。

**副指标**：论文默认报告 `pass^1 = pass@1 = E[r] = E[c/n]`，作为比较 agents 的主指标。此外 τ-bench 的 reward 定义为 $r=r_{\text{action}}\times r_{\text{output}}\in\{0,1\}$：$r_{\text{action}}$ 要求**最终数据库状态与唯一 ground truth 完全一致**，$r_{\text{output}}$ 要求 agent 对用户的回复包含所有必要信息（作为子串）。

**参考实现（官方代码）**，`display_metrics()` 逐字为：

```python
from math import comb
...
num_trials = len(set([r.trial for r in results]))
average_reward = sum(rewards) / len(rewards)          # = pass^1
c_per_task_id = {}
for result in results:
    c_per_task_id[result.task_id] = c_per_task_id.get(result.task_id, 0) + (1 if is_successful(result.reward) else 0)

pass_hat_ks = {}
for k in range(1, num_trials + 1):
    sum_task_pass_hat_k = 0
    for c in c_per_task_id.values():
        sum_task_pass_hat_k += comb(c, k) / comb(num_trials, k)
    pass_hat_ks[k] = sum_task_pass_hat_k / len(c_per_task_id)
```

来源：[tau_bench/run.py](https://raw.githubusercontent.com/sierra-research/tau-bench/main/tau_bench/run.py)

> 注意这里的实现细节：**每个任务都跑满 `num_trials` 次**，然后用组合数 $\binom{c}{k}/\binom{n}{k}$ 一次算出所有 k 的 pass^k。这与 Codex 的 pass@k 估计器结构相同，只是取了"全部成功"而非"至少一次成功"。

---

### C.12 pass@k 的无偏估计公式（Chen et al. 2021, Codex）

```latex
\text{pass@}k \;=\; \mathbb{E}_{\text{Problem}}\left[\,1-\frac{\dbinom{n-c}{k}}{\dbinom{n}{k}}\,\right]
```

其中 $n$ = 每题生成的样本数，$c$ = 通过单元测试的样本数，$k$ = 允许尝试的次数。

**为什么需要这个估计器**：朴素做法是对每题采样恰好 k 个样本、看是否有成功的，方差很大。Codex 的做法是**每题采样 $n \ge k$ 个样本**，然后用上式作为无偏估计——它等价于"从 n 个样本中不放回地随机抽 k 个，至少 1 个成功的概率"。这一点与 τ-bench 的实现完全一致：τ-bench 也是跑满 $n$ 次 trial，再把所有 $k \le n$ 的 pass^k / pass@k 一次算出来。

**与 pass^k 的关系与区别**：

- **数学结构相同**，都是"从 $n$ 个结果中抽 $k$ 个"的超几何分布概率；
- **分子不同**：pass@k 关心"抽到的 k 个里至少 1 个成功"→ 用 $\binom{n-c}{k}$（全失败组合数）；pass^k 关心"抽到的 k 个全部成功"→ 用 $\binom{c}{k}$；
- **单调性相反**：pass@k 随 k 递增（试得越多越可能成功）、pass^k 随 k 递减（要求全部成功，k 越大越难）；
- **解读相反**：pass@k 高说明"有潜力"，pass^k 高说明"可信赖"。τ-bench 论文明确指出前者度量 discovery，后者度量 reliability。

**τ-bench 论文的实测数据**（用于理解量级）：gpt-4o 用 function calling 的 pass^1 约 61%（τ-retail）/ 约 35%（τ-airline）；同一模型 **pass^8 在 τ-retail 上降到约 25%**。论文摘要亦给出 *"pass^8 < 25% in retail"*。

来源：[arXiv:2406.12045v1 §3](https://arxiv.org/html/2406.12045v1)；Codex 估计器原式见 Chen et al. 2021 *Evaluating Large Language Models Trained on Code*（本文的 τ-bench 引用为 [5]）

---

### C.13 τ-bench 评测协议细节

逐项核实（全部来自论文正文与官方代码）：

| 项目 | 设置 | 来源 |
|---|---|---|
| **user simulator 模型** | 论文 v1 使用 **`gpt-4-0613`**（原文：*"We use a language model (gpt-4-0613) to simulate a human user interacting with the agent."*）。**注意**：官方代码 `RunConfig.user_model` 的默认值是 **`"gpt-4o"`**，即代码默认与论文 v1 不同 | [论文 §3](https://arxiv.org/html/2406.12045v1)、[types.py](https://raw.githubusercontent.com/sierra-research/tau-bench/main/tau_bench/types.py) |
| **user simulator 机制** | user state = system prompt（含任务 instruction）+ 与 agent 的完整对话历史。**user 看不到 agent 与 API tools 的交互历史**。user 输出 `###STOP###` 时 episode 结束 | [论文 §3](https://arxiv.org/html/2406.12045v1) |
| **agent 温度** | `RunConfig.temperature` 默认 **`0.0`** | [types.py](https://raw.githubusercontent.com/sierra-research/tau-bench/main/tau_bench/types.py) |
| **每任务采样次数** | `RunConfig.num_trials` 默认 **`1`**；官方脚本按需设大（论文报告 pass^8 即 `num_trials=8`），代码会一次性输出所有 $k\in[1,n]$ 的 pass^k | [types.py](https://raw.githubusercontent.com/sierra-research/tau-bench/main/tau_bench/types.py)、[run.py](https://raw.githubusercontent.com/sierra-research/tau-bench/main/tau_bench/run.py) |
| **报告的 k 值** | 论文主表用 **pass^1**（=平均 reward）；稳定性分析用 **pass^4 与 pass^8**（摘要与正文均以 **pass^8 < 25% in retail** 作为核心结论）。**代码层面支持任意 $k \le n$** | [论文摘要与 §5.1](https://arxiv.org/html/2406.12045v1)、[run.py](https://raw.githubusercontent.com/sierra-research/tau-bench/main/tau_bench/run.py) |
| **可复现性设定** | `seed=10`、`task_split="test"`、`user_strategy="llm"`、`max_concurrency=1`（默认） | [types.py](https://raw.githubusercontent.com/sierra-research/tau-bench/main/tau_bench/types.py) |
| **environment** | `retail` / `airline` 两个 domain；`RunConfig.env` 默认 `"retail"` | [run.py](https://raw.githubusercontent.com/sierra-research/tau-bench/main/tau_bench/run.py) |
| **agent 策略** | `tool-calling`（native function calling，默认）/ `act` / `react` / `few-shot` | [run.py](https://raw.githubusercontent.com/sierra-research/tau-bench/main/tau_bench/run.py) |

**关键洞察（对本项目的直接含义）**：agent 温度默认 0.0，**随机性主要来自 user simulator 的采样**。因此若要复现 pass^k 的"对话扰动"效应，必须保证 user simulator 走采样路径（温度 > 0），否则同一任务重跑 k 次只会得到几乎相同的轨迹、pass^k 会虚高。**user simulator 的具体采样温度未在论文正文或默认 `RunConfig` 中显式给出，标记为"未核实"。**

---

## D. 小模型工具调用微调的实践经验

### D.14 Qwen2.5-1.5B/0.5B 做 tool calling 微调的可行性与常见坑

**先说最重要的一条结论（版本事实）**：Qwen2.5 系列的**原生 function calling 能力并非所有尺寸都具备**。Qwen 官方博客在介绍 Qwen2.5 的 tool calling 时，示例与部署命令**只使用 7B**：

> *"Furthermore, Qwen2.5 supports vllm's built-in tool calling. This functionality requires `vllm>=0.6`. … `vllm serve Qwen/Qwen2.5-7B-Instruct --enable-auto-tool-choice --tool-call-parser hermes`"*
>
> *"Qwen2.5's chat template also includes a tool calling template, meaning that you can use Hugging Face transformers' tool calling support."*
>
> *"The vllm / Ollama / transformers tool calling support uses a tool calling template inspired by Nous' Hermes."*

来源：[Qwen2.5 官方博客](https://qwenlm.github.io/blog/qwen2.5/)

**但**：Qwen 官方博客**并未明确声明** 0.5B/1.5B/3B 不支持 function calling。社区广泛流传"Qwen2.5 只有 7B/14B/32B/72B 支持 function calling，0.5B/1.5B/3B 不支持"这一说法，**我在本次调研中未能从 Qwen 官方一手来源（模型卡 / 技术报告 / 文档）核实该限制的原文表述**。因此：

- 我**不能**断言 1.5B 不支持；
- 但可以确定的是：官方 tool calling 的演示与推荐配置均以 7B 为基准，**1.5B 的工具调用能力不应当被默认为可用，必须实测**；
- 这恰好是**本项目做 DPO 偏好对齐的价值所在**：把"工具名选择 / 参数 JSON 合法性"作为偏好信号，直接优化 1.5B 上最弱的那一环。

**`apply_chat_template(..., tools=[...])` 的行为**：Qwen2.5 的 chat template（`tokenizer_config.json` 中的 Jinja）**内置了 tool calling 模板**，因此当传入 `tools=` 时，模板会自动把工具定义渲染成 Hermes/Nous 风格的 `<tools></tools>` 段与系统提示。Qwen 官方原话确认了这三点：① chat template 包含 tool calling 模板；② 可直接使用 transformers 的 tool calling 支持；③ 模板风格取自 [Nous' Hermes](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B)。**该 Jinja 模板的逐行内容我未能核实**（huggingface.co 在本环境不可达，EAI_AGAIN）。

**Hermes / Qwen2.5 / Functionary 风格对比**：

| | Hermes / Qwen2.5 | Functionary |
|---|---|---|
| 工具定义位置 | 渲染进 system prompt 的 `<tools>\n{json schema 逐行}\n</tools>` | 渲染进 system prompt（v2 用 Python 函数签名字符串形式） |
| 调用语法 | `<tool_call>\n{"name": ..., "arguments": {...}}\n</tool_call>` | `<|from|=function_name>{"arg": ...}<|recipient|>` 之类的专用 special token |
| 结果回传 | `<tool_response>\n...\n</tool_response>`，放进 **user** 角色（Qwen-Agent 会把连续的 function 结果合并进 user 消息） | `role="tool"` 消息 / 专用 token |
| 并行调用 | 支持，多个 `<tool_call>` 块并列 | 支持 |
| 解析方式 | 纯文本 + XML 标签（可正则），vLLM 有 `--tool-call-parser hermes` | 依赖 special token，需要 tokenizer 层面的支持 |

Qwen-Agent 默认模板 `NousFnCallPrompt.FN_CALL_TEMPLATE` 的**逐字内容**（这是 Qwen 官方对 Hermes 风格的定义）：

```text
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tool_descs}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
```

对应代码（原文）：

```python
fc = {'name': fn_call.name, 'arguments': arguments}
fc = json.dumps(fc, ensure_ascii=False)
fc = f'<tool_call>\n{fc}\n</tool_call>'
...
content = [ContentItem(text='<tool_response>\n')] + content + [ContentItem(text='\n</tool_response>')]
```

来源：[Qwen-Agent `nous_fncall_prompt.py`](https://raw.githubusercontent.com/QwenLM/Qwen-Agent/main/qwen_agent/llm/fncall_prompts/nous_fncall_prompt.py)

**Qwen2.5 的 ChatML 特殊 token**：Qwen2.5 沿用 ChatML 风格 `<|im_start|>` / `<|im_end|>`，role 为 `system` / `user` / `assistant`。**注意**：vLLM / transformers 的 Hermes 解析器要求模型**不**把工具定义塞进 `<|im_start|>system` 之外的奇怪位置；Qwen2.5 的 chat template 会自己处理。来源：[Qwen2.5 官方博客（transformers 用法示例）](https://qwenlm.github.io/blog/qwen2.5/)

**LoRA rank / 学习率 / 序列长度建议**：

> **标注**：本次调研**未能找到**针对 Qwen2.5-1.5B 工具调用 DPO/SFT 的、带具体超参数值的**同行评审论文或官方配方**。"r=16 / α=32 / lr=1e-4～2e-4（SFT）或 5e-6～1e-6（DPO）/ seq 2048"这类数值属于**社区通行做法**，而非本次核实到的一手结论。以下为工程推断，请以实测为准。

可用的**一手锚点**只有两个：

1. **DPO 论文的官方 SFT/DPO 超参**：DPO 阶段 β=0.1、batch 64、RMSprop、lr=1e-6、150 步线性 warmup（TL;DR 用 β=0.5）。来源见 §A.4。
2. **TRL 的 LoRA/DPO 官方实践**：TRL v0.13.0 文档明确说明 LoRA + DPO 时 `ref_model=None`，并给出三种 reference 构造方案（§B.8）；TRL 的 DPO Quick start 用 `Qwen2-0.5B-Instruct` 做演示，说明**0.5B 级别的模型反正是能跑通 DPO 流程的**（但那是通用对话偏好，不是工具调用）。来源：[v0.13.0 dpo_trainer.mdx](https://raw.githubusercontent.com/huggingface/trl/v0.13.0/docs/source/dpo_trainer.mdx)

**1.5B + LoRA 的常见坑清单**：

1. **必须先 SFT 再做 DPO**。TRL 文档原话：*"The first step is to train an SFT model, to ensure the data we train on is in-distribution for the DPO algorithm."* 对 1.5B 工具调用，尤其重要——如果模型根本不会输出 `<tool_call>` 语法，DPO 的 chosen/rejected 都会是"格式错误"，偏好信号无效。**推荐路径：SFT（教格式）→ DPO（修选择与参数）。**
2. **`ref_model=None` 的语义陷阱**（§B.8 第 4 条）：LoRA 未 merge 时，$\pi_{\text{ref}}$ 是 base 模型。如果你在 base 上先做了 LoRA SFT，然后**继续用同一个 adapter** 做 DPO 且 `ref_model=None`，那么 reference 是**原始 base 而非你的 SFT 模型**——这会严重削弱 KL 约束的意义。正确做法是 **merge SFT adapter 进 base，再新建 LoRA 做 DPO**（TRL 方案 2），或用双 adapter（方案 3）。
3. **序列长度**：多轮 + tools schema 极易超长。`max_length` 必须覆盖 `prompt + completion` 的总和。1.5B 在 3090（24GB）上，bf16 + LoRA + gradient checkpointing 大致可支撑 `max_length=2048~4096`、`per_device_batch=1~2` + 梯度累积。**注意 padding 侧**：DPO 的 preference collator 对 prompt 用 left padding、对 completion 用 right padding（`PreferenceCollator` 的 `pad(..., padding_side="left")` for prompt），这与自回归生成一致。
4. **prompt 前缀 token 对齐**（§B.10 第 2 条）：工具调用的 chosen/rejected 常常只差几个 token（工具名或参数值），TRL 会校验前缀一致性。**务必用 `trl.apply_chat_template` 统一渲染，不要手工拼字符串。**
5. **不要把 tools schema 放进 chosen/rejected**。schema 属于 prompt（上下文），不属于被优化的 completion。放进 completion 会让 loss 去拟合 schema 文本，浪费梯度。
6. **`<tool_call>` 内必须是合法 JSON**。Qwen-Agent 用 `json5.loads` 容错解析并会对失败打 `warning: Invalid json tool-calling arguments`，但**训练目标不应该依赖容错**——chosen 必须是严格合法 JSON。
7. **不要用 `loss_type="simpo"/"orpo"`**（§A.5）：TRL 的 `DPOTrainer` 不支持这两个取值，会抛 `ValueError: Unknown loss type`。
8. **`beta` 不要照搬 0.1**：工具调用的 chosen/rejected 差异往往很小（一个工具名），log-ratio 天然接近 0，$\sigma(\cdot)$ 权重接近 0.5，梯度不会饱和但信号弱。若发现 `rewards/margins` 长期贴 0，可以考虑**调大 β**（0.1 → 0.3~0.5）以放大差异信号，或先用 SFT 把 chosen 的 log-prob 抬起来。**（此为基于 §A.3 梯度分析的推断，未见文献直接验证。）**

---

### D.15 Qwen2.5 的 chat template 如何注入 tools，以及 Hermes / Functionary 风格区别

**（答案已并入 §D.14 的两张表与模板原文；此处补充 API 层面的精确行为）**

`apply_chat_template(messages, tools=[...], tokenize=False, add_generation_prompt=True)` 在 Qwen2.5 上的行为：

1. **工具定义被渲染进系统提示段**。Qwen2.5 的 chat template 内置了 Hermes 风格的工具段（`<tools>...</tools>` 加 `# Tools` 说明），无需用户自己拼。
2. **`add_generation_prompt=True`** 时会在末尾生成 `<|im_start|>assistant\n`，提示模型开始输出——这是工具调用的正确起手式。
3. **是否注入取决于 `tools` 是否非空**。TRL 在 v1.0.0 有一条相关修复：*"Pass tools as `None` to `apply_chat_template` when it is an empty list"*（PR [#5380](https://github.com/huggingface/trl/pull/5380)），说明**空 `tools=[]` 与 `tools=None` 曾经有行为差异**——这是训练脚本里的实际坑（传空列表可能渲染出空的 `<tools></tools>` 段）。
4. **模板本身无法从本环境直接读取**（huggingface.co 不可达）。若需要审计模板内容，在能联网的机器上执行：
   ```python
   from transformers import AutoTokenizer
   print(AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct").chat_template)
   ```

**Hermes 与 Functionary 的核心差异（一句话版）**：Hermes/Qwen2.5 用**可读文本 + XML 标签**（`<tool_call>{json}</tool_call>`），解析器是文本级正则，工具结果走 `<tool_response>` 并常以 user 角色回传；Functionary 用**专用 special token**（`<|from|=...>`, `<|recipient|>` 等）编码调用与角色切换，解析依赖 tokenizer 词表，且对多轮 tool 角色的表达更严格。

**与 vLLM 部署的对应关系**：Hermes 风格在 vLLM 中通过 `--enable-auto-tool-choice --tool-call-parser hermes` 启用，Qwen2.5 官方即推荐这一组合；Functionary 需要 `--tool-call-parser functionary`（**具体 parser 名称未核实**）。来源：[Qwen2.5 官方博客](https://qwenlm.github.io/blog/qwen2.5/)、[Qwen-Agent `nous_fncall_prompt.py`](https://raw.githubusercontent.com/QwenLM/Qwen-Agent/main/qwen_agent/llm/fncall_prompts/nous_fncall_prompt.py)

---

## §E 对本项目的落地建议（1.5B + LoRA + 单卡 3090）

### E.1 固定 TRL 版本：**0.13.0**（主推），次选 0.14.0

**推荐理由（逐条对应上文核实结果）：**

| 判据 | 0.13.0 的情况 |
|---|---|
| API 是否已稳定 | ✅ `DPOConfig` + `processing_class` 已是唯一推荐路径；`tokenizer=` 仍可用（到 0.14.0 才移除），兼容性最宽 |
| 是否会撞上弃用警告 | ✅ 位置参数弃用**恰好到期于 0.13.0**，也就是说用 0.13.0 时你写的 `DPOConfig(beta=...)` 是"目标写法"，不会再看到 "please use the DPOConfig" 的警告噪声 |
| `max_prompt_length` / `max_length` | ✅ **两者都正常可用**（`max_length` 默认 None、`max_prompt_length` 默认 512），你可以显式控制多轮 prompt 与 completion 各自的长度上限。这是 **0.14.0 之后到 0.27.0 之间最实用的窗口**——0.14.0 起 `max_length` 默认变 1024，0.28.0 起 `max_prompt_length` 进入弃用 |
| 会话式数据集直通 | ✅ v0.12.0 已支持（PR #2131），可直接喂 `prompt`/`chosen`/`rejected` 为 message list 的数据集 |
| `loss_type` 覆盖面 | ✅ 13 种（含 `discopop`, `apo_zero/down`, `ipo`, `robust`, `hinge`），足够做消融 |
| 工具调用相关 API | ⚠️ **无** `DPOConfig.tools`（v0.14.0 才引入）。但本项目推荐的做法本就是**把 tools schema 写进 prompt 文本**，不依赖该参数——所以这不是阻塞项 |
| 文档可得性 | ✅ `docs/source/dpo_trainer.mdx`、`dataset_formats.mdx` 都在 tag 上可读，且 `tokenizer`/`processing_class` 双写法有明确的迁移注解 |

**为什么不选 1.x**：v1.x 需要 `transformers>=4.56.2`（见 PyPI `requires_dist`），且参数面已经收敛（`max_prompt_length`/`max_completion_length`/`reference_free`/`rpo_alpha`/`tools` 全部移除，`truncation_mode` 默认改为 `keep_start`、`keep_end` 本身被弃用）。对复现性优先的项目，**锁定一个中期的、文档完整的版本**比追最新更有价值。若需 v1.x，请以 `main/MIGRATION.md` 为准。

**次选 0.14.0**：唯一好处是拿到 `DPOConfig.tools`（全局工具列表）和新 `PreferenceCollator`。代价：`tokenizer=` 报错（必须写 `processing_class=`），`max_length` 默认变 1024。**如果沿用 0.13.0 且把 tools 写进 prompt 文本，则完全不需要这个特性。**

**锁定方式**：

```bash
pip install "trl==0.13.0"
# 配套（0.13.0 的合理生态版本区间，请自行 pin 到实测可用的具体补丁号）
pip install "transformers>=4.46,<4.50" "peft>=0.13" "datasets>=3.0" "accelerate>=0.34"
```

> `transformers` 的具体上限**未核实**（TRL 0.13.0 未在 PyPI metadata 中给出 `requires_dist`——PyPI JSON 只保留最新版的 `requires_dist`）。0.12.0 release notes 提到 "Bump the minimum transformers version to v4.46"（PR #2245），可作下界依据。**请以本地实测为准，并在 `requirements.txt` 中写死。**

### E.2 推荐参数值

```python
args = DPOConfig(
    output_dir="outputs/qwen2.5-1.5b-toolcall-dpo",

    # ---- 算法 ----
    beta=0.1,                    # DPO 论文缺省 + TRL 缺省。若 rewards/margins 长期贴 0，再试 0.3
    loss_type="sigmoid",         # 原始 DPO。备选消融：ipo（防过拟合）、hinge、robust
    label_smoothing=0.0,         # 若偏好标注有噪声（规则打分必然有），提到 0.1（cDPO / Robust DPO）
    use_weighting=False,         # WPO，先关

    # ---- 长度（0.13.0 双旋钮可用）----
    max_length=3072,             # prompt + completion 总上限
    max_prompt_length=2560,      # 多轮历史 + tools schema 的预算是大头
    truncation_mode="keep_end",  # 0.13.0 默认；保留对话末尾（工具调用决策所在处）

    # ---- 优化 ----
    learning_rate=5e-6,          # DPO 阶段要小。论文用 1e-6，LoRA 场景社区常用 5e-6~1e-6
    num_train_epochs=1,          # 偏好数据通常 1 epoch；多轮会过拟合
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,   # 有效 batch 16
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="adamw_torch",         # 论文用 RMSprop，但 LoRA 场景 AdamW 更常见
    lr_scheduler_type="linear",
    warmup_ratio=0.03,

    # ---- 数值与日志 ----
    bf16=True,                   # 3090 是 Ampere，支持 bf16；比 fp16 稳
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    eval_strategy="no",
    report_to="none",            # 或 "wandb"
    remove_unused_columns=False, # DPO 数据列名特殊，保持 False 更安全

    # ---- 性能 ----
    precompute_ref_log_probs=False,   # 24GB 单卡：LoRA + disable_adapter 已足够省；
                                      # 若 OOM 再开 True（会预先跑一遍 ref forward）
    dataset_num_proc=4,
)

lora_config = LoraConfig(
    r=16,                # 1.5B 工具调用：16 是社区起点；欠拟合改 32
    lora_alpha=32,       # 惯例 alpha = 2*r
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
```

**关键参数的推理依据：**

- **`beta=0.1`**：DPO 论文 Appendix B 明文缺省 + TRL `DPOConfig` 缺省（多版本一致）。**不要**照搬 SimPO 的 β=2.0——那个 β 在 SimPO 里只是 sigmoid 的缩放因子，语义与 DPO 的 β 完全不同。
- **`max_prompt_length=2560` / `max_length=3072`**：多轮 + tools schema 是长度杀手。留 512 token 给 completion（一次 `<tool_call>{json}</tool_call>` 通常 < 200 token）。注意 0.13.0 下 `max_length` 只在 prompt 超限时起截断作用（见 `_truncate_tokens`：先按 `max_prompt_length` 截 prompt，若仍超长再截 response 到 `max_length - max_prompt_length`）。
- **`learning_rate=5e-6`**：DPO 阶段比 SFT 小 1~2 个数量级是标准做法。论文 1e-6 是 full-FT 设定；LoRA 参数少、梯度尺度不同，5e-6 是更安全的起点。
- **`precompute_ref_log_probs=False`**：LoRA + `ref_model=None` 时 reference 通过 `disable_adapter()` 得到，**不需要第二个模型常驻显存**（§B.8），显存开销可忽略，没必要额外预先算一遍。
- **`bf16=True`**：3090 是 Ampere，支持 bf16，比 fp16 数值更稳。

### E.3 数据格式长什么样

**推荐的 conversational preference 格式（TRL 0.13.0 直通，无需预处理）：**

```json
{
  "prompt": [
    {"role": "system",
     "content": "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get current weather.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}}\n{\"type\": \"function\", \"function\": {\"name\": \"send_email\", \"description\": \"Send an email.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"to\": {\"type\": \"string\"}, \"body\": {\"type\": \"string\"}}, \"required\": [\"to\", \"body\"]}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"},

    {"role": "user", "content": "帮我看看北京现在天气怎么样"},
    {"role": "assistant", "content": "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Beijing\"}}\n</tool_call>"},
    {"role": "user", "content": "<tool_response>\n{\"temperature\": 26.1, \"unit\": \"celsius\"}\n</tool_response>"},
    {"role": "user", "content": "那顺便给 zhang@example.com 发个邮件说一声"}
  ],

  "chosen": [
    {"role": "assistant", "content": "<tool_call>\n{\"name\": \"send_email\", \"arguments\": {\"to\": \"zhang@example.com\", \"body\": \"北京当前气温 26.1°C\"}}\n</tool_call>"}
  ],

  "rejected": [
    {"role": "assistant", "content": "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"zhang@example.com\"}}\n</tool_call>"}
  ]
}
```

**构造与质检清单：**

1. **`chosen` / `rejected` 只放最后那一条 assistant 消息**（单轮），历史全部在 `prompt` 里。理由见 §B.10。
2. **两者的 `prompt` 必须逐字节相同**（用 `dict` 引用同一对象即可避免手写不一致）。
3. **rejected 的类型要覆盖真实失败模式**，按优先级：
   - 工具名错误（选了不存在的工具 / 选了语义相近但错的工具）← 对 1.5B 最有价值
   - 参数值错误（把邮箱当地名、把城市当邮箱）
   - 参数缺失或多余（schema 违规）
   - JSON 不合法（漏引号、多逗号、用了单引号）
   - 该调用工具却直接编造答案（hallucinated answer，不产出 `<tool_call>`）
   - 该追问澄清却强行调用工具
4. **tools schema 放在 system 消息里**（0.13.0 没有 `DPOConfig.tools`，且这样每行样本可以自带不同的工具集，比全局 tools 更贴近真实场景）。
5. **工具结果用 `<tool_response>` 包住并置于 `user` 角色**——这是 Qwen-Agent 官方模板的做法（`content = [ContentItem(text='<tool_response>\n')] + content + [ContentItem(text='\n</tool_response>')]`），与 Qwen2.5 的 Hermes 风格一致。
6. **先用 `trl.apply_chat_template` 干跑一遍**，确认渲染出的字符串里 `<tool_call>` 与 `<|im_start|>assistant` 位置正确，再启动训练：
   ```python
   from trl import apply_chat_template
   from transformers import AutoTokenizer
   tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
   print(apply_chat_template(dataset[0], tok)["prompt"][-300:])
   ```
7. **数量级**：DPO 对小数据的敏感度远高于 SFT。1.5B 工具调用偏好集建议 **≥ 2k 条**、覆盖 ≥ 3 个工具的交叉混淆对；`rewards/accuracies` 若在 100 步内冲上 0.9 且 `rewards/margins` 继续拉大，说明数据太易，需要加强 rejected 的难度（hard negatives）。

### E.4 验收指标

- **训练侧**：`rewards/accuracies`（chosen > rejected 的比例，目标 0.7~0.85 区间，过高=数据太易）、`rewards/margins`、`rewards/chosen`、`rewards/rejected`。TRL 文档对这四个指标的定义见 [v0.13.0 dpo_trainer.mdx "Logged metrics"](https://raw.githubusercontent.com/huggingface/trl/v0.13.0/docs/source/dpo_trainer.mdx)。
- **评测侧**：**用 pass^k 而不是只看 pass^1**。按 §C 的定义，在工具调用任务上构造"同一语义、不同措辞"的用户请求，每任务采样 `n=8` 次，报告 `pass^1` 与 `pass^8`。1.5B 的典型现象是 pass^1 尚可但 pass^8 塌陷——这正是 DPO 该改善的地方，也是本项目最有说服力的实验结论。
- **禁止只看 loss**：DPO 的 loss 下降与偏好质量提升不必然同向。

### E.5 风险与未核实项清单（写文档时请照抄）

| 项 | 状态 |
|---|---|
| Qwen2.5-0.5B/1.5B/3B 是否原生支持 function calling | **未核实**（官方博客只以 7B 为例，未找到明确的尺寸限制原文） |
| SimPO 论文 Appendix B 的 β、γ 具体推荐值 | **未核实**（HTML 正文截断） |
| TRL v1.x 中 DPO 从数据集的哪个列读取 tools | **未核实** |
| τ-bench user simulator 的具体采样温度 | **未核实**（`RunConfig` 只给出 agent 的 `temperature=0.0`） |
| Functionary 在 vLLM 中的 parser 名称 | **未核实** |
| Qwen2.5-1.5B 工具调用 LoRA 的推荐 rank/lr 的一手文献 | **未找到**，§E.2 的数值为社区惯例 + DPO 论文锚点的工程推断 |
| TRL 0.13.0 的 `transformers` 上界 | **未核实**（PyPI 仅保留最新版 `requires_dist`） |

---

## 参考来源汇总

**论文 / 官方文档**

- Rafailov et al., *Direct Preference Optimization*, [arXiv:2305.18290v2 (HTML)](https://arxiv.org/html/2305.18290v2) ｜ [abs](https://arxiv.org/abs/2305.18290) ｜ [Appendix B 全文转载](https://hackernoon.com/lite/dpo-hyperparameters-and-implementation-details)
- Yao et al., *τ-bench*, [arXiv:2406.12045](https://arxiv.org/html/2406.12045v1) ｜ [abs](https://arxiv.org/abs/2406.12045) ｜ [官方代码 run.py](https://raw.githubusercontent.com/sierra-research/tau-bench/main/tau_bench/run.py) ｜ [types.py](https://raw.githubusercontent.com/sierra-research/tau-bench/main/tau_bench/types.py)
- Azar et al., IPO, [arXiv:2310.12036](https://arxiv.org/abs/2310.12036)
- Ethayarajh et al., KTO, [arXiv:2402.01306](https://arxiv.org/abs/2402.01306)
- Hong et al., ORPO, [arXiv:2403.07691](https://arxiv.org/abs/2403.07691)
- Meng et al., SimPO, [arXiv:2405.14734 (HTML v3)](https://arxiv.org/html/2405.14734v3)
- Qwen Team, [Qwen2.5 官方博客](https://qwenlm.github.io/blog/qwen2.5/)
- Qwen Team, [Qwen 文档 - Function Calling](https://qwen.readthedocs.io/en/latest/framework/function_call.html)
- Qwen-Agent, [`nous_fncall_prompt.py`](https://raw.githubusercontent.com/QwenLM/Qwen-Agent/main/qwen_agent/llm/fncall_prompts/nous_fncall_prompt.py)

**TRL 源码（按版本标注的 tag，`raw.githubusercontent.com`）**

- [v0.9.6 dpo_config.py](https://raw.githubusercontent.com/huggingface/trl/v0.9.6/trl/trainer/dpo_config.py) ｜ [v0.10.1 dpo_config.py](https://raw.githubusercontent.com/huggingface/trl/v0.10.1/trl/trainer/dpo_config.py) ｜ [v0.11.0 dpo_config.py](https://raw.githubusercontent.com/huggingface/trl/v0.11.0/trl/trainer/dpo_config.py) ｜ [v0.12.0 dpo_config.py](https://raw.githubusercontent.com/huggingface/trl/v0.12.0/trl/trainer/dpo_config.py) ｜ [v0.13.0 dpo_config.py](https://raw.githubusercontent.com/huggingface/trl/v0.13.0/trl/trainer/dpo_config.py) ｜ [v0.14.0 dpo_config.py](https://raw.githubusercontent.com/huggingface/trl/v0.14.0/trl/trainer/dpo_config.py) ｜ [v0.25.0](https://raw.githubusercontent.com/huggingface/trl/v0.25.0/trl/trainer/dpo_config.py) ｜ [v0.27.0](https://raw.githubusercontent.com/huggingface/trl/v0.27.0/trl/trainer/dpo_config.py) ｜ [v0.28.0](https://raw.githubusercontent.com/huggingface/trl/v0.28.0/trl/trainer/dpo_config.py) ｜ [v0.29.0](https://raw.githubusercontent.com/huggingface/trl/v0.29.0/trl/trainer/dpo_config.py) ｜ [main](https://raw.githubusercontent.com/huggingface/trl/main/trl/trainer/dpo_config.py)
- [v0.10.1 dpo_trainer.py](https://raw.githubusercontent.com/huggingface/trl/v0.10.1/trl/trainer/dpo_trainer.py) ｜ [v0.12.0 dpo_trainer.py](https://raw.githubusercontent.com/huggingface/trl/v0.12.0/trl/trainer/dpo_trainer.py) ｜ [main dpo_trainer.py](https://raw.githubusercontent.com/huggingface/trl/main/trl/trainer/dpo_trainer.py)
- [v0.13.0 docs/source/dpo_trainer.mdx](https://raw.githubusercontent.com/huggingface/trl/v0.13.0/docs/source/dpo_trainer.mdx) ｜ [v0.13.0 docs/source/dataset_formats.mdx](https://raw.githubusercontent.com/huggingface/trl/v0.13.0/docs/source/dataset_formats.mdx) ｜ [main MIGRATION.md](https://raw.githubusercontent.com/huggingface/trl/main/MIGRATION.md)
- [v0.13.0 仓库树（含 orpo_trainer.py）](https://api.github.com/repos/huggingface/trl/git/trees/v0.13.0?recursive=1)
- Release notes：[v0.12.0](https://api.github.com/repos/huggingface/trl/releases/tags/v0.12.0) ｜ [v0.26.0](https://api.github.com/repos/huggingface/trl/releases/tags/v0.26.0) ｜ [v1.0.0](https://api.github.com/repos/huggingface/trl/releases/tags/v1.0.0)
- 关键 PR：[#2162 `tokenizer`→`processing_class`](https://github.com/huggingface/trl/pull/2162) ｜ [#2141 WPO](https://github.com/huggingface/trl/pull/2141) ｜ [#4969 Deprecate DPOConfig params](https://github.com/huggingface/trl/pull/4969) ｜ [#4300 GRPO agent training](https://github.com/huggingface/trl/pull/4300)
- [PyPI trl JSON（最新版本号与依赖）](https://pypi.org/pypi/trl/json)

**环境限制说明**：本调研过程中 `huggingface.co` 在该环境不可达（`getaddrinfo EAI_AGAIN`），因此所有 TRL / Transformers 的一手证据均取自 `raw.githubusercontent.com` 的 **tag 化源码**（等效于官方文档的版本快照），Qwen2.5 的模型卡与 chat template Jinja 原文未能直接读取，相关条目已标注"未核实"。
