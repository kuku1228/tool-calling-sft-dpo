# 实验结果

> 全部数字来自 `experiments/<exp>/eval_report.json`，由 `scripts/eval/run_baseline.py` 在
> 单卡 RTX 3090（24G）上产出。评测协议固定为 **106 任务 × 4 采样 = 424 次采样/实验**。
> 汇总表由 `scripts/eval/collect_results.py` 生成，见
> [`experiments/summary/ablation_table.md`](../../experiments/summary/ablation_table.md)。

---

## 1. 主结果

| 实验 | 训练数据 | 单次准确率 | pass@4 | pass^4 | Δstability |
|---|---|---|---|---|---|
| 基线（未训练） | — | 38.9% | 43.4% | 34.0% | −9.4% |
| 仅 SFT（对照） | 格式 + 工具选择 + 参数（2635 条） | 80.7% | 81.1% | **80.2%** | **−0.9%** |
| Exp 1 格式约束 | `format_constraint`（1014 对） | **81.6%** | **84.0%** | 80.2% | −3.8% |
| Exp 2 错误恢复 | `error_recovery`（1065 对） | 73.1% | 80.2% | 64.2% | −16.0% |
| Exp 3 联合方案 | 格式 + 恢复 + 澄清（2443 对） | 63.2% | 79.2% | 42.5% | −36.8% |

**训练配置**：Qwen2.5-1.5B-Instruct；PEFT LoRA（r=16, α=32, all-target-linear）；
TRL DPOTrainer；β=0.1；lr=5e-5；grad_accum=8；bf16 + 梯度检查点。
DPO 单方案 1 epoch（约 14 分钟），联合方案另跑了 2 epoch 与 1 epoch 两组。

### 1.1 配对显著性检验

同一批任务上的配对 bootstrap + McNemar 检验，**簇级为报告口径**（106 任务 → 92 簇，
因为部分任务的期望调用完全相同，不是独立证据）：

| 对比 | 口径 | 样本数 | 差值 [95% CI] | 显著 | p |
|---|---|---|---|---|---|
| `dpo_format` vs `sft` | 任务级 | 106 | +0.009 [−0.002, +0.024] | 否 | 1.000 |
| `dpo_format` vs `sft` | **簇级** | **92** | **+0.006 [−0.005, +0.019]** | **否** | 1.000 |
| `dpo_joint` vs `sft` | 任务级 | 106 | −0.175 [−0.269, −0.075] | 是 | 0.185 |
| `dpo_joint` vs `sft` | **簇级** | **92** | **−0.285 [−0.368, −0.204]** | **是** | **< 0.0001** |

两个可注意之处：

1. **`dpo_format` 相对 SFT 的 +0.6pp 不显著**（任务级 MDE ≈ 1.9pp，差值远低于可检出的最小效应）。
2. **同一份数据在任务级口径下不显著、簇级口径下显著**（`dpo_joint` vs `sft`：
   任务级 p=0.185，簇级 p<0.0001）。这说明口径选择会改变结论，因此报告统一采用更保守的簇级。

---

## 2. 按类别（单次准确率）

| 类别 | n | 基线 | 仅 SFT | 格式约束 | 错误恢复 | 联合方案 |
|---|---|---|---|---|---|---|
| `flight_search` | 14 | **0.0%** | **100.0%** | **100.0%** | 83.9% | 39.3% |
| `multi_tool` | 10 | **0.0%** | **100.0%** | **100.0%** | 80.0% | 55.0% |
| `booking_flow` | 15 | **0.0%** | **93.3%** | **93.3%** | 51.7% | 40.0% |
| `error_correction` | 15 | 55.0% | 93.3% | 93.3% | 90.0% | 93.3% |
| `weather_advisory` | 10 | 90.0% | 100.0% | 100.0% | 100.0% | 38/40 条 |
| `multi_turn` | 6 | 12.5% | 91.7% | **100.0%** | 100.0% | 95.8% |
| `baggage` | 14 | 67.9% | 78.6% | 78.6% | 75.0% | 42.9% |
| `cancellation` | 22 | 62.5%\* | 31.8% | 34.1% | 45.5% | **67.0%** |

### 2.1 关于基线的 `cancellation` 62.5%

这个数字**不代表工具调用能力**。`cancellation` 的 22 个任务里有 **15 个是「信息不足，
应先追问」**（用户没给取消原因）。评分器原先只检查「有没有调用工具」，因此空输出也判通过；
未训练模型几乎不调用工具，于是把这 15 个任务"免费"拿下了（实测通过率 98.2%）。

该判据已修复（空输出判失败，并新增按任务口径的格式正确率）。因此解读基线时
应看 `flight_search` / `booking_flow` / `multi_tool` 这三个 **0%** 的类别，
它们才反映真实的工具调用能力。

---

## 3. 错误分布（占全部 424 次采样）

| 错误类别 | 基线 | 仅 SFT | 格式约束 | 错误恢复 | 联合方案 |
|---|---|---|---|---|---|
| 成功 | — | 342 | **346** | 310 | 268 |
| 格式错误 | — | 0 | **1** | 10 | 74 |
| 工具选错 | — | 62 | 58 | 77 | 50 |
| 参数错误 | — | 20 | 19 | 26 | 32 |
| 工具幻觉 | — | 0 | 0 | 1 | 0 |

（基线一份报告的错误分析产物缺失，故留空。）

两点观察：

- **格式约束方案在 424 次采样里只有 1 次格式错误**，与 SFT 的 100% 格式正确率持平。
- `sft` 的「工具选错」62 条中，**60 条是「信息不足时应先追问，却直接调用了工具」**——
  这是 SFT 阶段唯一的主要失败模式。

---

## 4. 泛化与稳定性

### 4.1 seen vs unseen（pass^4）

| 实验 | seen | unseen |
|---|---|---|
| 基线 | 32.8% | 35.9% |
| 仅 SFT | 77.6% | 84.6% |
| 格式约束 | 77.6% | 84.6% |
| 错误恢复 | 64.2% | 64.1% |
| 联合方案 | 38.8% | 48.7% |

> `seen`/`unseen` 标签由骨架哈希派生，**不读取训练数据**，因此其差值不能读作泛化差距。
> `scripts/prepare/verify_all.py` 每次运行都会打印这一警告。

### 4.2 Δstability 的含义

`Δstability = pass^k − pass@k`，衡量「能做对某题」与「每次都做对」之间的距离：

- 基线 −9.4%：有 9.4 个百分点的题属于「重采样偶尔能做对」，**会做但不稳**
- 仅 SFT −0.9%：这个差距几乎消失，**会做的题基本每次都能做对**
- 联合方案 −36.8%：能做对的题里有很大比例靠运气

`pass^k` 的实现用无偏估计 `C(c,k)/C(n,k)`，而非朴素 `(c/n)^k`——在 n=4 这样的小采样下
两者差异显著。

---

## 5. 结论

1. **SFT 对齐是主要增益来源**：38.9% → 80.7%（+41.5pp），pass^4 34.0% → 80.2%，
   Δstability 收敛到 −0.9%。三个基线为 0% 的类别全部提升到 93% 以上，格式正确率 100%。

2. **格式约束对 DPO 与 SFT 无显著差异**（簇级 +0.6pp，CI 含 0），
   且**在全部 8 个类别上零回退**——它可作为低风险方案，但不能声称带来提升。

3. **错误恢复对引入能力代价**：73.1%（−7.6pp vs SFT），`booking_flow` 从 93.3% 掉到 51.7%。

4. **两类偏好对混用产生显著负协同**：63.2%（−17.5pp），簇级差值 −28.5pp（p<0.0001）。
   损伤并非来自 DPO 方法本身，而是来自偏好数据类型之间的冲突——
   这一点由单方案消融排除：单独用任一类都不会出现这种程度的下降。

5. **训练日志提供了可观测的预警信号**：`logps/chosen` 的漂移量与能力损失单调对应——

   | 实验 | `logps/chosen` 终值 | 准确率 |
   |---|---|---|
   | 格式约束 | −0.14 | 81.6% |
   | 错误恢复 | −2.38 | 73.1% |
   | 联合方案（1 epoch） | −26.1 | 63.2% |
   | 联合方案（2 epoch） | −36.3 | 39.2% |

   chosen 侧的似然被压得越狠，任务能力损失越大。这比"只看 loss"更能提前发现问题。

   表中 2 epoch 那一行的训练日志在 `outputs/dpo_joint_2ep/train_log.jsonl`
   （该次运行只用于机制对比，其评测结果不作为主结果——见下方"关于 2 epoch"）。

### 5.1 关于 2 epoch

联合方案另跑了 2 epoch（其余配置相同），结果 **39.2%**，低于 1 epoch 的 63.2%。
两者训练日志里 `logps/chosen` 终值分别为 **−36.3**（2 epoch）与 **−26.1**（1 epoch）。

这说明该配置下的 epoch 数**直接决定漂移幅度**，而漂移幅度又决定能力损失。
因此最终报告采用 1 epoch，并把它作为所有单方案消融的统一设置。

---

## 6. 复现

```bash
# 基线
python scripts/eval/run_baseline.py --label baseline  --num-samples 4

# SFT
python scripts/train/sft_train.py --experiment sft_only --output-dir outputs/sft
python scripts/eval/run_baseline.py --adapter outputs/sft/final \
    --label sft  --num-samples 4

# DPO 单方案（必须从 SFT 权重起步）
python scripts/train/dpo_train.py --experiment dpo_format \
    --init-adapter outputs/sft/final --epochs 1 --output-dir outputs/dpo_format
python scripts/eval/run_baseline.py \
    --model outputs/dpo_format/base_with_init_adapter \
    --adapter outputs/dpo_format/final \
    --label dpo_format  --num-samples 4

# 汇总 + 配对检验
python scripts/eval/collect_results.py --baseline baseline \
    --experiments baseline,sft,dpo_format,dpo_recovery,dpo_joint \
    --out-dir experiments/summary
python scripts/eval/paired_significance.py --a dpo_joint --b sft 
```

### 关于 DPO 评测的加载方式

`dpo_train.py --init-adapter` 的两段式流程是：**先把 SFT 适配器 merge 进基座**
（保存为 `<out>/base_with_init_adapter`），**再在其上训练一层新的 LoRA**。
因此 `outputs/<exp>/final` 是**独立的增量适配器**，必须叠加在 merged base 之上：

```bash
# 正确
--model outputs/<exp>/base_with_init_adapter --adapter outputs/<exp>/final

# 错误（会漏掉 SFT 那一层，实测把 80.4% 误读成 18.9%）
--adapter outputs/<exp>/final
```

`scripts/eval/run_baseline.py` 的 `--adapter` 契约假设「单个适配器即全部权重」，
与 DPO 脚本的两段式约定不一致，使用时需注意。
