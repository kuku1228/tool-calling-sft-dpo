## 消融实验对比

| 实验 | 单次准确率 | pass@4 | pass^4 | Δstability | 相对基线(pass^4) | 效应量 |
|---|---|---|---|---|---|---|
| Exp 1：格式约束偏好对 | 81.6% | 84.0% | 80.2% | -3.8% | +136.1% | 大 (h=-0.97) |
| Exp 2：错误恢复偏好对 | 73.1% | 80.2% | 64.2% | -16.0% | +88.9% | 中 (h=-0.61) |
| baseline | 38.9% | 43.4% | 34.0% | -9.4% | — | — |
| sft | 80.7% | 81.1% | 80.2% | -0.9% | +136.1% | 大 (h=-0.97) |
| dpo_joint | 63.2% | 79.2% | 42.5% | -36.8% | +25.0% | 可忽略 (h=-0.18) |

### 分组明细（pass^4）

**按难度**

| 实验 | easy | hard | medium |
|---|---|---|---|
| Exp 1：格式约束偏好对 | 83.3% | 76.1% | 93.3% |
| Exp 2：错误恢复偏好对 | 83.3% | 53.7% | 80.0% |
| baseline | 66.7% | 19.4% | 46.7% |
| sft | 87.5% | 74.6% | 93.3% |
| dpo_joint | 41.7% | 31.3% | 93.3% |

**按可见性**

| 实验 | seen | unseen |
|---|---|---|
| Exp 1：格式约束偏好对 | 77.6% | 84.6% |
| Exp 2：错误恢复偏好对 | 64.2% | 64.1% |
| baseline | 32.8% | 35.9% |
| sft | 77.6% | 84.6% |
| dpo_joint | 38.8% | 48.7% |

### 数据来源

- `dpo_format`: `/root/autodl-tmp/agent-optimization/experiments/dpo_format/eval_report.json`
- `dpo_recovery`: `/root/autodl-tmp/agent-optimization/experiments/dpo_recovery/eval_report.json`
- `baseline`: `/root/autodl-tmp/agent-optimization/experiments/baseline/eval_report.json`
- `sft`: `/root/autodl-tmp/agent-optimization/experiments/sft/eval_report.json`
- `dpo_joint`: `/root/autodl-tmp/agent-optimization/experiments/dpo_joint/eval_report.json`

### 统计说明

置信区间由 **任务级 bootstrap**（1000 次重采样任务）得到，而不是按采样样本重采样 —— 同一任务的多次采样高度相关，按样本算会严重低估区间宽度。

- `dpo_format`: pass^4 = 80.2% (95%CI [0.0%, 100.0%])，基线 34.0% (95%CI [0.0%, 100.0%])；置信区间与基线重叠 → **不能断言有真实提升**，需增加任务数或采样次数
- `dpo_recovery`: pass^4 = 64.2% (95%CI [0.0%, 100.0%])，基线 34.0% (95%CI [0.0%, 100.0%])；置信区间与基线重叠 → **不能断言有真实提升**，需增加任务数或采样次数

以观测到的 pass^4 差值 +30.2% 计算，要在 80% 功效下检出该效应，每组约需 **44** 个任务（当前 106 个）。
