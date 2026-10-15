## 消融实验对比（⚠️ SIMULATED — 由 mock 流水线产出，不是真实模型结果）

| 实验 | 单次准确率 | pass@2 | pass^2 | Δstability | 相对基线(pass^2) | 效应量 |
|---|---|---|---|---|---|---|
| baseline_airline | 41.7% | 63.3% | 20.0% | -43.3% | — | — |
| 对照：仅 SFT | 41.7% | 63.3% | 20.0% | -43.3% | +0.0% | 可忽略 (h=+0.00) |
| Exp 1：格式约束偏好对 | 41.7% | 63.3% | 20.0% | -43.3% | +0.0% | 可忽略 (h=+0.00) |
| Exp 2：错误恢复偏好对 | 41.7% | 63.3% | 20.0% | -43.3% | +0.0% | 可忽略 (h=+0.00) |
| Exp 3：联合方案 | 41.7% | 63.3% | 20.0% | -43.3% | +0.0% | 可忽略 (h=+0.00) |

### 分组明细（pass^2）

**按难度**

| 实验 | hard |
|---|---|
| baseline_airline | 20.0% |
| 对照：仅 SFT | 20.0% |
| Exp 1：格式约束偏好对 | 20.0% |
| Exp 2：错误恢复偏好对 | 20.0% |
| Exp 3：联合方案 | 20.0% |

**按可见性**

| 实验 | seen | unseen |
|---|---|---|
| baseline_airline | 30.0% | 0.0% |
| 对照：仅 SFT | 30.0% | 0.0% |
| Exp 1：格式约束偏好对 | 30.0% | 0.0% |
| Exp 2：错误恢复偏好对 | 30.0% | 0.0% |
| Exp 3：联合方案 | 30.0% | 0.0% |

### 数据来源

- `baseline_airline`: `D:\HuaweiMoveData\Users\马琬蕊\Desktop\项目2\agent-optimization\experiments\baseline_airline\eval_report.mock.json`
- `sft_only`: `D:\HuaweiMoveData\Users\马琬蕊\Desktop\项目2\agent-optimization\experiments\sft_only\eval_report.mock.json`
- `dpo_format`: `D:\HuaweiMoveData\Users\马琬蕊\Desktop\项目2\agent-optimization\experiments\dpo_format\eval_report.mock.json`
- `dpo_recovery`: `D:\HuaweiMoveData\Users\马琬蕊\Desktop\项目2\agent-optimization\experiments\dpo_recovery\eval_report.mock.json`
- `dpo_joint`: `D:\HuaweiMoveData\Users\马琬蕊\Desktop\项目2\agent-optimization\experiments\dpo_joint\eval_report.mock.json`

### 统计说明

置信区间由 **任务级 bootstrap**（1000 次重采样任务）得到，而不是按采样样本重采样 —— 同一任务的多次采样高度相关，按样本算会严重低估区间宽度。

- `sft_only`: pass^2 = 20.0% (95%CI [0.0%, 100.0%])，基线 20.0% (95%CI [0.0%, 100.0%])；置信区间与基线重叠 → **不能断言有真实提升**，需增加任务数或采样次数
- `dpo_format`: pass^2 = 20.0% (95%CI [0.0%, 100.0%])，基线 20.0% (95%CI [0.0%, 100.0%])；置信区间与基线重叠 → **不能断言有真实提升**，需增加任务数或采样次数
- `dpo_recovery`: pass^2 = 20.0% (95%CI [0.0%, 100.0%])，基线 20.0% (95%CI [0.0%, 100.0%])；置信区间与基线重叠 → **不能断言有真实提升**，需增加任务数或采样次数
- `dpo_joint`: pass^2 = 20.0% (95%CI [0.0%, 100.0%])，基线 20.0% (95%CI [0.0%, 100.0%])；置信区间与基线重叠 → **不能断言有真实提升**，需增加任务数或采样次数

