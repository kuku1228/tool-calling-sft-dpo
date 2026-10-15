# AutoDL 云端部署与运行手册（逐条勾选版）

> **文档定位**：给「本地已跑通 mock、明天要上云跑真实实验」的人用的操作手册。
> 每一条都有**预期输出**，对不上就停下排查，不要硬往下走。
>
> **前置状态**（本地已完成，无需重做）：
> - 单元测试 50/50 通过、43 个文件 doctest 全过
> - mock 流水线端到端跑通（注入一致性 ≥ 94%）
> - 偏好数据可构造（约 3,900 条）、任务集已固化
> - **尚未跑过任何真实 GPU 评测/训练**
>
> **本手册目标**：在一台租来的 24G 单卡机器上，产出
> `experiments/baseline_airline/eval_report.json` 等**真实**结果。

---

## 0. 明天的时间线（先看这个）

| 时段 | 做什么 | 预估 |
|---|---|---|
| 0:00–0:10 | 本地打包 + 租机 + 上传 | 10 分 |
| 0:10–0:25 | 装依赖、下模型 | 15 分 |
| 0:25–0:35 | **冒烟验证（关键，别跳过）** | 10 分 |
| 0:35–1:05 | 全量真实基线评测 | 30 分 |
| 1:05–1:20 | 错误诊断 | 15 分 |
| 1:20–1:35 | 偏好数据构造与体检 | 15 分 |
| 1:35–3:30 | 四组训练 + 评测 | 约 2 小时 |
| 3:30–4:00 | 汇总、出图、下载结果 | 30 分 |
| | **合计** | **约 4 小时** |

**成本**：3090（¥1.68/h，学生 85 折 ¥1.43）≈ **¥6–8**。
4090（¥3.49/h）≈ **¥12–15**，耗时减半。

> ⚠️ 价格来自第三方汇总页（[来源](https://yangmao.ai/zh/compute/autodl/)，标注验证日期 2026-04-21），
> 实际以下单页为准。**关机后 GPU 不计费**，但数据盘继续收 ¥0.1/GB/天。

---

## 1. 本地准备（租机之前，10 分钟）

### 1.1 确认本地仍然健康

```powershell
cd "D:\HuaweiMoveData\Users\马琬蕊\Desktop\项目2\agent-optimization"
python tests/run_tests.py
```

**预期**：`结果: 50 通过, 0 失败`

### 1.2 打包代码（排除大文件与产物）

```powershell
cd "D:\HuaweiMoveData\Users\马琬蕊\Desktop\项目2"
# 只打包代码与配置，不含 mock 产物、缓存、__pycache__
Compress-Archive -Path "agent-optimization\*" `
    -DestinationPath "agent-optimization-upload.zip" -Force `
    -CompressionLevel Optimal
Get-Item "agent-optimization-upload.zip" | Select-Object Name, @{n='MB';e={[math]::Round($_.Length/1MB,1)}}
```

**实际产出**：`agent-optimization-upload.zip` = **2.89 MB / 104 个文件**。

已排除的内容（都在云端可重新生成，或本来就是 mock 产物）：

| 排除项 | 原因 |
|---|---|
| `outputs/`（138 MB） | mock 训练产物，无价值 |
| `experiments/mock`、`experiments/summary` | mock 评测报告与汇总表 |
| `experiments/` 下除任务集外的其他文件 | 557 个 mock 失败案例 |
| `__pycache__/`、`.tmp/` | 缓存与临时文件 |

**已包含**的关键内容：

- `data/preferences/`（18 个 jsonl，共 20508 条偏好记录）
  —— 让本地与云端用**同一份数据文件**，排除「数据不同」这个变量
- `experiments/tasks_airline_v1.json`（固化的 100 任务集）
- `src/`、`scripts/`、`configs/`、`tests/`、`docs/` 全部代码与文档

> ⚠️ **排除规则必须按相对路径匹配，不能按目录名。**
> 本项目开发期踩过：用目录名 `mock` 排除时，把 `src/mock/`（mock 策略模块
> 本身）也排除了 —— 云端 5 个测试、9 个流水线步骤直接失败。
> 打包后**务必解压到干净目录跑一遍 `tests/run_tests.py`** 再上传。

> **为什么打包而不是直接传文件夹**：Windows 用户名是中文（`马琬蕊`），
> 某些上传工具对中文路径 + 大量小文件会报错或极慢。zip 最稳。

### 1.3 记录本地基线状态（明天回来对比）

```powershell
python scripts/deploy/which_python.py > local_env_snapshot.txt
python -c "import sys; sys.path.insert(0,'.'); from src.config import describe; print(describe())" >> local_env_snapshot.txt
```

留着这份快照，明天如果云上行为不同，可以对比解释器/版本差异。

---

## 2. 租机器（5 分钟）

### 2.1 配置选择

| 项 | 选择 | 说明 |
|---|---|---|
| **GPU** | RTX 3090 / 24G | 首选。1.5B + LoRA 显存占用约 5–8G，离 24G 上限很远，4090 的额外性能不值 2 倍价钱 |
| 备选 | RTX 4090 / 24G | 若你时间紧、想快点跑完，或要多次调参 |
| **镜像** | **干净的 PyTorch 2.1+ 基础镜像** | 见下方警告 |
| **数据盘** | 50 G | 模型 3G + 数据 30M + checkpoint；`save_total_limit=3` 已限制 |
| 计费 | 按量 | 跑完**立刻关机** |

> ⚠️ **不要选「大模型全家桶 / Qwen 一站式」之类的预装镜像。**
> 你的 `requirements.txt` 要求 `transformers>=4.45.0,<4.50.0` 与 `trl<=0.13.0`；
> 预装镜像里的版本往往冲突，降级比从干净环境装更费时间。
> 选**只预装 CUDA + PyTorch** 的基础镜像，其余自己 pip 装。

### 2.2 租到后先做三件事

```bash
# 1) 确认 GPU 真的可见
nvidia-smi
# 预期：看到 RTX 3090 / 4090，显存 24G，CUDA Version 12.x

# 2) 确认预装的 torch 认 GPU
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 预期：2.x.x True   ← 如果是 False，说明镜像的 torch 是 CPU 版，换镜像

# 3) 把工作目录放到数据盘（系统盘小且贵）
mkdir -p /root/autodl-tmp && cd /root/autodl-tmp
df -h /root/autodl-tmp
```

> 若 `torch.cuda.is_available()` 是 `False`，**先别继续**。
> 整条训练链路都会失败，而且报错信息会误导你以为是代码问题。

### 2.3 配置国内下载加速（重要）

```bash
# AutoDL 提供学术加速，pip 与 HF 都受益
source /etc/network_turbo 2>/dev/null || echo "（该镜像无此脚本，跳过）"

export HF_ENDPOINT=https://hf-mirror.com
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
```

---

## 3. 上传代码（3 分钟）

用 AutoDL 控制台的「文件传输」上传 `agent-optimization-upload.zip` 到
`/root/autodl-tmp/`，然后：

```bash
cd /root/autodl-tmp
unzip -q agent-optimization-upload.zip -d agent-optimization
cd agent-optimization
ls
# 预期：README.md configs data docs scripts src tests requirements.txt ...
```

> 若中文文件名（`项目方案.md`）解压后乱码：不影响代码运行，可忽略。
> 想避免的话在本地打包前把它临时移出即可。

---

## 4. 装依赖（约 10 分钟）

```bash
cd /root/autodl-tmp/agent-optimization

# 先看镜像里已有什么，避免重复装大包
python -c "
import importlib
for m in ['torch','transformers','trl','peft','datasets','numpy','matplotlib','yaml']:
    try:
        mod = importlib.import_module(m); print(f'  {m:14s} {getattr(mod,\"__version__\",\"?\")}')
    except ImportError: print(f'  {m:14s} 未安装')
"
```

**已装的不要动**（尤其 torch）。只装缺的：

```bash
# 核心训练依赖（torch 已由镜像提供，这里不重复装）
pip install -q "transformers>=4.45.0,<4.50.0" "peft>=0.7.0" "datasets>=2.16.0"
# trl 必须锁 0.13.0 —— 见 docs/ops/trl_version_notes.md
pip install -q "trl==0.13.0"
# 出图与配置
pip install -q matplotlib pyyaml tensorboard
```

**验证版本组合**：

```bash
python -c "
import torch, transformers, trl, peft, datasets
print('torch       ', torch.__version__, 'cuda=', torch.cuda.is_available())
print('transformers', transformers.__version__)
print('trl         ', trl.__version__)
print('peft        ', peft.__version__)
print('datasets    ', datasets.__version__)
"
```

**预期**：

```
torch        2.x.x  cuda= True
transformers 4.45.x ~ 4.49.x
trl          0.13.0
peft         0.x.x
datasets     2.x.x ~ 4.x
```

> ⚠️ 若 `trl` 装成了 `1.x`：`pip install -q "trl==0.13.0" --force-reinstall`。
> 项目有兼容层能容忍 0.9–0.13，但 1.x 未验证过（见 `src/training/compat.py` 的告警）。

---

## 5. 下载模型（约 5 分钟）

```bash
export HF_ENDPOINT=https://hf-mirror.com
pip install -q -U "huggingface_hub[cli]"

huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct \
    --local-dir /root/autodl-tmp/models/Qwen2.5-1.5B-Instruct
```

**预期**：下载约 3 GB，目标目录里出现
`config.json`、`model.safetensors`、`tokenizer.json`、`vocab.json` 等。

```bash
du -sh /root/autodl-tmp/models/Qwen2.5-1.5B-Instruct
# 预期：约 3.0G
```

**设置环境变量**（写进 `~/.bashrc` 免得断线后忘）：

```bash
cat >> ~/.bashrc <<'EOF'
export AGENTOPT_MODEL_LOCAL=/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct
export AGENTOPT_HF_ENDPOINT=https://hf-mirror.com
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
EOF
source ~/.bashrc
```

**验证项目能读到模型**：

```bash
cd /root/autodl-tmp/agent-optimization
python -c "
import sys; sys.path.insert(0,'.')
from src.config import CFG, describe
print(describe())
print('模型解析路径:', CFG.model.resolved)
"
# 预期：模型解析路径 = /root/autodl-tmp/models/Qwen2.5-1.5B-Instruct
```

---

## 6. ⚠️ 冒烟验证（关键，别跳过，10 分钟）

这一步用 8 个任务 × 2 采样跑**真实模型**。它的价值：在花全量机时之前，
确认「模型能加载、能生成、输出能被正确解析」。

```bash
cd /root/autodl-tmp/agent-optimization

# 先跑本地测试（云上也要过一遍，确认解压没丢文件）
python tests/run_tests.py
# 预期：50 通过, 0 失败

# 再跑 8 任务的真实评测
python scripts/eval/run_baseline.py \
    --config configs/eval/eval_baseline.yaml \
    --label baseline_smoke \
    --task-limit 8 \
    --num-samples 2
```

**这一步要重点看四件事**：

1. **模型真的加载了**：日志里出现 `加载模型 ...` 与显存报告（约 3–4 GB）
2. **生成不是空/乱码**：随便找一个 `predictions.jsonl` 里的样本看看
3. **格式正确率不是 0%**：日志里 `格式正确率 XX%` —— 如果是 **0%**，
   说明 chat template 把 `<tool_call>` 破坏了，**立刻停下**（这是最典型的坑）
4. **解析器在工作**：错误分布里应该混有格式/参数/工具类错误，
   而不是 100% 全是 `FORMAT_ERROR`

```bash
# 看几条实际生成
python -c "
import json
rows=[json.loads(l) for l in open('experiments/baseline_smoke/predictions.jsonl',encoding='utf-8')]
for r in rows[:3]:
    print('---', r['task_id'], 'finish=', r['finish_reason'])
    print(repr(r['text'][:300]))
"
```

**判断标准**：

| 现象 | 结论 |
|---|---|
| 输出是 `<tool_call>{"name": ...}</tool_call>` | ✅ 正常，继续 |
| 输出是大段自然语言、无标签 | ⚠️ 可能是模型能力问题（正常），也可能是 prompt 问题 —— 看格式率是否 > 20% |
| 输出是乱码 / 重复token / 空 | ❌ **停下**，报给我 |
| 格式正确率 = 0% 且全部 FORMAT_ERROR | ❌ **停下**，很可能是 chat template 问题 |

> 冒烟结果**不是**项目结论（任务太少、置信区间极宽），
> 它的唯一用途是确认链路通畅。跑完可以删掉：
> `rm -rf experiments/baseline_smoke`

---

## 7. 全量真实基线（Week 1，约 30 分钟）

```bash
cd /root/autodl-tmp/agent-optimization

# 用 nohup 后台跑，断线不丢
nohup python scripts/eval/run_baseline.py \
    --config configs/eval/eval_baseline.yaml \
    > baseline.log 2>&1 &

tail -f baseline.log          # Ctrl+C 只退出查看，不杀进程
```

**预期**：100 任务 × 4 采样 = 400 次生成，约 20–35 分钟（3090）。
日志里每 10 个任务打印一次进度。

**跑完的标志**：出现

```
用时 XXXX.Xs
  报告        : .../experiments/baseline_airline/eval_report.json
  原始预测    : ...（400 条）
```

**检查产物**：

```bash
ls -la experiments/baseline_airline/
# 预期：eval_report.json  predictions.jsonl  error_analysis.json  failure_cases/
```

> ⚠️ **确认真实性**：打开 `eval_report.json`，`metadata.is_mock` 必须是 `false`。
> 如果文件名里带 `.mock`，说明你的环境变量 `AGENTOPT_FORCE_MOCK` 被设成了 1，检查一下。

---

## 8. 错误诊断（15 分钟）

```bash
python scripts/eval/analyze_errors.py --label baseline_airline
```

**预期输出**：错误类型分布表 + 失败子类型 Top-N + 按难度/可见性分组 + 图表。

**这一步产出的东西直接决定后续方案**，请留意：

| 观察 | 含义与后续动作 |
|---|---|
| `FORMAT_ERROR` 占比最高（> 30%） | 与项目假设一致 → 格式约束偏好对是重点 |
| `HALLUCINATION` 占比高 | 工具名幻觉严重 → 加强工具选择偏好对 |
| `PARAM_ERROR` 占比高 | 参数语义是主要瓶颈 → 加强参数约束偏好对 |
| `format_ok_value_wrong_rate` 很高 | 模型会「格式对但实体乱填」→ 必须在偏好数据里加大硬负样本比例 |
| `unseen` 准确率明显低于 `seen` | 泛化差 → 说明任务集划分合理，后续重点看 unseen 上的提升 |

**把这一段输出发给我**，我会据此判断偏好数据配比要不要调，
而不是照原方案盲目跑。

---

## 8.5 关于可复现性（重要，已修复）

本项目开发期发现并修掉了一个会破坏实验结果可比性的缺陷：

**数据生成曾用 Python 内建 `hash()` 做随机种子。** Python 3.3 起 `hash()`
对字符串**按进程随机化**（`PYTHONHASHSEED` 机制），结果是**同一次实验的
两次运行得到不同的偏好数据** —— 不报错、不警告，只是内容悄悄变了。

实测证据（修复前，同一命令连跑两次）：

```
对数量: 140  内容 sha256: 88b9cc8bbb2a32d0
对数量: 140  内容 sha256: 743e038fbfcc851f     ← 条数一样，内容不同
```

修复后（连跑三次）：

```
对数量: 140  内容 sha256: 4e5dc42113b221b9
对数量: 140  内容 sha256: 4e5dc42113b221b9
对数量: 140  内容 sha256: 4e5dc42113b221b9
```

**已改为 `src/utils/hashing.py` 的 `stable_seed()`**（基于 `hashlib.md5`，
跨进程/跨平台/跨 Python 版本稳定）。并有回归测试
`tests/test_methodology.py::test_data_generation_is_reproducible` 守住，
它同时做静态检查：源码里不应再出现内建 `hash()` 做种子。

**这对你意味着什么**：

- 云端与本地用同一份 `data/preferences/`（已打进上传包），
  即使不重新生成也完全可比
- 若在云端重新生成，结果与本地**逐字节一致**（已实测）
- 你不需要设 `PYTHONHASHSEED`；修好的代码已经不依赖它

---

## 9. 偏好数据构造与体检（Week 2，15 分钟）

```bash
# 构造（约 1 分钟）
python scripts/prepare/build_preferences.py

# 体检 —— 这一步会给出**真实的 max_length 建议**
python scripts/prepare/inspect_preferences.py
```

**重点看 `inspect_preferences.py` 的这三项**：

1. **prompt 一致性检查**：必须是 `✓ 训练 prompt 与评测 prompt 前缀一致`。
   不一致会导致「训练 loss 下降但评测指标不动」—— 必须停下修。
2. **建议配置**：形如 `建议配置: max_length=XXXX max_prompt_length=YYYY`。
   **记下这两个数**，下一步要用。
3. **截断后退化率**：必须 < 5%。若超了，按它的提示调大 `max_length`。

```bash
# 把建议值写进训练配置（把 XXXX / YYYY 换成实际值）
python - <<'PY'
import json, pathlib
for name in ("dpo_format","dpo_recovery","dpo_joint"):
    p = pathlib.Path(f"configs/train/{name}.yaml")
    d = json.loads(p.read_text(encoding="utf-8"))
    d["train"]["max_length"] = XXXX        # ← 换成实际建议值
    d["train"]["max_prompt_length"] = YYYY # ← 换成实际建议值
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print("已更新", p)
PY
```

---

## 10. 训练（Week 3，约 2 小时）

### 10.1 先 dry-run 确认不 OOM

```bash
# 这一条不训练，只做数据体检 + 打印建议 max_length
python scripts/train/dpo_train.py --experiment dpo_joint --dry-run
```

**预期**：打印数据统计 + token 长度分位数 + 截断率，最后 `--dry-run：体检完成`。

### 10.2 单组真跑（先验证一组，别一次全开）

```bash
nohup python scripts/train/dpo_train.py \
    --experiment dpo_joint \
    --config configs/train/dpo_joint.yaml \
    > dpo_joint.log 2>&1 &

tail -f dpo_joint.log
```

> ⚠️ **必须接监控回调才有输出。** TRL 的 `Trainer` 默认只把指标交给
> `report_to`（tensorboard），**不打印到控制台**。本项目已在
> `dpo_train.py` 里挂载了 `src/training/train_monitor.py` 的回调，
> 所以你会看到实时指标流。若日志里迟迟没有 `step` 行，检查那行
> `已挂载训练监控回调` 是否出现。

**你会看到这样的实时输出**：

```
--------------------------------------------------------------------------
DPO 训练监控（每 logging step 刷新）
  理论起点: loss = log 2 = 0.6931（此时 π_θ = π_ref）
  健康信号: loss 下降 / rewards/margin 单调上升 / rewards/accuracy 从 0.5 上升
  总步数  : 250
--------------------------------------------------------------------------
  step    10  loss=0.6935  rewards/accuracy=0.5100  rewards/margin=0.0012  grad_norm=1.3200  learning_rate=2.1e-05
  step    20  loss=0.6801  rewards/accuracy=0.5400  rewards/margin=0.0088  grad_norm=1.2900  learning_rate=4.2e-05
  ...
```

**要盯的三个信号**（判据来自 DPO 的数学性质，不是拍脑袋的经验值）：

| 信号 | 健康表现 | 异常表现与含义 |
|---|---|---|
| `loss` | 从 `0.6931` 下降 | **明显偏离 0.6931** → 偏好对被截断成两侧相同，或 chosen/rejected 标反了。监控会**自动告警** |
| `rewards/margin` | 单调上升 | 不动 → 两侧语义等价，梯度为零（DPO 的 loss 恒为 log 2） |
| `rewards/accuracy` | 从 ~0.5 上升 | 一直贴 0.5 → 模型没学到有效偏好 |

**训练结束会自动打印健康度判读**，并写入 `outputs/<exp>/train_health.txt`：

```
DPO 训练健康度判读
==============================================================
记录条数: 25
  loss            0.6935 → 0.1753   （log 2 = 0.6931）
    ✓ 初始 loss 接近 log 2，符合 DPO 的理论起点
    ✓ loss 下降 74.7%
  rewards/margin  +0.0012 → +0.5000
    ✓ margin 上升，说明模型在拉大 chosen 与 rejected 的隐式奖励差
  rewards/accuracy 0.5100 → 0.9200
    ✓ accuracy 超过 0.5，模型能区分 chosen/rejected
==============================================================
```

出现 ⚠️ 时按提示排查，**不要硬跑完四组** —— 数据问题不会随训练步数自动消失，
只会浪费机时。

**逐 step 的完整记录**在 `outputs/<exp>/train_log.jsonl`（含所有指标），
即使训练中途崩溃也不会丢已记录的部分。

**显存排查**（另开一个终端）：

```bash
watch -n 5 nvidia-smi
# 预期：显存占用 6–10 GB（LoRA + bs=2 + max_length≈1536）
# 若接近 24G 或 OOM：减小 batch_size 或 max_length
```

### 10.3 全部实验一键跑

确认单组没问题后：

```bash
# 训练 + 评测 + 汇总，全矩阵（含 SFT 对照组）
nohup python scripts/eval/run_experiment.py --all --with-sft > matrix.log 2>&1 &
tail -f matrix.log
```

**预期**：12 个步骤全部 `✓`（导出任务集 / 构造数据 / 基线评测 /
训练+评测 × 4 组实验 / 汇总）。

> `--with-sft` 会额外跑 SFT 对照组，它走 `scripts/train/sft_train.py`
> （不是 DPO 脚本 —— 两者的训练目标不同）。

---

## 11. 汇总与下载（30 分钟）

```bash
# 汇总（默认只读真实结果；不加 --include-mock）
python scripts/eval/collect_results.py

# 看一下主表
cat experiments/summary/ablation_table.md
```

**预期**：生成
`experiments/summary/ablation_table.md`（可直接贴进 README）、
`ablation_table.json`、以及 `docs/figures/` 下的对比图。

### 下载结果（关机前必做！）

```bash
cd /root/autodl-tmp
tar -czf results.tar.gz \
    agent-optimization/experiments \
    agent-optimization/outputs \
    agent-optimization/docs/figures \
    agent-optimization/data/preferences
du -sh results.tar.gz
```

然后从 AutoDL 控制台下载 `results.tar.gz`。

**必须下载的**：
- `experiments/*/eval_report.json` —— 报告的唯一数据来源
- `experiments/summary/ablation_table.md` —— 汇总表
- `outputs/*/final/` —— LoRA 适配器（想复现评测时需要）
- `outputs/*/train_health.txt` + `train_log.jsonl` —— 训练健康度证据
- `docs/figures/*.png`

> ⚠️ **关机不删数据盘，但数据盘按 ¥0.1/GB/天 计费。**
> 结果下载完、确认无误后，可以释放实例。

---

## 12. 把结果发给我

下载完后，把 `results.tar.gz` 解压覆盖回本地项目目录，然后：

```powershell
cd "D:\HuaweiMoveData\Users\马琬蕊\Desktop\项目2\agent-optimization"
python tests/run_tests.py          # 确认本地仍健康
python scripts/eval/collect_results.py   # 本地重新汇总一次
```

**然后发我这几样**（可只发关键部分）：

```powershell
# 1) 主表
Get-Content experiments\summary\ablation_table.md

# 2) 各实验的 headline
python -c "
import json, glob
for f in sorted(glob.glob('experiments/*/eval_report.json')):
    d = json.load(open(f, encoding='utf-8'))
    print(f, d['headline'])
"
```

拿到后我会：
1. 把 `docs/diagnosis/baseline_diagnosis.md` 的 **23 处占位符**填上真实数字
2. 把 `docs/ablation/ablation_report.md` 的消融表填完，并写结果分析
3. 更新 README 第 7 节
4. **重点报告与预期不符的地方** —— 包括若 `sft_only` 对照组推翻了核心论点

---

## 13. 故障排查表

| 症状 | 可能原因 | 处理 |
|---|---|---|
| `torch.cuda.is_available()` = False | 镜像是 CPU 版 torch | 换镜像重租 |
| 模型下载卡住 / 超时 | 没设 HF 镜像 | `export HF_ENDPOINT=https://hf-mirror.com`；已有模型可设 `HF_HUB_OFFLINE=1` |
| `ModuleNotFoundError: trl` | 没装或装错版本 | `pip install "trl==0.13.0"` |
| `TypeError` 构造 DPOTrainer 失败 | TRL 版本不匹配 | 看报错里打印的兼容层报告；锁 `trl==0.13.0` |
| **CUDA out of memory** | batch/max_length 过大 | 降 `--batch-size 1`，或按体检建议降 `max_length`，或确认 `gradient_checkpointing: true` |
| loss 一直贴着 0.6931 | 数据两侧语义等价 / 被截断成相同 | 跑 `inspect_preferences.py`，看审计与截断退化率 |
| loss 下降但评测指标不动 | prompt 分布不一致 | 跑 `inspect_preferences.py` 的一致性检查 |
| 训练日志里没有 `step` 行 | 监控回调没挂上 | 检查日志是否有「已挂载训练监控回调」；没有则看它后面的告警 |
| 训练结束打印 ⚠️（loss 偏离 log 2 / margin 不上升） | 偏好对数据问题 | **先修数据再重训**，别硬跑完四组；跑 `inspect_preferences.py` |
| `train_health.txt` 说 accuracy 停在 0.5 | 模型没学到偏好 | 检查 lr 是否过小、β 是否过大（β 大→z 被压小→梯度小） |
| 格式正确率 = 0% | chat template 破坏了 `<tool_call>` | **停下报我**，不要继续烧机时 |
| 评测结果文件名带 `.mock` | `AGENTOPT_FORCE_MOCK=1` 残留 | `unset AGENTOPT_FORCE_MOCK` |
| `collect_results.py` 说"没有找到任何评测报告" | 只有 mock 结果，默认被拒绝 | 确认 `eval_report.json`（无 `.mock`）存在 |
| 断线后任务没了 | 没用 nohup / tmux | 用 `nohup ... &` 或 `tmux new -s exp` |
| 显存够但速度极慢 | 数据在系统盘、或 CPU 抢占 | 确认代码与模型都在 `/root/autodl-tmp` |

---

## 14. 一句话速查（明天照着敲）

```bash
# ---- 租机后 ----
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"   # 必须 True
mkdir -p /root/autodl-tmp && cd /root/autodl-tmp
source /etc/network_turbo; export HF_ENDPOINT=https://hf-mirror.com

# ---- 上传解压后 ----
cd /root/autodl-tmp/agent-optimization
pip install -q "transformers>=4.45.0,<4.50.0" peft datasets "trl==0.13.0" matplotlib pyyaml
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir /root/autodl-tmp/models/Qwen2.5-1.5B-Instruct
export AGENTOPT_MODEL_LOCAL=/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct

# ---- 冒烟（必须过）----
python tests/run_tests.py
python scripts/eval/run_baseline.py --config configs/eval/eval_baseline.yaml --label baseline_smoke --task-limit 8 --num-samples 2

# ---- 正式流程 ----
nohup python scripts/eval/run_baseline.py --config configs/eval/eval_baseline.yaml > baseline.log 2>&1 &
python scripts/eval/analyze_errors.py --label baseline_airline
python scripts/prepare/build_preferences.py
python scripts/prepare/inspect_preferences.py       # 记下 max_length 建议
python scripts/train/dpo_train.py --experiment dpo_joint --dry-run
nohup python scripts/eval/run_experiment.py --all --with-sft > matrix.log 2>&1 &
python scripts/eval/collect_results.py

# ---- 下载 ----
tar -czf results.tar.gz agent-optimization/experiments agent-optimization/outputs agent-optimization/docs/figures
```

---

## 15. 明天开工时可以直接问我

把这两样发我，我就知道该接着做什么：

1. `nvidia-smi` 的输出（确认卡型与显存）
2. 第 6 步冒烟验证的日志（确认链路通畅）

如果冒烟就出问题，**先别继续**，把报错发我 —— 机上排查比在本地猜快得多。
