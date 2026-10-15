# 环境搭建与运行手册

## 文档定位

这份文档回答的问题是：**从一台什么都没装的机器开始，怎么把这个项目跑到出结果；以及跑挂了怎么定位。**

读者是：（1）第一次 clone 下来想验证代码的人（很可能没有 GPU）；（2）要上 AutoDL 租卡跑真实实验的人；（3）训练崩了 / 结果不对，来找排查步骤的人。

三条前提：

1. 本项目按**单卡 3090 / 4090（24G）**设计。没有多卡、没有 DeepSpeed、没有 online RL。
2. **真实 GPU 实验尚未运行过**。本文里的显存数字是**估算**，不是实测；命令清单是设计意图，不是「已验证跑通」的记录。凡属估算的地方都标了。
3. 命令分两类：**CPU 冒烟**（不需要 GPU，确定性可验证）与 **GPU 真实运行**（需要模型与显卡）。两类不要混着看。

相关文档：

- [`docs/ops/trl_version_notes.md`](trl_version_notes.md) —— TRL 版本问题的全部细节
- [`docs/method/preference_data_design.md`](../method/preference_data_design.md) —— 数据构造与质量门禁
- [`docs/references/DPO_tool_calling_research_memo.md`](../references/DPO_tool_calling_research_memo.md) —— 超参与 API 的一手核实
- [`README.md`](../../README.md) —— 项目概览与实验状态

---

## 1. 两种环境

本项目的依赖清单按**运行路径**分段，不是「都要装」。核心设计约束在 `requirements.txt` 的注释里写得很明确：

> 为了让 `python scripts/train/mock/run_all_mock.py` 在一台没有任何深度学习栈的机器上也能端到端跑通，核心模块只依赖标准库 + numpy。

| | CPU 冒烟环境 | GPU 训练环境 |
|---|---|---|
| **目的** | 验证代码正确、流水线通畅、数据可构造 | 跑真实基线评测与 DPO 训练 |
| **依赖** | `numpy>=1.24.0`、`pandas>=2.0.0`（`matplotlib` 可选） | 上面全部 + `torch`、`transformers`、`trl`、`peft`、`datasets`、`accelerate`、`tensorboard`、`PyYAML` |
| **磁盘** | < 1 GB | 模型权重约 3.1 GB + checkpoint + HF 缓存，预留 ≥ 40 GB |
| **时间** | 测试 2 秒，快速冒烟约 10 秒 | 基线评测约 10 分钟起；单次 DPO 训练按小时计 |
| **能否得出项目结论** | ❌ **绝对不能** | ✅（这才是唯一的数据来源） |

### 1.1 CPU 冒烟环境

```bash
# 最小可用：只跑测试与 mock 流水线
python -m pip install --upgrade pip
pip install "numpy>=1.24.0" "pandas>=2.0.0"

# 可选：出图（缺了不会报错，脚本会降级为不画图）
pip install "matplotlib>=3.7.0"

# 可选：让 configs/*.yaml 用 yaml.safe_load 而不是 json.loads 解析
# （不装也行，项目的 yaml 都写成 JSON 兼容子集，见 §2）
pip install "PyYAML>=6.0"
```

或者用 `pyproject.toml` 的 extra：

```bash
pip install -e ".[core]"
```

验证：

```bash
python tests/run_tests.py                          # 期望：48 通过, 0 失败（约 2s）
python scripts/train/mock/run_all_mock.py --quick  # 期望：退出码 0，末尾打印"✓ 流水线跑通"
```

**实测（本次审计环境）**：Python 3.12.9 / numpy 2.3.5 / pandas 3.0.0rc0 / PyYAML 已装 →
`python tests/run_tests.py` 输出 `结果: 48 通过, 0 失败, 用时 2.05s`。**没有跑 `run_all_mock.py`**（它会写 `outputs/mock_pipeline/`，本次审计不写文件）。

### 1.2 GPU 训练环境

```bash
# 1) torch：必须与 CUDA 版本匹配。3090/4090 是 sm_86，cu121 最稳。
#    AutoDL 镜像通常已预装，先检查再决定装不装：
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 2) 其余依赖
pip install "transformers>=4.45.0,<4.50.0" \
            "peft>=0.7.0" \
            "datasets>=2.16.0" \
            "accelerate>=0.25.0" \
            "tensorboard>=2.15.0" \
            "PyYAML>=6.0"

# 3) TRL：⚠️ 见下方注意事项
pip install "trl==0.13.0"
```

或者：

```bash
pip install -e ".[gpu]"
```

#### ⚠️ 安装 TRL 时的注意事项

`requirements.txt` 当前写的是 `trl>=0.9.0,<0.13.0`，而项目文档（本文、README、`compat.recommended_pins()`）声称锁 **0.13.0**。**`<0.13.0` 排除了 0.13.0。**

```bash
pip install -r requirements.txt   # → 会装到 0.12.x，与文档不一致
pip install "trl==0.13.0"         # → 与文档一致（推荐）
```

两者的差别、以及为什么推荐 0.13.0，见 [`trl_version_notes.md`](trl_version_notes.md) §4。**这一条不影响流水线能否跑通（`compat.py` 在 0.12.x 上也能工作），但会让 `compat_report.json` 里记的版本与文档不符。**

#### 安装后自检

```bash
python -c "from src.training.compat import make_compat_report,recommended_pins,warn_if_versions_drift;r=make_compat_report();print(r.summary());print();print('推荐:',recommended_pins());print('警告:',warn_if_versions_drift(r))"
```

（上面假定当前目录是仓库根，`src` 可导入。若不在仓库根，加 `-c` 前先 `cd` 过去，或临时设 `PYTHONPATH=.`。）

期望在 `summary()` 里看到 `processing_class : True` 与 `DPOConfig : True`，且**没有** `⚠️ 被丢弃的字段`。如果出现了 `dropped_fields`，先解决它再训练——否则你设的超参会被静默忽略（见 `trl_version_notes.md` §7.3）。

---

## 2. 为什么 Mock 路径只依赖标准库 + numpy

这不是「省事」，是这个项目的**主要工程约束**之一。设计意图有四层：

### 2.1 让没有 GPU 的人也能验证整条流水线

项目的对标目标是「诊断 → 数据构造 → 消融 → 结论」这个研究闭环。其中**除了「策略模型」之外，每一环都是真实代码**：

| 环节 | mock 路径是否真实 | 说明 |
|---|---|---|
| 工具调用解析 | ✅ 真实 | `src/eval/tool_call_parser.py` |
| 语义等价判定 | ✅ 真实 | `src/eval/equivalence.py` |
| 评分与错误分类 | ✅ 真实 | `src/eval/grader.py` + `error_taxonomy.py` |
| 偏好数据构造与质量门禁 | ✅ 真实 | `preference_builder.py`（**与 GPU 路径完全同一份代码**） |
| DPO 损失 | ✅ 真实 | `dpo_loss.py` 纯 Python 实现 + 10 项解析自检 |
| 训练循环与曲线 | ✅ 真实优化 | `mock_trainer.py`，用解析梯度真的下降 |
| 评测指标与置信区间 | ✅ 真实 | `metrics.py`（任务级 bootstrap） |
| 画图与汇总 | ✅ 真实 | `plots.py`、`collect_results.py` |
| **策略模型** | ❌ **合成** | `src/mock/mock_policy.py` |

也就是说：**mock 路径里被替换掉的只有「模型」这一个组件。** 所以它能抓出真实 bug。README 记录了它实际抓到的 6 个，其中三个是纯代码问题（错误类型键大小写不一致、偏好对去重键过粗、变量名遮蔽），两个是方法论问题（成功概率与错误率脱钩、纠错话术与错误类型不匹配），一个是数值问题（`log_sigmoid` 负值区精度损失）。

**这些 bug 在 GPU 上要烧掉几小时才会暴露**：因为 GPU 路径每改一次代码就要重新跑一遍评测。CPU 路径把「代码对不对」和「模型学得好不好」彻底解耦。

### 2.2 实现手段：延迟导入 + 零重依赖的 config

1. **`src/config.py` 只依赖标准库**（docstring 原文）：「YAML 解析优先用 PyYAML，缺失时降级到 `json.loads`（本项目的 yaml 配置文件都写成 JSON 兼容子集）。这样 `import src.config` 在裸 CPU 环境永远成功。」
2. **第三方库全部延迟导入**。`torch` / `transformers` / `trl` / `peft` / `datasets` 只在真正需要时在函数体内 import。所以 `python -c "from src.config import describe; print(describe())"` 在没有 torch 的机器上会打印 `深度学习栈 : torch 未安装（Mock 模式可用）`，而不是崩掉。
3. **画图是可降级项**。`collect_results.py` 里对应的是 `if made: print("图表:") else: print("（未生成图：matplotlib 不可用）")`。

### 2.3 「yaml 写成 JSON 兼容子集」是有意保留的降级路径

`configs/*.yaml` 的内容实际上都是合法 JSON（键加双引号、不用锚点、不用多行标量），所以 `load_config_file` 可以：

```python
try:
    import yaml
    return yaml.safe_load(text) or {}
except ImportError:
    return json.loads(text)
```

**这不是临时妥协。** 它保证「装没装 PyYAML」不改变任何超参的解析结果——而在训练项目里，一个把 `1.0e-5` 读成字符串 `"1.0e-5"` 的解析差异足以让实验白跑。

### 2.4 mock 的边界：它证明不了什么

`run_all_mock.py` 跑通**只说明代码没坏**，不说明方法有效。仓库为此做了三重隔离（详见 §7.2）。`MockPolicyCfg.baseline_error_rates`（`FORMAT_ERROR 0.180` / `HALLUCINATION 0.105` / `WRONG_TOOL 0.075` / `PARAM_ERROR 0.160` / `NO_RECOVERY 0.060`）与各实验的 `mitigation_factors` 是**待验证的假设值**，不是结论——它们唯一的用途是让「改善某一类错误 → 准确率上升」这条因果链在流水线里可执行、可自检。**真实数字一律来自 `experiments/*/eval_report.json`。**

---

## 3. AutoDL / 云 GPU 部署

### 3.1 系统盘 vs 数据盘：模型与 checkpoint 都放数据盘

AutoDL 实例的典型布局：

| 路径 | 性质 | 容量 | 用途 |
|---|---|---|---|
| `/root/` | **系统盘** | 通常 30~50 GB，与镜像共享 | 代码、配置。**不要放模型权重** |
| `/root/autodl-tmp/` | **数据盘** | 通常 50~100 GB，按量计费 | 模型权重、HF 缓存、训练 checkpoint、日志 |

为什么这条很重要：

- Qwen2.5-1.5B-Instruct 的 bf16 权重约 **3.1 GB**。看起来不大，但 HF 的默认缓存 `~/.cache/huggingface/` 在**系统盘**上；试过 0.5B / 1.5B / 3B 或者反复重下，系统盘很容易被填满。
- 训练产物会持续增长：`outputs/<exp>/final/`（LoRA adapter，很小）、`base_with_init_adapter/`（**一个完整模型的副本，约 3.1 GB**）、`checkpoint-*/`（`save_total_limit=3` 意味着同时存在 3 份）、`logs/`（tensorboard）。
- **系统盘满了会让训练在保存 checkpoint 时崩掉**，而此时可能已经跑了 40 分钟。这个失败模式完全可以预防。

推荐做法：

```bash
# 1) 把 HF 缓存放数据盘（放进 ~/.bashrc 持久化）
export HF_HOME=/root/autodl-tmp/hf && mkdir -p "$HF_HOME"

# 2) 把模型下到数据盘，再用 AGENTOPT_MODEL_LOCAL 指过去
mkdir -p /root/autodl-tmp/models
export AGENTOPT_MODEL_LOCAL=/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct

# 3) 把 outputs/ 挪到数据盘（仓库本身在系统盘时）
mkdir -p /root/autodl-tmp/agentopt-outputs && ln -sfn /root/autodl-tmp/agentopt-outputs outputs
```

**注意第 3 步的软链接要在 `run_dpo.sh` 创建 `outputs/${EXPERIMENT}` 之前建好**，否则它会在系统盘上创建真实目录。

### 3.2 环境变量覆盖机制

设计原则在 `src/config.py` 里写明：

> **可覆盖**：所有字段都能用环境变量覆盖（见 `_env_override`），方便在 AutoDL 上不改进仓库代码就切模型路径。

`apply_env_overrides()` 在模块导入时被调用（`CFG = Config().apply_env_overrides()`），所以**在 `python` 进程启动之前** export 才会生效。

| 环境变量 | 映射到 | 说明 |
|---|---|---|
| `AGENTOPT_MODEL` | `CFG.model.name` | 换模型名，例如 `Qwen/Qwen2.5-3B-Instruct` |
| `AGENTOPT_MODEL_LOCAL` | `CFG.model.local_path` | 本地权重路径。`ModelCfg.resolved` = `local_path or name`，**优先于 name** |
| `AGENTOPT_HF_ENDPOINT` | `CFG.model.hf_endpoint` | 同时 `os.environ.setdefault("HF_ENDPOINT", ...)` |
| `AGENTOPT_DEVICE` | `CFG.model.device` | `auto` / `cuda` / `cpu` / `mps` |
| `AGENTOPT_DTYPE` | `CFG.model.dtype` | 默认 `bfloat16` |
| `AGENTOPT_RUN_TAG` | `CFG.run_tag` | 实验标识，会写进所有产物 |
| `AGENTOPT_SEED` | `CFG.seed` | **只接受纯数字**（`seed_env.isdigit()` 检查），并且**只覆盖顶层 `CFG.seed`**，不改 `CFG.train.seed` / `CFG.eval` 里的随机性 |
| `AGENTOPT_FORCE_MOCK` | （模块级 `FORCE_MOCK`） | ⚠️ **见下方警告** |

映射表的实际代码（`apply_env_overrides` 里的 `mapping`）：

```python
mapping = {
    "AGENTOPT_MODEL":        ("model", "name"),
    "AGENTOPT_MODEL_LOCAL":  ("model", "local_path"),
    "AGENTOPT_HF_ENDPOINT":  ("model", "hf_endpoint"),
    "AGENTOPT_DEVICE":       ("model", "device"),
    "AGENTOPT_DTYPE":        ("model", "dtype"),
    "AGENTOPT_RUN_TAG":      ("",      "run_tag"),
}
```

#### ⚠️ `AGENTOPT_FORCE_MOCK` 不在这张表里

```python
FORCE_MOCK: bool = os.environ.get("AGENTOPT_FORCE_MOCK", "0") not in ("0", "", "false", "False")
```

它只在模块级被读成一个常量，而 `FORCE_MOCK` 全仓库**只有一处使用**——`describe()` 里打印 `强制 Mock : {FORCE_MOCK}`。**所以 `AGENTOPT_FORCE_MOCK=1` 只会改变一行打印输出，不会让任何加载真实模型的代码路径退化为 mock。** `src/config.py` 里那条 docstring（「所有会加载真实模型的代码路径都会退化为 mock」）与实现不符。**要跑 mock 必须显式加 `--mock`：**

```bash
python scripts/eval/run_baseline.py --mock --mock-policy baseline
python scripts/train/dpo_train.py --experiment dpo_joint --mock
python scripts/eval/run_experiment.py --all --mock
```

#### `AGENTOPT_RUN_TAG` 的实际作用

`run_dpo.sh` 给它一个默认值（`${AGENTOPT_RUN_TAG:-$(hostname)-$(date +%m%d-%H%M)}`），然后**只用于打印**。它进了 `CFG.run_tag`，但**目前没有脚本把它写进产物**——`train_config.json` 写的是 `experiment` / `model_name` / `init_adapter` / 超参，不含 `run_tag`。要靠它区分「同一实验的多次运行」，得自己记下来。

### 3.3 HF 镜像端点与离线模式

国内机器直连 huggingface.co 基本不可用。两种策略：

```bash
# 策略 A：走镜像（首次下载模型时）
export AGENTOPT_HF_ENDPOINT=https://hf-mirror.com
# config.py 在 hf_endpoint 非空时执行 os.environ.setdefault("HF_ENDPOINT", ...)
# run_dpo.sh 已把它设为默认值：export HF_ENDPOINT="${AGENTOPT_HF_ENDPOINT}"

# 策略 B：强制离线（模型已下好之后，推荐）
export HF_HUB_OFFLINE=1        # run_dpo.sh 里默认是 0
```

**为什么训练时应该设 `HF_HUB_OFFLINE=1`**：训练脚本会在**训练中途**重新解析模型名（尤其是 `--init-adapter` 之后要 `save_pretrained` 再加载）。此时网络抖动会让 HF 客户端挂起重试，表现为「训练卡住不动」而不是报错。它与 `AGENTOPT_MODEL_LOCAL` 配合最好——本地路径已指向数据盘上的权重，网络完全不需要参与。两者互斥的用法是「首次下载」：先不带它下模型，下完再加。

推荐组合：

```bash
export AGENTOPT_MODEL_LOCAL=/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct
export AGENTOPT_HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1        # 权重已下好，彻底断网
```

`HF_HUB_OFFLINE=1` 与「首次下载」互斥——先不带它下模型，下完之后再加。

### 3.4 缓解长序列 OOM

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

`run_dpo.sh` 里已经默认设置：

```bash
# 显存碎片整理，缓解长序列训练的 OOM
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
```

**它解决的是什么问题**：PyTorch 默认的 CUDA 分配器按块缓存，长序列训练会产生大量「大小不一、生命周期交错」的临时张量。结果是**总空闲显存够、但没有一块连续空间能满足一次大分配**，于是报 OOM，而 `torch.cuda.memory_summary()` 显示还有好几个 GB 空着。`expandable_segments:True` 让分配器可以扩展已有段而不是要求整块新段，专门治这个。

**它解决不了什么**：真实的显存不足。如果 `max_length` 真的超出了卡的容量，加这个环境变量只会让 OOM 来得晚一点、报错信息更难懂。显存预算见 §4。

其他相关设置（本项目已配好）：

| 设置 | 位置 | 作用 |
|---|---|---|
| `gradient_checkpointing=True` | `TrainCfg` / `configs/*` | 用计算换显存，是长序列训练的关键 |
| `bf16=True` + `fp16=False` | `compat.build_dpo_config` | 3090/4090 是 Ampere，支持 bf16，比 fp16 数值稳 |
| `dataloader_num_workers=0` | `compat.build_dpo_config` | 避免多 worker 复制数据与 tokenizer，减少主机内存与偶发挂起 |
| `remove_unused_columns=False` | `compat.build_dpo_config` | DPO 的列名特殊，关掉更安全 |
| `save_total_limit=3` | `TrainCfg` | 控制 checkpoint 数量，防止数据盘被吃满 |
| `TOKENIZERS_PARALLELISM=false` | `run_dpo.sh` | 消除 fork 警告（多进程 tokenizer 的经典噪声） |

### 3.5 用 `nohup` 跑训练：断连不丢

AutoDL 的 Web SSH 会超时断连。训练脚本要脱离终端：

```bash
cd /root/agent-optimization      # 换成你的仓库路径

# 方式 1：run_dpo.sh 自己就把日志 tee 到文件
nohup bash scripts/train/run_dpo.sh dpo_joint > /dev/null 2>&1 &
# 日志会同时在 outputs/dpo_joint/train_<时间戳>.log

# 方式 2：手动跑，日志重定向
mkdir -p outputs/dpo_joint
nohup python scripts/train/dpo_train.py \
    --experiment dpo_joint \
    --config configs/train/dpo_joint.yaml \
    --output-dir outputs/dpo_joint \
    > outputs/dpo_joint/train_$(date +%Y%m%d-%H%M%S).log 2>&1 &

# 记下 PID
echo $! > outputs/dpo_joint/train.pid
```

`run_dpo.sh` 的日志机制值得说明——它把四步都 tee 到**同一个文件**（`LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d-%H%M%S).log"`，每个 `python ... 2>&1 | tee -a "${LOG_FILE}"`）。所以「环境自检 → dry-run 体检 → 训练 → 训练后评测」的完整输出在一个文件里，事后排查不需要拼多个日志。

```bash
tail -f outputs/dpo_joint/train_*.log                 # 看进度
grep -E "loss=|step |⚠️|✗|错误" outputs/dpo_joint/*.log # 只看关键行
ps aux | grep dpo_train | grep -v grep                # 断连后确认还活着
```

`run_dpo.sh` 开头有 `set -euo pipefail`，所以任何一步失败都会让整个脚本退出。这既是保护也是坑：**如果训练成功后评测失败，你不会看到「训练完成」的总结块**，但 adapter 已经落盘了（在 `[3/4]` 之后、`[4/4]` 之前）。这时手动补跑评测即可。

---

## 4. 显存预算估算

### 4.1 ⚠️ 先说清这份估算的性质

**以下全部是估算，不是实测。** 本项目**没有在真机上跑过任何 DPO 训练**。估算的价值在于「帮你决定第一次试哪个 `max_length`」，不在于「告诉你确切的显存占用」。

真实数值必须用这两步确认：

```bash
# 第 1 步：数据侧 —— 量出真实 token 长度分布，拿到推荐的 max_length
python scripts/train/dpo_train.py --experiment dpo_joint --dry-run
# 产物：outputs/dpo_joint/data_health.json 的 token_length 与 recommended_max_length 段

# 第 2 步：训练侧 —— 用推荐的 max_length 真跑一次，看峰值显存
python scripts/train/dpo_train.py --experiment dpo_joint --max-steps 2
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

`run_dpo.sh` 的 `[1/4] 环境自检` 段也会用 `torch.cuda.mem_get_info()` 打印卡上当前显存。

### 4.2 为什么 DPO 的显存是普通 LM 训练的 2~3 倍

`TrainCfg` 里就写着这个结论（`#: DPO 的 logprob 计算峰值显存约是普通 LM 的 2-3 倍，序列长度要收着点。`）。原因是 **DPO 每次前向要算两套 log-prob**：

| | 普通 LM 训练 | DPO |
|---|---|---|
| policy 前向（chosen + rejected） | 1 次 | 1 次 |
| policy 反向 | 1 次 | 1 次 |
| **reference 前向（chosen + rejected）** | — | **1 次**（`torch.no_grad()`） |
| 显存里要留的中间量 | policy 的激活 | policy 的激活 **+** reference 的激活 + 两套 log-prob |

具体到 TRL 的实现（`DPOTrainer.concatenated_forward` / `compute_ref_log_probs`）：

1. **policy 与 reference 各跑一次「chosen+rejected 拼接」的前向**。LoRA + `ref_model=None` 时 reference 通过 `disable_adapter()` 在**同一个模型**上跑，所以权重只有一份（约 3.1 GB），但**激活与 logits 是两份**。
2. **reference 那次在 `torch.no_grad()` 下**，不建反向图、不存反向激活——这是它比 policy 那次便宜的地方。
3. **loss 需要四个标量**：`logp_policy_chosen` / `logp_policy_rejected` / `logp_ref_chosen` / `logp_ref_rejected`。
4. **`batch_size=2` 实际意味着 4 条序列**（2 选 + 2 拒）。这是最容易误算的一点。

TRL 0.13.0 会尽量只保留最后一个位置的 logits（避免不断生成 `[B, S, 152k]` 这种大张量）。所以**logits 通常不是主瓶颈，激活才是**——这也是为什么 `max_length` 的影响比 `batch_size` 更显著。

### 4.3 估算表：LoRA r=16/all-linear + β=0.1 + 1.5B 模型

配置前提（全部来自本项目的真实配置）：

| 项 | 值 | 来源 |
|---|---|---|
| 模型 | Qwen2.5-1.5B-Instruct，28 层，hidden 1536，intermediate 8960 | 模型结构 |
| 精度 | bf16（2 字节/参数） | `TrainCfg.bf16=True` |
| 基座权重 | 约 **3.1 GB**（1.54B × 2B） | — |
| LoRA | `r=16`，`lora_alpha=32`，`target_modules="all-linear"` → 可训练参数约 **18.5M** | `TrainCfg.lora_r/lora_alpha/lora_target_modules` |
| LoRA 参数 + 梯度 + AdamW 状态 | 约 **0.3 GB**（fp32 master 74 MB + bf16 36 MB + 梯度 36 MB + AdamW 一阶二阶 148 MB） | 估算 |
| batch | `per_device_train_batch_size=2` × `gradient_accumulation_steps=8` → 有效 batch 16，**但单次前向只有 2** | `TrainCfg` |
| gradient checkpointing | 开启 | `TrainCfg=True` |
| CUDA context + 分配器碎片 | 约 **0.5 ~ 1.0 GB** | 经验值 |

按 `max_length`：

| `max_length` | 激活与临时量（估） | 常驻（权重+优化器+context） | **合计（估）** | 24G 卡余量 | 建议 |
|---|---|---|---|---|---|
| **1024** | ~0.8 GB | ~4.0 GB | **~4.8 GB** | 很宽裕 | 冒烟与快速迭代的首选。也够跑通全流程 |
| **1536** | ~1.2 GB | ~4.0 GB | **~5.2 GB** | 宽裕 | **本项目默认值**。建议第一次真跑就用它 |
| **2048** | ~1.6 GB | ~4.0 GB | **~5.6 GB** | 宽裕 | 如果 `--dry-run` 建议值落在这里，放心用 |
| **3072** | ~2.4 GB | ~4.0 GB | **~6.4 GB** | 仍宽裕 | 只有在 `prompt+completion p99 > 2048` 时才需要 |

**⚠️ 这张表看起来「显存远远用不完」，与项目里「`max_length` 要收着点」的谨慎口径不一致。** 两者都可能对，因为估算里最不确定的是「激活」这一项：

- 激活按 `gradient_checkpointing=True` 估（只存层输入、反向时重算）。**如果 checkpointing 没生效**（例如 `gradient_checkpointing_enable()` 被后续调用覆盖），激活会涨到数倍，`max_length=3072` 就可能真的接近 24 GB。
- 表里假设 `batch_size=2`。**提到 4 或 8，激活量近似线性增长。**
- 没算 `--init-adapter` 路径的额外开销：那一步会 merge adapter 并 `save_pretrained` 一个完整模型副本到磁盘（不占显存），但 merge 过程中会短暂出现两份权重视图。

**正确用法：用这张表选一个「肯定安全」的起点（1024 或 1536），真跑一次量出峰值，再决定要不要往上加。** 不要因为表里写着 6.4 GB 就直接上 3072。本项目默认 `1536`，是保守选择。

三个「不用改代码就能省显存」的旋钮，按性价比排序：①**降 `max_length`**（最有效，激活近似线性）；②**`per_device_train_batch_size` 降到 1、`gradient_accumulation_steps` 翻到 16**（有效 batch 不变，峰值激活减半）；③**`precompute_ref_log_probs=True`**（reference log-prob 训练前算好写回数据集，训练时不跑 reference 前向；本项目默认 `False`，因为 LoRA + `disable_adapter()` 本就不需要第二个模型常驻，省的不多）。

实测峰值（单卡 RTX 3090 24G，`nvidia-smi` 轮询）：

| 阶段 | 峰值显存 | 配置 |
|---|---|---|
| DPO 训练（`dpo_format`，1014 对） | **15.8 GB** | `max_length=1280`、`batch_size=2`、`grad_accum=8`、`gradient_checkpointing=True`、bf16 |
| DPO 训练（`dpo_joint`，2443 对） | 15.8 GB | 同上 |
| 评测（`run_baseline.py`，推理） | **2.9 GB** | `max_new_tokens=1536`、无优化器状态 |

对应 `data_health.json` 的长度实测：prompt+completion p99 = 1107，
因此 `max_length=1280` 有余量（截断率 0.0%）。

**结论**：24G 卡上 `max_length=1280` + `batch_size=2` 的峰值约 15.8 GB，
留出约 8 GB 余量。若要提到 `max_length=3072`，按估算表会接近 24 GB 上限，
需要先把 `batch_size` 降到 1 或开 `precompute_ref_log_probs`。

### 4.4 比估算更直接的判断方法

```bash
python scripts/train/dpo_train.py --experiment dpo_joint --dry-run
# 打印：长度 p99 / 当前 max_length 下的截断率 / 建议的 max_length
```

- **`截断后退化率 > 5%`** → 必须调大 `max_length`。退化的样本（截断后 chosen 与 rejected 相同）对 DPO 是纯噪声，loss 恒为 `log 2`。
- **`prompt+comp p99` 远小于 `max_length`** → 可把 `max_length` 降到 p99 附近，省显存换更大的 batch。
- **`prompt p99 > max_prompt_length`** → 改 `max_prompt_length`，不是 `max_length`（0.13.0 上两者分工不同，见 [`trl_version_notes.md`](trl_version_notes.md) §2.1）。

---

## 5. 完整运行顺序（Week 1 → Week 4）

每一步都给出**预期产出**，用来判断「跑完了没有」。前置：

```bash
cd /root/agent-optimization      # 或你的本地仓库路径
export AGENTOPT_MODEL_LOCAL=/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct
export AGENTOPT_HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1          # 权重下好之后再设
```

### 1. `python tests/run_tests.py`（CPU，~2s）

改代码之后立刻跑。期望 `结果: 48 通过, 0 失败`、退出码 0。**失败就先别往下走**——核心库有问题，后面的都不可信。

### 2. `python scripts/train/mock/run_all_mock.py`（CPU）

上 GPU 之前的第一道关。执行「评测 → 错误分类 → 偏好数据构造 → mock 训练（真实 DPO 损失 + 真实梯度下降）→ 再评测 → 出图」。

- **`--quick` 约 10 秒**：它把 `num_samples_per_task` 降到 2、`primary_k` 降到 2、任务数降到 30、`n_format=100` / `n_recovery=100` / `n_tool_selection=60`、mock 训练 30 步。不带 `--quick` 时用 `CFG.preference` 默认值（1200 / 1200 / 600）、mock 训练 120 步。
- 脚本开头先跑 DPO 损失自检，**自检失败直接退出码 1**，不继续跑流水线。这是有意的——数学错了后面全是浪费时间。
- 预期产出：`outputs/mock_pipeline/` 下的 `pipeline_summary.mock.json`、`predictions_*.mock.jsonl`、`preferences_<exp>.{train,dev}.mock.jsonl`、`train/<exp>/{train_log.jsonl, training_curves.json, final/adapter_config.json, train_config.json}`、`eval_report_<exp>.mock.json`；`docs/figures/` 下若干 png；终端末尾 `✓ 流水线跑通` + 退出码 0。
- **通过了也只说明「代码没坏」，不是「方法有效」。**

### 3. `python scripts/prepare/dump_tasks.py`（CPU）

Week 1 第一步，必须在任何评测之前。

- 预期产出：`experiments/tasks_airline_v1.json`（约 80 KB，100 任务）、`experiments/tasks_airline_v1_tools.json`（约 8 KB，6 工具 schema + 渲染好的文本手册）；终端打印任务集 summary、`BASE_DATE = 2025-07-14`、纠错轮与易错任务数。
- **为什么必须落盘**：任务集由 `src/data/task_manifest.py` **代码生成**。不同机器上代码版本有微小差异 → 任务不同 → 结果不可比。固化后可以直接 diff 两个版本的任务集，而不是 diff 代码。
- 默认会跑 `_self_check()`（ID 唯一、目标轮合法、期望工具在白名单、相对日期已换算），失败会中止。**不要用 `--no-check` 跳过。**

### 4. `python scripts/eval/run_baseline.py --config configs/eval/eval_baseline.yaml`（**GPU**）

Week 1。这是诊断数据的唯一来源，所有后续实验都与它对比。第一次真正加载模型。约 10 分钟起。

- 预期产出（默认 `experiments/baseline_airline/`）：`eval_report.json`（**报告只引用这个文件**）、`predictions.jsonl`（逐次采样原始输出，所有分析的唯一来源）、`error_analysis.json`（错误分布 + 格式正确率 + `format_ok_value_wrong` 退化监控）、`failure_cases/`（最多 40 个失败案例，含 `raw_output`）。
- **`label` 与输出目录的优先级**（`run_baseline.py`）：
  - label：显式 `--label` > adapter 目录的父目录名 > `mock_<policy>`（当 `--mock`）> `baseline`。
  - 输出目录：显式 `--out-dir` > 配置文件 `output.dir` > `experiments/<label>`。

  所以带 `--config configs/eval/eval_baseline.yaml` 时会落到配置里声明的 `experiments/baseline_airline/`；**不带 `--config` 时落到 `experiments/baseline/`**。两者不一致会让 `collect_results.py` 找不到结果——要么统一传 `--config`，要么统一传 `--label` + `--out-dir`。
- 评测某个 checkpoint 时显式给 label，避免目录名变成 adapter 的父目录名：

  ```bash
  python scripts/eval/run_baseline.py --adapter outputs/dpo_joint/final --label dpo_joint \
      --out-dir experiments/dpo_joint
  ```

### 5. `python scripts/eval/analyze_errors.py --label baseline`（CPU）

- 预期产出：更新 `experiments/baseline/error_analysis.json`（这一份含 `cross_tab` 与 `recovery` 段，步骤 4 那份没有）；`docs/figures/` 下的 `error_distribution_baseline.png`、`success_by_difficulty_baseline.png`、`success_by_visibility_baseline.png`；若 label 以 `baseline` 开头且**不是** mock，额外写一份不带 label 的默认名（方便 README 引用）。
- 终端会打印：错误类别分布、子类型 Top10、按难度/可见性交叉、**错误恢复分析**（恢复率 / 重复首轮错误率 / 换工具率 / 逐 `mistake_key` 明细）、每类错误的改进手段。
- **这一步是 Week 2 数据设计的事实基础**：如果诊断出的分布与 `CATEGORY_REMEDIES_ZH` 的假设不符，应该**先改假设**，而不是照原计划造数据。

实测基线（`experiments/baseline/error_analysis.json`，424 次采样）：

```
错误类别分布（占全部采样）：
  SUCCESS       38.9%      FORMAT_ERROR  33.3%
  PARAM_ERROR   26.2%      WRONG_TOOL     1.7%
  HALLUCINATION  0.0%      REFUSED/UNKNOWN 0.0%

占失败的比例：
  FORMAT_ERROR  54.4%      PARAM_ERROR   42.9%      WRONG_TOOL  2.7%
```

两类错误（格式 + 参数）覆盖 **97.3%** 的失败，而**工具幻觉为 0**。
这直接决定四类偏好对的规模取向：格式与参数类是主要矛盾，
工具选择类在基线阶段边际收益有限（但其 `chosen/rejected` 设计仍保留，
因为幻觉会在训练后阶段出现——见 `docs/ablation/results.md` §3）。

> 该次运行的 `error_analysis.json` 未包含 `recovery` 段（真实多轮恢复评测未运行），
> 因此没有 `recovery_rate` / `repeat_first_mistake_rate`。要得到它们需跑
> `scripts/eval/eval_recovery.py`。

### 6. `python scripts/prepare/build_preferences.py`（CPU）

Week 2。

- 预期产出（`data/preferences/`）：`all.{train,dev}.jsonl`；按实验切片的 `dpo_format.*` / `dpo_recovery.*` / `dpo_joint.*` / `dpo_tool.*` / `dpo_joint_param.*` / `dpo_all.*` / `sft_only.*` / `dpo_tool_param.*`（各一对 train/dev）；`build_report.json`（统计 + 质量门禁报告，可审计）。
- **尺寸预期**：默认目标是 `n_format=1200` / `n_recovery=1200` / `n_tool_selection=600` / `n_param=n_format=1200`，但**格式约束类有去重上限**——2025 年本次审计测得约 **924** 条（见 [`preference_data_design.md`](../method/preference_data_design.md) §7.2，那里也给了自己复测的命令）。所以默认目标下大约得到 924 / 1200 / 600 / 1200 ≈ 3924 条，而不是配置里写的 4200。
- `--dry-run` 不写文件，只打印统计与门禁报告，适合先探规模。
- 终端重点看：每类的「目标 N / 实得 M」——**如果实得远小于目标，问题在负样本多样性，调大 `n_*` 只会让 guard 空转**；以及门禁报告（`sides_semantically_equal` 通常是最大的一项，这是正常的）。- 若 `pairs` 为空，脚本返回 1 并提示「请检查错误率配置与质量门禁」。

### 7. `python scripts/prepare/inspect_preferences.py`（CPU，装了 transformers 才有真 token 长度）

**占用 GPU 之前必须做这一步。** 它回答四个问题（脚本 docstring）：①两侧语义等价；②chosen 格式不合规；③**prompt 与评测分布不一致**；④长度超出 `max_length`（截断后两侧可能变成一样）。

- 预期产出：**只打印，不写文件**。退出码 0 = 数据可用，1 = 有问题。
- 关键几行：

  ```
  ── token 长度检查
     tokenizer: Qwen/Qwen2.5-1.5B-Instruct（采样 N 条）
     prompt  token: p50=... p90=... p99=... max=...
     prompt+comp : p50=... p90=... p99=... max=...
     建议配置: max_length=... max_prompt_length=...
     截断率: prompt=...% completion=...%
     截断后退化(两侧相同)率: ...%
  ```

  以及：
  ```
  ── prompt 一致性检查（训练分布 vs 评测分布）
     ✓ 训练 prompt 与评测 prompt 前缀一致
  ```

- **`建议配置` 那两行就是你要写进 `--max-length` / `--max-prompt-length` 的值。**
- `rejected 含幻觉工具 N 条（预期行为：这是工具选择负样本）` —— 这条**不是**问题，脚本把它单独排除在 `problem_counts` 之外。

### 步骤 8 —— `python scripts/train/dpo_train.py --experiment dpo_joint --dry-run`

- **何时**：真正训练前的最后一道关，**不花 GPU 钱**。
- **需要什么**：numpy / pandas。装了 transformers 才有真实 token 长度体检。
- **它做什么**（三步）：DPO 损失自检 → 数据体检 → 退出（不训练）。
- **预期产出**：`outputs/dpo_joint/data_health.json`，含 `stats` / `audit` / `prompt_consistency` / `token_length` / `recommended_max_length` / `truncation` 六段。
- **终端要看到**：

  ```
  ✓ 数据审计通过
  ✓ prompt 一致性通过（检查 N 条）
  长度: prompt p99=...  prompt+comp p99=...  max=...
  当前 max_length=1536 / max_prompt_length=1024 → 截断率 completion=..., 截断后退化=...
  建议命令: 加 --max-length N --max-prompt-length M
  ```

- **`--dry-run` 报数据审计失败**是最常见的卡点。见 §6 的排查表。

### 步骤 9 —— `bash scripts/train/run_dpo.sh dpo_joint`

- **何时**：Week 3。
- **需要什么**：**GPU** + 全部 GPU 依赖。
- **它做四件事**：环境自检（torch / transformers / trl / peft / datasets + 显存）→ dry-run 数据体检 → 训练 → 训练后评测。
- **训练前核对数据来源**：脚本会优先使用 `--config configs/train/<exp>.yaml` 里 `pairs.train_file` / `dev_file` 声明的路径，找不到才回落到内置候选顺序。先确认它们指向**真实数据**而不是 mock 流水线的产物：

  ```bash
  Select-String -Path configs/train/*.yaml -Pattern 'train_file|dev_file'    # Windows
  grep -n 'train_file\|dev_file' configs/train/*.yaml                        # Linux
  ```

  **不要指向 `outputs/mock_pipeline/*.mock.jsonl`**——那是合成数据，用它做真实训练只会得到一份无法解释的结果。正确目标形如 `data/preferences/dpo_joint.train.jsonl`（由步骤 6 生成）。

- **预期产出**：
  - `outputs/dpo_joint/train_<时间戳>.log`（四步的全部输出）
  - `outputs/dpo_joint/base_with_init_adapter/`（**仅当传了 `--init-adapter`**）
  - `outputs/dpo_joint/final/` —— LoRA adapter（`adapter_config.json` + `adapter_model.safetensors`）
  - `outputs/dpo_joint/checkpoint-*/` —— 每 50 步，最多保留 3 份
  - `outputs/dpo_joint/logs/` —— tensorboard
  - `outputs/dpo_joint/train_config.json` —— 超参快照 + `final_metrics`
  - `outputs/dpo_joint/compat_report.json` —— **兼容层报告，含 `dropped_fields`**
  - `experiments/dpo_joint/eval_report.json` 等（`[4/4]` 评测）
- **sanity check**：`compat_report.json` 里 `dropped_fields` 必须是空数组。非空说明你设的超参被静默丢弃了。
- **建议先 2 步试跑**：`python scripts/train/dpo_train.py --experiment dpo_joint --max-steps 2`，确认能跑通、显存够、loss 是 `log 2` 附近起步，再上全量。
- **可选：SFT → DPO 续训**。如果先做了 LoRA SFT，用 `--init-adapter <sft_adapter_dir>` 把 SFT adapter merge 进 base 再新建 LoRA。理由见 [`trl_version_notes.md`](trl_version_notes.md) §6。

### 步骤 10 —— `python scripts/eval/run_experiment.py --all`

- **何时**：要跑完整消融矩阵时。
- **它做什么**（固定成一条流水线，避免「Exp1 用 4 次采样、Exp2 用 3 次」这类协议不一致）：

  1. `dump_tasks.py`（导出任务集）
  2. `build_preferences.py`（构造偏好数据）
  3. `run_baseline.py --label baseline`（基线）
  4. 对每个实验：`dpo_train.py` → `run_baseline.py --label <exp> --adapter outputs/<exp>/final`
  5. `collect_results.py`（汇总）

  默认的 `TRAIN_EXPERIMENTS = ("dpo_format", "dpo_recovery", "dpo_joint")`，加 `--with-sft` 会插入 `sft_only`。

- **预期产出**：`experiments/summary/run_record.json`（执行记录，含 `protocol` / `experiments` / `steps` / `config_hash` / `wall_time_s`）。
- **`config_hash` 的作用**：对 `configs/default.yaml` + `src/config.py` 做 sha256 取前 16 位。只靠时间戳不够——配置文件被改过但时间戳没变（例如 `git checkout` 回滚）是很常见的。
- **可以跳过已完成的步骤**：`--skip-baseline` / `--skip-prepare` / `--skip-train` / `--skip-eval`。
- **`--dry-run` 只打印将要执行的命令**，适合先核对一遍。
- **注意**：`run_experiment.py` 调 `dpo_train.py` 时会传 `--config configs/train/<exp>.yaml`，所以它用的是配置里声明的数据文件。跑 `--all` 之前先按步骤 9 那条命令核对三个 `configs/train/*.yaml` 的 `train_file` / `dev_file`。

### 步骤 11 —— `python scripts/eval/collect_results.py`

- **何时**：所有实验评测完成之后。
- **需要什么**：numpy / pandas。matplotlib 可选（缺了不出图）。
- **它做什么**：发现 `experiments/<exp>/eval_report.json`（以及 `experiments/mock/<exp>/eval_report.mock.json`），生成对比表 + 效应量 + 统计说明 + 图。
- **预期产出**（`experiments/summary/`）：`ablation_table.md`（**可直接贴进 README 第 7 节**）、`ablation_table.json`（机器可读，含 `headlines` / `by_difficulty` / `by_visibility` / `comparisons`）、`ablation_summary.txt`；以及 `docs/figures/` 下的 `ablation_comparison.png` 与 `pass_k_curve.png`。
- **它比「一张准确率表」多给了什么**：
  - **Δstability**（`pass^k − pass@k`，越接近 0 越稳定）。
  - **相对基线变化**与 **Cohen's h 效应量**（分「可忽略 / 小 / 中 / 大」四档）。
  - **统计说明**：用**任务级 bootstrap**（1000 次重采样任务）算 95% CI，而不是按采样样本重采样——同一任务的多次采样高度相关，按样本算会严重低估区间宽度。
  - **逐实验判定**：CI 与基线重叠 → 「**不能断言有真实提升**，需增加任务数或采样次数」；不重叠 → 「有统计支持」。
  - **样本量建议**：按观测到的差值反算「要在 80% 功效下检出该效应，每组约需 N 个任务」。
- **默认拒绝 mock**：必须显式加 `--include-mock`。加了之后输出文件名带 `.mock` 后缀、表格标题标注 SIMULATED。**带 mock 的汇总表不能作为项目结论。**
- 若报「没有找到任何评测报告」：`experiments/` 下没有 `eval_report.json`，先去跑步骤 4。

---

## 6. 常见问题排查表

| 症状 | 可能原因 | 处理 |
|---|---|---|
| **CUDA out of memory**（`max_length=1536` 就炸） | ① `gradient_checkpointing` 没生效；② 别的东西占了显存；③ `batch_size` 被改了 | ① `nvidia-smi` 看有没有残留进程（`kill` 掉）；② 把 `per_device_train_batch_size` 降到 1、`gradient_accumulation_steps` 翻到 16；③ 设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；④ 降 `max_length`。**先量再调，别猜**（§4.4） |
| **OOM 时显存显示还有几个 GB 空着** | 分配器碎片，不是真的不够 | `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（`run_dpo.sh` 已默认设） |
| **loss 停在 `log 2 ≈ 0.6931` 附近不动** | ① 数据里 chosen 与 rejected 语义等价（两侧一样 → `z ≡ 0`）；② **chosen/rejected 被标反了**；③ β 太小 + lr 太小，梯度幅度不足 | ① 跑 `inspect_preferences.py`，看 `sides_equivalent` 是否为 0；② 抽查 `data/preferences/<exp>.train.jsonl` 的前几条，确认 chosen 是"正确"那侧；③ `--beta 0.3` 或 `--lr 1e-4` 试一次。**注意 `log 2` 是 DPO 的正确起点**——训练刚开始时就应该在 0.6931，不是「卡住」。要看它有没有在 50~100 步内开始下降 |
| **loss 下降但评测指标不动** | ① **训练 prompt 与评测 prompt 分布不一致**（最可能）；② 截断把偏好对砍成了噪声；③ 学到了格式捷径 | ① 跑 `python scripts/train/dpo_train.py --experiment <exp> --dry-run`，看 `✓ prompt 一致性通过` 还是 `✗ prompt 不一致`；② 看 `data_health.json` 的 `truncation.degenerate_after_truncation_rate`，> 5% 就调大 `max_length`；③ 看 `error_analysis.json` 的 `format_ok_value_wrong_rate` 是否上升——升了就是走了捷径，需要增加硬负样本比例 |
| **`pass^k` 恒等于 accuracy** | 同一任务的多次采样结果完全相同，随机性被抹掉了 | ① 检查 `CFG.model.temperature`（默认 0.7，**必须 > 0**；0 会让 4 次采样结果一致，`pass^k ≡ accuracy`）；② 检查任务的 `difficulty` 是否真的产生了异质性——如果所有任务难度一致，`pass^k` 会趋近 `accuracy^k`；③ 看 `eval_report.json` 的 `per_task[i].outcomes`（长度 = 采样次数的布尔列表），确认同一 `task_id` 的 4 个 `outcomes` 不全相同 |
| **`--dry-run` 报数据审计失败** | `inspect_preferences.audit_rows` 发现了 `sides_equivalent` / `chosen_not_canonical` / `chosen_unknown_tool` / `prompt_missing_assistant_head` / `empty_side` 之一 | 看 `outputs/<exp>/data_health.json` 的 `audit.problem_counts` 与 `audit.examples`。`prompt_missing_assistant_head` 说明 prompt 没有以 `<|im_start|>assistant` 结尾（构造逻辑被改过）；`chosen_unknown_tool` 说明 chosen 用了幻觉工具名（严重，要查 `preference_builder`） |
| **找不到偏好数据**（`FileNotFoundError: 找不到训练数据`） | `data/preferences/<exp>.train.jsonl` 不存在，且 `configs/train/<exp>.yaml` 声明的 `pairs.train_file` 也不存在 | 先跑 `python scripts/prepare/build_preferences.py`。报错信息里会列出所有尝试过的候选路径，按那个清单核对 |
| **训练跑起来了，但数据来源可疑** | `configs/train/*.yaml` 的 `pairs.train_file` 指向 `outputs/mock_pipeline/*.mock.jsonl`（合成数据） | 用 `grep -n train_file configs/train/*.yaml` 核对。改成 `data/preferences/...`，或删掉这两行让它走 `resolve_data_file` 的内置候选顺序 |
| **评测结果目录出现在意料之外的位置** | `run_baseline.py` 的 label/out_dir 优先级与预期不同 | 优先级：label = 显式 `--label` > adapter 父目录名 > `mock_<policy>` > `baseline`；out_dir = 显式 `--out-dir` > 配置文件 `output.dir` > `experiments/<label>`。**传了 `--config` 与不传会落到不同目录**（见 §5 步骤 4） |
| **`trl` 版本报 `TypeError`**（构造 `DPOTrainer` 失败） | TRL 版本与 `compat.py` 的适配分支不匹配 | 本项目在版本不匹配时会抛出**带完整诊断**的 `TypeError`（含原始错误 / 实际传入的参数 / 签名接受集 / `CompatReport` 摘要）。把那段完整贴出来。排查步骤见 [`trl_version_notes.md`](trl_version_notes.md) §11 |
| **训练跑着跑着「卡住」不动** | HF 客户端在重试网络请求（不是死锁） | `export HF_HUB_OFFLINE=1`（前提：权重已在本地）。如果是首次下载模型，用 `AGENTOPT_HF_ENDPOINT=https://hf-mirror.com` 并等它下完 |
| **模型下载卡住 / 速度极慢** | 直连 huggingface.co 不可达 | `export AGENTOPT_HF_ENDPOINT=https://hf-mirror.com`。已经设了还慢 → 用 `huggingface-cli download` 单独下到 `/root/autodl-tmp/models/`，然后用 `AGENTOPT_MODEL_LOCAL` 指过去，之后设 `HF_HUB_OFFLINE=1` |
| **系统盘满了 / 保存 checkpoint 时崩** | 模型缓存与 outputs 都在系统盘 | 见 §3.1。`HF_HOME` 挪到数据盘 + `outputs/` 做软链接 |
| **Windows 上路径 / 编码问题** | 中文路径、GBK 控制台、反斜杠 | ① 仓库路径尽量不含中文（本项目的开发路径里含中文，在 Linux 上没问题，但 Windows 工具链偶发异常）；② 控制台乱码是 GBK 问题，设 `chcp 65001` 或 `$env:PYTHONIOENCODING="utf-8"`；③ **不要在 Windows 上跑 `run_dpo.sh`**（bash 脚本），Windows 上只用 CPU 冒烟路径，训练上 Linux；④ 所有文件读写都用 `encoding="utf-8"`（本项目已统一） |
| **`bash scripts/train/run_dpo.sh` 报 `command not found: python`** | AutoDL 镜像里可能只有 `python3` | 改脚本里的 `python` 为 `python3`，或建别名 |
| **`train_config.json` 里的 β 与命令行传的不一样** | `configs/train/<exp>.yaml` 的 `train.beta` 覆盖了 `CFG.train.beta`，而 CLI 的参数优先级在两者之上 | 优先级是 **命令行 > 配置文件 > 全局默认**（`dpo_train.py` 的 `train_kwargs` 构造逻辑）。同时确认 `compat_report.json` 的 `dropped_fields` 为空——非空说明版本丢弃了你的设置 |
| **`collect_results.py` 报「没有找到任何评测报告」** | `experiments/` 下没有 `eval_report.json` | 先跑基线评测（步骤 4）。或者你就是想看 mock 结果 → 加 `--include-mock`，但要知道那不是结论 |
| **`collect_results.py` 报「混有 MOCK 与真实结果」** | 部分实验只有 `.mock.json`、部分有真实 report | 对比表会失去意义。`--include-mock` 会让 mock 结果优先被选中——**在已经有真实结果的目录里不要加这个参数** |
| **`analyze_errors.py` 报 `FileNotFoundError: 找不到 predictions.jsonl`** | 该 label 目录下只有 `.mock` 版本 | `analyze_errors.py` 硬编码读 `predictions.jsonl`（**不读** `predictions.mock.jsonl`），且只读 `eval_report.json`（不读 `.mock.json`）。所以它**只适用于真实结果** |
| **图表没生成，但脚本没报错** | matplotlib 不可用 | 这是设计好的降级行为。`pip install matplotlib` 即可 |

---

## 7. 产物目录说明

### 7.1 四个顶层目录各放什么

| 目录 | 内容 | 是否入库 | 说明 |
|---|---|---|---|
| **`outputs/`** | **训练产物**：LoRA adapter、checkpoint、训练曲线、tensorboard 日志、配置快照、`compat_report.json`、`data_health.json`、mock 流水线的全部产物 | ❌ 大文件 | `.gitignore` 排除 `outputs/**/checkpoint-*/`、`outputs/**/*.safetensors`、`optimizer.pt`、`scheduler.pt`、`rng_state.pth`、`trainer_state.json` |
| **`experiments/`** | **评测任务定义 + 评测结果**：`tasks_airline_v1.json`（固化的 100 任务集）、`<exp>/eval_report.json`、`predictions.jsonl`、`error_analysis.json`、`failure_cases/`、`summary/` | ✅ **入库**（核心证据） | `.gitignore` 里 `experiments/**/eval_report.json` 是被注释掉的，附注说明「但它是项目核心证据，默认保留。若过大可取消下面注释」 |
| **`data/`** | **生成的偏好数据**：`preferences/*.jsonl`、`build_report.json`；`sft/`、`raw/` 为预留 | ❌ 不入库 | `.gitignore` 排除 `data/preferences/*.jsonl`、`data/sft/*.jsonl`、`data/raw/*.jsonl`（保留 `.gitkeep`）。**数据能用代码重新生成，所以不入库；但 `build_report.json` 也不入库，意味着门禁统计只存在于本地** |
| **`docs/figures/`** | **所有图表的输出目录** | ✅ 入库 | `make_all_figures` / `plot_*` 系列的写入目标。`configs/*.yaml` 里没有单独的配图目录配置——`collect_results.py` 有 `--figures-dir`（默认 `docs/figures`），`analyze_errors.py` 同样 |
| **`logs/`** | `Config.ensure_dirs()` 会创建，但**当前没有脚本往里写**（`run_dpo.sh` 的日志写到 `outputs/<exp>/`） | — | 预留 |
| **`configs/`** | yaml 配置（JSON 兼容子集） | ✅ 入库 | `default.yaml` + `train/` + `eval/` |

### 7.2 `eval_report.json`（真实）vs `eval_report.mock.json`（模拟）

这是整个项目**最重要的一条隔离线**。

| | `eval_report.json` | `eval_report.mock.json` |
|---|---|---|
| **产生者** | `run_baseline.py`（不加 `--mock`，真实模型） | `run_baseline.py --mock` 或 `mock_pipeline` |
| **`metadata.engine`** | `"hf"` | `"mock"` |
| **`metadata.is_mock`** | `false` | `true` |
| **配套文件** | `predictions.jsonl`、`error_analysis.json`、`failure_cases/` | `predictions.mock.jsonl`、`error_analysis.mock.json` |
| **位置** | `experiments/<exp>/` | `experiments/mock/<exp>/`（或与真实结果同目录，用后缀区分） |
| **报告能否引用** | ✅ **唯一合法来源** | ❌ **绝不能作为项目结论** |
| **`collect_results.py` 默认行为** | 读取 | **拒绝读取**（需显式 `--include-mock`） |
| **`analyze_errors.py`** | 读取 | **不读取**（硬编码文件名） |

**为什么 mock 结果绝不能当结论**（`collect_results.py` docstring 的原话）：

> 原因：mock 的指标是我们**喂进去的假设**。如果它们混进正式报告，整个项目就变成了「自己写死结论再自己验证」。

具体地说：mock 的准确率与 pass^k 完全由 `MockPolicyCfg.mitigation_factors` 与 `recovery_gain` 这两张表决定。表里写着 `dpo_joint` 的 `FORMAT_ERROR: 0.35`，那 mock 跑出来格式错误就会降 65%——**这是输入，不是发现**。把这条曲线放进报告，读者以为在看在真实模型上的效果，实际上是在看自己写下的假设。

三重隔离机制（README 的原话）：①所有 mock 产物文件名带 `.mock` 后缀；②产物内部带 `"is_mock": true`；③`collect_results.py` 默认拒绝读取 mock 报告，必须显式加 `--include-mock`，即使加了输出文件名与表格标题也标注 SIMULATED。

还有一条**防止覆盖**的保护：`run_baseline.py` 发现目标目录已存在 `eval_report.json`（且其 `metadata.is_mock` 为假）时，会打印警告并**只写 `.mock.json`，绝不覆盖真实结果**。

**`data/preferences/build_report.json` 里的 `is_mock` 是另一回事**：它是**硬编码 `False`**。含义是「这是真实的数据构造，不是模拟」——确实为真，因为 `preference_builder` 在 mock 与真实路径下是同一份代码。但这也意味着：**把 `--out-dir` 指到 `outputs/mock_pipeline/` 时，那份 `build_report.json` 仍会写 `is_mock: false`。** 靠这个字段区分真假数据不可靠，要看目录名。

### 7.3 一个完整的产物清单（真实 GPU 路径跑完之后）

```
agent-optimization/
├── configs/                                    ← 入库
│   ├── default.yaml
│   ├── eval/eval_baseline.yaml
│   └── train/{dpo_format,dpo_recovery,dpo_joint}.yaml
├── data/preferences/                           ← 不入库，可重新生成
│   ├── all.{train,dev}.jsonl
│   ├── {dpo_format,dpo_recovery,dpo_joint,sft_only,
│   │    dpo_tool,dpo_joint_param,dpo_all,dpo_tool_param}.{train,dev}.jsonl
│   └── build_report.json                       门禁审计（也不入库）
├── docs/figures/                               ← 入库
│   ├── error_distribution.png                  基线专用，不带 label
│   ├── error_distribution_baseline.png
│   ├── success_by_difficulty.png / success_by_visibility.png
│   ├── ablation_comparison.png / pass_k_curve.png / training_curves.png
├── experiments/                                ← 入库（核心证据）
│   ├── tasks_airline_v1.json                   100 任务集
│   ├── tasks_airline_v1_tools.json             6 工具 schema
│   ├── baseline/  dpo_format/  dpo_recovery/  dpo_joint/
│   │   ├── eval_report.json                    ★ 报告只引用这个
│   │   ├── predictions.jsonl
│   │   ├── error_analysis.json
│   │   └── failure_cases/case_*.json
│   └── summary/{ablation_table.md, ablation_table.json,
│                ablation_summary.txt, run_record.json}
└── outputs/                                    ← 不入库（大文件）
    ├── dpo_joint/
    │   ├── final/adapter_config.json           LoRA adapter
    │   ├── checkpoint-50/100/150/             每 50 步，最多 3 份
    │   ├── logs/                              tensorboard
    │   ├── train_<时间戳>.log
    │   ├── train_config.json                  超参快照 + final_metrics
    │   ├── compat_report.json                 ★ 检查 dropped_fields
    │   └── data_health.json                   ★ 长度体检与建议值
    └── mock_pipeline/                         mock 流水线（全部带 .mock）
```

---

## 8. 环境搭建的验收清单

上 GPU 之前按顺序确认这 8 项。任何一项不过，先停下。

| # | 检查 | 通过标准 |
|---|---|---|
| 1 | `python tests/run_tests.py` | `48 通过, 0 失败` |
| 2 | `python scripts/train/mock/run_all_mock.py --quick` | 退出码 0，末尾 `✓ 流水线跑通` |
| 3 | `python -c "import torch;print(torch.cuda.is_available())"` | `True` |
| 4 | `make_compat_report().summary()` | `processing_class : True`、`DPOConfig : True`、**无 `⚠️ 被丢弃的字段`** |
| 5 | `pip show trl` 的版本 | 与 [`trl_version_notes.md`](trl_version_notes.md) §4 的说法一致（知道自己在哪个版本上） |
| 6 | `python scripts/prepare/dump_tasks.py` | 写出 `experiments/tasks_airline_v1.json`，自检通过 |
| 7 | `python scripts/prepare/inspect_preferences.py` | 退出码 0、`✓ 数据可用`，记下建议的 `max_length` |
| 8 | `python scripts/train/dpo_train.py --experiment dpo_joint --dry-run` | `✓ 数据审计通过` + `✓ prompt 一致性通过`，记下 `data_health.json` 的截断率 |

第 8 项之后，用第 7 项给出的长度做一次 2 步试跑：

```bash
python scripts/train/dpo_train.py --experiment dpo_joint \
    --max-length <建议值> --max-prompt-length <建议值> --max-steps 2
```

确认三件事：**不 OOM**、**loss 从 `log 2 ≈ 0.6931` 附近起步**、**`outputs/dpo_joint/compat_report.json` 的 `dropped_fields` 为空**。三项都过了，再上全量训练。
