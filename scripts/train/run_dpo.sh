#!/usr/bin/env bash
# ============================================================================
# 真实 DPO 训练脚本（在 GPU 机器上运行）
# ============================================================================
# 用法:
#   bash scripts/train/run_dpo.sh dpo_joint              # 训练联合方案
#   bash scripts/train/run_dpo.sh dpo_format 2           # 指定实验与 epoch 数
#   MODE=dry bash scripts/train/run_dpo.sh dpo_joint     # 只做数据体检
#
# 设计说明:
#   - 所有环境变量都可被外部覆盖（不硬编码路径），方便换机器
#   - HF_HUB_OFFLINE / HF_ENDPOINT 显式处理，避免训练中途卡在联网
#   - 训练前先 dry-run 做数据体检，确认 max_length 再真跑（省 GPU 钱）
#   - 用 nohup + 日志文件跑，断连不丢训练
# ============================================================================
set -euo pipefail

EXPERIMENT="${1:-dpo_joint}"
EPOCHS="${2:-}"

# ---- 仓库根（脚本所在目录的上两级）----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# ---- 可覆盖的环境变量 ----
# 模型：优先用本地路径，避免每个 step 都走 HF 网络
export AGENTOPT_MODEL_LOCAL="${AGENTOPT_MODEL_LOCAL:-}"
export AGENTOPT_HF_ENDPOINT="${AGENTOPT_HF_ENDPOINT:-https://hf-mirror.com}"
export AGENTOPT_RUN_TAG="${AGENTOPT_RUN_TAG:-$(hostname)-$(date +%m%d-%H%M)}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_ENDPOINT="${AGENTOPT_HF_ENDPOINT}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
# 设 HF_HUB_OFFLINE=1 可强制离线（模型已下好时更稳）
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

# 显存碎片整理，缓解长序列训练的 OOM
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODE="${MODE:-train}"
CONFIG="configs/train/${EXPERIMENT}.yaml"
OUT_DIR="outputs/${EXPERIMENT}"
LOG_DIR="${OUT_DIR}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d-%H%M%S).log"

echo "=============================================================="
echo "DPO 训练"
echo "=============================================================="
echo "  仓库根       : ${REPO_ROOT}"
echo "  实验         : ${EXPERIMENT}"
echo "  配置         : ${CONFIG}"
echo "  运行标识     : ${AGENTOPT_RUN_TAG}"
echo "  CUDA 设备    : ${CUDA_VISIBLE_DEVICES}"
echo "  模型本地路径 : ${AGENTOPT_MODEL_LOCAL:-（未设置，用 HF 名称）}"
echo "  HF 端点      : ${HF_ENDPOINT}"
echo "  日志         : ${LOG_FILE}"
echo "=============================================================="

# ---- 环境自检 ----
echo
echo "[1/4] 环境自检"
python - <<'PY'
import sys
print(f"  python {sys.version.split()[0]}")
try:
    import torch
    print(f"  torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info()
        print(f"  显存: {free/1024**3:.1f} GB free / {total/1024**3:.1f} GB total")
except ImportError:
    print("  ✗ torch 未安装")
    sys.exit(1)
for m in ("transformers", "trl", "peft", "datasets"):
    try:
        mod = __import__(m)
        print(f"  {m} {getattr(mod, '__version__', '?')}")
    except ImportError:
        print(f"  ✗ {m} 未安装")
        sys.exit(1)
PY

# ---- 数据体检（先确认 max_length，再花 GPU 时间）----
echo
echo "[2/4] 数据体检（dry-run）"
python scripts/train/dpo_train.py \
    --experiment "${EXPERIMENT}" \
    --config "${CONFIG}" \
    --dry-run \
    2>&1 | tee -a "${LOG_FILE}"

if [ "${MODE}" = "dry" ]; then
    echo
    echo "MODE=dry：体检完成，未开始训练。"
    exit 0
fi

# ---- 训练 ----
echo
echo "[3/4] 训练"
EXTRA_ARGS=()
if [ -n "${EPOCHS}" ]; then
    EXTRA_ARGS+=(--epochs "${EPOCHS}")
fi
# 若体检给出了建议的 max_length，可用环境变量覆盖
if [ -n "${MAX_LENGTH:-}" ]; then
    EXTRA_ARGS+=(--max-length "${MAX_LENGTH}")
fi
if [ -n "${MAX_PROMPT_LENGTH:-}" ]; then
    EXTRA_ARGS+=(--max-prompt-length "${MAX_PROMPT_LENGTH}")
fi

python scripts/train/dpo_train.py \
    --experiment "${EXPERIMENT}" \
    --config "${CONFIG}" \
    --output-dir "${OUT_DIR}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee -a "${LOG_FILE}"

# ---- 训练后评测 ----
echo
echo "[4/4] 训练后评测"
python scripts/eval/run_baseline.py \
    --adapter "${OUT_DIR}/final" \
    --label "${EXPERIMENT}" \
    --out-dir "experiments/${EXPERIMENT}" \
    2>&1 | tee -a "${LOG_FILE}"

echo
echo "=============================================================="
echo "完成"
echo "  LoRA 适配器 : ${OUT_DIR}/final"
echo "  评测报告    : experiments/${EXPERIMENT}/eval_report.json"
echo "  训练日志    : ${LOG_FILE}"
echo
echo "汇总（把所有实验一起对比）:"
echo "  python scripts/eval/collect_results.py"
echo "=============================================================="
