#!/usr/bin/env bash
# ============================================================================
# 四组消融一键跑完（Week 3 的入口）
# ============================================================================
# 用法:
#   bash scripts/eval/run_ablation.sh                 # 全部（基线 + 4 个训练实验）
#   bash scripts/eval/run_ablation.sh --baseline-only # 只跑基线
#   bash scripts/eval/run_ablation.sh --mock          # CPU 冒烟（不需要 GPU）
#   bash scripts/eval/run_ablation.sh --skip-train    # 只重新评测已有 checkpoint
#   bash scripts/eval/run_ablation.sh --dry-run       # 只看会执行什么
#
# 设计说明:
#   run_experiment.py 是唯一的执行编排器，本脚本只是它的 GPU 环境包装：
#   设置 CUDA / HF 相关环境变量、把日志落盘、然后用 nohup 可选后台运行。
#   这样「实验流程」只有一处定义，不会出现 shell 与 python 两套逻辑漂移。
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_ENDPOINT="${AGENTOPT_HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export AGENTOPT_RUN_TAG="${AGENTOPT_RUN_TAG:-$(hostname)-$(date +%m%d-%H%M)}"

LOG_DIR="experiments/summary"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/ablation_$(date +%Y%m%d-%H%M%S).log"

echo "=============================================================="
echo "消融实验矩阵"
echo "=============================================================="
echo "  运行标识 : ${AGENTOPT_RUN_TAG}"
echo "  CUDA     : ${CUDA_VISIBLE_DEVICES}"
echo "  参数     : $*"
echo "  日志     : ${LOG_FILE}"
echo "=============================================================="

# GPU 冒烟：先确认设备可见
if [[ "$*" != *"--mock"* && "$*" != *"--dry-run"* ]]; then
    python - <<'PY'
import sys
try:
    import torch
except ImportError:
    print("  ✗ torch 未安装。若只想验证流水线，请加 --mock"); sys.exit(1)
print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("  ✗ 未检测到可用 GPU。若只想验证流水线，请加 --mock"); sys.exit(1)
print(f"  GPU: {torch.cuda.get_device_name(0)}")
free, total = torch.cuda.mem_get_info()
print(f"  显存: {free/1024**3:.1f} GB free / {total/1024**3:.1f} GB total")
PY
fi

python scripts/eval/run_experiment.py "$@" 2>&1 | tee -a "${LOG_FILE}"

echo
echo "=============================================================="
echo "完成。汇总结果:"
echo "  experiments/summary/ablation_table.md     ← 可直接贴进 README"
echo "  experiments/summary/ablation_table.json"
echo "  docs/figures/ablation_comparison.png"
echo "  docs/figures/pass_k_curve.png"
echo "  日志: ${LOG_FILE}"
echo "=============================================================="
