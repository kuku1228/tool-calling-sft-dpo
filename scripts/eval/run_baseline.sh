#!/usr/bin/env bash
# ============================================================================
# 基线评测（在 GPU 机器上运行，Week 1 的入口）
# ============================================================================
# 用法:
#   bash scripts/eval/run_baseline.sh                       # 全部 100 任务
#   bash scripts/eval/run_baseline.sh 20                    # 只用前 20 个任务（快速验证）
#   LABEL=baseline_lite EXTRA="--lite-system" bash scripts/eval/run_baseline.sh
#   ADAPTER=outputs/dpo_joint/final LABEL=dpo_joint bash scripts/eval/run_baseline.sh
#
# 设计说明:
#   - 评测协议（采样数、温度、k 值）由 configs/eval/*.yaml 统一提供，
#     脚本不重复定义，避免命令行与配置漂移导致历史结果失去可比性
#   - 跑完自动做错误分析并出图，一步到位
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TASK_LIMIT="${1:-0}"
LABEL="${LABEL:-baseline}"
CONFIG="${CONFIG:-configs/eval/eval_baseline.yaml}"
ADAPTER="${ADAPTER:-}"
NUM_SAMPLES="${NUM_SAMPLES:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_ENDPOINT="${AGENTOPT_HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

OUT_DIR="experiments/${LABEL}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/eval_$(date +%Y%m%d-%H%M%S).log"

echo "=============================================================="
echo "基线 / checkpoint 评测"
echo "=============================================================="
echo "  标签     : ${LABEL}"
echo "  配置     : ${CONFIG}"
echo "  适配器   : ${ADAPTER:-（无，用原始基座）}"
echo "  任务上限 : ${TASK_LIMIT:-0}（0=全部）"
echo "  采样次数 : ${NUM_SAMPLES:-（用配置默认）}"
echo "  输出     : ${OUT_DIR}"
echo "  日志     : ${LOG_FILE}"
echo "=============================================================="

ARGS=(
    --config "${CONFIG}"
    --label "${LABEL}"
    --out-dir "${OUT_DIR}"
)
if [ -n "${ADAPTER}" ]; then
    ARGS+=(--adapter "${ADAPTER}")
fi
if [ "${TASK_LIMIT}" != "0" ]; then
    ARGS+=(--task-limit "${TASK_LIMIT}")
fi
if [ -n "${NUM_SAMPLES}" ]; then
    ARGS+=(--num-samples "${NUM_SAMPLES}")
fi
# 允许通过 EXTRA 追加任意参数，例如 --lite-system / --tools xxx
if [ -n "${EXTRA:-}" ]; then
    # shellcheck disable=SC2206
    ARGS+=(${EXTRA})
fi

echo
echo "[1/2] 评测"
python scripts/eval/run_baseline.py "${ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"

echo
echo "[2/2] 错误分析"
python scripts/eval/analyze_errors.py \
    --persist-dir "${OUT_DIR}" \
    2>&1 | tee -a "${LOG_FILE}"

echo
echo "=============================================================="
echo "完成"
echo "  报告      : ${OUT_DIR}/eval_report.json"
echo "  错误分析  : ${OUT_DIR}/error_analysis.json"
echo "  失败案例  : ${OUT_DIR}/failure_cases/"
echo "  图表      : docs/figures/"
echo "  日志      : ${LOG_FILE}"
echo
echo "下一步: python scripts/prepare/build_preferences.py"
echo "=============================================================="
