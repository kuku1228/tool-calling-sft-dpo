#!/usr/bin/env bash
# ============================================================================
# Mock 流水线（CPU 无需 GPU，上 GPU 前的第一道关）
# ============================================================================
# 用法:
#   bash scripts/train/mock/run_mock.sh            # 完整规模
#   bash scripts/train/mock/run_mock.sh --quick    # 约 10 秒
#
# 这一步跑通只说明「代码没坏」，不说明「方法有效」。
# 所有产物都带 .mock 标记，汇总脚本默认拒绝读取。
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

# Mock 路径强制不碰 GPU：避免误用真实模型，也避免在无 GPU 机器上报错
export CUDA_VISIBLE_DEVICES=""
export AGENTOPT_FORCE_MOCK=1

echo "=============================================================="
echo "Mock 流水线（CPU）"
echo "=============================================================="
echo "  FORCE_MOCK=1  CUDA_VISIBLE_DEVICES=（已清空）"
echo "=============================================================="

echo
echo "[1/2] 单元测试"
python tests/run_tests.py

echo
echo "[2/2] 端到端流水线"
python scripts/train/mock/run_all_mock.py "$@"

echo
echo "=============================================================="
echo "完成。产物位于 outputs/mock_pipeline/ 与 experiments/mock/"
echo "⚠️ 全部为 SIMULATED 数据，不能作为项目结论。"
echo "=============================================================="
