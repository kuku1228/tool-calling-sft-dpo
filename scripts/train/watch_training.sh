#!/usr/bin/env bash
# 训练进度监控 —— 区分「正常完成」与「异常退出」
#
# 为什么需要区分：早期版本的监控只判断「进程是否还在」，进程一消失就打印
# 「训练已结束」。而 DPO 训练在 step 50 因评测 OOM 崩过一次 ——
# 脚本却报告「已结束」，看起来像跑完了。真实状态要靠**产物**判断：
#
#   正常完成 → 有 outputs/<exp>/final/ + 训练的收尾日志
#   异常退出 → 没有 final/，且日志尾部有 Traceback
#
# 用法：
#   bash scripts/train/watch_training.sh                # 默认看 dpo_joint
#   bash scripts/train/watch_training.sh sft_only 5     # 实验名 + 间隔秒数
set -u

EXP="${1:-dpo_joint}"
INTERVAL="${2:-30}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

LOG="${EXP}.log"
OUT="outputs/${EXP}"
TOTAL="${TOTAL_STEPS:-248}"

echo "监控实验: ${EXP}   日志: ${LOG}   间隔: ${INTERVAL}s"
echo "（Ctrl+C 只停止监控，不会杀掉训练）"
echo

finished=0
for _ in $(seq 1 400); do
    if ! pgrep -f "dpo_train.py.*--experiment ${EXP}" > /dev/null 2>&1 \
       && ! pgrep -f "sft_train.py.*--experiment ${EXP}" > /dev/null 2>&1; then
        finished=1
        echo "──────────────────────────────────────────────────────────"
        if [ -d "${OUT}/final" ] && ! grep -q "Traceback" "$LOG" 2>/dev/null; then
            echo "✓ 训练正常完成（${OUT}/final 已生成，日志无 Traceback）"
        elif grep -q "Traceback" "$LOG" 2>/dev/null; then
            echo "✗ 训练异常退出 —— 日志里有 Traceback："
            echo
            grep -nE "Error|OutOfMemory|Exception" "$LOG" | tail -5
            echo
            echo "--- 日志尾部 12 行 ---"
            tail -12 "$LOG"
        else
            echo "✗ 进程消失，但既没有 final/ 也没有 Traceback —— 可能被外部杀掉"
            echo "--- 日志尾部 12 行 ---"
            tail -12 "$LOG"
        fi
        break
    fi

    # 进度条（形如 50/248 [10:42<42:25）
    prog=$(grep -oE "[0-9]+/${TOTAL} \[[0-9:]+<[0-9:]+" "$LOG" 2>/dev/null | tail -1)
    [ -z "$prog" ] && prog="(尚未开始)"

    # 最近一条监控记录
    last=$(python - "$OUT" <<'PY' 2>/dev/null
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "train_log.jsonl"
if p.exists():
    rows = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
    if rows:
        d = rows[-1]
        f = lambda k: ("%.4f" % d[k]) if isinstance(d.get(k), (int, float)) else "  n/a "
        print("step=%-4s loss=%s margins=%s acc=%s" % (
            d.get("step"), f("loss"), f("rewards/margins"), f("rewards/accuracies")))
PY
)
    [ -z "$last" ] && last="(无记录)"

    gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1)

    echo "$(date +%H:%M:%S)  [${prog}]  ${last}  | GPU ${gpu}"

    # 评测失败是已知风险点：提前把显存情况打出来
    if [ -n "$prog" ]; then
        step=$(echo "$prog" | grep -oE "^[0-9]+")
        if [ "${step:-0}" -ge 45 ] && [ "${step:-0}" -le 55 ]; then
            echo "    ↑ 接近 eval_steps=50 —— 这是上次 OOM 的位置，注意观察"
        fi
    fi

    sleep "$INTERVAL"
done

if [ "$finished" -eq 0 ]; then
    echo "(监控循环到达上限，训练可能仍在进行；重新运行本脚本继续观察)"
fi

echo
echo "=========================================================="
echo "最终产物"
ls -la "${OUT}/" 2>/dev/null | head -12
[ -f "${OUT}/train_health.txt" ] && { echo; echo "--- 训练健康度 ---"; cat "${OUT}/train_health.txt"; }
