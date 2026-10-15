"""训练进度监控 —— 一条命令，跑完自动报告。

## 为什么需要它（踩过的坑）

1. **不能只看进程是否存活。** 进程消失可能是**正常完成**，也可能是 **OOM 崩溃**。
   唯一可靠的判据是**产物目录**（``outputs/<run>/final``）是否生成。
   实测踩过：进程没了、以为完成、实际崩了。

2. **不能用 ``| tail`` 管道看输出。** 它会把输出全藏到结束才显示 ——
   训练要 30 分钟，等于全程看不见进度。

3. **必须检查截断率。** 实测踩过 23.2% 的 prompt 被静默截断，
   而 loss 曲线、进度条**全都正常**。

## 用法

```bash
# 前台监控（每 30 秒刷新一行，训练结束自动打印结果）
python scripts/train/monitor.py outputs/sft_v6 sft_v6.log

# 或后台跑，自己看日志
nohup python -u scripts/train/monitor.py outputs/sft_v6 sft_v6.log > monitor.log 2>&1 &
```

**退出码**：0 = 正常完成；1 = 异常结束（无 ``final/``）；2 = 超时仍在跑。
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]

#: 训练进度行，例如 ``  120/336 [10:52<19:24, ...]``
PROGRESS_RE = re.compile(r"(\d+)/(\d+)\s*\[(\d+:\d+)<(\d+:\d+)")

#: 日志里必须确认的项（banner 后的开头部分）
HEADER_CHECKS = (
    ("训练样本", r"输入偏好对\s*(\d+)\s*条"),
    ("SFT 样本", r"SFT 样本\s*(\d+)\s*条"),
    ("按类型", r"按类型:\s*(\{[^}]*\})"),
    ("截断率", r"截断率\s*=?\s*([\d.]+%)"),
    ("max_length", r"max_length\s*=?\s*(\d+)"),
)


def process_alive(pattern: str) -> bool:
    """训练进程是否还在。"""
    try:
        r = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
        return r.returncode == 0
    except FileNotFoundError:
        # Windows / 无 pgrep：退化为看产物
        return True


def last_progress(log: Path) -> tuple[int, int, str] | None:
    if not log.exists():
        return None
    last = None
    with log.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = PROGRESS_RE.search(line)
            if m:
                last = (int(m.group(1)), int(m.group(2)), m.group(4))
    return last


def header_report(log: Path) -> list[str]:
    """从日志开头抽取必须确认的项。"""
    if not log.exists():
        return ["  （日志尚未生成）"]
    head = "".join(log.open(encoding="utf-8", errors="replace").readlines()[:80])
    out = []
    for name, pat in HEADER_CHECKS:
        m = re.search(pat, head)
        out.append(f"    {name:<12} {m.group(1) if m else '⚠️ 未找到'}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="训练进度监控")
    ap.add_argument("out_dir", help="产物目录，如 outputs/sft_v6")
    ap.add_argument("log", help="训练日志路径")
    ap.add_argument("--pattern", default="sft_train.py|dpo_train.py|dose_response.py",
                    help="pgrep 匹配的训练进程名")
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--timeout-min", type=int, default=180)
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    log = Path(args.log)
    final = out_dir / "final"

    print("=" * 66)
    print(f"监控 {out_dir}   （每 {args.interval}s 刷新）")
    print("=" * 66)
    print("判据：**产物 final/ 是否生成** —— 不是看进程是否存活")
    print("-" * 66)

    # ⚠️ 头部检查**必须在循环里反复重试**，不能只在启动时读一次。
    #
    # 实测踩过：把 `train.py &` 和 `monitor.py` 连着敲，监控启动时
    # 训练还没写出任何日志 → 5 项全报「⚠️ 未找到」，看起来像监控坏了。
    # 实际只是**时序问题**：日志要 ~10 秒才出现前几行。
    #
    # 所以：头部没读全就继续等，读全了才停。
    header_done = False
    header_shown: list[str] = []
    t0 = time.time()
    while not header_done and (time.time() - t0) < 120:
        rep = header_report(log)
        if all("未找到" not in x and "尚未生成" not in x for x in rep):
            header_shown = rep
            header_done = True
            break
        if time.time() - t0 > 3:          # 前 3 秒安静等待
            print(f"  [{time.strftime('%H:%M:%S')}] 等待日志头部…")
        time.sleep(1)

    if header_shown:
        print("日志开头的关键项：")
        for ln in header_shown:
            print(ln)
        print()
    else:
        print("⚠️ 120 秒内没读到日志头部 —— 检查日志里是否有报错：")
        for ln in header_report(log):
            print(ln)
        try:
            for ln in log.open(encoding="utf-8",
                               errors="replace").readlines()[:25]:
                print("   ", ln.rstrip()[:110])
        except Exception:
            pass
        print()

    t0 = time.time()
    while True:
        alive = process_alive(args.pattern)
        prog = last_progress(log)
        if not alive:
            print()
            print("-" * 66)
            if final.exists():
                print("✓ 训练完成（产物已生成）")
                print(f"  产物: {final}")
                # 打印最后几行
                try:
                    tail = log.open(encoding="utf-8",
                                    errors="replace").readlines()[-8:]
                    print("  日志末尾：")
                    for ln in tail:
                        print("   ", ln.rstrip()[:100])
                except Exception:
                    pass
                return 0
            print("✗ **异常结束**：进程已退出但产物 final/ 不存在")
            print("   最可能的原因：OOM 或数据错误")
            try:
                tail = log.open(encoding="utf-8",
                                errors="replace").readlines()[-25:]
                print("  日志末尾（25 行）：")
                for ln in tail:
                    print("   ", ln.rstrip()[:110])
            except Exception:
                pass
            return 1

        if prog:
            done, total, eta = prog
            pct = 100 * done / max(1, total)
            print(f"[{time.strftime('%H:%M:%S')}] {done}/{total} "
                  f"({pct:.0f}%)  剩余 {eta}")
        else:
            el = int(time.time() - t0)
            print(f"[{time.strftime('%H:%M:%S')}] 等待首个进度行…（{el}s）")

        if (time.time() - t0) / 60 > args.timeout_min:
            print(f"\n⚠️ 超过 {args.timeout_min} 分钟仍在跑 —— 交给你去判断")
            return 2
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
