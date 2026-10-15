"""数据完整性预检 —— 改 `data/` 或任务集后**必须**跑一遍。

## 为什么需要它（专家建议）

我写了 159 个测试和一堆检查器，但它们**只在被调用时才跑**。
风险场景很具体：

> 我改了任务集或训练语料 → 忘了跑检查器 → 提交 → 后面的实验
> 建立在**未被检查**的数据上，而当时看起来一切正常。

这个项目里最严重的一类错误（数据污染、机场码漂移、实体泄漏）
**全都不报错**。所以「记得跑检查」不能靠人。

## 用法

```bash
# 手动
python scripts/prepare/verify_all.py

# 装成 git pre-commit 钩子（改 data/ 或任务集文件时自动跑）
python scripts/prepare/install_hooks.py
```

**退出码非 0 表示数据有问题，不要提交。**
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


#: 预检项：(名称, 要跑的 python 代码片段, 失败时的说明)
#:
#: ⚠️ 每项都必须是**可失败的** —— 会抛异常或返回非零才算过。
#: 「打印一堆数字然后退出 0」不算检查。
CHECKS: list[tuple[str, str, str]] = [
    (
        "任务集结构约束",
        "import sys; sys.path.insert(0,'.');"
        "from src.data.task_manifest_v2 import verify_task_set;"
        "from src.data.task_set_v2_specs import build_specs;"
        "r=verify_task_set(build_specs());"
        "assert not r['violations'], r['violations'];"
        "print(f\"  {r['n']} 任务, 骨架比 {r['skeleton_ratio']:.3f}, 违规 0\")",
        "任务集违反结构约束（骨架重复/实体复用/过去日期/空参数等）",
    ),
    (
        "训练语料实体覆盖",
        "import sys; sys.path.insert(0,'.');"
        "from src.data.recovery_corpus import assert_training_covers_eval_entities as A;"
        "rep=A(); print(f\"  评测需 {rep['eval_codes']} 码, 训练覆盖 {rep['syn_codes']}, 缺失 {len(rep['missing_codes'])}\")",
        "训练语料没覆盖评测需要的机场码/工具 —— 那些任务**必然失败**",
    ),
    (
        "评测/训练实体泄漏",
        "import sys; sys.path.insert(0,'.');"
        "from src.data.recovery_corpus import find_expected_leaks as F;"
        "leaks=F(task_set='v3');"
        "assert not leaks, f'实体泄漏: {leaks}';"
        "print('  实体泄漏 0 个')",
        "有评测任务的期望调用与训练样本完全相同 —— 可能靠记忆答对",
    ),
    (
        "任务可解性（期望值可推导）",
        "import sys; sys.path.insert(0,'.');"
        "from src.data.task_manifest import load_tasks_by_version;"
        "from scripts.prepare.audit_task_solvability import audit_task;"
        "rows=[audit_task(t) for t in load_tasks_by_version('v3')];"
        "bad=[r for r in rows if not r['ok']];"
        "assert not bad, [(r['task_id'], r['hard_fails']) for r in bad[:3]];"
        "print(f'  {len(rows)} 任务全部可解')",
        "有任务的期望值无法从用户话/观测/纠正轮推出 —— "
        "**永远答不对**，而报告会把它记成「模型参数错」",
    ),
    (
        "数据划分交叠",
        "import sys, json; sys.path.insert(0,'.');"
        "bad=[];"
        "\nfor n in ('sft_only','dpo_joint'):\n"
        "    tr=[json.loads(l) for l in open(f'data/preferences/{n}.train.jsonl',encoding='utf-8') if l.strip()];\n"
        "    dv=[json.loads(l) for l in open(f'data/preferences/{n}.dev.jsonl',encoding='utf-8') if l.strip()];\n"
        "    ov={r['prompt'] for r in tr} & {r['prompt'] for r in dv};\n"
        "    if ov: bad.append((n,len(ov)))\n"
        "assert not bad, f'train/dev prompt 交叠: {bad}';"
        "print('  train/dev 交叠 0')",
        "训练与验证集有逐字相同的 prompt",
    ),
    (
        "可见性标签有效性",
        "import sys; sys.path.insert(0,'.');"
        "from src.data.task_manifest_v2 import report_visibility_warning as W;"
        "w=W();"
        # ⚠️ 这一项是**诊断式**检查：标签有效才算「通过」，
        # 无效时以非零退出（但 WARN_ONLY 会让它只警告不阻断）。
        "assert not w, w.splitlines()[0] if w else '';"
        "print('  seen/unseen 标签有效')",
        "（非致命）seen/unseen 标签没有含义，报告里不得用于泛化结论",
    ),
]

#: 这些项失败只警告、不阻断（例如可见性标签问题是「要声明」而非「要修」）
WARN_ONLY = {"可见性标签有效性"}


def run_check(name: str, code: str, why: str) -> tuple[bool, str]:
    """跑一项检查。返回 ``(是否通过, 输出)``。

    ⚠️ 必须显式 ``encoding="utf-8"``：Windows 上 ``subprocess`` 默认按
    **GBK** 解码子进程输出，而我们的检查器会打印中文 ——
    实测直接抛 ``UnicodeDecodeError``（在 `_readerthread` 里，
    所以主进程看到的是「线程异常」而不是有用的报错）。
    """
    env = dict(**__import__("os").environ)
    env["PYTHONIOENCODING"] = "utf-8"
    p = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(_REPO), capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=env,
    )
    out = (p.stdout or "") + (p.stderr or "")
    return p.returncode == 0, out.strip()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="数据完整性预检")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    print("=" * 74)
    print("数据完整性预检")
    print("=" * 74)

    failed: list[tuple[str, str]] = []
    warned: list[tuple[str, str]] = []

    for name, code, why in CHECKS:
        ok, out = run_check(name, code, why)
        mark = "✅" if ok else ("⚠️" if name in WARN_ONLY else "❌")
        print(f"\n{mark} {name}")
        if not args.quiet and out:
            for ln in out.splitlines()[:6]:
                print(f"   {ln}")
        if not ok:
            if name in WARN_ONLY:
                warned.append((name, why))
            else:
                failed.append((name, why))
                print(f"   ⚠️ {why}")

    print()
    print("=" * 74)
    if failed:
        print(f"预检失败：{len(failed)} 项阻断性问题")
        for name, why in failed:
            print(f"  ❌ {name} —— {why}")
        print("\n**不要提交。** 这些错误不会在训练/评测时暴露，")
        print("只会让结论静默地错。")
        return 1
    if warned:
        print(f"预检通过，但有 {len(warned)} 项警告（需在报告里声明）：")
        for name, why in warned:
            print(f"  ⚠️ {name} —— {why}")
        return 0
    print("预检通过 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
