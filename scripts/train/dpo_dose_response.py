"""C1：clarify 偏好对的**受控剂量-反应**实验。

## 这个实验要回答什么

「DPO 把 clarify 偏好对过度泛化，导致任务能力崩塌」—— 这是我从
**一次失败**里读出的机制假设。一次失败不能确立机制，**剂量-反应能**。

如果损害真的由 clarify 偏好对造成，那么在其他条件完全相同、
**只改 clarify 对占比**时：

* ``clarify_instead`` 错误率应随占比**单调上升**
* 任务准确率应随占比**单调下降**

三条曲线都单调 → 机制确立。否则我的假设就是错的（也如实报告）。

## 为什么必须在「同一份数据」上做切分

早期我打算分别生成三份数据，但那样会引入**混淆变量**：
不同生成的 prompt 分布、实体、长度都可能不同，
剂量效应就和数据差异混在一起了。

所以本脚本**只切分同一份 ``dpo_joint`` 偏好数据**，
三个变体的非 clarify 部分来自**同一个池**，
只有 clarify 的**数量**不同。

## 严格共用的量（剂量实验的全部意义所在）

| 项 | 处理 |
|---|---|
| 起点 LoRA | 同一次 SFT 的 ``final``（``--init-adapter``） |
| β | 相同 |
| lr | 相同 |
| epochs | 相同 |
| 有效 batch | 相同 |
| 随机种子 | 数据采样用同一 seed |
| **总数** | **固定 1000 对** —— 否则「对多了」本身会改变训练步数 |
| **只有** clarify 占比变 | 0% / 30% / 100% |

**固定总数是关键**：若 100% 变体只有 300 对而 0% 变体有 2000 对，
那两者的**梯度步数不同**，观测到的差异可能是步数造成的，不是剂量造成的。

## 数据量的现实约束（必须记录）

``dpo_joint`` 里 clarify 只有 **304 对**。所以：

* 100% 变体需要从 304 对里**有放回**重采样到 1000 对
  （脚本会打印提示，报告里必须写明）
* 若 SFT/DPO 重训后 clarify 变多，可提高 ``--total``

## 用法

```bash
# 1) 只造数据并体检（不需要 GPU）
python scripts/train/dpo_dose_response.py --prepare-only

# 2) 三个变体依次训练（需要 GPU，约 45 分钟）
nohup python -u scripts/train/dpo_dose_response.py \
    --sft-adapter outputs/sft/final \
    --beta 0.5 --lr 5e-5 --epochs 1 --total 1000 \
    > dose_response.log 2>&1 &

# 3) 训练完自动评测三个变体
python scripts/train/dpo_dose_response.py --eval-only
```
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

#: clarify 的 pair_type 名（见 src/data/preference_builder.py）
CLARIFY = "clarify_boundary"

#: 三档剂量（clarify 对的占比）
DOSES = (0.0, 0.3, 1.0)


def load_pairs(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.open(encoding="utf-8") if ln.strip()]


def split_by_type(pairs: list[dict]) -> tuple[list[dict], list[dict]]:
    """切成 (clarify 对, 非 clarify 对)。

    >>> ps = [{"pair_type": "clarify_boundary"}, {"pair_type": "format_constraint"}]
    >>> c, n = split_by_type(ps)
    >>> len(c), len(n)
    (1, 1)
    """
    clarify = [p for p in pairs if p.get("pair_type") == CLARIFY]
    other = [p for p in pairs if p.get("pair_type") != CLARIFY]
    return clarify, other


def build_variant(
    clarify: list[dict],
    other: list[dict],
    *,
    dose: float,
    total: int,
    seed: int = 42,
) -> tuple[list[dict], dict]:
    """按剂量组装一个变体：**总数固定**为 ``total``，clarify 占 ``dose``。

    返回 ``(偏好对列表, 统计字典)``。

    * 需要放回重采样时会在统计里标记 ``clarify_resampled=True`` ——
      报告必须写明这一点（有放回会让同一条数据被多次见到）。
    * 总数固定是为了让三个变体的**梯度步数相同**；否则观测到的差异
      可能来自步数而不是剂量。

    >>> c = [{"pair_type": "clarify_boundary"}] * 5
    >>> o = [{"pair_type": "format_constraint"}] * 20
    >>> v, st = build_variant(c, o, dose=0.5, total=10, seed=1)
    >>> len(v), st["n_clarify"], st["n_other"]
    (10, 5, 5)
    """
    rng = random.Random(seed)
    n_clarify = int(round(total * dose))
    n_other = total - n_clarify
    resampled = False

    def _take(pool: list[dict], n: int) -> list[dict]:
        nonlocal resampled
        if n == 0:
            return []
        if len(pool) >= n:
            return rng.sample(pool, n)
        # 池子不够：有放回重采样（会重复见到同一条数据）
        if pool:
            resampled = True
            return [rng.choice(pool) for _ in range(n)]
        return []

    v = _take(clarify, n_clarify) + _take(other, n_other)
    rng.shuffle(v)
    stats = {
        "dose": dose,
        "n_total": len(v),
        "n_clarify": sum(1 for p in v if p.get("pair_type") == CLARIFY),
        "n_other": sum(1 for p in v if p.get("pair_type") != CLARIFY),
        "clarify_pool": len(clarify),
        "other_pool": len(other),
        "clarify_resampled": resampled,
    }
    return v, stats


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def dose_label(dose: float) -> str:
    return f"dose{int(round(dose * 100)):03d}"


def prepare(args) -> list[dict]:
    src = Path(args.data_file)
    if not src.exists():
        print(f"  ✗ 找不到偏好数据 {src}")
        print("     先跑：python scripts/prepare/build_preferences.py")
        return []
    pairs = load_pairs(src)
    clarify, other = split_by_type(pairs)
    print(f"  源数据 {src}: {len(pairs)} 对")
    print(f"    clarify   {len(clarify):>5} 对 ({len(clarify)/len(pairs):.1%})")
    print(f"    非 clarify {len(other):>5} 对")
    print()

    prepared = []
    for dose in DOSES:
        v, st = build_variant(clarify, other, dose=dose, total=args.total, seed=args.seed)
        out = Path("data/preferences") / f"{dose_label(dose)}.train.jsonl"
        write_jsonl(v, out)
        st["path"] = str(out)
        st["label"] = dose_label(dose)
        prepared.append(st)
        flag = "  ⚠️ clarify 有放回重采样" if st["clarify_resampled"] else ""
        print(f"  {out}")
        print(f"     总数 {st['n_total']}  clarify {st['n_clarify']} "
              f"({st['n_clarify']/st['n_total']:.1%})  非 clarify {st['n_other']}{flag}")

    # 一致性自检：三个变体总数必须相同（否则剂量效应与步数效应混淆）
    totals = {s["n_total"] for s in prepared}
    if len(totals) != 1:
        print(f"\n  ✗⚠️ 三个变体的总数不一致: {totals} —— "
              "那会让「梯度步数」成为混淆变量，剂量实验失效")
        return []
    print(f"\n  ✅ 三个变体总数一致（{totals.pop()} 对）—— 梯度步数相同，"
          "唯一变量是 clarify 占比")
    return prepared


def train_one(args, st: dict) -> bool:
    """训练一个变体。返回是否成功。"""
    out_dir = Path(args.out_root) / st["label"]
    if (out_dir / "final").exists() and not args.force:
        print(f"  ⏭  {out_dir}/final 已存在，跳过（--force 可强制重跑）")
        return True
    cmd = [
        sys.executable, "-u", "scripts/train/dpo_train.py",
        "--experiment", args.experiment,
        "--data-file", st["path"],
        "--output-dir", str(out_dir),
        "--beta", str(args.beta),
        "--lr", str(args.lr),
        "--epochs", str(args.epochs),
        "--seed", str(args.seed),
    ]
    if args.sft_adapter:
        cmd += ["--init-adapter", args.sft_adapter]
    if args.mock:
        cmd.append("--mock")
    if args.max_steps > 0:
        cmd += ["--max-steps", str(args.max_steps)]
    print(f"\n  ▶ {st['label']}（clarify {st['n_clarify']}/{st['n_total']}）")
    print(f"    {' '.join(cmd[:8])} …")
    rc = subprocess.run(cmd, cwd=str(_REPO)).returncode
    ok = rc == 0 and (out_dir / "final").exists()
    if not ok:
        print(f"  ✗ {st['label']} 失败（rc={rc}，final 是否存在="
              f"{(out_dir / 'final').exists()}）")
    return ok


def eval_one(args, st: dict) -> bool:
    out_dir = Path(args.out_root) / st["label"]
    if not (out_dir / "final").exists():
        print(f"  ✗ {st['label']} 无 final，跳过评测")
        return False
    cmd = [
        sys.executable, "-u", "scripts/eval/run_baseline.py",
        "--label", st["label"],
        "--task-set", args.task_set,
        "--num-samples", str(args.num_samples),
        "--adapter", str(out_dir / "final"),
    ]
    if args.mock:
        # ⚠️ mock 训练必须配 mock 评测：否则评测会去加载真实模型，
        # 在没有 GPU / transformers 的机器上直接 ModuleNotFoundError。
        # 而且「训练用 mock、评测用真模型」这个组合本身就不一致，
        # 得出的数字不能和 mock 训练产物配对。
        cmd += ["--mock", "--mock-policy", args.experiment]
    print(f"  ▶ 评测 {st['label']} …")
    return subprocess.run(cmd, cwd=str(_REPO)).returncode == 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="C1 clarify 剂量-反应实验")
    ap.add_argument("--data-file", default="data/preferences/dpo_joint.train.jsonl")
    ap.add_argument("--experiment", default="dpo_joint",
                    help="用哪个实验的偏好对组合（默认 dpo_joint）")
    ap.add_argument("--out-root", default="outputs/dose")
    ap.add_argument("--total", type=int, default=1000,
                    help="每个变体的偏好对总数（三个变体相同，固定梯度步数）")
    ap.add_argument("--sft-adapter", default=None,
                    help="SFT 的 final 目录（作为三个变体的共同起点）")
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--task-set", default="v3")
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--mock", action="store_true", help="mock 训练（不需要 GPU）")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="已有 final 也重跑")
    args = ap.parse_args(argv)

    print("=" * 74)
    print("C1：clarify 偏好对剂量-反应")
    print("=" * 74)
    if not args.sft_adapter and not args.prepare_only:
        print("  ⚠️ 未给 --sft-adapter：三个变体将从**基座模型**开始 —— "
              "那测的是「从零做 DPO」，与「在 SFT 之上做 DPO」不是同一个问题。")
        print("     建议：--sft-adapter outputs/sft/final")

    prepared = prepare(args)
    if not prepared:
        return 1
    if args.prepare_only:
        return 0

    if args.eval_only:
        results = [eval_one(args, st) for st in prepared]
        return 0 if all(results) else 1

    ok = True
    for st in prepared:
        if not train_one(args, st):
            ok = False
            print("  ⚠️ 一个变体失败后仍继续跑其余的 —— "
                  "部分结果也比「因为一个失败而全军覆没」有用")

    print("\n" + "=" * 74)
    print("训练完成，开始评测三个变体")
    print("=" * 74)
    for st in prepared:
        eval_one(args, st)

    print("\n" + "=" * 74)
    print("下一步：跑配对检验看三条曲线是否单调")
    print("=" * 74)
    labels = ",".join(st["label"] for st in prepared)
    print(f"  python scripts/eval/paired_significance.py "
          f"--baseline {prepared[0]['label']} --against {labels}")
    print("\n  然后按 ``clarify_instead`` 错误数与 accuracy 分别画剂量曲线：")
    print("    · clarify_instead 随剂量**单调上升**？")
    print("    · accuracy 随剂量**单调下降**？")
    print("  两条都单调 → 机制确立；否则我的假设是错的（如实报告）。")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
