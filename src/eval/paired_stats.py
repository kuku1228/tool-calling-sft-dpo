"""配对统计检验（任务级）。

## 为什么需要它（我之前的判据是错的）

我长期用「两个实验的 95% CI 是否重叠」判断差异是否显著。**那是错的**：

* CI 重叠 ≠ 无差异 —— 两个区间各自包含各自的不确定性，重叠只说明
  「存在共同可能值」，**不构成「差异不显著」的证据**
* 两个实验跑在**同一批任务**上，这是**配对数据**，必须用配对检验

正确做法是在同一批任务上做配对比较。

## 为什么重采样单位是「任务」而不是「样本」

每个任务有 ``num_samples`` 次采样，同一任务的多次采样**高度相关**。
按样本重采样会把它们当成独立观测，**严重低估区间宽度**。
有效独立样本量是**任务数**，不是样本数。

## 实测校准（本项目）

σ≈0.5（任务级二值结果在 p≈0.5 附近）时：

    100 任务 → MDE ≈ 0.14    （即「差 3 个百分点」这类结论不可能成立）
    400 任务 → MDE ≈ 0.07

这也是为什么「扩到 400 个任务」曾经看起来有吸引力 —— 但 400 个任务里
只有 208 种骨架，**有效独立样本不是 400**，那个更窄的 CI 是虚假精度。
"""

from __future__ import annotations

import collections
import json
from pathlib import Path


import math
import random

from src.data.task_manifest import load_tasks_by_version


def paired_bootstrap(
    a: dict[str, float],
    b: dict[str, float],
    *,
    n_boot: int = 2000,
    seed: int = 7,
) -> tuple[float, float, float]:
    """对**任务**重采样，返回 ``(观测差值, CI 下界, CI 上界)``。

    差值定义 ``mean(a) - mean(b)``，即「A 优于 B 多少」。

    >>> a = {"t1": 1.0, "t2": 1.0, "t3": 0.0, "t4": 1.0}
    >>> b = {"t1": 0.0, "t2": 0.0, "t3": 0.0, "t4": 1.0}
    >>> obs, lo, hi = paired_bootstrap(a, b, n_boot=200, seed=1)
    >>> round(obs, 3)
    0.5
    >>> lo <= obs <= hi
    True
    """
    common = sorted(set(a) & set(b))
    if not common:
        return 0.0, 0.0, 0.0
    diffs = [a[k] - b[k] for k in common]
    n = len(diffs)
    obs = sum(diffs) / n
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        boots.append(s / n)
    boots.sort()
    return obs, boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]


def mcnemar(
    a: dict[str, float], b: dict[str, float], *, thr: float = 0.5
) -> dict[str, float]:
    """任务级 McNemar：按 ``thr`` 把任务二值化为「会 / 不会」。

    返回 ``b``（A 对 B 错）、``c``（B 对 A 错）、``n_discordant`` 与
    **精确二项检验**的双侧 p 值。

    为什么用精确检验而不是卡方：不一致对数常常很少（``b+c`` 可能 < 10），
    卡方近似在这种情形下不可靠。

    >>> a = {"t1": 1.0, "t2": 1.0, "t3": 1.0}
    >>> b = {"t1": 1.0, "t2": 0.0, "t3": 0.0}
    >>> r = mcnemar(a, b)
    >>> r["b"], r["c"]
    (2, 0)
    >>> round(r["p_value"], 3)
    0.5
    """
    common = sorted(set(a) & set(b))
    b_cnt = sum(1 for k in common if a[k] >= thr and b[k] < thr)
    c_cnt = sum(1 for k in common if b[k] >= thr and a[k] < thr)
    n = b_cnt + c_cnt
    if n == 0:
        return {"b": 0, "c": 0, "n_discordant": 0, "p_value": 1.0}
    # 双侧精确二项检验：H0 下 b ~ Binomial(n, 0.5)
    k = min(b_cnt, c_cnt)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n) * 2
    return {"b": b_cnt, "c": c_cnt, "n_discordant": n, "p_value": min(1.0, p)}


def mde(
    n_tasks: int, *, sigma: float, alpha: float = 0.05, power: float = 0.80
) -> float:
    """最小可检测效应（配对设计，任务级）。

    ``(z_{1-alpha/2} + z_{power}) * sigma / sqrt(n)``

    >>> mde(100, sigma=0.5) > mde(400, sigma=0.5)
    True
    >>> mde(0, sigma=0.5) > 0
    True
    """
    z_a = 1.959963985  # alpha=0.05 双侧
    z_p = 0.841621234  # power=0.80
    return (z_a + z_p) * sigma / math.sqrt(max(1, n_tasks))


def sigma_of(a: dict[str, float], b: dict[str, float]) -> float:
    """配对差值的样本标准差（用于 MDE）。"""
    common = sorted(set(a) & set(b))
    if len(common) < 2:
        return 0.0
    d = [a[k] - b[k] for k in common]
    m = sum(d) / len(d)
    var = sum((x - m) ** 2 for x in d) / (len(d) - 1)
    return math.sqrt(var)


def task_scores(d: dict[str, list[bool]]) -> dict[str, float]:
    """``{task_id: [是否正确,...]}`` → ``{task_id: 该任务答对比例}``。

    >>> task_scores({"t": [True, True, False, False]})
    {'t': 0.5}
    """
    return {k: (sum(v) / len(v) if v else 0.0) for k, v in d.items()}


# ---------------------------------------------------------------------------
# 簇（cluster）：同一「期望调用」的多个任务
# ---------------------------------------------------------------------------
def expected_keys(version: str = "v3") -> dict[str, str]:
    """``{task_id: 期望调用指纹}``。

    ## 为什么需要它

    评测集里有些任务的**正确答案完全相同**。最明显的一例：
    所有「应先追问」任务的期望调用都是空的（正确行为 = 不调用任何工具），
    所以它们的期望调用指纹**全是同一个**。

    实测 v3：106 个任务只有 **92 个**不同期望调用 ——
    其中一个是 **15 个 clarify 任务**共享的簇。

    **它们不是 15 个独立证据。** 把它们当独立样本会让
    不确定区间偏窄、且把「一个行为」的权重放大 15 倍。

    >>> k = expected_keys("v3")
    >>> len(k) > 90
    True
    >>> len(set(k.values())) < len(k)      # 存在重复的期望调用
    True
    """
    from src.data.task_manifest import load_tasks_by_version

    tasks = load_tasks_by_version(version)
    out: dict[str, str] = {}
    for t in tasks:
        a = t.expected.args or {}
        out[t.task_id] = "|".join([
            str(t.expected.tool),
            *[f"{k}={v}" for k, v in sorted((k, str(v)) for k, v in a.items())],
        ])
    return out


def cluster_scores(
    scores: dict[str, float], cluster_of: dict[str, str]
) -> dict[str, float]:
    """把任务分数按簇求平均 → ``{cluster_id: 簇平均分}``。

    簇内取平均，而不是取最大或求和 —— 我们要的是
    「这个**行为**做对了多少」，而不是「这一族任务总共贡献多少分」。

    >>> s = {"t1": 1.0, "t2": 0.0, "t3": 1.0}
    >>> c = {"t1": "A", "t2": "A", "t3": "B"}
    >>> cluster_scores(s, c)
    {'A': 0.5, 'B': 1.0}
    """
    acc: dict[str, list[float]] = {}
    for tid, v in scores.items():
        cid = cluster_of.get(tid)
        if cid is None:
            continue
        acc.setdefault(cid, []).append(v)
    return {k: sum(v) / len(v) for k, v in acc.items()}


def paired_bootstrap_clustered(
    a: dict[str, float],
    b: dict[str, float],
    cluster_of: dict[str, str],
    *,
    n_boot: int = 2000,
    seed: int = 7,
) -> tuple[float, float, float, int]:
    """**按簇**重采样的配对 bootstrap。

    返回 ``(观测差值, CI 下界, CI 上界, 有效簇数)``。

    与 :func:`paired_bootstrap` 的区别：后者按**任务**重采样，
    会把同一个行为重复计数。本函数先按簇取平均，再对**簇**重采样，
    于是有效样本量是**簇数**而不是任务数。

    >>> a = {"t1": 1.0, "t2": 1.0, "t3": 0.0}
    >>> b = {"t1": 0.0, "t2": 0.0, "t3": 0.0}
    >>> c = {"t1": "X", "t2": "X", "t3": "Y"}
    >>> obs, lo, hi, n = paired_bootstrap_clustered(a, b, c, n_boot=200, seed=1)
    >>> n
    2
    >>> round(obs, 3)        # 簇 X=1.0, 簇 Y=0.0 → 均值 0.5
    0.5
    >>> lo <= obs <= hi
    True
    """
    ca = cluster_scores(a, cluster_of)
    cb = cluster_scores(b, cluster_of)
    common = sorted(set(ca) & set(cb))
    if not common:
        return 0.0, 0.0, 0.0, 0
    diffs = [ca[k] - cb[k] for k in common]
    n = len(diffs)
    obs = sum(diffs) / n
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        boots.append(s / n)
    boots.sort()
    return obs, boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)], n


def sigma_of_clustered(
    a: dict[str, float], b: dict[str, float], cluster_of: dict[str, str]
) -> float:
    """按簇的配对差值标准差（用于簇级 MDE）。

    >>> a = {"t1": 1.0, "t2": 1.0}; b = {"t1": 0.0, "t2": 0.0}
    >>> c = {"t1": "X", "t2": "X"}
    >>> sigma_of_clustered(a, b, c)
    0.0
    """
    ca = cluster_scores(a, cluster_of)
    cb = cluster_scores(b, cluster_of)
    common = sorted(set(ca) & set(cb))
    if len(common) < 2:
        return 0.0
    d = [ca[k] - cb[k] for k in common]
    m = sum(d) / len(d)
    return math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1))


def mcnemar_clustered(
    a: dict[str, float],
    b: dict[str, float],
    cluster_of: dict[str, str],
    *,
    thr: float = 0.5,
) -> dict[str, float]:
    """**按簇**的 McNemar —— 先把簇内平均，再二值化。

    不这么做的话，15 个 clarify 任务会在不一致对表里占 15 行，
    而它们其实只反映**一个**行为。
    """
    ca = cluster_scores(a, cluster_of)
    cb = cluster_scores(b, cluster_of)
    return mcnemar(ca, cb, thr=thr)


def cluster_ci(
    scores: dict[str, float],
    cluster_of: dict[str, str],
    *,
    n_boot: int = 2000,
    seed: int = 7,
) -> dict:
    """**单组**的簇级置信区间。

    与 :func:`paired_bootstrap_clustered` 不同，这里只有一组
    （不是两组比较）—— 用于给「某个模型的准确率」加上更保守的区间。

    ⚠️ 为什么需要它：``EvalReport`` 的主指标用**任务级** bootstrap。
    而当多个任务的正确答案完全相同时（实测：15 个 clarify 任务的
    期望调用都是空的），按任务重采样会把**一个行为**重复计数，
    区间因此偏窄。簇级口径先把同簇任务取平均、再按簇重采样。

    >>> s = {"t1": 1.0, "t2": 1.0, "t3": 0.0, "t4": 0.0}
    >>> c = {"t1": "A", "t2": "A", "t3": "B", "t4": "C"}
    >>> r = cluster_ci(s, c, n_boot=200, seed=1)
    >>> r["n_clusters"]
    3
    >>> round(r["mean"], 3)      # (1.0 + 0.0 + 0.0) / 3
    0.333
    >>> r["ci_lo"] <= r["mean"] <= r["ci_hi"]
    True
    """
    cs = cluster_scores(scores, cluster_of)
    keys = sorted(cs)
    if not keys:
        return {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "n_clusters": 0,
                "n_tasks": len(scores)}
    vals = [cs[k] for k in keys]
    n = len(vals)
    mean = sum(vals) / n
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        acc = 0.0
        for _ in range(n):
            acc += vals[rng.randrange(n)]
        boots.append(acc / n)
    boots.sort()
    return {
        "mean": mean,
        "ci_lo": boots[int(0.025 * n_boot)],
        "ci_hi": boots[int(0.975 * n_boot)],
        "n_clusters": n,
        "n_tasks": len(scores),
        "cluster_shrinkage": len(scores) - n,
    }


def compare(
    a_label: str,
    b_label: str,
    A: dict[str, list[bool]],
    B: dict[str, list[bool]],
    *,
    thr: float = 0.5,
    n_boot: int = 2000,
    cluster_of: dict[str, str] | None = None,
) -> dict:
    """完整配对对比。

    同时给出**任务级**与**簇级**两套结果。簇级是更保守的口径：
    它把「同一期望调用的多个任务」压成一个观测。

    ``cluster_of`` 为 ``None`` 时只出任务级结果（向后兼容）。
    """
    a, b = task_scores(A), task_scores(B)
    common = sorted(set(a) & set(b))
    n = len(common)
    obs, lo, hi = paired_bootstrap(a, b, n_boot=n_boot)
    sig = sigma_of(a, b)
    out: dict = {
        "a": a_label,
        "b": b_label,
        "n_common": n,
        "n_only_a": len(set(a) - set(b)),
        "n_only_b": len(set(b) - set(a)),
        "mean_a": sum(a[k] for k in common) / n if n else 0.0,
        "mean_b": sum(b[k] for k in common) / n if n else 0.0,
        "diff": obs,
        "ci_lo": lo,
        "ci_hi": hi,
        "ci_excludes_zero": (lo > 0) or (hi < 0),
        "mcnemar": mcnemar(a, b, thr=thr),
        "sigma": sig,
        "mde": mde(n, sigma=sig) if n else 0.0,
    }

    if cluster_of:
        cobs, clo, chi, cn = paired_bootstrap_clustered(
            a, b, cluster_of, n_boot=n_boot
        )
        csig = sigma_of_clustered(a, b, cluster_of)
        out["clustered"] = {
            "n_clusters": cn,
            "diff": cobs,
            "ci_lo": clo,
            "ci_hi": chi,
            "ci_excludes_zero": (clo > 0) or (chi < 0),
            "mcnemar": mcnemar_clustered(a, b, cluster_of, thr=thr),
            "sigma": csig,
            "mde": mde(cn, sigma=csig) if cn else 0.0,
        }
    return out


def load_graded(
    label: str, exp_root: Path, *, grade_fn=None
) -> dict[str, list[bool]]:
    """读某个实验的 ``predictions.jsonl`` 并**重新评分**。

    返回 ``{task_id: [该任务每次采样是否通过]}``。

    ⚠️ **必须重新评分**：``predictions.jsonl`` 的字段只有
    ``task_id / text / sample_index / finish_reason / latency_s /
    num_tokens / policy / meta`` —— **没有 ``passed``**。
    早期我误以为有，脚本会读到一个恒为 ``None`` 的字段、把所有样本
    当成失败，于是**任何两组对比都得出「无差异」**（不报错）。
    """
    exp = exp_root / label
    path = None
    for cand in ("predictions.jsonl", "predictions.mock.jsonl"):
        if (exp / cand).exists():
            path = exp / cand
            break
    if path is None:
        contents = sorted(p.name for p in exp.iterdir()) if exp.exists() else "目录不存在"
        raise FileNotFoundError(f"{exp} 下找不到 predictions.jsonl（内容: {contents}）")

    by_id = {t.task_id: t for t in load_tasks_by_version("v1")}
    by_id.update({t.task_id: t for t in load_tasks_by_version("v3")})

    out: dict[str, list[bool]] = collections.defaultdict(list)
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        task = by_id.get(r.get("task_id", ""))
        if task is None:
            continue
        if grade_fn is None:
            from src.eval.grader import grade as grade_fn  # type: ignore[assignment]
            _g = grade_fn(task, r.get("text", ""))
        else:
            _g = grade_fn(task, r.get("text", ""))
        out[task.task_id].append(bool(_g.passed))
    return dict(out)
