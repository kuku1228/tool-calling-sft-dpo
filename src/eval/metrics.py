"""指标计算：pass@k、pass^k、bootstrap 置信区间与统计量。

本项目报告哪些指标
------------------
====================  ==========================================================
指标                   定义与用途
====================  ==========================================================
``accuracy``          单次采样正确率。最直观，但不反映稳定性。
``pass@k``            一个任务采 k 次，至少一次对的概率（无偏估计）。
                      k=1 时等于 accuracy，所以它和 accuracy 不独立。
``pass^k``            一个任务采 k 次，**k 次全对**的概率。τ-bench 的主指标。
                      它同时惩罚「不会做」和「做得不稳」，是 DPO 最该改善的量。
``pass^k_by_*``       按难度 / 类别 / seen-unseen 切分的 pass^k。
``Δstability``        pass^k − pass@k。越接近 0 越稳定；负得越多说明「会做但飘」。
====================  ==========================================================

pass@k 与 pass^k 的无偏估计
---------------------------
n = 每任务采样次数，c = 其中正确的次数，k ≤ n。

pass@k（Chen et al. 2021, HumanEval）——「至少一次成功」::

    pass@k = 1 − C(n−c, k) / C(n, k)

pass^k（τ-bench, Yao et al. 2024）——「k 次全部成功」::

    pass^k = C(c, k) / C(n, k)

两者都用组合数而不是 ``c/n`` 的幂，因为在 n 较小时（本项目 n=4）
朴素估计 ``(c/n)^k`` 是**有偏**的：它把「从同一批样本里取 k 个」
当成了「独立重采 k 次」。

n = k 的退化情形
----------------
当 k = n = 4 时，``C(n,k)=C(4,4)=1``，两个估计都退化为「k 次采样的
观测结果」，不再有估计方差：
  - ``pass^n = C(c,n)/1 = 1`` 当且仅当 c = n，否则 0
  - 这正是 τ-bench 报告 pass^4 时的行为（n 通常就等于报告的 k）

为了在报告里既保留这个定义、又不至于只剩 0/1 两档，
``pass_hat_k`` 同时返回 ``naive_all``（``(c/n)^k``，有偏但连续），
在 n == k 时作为**参考值**放在报告里，并在文档中标注其为有偏估计。
主指标仍然用无偏公式。

置信区间
--------
任务级指标是不独立的多项分布比率，所以用 **bootstrap 重采样任务**（而非样本）
来估 CI —— 重采样单位必须和分析单位一致，否则 CI 会被高估精度。
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

# ============================================================================
# 基础统计
# ============================================================================


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def percentile(xs: Sequence[float], q: float) -> float:
    """线性插值分位数（与 numpy.percentile 一致）。

    >>> percentile([1, 2, 3, 4], 50)
    2.5
    """
    if not xs:
        return 0.0
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"q 必须在 [0,100]，收到 {q}")
    s = sorted(xs)
    if len(s) == 1:
        return float(s[0])
    pos = (len(s) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(s[int(pos)])
    return float(s[lo] + (s[hi] - s[lo]) * (pos - lo))


def stdev(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def pooled_stdev(groups: Iterable[Sequence[float]]) -> float:
    """合并组内标准差（智利项目里用来估 group advantage 的组内方差）。

    用途：诊断「group saturation」—— 当组内方差接近 0 时，
    GRPO 的 advantage 会消失。本项目虽然用 DPO，但在 mock 策略里
    同样报告这个量，用来对照说明「为什么 DPO 没有 group 结构所以不受此困扰」。
    """
    num = 0.0
    den = 0
    for g in groups:
        n = len(g)
        if n < 2:
            continue
        m = mean(g)
        num += sum((x - m) ** 2 for x in g)
        den += n - 1
    return math.sqrt(num / den) if den else 0.0


def log_comb(n: int, k: int) -> float:
    """log(C(n,k))，用 lgamma 避免大数溢出。"""
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def comb_ratio(numerator_n: int, numerator_k: int, denom_n: int, denom_k: int) -> float:
    """C(a,b)/C(c,d)，对数域计算。

    >>> comb_ratio(4, 2, 4, 2)
    1.0
    """
    if denom_k < 0 or denom_k > denom_n:
        return 0.0
    if numerator_k < 0 or numerator_k > numerator_n:
        return 0.0
    val = log_comb(numerator_n, numerator_k) - log_comb(denom_n, denom_k)
    if val < -700:
        return 0.0
    # 对数域计算会引入 ~1e-16 的浮点噪声（例如 0.24999999999999856）。
    # 指标最终只报到小数点后 4 位，这里统一修约到 12 位，
    # 既消除噪声又远高于任何实际需要的精度。
    return round(math.exp(val), 12)


# ============================================================================
# pass@k / pass^k
# ============================================================================


def pass_at_k_unbiased(n: int, c: int, k: int) -> float:
    """无偏 pass@k：n 次采样中 c 次正确，随机取 k 次至少一次正确的概率。

    >>> pass_at_k_unbiased(4, 1, 1)
    0.25
    >>> round(pass_at_k_unbiased(4, 2, 2), 6)
    0.833333
    >>> pass_at_k_unbiased(4, 4, 2)
    1.0
    """
    if k <= 0:
        raise ValueError("k 必须为正")
    if n <= 0:
        return 0.0
    if k > n:
        # 无法从 n 个样本估 k>n 的 pass@k，返回定义上的上界近似
        k = n
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - comb_ratio(n - c, k, n, k)


def pass_hat_k_unbiased(n: int, c: int, k: int) -> float:
    """无偏 pass^k：n 次采样中 c 次正确，随机取 k 次**全部**正确的概率。

    >>> pass_hat_k_unbiased(4, 1, 1)
    0.25
    >>> round(pass_hat_k_unbiased(4, 2, 2), 6)
    0.166667
    >>> pass_hat_k_unbiased(4, 4, 2)
    1.0
    >>> pass_hat_k_unbiased(4, 3, 2)
    0.5
    """
    if k <= 0:
        raise ValueError("k 必须为正")
    if n <= 0:
        return 0.0
    if k > n:
        k = n
    if c < k:
        return 0.0
    return comb_ratio(c, k, n, k)


def pass_hat_k_naive(n: int, c: int, k: int) -> float:
    """有偏的朴素 pass^k 估计 ``(c/n)^k``。

    只在 n == k（无偏估计退化为 0/1）时作为参考值报告。
    """
    if n <= 0:
        return 0.0
    return (c / n) ** k


# ============================================================================
# 任务级结果容器
# ============================================================================


@dataclass
class SampledTaskResult:
    """一个任务的多次采样结果。"""

    task_id: str
    #: 每次采样是否正确（True/False）
    outcomes: list[bool]
    category: str = ""
    difficulty: str = ""
    visibility: str = ""
    #: 每次采样的错误类别（与 outcomes 等长）
    categories: list[str] = field(default_factory=list)
    #: 每次采样的原始输出（可选，用于抽检；过大会被截断）
    texts: list[str] = field(default_factory=list)
    #: 每次采样的解析结果摘要
    predictions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.outcomes)

    @property
    def c(self) -> int:
        return sum(1 for o in self.outcomes if o)

    @property
    def accuracy(self) -> float:
        return self.c / self.n if self.n else 0.0

    @property
    def pass_k_values(self) -> dict[int, float]:
        return {k: pass_at_k_unbiased(self.n, self.c, k) for k in range(1, self.n + 1)}

    def pass_hat(self, k: int) -> float:
        return pass_hat_k_unbiased(self.n, self.c, k)

    def to_dict(self, include_text: bool = False) -> dict[str, Any]:
        d: dict[str, Any] = {
            "task_id": self.task_id,
            "category": self.category,
            "difficulty": self.difficulty,
            "visibility": self.visibility,
            "n": self.n,
            "c": self.c,
            "accuracy": round(self.accuracy, 4),
            "outcomes": [bool(o) for o in self.outcomes],
            "error_categories": self.categories,
            "predictions": self.predictions,
        }
        if include_text:
            d["texts"] = self.texts
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> SampledTaskResult:
        return SampledTaskResult(
            task_id=d["task_id"],
            outcomes=list(d["outcomes"]),
            category=d.get("category", ""),
            difficulty=d.get("difficulty", ""),
            visibility=d.get("visibility", ""),
            categories=list(d.get("error_categories", [])),
            texts=list(d.get("texts", [])),
            predictions=list(d.get("predictions", [])),
        )


@dataclass
class MetricValue:
    """一个带置信区间的指标。"""

    name: str
    value: float
    lo: float = 0.0
    hi: float = 0.0
    n_units: int = 0
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": round(self.value, 6),
            "ci95": [round(self.lo, 6), round(self.hi, 6)],
            "n_units": self.n_units,
            "note": self.note,
        }

    def pretty(self, as_percent: bool = True) -> str:
        if as_percent:
            return f"{self.name} = {self.value:.1%} [{self.lo:.1%}, {self.hi:.1%}]"
        return f"{self.name} = {self.value:.3f} [{self.lo:.3f}, {self.hi:.3f}]"


# ============================================================================
# 聚合
# ============================================================================


def _bootstrap_ci(
    unit_values: Sequence[float],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    statistic: Callable[[Sequence[float]], float] = mean,
) -> tuple[float, float]:
    """对**任务级**指标做 bootstrap 重采样置信区间。

    重采样单位是任务而不是采样样本 —— 同一任务的多次采样高度相关，
    按样本重采样会严重低估置信区间宽度。

    >>> lo, hi = _bootstrap_ci([1.0, 1.0, 0.0, 0.0], n_boot=200, seed=1)
    >>> lo <= 0.5 <= hi
    True
    """
    if not unit_values:
        return 0.0, 0.0
    point = statistic(unit_values)
    if len(unit_values) == 1:
        return point, point
    rng = random.Random(seed)
    m = len(unit_values)
    stats: list[float] = []
    for _ in range(n_boot):
        sample = [unit_values[rng.randrange(m)] for _ in range(m)]
        stats.append(statistic(sample))
    stats.sort()
    lo = stats[max(0, int((alpha / 2) * n_boot) - 1)]
    hi = stats[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return lo, hi


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 区间 —— 小样本比率（本项目每个难度组只有 10-20 个任务）
    比正态近似可靠得多。

    >>> lo, hi = wilson_interval(10, 20)
    >>> round(lo, 3), round(hi, 3)
    (0.299, 0.701)
    """
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def cohens_h(p1: float, p2: float) -> float:
    """比例差异的效应量（Cohen's h）。

    |h| 的经验解读：0.2 小、0.5 中、0.8 大。
    项目里用来判断「改进是否只是噪声」。

    >>> round(cohens_h(0.4, 0.6), 3)
    -0.403
    """
    p1 = min(max(p1, 1e-9), 1 - 1e-9)
    p2 = min(max(p2, 1e-9), 1 - 1e-9)
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def required_sample_size(effect: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """给定效应量下每组需要的任务数（两比例比较的粗略估计）。

    用途：**在跑实验之前**判断 100 个任务够不够检测出目标提升。
    这个问题在开题阶段就应当回答，否则实验跑完才发现
    「提升 3% 但置信区间跨 0」会浪费一轮 GPU 预算。

    >>> required_sample_size(0.10) > required_sample_size(0.20)
    True
    """
    effect = abs(effect)
    if effect <= 0:
        return 10**9
    z_a = 1.959964 if abs(alpha - 0.05) < 1e-6 else 1.644854
    z_b = 0.841621 if abs(power - 0.8) < 1e-6 else 1.281552
    # 两比例、方差取保守的 0.25（p=0.5）
    return math.ceil(2 * (z_a + z_b) ** 2 * 0.25 / (effect**2))


@dataclass
class EvalReport:
    """一次完整评测的聚合结果。"""

    label: str
    num_tasks: int
    num_samples_per_task: int
    results: list[SampledTaskResult]
    k_values: tuple[int, ...] = (1, 2, 4)
    primary_k: int = 4
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # 主指标
    # ------------------------------------------------------------------
    def accuracy(self, results: Sequence[SampledTaskResult] | None = None) -> MetricValue:
        rs = list(results if results is not None else self.results)
        if not rs:
            return MetricValue("accuracy", 0.0, 0.0, 0.0, 0)
        per_task = [r.accuracy for r in rs]
        lo, hi = _bootstrap_ci(per_task, seed=42)
        return MetricValue(
            "accuracy",
            mean(per_task),
            lo,
            hi,
            n_units=len(rs),
            note="单次采样平均正确率（任务级 bootstrap 95% CI）",
        )

    def pass_at(
        self, k: int, results: Sequence[SampledTaskResult] | None = None
    ) -> MetricValue:
        rs = list(results if results is not None else self.results)
        if not rs:
            return MetricValue(f"pass@{k}", 0.0, 0.0, 0.0, 0)
        per_task = [pass_at_k_unbiased(r.n, r.c, k) for r in rs]
        lo, hi = _bootstrap_ci(per_task, seed=42 + k)
        return MetricValue(
            f"pass@{k}",
            mean(per_task),
            lo,
            hi,
            n_units=len(rs),
            note=f"n={self.num_samples_per_task} 采样的无偏估计",
        )

    def pass_hat(
        self, k: int, results: Sequence[SampledTaskResult] | None = None
    ) -> MetricValue:
        rs = list(results if results is not None else self.results)
        if not rs:
            return MetricValue(f"pass^{k}", 0.0, 0.0, 0.0, 0)
        per_task = [pass_hat_k_unbiased(r.n, r.c, k) for r in rs]
        per_task_naive = [pass_hat_k_naive(r.n, r.c, k) for r in rs]
        lo, hi = _bootstrap_ci(per_task, seed=42 + 100 + k)
        note = f"k 次全对的无偏估计（n={self.num_samples_per_task}）"
        if len(rs) and rs[0].n == k:
            note += f"；n==k 时退化为 0/1，参考值 (c/n)^k = {mean(per_task_naive):.3f}（有偏）"
        return MetricValue(f"pass^{k}", mean(per_task), lo, hi, n_units=len(rs), note=note)

    def delta_stability(self, k: int | None = None) -> MetricValue:
        """pass^k − pass@k。衡量「会做但不稳定」的程度。

        越接近 0 越稳定。DPO 的正确目标是把这一项推向 0
        （即在保持 pass@k 的同时把 pass^k 拉上来）。
        """
        k = k or self.primary_k
        p_at = self.pass_at(k)
        p_hat = self.pass_hat(k)
        diff_per_task = [
            pass_hat_k_unbiased(r.n, r.c, k) - pass_at_k_unbiased(r.n, r.c, k) for r in self.results
        ]
        lo, hi = _bootstrap_ci(diff_per_task, seed=7)
        return MetricValue(
            f"Δstability(pass^{k}−pass@{k})",
            p_hat.value - p_at.value,
            lo,
            hi,
            n_units=len(self.results),
            note="越接近 0 越稳定",
        )

    # ------------------------------------------------------------------
    # 分组
    # ------------------------------------------------------------------
    def group_by(self, key: str) -> dict[str, list[SampledTaskResult]]:
        out: dict[str, list[SampledTaskResult]] = {}
        for r in self.results:
            out.setdefault(getattr(r, key, "") or "unknown", []).append(r)
        return out

    def by_difficulty(self) -> dict[str, dict[str, Any]]:
        return {
            k: self._slice_summary(v) for k, v in sorted(self.group_by("difficulty").items())
        }

    def by_category(self) -> dict[str, dict[str, Any]]:
        return {k: self._slice_summary(v) for k, v in sorted(self.group_by("category").items())}

    def by_visibility(self) -> dict[str, dict[str, Any]]:
        return {k: self._slice_summary(v) for k, v in sorted(self.group_by("visibility").items())}

    def _slice_summary(self, rs: list[SampledTaskResult]) -> dict[str, Any]:
        if not rs:
            return {}
        acc = self.accuracy(rs)
        return {
            "num_tasks": len(rs),
            "accuracy": round(acc.value, 4),
            "accuracy_ci95": [round(acc.lo, 4), round(acc.hi, 4)],
            **{
                f"pass@{k}": round(self.pass_at(k, rs).value, 4)
                for k in self.k_values
                if k <= (rs[0].n if rs else 0)
            },
            **{
                f"pass^{k}": round(self.pass_hat(k, rs).value, 4)
                for k in self.k_values
                if k <= (rs[0].n if rs else 0)
            },
        }

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------
    def to_dict(self, include_per_task_text: bool = False) -> dict[str, Any]:
        k = self.primary_k
        payload: dict[str, Any] = {
            "label": self.label,
            "num_tasks": self.num_tasks,
            "num_samples_per_task": self.num_samples_per_task,
            "primary_k": k,
            "metadata": self.metadata,
            "headline": {
                "accuracy": round(self.accuracy().value, 6),
                "pass_at_k": round(self.pass_at(k).value, 6),
                "pass_hat_k": round(self.pass_hat(k).value, 6),
                "delta_stability": round(self.delta_stability(k).value, 6),
            },
            "metrics": {
                **{f"accuracy": self.accuracy().to_dict()},
                **{f"pass@{kk}": self.pass_at(kk).to_dict() for kk in self.k_values},
                **{f"pass^{kk}": self.pass_hat(kk).to_dict() for kk in self.k_values},
                "delta_stability": self.delta_stability(k).to_dict(),
            },
            "by_difficulty": self.by_difficulty(),
            "by_category": self.by_category(),
            "by_visibility": self.by_visibility(),
            "per_task": [r.to_dict(include_text=include_per_task_text) for r in self.results],
        }
        return payload

    def write_json(self, path: str, include_per_task_text: bool = False) -> str:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(include_per_task_text), f, ensure_ascii=False, indent=1)
        return path

    @staticmethod
    def read_json(path: str) -> dict[str, Any]:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # 展示
    # ------------------------------------------------------------------
    def headline_lines(self) -> list[str]:
        k = self.primary_k
        acc = self.accuracy()
        p_at = self.pass_at(k)
        p_hat = self.pass_hat(k)
        ds = self.delta_stability(k)
        lines = [
            f"【{self.label}】 {self.num_tasks} 任务 × {self.num_samples_per_task} 采样",
            "-" * 66,
            f"  单次准确率 accuracy      {acc.value:>7.2%}  95%CI [{acc.lo:.2%}, {acc.hi:.2%}]",
            f"  pass@{k}（至少一次对）     {p_at.value:>7.2%}  95%CI [{p_at.lo:.2%}, {p_at.hi:.2%}]",
            f"  pass^{k}（{k} 次全对）       {p_hat.value:>7.2%}  95%CI [{p_hat.lo:.2%}, {p_hat.hi:.2%}]",
            f"  Δstability              {ds.value:>+7.2%}  （越接近 0 越稳定）",
        ]
        return lines

    def detail_lines(self) -> list[str]:
        lines = ["", "按难度:"]
        for k, v in self.by_difficulty().items():
            pk = f"pass^{self.primary_k}"
            lines.append(
                f"    {k:<8} n={v['num_tasks']:<4} acc={v['accuracy']:.1%}  "
                f"{pk}={v.get(pk, float('nan')):.1%}"
            )
        lines.append("按可见性（unseen = 训练未接触，泛化指标）:")
        for k, v in self.by_visibility().items():
            pk = f"pass^{self.primary_k}"
            lines.append(
                f"    {k:<8} n={v['num_tasks']:<4} acc={v['accuracy']:.1%}  "
                f"{pk}={v.get(pk, float('nan')):.1%}"
            )
        lines.append("按类别:")
        for k, v in self.by_category().items():
            pk = f"pass^{self.primary_k}"
            lines.append(
                f"    {k:<18} n={v['num_tasks']:<4} acc={v['accuracy']:.1%}  "
                f"{pk}={v.get(pk, float('nan')):.1%}"
            )
        return lines


# ============================================================================
# 多实验对比
# ============================================================================


@dataclass
class Comparison:
    """两个实验之间的对比结论。"""

    baseline: str
    treatment: str
    metric: str
    base_value: float
    treat_value: float

    @property
    def abs_delta(self) -> float:
        return self.treat_value - self.base_value

    @property
    def rel_delta(self) -> float:
        return self.abs_delta / self.base_value if self.base_value else float("nan")

    @property
    def effect_size(self) -> float:
        return cohens_h(self.base_value, self.treat_value)

    @property
    def effect_label(self) -> str:
        h = abs(self.effect_size)
        if h < 0.2:
            return "可忽略"
        if h < 0.5:
            return "小"
        if h < 0.8:
            return "中"
        return "大"

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline,
            "treatment": self.treatment,
            "metric": self.metric,
            "baseline_value": round(self.base_value, 6),
            "treatment_value": round(self.treat_value, 6),
            "abs_delta": round(self.abs_delta, 6),
            "rel_delta": round(self.rel_delta, 6) if self.rel_delta == self.rel_delta else None,
            "cohens_h": round(self.effect_size, 4),
            "effect_size": self.effect_label,
        }

    def pretty(self) -> str:
        sign = "+" if self.abs_delta >= 0 else ""
        rel = f"{self.rel_delta:+.1%}" if self.rel_delta == self.rel_delta else "n/a"
        return (
            f"{self.metric:<16} {self.base_value:.2%} → {self.treat_value:.2%}  "
            f"({sign}{self.abs_delta:.2%}, 相对 {rel}, 效应量 {self.effect_label} h={self.effect_size:+.3f})"
        )


def compare_reports(
    reports: dict[str, dict[str, Any]],
    *,
    baseline_key: str,
    metric_keys: tuple[str, ...] = ("accuracy", "pass_at_k", "pass_hat_k", "delta_stability"),
) -> list[Comparison]:
    """用已经写盘的 report dict 做对比。

    Args:
        reports: ``{实验名: report dict}``（``EvalReport.to_dict()`` 的输出）
        baseline_key: 作为基线的实验名。
    """
    out: list[Comparison] = []
    base = reports.get(baseline_key, {})
    base_head = base.get("headline", {})
    for name, rep in reports.items():
        if name == baseline_key:
            continue
        head = rep.get("headline", {})
        for mk in metric_keys:
            if mk in head and mk in base_head:
                out.append(
                    Comparison(
                        baseline=baseline_key,
                        treatment=name,
                        metric=mk,
                        base_value=float(base_head[mk]),
                        treat_value=float(head[mk]),
                    )
                )
    return out
