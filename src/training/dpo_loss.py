"""DPO 损失的纯 Python 实现（无 torch 依赖）。

为什么要单独写一份
------------------
1. **可验证**：``self_check()`` 用解析解验证实现是否与论文公式一致。
   这一条在 GPU 上跑不了（要先装 torch），而在 CPU 上随时能跑。
2. **mock 训练器要用**：它需要在没有 torch 的环境里做「真实」的
   梯度下降，从而产出**自洽**的 loss 曲线（改了公式曲线就会变），
   而不是一条硬编码的下降曲线。
3. **文档与代码一致**：文档里写的公式就是这里实现的公式。

数学
----
DPO 的出发点是 RLHF 目标::

    max_π  E[r(x,y)] − β · KL(π(y|x) ‖ π_ref(y|x))

可以证明最优策略有闭式解::

    π*(y|x) ∝ π_ref(y|x) · exp(r(x,y)/β)

反解出隐式 reward::

    r(x,y) = β · log(π_θ(y|x) / π_ref(y|x)) + β · log Z(x)

代入 Bradley-Terry 偏好模型 P(y_w ≻ y_l) = σ(r_w − r_l)，
配分项 log Z(x) 在差值中抵消，得到 DPO 损失::

    L = − E[ log σ( β · [ (log π_θ(y_w) − log π_ref(y_w))
                        − (log π_θ(y_l) − log π_ref(y_l)) ] ) ]

记::

    z = β · [ (πc − πr) − (refc − refr) ]      ← 隐式 reward 的 margin × β
    L = −log σ(z) = softplus(−z) = log(1 + e^{−z})

梯度（本模块用于 mock 训练的解析梯度）::

    ∂L/∂ logπ_θ(y_w) = −β · σ(−z)
    ∂L/∂ logπ_θ(y_l) = +β · σ(−z)

三个关键性质（也是 ``self_check`` 验证的内容）
----------------------------------------------
1. **对称无偏**：当 π_θ = π_ref 时，z = 0，L = log 2 ≈ 0.6931。
   这是 DPO loss 的起点 —— 任何偏离 log 2 的初始 loss 都说明
   实现有问题（或者数据里 chosen/rejected 被标反了）。
2. **单调性**：margin z 越大，loss 越小。
3. **饱和**：z → +∞ 时 L → 0；z → −∞ 时 L ≈ −z（线性）。
   饱和意味着「一旦学得很对，梯度就没了」—— 这正是 DPO
   后期改进空间变小的原因，也是要报告 margin 分布的原因。

β 的作用
--------
- β 大 → 对参考模型偏离的惩罚大 → 保守，z 被压小，loss 更大
- β 小 → 允许大幅偏离 → 但梯度也被 β 线性缩小（见梯度公式）

注意梯度里那个 β：β 小的时候**损失值**可能更小（z 更大），
但**梯度幅度**也更小。这就是为什么 β 和 lr 必须一起调，
不能只改一个。DPO 论文主实验用 β=0.1、lr=1e-6（RMSprop）。
"""

from __future__ import annotations

import math

#: 数值稳定用的裁剪边界。exp 的入参超过这个范围就没意义了。
_CLAMP = 60.0


def sigmoid(x: float) -> float:
    """数值稳定的 sigmoid。

    >>> round(sigmoid(0.0), 6)
    0.5
    >>> round(sigmoid(100.0), 6)
    1.0
    >>> round(sigmoid(-100.0), 6)
    0.0
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-min(x, _CLAMP)))
    ex = math.exp(max(x, -_CLAMP))
    return ex / (1.0 + ex)


def log_sigmoid(x: float) -> float:
    """数值稳定的 log σ(x)。

    推导::

        σ(x)      = 1 / (1 + e^{-x})
        log σ(x)  = −log(1 + e^{−x})

    直接这样算在 ``x`` 很**负**时会让 ``e^{-x}`` 上溢，所以按符号分支::

        x ≥ 0:  log σ(x) = −log1p(e^{−x})        ← e^{-x} ∈ (0,1]，不会上溢
        x < 0:  log σ(x) =  x − log1p(e^{x})     ← e^{x} ∈ (0,1)，且变形后
                                                    避免了 log(1+大数) 的精度损失

    注意 x<0 分支**不是** ``−log1p(e^{x})``：那是 ``log σ(−x)`` 的算法，
    符号会反。这个错误很隐蔽 —— 在 x≈0 附近两者数值接近，
    只有在 x 明显偏离 0 时才暴露（``log σ(−20)`` 应为 −20，
    写成 ``−log1p(e^{−20})`` 会得到 −2e−9）。

    >>> round(log_sigmoid(0.0), 6)
    -0.693147
    >>> round(log_sigmoid(-20.0), 6)
    -20.0
    >>> log_sigmoid(-1000.0) < -999
    True
    >>> round(log_sigmoid(100.0), 12)
    -0.0
    """
    if x >= 0:
        # e^{-x} ∈ (0, 1]，log1p 在此区间精确
        return -math.log1p(math.exp(-x))
    # x < 0：e^{x} ∈ (0, 1)；x − log1p(e^x) 等价于 −log(1 + e^{−x})
    return x - math.log1p(math.exp(x))


def dpo_implicit_reward_margin(
    logp_policy_chosen: float,
    logp_policy_rejected: float,
    logp_ref_chosen: float,
    logp_ref_rejected: float,
    *,
    beta: float = 0.1,
) -> float:
    """DPO 的隐式 reward margin × β，即损失里的 ``z``。

    ``z = β · [ (logπ_θ(y_w) − logπ_ref(y_w)) − (logπ_θ(y_l) − logπ_ref(y_l)) ]``

    >>> z = dpo_implicit_reward_margin(-1.0, -2.0, -1.5, -1.5, beta=0.1)
    >>> round(z, 6)
    0.1
    """
    chosen_adv = logp_policy_chosen - logp_ref_chosen
    rejected_adv = logp_policy_rejected - logp_ref_rejected
    return beta * (chosen_adv - rejected_adv)


def dpo_loss_sigmoid(
    logp_policy_chosen: float,
    logp_policy_rejected: float,
    logp_ref_chosen: float,
    logp_ref_rejected: float,
    *,
    beta: float = 0.1,
) -> float:
    """DPO 的 sigmoid 损失 ``−log σ(z)``。

    >>> round(dpo_loss_sigmoid(-1.0, -1.0, -1.0, -1.0), 6)
    0.693147
    >>> round(dpo_loss_sigmoid(-1.0, -2.0, -1.5, -1.5, beta=0.5), 6)
    0.474077
    """
    z = dpo_implicit_reward_margin(
        logp_policy_chosen, logp_policy_rejected, logp_ref_chosen, logp_ref_rejected, beta=beta
    )
    return -log_sigmoid(z)


def dpo_loss_with_label_smoothing(
    logp_policy_chosen: float,
    logp_policy_rejected: float,
    logp_ref_chosen: float,
    logp_ref_rejected: float,
    *,
    beta: float = 0.1,
    smoothing: float = 0.0,
) -> float:
    """带标签平滑的 DPO 损失。

    平滑的作用：把「chosen 一定优于 rejected」这个硬假设软化，
    防止模型对噪声偏好对过度自信（小数据集上尤其有用）。

    ``smoothing=0`` 时退化为标准 DPO，所以它也是唯一的损失入口。

    >>> a = dpo_loss_with_label_smoothing(-1.0, -2.0, -1.5, -1.5, smoothing=0.0)
    >>> b = dpo_loss_sigmoid(-1.0, -2.0, -1.5, -1.5)
    >>> round(a - b, 12)
    0.0

    平滑对**已经学对**的样本（margin 越大、loss 越小）会抬高损失 ——
    这正是它防过拟合的机制。所以不能用「平滑后损失更大」笼统判断，
    要看具体样本处于哪个区间：

    >>> z_mid = dpo_loss_with_label_smoothing(-1.0, -6.0, -2.0, -2.0, smoothing=0.1)
    >>> dpo_loss_sigmoid(-1.0, -6.0, -2.0, -2.0)
    0.4740769841801067
    >>> round(z_mid - 0.4740769841801067, 4) > 0
    True
    >>> z_mid < 0.6
    True
    """
    z = dpo_implicit_reward_margin(
        logp_policy_chosen, logp_policy_rejected, logp_ref_chosen, logp_ref_rejected, beta=beta
    )
    if smoothing <= 0:
        return -log_sigmoid(z)
    # 标准的对称平滑：把标签 1 换成 (1-ε)，标签 0 换成 ε
    return -(1 - smoothing) * log_sigmoid(z) - smoothing * log_sigmoid(-z)


def dpo_loss_ipo(
    logp_policy_chosen: float,
    logp_policy_rejected: float,
    logp_ref_chosen: float,
    logp_ref_rejected: float,
    *,
    beta: float = 0.1,
) -> float:
    """IPO 损失（Identity Preference Optimization, Azar et al. 2024）。

    动机：DPO 在「chosen 明显优于 rejected」的样本上会持续优化，
    容易过拟合；IPO 用平方损失替代 log-sigmoid，把 margin 拉向
    一个**有限的目标值** ``1/(2β)``，从根本上避免无限推高 margin。

    ``L = (z − 1/(2β))²``，其中 z 的定义与 DPO 相同。
    最小值 0 出现在 ``z = 1/(2β)`` 处。

    >>> round(dpo_loss_ipo(-1.0, -1.0, -1.0, -1.0, beta=0.5), 6)
    1.0
    """
    z = dpo_implicit_reward_margin(
        logp_policy_chosen, logp_policy_rejected, logp_ref_chosen, logp_ref_rejected, beta=beta
    )
    target = 1.0 / (2.0 * beta) if beta > 0 else 0.0
    return (z - target) ** 2


#: 支持的 loss 类型。
#: ⚠️ 与 TRL 的 ``DPOConfig.loss_type`` **不是**同一套取值！
#: TRL 的 ``loss_type`` 支持 "sigmoid" / "hinge" / "ipo" / "kto_pair" /
#: "apo_zero" / "apo_down" 等，但**从来没有** "simpo" 或 "orpo" ——
#: 那两个在 TRL 里是独立的 ``CPOTrainer`` / ``ORPOTrainer``。
#: 本项目默认只用 "sigmoid"（DPO 论文原版），其余作为消融备选。
SUPPORTED_LOSSES: tuple[str, ...] = ("sigmoid", "ipo", "label_smoothing")


def compute_dpo_loss(
    logp_policy_chosen: float,
    logp_policy_rejected: float,
    logp_ref_chosen: float,
    logp_ref_rejected: float,
    *,
    loss_type: str = "sigmoid",
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> float:
    """统一入口，按 ``loss_type`` 分派。

    >>> round(compute_dpo_loss(-1, -1, -1, -1, beta=0.1), 6)
    0.693147
    >>> compute_dpo_loss(-1, -1, -1, -1, loss_type="ipo", beta=0.5) > 0
    True
    """
    if loss_type == "sigmoid":
        return dpo_loss_sigmoid(
            logp_policy_chosen, logp_policy_rejected, logp_ref_chosen, logp_ref_rejected, beta=beta
        )
    if loss_type == "ipo":
        return dpo_loss_ipo(
            logp_policy_chosen, logp_policy_rejected, logp_ref_chosen, logp_ref_rejected, beta=beta
        )
    if loss_type == "label_smoothing":
        return dpo_loss_with_label_smoothing(
            logp_policy_chosen,
            logp_policy_rejected,
            logp_ref_chosen,
            logp_ref_rejected,
            beta=beta,
            smoothing=label_smoothing or 0.1,
        )
    raise ValueError(f"未知 loss_type={loss_type!r}，可选: {SUPPORTED_LOSSES}")


# ============================================================================
# logprob 计算（token 级 → 序列级）
# ============================================================================


def log_prob_from_logits(
    logits: list[float],
    label: int,
    mask: float = 1.0,
) -> float:
    """从单个位置的 logits 与标签算出 log π(token)。

    ``log π = logits[label] − logsumexp(logits)``

    >>> import math
    >>> round(log_prob_from_logits([0.0, 0.0], 0), 6)
    -0.693147
    >>> log_prob_from_logits([0.0, 0.0], 0, mask=0.0)
    0.0
    """
    if mask == 0.0:
        return 0.0
    m = max(logits)
    lse = m + math.log(sum(math.exp(x - m) for x in logits))
    return float(logits[label]) - lse


def sequence_log_prob(
    logits_flat: list[float],
    labels: list[int],
    *,
    vocab_size: int,
    mask: list[float] | None = None,
) -> float:
    """整条序列的 log π(y|x)（按 token 求和）。

    Args:
        logits_flat: 展平的 logits，长度 = ``len(labels) * vocab_size``。
        labels: 每个位置的标签 token id。
        vocab_size: 词表大小。
        mask: 每个位置的权重（0 表示不计入，例如 padding）。

    >>> import math
    >>> lp = sequence_log_prob([0.0, 0.0, 0.0, 0.0], [0, 1], vocab_size=2)
    >>> round(lp, 6)
    -1.386294
    """
    total = 0.0
    for i, lab in enumerate(labels):
        seg = logits_flat[i * vocab_size : (i + 1) * vocab_size]
        w = 1.0 if mask is None else mask[i]
        total += log_prob_from_logits(seg, lab, mask=w)
    return total


# ============================================================================
# 自检：用解析解验证实现
# ============================================================================


def self_check(tol: float = 1e-9) -> dict[str, object]:
    """验证 DPO 损失实现符合论文公式。

    检查项：

    1. ``π_θ = π_ref`` 时 loss 恰为 ``log 2``（对称无偏的起点）
    2. margin 单调增时 loss 单调减
    3. ``z → +∞`` 时 loss → 0
    4. 梯度符号正确（对 chosen 是负、对 rejected 是正）
    5. 标签平滑退化性
    6. IPO 的目标 margin 行为

    Returns:
        dict，``passed`` 为 True 表示全部通过。

    >>> self_check()["passed"]
    True
    """
    checks: dict[str, object] = {}
    ok = True

    # 1. π_θ = π_ref → loss = log 2
    l0 = dpo_loss_sigmoid(-1.0, -2.0, -1.0, -2.0, beta=0.1)
    checks["initial_loss_is_log2"] = abs(l0 - math.log(2.0)) < tol
    ok &= bool(checks["initial_loss_is_log2"])

    # 2. 单调性
    losses = [
        dpo_loss_sigmoid(-1.0, -2.0 - k, -1.5, -1.5, beta=0.5) for k in range(6)
    ]
    checks["monotone_decreasing"] = all(
        losses[i] >= losses[i + 1] - tol for i in range(len(losses) - 1)
    )
    ok &= bool(checks["monotone_decreasing"])

    # 3. 饱和
    checks["saturates_to_zero"] = dpo_loss_sigmoid(50.0, -50.0, 0.0, 0.0, beta=1.0) < 1e-9
    ok &= bool(checks["saturates_to_zero"])

    # 4. 梯度符号（用有限差分验证）
    beta, h = 0.5, 1e-6
    base = (-1.0, -2.0, -1.5, -1.5)
    l_c_plus = dpo_loss_sigmoid(base[0] + h, base[1], base[2], base[3], beta=beta)
    l_c_minus = dpo_loss_sigmoid(base[0] - h, base[1], base[2], base[3], beta=beta)
    grad_c = (l_c_plus - l_c_minus) / (2 * h)
    l_r_plus = dpo_loss_sigmoid(base[0], base[1] + h, base[2], base[3], beta=beta)
    l_r_minus = dpo_loss_sigmoid(base[0], base[1] - h, base[2], base[3], beta=beta)
    grad_r = (l_r_plus - l_r_minus) / (2 * h)

    z = beta * ((base[0] - base[1]) - (base[2] - base[3]))
    sig_neg = sigmoid(-z)
    checks["grad_chosen_matches_analytic"] = abs(grad_c - (-beta * sig_neg)) < 1e-4
    checks["grad_rejected_matches_analytic"] = abs(grad_r - (beta * sig_neg)) < 1e-4
    checks["grad_chosen_negative"] = grad_c < 0
    checks["grad_rejected_positive"] = grad_r > 0
    ok &= all(
        bool(checks[k])
        for k in (
            "grad_chosen_matches_analytic",
            "grad_rejected_matches_analytic",
            "grad_chosen_negative",
            "grad_rejected_positive",
        )
    )

    # 5. 标签平滑退化性
    a = dpo_loss_with_label_smoothing(-1.0, -2.0, -1.5, -1.5, smoothing=0.0)
    b = dpo_loss_sigmoid(-1.0, -2.0, -1.5, -1.5)
    checks["label_smoothing_zero_is_dpo"] = abs(a - b) < tol
    ok &= bool(checks["label_smoothing_zero_is_dpo"])

    # 6. IPO 目标 margin：构造一个 z 恰等于 1/(2β) 的输入，损失应为 0。
    #    用 dpo_implicit_reward_margin 反推输入值，而不是硬编码数字 ——
    #    否则一旦公式微调，这个断言会静默失效。
    beta_ipo = 0.5
    target = 1.0 / (2 * beta_ipo)
    # 令 logp_policy_chosen = target/beta、其余为 0，则 z = beta * target/beta = target
    l_ipo = dpo_loss_ipo(target / beta_ipo, 0.0, 0.0, 0.0, beta=beta_ipo)
    checks["ipo_minimum_at_target"] = l_ipo < tol
    # 顺带确认 z 确实等于目标（防止上面反推写错）
    z_check = dpo_implicit_reward_margin(target / beta_ipo, 0.0, 0.0, 0.0, beta=beta_ipo)
    checks["ipo_target_margin_correct"] = abs(z_check - target) < tol
    ok &= bool(checks["ipo_minimum_at_target"]) and bool(checks["ipo_target_margin_correct"])

    checks["passed"] = bool(ok)
    checks["beta"] = "see individual checks"
    return checks


def report_self_check() -> str:
    """自检结果的人类可读报告。"""
    res = self_check()
    lines = ["DPO 损失实现自检", "-" * 46]
    for k, v in res.items():
        if k in ("passed", "beta"):
            continue
        lines.append(f"  {'✓' if v else '✗'} {k}")
    lines.append("-" * 46)
    lines.append(f"  总体: {'通过' if res['passed'] else '失败'}")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(report_self_check())
