"""测试：pass^k 指标、DPO 损失、偏好数据、mock 流水线一致性。

这一组测试守护的是「方法论正确性」——
指标定义错了、损失公式错了、数据构造错了，实验跑得再顺也没意义。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests._harness import A, test  # noqa: E402


# ============================================================================
# pass@k / pass^k
# ============================================================================


@test
def test_pass_at_k_unbiased_known_values() -> None:
    from src.eval.metrics import pass_at_k_unbiased

    # n=4, c=1 → 取 k=1 时概率 1/4
    A.close(pass_at_k_unbiased(4, 1, 1), 0.25, tol=1e-9)
    # n=4, c=2, k=2 → 1 - C(2,2)/C(4,2) = 1 - 1/6
    A.close(pass_at_k_unbiased(4, 2, 2), 1 - 1 / 6, tol=1e-9)
    # c=n 时任何 k 都是 1
    A.close(pass_at_k_unbiased(4, 4, 3), 1.0, tol=1e-9)
    # c=0 时恒为 0
    A.close(pass_at_k_unbiased(4, 0, 2), 0.0, tol=1e-9)
    # k=1 退化为 c/n
    A.close(pass_at_k_unbiased(8, 3, 1), 3 / 8, tol=1e-9)


@test
def test_pass_hat_k_unbiased_known_values() -> None:
    from src.eval.metrics import pass_hat_k_unbiased

    A.close(pass_hat_k_unbiased(4, 1, 1), 0.25, tol=1e-9)
    # n=4,c=2,k=2 → C(2,2)/C(4,2) = 1/6
    A.close(pass_hat_k_unbiased(4, 2, 2), 1 / 6, tol=1e-9)
    # n=4,c=3,k=2 → C(3,2)/C(4,2) = 3/6
    A.close(pass_hat_k_unbiased(4, 3, 2), 0.5, tol=1e-9)
    # c<k → 0
    A.close(pass_hat_k_unbiased(4, 1, 2), 0.0, tol=1e-9)
    # c=n → 1
    A.close(pass_hat_k_unbiased(4, 4, 4), 1.0, tol=1e-9)


@test
def test_pass_hat_k_is_stricter_than_pass_at_k() -> None:
    """pass^k 必须 <= pass@k（k>1 时严格小于）。

    这个不等式是两者语义的直接推论：k 次全对比至少一次对更难。
    如果实现违反了它，说明公式用反了。
    """
    from src.eval.metrics import pass_at_k_unbiased, pass_hat_k_unbiased

    for n in (4, 6, 8):
        for c in range(n + 1):
            for k in range(1, n + 1):
                pa = pass_at_k_unbiased(n, c, k)
                ph = pass_hat_k_unbiased(n, c, k)
                A.true(ph <= pa + 1e-12, f"n={n} c={c} k={k}: pass^{k}={ph} 不应大于 pass@{k}={pa}")


@test
def test_pass_hat_k_naive_differs_from_unbiased() -> None:
    """朴素估计 (c/n)^k 与无偏估计不同 —— 这正是不能图省事的原因。"""
    from src.eval.metrics import pass_hat_k_naive, pass_hat_k_unbiased

    n, c, k = 4, 2, 2
    A.close(pass_hat_k_naive(n, c, k), 0.25, tol=1e-9)
    A.close(pass_hat_k_unbiased(n, c, k), 1 / 6, tol=1e-9)
    A.true(
        abs(pass_hat_k_naive(n, c, k) - pass_hat_k_unbiased(n, c, k)) > 0.05,
        "朴素与无偏估计应当有可见差异",
    )


@test
def test_metric_helpers() -> None:
    from src.eval.metrics import (
        cohens_h,
        mean,
        percentile,
        required_sample_size,
        stdev,
        wilson_interval,
    )

    A.close(mean([1, 2, 3, 4]), 2.5)
    A.close(percentile([1, 2, 3, 4], 50), 2.5)
    A.close(percentile([1, 2, 3, 4], 0), 1.0)
    A.close(percentile([1, 2, 3, 4], 100), 4.0)
    A.true(stdev([1, 1, 1]) == 0.0)

    lo, hi = wilson_interval(10, 20)
    A.true(lo < 0.5 < hi, "Wilson 区间应包含点估计")
    A.true(0 <= lo <= hi <= 1)

    A.true(required_sample_size(0.10) > required_sample_size(0.20), "效应越小需要样本越多")
    A.true(abs(cohens_h(0.5, 0.5)) < 1e-9, "相同比例效应量应为 0")


@test
def test_eval_report_headline_and_groups() -> None:
    from src.eval.metrics import EvalReport, SampledTaskResult

    results = [
        SampledTaskResult("t1", [True, True, True, True], "flight_search", "easy", "seen"),
        SampledTaskResult("t2", [True, False, False, False], "flight_search", "easy", "seen"),
        SampledTaskResult("t3", [False, False, False, False], "recovery", "hard", "unseen"),
        SampledTaskResult("t4", [True, True, False, False], "recovery", "hard", "unseen"),
    ]
    rep = EvalReport("test", 4, 4, results, k_values=(1, 2, 4), primary_k=4)

    A.close(rep.accuracy().value, (1.0 + 0.25 + 0.0 + 0.5) / 4, tol=1e-6)
    # pass^4: 只有 t1 全对 → 1/4
    A.close(rep.pass_hat(4).value, 0.25, tol=1e-6)
    # Δstability = pass^4 - pass@4，必然 <= 0
    A.true(rep.delta_stability(4).value <= 1e-9, "Δstability 应当 <= 0")

    by_diff = rep.by_difficulty()
    A.eq(set(by_diff), {"easy", "hard"})
    A.true(by_diff["easy"]["accuracy"] > by_diff["hard"]["accuracy"], "easy 组准确率应更高")

    by_vis = rep.by_visibility()
    A.eq(set(by_vis), {"seen", "unseen"})

    d = rep.to_dict()
    A.true("headline" in d and "metrics" in d and "per_task" in d)
    A.eq(len(d["per_task"]), 4)

    # 序列化往返
    import json

    from tests._harness import TempDir

    with TempDir("evalreport") as tmp:
        p = tmp / "r.json"
        rep.write_json(str(p))
        back = json.loads(p.read_text(encoding="utf-8"))
        A.close(back["headline"]["accuracy"], rep.accuracy().value, tol=1e-6)


# ============================================================================
# DPO 损失
# ============================================================================


@test
def test_dpo_loss_self_check() -> None:
    from src.training.dpo_loss import self_check

    res = self_check()
    A.true(res["passed"], f"DPO 损失自检未通过: {res}")


@test
def test_dpo_loss_starts_at_log2() -> None:
    """π_θ = π_ref 时损失恰为 log 2 —— 这是 DPO 的起点。"""
    from src.training.dpo_loss import dpo_loss_sigmoid

    v = dpo_loss_sigmoid(-1.0, -2.0, -1.0, -2.0, beta=0.5)
    A.close(v, math.log(2.0), tol=1e-12)


@test
def test_dpo_loss_beta_scaling() -> None:
    """β 越大越保守：同样的 logprob 差，β 大时 z 大、loss 小。"""
    from src.training.dpo_loss import dpo_loss_sigmoid

    small = dpo_loss_sigmoid(-1.0, -3.0, -2.0, -2.0, beta=0.1)
    large = dpo_loss_sigmoid(-1.0, -3.0, -2.0, -2.0, beta=1.0)
    A.true(large < small, "β 大时 loss 应更小（z 更大）")


@test
def test_dpo_loss_saturates() -> None:
    from src.training.dpo_loss import dpo_loss_sigmoid

    # 正 margin 极大时 loss 趋于 0。
    # 阈值取 1e-20 而不是 1e-80：log σ(-200) 的值约 -200，
    # 任何「x + 小量」的浮点写法都只能保留约 1e-16 的相对精度，
    # 所以 -log σ(z) 的绝对精度下限就是 ~1e-16 量级。
    # 测试应当断言**公式正确**，而不是断言超出浮点能力的东西。
    v = dpo_loss_sigmoid(100.0, -100.0, 0.0, 0.0, beta=1.0)
    A.true(v < 1e-20, f"正 margin 极大时 loss 应趋于 0，实际 {v}")
    A.true(v > 0, "loss 不能为 0（sigmoid 永不为 1）")

    # 负 margin 极大时 loss ≈ -z（线性增长）
    v2 = dpo_loss_sigmoid(-100.0, 100.0, 0.0, 0.0, beta=1.0)
    A.close(v2, 200.0, tol=1e-6, msg="负 margin 极大时 loss 应约等于 -z")

    # 单调性：margin 越大 loss 越小
    prev = float("inf")
    for delta in range(0, 10):
        cur = dpo_loss_sigmoid(-1.0, -1.0 - delta, -1.5, -1.5, beta=0.5)
        A.true(cur <= prev + 1e-12, f"loss 应随 margin 增大而单调下降（delta={delta}）")
        prev = cur


@test
def test_log_sigmoid_is_numerically_stable() -> None:
    """验证 log_sigmoid 在两个方向都正确且不溢出。

    这里最容易犯的错误是**符号写反**：``log σ(x) = −log(1 + e^{−x})``，
    若在 x<0 分支误写成 ``−log1p(e^{x})``，在 x≈0 附近看不出问题，
    但 ``log σ(−20)`` 会得到 −2e−9 而不是 −20。
    所以下面的测试**必须覆盖远离 0 的取值**。
    """
    import math as _m

    from src.training.dpo_loss import log_sigmoid

    # 1. 与定义式 log σ(x) = -log(1 + e^{-x}) 在 x>=0 区间逐点一致
    for x in (0.0, 0.5, 1.0, 5.0, 20.0, 100.0):
        expected = -_m.log1p(_m.exp(-x))
        A.close(log_sigmoid(x), expected, tol=1e-15, msg=f"log_sigmoid({x}) 与定义式不符")

    # 2. 负值区：必须先确认量级对（防符号写反），再确认精度
    A.true(log_sigmoid(-20.0) < -19.9, f"log_sigmoid(-20) 应约为 -20，实际 {log_sigmoid(-20.0)}")
    A.true(log_sigmoid(-5.0) < -5.0, f"log_sigmoid(-5) 应小于 -5，实际 {log_sigmoid(-5.0)}")
    A.true(log_sigmoid(-1000.0) < -999.0, "极大负值不应被截断到 0")

    # 3. 恒等式 log σ(x) + log σ(-x) = -log(2 cosh(x)/... ) 的简化形式：
    #    σ(x) + σ(-x) = 1 → 用 log1p 反推验证一致性
    for x in (-3.0, -1.0, 1.0, 3.0):
        # log(1 - σ(x)) = log σ(-x)
        sig = 1.0 / (1.0 + _m.exp(-x))
        A.close(log_sigmoid(-x), _m.log1p(-sig), tol=1e-12, msg=f"log_sigmoid 在 x={x} 处不满足对称性")

    # 4. 单调性
    xs = [-10.0 + i for i in range(21)]
    vals = [log_sigmoid(x) for x in xs]
    A.true(all(vals[i] < vals[i + 1] for i in range(len(vals) - 1)), "log_sigmoid 应严格单调递增")

    # 5. 上界：log σ(x) 永远 < 0
    A.true(all(v < 0 for v in vals), "log_sigmoid 必须恒为负")


@test
def test_dpo_loss_types() -> None:
    from src.training.dpo_loss import compute_dpo_loss, dpo_loss_ipo

    A.true(compute_dpo_loss(-1, -1, -1, -1, loss_type="sigmoid") > 0)
    A.true(compute_dpo_loss(-1, -1, -1, -1, loss_type="ipo", beta=0.5) >= 0)
    A.true(compute_dpo_loss(-1, -1, -1, -1, loss_type="label_smoothing") > 0)
    A.raises(ValueError, lambda: compute_dpo_loss(-1, -1, -1, -1, loss_type="simpo"))

    # IPO 在目标 margin 处取最小
    beta = 0.5
    target = 1.0 / (2 * beta)
    A.close(dpo_loss_ipo(target / beta, 0.0, 0.0, 0.0, beta=beta), 0.0, tol=1e-12)


@test
def test_log_prob_from_logits() -> None:
    from src.training.dpo_loss import log_prob_from_logits, sequence_log_prob

    A.close(log_prob_from_logits([0.0, 0.0], 0), -math.log(2), tol=1e-12)
    A.close(log_prob_from_logits([0.0, 0.0], 0, mask=0.0), 0.0)
    # 均匀 logits 上两个 token 的序列 logprob = -2*log 2
    A.close(sequence_log_prob([0.0, 0.0, 0.0, 0.0], [0, 1], vocab_size=2), -2 * math.log(2), tol=1e-12)


# ============================================================================
# 偏好数据
# ============================================================================


@test
def test_preference_builder_produces_quality_pairs() -> None:
    from src.data.preference_builder import (
        PAIR_FORMAT,
        PAIR_RECOVERY,
        PreferenceBuilder,
        summarize_pairs,
    )

    b = PreferenceBuilder(seed=7)
    pairs = b.build_all(
        n_format=60, n_recovery=60, n_tool_selection=30, n_param=30, include_param=True
    )
    A.true(len(pairs) > 100, f"应当构造出足量偏好对，实际 {len(pairs)}")

    types = {p.pair_type for p in pairs}
    A.in_(PAIR_FORMAT, types, "应当包含格式约束对")
    A.in_(PAIR_RECOVERY, types, "应当包含错误恢复对")

    # 每条都必须满足基本结构
    for p in pairs:
        A.true(p.prompt.rstrip().endswith("<|im_start|>assistant"), "prompt 必须以 assistant 头结尾")
        A.true(p.chosen.strip() != p.rejected.strip(), "两侧不能完全相同")
        A.true(len(p.chosen) > 0 and len(p.rejected) > 0)
        A.true(p.split in ("train", "dev"))
        A.true(p.intent, "每条都应有教学意图说明")

    # summarize 不应崩
    A.true(len(summarize_pairs(pairs)) > 0)


@test
def test_preference_pairs_are_semantically_distinct() -> None:
    """chosen 与 rejected 在语义上必须真的不同。

    如果等价，这条数据对 DPO 就是噪声（loss 恒为 log 2）。
    """
    from src.data.preference_builder import PreferenceBuilder
    from src.eval.equivalence import equivalent
    from src.eval.tool_call_parser import extract_tool_calls

    pairs = PreferenceBuilder(seed=11).build_all(
        n_format=40, n_recovery=40, n_tool_selection=20, n_param=20
    )
    checked = 0
    degenerate = 0
    for p in pairs:
        c = extract_tool_calls(p.chosen)
        r = extract_tool_calls(p.rejected)
        if not (c.first and r.first and c.first.name == r.first.name):
            continue
        checked += 1
        same = True
        for k, v in c.first.arguments.items():
            if k not in r.first.arguments:
                same = False
                break
            ok, _ = equivalent(v, r.first.arguments[k], param_name=k)
            if not ok:
                same = False
                break
        if same and len(c.first.arguments) == len(r.first.arguments):
            degenerate += 1
    A.eq(degenerate, 0, f"有 {degenerate}/{checked} 条偏好对两侧语义等价（应为 0）")


@test
def test_recovery_pairs_have_coherent_corrections() -> None:
    """纠错话术提到的参数必须真的出现在首轮错误调用里。

    这条测试是为了防止「航班搜索场景 + 『city 参数要填中文城市名』」
    这类自相矛盾的训练数据重新出现。
    """
    import re

    from src.data.preference_builder import PAIR_RECOVERY, PreferenceBuilder
    from src.eval.tool_call_parser import extract_tool_calls

    pairs = PreferenceBuilder(seed=3).build_all(n_format=0, n_recovery=80, n_tool_selection=0, n_param=0)
    rec = [p for p in pairs if p.pair_type == PAIR_RECOVERY]
    A.true(len(rec) > 20, f"应当有足够多的恢复对，实际 {len(rec)}")

    violations = 0
    for p in rec:
        asst = re.findall(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", p.prompt, flags=re.S)
        corr = re.findall(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", p.prompt, flags=re.S)
        if not asst or not corr:
            continue
        wrong = extract_tool_calls(asst[-1])
        if wrong.first is None:
            continue
        wrong_args = set(wrong.first.arguments)
        mentioned = set(
            re.findall(
                r"\b(city|origin|destination|date|booking_id|flight_id|reason|new_baggage_kg)\b",
                corr[-1],
            )
        )
        if mentioned - wrong_args:
            violations += 1
    A.eq(violations, 0, f"有 {violations} 条恢复对的纠错话术提到了不存在的参数")


@test
def test_recovery_corpus_scenarios_valid() -> None:
    from src.data.recovery_corpus import build_scenarios, build_training_tasks, scenario_stats
    from src.eval.tool_call_parser import extract_tool_calls
    from src.env.tool_catalog import VALID_TOOL_NAMES

    sc = build_scenarios(n_variants_per_pattern=3)
    A.true(len(sc) >= 30, f"场景数应足够，实际 {len(sc)}")
    A.eq(len({s.scenario_id for s in sc}), len(sc), "场景 ID 必须唯一")

    for s in sc:
        A.true(s.user_request and s.wrong_reply and s.correction and s.correct_reply, "字段不能为空")
        A.true(s.correction != s.wrong_reply, "纠错话术不能与错误回复相同")
        # 正确回复若是工具调用，工具名必须在白名单
        call = extract_tool_calls(s.correct_reply).first
        if call is not None:
            A.in_(call.name, VALID_TOOL_NAMES, f"{s.scenario_id} 的纠正目标工具必须合法")

    stats = scenario_stats(sc)
    A.eq(stats["total"], len(sc))

    tasks = build_training_tasks(n_variants=3)
    A.true(len(tasks) > 0, "应当能转出训练任务")
    for t in tasks:
        A.eq(t.visibility, "train_only", "训练任务绝不能进入评测集")
        A.true(t.task_id.startswith("trainsyn_"))


@test
def test_experiment_slices_are_disjoint_by_type() -> None:
    from src.data.preference_builder import (
        EXPERIMENT_PAIR_TYPES,
        PAIR_CLARIFY,
        PAIR_FORMAT,
        PAIR_RECOVERY,
        PreferenceBuilder,
        build_experiment_dataset,
    )

    pairs = PreferenceBuilder(seed=5).build_all(
        n_format=40, n_recovery=40, n_tool_selection=20, n_param=20, n_clarify=20
    )
    fmt = build_experiment_dataset("dpo_format", pairs, split="train")
    rec = build_experiment_dataset("dpo_recovery", pairs, split="train")
    clr = build_experiment_dataset("dpo_clarify", pairs, split="train")
    joint = build_experiment_dataset("dpo_joint", pairs, split="train")

    A.true(all(p.pair_type == PAIR_FORMAT for p in fmt), "dpo_format 只应含格式对")
    A.true(all(p.pair_type == PAIR_RECOVERY for p in rec), "dpo_recovery 只应含恢复对")
    A.true(all(p.pair_type == PAIR_CLARIFY for p in clr), "dpo_clarify 只应含追问边界对")
    # dpo_joint = 格式 + 恢复 + 追问边界（实测后加入第三类）
    A.eq(
        len(joint),
        len(fmt) + len(rec) + len(clr),
        "联合方案应等于三类之和（dpo_joint 已含 clarify_boundary）",
    )
    A.true("sft_only" in EXPERIMENT_PAIR_TYPES, "必须定义 SFT 对照组（用于验证对比信号的必要性）")
    sft = build_experiment_dataset("sft_only", pairs, split="train")
    A.true(
        all(p.pair_type != PAIR_RECOVERY for p in sft),
        "SFT 对照组不应包含恢复对 —— 恢复能力无法由只有正确答案的数据教出来",
    )
    A.true(
        all(p.pair_type != PAIR_CLARIFY for p in sft),
        "SFT 对照组不应包含追问边界对 —— 「不该调用」无法由正例表达",
    )


@test
def test_train_dev_split_has_no_task_overlap() -> None:
    """train 与 dev 必须按 task_id 完全分离。

    否则 dev 会因为见过同一 prompt 的其他变体而虚高，起不到早停作用。
    """
    from src.data.preference_builder import PreferenceBuilder, build_experiment_dataset

    pairs = PreferenceBuilder(seed=9).build_all(
        n_format=80, n_recovery=80, n_tool_selection=40, n_param=40, n_clarify=40
    )
    tr = build_experiment_dataset("dpo_joint", pairs, split="train")
    dv = build_experiment_dataset("dpo_joint", pairs, split="dev")
    tr_ids = {p.task_id for p in tr}
    dv_ids = {p.task_id for p in dv}
    A.eq(tr_ids & dv_ids, set(), "train 与 dev 的 task_id 不能有交集")
    # dpo_joint 现在含三类（format / recovery / clarify_boundary）
    wanted = {"format_constraint", "error_recovery", "clarify_boundary"}
    A.eq(len(tr) + len(dv), len([p for p in pairs if p.pair_type in wanted]))


# ============================================================================
# mock 流水线
# ============================================================================


@test
def test_mock_policy_injection_consistency() -> None:
    """mock 注入的错误类型必须能被评分器判成同一类型。

    一致性低说明 mock 生成器与评分器口径漂移，
    此时 mock 产出的所有图表都不可信。
    """
    from src.data.task_manifest import load_tasks
    from src.mock.mock_policy import MockPolicy

    tasks = load_tasks()
    for exp in ("baseline", "dpo_joint"):
        policy = MockPolicy(exp, seed=42)
        chk = policy.verify_injection(tasks, num_samples=2)
        A.true(
            chk["consistency"] >= 0.85,
            f"{exp} 的注入一致性过低: {chk['consistency']:.1%}，混淆矩阵 {chk['confusion']}",
        )


@test
def test_mock_experiments_are_ordered() -> None:
    """各实验的准确率必须满足「联合 > 单方案 > 基线」。

    这验证的是「错误率假设 → 指标」这条映射链是否真的生效。
    如果某个实验的 mitigation 系数改了但指标没动，
    说明成功概率与错误率脱钩了（本项目早期版本就踩过这个坑）。
    """
    from src.data.task_manifest import load_tasks
    from src.eval.grader import run_evaluation
    from src.mock.mock_policy import MockPolicy

    tasks = load_tasks()
    acc: dict[str, float] = {}
    for exp in ("baseline", "dpo_format", "dpo_recovery", "dpo_joint"):
        _, rep = run_evaluation(
            MockPolicy(exp, seed=42).as_policy_fn(tasks),
            tasks,
            label=exp,
            num_samples_per_task=2,
            primary_k=2,
            progress=False,
        )
        acc[exp] = rep.accuracy().value

    A.true(acc["dpo_joint"] > acc["baseline"], f"联合方案应优于基线: {acc}")
    A.true(acc["dpo_format"] > acc["baseline"], f"格式方案应优于基线: {acc}")
    A.true(acc["dpo_recovery"] > acc["baseline"], f"恢复方案应优于基线: {acc}")
    A.true(acc["dpo_joint"] >= max(acc["dpo_format"], acc["dpo_recovery"]) - 0.02, f"联合方案应不弱于单方案: {acc}")


@test
def test_sft_only_cannot_learn_recovery() -> None:
    """本项目核心论点的可执行断言：**SFT 学不到错误恢复能力**。

    这是项目最重要的假设，也是最可能被推翻的假设。它必须被写成测试：
    一旦有人调整 mock 的假设系数使 SFT 在恢复率上也大幅提升，
    这个测试会失败并强制作者正视「论点可能不成立」。

    机制：``sft_only`` 的偏好数据**故意不含** ``error_recovery`` 对，
    因为 chosen 里没有「先错后改」的样本，恢复能力无从学起。
    因此它的恢复率应当贴近基线，而明显低于 dpo_recovery / dpo_joint。
    """
    from src.data.task_manifest import load_tasks
    from src.eval.grader import run_evaluation
    from src.mock.mock_policy import MockPolicy

    tasks = load_tasks()
    rec: dict[str, float] = {}
    for exp in ("baseline", "sft_only", "dpo_recovery", "dpo_joint"):
        _, rep = run_evaluation(
            MockPolicy(exp, seed=42).as_policy_fn(tasks),
            tasks,
            label=exp,
            num_samples_per_task=3,
            primary_k=3,
            progress=False,
        )
        analysis = rep.metadata.get("recovery_analysis", {})
        # 注意键名：单轮评分视角给出的是 recovered_rate（占样本比例），
        # 不是权威的 recovery_rate（改对次数/犯错次数，只有 eval_recovery.py 能算）。
        # 两者对「SFT 学不到恢复」这个假设的判定是同向的，所以用哪个都能验证。
        rec[exp] = analysis.get("recovered_rate", 0.0)

    A.true(all(v > 0 for v in rec.values()), f"各实验都应有恢复率数据: {rec}")

    # 核心断言：只有含 error_recovery 数据的方案才能真正提升恢复能力。
    # 用**绝对差距**而不是倍数比值 —— 基线恢复率本身可能很低，
    # 比率会因分母小而虚高，绝对值更能反映真实差距。
    margin = 0.15  # 15 个百分点
    A.true(
        rec["dpo_recovery"] - rec["sft_only"] > margin,
        f"DPO 恢复方案应显著优于 SFT 对照（绝对差 > {margin:.0%}）: {rec}",
    )
    A.true(
        rec["dpo_joint"] - rec["sft_only"] > margin,
        f"联合方案应显著优于 SFT 对照（绝对差 > {margin:.0%}）: {rec}",
    )
    # SFT 相对基线只应有小幅提升（它只有 NO_RECOVERY 系数的微弱间接效应）
    A.true(
        rec["sft_only"] - rec["baseline"] < 0.20,
        f"SFT 的恢复率相对基线不应有大幅提升（它没有恢复类数据）: {rec}",
    )


@test
def test_train_monitor_detects_unhealthy_training() -> None:
    """训练监控必须能区分「学到东西」与「没学到」。

    这个测试覆盖一个实际踩过的坑：监控回调没有 ``on_*`` 方法时，
    挂载到 Trainer 上会 ``AttributeError`` 崩掉；而更隐蔽的是
    「回调能挂上但从不输出任何东西」—— 手册让你盯 rewards/margin，
    实际日志里根本没有这个量。

    所以这里同时验证两件事：
    1. 回调具备完整接口，能被安全驱动
    2. 健康 / 不健康两种日志都能被正确判读
    """
    import json as _json

    from src.training.train_monitor import (
        LOG2,
        DPOMonitorCallback,
        summarize_dpo_health,
    )
    from tests._harness import TempDir

    class _State:
        def __init__(self, step: int) -> None:
            self.global_step = step
            self.epoch = 1.0

    class _Args:
        max_steps = 100

    with TempDir("monitor") as tmp:
        # ---------- 健康曲线 ----------
        good = tmp / "good"
        cb = DPOMonitorCallback(good, verbose=False)
        # 接口完整性：三个方法都必须存在且可调用
        for m in ("on_train_begin", "on_log", "on_train_end"):
            A.true(hasattr(cb, m), f"回调缺少 {m} 方法（挂到 Trainer 上会崩）")
        cb.on_train_begin(_Args(), _State(0), None)
        for step in range(1, 5):
            p = step / 4
            cb.on_log(
                _Args(),
                _State(step * 25),
                None,
                {
                    "loss": LOG2 * (1 - 0.7 * p),
                    "rewards/margin": 0.5 * p,
                    "rewards/accuracy": 0.5 + 0.4 * p,
                },
            )
        cb.on_train_end(_Args(), _State(100), None)

        A.true((good / "train_log.jsonl").exists(), "监控必须落盘 train_log.jsonl")
        recs = [_json.loads(l) for l in (good / "train_log.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        A.eq(len(recs), 4, "应记录 4 条")
        A.true(all("step" in r for r in recs), "每条记录都要有 step")

        report = summarize_dpo_health(good / "train_log.jsonl")
        A.true("✓" in report, f"健康曲线应判为健康: {report}")
        A.true("⚠️" not in report, f"健康曲线不应有告警: {report}")

        # ---------- 不健康曲线 ----------
        bad = tmp / "bad"
        cb2 = DPOMonitorCallback(bad, verbose=False)
        cb2.on_train_begin(_Args(), _State(0), None)
        for step in range(1, 4):
            cb2.on_log(
                _Args(),
                _State(step * 25),
                None,
                {"loss": 1.05, "rewards/margin": -0.001, "rewards/accuracy": 0.50},
            )
        cb2.on_train_end(_Args(), _State(100), None)
        report2 = summarize_dpo_health(bad / "train_log.jsonl")
        A.true("⚠️" in report2, f"不健康曲线必须给出告警: {report2}")
        A.true("log 2" in report2 or "偏离" in report2, "应指出 loss 偏离 log 2")

        # ---------- 边角情况 ----------
        A.true(len(summarize_dpo_health(tmp / "不存在.jsonl")) > 0, "文件不存在时应给提示而非崩溃")
        empty = tmp / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        A.true(len(summarize_dpo_health(empty)) > 0, "空日志应给提示而非崩溃")


@test
def test_sft_slice_and_dpo_slice_are_both_available() -> None:
    """SFT 与 DPO 两条训练路径都必须有可用的数据切片。

    ``sft_only`` 走 ``scripts/train/sft_train.py``（只用 chosen 算交叉熵），
    其余实验走 ``scripts/train/dpo_train.py``。两者用同一份偏好数据，
    但消费方式不同 —— 这里确认两边的切片都非空，避免出现
    「对照组没数据、静默跳过」的情况。
    """
    from src.data.preference_builder import PreferenceBuilder, build_experiment_dataset

    pairs = PreferenceBuilder(seed=21).build_all(
        n_format=60, n_recovery=60, n_tool_selection=30, n_param=30
    )
    for exp in ("sft_only", "dpo_format", "dpo_recovery", "dpo_joint"):
        tr = build_experiment_dataset(exp, pairs, split="train")
        A.true(len(tr) > 0, f"{exp} 的训练切片不能为空")
    # SFT 侧的实际转换：只用 chosen
    from scripts.train.sft_train import build_sft_records

    sft_pairs = build_experiment_dataset("sft_only", pairs, split="train")
    rows = [p.to_dict() for p in sft_pairs]
    recs = build_sft_records(rows)
    A.eq(len(recs), len(rows), "每条偏好对应产出一条 SFT 样本")
    for r, row in zip(recs, rows):
        A.true(row["chosen"] in r["text"], "SFT 文本必须包含 chosen")
        A.false(row["rejected"] in r["text"], "SFT 文本绝不能包含 rejected（对照组的定义）")


@test
def test_data_generation_is_reproducible() -> None:
    """同一随机种子必须产生**逐字节相同**的数据。

    这个测试覆盖一个实际踩过的坑：数据生成用了内建 ``hash()`` 做随机种子。
    Python 3.3 起 ``hash()`` 对字符串**按进程随机化**（PYTHONHASHSEED），
    所以同一次实验的两次运行会得到**不同的数据** —— 不报错、不警告，
    只是结果悄悄变了。

    后果很严重：消融实验失去可比性（你以为在重复同一个实验，
    其实每次都在换数据）；mock 流水线的「注入一致性」也无故抖动。

    所以这里用「内容哈希」而不是「条数」来断言。条数相同但内容不同
    正是这个 bug 的表现形态 —— 只查条数会漏掉它。
    """
    import ast
    import hashlib

    from src.data.preference_builder import PreferenceBuilder
    from src.data.recovery_corpus import build_scenarios
    from src.utils.hashing import stable_seed

    # 1) stable_seed 本身必须确定性
    A.eq(stable_seed("a", "b", 0), stable_seed("a", "b", 0), "stable_seed 必须确定性")
    A.true(stable_seed(1, 2) != stable_seed(2, 1), "参数顺序不同应得到不同种子")
    A.true(stable_seed("ab", "c") != stable_seed("a", "bc"), "分隔符应防止拼接歧义")
    A.true(0 <= stable_seed("x") <= 0xFFFFFFFF, "种子应落在合法范围")

    # 2) 同一进程内两次构造必须完全一致
    def _digest(pairs) -> str:
        blob = "|".join(p.chosen + "\x1f" + p.rejected for p in pairs).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    a = PreferenceBuilder(seed=42).build_all(
        n_format=40, n_recovery=40, n_tool_selection=20, n_param=20
    )
    b = PreferenceBuilder(seed=42).build_all(
        n_format=40, n_recovery=40, n_tool_selection=20, n_param=20
    )
    A.eq(len(a), len(b), "相同种子应得到相同条数")
    A.eq(_digest(a), _digest(b), "相同种子必须得到逐字节相同的偏好数据")

    # 3) 纠错语料同样必须稳定（它内部用了 hash 派生场景种子）
    s1 = build_scenarios(seed=42)
    s2 = build_scenarios(seed=42)
    d1 = hashlib.sha256("|".join(x.wrong_reply + x.correction for x in s1).encode()).hexdigest()
    d2 = hashlib.sha256("|".join(x.wrong_reply + x.correction for x in s2).encode()).hexdigest()
    A.eq(d1, d2, "纠错语料必须可复现")

    # 4) 不同种子应产生不同数据（否则种子参数形同虚设）
    c = PreferenceBuilder(seed=43).build_all(
        n_format=40, n_recovery=40, n_tool_selection=20, n_param=20
    )
    A.true(_digest(a) != _digest(c), "不同种子应产生不同数据")

    # 5) 源码里不应再出现内建 hash() 做种子（静态检查，防止回归）
    from pathlib import Path

    import re

    repo = Path(__file__).resolve().parents[1]
    offenders: list[str] = []
    for f in list((repo / "src").rglob("*.py")) + list((repo / "scripts").rglob("*.py")):
        if f.name == "hashing.py":
            continue  # 该文件的文档字符串里在解释这个问题
        # 逐文件剥掉文档字符串与注释后再检查 —— 否则「解释这个坑」的文字
        # 本身会被误报为违规代码（开发期就这样误报过 agent/loop.py）。
        raw = f.read_text(encoding="utf-8")
        try:
            tree = ast.parse(raw)
        except SyntaxError:
            continue
        doc_lines: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                body = getattr(node, "body", [])
                if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                        and isinstance(body[0].value.value, str):
                    doc_lines |= set(range(body[0].lineno, (body[0].end_lineno or body[0].lineno) + 1))
        for i, line in enumerate(raw.splitlines(), 1):
            if i in doc_lines:
                continue
            if line.lstrip().startswith("#"):
                continue
            if re.search(r"(?<![\w.])hash\(", line):
                offenders.append(f"{f.relative_to(repo)}:{i}")
    A.eq(
        offenders,
        [],
        "以下位置仍用内建 hash() 做种子（跨进程不稳定）: " + ", ".join(offenders),
    )


@test
def test_clarify_boundary_pairs_are_well_formed() -> None:
    """「该追问 vs 硬调工具」偏好对必须满足四条性质。

    这一类是**真实基线数据逼出来的**：SFT 之后剩下的唯一失败类型就是
    「信息不足时应先追问，却直接调用了工具」（20 条，占失败 100%）。
    原四类偏好对没有任何一类覆盖这个决策边界 —— SFT 的数据全是
    「正确调用」的正例，模型自然被推成「无脑调工具」。

    四条性质：

    1. ``chosen`` 不含工具调用（正确行为是追问）
    2. ``rejected`` 是**合法**的工具调用（不是乱码 —— 模型必须从
       「参数有没有依据」这个层面才能区分两侧）
    3. 两侧不相同（否则没有梯度信号）
    4. ``rejected`` 有足够多样性（否则会被去重压成一条）

    第 4 条对应一个实际踩过的坑：订单号提取用了 ``\\b``，
    而 Python 3 的 ``\\b`` 按 Unicode 语义判断，中文旁不成立，
    于是所有样本的 booking_id 都退化成占位值 ``BK000000``，
    被去重门禁压成**一条** —— 数据量看着正常，实际只剩一个样本。
    """
    from src.data.preference_builder import (
        PAIR_CLARIFY,
        PreferenceBuilder,
        extract_booking_id,
    )
    from src.data.task_manifest import load_tasks
    from src.eval.tool_call_parser import extract_tool_calls

    # 单独验证那个 Unicode 边界坑
    A.eq(extract_booking_id("我不想要BK9A4C6F这个订单了"), "BK9A4C6F", "中文旁的订单号必须能提取")
    A.eq(extract_booking_id("帮我取消订单BK7F2A9C"), "BK7F2A9C")
    A.eq(extract_booking_id("我要退票"), "", "没有订单号时返回空串")

    tasks = list(load_tasks())
    pairs = PreferenceBuilder(seed=42).build_clarify_pairs(tasks, 40)

    A.true(len(pairs) >= 20, f"应能构造出足够多的 clarify 对，实际 {len(pairs)}")
    A.true(all(p.pair_type == PAIR_CLARIFY for p in pairs), "类型必须是 clarify_boundary")

    # 性质 1：chosen 不能含工具调用
    for p in pairs:
        A.false("<tool_call>" in p.chosen, f"chosen 不该含工具调用: {p.chosen[:60]!r}")

    # 性质 2：rejected 必须能解析成合法调用
    for p in pairs:
        A.true(bool(extract_tool_calls(p.rejected).calls), f"rejected 应是合法调用: {p.rejected[:80]!r}")

    # 性质 3：两侧不同
    for p in pairs:
        A.true(p.chosen.strip() != p.rejected.strip(), "两侧不能相同（否则梯度为零）")

    # 性质 4：多样性 —— 不能被去重压成一条
    distinct_rejected = len({p.rejected for p in pairs})
    A.true(
        distinct_rejected >= 5,
        f"rejected 多样性过低（只有 {distinct_rejected} 种）—— 疑似订单号提取失败退化成占位值",
    )
    ids = {p.task_id for p in pairs if p.task_id}
    A.true(len(ids) >= 2, f"应覆盖多个任务，实际 {ids}")


@test
def test_sft_only_excludes_clarify_boundary() -> None:
    """SFT 对照组不能包含「该追问」边界对 —— 这是它能被证伪的前提。

    这一类的学习**必须依赖对比信号**：要让模型学会「此处不该调用」，
    就必须看到「硬调工具」这个反面例子。SFT 只用 chosen 做监督，
    无法表达「不该调用」—— 一个纯正例数据无法教会模型「何时不做某事」。

    所以 sft_only 的切片里绝不能有这个类型。若混进去了，
    「SFT 学不会该追问边界」这个论证就不成立 —— 它其实拿到了反面例子，
    只是用法不同，两个变量一起动了。
    """
    from src.data.preference_builder import (
        EXPERIMENT_PAIR_TYPES,
        PAIR_CLARIFY,
        PreferenceBuilder,
        build_experiment_dataset,
    )

    A.false(
        PAIR_CLARIFY in EXPERIMENT_PAIR_TYPES["sft_only"],
        "sft_only 的数据切片里不能包含 clarify_boundary —— 这是对照实验的前提",
    )
    # 而 dpo 侧必须包含（否则 DPO 也无从学起）
    A.true(
        PAIR_CLARIFY in EXPERIMENT_PAIR_TYPES["dpo_joint"],
        "dpo_joint 必须包含 clarify_boundary（实测唯一的失败类型）",
    )
    A.true(PAIR_CLARIFY in EXPERIMENT_PAIR_TYPES["dpo_clarify"])

    pairs = PreferenceBuilder(seed=11).build_all(
        n_format=30, n_recovery=0, n_tool_selection=0, n_param=0, n_clarify=20
    )
    sft = build_experiment_dataset("sft_only", pairs, split="train")
    A.false(any(p.pair_type == PAIR_CLARIFY for p in sft), "实际数据里 sft_only 也不能混入这一类")


@test
def test_prompt_consistency_checker_catches_mismatch() -> None:
    """一致性检查器必须能**真的**发现不一致。

    这个测试守的是一个会给「假通过」的缺陷。原实现有两个问题：

    1. **不带 ``task_id`` 的样本被完全跳过**（``if task is None: continue``）。
       于是一份一条 task_id 都没有的偏好数据会被判为「一致」，
       而它实际上一条都没被检查。实测中 ``sft_only.train.jsonl``
       的 2435 条全都没有 task_id。
    2. **只比对 ``[:400]``**，而 system prompt 约 1986 字符 ——
       工具定义部分（约 1500 字符）从来没被校验过。

    这里用「明显不同的 system prompt」构造一份数据，
    新检查器必须判为不一致。
    """
    from src.data.preference_builder import render_prompt, render_user_prompt
    from src.data.task_manifest import load_tasks
    from src.training.dpo_dataset import verify_prompt_consistency

    tasks = list(load_tasks())

    # 正确：用项目的渲染器
    good = render_prompt(tasks[0])
    r = verify_prompt_consistency([{"prompt": good, "task_id": "", "pair_type": "x"}], tasks)
    A.true(r["checked"] >= 1, "必须真的检查了样本（旧版会跳过无 task_id 的样本）")
    A.eq(r["num_mismatch"], 0, "正确渲染的数据应判为一致")
    A.true(r["consistent"])

    # 错误：换一个完全不同的 system prompt
    bad = render_user_prompt("X", system_prompt="完全不同的系统提示")
    r2 = verify_prompt_consistency([{"prompt": bad, "task_id": "", "pair_type": "x"}], tasks)
    A.eq(r2["num_mismatch"], 1, "system prompt 不同必须被检出")
    A.false(r2["consistent"], "不一致时必须判为不一致")
    A.true(len(r2["examples"]) > 0, "应给出差异示例")


@test
def test_frozen_eval_tasks_unchanged() -> None:
    """扩集**绝不能改变**原有 100 个任务 —— 否则历史结果全部作废。

    baseline / sft_v2 / sft_v3 / dpo_from_sft / dpo_sft_v3_b05 全部跑在
    这 100 个任务上。如果扩集顺手改了它们的措辞或期望值，那些数字就
    不可比了，而**这种改变是静默的**（评测照常跑、照常出数）。

    所以这里用逐任务指纹断言：100 个冻结任务的 JSON 序列化必须与
    ``experiments/_frozen_fingerprint.json`` 完全一致。
    """
    import hashlib
    import json
    from pathlib import Path as _P

    from src.data.task_manifest import TASK_SET_SIZES, build_tasks

    fp_path = _P("experiments/_frozen_fingerprint.json")
    # ⚠️ 这条断言消息以前写的是「应由 scripts 生成」，
    # **但当时没有那个脚本** —— 文件是我手工造的，一旦丢失就无法重建。
    # 实测后果：部署脚本的保护机制跳过了它，云端这个测试挂掉，
    # 而错误消息指向一个不存在的能力。
    # 现在补了 `scripts/prepare/freeze_v1_tasks.py`，消息也指向真实路径。
    A.true(
        fp_path.exists(),
        "缺少冻结指纹文件。重建：python scripts/prepare/freeze_v1_tasks.py"
        "（校验：--check）",
    )
    fp = json.loads(fp_path.read_text(encoding="utf-8"))

    base = build_tasks()
    A.eq(len(base), TASK_SET_SIZES["v1"], "v1 任务数应为 100")

    changed = []
    for t in base:
        digest = hashlib.sha256(
            json.dumps(t.to_dict(), ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()[:16]
        if fp.get(t.task_id) != digest:
            changed.append(t.task_id)
    A.eq(changed, [], f"这些冻结任务被改动了，历史结果将不可比：{changed[:5]}")


@test
def test_deprecated_400_task_set_is_gone() -> None:
    """400 任务的扩集方案已被**否决**，不应再存在于代码里。

    背景：我一度按「扩到 400 把 CI 压窄」推进，但专家指出——
    实测 400 个任务只有 208 种骨架、8 种签名，
    ``multi_tool`` 更是 53 个任务 8 种骨架。
    **有效独立样本根本不是 400**，那个 CI 是虚假精度。

    结论：改为「100 个真正独立的任务」。
    400 任务的生成器已删除；这条测试防止它被无意恢复。

    （v2 高质量 100 任务集的约束检查见 ``tests/test_task_set_v2.py``。）
    """
    import src.data.task_manifest as M

    A.false(
        hasattr(M, "build_tasks_v2"),
        "400 任务的 build_tasks_v2 已被否决，不应再存在",
    )


@test
def test_extra_param_only_flags_fabricated_fields() -> None:
    """「多给参数」只能判**编造**的字段，不能判 schema 里合法的参数。

    这是评分器最严重的一处口径错误，实测影响：

        sft_v3 的 42 条 extra_param 里，36 条是误判
         32x  cancel_booking  多给=('reason',)    ← reason 是 schema **必填**
          4x  book_flight     多给=('flight_id',) ← flight_id 是 schema **必填**

    即**模型给对了必填参数，反而被判成「多给了 schema 未定义的参数」**。
    它把 extra_param 抬成了头号失败原因（占全部失败的 58.8%）。

    根因：``expected.args`` 往往只是「答案的一部分」—— 有的任务有意不把
    需要模型自己获取的参数（flight_id）或自由文本（reason）列进去。
    但它仍是该工具的合法参数，不该算错。

    本测试守住三个方向：
    1. schema 必填但未评分的参数（reason / flight_id）→ 不算错
    2. schema 合法可选的参数（cabin）→ 不算错
    3. **schema 里没有的字段 → 必须仍判 extra_param**
    """
    from src.data.task_manifest import load_tasks
    from src.env.tool_catalog import TOOL_REGISTRY
    from src.eval.grader import grade

    ts = {t.task_id: t for t in load_tasks()}

    # --- cancel_booking：reason 是 schema 必填但不在 expected.args 里 ---
    t = ts["task_0057"]
    A.eq(t.expected.tool, "cancel_booking")
    A.true("reason" not in (t.expected.args or {}), "该任务的 expected.args 有意不含 reason")
    A.true(
        "reason" in TOOL_REGISTRY["cancel_booking"].required_params,
        "但 reason 是该工具 schema 的必填参数",
    )

    def _subtype(task, tool, args):
        import json as _json

        text = "<tool_call>" + _json.dumps(
            {"name": tool, "arguments": args}, ensure_ascii=False
        ) + "</tool_call>"
        return getattr(grade(task, text).classification.detail, "subtype", None)

    A.true(
        _subtype(t, "cancel_booking", {"booking_id": "BK2B7D5E", "reason": "身体不适"})
        != "extra_param",
        "给出 schema 必填的 reason 不得被判成 extra_param",
    )
    # 方案 3：编造字段必须仍被抓住
    A.eq(
        _subtype(
            t, "cancel_booking", {"booking_id": "BK2B7D5E", "reason": "身体不适", "zzz": 1}
        ),
        "extra_param",
        "schema 里没有的字段必须仍判 extra_param",
    )

    # --- book_flight：flight_id 必填、cabin 合法可选 ---
    t2 = ts["task_0097"]
    A.eq(t2.expected.tool, "book_flight")
    A.true("flight_id" not in (t2.expected.args or {}), "该任务不评 flight_id")
    A.true("flight_id" in TOOL_REGISTRY["book_flight"].required_params)
    A.true("cabin" in TOOL_REGISTRY["book_flight"].params, "cabin 是合法可选参数")

    for args in (
        {"passenger_name": "刘洋", "flight_id": "MU1000"},
        {"passenger_name": "刘洋", "cabin": "经济舱"},
        {"passenger_name": "刘洋", "flight_id": "MU1000", "cabin": "经济舱"},
    ):
        A.true(
            _subtype(t2, "book_flight", args) != "extra_param",
            f"schema 合法参数不得被判 extra_param：{args}",
        )
    A.eq(
        _subtype(t2, "book_flight", {"passenger_name": "刘洋", "zzz": 1}),
        "extra_param",
        "编造字段必须仍判 extra_param",
    )


@test
def test_training_data_never_contains_eval_tasks() -> None:
    """训练数据**绝不能**包含评测任务的 prompt —— 否则结论全部作废。

    这是本项目发现的最严重缺陷。原实现里 ``build_format_pairs`` /
    ``build_param_pairs`` / ``build_tool_selection_pairs`` 都直接吃
    ``pool = 评测任务 + 合成任务``，而 ``build_recovery_pairs`` 的注释却写着
    「评测与训练因此完全分离」—— **只有它做到了**。

    实测污染量：

        sft_only 的 2509 条训练样本里，1292 条（51.5%）的 prompt
        与评测 prompt **逐字相同**，且 chosen 就是该任务的正确答案。

    这等于在测试集上做梯度下降：SFT 的 95.0% 准确率因此虚高，
    与泛化能力无关。

    本测试同时守住四件事：
    1. 硬门禁：把评测任务传给 ``build_all`` 必须抛 ValueError
    2. 实际生成的偏好数据：没有任何一条的 prompt 与评测 prompt 逐字相同
    3. 合成语料覆盖评测所需的全部工具（否则那些任务必然全错）
    4. 被污染的黑名单与检测函数一致
    """
    from src.data.preference_builder import PreferenceBuilder
    from src.data.recovery_corpus import (
        CONTAMINATED_EVAL_TASKS,
        find_contaminated_eval_tasks,
        build_training_tasks,
    )
    from src.data.task_manifest import load_tasks

    ev = list(load_tasks())

    # --- 1) 硬门禁 ---
    try:
        PreferenceBuilder(seed=1).build_all(ev, n_format=5, n_recovery=0,
                                            n_tool_selection=0, n_param=0, n_clarify=0)
    except ValueError as exc:
        A.true("评测任务" in str(exc), f"门禁报错信息应说明原因：{exc}")
    else:  # pragma: no cover
        raise AssertionError("把评测任务传给 build_all 必须抛 ValueError")

    # --- 2) 合成语料覆盖全部评测所需工具 ---
    syn = build_training_tasks(seed=42)
    syn_tools = {t.expected.tool for t in syn if t.expected.tool}
    ev_tools = {t.expected.tool for t in ev if t.expected.tool}
    missing = ev_tools - syn_tools
    A.eq(missing, set(), f"合成语料缺少这些工具的训练样本，评测时必然全错：{missing}")
    A.true(
        any(t.expected.tool is None for t in syn),
        "合成语料需含「该追问」（expected.tool is None）任务 —— 那是否定性约束的唯一载体",
    )

    # --- 3) 实际数据：没有一条与评测 prompt 逐字相同 ---
    pairs = PreferenceBuilder(seed=42).build_all(
        n_format=40, n_recovery=40, n_tool_selection=20, n_param=20, n_clarify=20
    )
    from src.models.prompt_builder import build_plain_prompt

    ev_prompts = {build_plain_prompt(t).strip() for t in ev}
    hit = [p.task_id for p in pairs if p.prompt.strip() in ev_prompts]
    A.eq(hit, [], f"这些偏好对的 prompt 与评测 prompt 逐字相同：{hit[:5]}")

    # --- 4) 黑名单与实际检测一致 ---
    found = find_contaminated_eval_tasks(ev)
    A.eq(
        sorted(found),
        sorted(CONTAMINATED_EVAL_TASKS),
        "黑名单与实测的逐字撞车任务不一致 —— 语料变了就要同步更新",
    )


@test
def test_prefix_constant_actually_marks_synthetic_corpus() -> None:
    """``SYNTHETIC_TASK_PREFIX`` 的**语义**必须与名字一致。

    ## 为什么单独守这一条

    这个常量原先叫 ``EVAL_TASK_PREFIX``，而它的值是 ``"trainsyn_"`` ——
    **合成训练语料**的前缀。名字与语义完全相反，后果有两层：

    1. ``build_all`` 的报错信息写成「训练数据中混入了评测任务
       （样例 ``task_0001``）」，而 ``task_0001`` 恰恰是被错误混进来的那个；
    2. 更危险的是，门禁判据是 ``not task_id.startswith(该常量)``。
       若后人看到名字后"顺手修正"成 ``startswith(...)``，门禁会**静默失效**，
       而既有的 ``test_training_data_never_contains_eval_tasks`` 仍会通过 ——
       它只验证行为，不验证这个不变量。

    本测试直接把不变量钉死，与实现细节解耦：
    合成语料必须带该前缀，**评测任务必须一个都不带**。
    """
    from src.data.recovery_corpus import (
        SYNTHETIC_TASK_PREFIX,
        build_training_tasks,
    )
    from src.data.task_manifest import load_tasks, load_tasks_by_version
    from src.data.task_manifest_v2 import load_tasks_v3

    # 1) 常量名与语义一致：它是「合成」前缀，不是「评测」前缀
    A.eq(
        SYNTHETIC_TASK_PREFIX,
        "trainsyn_",
        "合成语料前缀被改动；若确需改动，须同步 build_all 的判据与报错文案",
    )

    syn = build_training_tasks(seed=42)
    off = [t.task_id for t in syn if not t.task_id.startswith(SYNTHETIC_TASK_PREFIX)]
    A.eq(off, [], f"合成语料里出现不带 {SYNTHETIC_TASK_PREFIX!r} 前缀的任务：{off[:5]}")

    # 2) 评测任务**绝不能**带这个前缀 —— 否则门禁会把它们当成合法训练数据放行
    ev_all = list(load_tasks()) + list(load_tasks_by_version("v1")) + list(load_tasks_v3())
    bad = [t.task_id for t in ev_all if t.task_id.startswith(SYNTHETIC_TASK_PREFIX)]
    A.eq(
        bad,
        [],
        f"评测任务带了合成语料前缀 {SYNTHETIC_TASK_PREFIX!r}：{bad[:5]} —— "
        "这会让 build_all 的门禁把它们判为合法训练数据",
    )

    # 3) 旧名保留为别名，且指向同一值（兼容既有调用方）
    from src.data.recovery_corpus import EVAL_TASK_PREFIX

    A.eq(EVAL_TASK_PREFIX, SYNTHETIC_TASK_PREFIX, "旧名别名应指向同一前缀值")


@test
def test_sft_only_slice_excludes_recovery_pairs() -> None:
    """SFT 对照组的偏好数据里不能出现 error_recovery。

    如果混进去了，「SFT 学不到恢复」这个论证就不成立 ——
    它其实拿到了恢复数据，只是训练目标不同，两个变量一起动了。
    """
    from src.data.preference_builder import (
        EXPERIMENT_PAIR_TYPES,
        PAIR_RECOVERY,
        PreferenceBuilder,
        build_experiment_dataset,
    )

    A.true("sft_only" in EXPERIMENT_PAIR_TYPES, "必须定义 sft_only 实验切片")
    A.false(
        PAIR_RECOVERY in EXPERIMENT_PAIR_TYPES["sft_only"],
        "sft_only 的数据切片里不能包含 error_recovery —— 这是整个对照实验的前提",
    )

    pairs = PreferenceBuilder(seed=13).build_all(
        n_format=40, n_recovery=40, n_tool_selection=20, n_param=20
    )
    sft = build_experiment_dataset("sft_only", pairs, split="train")
    A.true(len(sft) > 0, "sft_only 应当有数据")
    A.false(any(p.pair_type == PAIR_RECOVERY for p in sft), "实际数据里也不能混入恢复对")


@test
def test_mock_sampling_has_variance_across_attempts() -> None:
    """同一任务的不同采样必须产生不同结果。

    如果逐次采样用了同一个种子，4 次采样会完全相同，
    pass^k 会恒等于 accuracy，「稳定性」这个维度被静默抹掉。
    """
    from src.data.task_manifest import load_tasks
    from src.mock.mock_policy import MockPolicy

    tasks = load_tasks()[:40]
    policy = MockPolicy("baseline", seed=42)
    differing = 0
    for t in tasks:
        texts = {policy.sample(t, i).text for i in range(4)}
        if len(texts) > 1:
            differing += 1
    A.true(
        differing >= len(tasks) // 3,
        f"只有 {differing}/{len(tasks)} 个任务的多次采样结果不同 —— 采样种子可能没变化",
    )


@test
def test_mock_pipeline_produces_artifacts() -> None:
    """完整 mock 流水线要能跑通并产出产物。"""
    import json

    from src.mock.mock_pipeline import MockPipelineConfig, run_mock_pipeline
    from tests._harness import TempDir

    with TempDir("mockpipe") as tmp:
        cfg = MockPipelineConfig(
            num_samples_per_task=2,
            primary_k=2,
            task_limit=12,
            n_format=20,
            n_recovery=20,
            n_tool_selection=10,
            train_steps=10,
            train_epochs=1,
            eval_every=5,
            run_training=True,
            make_plots=False,
            out_root=str(tmp),
        )
        res = run_mock_pipeline(cfg, verbose=False)
        out = Path(res.out_dir)
        A.true((out / "pipeline_summary.mock.json").exists(), "应产出流水线摘要")
        A.true((out / "predictions_baseline.mock.jsonl").exists(), "应产出基线预测")
        A.eq(len(res.experiments), 5, "应覆盖 5 个实验")
        for exp in res.experiments:
            A.true(exp in res.reports, f"{exp} 应有报告")
            A.true("headline" in res.reports[exp])
        A.true(res.consistency["consistency"] >= 0.85, "注入一致性应达标")
        # mock 摘要必须显式标注，否则会被人误当成真实结果
        summary = json.loads((out / "pipeline_summary.mock.json").read_text(encoding="utf-8"))
        A.true(summary.get("is_mock") is True, "摘要必须带 is_mock 标记")
        A.true(
            "SIMULATED" in summary.get("warning", "") or "mock" in summary.get("warning", "").lower(),
            "摘要必须带免责说明",
        )
        # 目录结构不能出现 outputs/outputs 这类重复前缀
        A.false("/outputs/outputs" in out.as_posix() or "\\outputs\\outputs" in str(out), "路径不应重复拼接 outputs")
