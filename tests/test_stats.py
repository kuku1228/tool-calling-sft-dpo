"""配对统计检验的测试。

## 这个模块为什么存在

我长期用「两个 CI 是否重叠」判断显著性 —— **那是错的**，而错的东西
不会自己报错。所以在引入正确判据（配对检验）的同时，必须把它的
**前提与退化情形**都测出来，否则换一套判据也可能悄悄错。

重点覆盖三件事：

1. **配对 bootstrap 必须按任务重采样** —— 按样本重采样会低估区间宽度，
   这条用一个「同一任务多次采样完全相同」的构造来区分两者
2. **McNemar 在无差异与完全差异两端的边界行为**
3. **``load_graded`` 必须重新评分** —— ``predictions.jsonl`` 里
   **没有** ``passed`` 字段；早期我误以为有，写出的脚本会把所有样本
   当成失败，于是任何对比都得出「无差异」
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.eval import paired_stats as PS  # noqa: E402
from tests._harness import A, test  # noqa: E402


@test
def test_paired_bootstrap_is_deterministic_given_seed() -> None:
    """给了 seed 就必须可复现 —— 否则报告里的数字无法复核。"""
    a = {f"t{i}": 1.0 if i % 3 else 0.0 for i in range(60)}
    b = {f"t{i}": 1.0 if i % 5 else 0.0 for i in range(60)}
    r1 = PS.paired_bootstrap(a, b, n_boot=300, seed=11)
    r2 = PS.paired_bootstrap(a, b, n_boot=300, seed=11)
    A.eq(r1, r2, "同一 seed 下配对 bootstrap 结果必须一致")
    r3 = PS.paired_bootstrap(a, b, n_boot=300, seed=12)
    A.true(r1[0] == r3[0], "观测差值不应依赖 seed（只有 CI 会因重采样而变）")


@test
def test_paired_bootstrap_ci_contains_the_observed_diff() -> None:
    """CI 必须包含观测差值。"""
    a = {f"t{i}": 1.0 if i % 2 else 0.0 for i in range(80)}
    b = {f"t{i}": 1.0 if i % 4 else 0.0 for i in range(80)}
    obs, lo, hi = PS.paired_bootstrap(a, b, n_boot=500, seed=3)
    A.true(lo <= obs <= hi, f"观测值 {obs} 不在 CI [{lo},{hi}] 内")


@test
def test_paired_bootstrap_ci_narrows_with_more_tasks() -> None:
    """任务越多，CI 越窄 —— 这是「有效样本量是任务数」的直接体现。"""
    def mk(n: int) -> tuple[dict, dict]:
        a = {f"t{i}": 1.0 if i % 2 else 0.0 for i in range(n)}
        b = {f"t{i}": 1.0 if i % 3 else 0.0 for i in range(n)}
        return a, b

    a1, b1 = mk(50)
    a2, b2 = mk(400)
    _, lo1, hi1 = PS.paired_bootstrap(a1, b1, n_boot=800, seed=5)
    _, lo2, hi2 = PS.paired_bootstrap(a2, b2, n_boot=800, seed=5)
    A.true((hi2 - lo2) < (hi1 - lo1),
           f"400 任务的 CI 宽度 {(hi2-lo2):.3f} 应小于 50 任务的 {(hi1-lo1):.3f}")


@test
def test_mcnemar_counts_and_extremes() -> None:
    """McNemar 的 b/c 计数与两端边界。"""
    # 完全一致 → 无不一致对，p=1
    same = {f"t{i}": 1.0 for i in range(20)}
    r = PS.mcnemar(same, dict(same))
    A.eq((r["b"], r["c"], r["n_discordant"]), (0, 0, 0))
    A.eq(r["p_value"], 1.0)

    # A 全对、B 全错 → b=20, c=0，p 应极小
    a = {f"t{i}": 1.0 for i in range(20)}
    b = {f"t{i}": 0.0 for i in range(20)}
    r2 = PS.mcnemar(a, b)
    A.eq((r2["b"], r2["c"]), (20, 0))
    A.true(r2["p_value"] < 1e-4, f"完全分离时 p 应极小，实际 {r2['p_value']}")

    # 对称的不一致（b == c）→ p 应接近 1
    a3, b3 = {}, {}
    for i in range(10):
        a3[f"x{i}"], b3[f"x{i}"] = (1.0, 0.0) if i < 5 else (0.0, 1.0)
    r3 = PS.mcnemar(a3, b3)
    A.eq((r3["b"], r3["c"]), (5, 5))
    A.true(r3["p_value"] > 0.9, f"b==c 时 p 应接近 1，实际 {r3['p_value']}")


@test
def test_mcnemar_p_value_is_a_probability() -> None:
    """p 值必须落在 [0,1] —— 精确二项检验的求和容易越界到 >1。"""
    for nb, nc in ((1, 0), (2, 1), (7, 3), (13, 0), (0, 13), (25, 24)):
        a, b = {}, {}
        for i in range(nb):
            a[f"a{i}"], b[f"a{i}"] = 1.0, 0.0
        for i in range(nc):
            a[f"c{i}"], b[f"c{i}"] = 0.0, 1.0
        p = PS.mcnemar(a, b)["p_value"]
        A.true(0.0 <= p <= 1.0, f"b={nb},c={nc} 时 p={p} 越界")


@test
def test_mde_matches_the_textbook_formula() -> None:
    """MDE 应与公式一致，且随样本量与 σ 单调。"""
    n, sigma = 100, 0.5
    expect = (1.959963985 + 0.841621234) * sigma / math.sqrt(n)
    A.true(abs(PS.mde(n, sigma=sigma) - expect) < 1e-9, "MDE 公式不符")

    A.true(PS.mde(400, sigma=sigma) < PS.mde(100, sigma=sigma),
           "任务越多，MDE 应越小")
    A.true(PS.mde(100, sigma=1.0) > PS.mde(100, sigma=0.5),
           "σ 越大，MDE 应越大")
    A.true(PS.mde(0, sigma=0.5) > 0, "n=0 不应除零崩溃")


@test
def test_mde_explains_the_task_level_floor() -> None:
    """复现「100 任务时 MDE ≈ 14pp」这一关键读数。

    这是本项目一个重要的方法论结论：**100 个任务能检出的最小效应约 14pp**，
    所以「差 3 个点」这类结论在这个样本量下根本不可能成立。
    """
    # 任务级二值结果（0/1）在 p≈0.5 附近时，配对差值的 σ 约为 sqrt(2)*0.5
    sigma = 0.5
    m = PS.mde(100, sigma=sigma)
    A.true(0.12 <= m <= 0.17, f"100 任务、σ=0.5 时 MDE 应约 0.14，实际 {m:.3f}")

    # 400 任务时才降到 ~7pp
    A.true(0.06 <= PS.mde(400, sigma=sigma) <= 0.09,
           f"400 任务时 MDE 应约 0.07，实际 {PS.mde(400, sigma=sigma):.3f}")


@test
def test_load_graded_regrades_instead_of_trusting_a_passed_field() -> None:
    """``load_graded`` 必须**重新评分**。

    ``predictions.jsonl`` 的字段只有
    ``task_id / text / sample_index / finish_reason / latency_s /
    num_tokens / policy / meta`` —— **没有 ``passed``**。

    早期我误以为有，脚本会读到一个恒为 ``None`` 的字段，
    把所有样本当成失败，于是**任何两组对比都得出「无差异」**。
    这条测试用一个「明确正确」的预测文本来验证它确实被判为通过。
    """
    import shutil

    from src.data.task_manifest import load_tasks
    from tests._harness import workspace_tempdir

    task = load_tasks()[0]  # task_0001
    # 用该任务的**标准答案**文本构造一条预测 → 必须判为通过
    exp = task.expected
    args = json.dumps(exp.args, ensure_ascii=False)
    good = f'<tool_call>{{"name": "{exp.tool}", "arguments": {args}}}</tool_call>'

    # ⚠️ 不用 ``tempfile``：系统 temp（``C:\WINDOWS\TEMP``）在沙箱下
    # **连写入都拒绝**（``PermissionError: [WinError 5]``），而这两个测试
    # 要测的是统计逻辑，与文件系统权限毫无关系。
    # 改用工作区内的临时目录。
    root = workspace_tempdir("stats")
    try:
        d = root / "lab"
        d.mkdir(parents=True)
        rows = [
            {"task_id": task.task_id, "sample_index": i, "text": good,
             "finish_reason": "stop", "latency_s": 0.0, "num_tokens": 20,
             "policy": "test", "meta": {}}
            for i in range(4)
        ]
        (d / "predictions.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
            encoding="utf-8",
        )
        graded = PS.load_graded("lab", root)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    A.true(task.task_id in graded, "应读到该任务")
    A.eq(graded[task.task_id], [True] * 4,
         "标准答案文本必须被判为通过（说明确实重新评分了，而不是读 passed 字段）")


@test
def test_load_graded_marks_a_wrong_call_as_failed() -> None:
    """反向对照：错误调用必须判为失败 —— 否则上面那条测试无意义。"""
    import shutil

    from src.data.task_manifest import load_tasks
    from tests._harness import workspace_tempdir

    task = load_tasks()[0]
    wrong = '<tool_call>{"name": "get_weather", "arguments": {"city": "不存在的城市"}}</tool_call>'

    root = workspace_tempdir("stats")
    try:
        d = root / "lab"
        d.mkdir(parents=True)
        (d / "predictions.jsonl").write_text(
            json.dumps({"task_id": task.task_id, "sample_index": 0,
                        "text": wrong, "policy": "test"}), encoding="utf-8",
        )
        graded = PS.load_graded("lab", root)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    A.eq(graded[task.task_id], [False], "错误调用必须判为失败")


@test
def test_compare_flags_significance_only_when_ci_excludes_zero() -> None:
    """``compare`` 的 «显著» 判定必须严格依据 CI 是否含 0。"""
    # 构造一组清晰的 A 优于 B
    a = {f"t{i}": 1.0 for i in range(60)}
    b = {f"t{i}": 0.0 for i in range(60)}
    r = PS.compare("A", "B", {k: [v] * 3 for k, v in a.items()},
                   {k: [v] * 3 for k, v in b.items()})
    A.true(r["ci_excludes_zero"], "完全分离时应判显著")
    A.true(r["diff"] > 0.9, f"差值应接近 1，实际 {r['diff']}")

    # 完全相同 → 不显著
    r2 = PS.compare("A", "A2", {k: [v] * 3 for k, v in a.items()},
                    {k: [v] * 3 for k, v in a.items()})
    A.false(r2["ci_excludes_zero"], "两组完全相同时不应判显著")
    A.eq(r2["diff"], 0.0)

# ---------------------------------------------------------------------------
# 轨迹级 KL 漂移的汇总逻辑（不需要 GPU）
# ---------------------------------------------------------------------------
def _load_kl_module():
    """按文件路径加载 ``measure_kl_drift``。

    ⚠️ 不能 ``import scripts.eval.measure_kl_drift``：``scripts`` 不是包，
    而且那个模块顶层会 import torch/transformers。用文件加载可以只取
    ``summarize`` 这个**纯函数**。
    """
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts/eval/measure_kl_drift.py"
    spec = importlib.util.spec_from_file_location("_kl_drift_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rec(seq: float, tok: float) -> dict:
    return {"task_id": "t", "category": "c", "sample_index": 0,
            "n_tokens": 10, "seq_delta": seq, "token_delta": tok}


@test
def test_kl_summarize_reports_both_levels() -> None:
    """序列级与 token 级必须**分别**报告。

    这是本脚本存在的理由：``logps/chosen``（token 级）已经被实测证明
    会漏报（β=0.5 时它稳定在 −4.55，而准确率掉了 33.75pp）。
    只报 token 级就等于继续用那个失效的判据。
    """
    m = _load_kl_module()
    # 构造「token 级几乎不变、序列级明显漂移」的情形
    recs = [_rec(seq=s, tok=s / 100.0) for s in (0.0, 0.5, 1.0, 8.0)]
    r = m.summarize(recs, policy_dir="p", reference_dir="q", task_set="v3")
    A.true(abs(r["token_level"]["mean_delta"] - 0.02375) < 1e-9,
           f"token 级均值应约 0.024，实际 {r['token_level']['mean_delta']}")
    A.true(r["sequence_level"]["mean_delta"] > 2.0,
           "序列级均值应明显更大（这正是 token 级漏报的表现）")


@test
def test_kl_summarize_reports_spread_not_just_mean() -> None:
    """必须给出 std 与 median —— 只看均值会漏掉「漂移不均匀」。

    实测失效模式：均值稳定而部分类别崩掉。std 是发现它的第一手线索。
    """
    m = _load_kl_module()
    recs = [_rec(0.0, 0.0), _rec(0.0, 0.0), _rec(0.0, 0.0), _rec(20.0, 0.0)]
    r = m.summarize(recs, policy_dir="p", reference_dir="q", task_set="v3")
    A.true(r["sequence_level"]["std_delta"] > 5.0,
           f"极端不均匀时 std 应很大，实际 {r['sequence_level']['std_delta']}")
    A.eq(r["sequence_level"]["median_delta"], 0.0, "中位数应为 0")
    A.true(r["sequence_level"]["max_delta"] == 20.0, "应报告最大值")


@test
def test_kl_summarize_handles_empty_input() -> None:
    """空输入不得除零崩溃。"""
    m = _load_kl_module()
    r = m.summarize([], policy_dir="p", reference_dir="q", task_set="v3")
    A.eq(r["num_trajectories"], 0)
    A.eq(r["sequence_level"]["mean_delta"], 0.0)
    A.eq(r["thresholds"]["frac_seq_delta_gt_0.5"], 0.0)


@test
def test_kl_threshold_fractions_are_monotone() -> None:
    """阈值越高、超过它的比例应越低（单调）。"""
    m = _load_kl_module()
    recs = [_rec(s, 0.0) for s in (0.1, 0.6, 1.2, 2.5, 5.0)]
    r = m.summarize(recs, policy_dir="p", reference_dir="q", task_set="v3")
    th = r["thresholds"]
    A.true(th["frac_seq_delta_gt_0.5"] >= th["frac_seq_delta_gt_1.0"]
           >= th["frac_seq_delta_gt_2.0"], f"阈值比例应单调递减，实际 {th}")
    A.true(abs(th["frac_seq_delta_gt_1.0"] - 0.6) < 1e-9,
           f"Δ>1.0 应有 3/5，实际 {th['frac_seq_delta_gt_1.0']}")

# ---------------------------------------------------------------------------
# C1 剂量-反应的数据组装（不需要 GPU）
# ---------------------------------------------------------------------------
def _load_dose_module():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts/train/dpo_dose_response.py"
    spec = importlib.util.spec_from_file_location("_dose_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@test
def test_dose_variants_share_the_same_total() -> None:
    """三个剂量变体的**总数必须相同** —— 否则梯度步数成为混淆变量。

    若 0% 变体有 2000 对而 100% 变体只有 300 对，观测到的差异可能来自
    「训练步数不同」，而不是「clarify 剂量不同」。剂量实验的全部意义
    就在于**只改一个变量**。
    """
    m = _load_dose_module()
    c = [{"pair_type": m.CLARIFY, "i": i} for i in range(40)]
    o = [{"pair_type": "format_constraint", "i": i} for i in range(500)]
    for dose in m.DOSES:
        v, st = m.build_variant(c, o, dose=dose, total=100, seed=3)
        A.eq(st["n_total"], 100, f"dose={dose} 时总数应为 100，实际 {st['n_total']}")
        A.eq(len(v), 100)


@test
def test_dose_variants_have_the_intended_clarify_fraction() -> None:
    """clarify 占比必须等于剂量 —— 这是唯一的自变量。"""
    m = _load_dose_module()
    c = [{"pair_type": m.CLARIFY, "i": i} for i in range(40)]
    o = [{"pair_type": "format_constraint", "i": i} for i in range(500)]
    for dose, expect in ((0.0, 0), (0.3, 30), (1.0, 100)):
        _, st = m.build_variant(c, o, dose=dose, total=100, seed=3)
        A.eq(st["n_clarify"], expect,
             f"dose={dose} 时 clarify 应 {expect} 个，实际 {st['n_clarify']}")


@test
def test_dose_marks_resampling_when_pool_is_too_small() -> None:
    """池子不够时要**标记**有放回重采样，而不是静默重复。

    实测：``dpo_joint`` 里 clarify 只有 304 对，而 100% 剂量需要 1000 对，
    必须有放回重采样。报告里必须写明这一点（同一数据被多次见到）。
    """
    m = _load_dose_module()
    c = [{"pair_type": m.CLARIFY, "i": i} for i in range(5)]
    o = [{"pair_type": "format_constraint", "i": i} for i in range(500)]
    _, st = m.build_variant(c, o, dose=1.0, total=100, seed=3)
    A.true(st["clarify_resampled"], "池子 5 个要凑 100 个，必须标记为重采样")
    A.eq(st["clarify_pool"], 5)

    # 池子够时不应标记
    c2 = [{"pair_type": m.CLARIFY, "i": i} for i in range(500)]
    _, st2 = m.build_variant(c2, o, dose=0.3, total=100, seed=3)
    A.false(st2["clarify_resampled"], "池子充足时不应标记重采样")


@test
def test_dose_zero_has_no_clarify_pairs() -> None:
    """0% 剂量必须**一条 clarify 都没有** —— 它是基线臂。"""
    m = _load_dose_module()
    c = [{"pair_type": m.CLARIFY, "i": i} for i in range(50)]
    o = [{"pair_type": "format_constraint", "i": i} for i in range(500)]
    v, st = m.build_variant(c, o, dose=0.0, total=100, seed=9)
    A.eq(st["n_clarify"], 0)
    A.true(all(p["pair_type"] != m.CLARIFY for p in v),
           "0% 变体里不得出现 clarify 对")


@test
def test_dose_split_by_type_is_a_partition() -> None:
    """``split_by_type`` 必须是一个划分（不重不漏）。"""
    m = _load_dose_module()
    ps = ([{"pair_type": m.CLARIFY}] * 3
          + [{"pair_type": "format_constraint"}] * 4
          + [{"pair_type": "error_recovery"}] * 5)
    c, o = m.split_by_type(ps)
    A.eq(len(c), 3)
    A.eq(len(o), 9)
    A.eq(len(c) + len(o), len(ps), "两部分之和应等于原集合")


# ---------------------------------------------------------------------------
# C1 判定：剂量曲线
# ---------------------------------------------------------------------------
def _load_curve_module():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts/eval/dose_response_curve.py"
    spec = importlib.util.spec_from_file_location("_curve_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@test
def test_monotonic_detects_direction() -> None:
    """单调判定必须区分方向，且允许相等（非严格）。"""
    c = _load_curve_module()
    A.true(c.monotonic([0, 1, 2], direction="up"))
    A.true(c.monotonic([2, 1, 0], direction="down"))
    A.false(c.monotonic([0, 2, 1], direction="up"))
    A.false(c.monotonic([1, 2, 0], direction="down"))
    A.true(c.monotonic([1, 1, 1], direction="up"))
    A.true(c.monotonic([1, 1, 0], direction="down"))
    A.false(c.strictly_monotonic([1, 1, 1], direction="up"))
    A.true(c.strictly_monotonic([0, 1, 2], direction="up"))


@test
def test_dose_verdict_mechanism_holds() -> None:
    """两个指标都单调 → 判「机制成立」。"""
    c = _load_curve_module()
    per = [{"accuracy": 0.83, "clarify_instead": 20},
           {"accuracy": 0.70, "clarify_instead": 35},
           {"accuracy": 0.49, "clarify_instead": 54}]
    r = c.analyze([0.0, 0.3, 1.0], per)
    A.true(r["accuracy_monotone_down"], "accuracy 应判为单调下降")
    A.true(r["clarify_instead_monotone_up"], "clarify_instead 应判为单调上升")
    A.eq(r["verdict"], "机制成立")


@test
def test_dose_verdict_mechanism_fails_honestly() -> None:
    """都不单调 → 必须判「机制不成立」。

    这条保证脚本不会为了迎合预期而在非单调数据上
    判「成立」。我预期机制成立，但脚本必须能说不。
    """
    c = _load_curve_module()
    per = [{"accuracy": 0.50, "clarify_instead": 10},
           {"accuracy": 0.52, "clarify_instead": 9},   # 反向了
           {"accuracy": 0.49, "clarify_instead": 12}]
    r = c.analyze([0.0, 0.3, 1.0], per)
    A.false(r["accuracy_monotone_down"])
    A.false(r["clarify_instead_monotone_up"])
    A.eq(r["verdict"], "机制不成立")


@test
def test_dose_verdict_partial_when_only_one_metric_moves() -> None:
    """只有一个指标单调 → 判「部分成立」，不能升级成「成立」。

    ⚠️ 构造时要注意：第二个指标**不能是平的** —— 平线在非严格单调
    定义下也算「单调」（我第一次就写成了平的，测试因此失败，
    而失败原因是我的构造不对，不是代码不对）。
    要让它**反向动**，才真正不是单调。
    """
    c = _load_curve_module()
    per = [{"accuracy": 0.80, "clarify_instead": 10},
           {"accuracy": 0.70, "clarify_instead": 12},   # 升
           {"accuracy": 0.60, "clarify_instead": 11}]   # 又降 → 非单调
    r = c.analyze([0.0, 0.5, 1.0], per)
    A.true(r["accuracy_monotone_down"], "accuracy 应单调下降")
    A.false(r["clarify_instead_monotone_up"], "clarify_instead 非单调")
    A.eq(r["verdict"], "部分成立")


@test
def test_dose_verdict_flat_data_is_not_a_mechanism() -> None:
    """完全没变化的数据不得被判成「机制成立」。

    非严格单调允许相等，于是三个点同一个值也算「单调」。
    若只依赖它，「什么都没发生」会被当成「机制被证实」。
    所以必须同时给出严格判据，并让结论体现它。
    """
    c = _load_curve_module()
    per = [{"accuracy": 0.5, "clarify_instead": 10}] * 3
    r = c.analyze([0.0, 0.5, 1.0], per)
    A.true(r["accuracy_monotone_down"], "非严格判定会认为它单调")
    A.false(r["accuracy_strictly_down"], "严格判定必须说它没动")
    A.false(r["clarify_instead_strictly_up"])
    A.true("0" in r["verdict"],
           f"完全不变的数据结论里应体现「变化幅度为 0」，实际 {r['verdict']}")


@test
def test_dose_extract_metrics_reads_the_real_report_shape() -> None:
    """必须从**实际报告结构**里读到值 —— 字段位置错了会静默全 0。

    实测踩过：accuracy 在 ``headline.accuracy``（不是 ``overall``），
    错误子类型在**另一份** ``error_analysis.json`` 的
    ``distribution.subtype_counts``（不是 ``eval_report`` 的
    ``error_distribution``）。

    当时脚本不报错，只是所有计数都是 0 —— 那会让剂量曲线看起来
    「三条线全平」，把「字段找错了」误读成「剂量没有效应」。
    **这是会直接误导结论的错误**，所以要用真实结构测。
    """
    c = _load_curve_module()
    real_report = {
        "label": "dose000",
        "headline": {"accuracy": 0.679, "pass_at_k": 0.679},
        "metadata": {"is_mock": True},
        "per_task": [{"task_id": "t1"}],
    }
    real_analysis = {
        "distribution": {
            "counts": {"SUCCESS": 72, "FORMAT_ERROR": 8, "HALLUCINATION": 4,
                       "PARAM_ERROR": 14, "WRONG_TOOL": 8},
            "subtype_counts": {"SUCCESS": 72, "clarify_instead": 12,
                               "multi_tool": 3, "wrong_value": 9},
        }
    }
    m = c.extract_metrics(real_report, real_analysis)
    A.true(abs(m["accuracy"] - 0.679) < 1e-9,
           f"accuracy 应从 headline 读取，实际 {m['accuracy']}")
    A.eq(m["clarify_instead"], 12, "clarify_instead 应从 subtype_counts 读取")
    A.eq(m["multi_tool"], 3)
    A.eq(m["wrong_value"], 9)
    A.eq(m["format_error"], 8, "FORMAT_ERROR 在 counts 里")


@test
def test_dose_extract_metrics_reads_zero_without_error_analysis() -> None:
    """只给 eval_report（不给 error_analysis）时子类型应为 0 —— 不是崩溃。

    这种情形本身是**可疑的**（调用方应打印警告），但函数不能崩。
    """
    c = _load_curve_module()
    m = c.extract_metrics({"headline": {"accuracy": 0.5}})
    A.eq(m["accuracy"], 0.5)
    A.eq(m["clarify_instead"], 0)


@test
def test_dose_extract_metrics_tolerates_report_shape_changes() -> None:
    """旧结构的报告仍要能读（向后兼容）。"""
    c = _load_curve_module()
    A.eq(c.extract_metrics({"overall": {"accuracy": 0.5}})["accuracy"], 0.5)
    A.eq(c.extract_metrics({"accuracy": 0.7})["accuracy"], 0.7)
    A.eq(c.extract_metrics({})["accuracy"], 0.0)
    A.eq(c.extract_metrics({})["clarify_instead"], 0)
    # 子类型直接放在 distribution 下的旧写法
    A.eq(c.extract_metrics({}, {"distribution": {"clarify_instead": 4}})["clarify_instead"], 4)

# ---------------------------------------------------------------------------
# 簇级统计（同一「期望调用」的多个任务不是独立证据）
# ---------------------------------------------------------------------------
@test
def test_expected_keys_reveals_duplicate_expected_calls() -> None:
    """必须能查出「多个任务的正确答案完全相同」。

    实测 v3：106 个任务只有 **92** 个不同期望调用 ——
    其中一个簇是 **15 个 clarify 任务**共享的（它们的期望调用都是空的，
    因为正确行为都是「不调用工具」）。

    **它们不是 15 个独立证据。**
    """
    k = PS.expected_keys("v3")
    A.true(len(k) >= 100, "应覆盖全部任务")
    uniq = set(k.values())
    A.true(len(uniq) < len(k),
           f"应存在重复的期望调用（{len(k)} 任务 → {len(uniq)} 个不同）")


@test
def test_cluster_scores_average_within_cluster() -> None:
    """簇内必须取**平均**，不是求和也不是取最大。

    我们要的是「这个**行为**做对了多少」，而不是
    「这一族任务总共贡献多少分」—— 求和会把权重放大 15 倍。
    """
    s = {"t1": 1.0, "t2": 0.0, "t3": 1.0}
    c = {"t1": "A", "t2": "A", "t3": "B"}
    A.eq(PS.cluster_scores(s, c), {"A": 0.5, "B": 1.0})
    # 未在 cluster_of 里的任务被丢弃（而不是算作 0）
    A.eq(PS.cluster_scores({"t9": 1.0}, c), {})


@test
def test_clustered_bootstrap_uses_fewer_samples_than_task_level() -> None:
    """簇级有效样本数必须**不高于**任务数，且差值等于重复数。"""
    a = {f"t{i}": 1.0 for i in range(30)}
    b = {f"t{i}": 0.0 for i in range(30)}
    c = {f"t{i}": "X" for i in range(15)}
    c.update({f"t{i}": f"Y{i}" for i in range(15, 30)})
    _, _, _, n = PS.paired_bootstrap_clustered(a, b, c, n_boot=200, seed=1)
    A.eq(n, 16, "15 个共享簇 X + 15 个独立簇 = 16 个有效样本")


@test
def test_clustered_ci_is_wider_when_tasks_are_clustered() -> None:
    """把重复任务归簇后 CI 应更宽（或至少不更窄）—— 这是「更保守」的体现。

    若按任务重采样，15 个重复任务会被当成 15 份独立证据，
    CI 偏窄、显得「更显著」。归簇之后有效样本变少，CI 应变宽。
    """
    # 15 个任务共享同一个簇，另外 15 个各自独立
    a = {f"t{i}": (1.0 if i < 20 else 0.0) for i in range(30)}
    b = {f"t{i}": 0.0 for i in range(30)}
    c = {f"t{i}": "SHARED" for i in range(15)}
    c.update({f"t{i}": f"S{i}" for i in range(15, 30)})

    _, tlo, thi = PS.paired_bootstrap(a, b, n_boot=800, seed=5)
    _, clo, chi, n = PS.paired_bootstrap_clustered(a, b, c, n_boot=800, seed=5)
    A.true(n < 30, "有效样本应少于 30")
    A.true((chi - clo) >= (thi - tlo) - 0.02,
           f"簇级 CI 宽度 {chi-clo:.3f} 不应明显窄于任务级 {thi-tlo:.3f}")


@test
def test_compare_reports_both_scopes_and_flags_disagreement() -> None:
    """``compare`` 必须同时给出两套口径，且能发现结论不一致。"""
    a = {f"t{i}": [1.0, 1.0] for i in range(40)}
    b = {f"t{i}": [0.0, 0.0] for i in range(40)}
    c = {f"t{i}": ("S" if i < 20 else f"U{i}") for i in range(40)}
    r = PS.compare("A", "B", a, b, cluster_of=c)
    A.true("clustered" in r, "应含簇级结果")
    A.eq(r["n_common"], 40)
    A.eq(r["clustered"]["n_clusters"], 21, "20 个共享 + 20 个独立 = 21 簇")
    A.true(r["clustered"]["diff"] > 0.5)
    # 不给 cluster_of 时不应有 clustered（向后兼容）
    A.false("clustered" in PS.compare("A", "B", a, b))


@test
def test_clustered_mcnemar_does_not_count_duplicates_as_evidence() -> None:
    """簇级 McNemar 不应把同一簇的 15 个任务算成 15 个不一致对。"""
    a = {f"t{i}": 1.0 for i in range(15)}
    b = {f"t{i}": 0.0 for i in range(15)}
    c = {f"t{i}": "SAME" for i in range(15)}
    r = PS.mcnemar_clustered(a, b, c)
    A.eq(r["n_discordant"], 1,
         f"15 个同簇任务只应产生 1 个不一致对，实际 {r['n_discordant']}")

@test
def test_cluster_ci_is_wider_than_task_ci() -> None:
    """单组的簇级 CI 必须**不窄于**任务级 CI —— 这是「更保守」的可验证定义。

    ``EvalReport`` 的主指标按**任务** bootstrap。当多个任务的正确答案
    完全相同时（实测：15 个 clarify 任务共享同一个「不调用工具」的期望），
    按任务重采样会把一个行为重复计数，区间偏窄。
    簇级先取簇内平均再重采样，区间应变宽。
    """
    # 构造：14 个任务共享一簇，另 20 个各自独立
    scores = {f"t{i}": (1.0 if i < 17 else 0.0) for i in range(34)}
    cluster_of = {f"t{i}": "SHARED" for i in range(14)}
    cluster_of.update({f"t{i}": f"U{i}" for i in range(14, 34)})

    r = PS.cluster_ci(scores, cluster_of, n_boot=800, seed=3)
    A.eq(r["n_clusters"], 21, "14 个共享 + 20 个独立 = 21 簇")
    A.eq(r["n_tasks"], 34)
    A.eq(r["cluster_shrinkage"], 13)

    # 任务级 CI 宽度（用同族函数算，保证可比）
    _, tlo, thi = PS.paired_bootstrap(scores, {}, n_boot=800, seed=3)
    A.true((r["ci_hi"] - r["ci_lo"]) >= (thi - tlo) - 0.02,
           f"簇级 CI 宽 {r['ci_hi']-r['ci_lo']:.3f} 不应明显窄于"
           f"任务级 {thi-tlo:.3f}")


@test
def test_cluster_ci_handles_empty_and_single() -> None:
    """空输入与单簇不得除零崩溃。"""
    A.eq(PS.cluster_ci({}, {})["n_clusters"], 0)
    A.eq(PS.cluster_ci({}, {})["mean"], 0.0)
    r = PS.cluster_ci({"t1": 1.0}, {"t1": "A"}, n_boot=100, seed=1)
    A.eq(r["n_clusters"], 1)
    A.eq(r["mean"], 1.0)


@test
def test_cluster_ci_mean_matches_manual_computation() -> None:
    """簇级均值必须等于「簇内平均后再对簇取平均」—— 不是对所有任务取平均。

    这两者在簇大小不等时**不一样**，而后者会把大簇的权重放大。
    """
    # 簇 A 有 3 个任务（2 对 1 错 → 0.667），簇 B 有 1 个任务（全错 → 0）
    scores = {"a1": 1.0, "a2": 1.0, "a3": 0.0, "b1": 0.0}
    cluster_of = {"a1": "A", "a2": "A", "a3": "A", "b1": "B"}
    r = PS.cluster_ci(scores, cluster_of, n_boot=100, seed=1)
    # 簇 A 均值 2/3，簇 B 均值 0 → 簇级均值 = (2/3 + 0)/2 = 1/3
    A.true(abs(r["mean"] - 1 / 3) < 1e-9,
           f"簇级均值应为 1/3，实际 {r['mean']}")
    # 而按任务平均是 (1+1+0+0)/4 = 0.5 —— 两者不同，这正是要区分的
    A.true(abs(0.5 - r["mean"]) > 0.1, "簇级均值不应等于按任务平均")

# ---------------------------------------------------------------------------
# 格式正确率的「按任务」口径（修一个会误导结论的测量 bug）
# ---------------------------------------------------------------------------
@test
def test_format_accuracy_old_metric_misjudges_clarify_tasks() -> None:
    """**复现那个测量 bug**：旧口径把「应先追问」的正确答案判成格式错误。

    实测：v3 基线里 `cancellation` 任务成功率 **62.5%** 而格式正确率
    **0.0%** —— 那个 0.0% 是测量产物，因为该类别 22 个任务里
    **14 个是「应先追问」**，它们的正确输出是自然语言追问、
    本来就没有 `tool_call`。

    **若不分开报，会把它误读成「模型连格式都不会」。**
    """
    from src.eval.tool_call_parser import format_accuracy

    clarify_answer = "请问您取消这个订单的原因是什么？"
    # 旧口径：追问文本被判为格式错误
    A.eq(format_accuracy([clarify_answer]), 0.0,
         "旧口径确实会把追问判为格式错误（这正是 bug）")


@test
def test_format_accuracy_by_task_handles_clarify_correctly() -> None:
    """按任务口径必须把追问类任务的正确答案判为格式**正确**。"""
    from src.eval.tool_call_parser import (
        expected_format_ok,
        format_accuracy_by_task,
    )

    # 追问类：自然语言回复 = 格式正确；空输出 = 格式错误
    A.true(expected_format_ok(is_clarify_task=True, text="请问订单号是多少？"))
    A.false(expected_format_ok(is_clarify_task=True, text=""))
    A.false(expected_format_ok(is_clarify_task=True, text="   "))

    # 要调工具的类：必须有 canonical tool_call
    good = '<tool_call>{"name": "f", "arguments": {}}</tool_call>'
    A.true(expected_format_ok(is_clarify_task=False, text=good))
    A.false(expected_format_ok(is_clarify_task=False, text="我帮您查一下"))

    # 混合批次
    A.eq(format_accuracy_by_task([(True, "请问？"), (False, good)]), 1.0)
    A.eq(format_accuracy_by_task([(True, ""), (False, good)]), 0.5)
    A.eq(format_accuracy_by_task([]), 0.0)


@test
def test_format_accuracy_by_task_is_higher_for_clarify_heavy_categories() -> None:
    """在「追问类占多数」的类别上，按任务口径必须**明显高于**旧口径。

    这条是那个 bug 的**量化复现**：cancellation 里 14/22 是追问类，
    所以两个口径应该差很多。
    """
    from src.eval.tool_call_parser import (
        format_accuracy,
        format_accuracy_by_task,
    )

    # 模拟 cancellation：14 个追问（正确输出是文本）+ 8 个要调工具（输出合法调用）
    good = '<tool_call>{"name": "cancel_booking",'
    good += '"arguments": {"booking_id": "BK1", "reason": "行程变更"}}</tool_call>'
    pairs = [(True, "请问您取消的原因是什么？")] * 14 + [(False, good)] * 8
    texts = [tx for _, tx in pairs]

    old = format_accuracy(texts)
    new = format_accuracy_by_task(pairs)
    A.true(abs(old - 8 / 22) < 1e-9, f"旧口径应约 {8/22:.3f}，实际 {old:.3f}")
    A.eq(new, 1.0, "按任务口径应为 100%（两类都格式正确）")
    A.true(new - old > 0.5, f"两个口径应差很多，实际差 {new-old:.3f}")

# ---------------------------------------------------------------------------
# 分类别对照（回答「补了语料之后那三个 0% 修好了吗」）
# ---------------------------------------------------------------------------
def _load_compare_module():
    import importlib.util

    path = _REPO / "scripts/eval/compare_by_category.py"
    spec = importlib.util.spec_from_file_location("_cmp_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@test
def test_compare_by_category_uses_task_aware_format() -> None:
    """对照表里的格式正确率必须是**按任务口径**。

    旧口径对「应先追问」任务（正确输出是自然语言追问）会判为格式错误 ——
    实测 `cancellation` 因此显示 0.0%。若对照表用旧口径，
    **会让人以为那一类连格式都不会**，而它其实做得不错。
    """
    import importlib.util

    path = _REPO / "scripts/eval/compare_by_category.py"
    src = path.read_text(encoding="utf-8")
    A.true("expected_format_ok" in src,
           "对照表必须用按任务口径的格式判定")


@test
def test_compare_by_category_render_includes_all_labels_and_delta() -> None:
    """渲染必须含每个实验的列，以及「变化」列。"""
    m = _load_compare_module()
    all_cats = {
        "booking_flow": {
            "a": {"n": 10, "acc": 0.0, "fmt": 0.0},
            "b": {"n": 10, "acc": 0.5, "fmt": 0.9},
        },
        "weather_advisory": {
            "a": {"n": 5, "acc": 0.9, "fmt": 0.9},
            "b": {"n": 5, "acc": 0.9, "fmt": 1.0},
        },
    }
    md = m.render(all_cats, ["a", "b"])
    A.true("booking_flow" in md)
    A.true("weather_advisory" in md)
    A.true("变化" in md, "应有「变化」列")
    A.true("+50.0%" in md, f"booking_flow 的变化应为 +50.0%，实际：{md}")
    A.true("按任务口径" in md, "应说明格式用的口径")
    # 总体段
    A.true("### 总体" in md)

# ---------------------------------------------------------------------------
# 失败归因（我在这里错了两次，必须锁住）
# ---------------------------------------------------------------------------
def _load_attr_module():
    import importlib.util

    path = _REPO / "scripts/eval/attribute_failures.py"
    spec = importlib.util.spec_from_file_location("_attr_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@test
def test_attribution_counts_only_failed_samples() -> None:
    """归因**只看 passed=False** 的样本。

    ## 我错过的第一件事

    `clarify_instead` 是「该轮期望不调用工具，模型也没调用」的标记 ——
    在「应先追问」任务上它是**正确答案**，判 ``SUCCESS``。

    我把它读成「过度追问」，于是以为有 62 条过度追问（占失败 70%），
    实际只有 **2 条**。**用错误类别表做归因会把「做对了」算成「做错了」。**
    """
    m = _load_attr_module()

    class _E:
        tool = "f"
        args = {}

    class _T:
        expected = _E()
        category = "c"

        class turns:  # noqa: N801
            pass

    _T.turns = [type("Turn", (), {"content": "u"})()]
    tasks = {"t1": _T()}

    rep = {
        "label": "x",
        "per_task": [
            {"task_id": "t1", "outcomes": [True, False],
             "error_categories": ["SUCCESS", "PARAM_ERROR"],
             "predictions": [
                 {"classification": {"detail": {"subtype": "clarify_instead"}}},
                 {"classification": {"detail": {"subtype": "wrong_value"}}},
             ]},
        ],
    }
    r = m.attribute(rep, tasks)
    A.eq(r["n_samples"], 2, "总样本应数全部")
    A.eq(r["n_fail"], 1, "失败只应数 passed=False 的")
    A.eq(r["by_cat"], {"PARAM_ERROR": 1}, "正确样本不得进入归因")
    A.eq(r["by_subtype"], {"wrong_value": 1})


@test
def test_attribution_separates_two_opposite_decision_errors() -> None:
    """必须分开统计两类**方向相反**的决策错误。

    * **过度追问**：信息够，模型却停下追问 → `clarify_instead`
    * **该追问却调工具** → `should_clarify`（类别 WRONG_TOOL）

    混在一起算会得出「有 62 条过度追问」这种完全错误的结论。
    """
    m = _load_attr_module()

    class _E1:  # 要调工具
        tool = "search_flights"
        args = {}

    class _E2:  # 应追问
        tool = None
        args = {}

    class _T:
        def __init__(self, tool):
            self.expected = type("E", (), {"tool": tool, "args": {}})()
            self.category = "c"
            self.turns = [type("Turn", (), {"content": "u"})()]

    tasks = {"tool_task": _T("search_flights"), "clar_task": _T(None)}
    rep = {
        "label": "x",
        "per_task": [
            # 要调工具却追问 → 过度追问
            {"task_id": "tool_task", "outcomes": [False],
             "error_categories": ["WRONG_TOOL"],
             "predictions": [{"classification":
                              {"detail": {"subtype": "clarify_instead"}}}]},
            # 应追问却调工具 → 该追问却调工具
            {"task_id": "clar_task", "outcomes": [False],
             "error_categories": ["WRONG_TOOL"],
             "predictions": [{"classification":
                              {"detail": {"subtype": "should_clarify"}}}]},
        ],
    }
    r = m.attribute(rep, tasks)
    A.eq(len(r["over_clarify"]), 1, "过度追问应有 1 条")
    A.eq(len(r["under_clarify"]), 1, "该追问却调工具应有 1 条")
    A.eq(r["over_clarify"][0]["task_id"], "tool_task")
    A.eq(r["under_clarify"][0]["task_id"], "clar_task")


@test
def test_attribution_reads_error_categories_field_not_categories() -> None:
    """必须读 ``error_categories`` 字段 —— 猜错字段会让结果**全变 "?"**。

    实测踩过：我写成了 ``r.get("categories")``，于是所有错误的 error
    都是 ``"?"``，两类决策错误都统计为 **0** ——
    **看起来像「没有问题」，实际是脚本坏了。**
    """
    import importlib.util  # noqa: F401

    src = (_REPO / "scripts/eval/attribute_failures.py").read_text(encoding="utf-8")
    A.true("error_categories" in src, "必须读 error_categories 字段")
    A.true("predictions" in src and "subtype" in src,
           "子类型必须从 predictions[i].classification.detail.subtype 取")
