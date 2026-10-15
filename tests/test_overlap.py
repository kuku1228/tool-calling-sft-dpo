"""评测集与训练语料**重合层级**的审计测试。

## 为什么单独一个测试模块

我原先只检查「prompt 是否逐字相同」，得出「0 重合、干净」的结论 ——
**那只覆盖了一层**。实测把层级拆开后发现另外几层都有问题：

| 层级 | 判据 | v3 实测 |
|---|---|---|
| 1 | prompt 逐字相同 | 0 ✅ |
| 2 | 完整期望相同（工具+参数名+参数值） | **20** ⚠️ |
| 3 | 同工具 + 同参数名 | 85 |
| 4 | 同句式、实体不同 | 0 |
| 5 | 全新 | 1 |

所以「数据划分干净」这个结论**必须分层说**，不能一句话带过。
这个模块把分层判据锁住。
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tests._harness import A, test  # noqa: E402


def _load_audit_module():
    import importlib.util

    path = _REPO / "scripts/eval/audit_eval_overlap.py"
    spec = importlib.util.spec_from_file_location("_overlap_audit", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeTask:
    def __init__(self, prompt: str, tool, args: dict):
        self.turns = [type("Turn", (), {"content": prompt})()]
        self.expected = type("E", (), {"tool": tool, "args": args})()
        self.task_id = "fake"
        self.category = "fake"
        self.visibility = "seen"


@test
def test_expected_key_uses_values_not_just_names() -> None:
    """层级 2 的判据必须**含参数值** —— 否则它退化成层级 3。

    早期我只有「同工具同参数名」这一个判据（层级 3），它无法区分
    「训练里有 get_weather(city=三亚)」与「训练里只有 get_weather(city=别的)」。
    前者是可能靠记忆答对的泄漏，后者只是学会了用法。
    """
    m = _load_audit_module()
    a = _FakeTask("p", "get_weather", {"city": "三亚"})
    b = _FakeTask("q", "get_weather", {"city": "三亚"})
    c = _FakeTask("r", "get_weather", {"city": "上海"})
    A.eq(m.expected_key(a), m.expected_key(b), "参数值相同应判为同层")
    A.true(m.expected_key(a) != m.expected_key(c),
           "参数值不同必须区分开（否则漏报泄漏）")
    # 但签名相同 —— 层级 3 分不开它们
    A.eq(m.signature_key(a), m.signature_key(c),
         "签名只含参数名，两者应相同")


@test
def test_overlap_level_orders_the_five_tiers() -> None:
    """五层判据必须严格递进（数字越小重合越强）。"""
    m = _load_audit_module()
    syn_prompts = {"P1"}
    syn_expected = {("t", (("a", "1"),))}
    syn_sig = {("t", ("a",))}
    syn_skel = {"SKEL"}

    def skel_fn(x: str) -> str:
        return "SKEL" if x == "P4" else "OTHER"

    kw = dict(syn_prompts=syn_prompts, syn_expected=syn_expected,
              syn_sig=syn_sig, syn_skel=syn_skel, skeleton_fn=skel_fn)

    cases = [
        (_FakeTask("P1", "z", {"q": 9}), 1, "prompt 相同最优先"),
        (_FakeTask("P2", "t", {"a": 1}), 2, "期望相同"),
        (_FakeTask("P3", "t", {"a": 2}), 3, "签名相同（值不同）"),
        (_FakeTask("P4", "w", {"z": 9}), 4, "只有句式相同"),
        (_FakeTask("P5", "w", {"z": 9}), 5, "全不同"),
    ]
    for task, expect, why in cases:
        A.eq(m.overlap_level(task, **kw), expect, why)


@test
def test_audit_reports_zero_prompt_overlap_for_v3() -> None:
    """v3 的层级 1 必须是 0 —— 这是「没有被直接污染」的证据。"""
    m = _load_audit_module()
    rep = m.audit("v3")
    A.eq(rep["by_level"]["1"], 0, "v3 不应有 prompt 逐字相同的任务")
    A.true(rep["n_tasks"] >= 100)


@test
def test_audit_separates_entity_leaks_from_clarify_duplicates() -> None:
    """层级 2 必须**分成两类** —— 它们的处置完全不同。

    * **实体泄漏**（非 clarify）：可能靠记忆答对，应**从指标里剔除**
    * **clarify 必然重合**：期望调用是空的，所以「完整期望相同」是
      **设计使然**，不是泄漏 —— 剔除它们会误删 15 个任务

    把两类混在一起会导致两个错误方向之一：要么漏掉真泄漏，
    要么排除掉大量合法任务。
    """
    m = _load_audit_module()
    rep = m.audit("v3")
    lv2_total = rep["by_level"]["2"]
    el, cd = rep["entity_leaks"], rep["clarify_duplicates"]
    A.eq(len(el) + len(cd), lv2_total,
         "层级 2 应被完整分成实体泄漏与 clarify 两类")
    A.true(all(not r["is_clarify"] for r in el),
           "实体泄漏里不应有 clarify 任务")
    A.true(all(r["is_clarify"] for r in cd),
           "clarify 重复里应全是 clarify 任务")


@test
def test_audit_clean_set_excludes_only_entity_leaks() -> None:
    """干净子集 = 剔除实体泄漏，**保留** clarify 任务。"""
    m = _load_audit_module()
    rep = m.audit("v3")
    A.eq(len(rep["clean_ids"]), rep["n_tasks"] - len(rep["entity_leaks"]),
         "干净子集大小应等于总数减去实体泄漏数")
    # clarify 任务必须在干净子集里
    cd_ids = {r["task_id"] for r in rep["clarify_duplicates"]}
    A.true(cd_ids <= set(rep["clean_ids"]),
           "clarify 任务不应被剔除（它们的重合是设计使然）")


@test
def test_find_expected_leaks_matches_the_audit() -> None:
    """两个独立实现的检测结果必须一致。

    ``find_expected_leaks``（src 里，供评测排除用）与
    ``audit_eval_overlap``（脚本里，供报告用）是**两处实现**。
    两处不一致会让「评测排除了 A、报告声称排除了 B」。
    """
    from src.data.recovery_corpus import find_expected_leaks

    m = _load_audit_module()
    rep = m.audit("v3")
    A.eq(sorted(find_expected_leaks(task_set="v3")),
         sorted(rep["contaminated_ids"]),
         "src 与脚本两处检测的实体泄漏应完全一致")


@test
def test_find_expected_leaks_excludes_clarify_tasks() -> None:
    """``find_expected_leaks`` 必须排除 ``tool is None`` 的任务。

    那类任务的期望调用是空的（``{}``），与合成语料里的 clarify 任务
    **必然相同**。若不排除，v3 会一下子被排除 15 个任务，
    而那 15 个是合法的评测任务。
    """
    from src.data.recovery_corpus import find_expected_leaks

    leaks = find_expected_leaks(task_set="v3")
    A.true(len(leaks) <= 10,
           f"实体泄漏应只有少数，实际 {len(leaks)} 个（可能把 clarify 也算进来了）")
    A.true(all(not x.startswith("taskv2_006") for x in leaks) or len(leaks) <= 10,
           "不应把 cancellation 里的 clarify 任务算作泄漏")

@test
def test_visibility_labels_are_flagged_as_meaningless() -> None:
    """``seen``/``unseen`` 标签必须被检查出「没有含义」。

    ``visibility`` 的赋值**从不读取训练数据**：
    v1 是 ``counter % 3``、v3 是 ``sha256(skeleton) % 3``。
    它假设「合成语料只覆盖评测分布的一部分」。

    而我为了修「33 个机场码模型从未见过」，把合成语料扩到覆盖
    **全部 60 个机场码** —— 于是每个评测任务的签名都在训练里出现过。
    实测 ``unseen`` 组的签名覆盖率是 **38/38 = 100%**。

    **「seen 与 unseen 的差值 = 泛化差距」这个说法因此不成立。**

    这条测试的作用是：一旦有人（包括未来的我）又想用 seen/unseen
    的差值讲泛化，检查器会立刻说「这个标签不可用」。
    """
    from src.data.task_manifest_v2 import (
        report_visibility_warning,
        verify_visibility_is_meaningful,
    )

    rep = verify_visibility_is_meaningful()
    A.true(rep["total"] > 0)
    A.true(rep["unseen"] > 0, "应有 unseen 任务可供检查")
    # 当前配置下必须被判为「无含义」
    A.false(rep["meaningful"],
            "合成语料已覆盖整个评测分布，visibility 不应被判为有含义")
    A.true(rep["unseen_all_covered"],
           f"unseen 组应全部被覆盖，实际 {rep['unseen_signature_covered']}/{rep['unseen']}")
    # 警告文本必须非空且说清原因
    w = report_visibility_warning()
    A.true(len(w) > 0, "标签无含义时必须给出警告")
    A.true("泛化" in w, "警告里必须点明「不要解释为泛化差距」")


@test
def test_audit_effective_independent_sample_note() -> None:
    """15 个 clarify 任务的正确答案是**同一个行为**，不应算 15 个独立样本。

    它们期望调用都是 ``{}``（不调用工具），所以「完整期望相同」
    对它们必然成立。这在审计里必须被单独列出 ——
    否则会把「一个行为被重复计数 15 次」当成 15 个独立证据。
    """
    m = _load_audit_module()
    rep = m.audit("v3")
    dup = rep["duplicate_expected_groups"]
    A.true("{}" in dup, "空期望调用应被列为一个重复组")
    A.true(len(dup["{}"]) >= 10,
           f"clarify 任务应有 10 个以上，实际 {len(dup['{}'])}")
    A.eq(len(dup["{}"]), len(rep["clarify_duplicates"]),
         "空期望组的大小应与 clarify 重复数一致")

@test
def test_no_entity_leaks_remain_after_task_set_fix() -> None:
    """任务集层面修完后，**实体泄漏必须为 0**。

    原先有 5 个（4 个 ``get_weather(city=...)`` + 1 个
    ``search_flights(YNT→ZUH, 2026-12-10)``）。修法是**换评测实体**，
    而不是过滤训练语料 —— 实测过滤会删掉 41 个训练任务、把
    ``check_baggage`` 与 ``get_weather`` 的样本全删光。

    这条测试守的是「不要回退」：任何人（包括我）若把评测城市改回
    训练用过的池子，这里会立刻失败。
    """
    from src.data.recovery_corpus import find_expected_leaks

    leaks = find_expected_leaks(task_set="v3")
    A.eq(leaks, [], f"实体泄漏应已清零，实际还有 {leaks}")


@test
def test_weather_cities_avoid_training_weather_pool() -> None:
    """评测的天气城市**必须避开**训练语料的天气池。

    训练天气池是 ``recovery_corpus._WEATHER_CITIES``（固定 7 城）。
    评测原来用了其中的 4 个（三亚/上海/北京/南京）→ 期望调用完全相同
    → 层级 2 泄漏。

    这类泄漏**只能靠实体池设计避免**：训练与评测都覆盖同一空间时，
    组合必然碰撞（实测 multi_tool 的 10 条航线全部与训练重合，
    只有 1 条连日期也相同）。
    """
    import src.data.recovery_corpus as R
    from src.data.task_set_v2_specs import _WEATHER_CITY

    overlap = set(_WEATHER_CITY) & set(R._WEATHER_CITIES)
    A.eq(overlap, set(),
         f"评测天气城市与训练池重合：{sorted(overlap)} —— 会产生实体泄漏")


@test
def test_multi_tool_dates_avoid_training_route_dates() -> None:
    """``multi_tool`` 的「航线 + 日期」三元组不得与训练完全相同。

    ⚠️ 判据的精确边界（专家指出）：
    * **航线相同、日期不同** = 层级 3 **合理泛化**（该学会规则）
    * **航线 + 日期都相同** = 层级 2 **泄漏**

    实测 multi_tool 的 10 条航线**全部**与训练重合（训练覆盖全部 60 城），
    但只有 1 条的日期也相同 —— 只需修那一条。
    """
    from src.data.recovery_corpus import build_training_tasks
    from src.data.task_manifest_v2 import load_tasks_v3

    syn = build_training_tasks(seed=42)
    train_triples = set()
    for x in syn:
        a = x.expected.args or {}
        if a.get("origin") and a.get("destination"):
            train_triples.add((a["origin"], a["destination"], a.get("date")))

    bad = []
    for x in load_tasks_v3():
        if x.category != "multi_tool":
            continue
        a = x.expected.args or {}
        triple = (a.get("origin"), a.get("destination"), a.get("date"))
        if triple in train_triples:
            bad.append(x.task_id)
    A.eq(bad, [], f"这些 multi_tool 任务的航线+日期与训练完全相同：{bad}")


@test
def test_entity_tiers_cover_all_tasks_and_are_meaningful() -> None:
    """实体重合度三档必须覆盖全部任务，且**三档都非空**。

    这是替代没有含义的 ``seen``/``unseen`` 的方案（专家建议）。
    若某一档为空，说明分层退化（例如全部任务都是 seen），
    那样它和原来的 seen/unseen 一样没用。
    """
    m = _load_audit_module()
    rep = m.audit("v3")
    tiers = rep["entity_tiers"]
    total = sum(tiers.values())
    A.eq(total, rep["n_tasks"], "三档应覆盖全部任务")
    for tier in ("entity_new", "entity_partial", "entity_seen"):
        A.true(tiers.get(tier, 0) > 0,
               f"{tier} 档为空 —— 分层退化，失去信息量")

@test
def test_training_corpus_covers_observation_and_multi_tool_skills() -> None:
    """训练语料必须覆盖**技能**，不只是实体。

    ## 这是 v3 基线三个 0% 的根因（实测查出）

    实体覆盖检查**全绿**（52 个机场码都在），但：

        带观测轮（多步流程）的合成任务: 0 / 181
        multi_tool 训练任务:            0 / 181

    而评测有 **31 个带观测轮的任务** + **10 个 multi_tool 任务**。
    症状：
    * `booking_flow` 58/60 是 `action_hallucination`（从不调 `book_flight`）
    * `multi_tool` 38/40 是「调了航班没调天气」（只做一半）

    **实体覆盖检查抓不到这个** —— 所以必须单独守技能覆盖。
    """
    from src.data.recovery_corpus import (
        assert_training_covers_eval_entities,
        build_training_tasks,
    )

    rep = assert_training_covers_eval_entities()
    A.eq(rep["skill_gaps"], [], f"技能缺口：{rep['skill_gaps']}")
    A.true(rep["syn_obs_tasks"] > 0,
           "训练语料必须含带观测轮的多步任务（否则 booking_flow 学不会）")
    A.true(rep["syn_multi_tool"] > 0,
           "训练语料必须含 multi_tool 任务（否则只学会调一半）")

    syn = build_training_tasks(seed=42)
    obs = [x for x in syn if any(t.role == "tool" for t in x.turns)]
    A.true(len(obs) >= 20, f"带观测的任务应 >= 20，实际 {len(obs)}")


@test
def test_observation_tasks_require_reading_the_observation() -> None:
    """观测驱动的任务必须**不能被用户话直接回答**。

    这是关键判据：原来的 `_synthetic_booking_tasks` 把 `flight_id`
    直接写在**用户话里**（「帮我订 MU1000 这一班」），模型学的是
    「用户报航班号我就订」。而评测要求「**观测里给了航班号，所以我要订它**」。

    所以：**期望的 `flight_id` 不得出现在用户句里**。
    """
    from src.data.recovery_corpus import _synthetic_observation_tasks

    ts = _synthetic_observation_tasks()
    A.true(len(ts) >= 12, f"应至少 12 个，实际 {len(ts)}")
    bad = []
    for x in ts:
        fid = (x.expected.args or {}).get("flight_id", "")
        user_text = " ".join(t.content or "" for t in x.turns if t.role == "user")
        if fid and fid in user_text:
            bad.append(x.task_id)
    A.eq(bad, [], f"这些任务的 flight_id 泄露在用户句里（学不到读观测）：{bad[:5]}")

    # 反向：flight_id 必须**出现在观测里**（否则模型无从得知）
    missing = []
    for x in ts:
        fid = (x.expected.args or {}).get("flight_id", "")
        obs_ids = set()
        for t in x.turns:
            if t.role == "tool" and isinstance(t.observation, dict):
                for f in ((t.observation.get("result") or {}).get("flights") or []):
                    obs_ids.add(f.get("flight_id"))
        if fid not in obs_ids:
            missing.append(x.task_id)
    A.eq(missing, [], f"这些任务的期望 flight_id 不在观测里：{missing[:5]}")

@test
def test_clarify_training_has_multiple_missing_modes() -> None:
    """追问训练样本必须覆盖**多种缺失模式**，不能只加同模式副本。

    专家指出评测集 15 个 clarify 只有 **2 种**缺失模式，
    「有效样本是 1 个行为、15 是触发方式数」。加副本只让模型记措辞。

    实测背景：补多步语料（40 个，全是「要调工具」）把追问占比
    11.8% → 9.6%，`cancellation` 随之 62.5% → 31.8%。
    """
    from src.data.recovery_corpus import build_training_tasks

    syn = build_training_tasks(seed=42)
    clarify = [t for t in syn if t.expected.tool is None]
    modes = {t.mistake_hint for t in clarify if t.mistake_hint}
    A.true(len(modes) >= 8,
           f"缺失模式应 >= 8 种，实际 {len(modes)}：{sorted(modes)[:6]}")
    # 追问占比不能低于评测（14.2%），否则会被工具样本淹没
    share = len(clarify) / len(syn)
    A.true(share >= 0.14,
           f"追问占比 {share:.1%} 低于评测的 14.2% —— 会被工具样本淹没")


@test
def test_empty_output_is_not_graded_as_correct_clarify() -> None:
    """**空输出不得判为「正确追问」** —— 这是实测查出的严重漏洞。

    原来 `if not calls and expected_tool is None` 无条件返回 SUCCESS，
    于是空字符串也算「正确地向用户追问信息」。

    后果：未训练的基座在 15 个「应先追问」任务上通过 **98.2%**（55/56），
    其中 26/56 是「其他文本」甚至空输出 —— **那个 98.2% 是免费的**，
    让 baseline 总体准确率虚高约 **14 个百分点**（15/106 × 98.2%）。
    """
    from src.data.task_manifest import ExpectedCall, Task, Turn
    from src.eval.grader import grade

    t = Task(task_id="x", category="c", difficulty="e", visibility="train_only",
             turns=[Turn("user", "我不想要BK000001这个订单了")],
             target_turn=1, target_turn_kind="initial",
             expected=ExpectedCall(tool=None, args={}, note=""))
    # 空输出必须失败
    A.false(grade(t, "").passed, "空输出不得判为通过")
    A.false(grade(t, "   ").passed, "纯空白不得判为通过")
    # 合理追问必须通过
    A.true(grade(t, "请问您取消的原因是什么？").passed)
    A.true(grade(t, "请提供订单号").passed, "不带问号的追问也应通过")
    # 调了工具必须失败
    A.false(grade(t, '<tool_call>{"name": "cancel_booking",'
                     '"arguments": {"booking_id": "BK000001"}}</tool_call>').passed)

@test
def test_sft_data_has_no_tool_call_examples_for_clarify() -> None:
    """**关键实测发现**：`sft_only` 的 chosen 里 **0 条**不含 tool_call。

    纯正例监督在结构上无法表达「此处不该调用工具」。后果（v3 基线）：

        60/89 的失败是「信息不足时应先追问，却直接调用了工具」

    而补追问**任务**（合成语料 20→58）**完全没用** —— 因为那些任务
    从没进过 SFT 数据（`sft_only` 的 pair_type 组合里没有
    `clarify_boundary`）。

    这条测试把 `sft_only` 的这个**已知性质**固定下来，
    这样 `sft_clarify` 变体的对比才有意义。
    """
    import json
    from pathlib import Path as _P

    from src.data.preference_builder import EXPERIMENT_PAIR_TYPES as E

    # sft_only 不含 clarify_boundary —— 这是设计（对照组）
    A.true("clarify_boundary" not in E["sft_only"],
           "sft_only 应保持「无追问样本」的对照性质")
    # sft_clarify 含它 —— 这是实验组
    A.true("clarify_boundary" in E["sft_clarify"],
           "sft_clarify 必须包含 clarify_boundary")

    p = _REPO / "data/preferences/sft_only.train.jsonl"
    if p.exists():
        rows = [json.loads(ln) for ln in p.open(encoding="utf-8") if ln.strip()]
        nn = sum(1 for r in rows if "<tool_call>" not in r.get("chosen", ""))
        A.eq(nn, 0,
             f"sft_only 的 chosen 不应含「不调工具」样本，实际 {nn} 条 —— "
             "若变了，说明对照性质被破坏")

    p2 = _REPO / "data/preferences/sft_clarify.train.jsonl"
    if p2.exists():
        rows2 = [json.loads(ln) for ln in p2.open(encoding="utf-8") if ln.strip()]
        nn2 = sum(1 for r in rows2 if "<tool_call>" not in r.get("chosen", ""))
        share = nn2 / len(rows2)
        A.true(0.08 <= share <= 0.20,
               f"sft_clarify 的「不调工具」占比应接近评测的 14.2%，实际 {share:.1%}")

@test
def test_all_v3_tasks_are_solvable() -> None:
    """每个任务的期望值都必须能从**用户话/观测/纠正轮**推出。

    ## 实测查出 2 个「不可能完成」的任务（都是我的 bug）

    ```
    taskv2_0093  用户「后天从青岛到长沙的航班有哪些」
                 BASE_DATE_V2 = 2026-10-06 → 「后天」应是 2026-10-08
                 期望 date = 2026-11-08        ← 差 33 天，用户从未说过

    taskv2_0100  用户「查一下订单 bk8e51a3 的行李额度」
                 期望 booking_id = BK3A5C18     ← 与用户原话完全无关
    ```

    第二个的根因很具体：我做全局订单号替换时**只改了期望、漏改了用户话**。

    **危害**：这两个任务永远答不对，而报告把它们记成「模型参数值错」——
    **把标注错误记成了模型缺陷**。`taskv2_0093` 还进了「过度追问」的统计。
    """
    import importlib.util

    from src.data.task_manifest import load_tasks_by_version

    path = _REPO / "scripts/prepare/audit_task_solvability.py"
    spec = importlib.util.spec_from_file_location("_solv_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rows = [mod.audit_task(t) for t in load_tasks_by_version("v3")]
    bad = [(r["task_id"], r["hard_fails"]) for r in rows if not r["ok"]]
    A.eq(bad, [], f"这些任务的期望值无法推导（永远答不对）：{bad[:3]}")


#: 整体豁免的切片：其**全部**偏好对都不是工具调用，因此不适用工具覆盖判据。
#:
#: 判据是「不要静默丢掉一个工具」，而这些切片的 `chosen` 侧设计上就是
#: 自然语言澄清，一条工具调用都不该有 —— 用同一条判据去要求它们是错的。
#:
#: ⚠️ 光靠这张手工名单不够：`dose100`（clarify 占比 100% 的剂量档）
#: 同样没有工具调用，但它由 `dpo_dose_response.py --prepare-only` 生成，
#: 名字里带剂量后缀，不会有人记得回来补名单。所以测试里**同时**做了
#: 数据驱动的判定：**任何一条工具调用都没有的切片自动豁免**，
#: 并打印原因。名单只用于「明知有工具却仍要整片豁免」的少数情形。
_SLICES_WITHOUT_TOOL_CALLS = {
    # dpo_clarify 是 100% clarify_boundary 的单因素切片，用来直接测
    # 「只喂澄清对会怎样」。它没有工具调用是**预期行为**，不是缺口。
    "dpo_clarify",
}

#: 允许在**个别单因素切片**里缺席的工具，附缺席原因。
#:
#: 判据不是「覆盖越全越好」，而是「不能让一个切片静默丢掉整个工具」：
#: 若某工具在某个实验的训练数据里一条都没有，那些任务在该实验上
#: **0% 是保证的**，而报告会把它记成「该方法无效」。
_SLICE_TOOL_EXCEPTIONS = {
    # dpo_recovery 只含 error_recovery 对，而纠错语料 84 个场景里
    # `cancel_booking` 的**正解数为 0**（`fake_reason_*` 那类正解是
    # 「先追问原因」，转不成工具调用任务）。因此该切片天然覆盖不到
    # 这个工具，需要在实验解读时说明：它的绝对值会因 cancellation
    # 类别（v3 有 22 个任务）而偏低，**不宜用它论证整体能力**。
    ("dpo_recovery", "cancel_booking"),
}


@test
def test_no_slice_silently_drops_a_whole_tool() -> None:
    """任何实验切片都不能**静默**丢掉一整个工具。

    ## 为什么单独守这一条

    `assert_training_covers_eval_entities()` 检查的是**合成语料整体**
    的工具/机场码覆盖，它在所有切片上都通过。但真正喂进训练的是
    `data/preferences/<exp>.train.jsonl` —— 每个实验用自己的 pair_type
    子集重建池子，**单个切片可能整块缺一个工具**。

    实测（v3，本次新增的逐切片审计）：

    | 切片 | 缺机场码 | 缺工具 |
    |---|---|---|
    | all / sft_only / dpo_format | 0 | — |
    | dpo_joint | 0 | — |
    | **dpo_recovery** | **40** | **cancel_booking** |

    整体断言看不到这个。后果：`dpo_recovery` 上 cancellation 相关的
    22 个任务**必然 0%**，而报告若把该实验的总体准确率当"恢复能力的
    效果"来读，就是把语料缺口算成了方法缺陷。

    本测试把「缺整个工具」变成显式失败，已知且可接受的情形必须写进
    ``_SLICE_TOOL_EXCEPTIONS`` 并附原因 —— 不允许静默通过。
    """
    import collections
    import json
    import re

    from src.data.task_manifest import load_tasks_by_version

    ev_tools = {
        t.expected.tool for t in load_tasks_by_version("v3") if t.expected.tool
    }
    pref_dir = _REPO / "data" / "preferences"
    files = sorted(pref_dir.glob("*.train.jsonl"))
    A.true(len(files) > 0, "偏好数据目录里应有 *.train.jsonl —— 先跑 build_preferences.py")

    silent: list[tuple[str, str]] = []
    empty_slices: list[str] = []
    for path in files:
        stem = path.name.replace(".train.jsonl", "")
        if stem in _SLICES_WITHOUT_TOOL_CALLS:
            continue
        tools: set[str] = set()
        n_rows = 0
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            n_rows += 1
            chosen = json.loads(line).get("chosen", "")
            m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", chosen, re.S)
            if not m:
                continue
            try:
                name = json.loads(m.group(1)).get("name")
            except Exception:
                continue
            if name:
                tools.add(name)

        # 数据驱动豁免：整片一条工具调用都没有 → 这是「纯澄清类」切片
        # （如剂量档 dose100），工具覆盖判据对它不适用。
        if not tools:
            empty_slices.append(f"{stem}({n_rows} 条，无工具调用)")
            continue

        # 只关心「评测确实需要」的工具
        for missing in sorted(ev_tools - tools):
            if (stem, missing) in _SLICE_TOOL_EXCEPTIONS:
                continue
            silent.append((stem, missing))

    A.eq(
        silent,
        [],
        "这些实验切片**完全没有**某个评测所需工具的训练样本，"
        f"相关任务必然 0%（要么补语料，要么写进 _SLICE_TOOL_EXCEPTIONS 并说明）：{silent}",
    )

    # 例外清单本身也要有效：声明的例外必须**真的**缺席，
    # 否则说明语料已补齐而清单没更新（会掩盖新出现的缺口）。
    stale: list[tuple[str, str]] = []
    for stem, tool in _SLICE_TOOL_EXCEPTIONS:
        p = pref_dir / f"{stem}.train.jsonl"
        if not p.is_file():
            continue
        present = False
        for line in p.open(encoding="utf-8"):
            if f'"{tool}"' in line:
                present = True
                break
        if present:
            stale.append((stem, tool))
    A.eq(stale, [], f"例外清单已过期（该工具其实已覆盖，应删掉例外）：{stale}")
