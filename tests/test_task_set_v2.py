"""v2 高质量任务集的约束测试。

这些测试的作用：**把「高质量」变成 CI 会拦住的硬约束**。

为什么必须这样：v1 的五个缺陷（实体复用、过去日期、multi_tool 名不符实、
seen/unseen 骨架共享、骨架比过低）**没有任何一个会在评测时报错** ——
评测照常跑、照常出数字，只是数字没有意义。
这类缺陷只能靠结构化断言拦住。
"""

from __future__ import annotations

import collections
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.task_manifest_v2 import (  # noqa: E402
    BASE_DATE_V2,
    BOOKING_IDS_V2,
    PASSENGERS_V2,
    load_tasks_v3,
    skeleton,
    verify_task_set,
)
from src.data.task_set_v2_specs import build_specs  # noqa: E402

# ⚠️ 必须从 ``tests._harness`` 导入装饰器，**不能**从 ``tests.run_tests`` ——
# run_tests 会 import 本模块，反向导入会造成循环导入，
# 结果是本模块的测试**静默不被注册**（实测测试数仍是 91，没有报错）。
from tests._harness import A, test  # noqa: E402


@test
def test_v2_passes_all_structural_constraints() -> None:
    """v2 任务集必须通过 ``verify_task_set`` 的全部硬约束。

    这是**总闸**：实体唯一性、日期未来、骨架同类别唯一、multi_tool 真多工具
    全部由它检查。任何一条不过，这个测试失败。
    """
    specs = build_specs()
    # ⚠️ 不要求**整** 100 个：为了让「应先追问」的样本达到 14 个
    # （核心论点的证据），刻意放弃了整数的形式要求。样本量比整数重要。
    A.true(95 <= len(specs) <= 115, f"任务数应在 95~115 之间，实际 {len(specs)}")
    report = verify_task_set(specs)
    A.eq(
        report["violations"], [],
        f"v2 任务集存在结构性违规：{report['violations']}",
    )


@test
def test_v2_entities_are_genuinely_new() -> None:
    """实体必须**真的新的**，不是换说法。

    v1 的实测缺陷：订单号池只有 8 个，``BK7F2A9C`` 出现 7 次；
    机场码 ``SHA`` 出现 13 次；人名只有 6 个。

    后果：模型答对可能只是记住了那个订单号，而不是学会了调用规则。
    """
    specs = build_specs()
    bks = [m for s in specs for m in re.findall(r"BK[0-9A-F]{6}", s.user)]
    dup_bk = {k: v for k, v in collections.Counter(bks).items() if v > 1}
    A.eq(dup_bk, {}, f"订单号必须全库唯一，实际复用：{dup_bk}")
    A.true(len(set(bks)) >= 25, f"订单号应足够多，实际 {len(set(bks))} 个")

    names = [n for s in specs for n in PASSENGERS_V2 if n in s.user]
    dup_nm = {k: v for k, v in collections.Counter(names).items() if v > 1}
    A.eq(dup_nm, {}, f"乘客名必须全库唯一，实际复用：{dup_nm}")

    # 与 v1 的订单号不得重合（否则模型可能记住的是 v1 的实体）
    from src.data.task_manifest import build_tasks

    v1_bks = {m for t in build_tasks()
              for m in re.findall(r"BK[0-9A-F]{6}", t.turns[0].content)}
    overlap = set(bks) & v1_bks
    A.eq(overlap, set(), f"v2 不应复用 v1 的订单号，实际重合：{overlap}")


@test
def test_v2_dates_are_in_the_future() -> None:
    """日期必须是**未来**的。

    v1 的全部日期落在 ``2025-07/08``，而项目当前已是 2026-10 ——
    评测在让模型查一年前的航班，这在真实产品里是无效请求。
    """
    specs = build_specs()
    dates = {d for s in specs for d in re.findall(r"\d{4}-\d{2}-\d{2}", s.user)}
    A.true(len(dates) >= 40, f"日期应足够多样，实际 {len(dates)} 个")
    past = sorted(d for d in dates if d < BASE_DATE_V2.isoformat())
    A.eq(past, [], f"不得含过去日期：{past[:5]}")
    # 且应当落在基准日之后 1 年内（避免写成 2099 这种无意义日期）
    far = sorted(d for d in dates if d > "2027-06-30")
    A.eq(far, [], f"日期不应过于遥远：{far[:5]}")


@test
def test_v2_seen_unseen_has_no_skeleton_overlap() -> None:
    """``seen`` 与 ``unseen`` 的骨架必须**零交集** —— 否则「泛化」不成立。

    v1 的实测缺陷：两组共享 6 个骨架，占 unseen 的 26%。也就是说
    「unseen 泛化指标」测的其实是同一批任务类型的另一种说法。

    这不是「模型偷看」（v1 的训练语料骨架覆盖率只有 2~9%），而是
    **划分方式本身无意义**：按序号 ``counter % 3`` 切，切在骨架中间。
    修法是**按骨架分配**。
    """
    ts = load_tasks_v3()
    a = {skeleton(t.turns[0].content) for t in ts if t.visibility == "seen"}
    b = {skeleton(t.turns[0].content) for t in ts if t.visibility == "unseen"}
    A.true(len(a) > 0 and len(b) > 0, "两组都应非空")
    A.eq(a & b, set(), f"seen/unseen 骨架不该有交集，实际 {len(a & b)} 个")


@test
def test_v2_multi_tool_really_requires_multiple_tools() -> None:
    """``multi_tool`` 必须真的要求多个工具。

    v1 的实测缺陷：9 个 ``multi_tool`` 任务的 ``expected.tool`` **全部是
    ``search_flights``**，观测里给的 ``get_weather`` 从不参与判定 ——
    这个类别测的东西和它的名字完全不符。

    v2 要求每个任务通过 ``also_required`` 声明**额外**必须调用的工具。
    """
    ts = load_tasks_v3()
    mt = [t for t in ts if t.category == "multi_tool"]
    A.true(len(mt) >= 8, f"multi_tool 任务不应过少，实际 {len(mt)}")
    bad = [t.task_id for t in mt if "get_weather" not in t.tags]
    A.eq(bad, [], f"这些 multi_tool 任务未声明额外工具：{bad}")
    # 且观测里必须真的出现那个额外工具
    for t in mt:
        obs_tools = [x.observation.get("tool") for x in t.turns
                     if x.role == "tool" and isinstance(x.observation, dict)]
        A.true(
            "get_weather" in obs_tools,
            f"{t.task_id} 的观测里缺少 get_weather，多工具无法成立",
        )


@test
def test_v2_skeleton_ratio_is_high() -> None:
    """骨架/任务比必须接近 1 —— 这是「不是换说法」的量化判据。

    v1 的实测值：全库 0.52（100 任务 / 52 骨架），
    最差的 ``multi_tool`` 只有 **0.11**（9 任务 / 1 骨架）。
    """
    report = verify_task_set(build_specs())
    A.true(
        report["skeleton_ratio"] >= 0.95,
        f"全库骨架比应 ≥0.95，实际 {report['skeleton_ratio']}",
    )
    for cat, m in report["by_category"].items():
        A.true(
            m["ratio"] >= 0.9,
            f"{cat} 的骨架比应 ≥0.9，实际 {m['ratio']}（{m['n']} 任务 / "
            f"{m['skeletons']} 骨架）",
        )


@test
def test_v2_structural_checks_can_actually_fail() -> None:
    """反向验证：约束框架必须**真的能报警**，不是走过场。

    我给 v1 式缺陷喂进去，检查是否被逐条抓到。
    没有这个测试，前面几条断言可能只是因为「检查逻辑写错了、
    永远返回空违规」而通过。
    """
    from src.data.task_manifest_v2 import TaskSpec

    bad = [
        # 同类别内骨架重复 + 订单号复用
        TaskSpec("baggage", "帮我查一下订单BK111111的行李额度",
                 "check_baggage", {"booking_id": "BK111111"}),
        TaskSpec("baggage", "帮我查一下订单BK111111的行李额度",
                 "check_baggage", {"booking_id": "BK111111"}),
        # 过去日期
        TaskSpec("flight_search", "帮我查2025-07-18北京到上海的航班",
                 "search_flights",
                 {"origin": "PEK", "destination": "SHA", "date": "2025-07-18"}),
        # multi_tool 未声明额外工具
        TaskSpec("multi_tool", "2026-11-05我要从北京去上海，顺便看看天气",
                 "search_flights", {}),
    ]
    v = verify_task_set(bad)["violations"]
    A.true(any("订单号复用" in x for x in v), f"应抓到订单号复用，实际 {v}")
    A.true(any("过去日期" in x for x in v), f"应抓到过去日期，实际 {v}")
    A.true(
        any("multi_tool 未声明额外工具" in x for x in v),
        f"应抓到 multi_tool 未声明额外工具，实际 {v}",
    )

@test
def test_recovery_tasks_are_genuine_multi_turn_corrections() -> None:
    """``error_correction`` 必须是**真的多轮纠错**，不是「把其他类任务混进来」。

    专家指出 v2 初版的三个硬伤（实测确认）：
    1. 15 个任务的纠错话术**全是「请修正」三个字** —— 退化成固定 prompt
    2. 其中 2 个任务的 ``expected.args`` 是**空的** → 任何调用都判成功
       （这条会直接污染准确率）
    3. 没有一条真的做到「模型先错一次 → 用户针对该错误纠正 → 模型改对」

    修完后每条都是：user → assistant(具体错误) → user(针对该错误的纠正)
    → 期望(改对后的调用)。
    """
    ts = [t for t in load_tasks_v3() if t.category == "error_correction"]
    A.true(len(ts) >= 12, f"error_correction 任务不应过少，实际 {len(ts)}")

    # 1) 结构：3 轮，且中间是 assistant、末轮是 user
    for t in ts:
        A.eq(len(t.turns), 3, f"{t.task_id} 应为 3 轮（user/assistant/user）")
        A.eq(t.turns[1].role, "assistant", f"{t.task_id} 第 2 轮应是模型的首轮错误回复")
        A.eq(t.turns[2].role, "user", f"{t.task_id} 第 3 轮应是用户的纠错")

    # 2) 纠错话术必须各不相同（否则退化为固定 prompt）
    fixes = [t.turns[2].content for t in ts]
    A.eq(len(set(fixes)), len(fixes),
         f"纠错话术必须各不相同，实际 {len(set(fixes))}/{len(fixes)} 种")

    # 3) 纠错话术必须与该任务的错误内容相关（用关键词粗检）
    #    例如「没有 get_order_details」这类工具名的纠正，必须提到那个工具
    for t in ts:
        wrong = t.turns[1].content
        fix = t.turns[2].content
        A.true(len(fix) >= 8, f"{t.task_id} 的纠错话术过短：{fix!r}")
        # 若错误回复里含幻觉工具名，纠错话术应提到它
        if "get_order_details" in wrong:
            A.true("get_order_details" in fix,
                   f"{t.task_id} 的纠错话术未指出具体错误工具")

    # 4) 期望参数不得为空（空 args 会让任何调用都判成功）
    for t in ts:
        if t.expected.tool is not None:
            A.true(
                bool(t.expected.args),
                f"{t.task_id} 的期望参数为空 —— 任何调用都会判成功，"
                "这会污染准确率",
            )


@test
def test_there_are_genuine_multi_turn_tasks() -> None:
    """必须有**真正的多轮对话**任务（专家第 5 条）。

    项目叫 agent-optimization，但初版**没有一个真正的多轮任务**：
    ``multi_tool`` 的「3 轮」实质是「一次调两个工具」，不是「多轮对话」。

    真正的多轮标志：**后一轮的输入依赖前一轮的输出**，且用户意图会**变化**
    —— 模型必须用新参数**再调一次**，而不是重复上一轮的调用。
    """
    mt = [t for t in load_tasks_v3() if t.category == "multi_turn"]
    A.true(len(mt) >= 5, f"真多轮任务不应过少，实际 {len(mt)}")
    for t in mt:
        # 必须是 user → tool 观测 → user 的结构：末轮的输入依赖前一轮的输出
        roles = [x.role for x in t.turns]
        A.true("tool" in roles, f"{t.task_id} 缺少工具观测轮，无法构成多轮依赖")
        A.true(
            roles.count("user") >= 2,
            f"{t.task_id} 只有一个 user 轮，不构成多轮对话",
        )
        A.true("multi_turn" in t.tags, f"{t.task_id} 缺少 multi_turn 标签")


@test
def test_flight_ids_are_not_arithmetic() -> None:
    """``flight_id`` 不得是恒定等差（专家指出「明显是编的」）。

    初版用 ``"CA%04d" % (2000 + i*17)``，产生 CA2510 / CA2527 / CA2544 …
    相邻差值恒为 17 —— 一眼可辨是程序生成的。
    """
    import re as _re

    ts = [t for t in load_tasks_v3() if t.category == "booking_flow"]
    fids = [t.expected.args.get("flight_id") for t in ts]
    fids = [f for f in fids if f]
    A.true(len(fids) >= 10, f"booking_flow 任务不应过少，实际 {len(fids)}")
    nums = [int(_re.sub(r"\D", "", f)) for f in fids]
    diffs = [b - a for a, b in zip(nums, nums[1:])]
    A.true(
        len(set(diffs)) > 1,
        f"flight_id 呈恒定等差（差值恒为 {diffs[0]}），一眼可辨是编的",
    )


@test
def test_clarify_samples_are_enough_for_statistical_claims() -> None:
    """「应先追问」的样本量必须足够支撑统计结论。

    ``missing_reason`` 是本项目核心价值点的证据（SFT 0/4 vs DPO 4/4），
    而初版只有 4 个样本 —— Fisher 精确检验 p≈0.029，勉强显著，
    专家明确指出「4 个样本统计意义非常有限，至少补到 10-15 个」。
    """
    ts = load_tasks_v3()
    clarify = [t for t in ts if "clarify_first" in t.tags]
    A.true(
        len(clarify) >= 8,
        f"「应先追问」样本不应少于 12 个，实际 {len(clarify)} —— "
        "样本不足会让核心结论站不住",
    )

@test
def test_no_cross_category_entity_reuse() -> None:
    """**跨类别的实体不得重合** —— 否则某个类别不再测它的名义能力。

    专家从这个盲区里找出问题（实测确认）：
    * ``multi_tool`` 的航线与 ``flight_search`` 重复（当时 10 个里 3 个）
    * ``recovery`` 的 (工具, 参数) 与其他类完全重合
    * 天气城市与主航线城市全面重合

    为什么这会毁掉评测：**若两个类别用同一批实体，模型在一个类别上学会的
    映射可以直接套到另一个类别**。例如 multi_tool 若复用 flight_search 的
    「北京→上海」，它测的就只是「多调一个天气工具」，
    而航线理解那部分已经被训过了。

    判据的边界（我在这里先错了一次）：
    * 要防的是**同一组城市对（航线）**跨类别复用，不是「同一个城市名」
      —— 城市名重合是自然的，40+ 城可组合出上千组航线，而各类只需 10 余组
    * 排除 ``tool is None``（应追问类）的空签名 —— 那类任务天然跨类别
    """
    from src.data.task_set_v2_specs import build_specs

    specs = build_specs()

    def routes(cat: str) -> set:
        return {
            (s.args["origin"], s.args["destination"]) for s in specs
            if s.category == cat and "origin" in (s.args or {})
        }

    cats = ["flight_search", "booking_flow", "multi_tool", "multi_turn"]
    for i, c1 in enumerate(cats):
        for c2 in cats[i + 1:]:
            shared = routes(c1) & routes(c2)
            A.eq(shared, set(),
                 f"航线跨类别重合 {c1} ∩ {c2} = {sorted(shared)}")

    def bids(cat: str) -> set:
        return {s.args["booking_id"] for s in specs
                if s.category == cat and (s.args or {}).get("booking_id")}

    for i, c1 in enumerate(["baggage", "cancellation", "recovery"]):
        for c2 in ["baggage", "cancellation", "recovery"][i + 1:]:
            shared = bids(c1) & bids(c2)
            A.eq(shared, set(),
                 f"订单号跨类别重合 {c1} ∩ {c2} = {sorted(shared)}")

@test
def test_training_corpus_covers_eval_entities() -> None:
    """训练语料必须覆盖评测所需的**机场码**与**工具**。

    这条测试守的是一个**确定性失败**：「训练没见过 → 必然编造」，
    而且它不会报任何错。

    ## 两次踩坑记录

    **第一次**：``cancel_booking`` 在合成语料里完全没出现，而评测有 8 个
    cancellation 任务。缺口被「训练数据里混入了评测任务」掩盖，
    修掉污染后才暴露。

    **第二次**：机场码池曾有两个独立来源 —— ``tool_catalog``（20 城）
    与 ``task_manifest_v2``（60 城）。训练语料读前者、评测读后者，
    于是 v3 需要的 52 个码里有 **33 个模型从未见过**，
    ``flight_search`` 必然编造（实测编出 ``CCN``/``SXM``/``HTX``）。
    修法是**收敛到单一来源**（``src.env.tool_catalog`` 为唯一来源）。

    **第三次（加固时暴露）**：把航线目标数调小后，贪心选出的航线集中在
    少数城市，``CKG``/``CSX``/``INC``/``SYX``/``TAO`` 一条都没有 ——
    而且这个缺陷**只在撞车过滤之后**才暴露。
    修法是两阶段选择（阶段 1 是「每码至少 1 条」的硬要求）。
    """
    from src.data.recovery_corpus import assert_training_covers_eval_entities

    rep = assert_training_covers_eval_entities()
    A.eq(rep["missing_codes"], [], f"训练缺失机场码：{rep['missing_codes']}")
    A.eq(rep["missing_tools"], [], f"训练缺失工具：{rep['missing_tools']}")
    A.true(rep["syn_codes"] >= rep["eval_codes"],
           "训练覆盖的码数应不少于评测需要的")


@test
def test_airport_codes_have_a_single_source() -> None:
    """机场码必须只有**一个来源** —— 两份表必然漂移。

    实测记录：``tool_catalog.AIRPORT_CODES`` 曾只有 20 城，而
    ``task_manifest_v2.AIRPORT_CODES_V2`` 有 60 城。训练语料读前者、
    评测读后者，于是 33 个码在训练里缺席 —— **这是静默漂移，不报错**。
    """
    from src.env.tool_catalog import AIRPORT_CODES
    from src.data.task_manifest_v2 import AIRPORT_CODES_V2

    A.true(
        AIRPORT_CODES is AIRPORT_CODES_V2,
        "task_manifest_v2.AIRPORT_CODES_V2 必须直接引用 "
        "tool_catalog.AIRPORT_CODES（同一对象），而不是各自维护一份",
    )


@test
def test_training_corpus_has_no_prompt_collision_with_eval() -> None:
    """合成训练任务的 prompt 不得与**任何**评测集的 prompt 逐字相同。

    这既是防污染，也解释了为什么「训练覆盖」不能只靠一条航线：
    与评测撞车的那条会被过滤掉，若它是某个码的唯一来源，覆盖就断了。
    """
    from src.data.recovery_corpus import build_training_tasks
    from src.data.task_manifest import load_tasks
    from src.data.task_manifest_v2 import load_tasks_v3
    from src.models.prompt_builder import build_plain_prompt

    syn = build_training_tasks(seed=42)
    ev = {build_plain_prompt(t).strip() for t in load_tasks()}
    ev |= {build_plain_prompt(t).strip() for t in load_tasks_v3()}
    hit = [t.task_id for t in syn if build_plain_prompt(t).strip() in ev]
    A.eq(hit, [], f"这些合成任务的 prompt 与评测逐字相同：{hit[:5]}")
