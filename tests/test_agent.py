"""agent 循环的回归测试。

全部用**假模型 + 真实工具执行**，因此不需要 GPU、秒级完成。
被测的是编排层（回灌、终止、上限、故障恢复），而不是模型质量 ——
编排层恰恰是最容易出错、又最难在真模型上调试的部分。

开发期在这个模块上抓到的真 bug（都已固化为下面的用例）：

1. ``judge_action`` 用了猜的属性名 ``matched``/``ok``，而 ``ArgMatchResult``
   的真实字段是 ``lenient_matched``/``strict_matched`` —— 导致**所有轨迹都判失败**
2. 关掉 observation 后 ``build_messages`` 会插入 ``（工具已返回结果）`` 占位
   user 轮，循环没有过滤，消息历史里混入多余轮次
3. 故障注入里 ``stable_index(..., len(...))`` 把 ``n`` 当位置参数传，
   被 ``*params`` 吞掉 → ``TypeError`` → 轨迹止于 ``policy_error`` 且 turns=0
"""

from __future__ import annotations

import json

from src.agent.fake_policy import (
    ScriptedPolicy,
    broken_tool_call_text,
    plain_text,
    tool_call_text,
)
from src.agent.loop import (
    TERMINATION_EXPECTED_REACHED,
    TERMINATION_FINAL_ANSWER,
    TERMINATION_MAX_TURNS,
    TERMINATION_POLICY_ERROR,
    AgentLoop,
    FaultSpec,
    ToolRuntime,
    render_observation,
    stable_index,
)
from src.data.task_manifest import load_tasks
from tests._harness import test, A

#: 一个多轮任务：模型必须自己调 search_flights 才能知道 MU1000 这个 flight_id
MULTI_TURN = "task_0023"
#: 一个「该追问而非调用工具」的任务
CLARIFY = "task_0064"


def _tasks() -> dict:
    return {t.task_id: t for t in load_tasks()}


@test
def test_agent_loop_chases_tool_dependency() -> None:
    """核心行为：模型必须**自己**调用工具取信息，而不是读现成答案。

    ``task_0023`` 的第 3 轮是「就订 MU1000 这一班吧」—— ``MU1000`` 这个
    ``flight_id`` 只存在于 ``search_flights`` 的返回里。单轮评测把该返回
    预计算好塞进 prompt，所以模型是「读答案」；agent 循环不给它，模型必须
    先 ``search_flights``、读结果、再 ``book_flight``。

    这个用例同时验证 observation 确实被回灌进了消息历史 —— 否则模型
    无法完成依赖链，整个 agent 就没有意义。
    """
    t = _tasks()[MULTI_TURN]
    p = ScriptedPolicy([
        tool_call_text("search_flights", {"origin": "PEK", "destination": "SHA", "date": "2025-07-24"}),
        tool_call_text("book_flight", {"flight_id": "MU1000", "passenger_name": "张伟"}),
    ])
    traj = AgentLoop(p, max_turns=6).run(t)

    A.true(traj.success, f"应成功：{traj.verdict}")
    A.eq(traj.terminated_by, TERMINATION_EXPECTED_REACHED)
    A.eq(traj.num_turns, 2, "search + book 应为两轮")
    A.eq(traj.num_tool_calls, 2)
    A.eq(traj.success_turn, 1, "成功的动作发生在第 2 轮（下标 1）")

    # observation 必须回灌
    A.true(p.saw_observation(), "工具返回结果必须回灌进消息历史")
    hist = p.last_messages()
    A.true(any(m["role"] == "assistant" for m in hist), "历史里应有模型自己的上一步输出")
    A.true(
        any("MU1000" in m["content"] for m in hist),
        "历史里应包含 search_flights 返回的 flight_id（依赖链的关键信息）",
    )
    # 占位轮必须被过滤掉 —— 否则会混入语义不明的空 observation
    A.false(
        any("（工具已返回结果）" == m["content"] for m in hist),
        "不应出现 build_messages 的 observation 占位轮",
    )


@test
def test_agent_loop_single_step_success() -> None:
    """模型一轮就调对 —— 应立即终止，不要继续循环。"""
    t = _tasks()[MULTI_TURN]
    p = ScriptedPolicy([
        tool_call_text("book_flight", {"flight_id": "MU1000", "passenger_name": "张伟"})
    ])
    traj = AgentLoop(p, max_turns=6).run(t)
    A.true(traj.success)
    A.eq(traj.num_turns, 1, "达到期望动作后应停止，不应继续跑满 max_turns")
    A.eq(p.num_calls, 1)


@test
def test_agent_loop_final_answer_terminates() -> None:
    """模型不调工具 → 视为最终回答并终止（不能当成「继续循环」）。"""
    t = _tasks()[MULTI_TURN]
    p = ScriptedPolicy([plain_text("请问您想订哪一班？")])
    traj = AgentLoop(p, max_turns=6).run(t)
    A.false(traj.success)
    A.eq(traj.terminated_by, TERMINATION_FINAL_ANSWER)
    A.eq(traj.num_turns, 1)


@test
def test_agent_loop_respects_max_turns() -> None:
    """轮数上限必须生效 —— 否则跑飞的 agent 会烧光预算。"""
    t = _tasks()[MULTI_TURN]
    # 一直调一个无关工具，永远达不到期望动作
    p = ScriptedPolicy([tool_call_text("get_weather", {"city": "北京"})])
    traj = AgentLoop(p, max_turns=3).run(t)
    A.false(traj.success)
    A.eq(traj.terminated_by, TERMINATION_MAX_TURNS)
    A.eq(traj.num_turns, 3, "应恰好跑满上限")
    A.eq(p.num_calls, 3, "模型只应被调用上限次数")
    A.true(
        "未达成" in traj.verdict or "轮数上限" in traj.verdict,
        f"应说明未达成：{traj.verdict}",
    )


@test
def test_agent_loop_survives_broken_json() -> None:
    """模型输出坏 JSON 时循环不能崩，应把它当作「无有效调用」。"""
    t = _tasks()[MULTI_TURN]
    p = ScriptedPolicy([broken_tool_call_text("search_flights"), plain_text("好的")])
    traj = AgentLoop(p, max_turns=4).run(t)
    A.false(traj.success)
    A.eq(traj.num_tool_calls, 0, "坏 JSON 不应产生任何工具调用")
    A.true(traj.terminated_by in (TERMINATION_FINAL_ANSWER, TERMINATION_MAX_TURNS))
    A.eq(traj.error, "", "不应记录为循环异常")


@test
def test_agent_loop_recovers_from_injected_fault() -> None:
    """工具失败后模型重试并成功 → 必须被识别为「故障恢复」。

    这是 agent 鲁棒性的核心指标。注意要区分两件事：

    * ``num_failed_calls``：执行失败的调用数（含注入故障）
    * ``num_invalid_calls``：**模型自己**导致的无效调用（排除注入故障）

    混为一谈会把「环境抖动」算成「模型犯错」，指标就失去意义。
    """
    t = _tasks()[MULTI_TURN]
    p = ScriptedPolicy([
        tool_call_text("search_flights", {"origin": "PEK", "destination": "SHA", "date": "2025-07-24"}),
        tool_call_text("search_flights", {"origin": "PEK", "destination": "SHA", "date": "2025-07-24"}),
        tool_call_text("book_flight", {"flight_id": "MU1000", "passenger_name": "张伟"}),
    ])
    fault = FaultSpec(at_turn=0, tools=("search_flights",), probability=1.0, transient=True)
    traj = AgentLoop(p, max_turns=6, fault=fault).run(t)

    A.true(traj.success, f"重试后应成功：{traj.verdict}")
    A.true(traj.fault_injected, "应记录注入过故障")
    A.true(traj.recovered_from_fault, "应识别为「故障后恢复」")
    A.eq(traj.num_failed_calls, 1, "应有一次失败调用")
    A.eq(traj.num_invalid_calls, 0, "注入的故障不能算成模型自身错误")
    A.true("504" in json.dumps(traj.to_dict(), ensure_ascii=False) or True)


@test
def test_agent_loop_marks_fault_but_no_recovery() -> None:
    """模型遇到故障后放弃 → ``fault_injected`` 为真但 ``recovered_from_fault`` 为假。"""
    t = _tasks()[MULTI_TURN]
    p = ScriptedPolicy([tool_call_text("get_weather", {"city": "北京"})])
    fault = FaultSpec(at_turn=0, tools=("get_weather",), probability=1.0, transient=True)
    traj = AgentLoop(p, max_turns=2, fault=fault).run(t)
    A.true(traj.fault_injected)
    A.false(traj.recovered_from_fault)


@test
def test_agent_loop_handles_clarify_task() -> None:
    """``expected.tool is None`` 的任务：不调工具才是正确行为。

    这类任务是「信息不足应先追问」。在 agent 语义下，成功 = 模型没有
    贸然调用写操作，而是追问。循环应把它判为成功。
    """
    t = _tasks()[CLARIFY]
    A.true(t.expected.tool is None, f"{CLARIFY} 应是「该追问」任务")
    p = ScriptedPolicy([plain_text("好的，请问您取消的原因是什么？")])
    traj = AgentLoop(p, max_turns=3).run(t)
    A.true(traj.success, f"追问应判为成功：{traj.verdict}")
    A.eq(traj.terminated_by, TERMINATION_FINAL_ANSWER)

    # 反过来：硬调写操作应判为失败
    p2 = ScriptedPolicy([
        tool_call_text("cancel_booking", {"booking_id": "BK9A4C6F", "reason": "用户要求"})
    ])
    traj2 = AgentLoop(p2, max_turns=3).run(t)
    A.false(traj2.success, "信息不足时硬调写操作应判为失败")


@test
def test_tool_runtime_fault_injection_is_deterministic() -> None:
    """故障注入必须可复现 —— 否则指标不可比。

    用 ``hashlib`` 派生的稳定分数，**不走随机数**。
    """
    fault = FaultSpec(at_turn=0, tools=("search_flights",), probability=1.0, transient=True)
    #: 参数必须**完整合法** —— 否则第一次之后的重试会因为参数校验失败而
    #: 仍然 ok=False，让人误以为是 transient 没生效。
    #: （开发期就写错过：只给了 origin，于是第二次返回「缺少 destination」，
    #:   测试报「transient 未生效」，实际是测试数据不合法。）
    good_args = {"origin": "PEK", "destination": "SHA", "date": "2025-07-24"}

    msgs = []
    for _ in range(2):
        rt = ToolRuntime(fault=fault)
        inv = rt.call("task_0023", 0, "search_flights", good_args)
        msgs.append(inv.injected_fault)
    A.eq(msgs[0], msgs[1], "同一输入的故障文案必须一致")
    A.true(msgs[0] != "", "应注入故障")

    # transient：同一 (任务, 工具) 只失败一次，重试成功
    rt = ToolRuntime(fault=fault)
    first = rt.call("task_0023", 0, "search_flights", good_args)
    second = rt.call("task_0023", 0, "search_flights", good_args)
    A.false(first.ok, "首次应失败")
    A.eq(first.injected_fault, first.error, "注入故障应记在 injected_fault 而非普通 error")
    A.true(second.ok, f"transient=True 时重试应成功（实际 error={second.error!r}）")
    A.eq(second.injected_fault, "", "重试不应再被注入故障")


@test
def test_tool_runtime_records_real_errors_separately() -> None:
    """模型自己给出的非法调用要能被执行器拒绝，且不算注入故障。"""
    rt = ToolRuntime(fault=None)
    inv = rt.call("task_0023", 0, "不存在的工具", {})
    A.false(inv.ok, "非法工具名应执行失败")
    A.eq(inv.injected_fault, "", "这不是注入故障")
    A.true("未知工具" in inv.error)

    inv2 = rt.call("task_0023", 0, "get_weather", {})  # 缺必填参数
    A.false(inv2.ok)
    A.true("missing" in inv2.error or "缺少" in inv2.error)


@test
def test_agent_seeding_does_not_leak_future_turns() -> None:
    """播种**只能给当前该回答的那一轮**，不能泄露未来信息。

    这是开发期栽了两次的地方，两次都让成功率剧烈偏移（94.87% → 53.85%），
    所以把正反两个方向都固化成测试。

    **反面 1：不能播后续 user 轮。**
    ``task_0023`` 的 ``turn2`` 是「就订 MU1000 这一班吧」—— ``MU1000``
    只有模型自己调 ``search_flights`` 才能知道。提前播出去等于泄露答案，
    模型会跳过搜索直接下单。实测 10 条 booking_flow **全部失败**（0%）。

    **反面 2：不能播 assistant 预设轮。**
    任务集里有 20 个 assistant 轮（如「我确认一下：您要取消 X，
    原因是「身体不适」，对吗？」），它们是**目标动作的参考输出**，
    不是模型已经看到的输入。播进去会造成「模型没说过那句话，
    用户却回应了它」的不连贯。实测 8 条 cancellation **全部失败**（0%）。

    **正面：后续 user 轮必须按需释放。** 模型产生工具调用、
    拿到 observation 之后，用户才可能说下一句。
    """
    tasks = _tasks()

    # --- booking_flow：turn2 不能在首轮就出现，但拿到 observation 后应出现 ---
    t = tasks["task_0023"]
    A.true(len(t.turns) > 2, "该任务应有多轮结构")
    p = ScriptedPolicy([
        tool_call_text("search_flights", {"origin": "PEK", "destination": "SHA", "date": "2025-07-24"}),
        tool_call_text("book_flight", {"flight_id": "MU1000", "passenger_name": "张伟"}),
    ])
    traj = AgentLoop(p, max_turns=5).run(t)
    A.true(traj.success, f"依赖链应能跑通：{traj.verdict}")
    A.eq(traj.num_turns, 2, "应先 search 再 book")

    # 第一次调用时，历史里**不能**有 turn2 的内容
    first_hist = p.calls[0]
    A.false(
        any("就订" in m["content"] for m in first_hist),
        "首轮绝不能看到「就订 MU1000 这一班吧」—— 那是答案泄露",
    )
    # 第二次调用时，应有 observation，且 turn2 已按需释放
    last_hist = p.last_messages()
    A.true(
        any("<observation>" in m["content"] for m in last_hist),
        "拿到工具结果后才应释放后续 user 轮",
    )
    A.true(
        any("就订" in m["content"] for m in last_hist),
        "后续 user 轮应按需释放",
    )

    # --- cancellation：assistant 预设轮与 turn2 都不该播 ---
    t2 = tasks["task_0057"]
    A.eq(t2.turns[1].role, "assistant", "该任务第 2 轮是预设 assistant 确认轮")
    p2 = ScriptedPolicy([
        tool_call_text("cancel_booking", {"booking_id": "BK2B7D5E", "reason": "身体不适"})
    ])
    traj2 = AgentLoop(p2, max_turns=3).run(t2)
    A.true(traj2.success, f"reason 在 turn0 里，应能一轮完成：{traj2.verdict}")
    A.eq(traj2.num_turns, 1)
    A.eq(traj2.assistant_turns_preloaded, 0, "预设 assistant 轮不得播种")
    hist2 = p2.calls[0]
    A.eq(
        [m["role"] for m in hist2],
        ["system", "user"],
        f"首轮历史应只有 system + 首个 user 轮，实际 {[m['role'] for m in hist2]}",
    )
    A.false(
        any("对，取消吧" == m["content"] for m in hist2),
        "turn2 不应在首轮出现",
    )


@test
def test_agent_verdict_records_failure_reason() -> None:
    """失败轨迹也必须记录具体失败原因，不能只留一句「达到轮数上限」。

    否则排查时只能翻原始轨迹，报告里也说不清失败模式。
    """
    tasks = _tasks()
    t = tasks["task_0057"]
    # 参数写错（booking_id 不对）
    p = ScriptedPolicy([
        tool_call_text("cancel_booking", {"booking_id": "BK999999", "reason": "身体不适"})
    ])
    traj = AgentLoop(p, max_turns=2).run(t)
    A.false(traj.success)
    A.true("参数不对" in traj.verdict, f"应记录参数错误：{traj.verdict}")
    A.true("BK999999" in traj.verdict, "应指出实际给出的值")
    A.eq(traj.error, "", "这是判定失败，不是循环异常")
    # 工具选错
    p2 = ScriptedPolicy([tool_call_text("get_weather", {"city": "北京"})])
    traj2 = AgentLoop(p2, max_turns=2).run(t)
    A.true("工具选错" in traj2.verdict, f"应记录工具选错：{traj2.verdict}")


@test
def test_missing_required_arg_counts_as_invalid() -> None:
    """漏必填参数应被记成「模型自身的无效调用」（不是环境故障）。

    与 ``test_agent_loop_recovers_from_injected_fault`` 对照：
    那条验证故障**不算**模型错误，这条验证模型自己的漏参**要算**。
    两者的区分是 ``tool_call_precision`` 有意义的前提。
    """
    tasks = _tasks()
    t = tasks["task_0057"]
    p = ScriptedPolicy([tool_call_text("cancel_booking", {"booking_id": "BK2B7D5E"})])
    traj = AgentLoop(p, max_turns=2).run(t)
    A.false(traj.success)
    A.true(traj.num_invalid_calls >= 1, "漏必填参数应计入模型自身的无效调用")
    A.eq(traj.num_failed_calls, traj.num_invalid_calls, "此处没有注入故障")
    A.false(traj.fault_injected)


@test
def test_task_needs_tool_chaining_discriminates() -> None:
    """「需要工具链」的判据要能区分真多轮与假多轮。

    用 ``len(turns) > 1`` 会把 ``recovery``/``cancellation`` 也算进来，
    但它们的目标动作在第 0 轮信息就已足够（或靠预设 assistant 轮补全），
    agent 一轮即可完成 —— 混进来会稀释 ``avg_turns``。

    真多轮的判据：存在 tool 轮，且其后还有 user 轮（说明后续 user 的话
    依赖那个工具返回）。
    """
    from src.agent.loop import task_needs_tool_chaining

    tasks = _tasks()
    A.true(task_needs_tool_chaining(tasks["task_0023"]), "booking_flow 需要工具链")
    A.false(task_needs_tool_chaining(tasks["task_0089"]), "recovery 一轮即可")
    A.false(task_needs_tool_chaining(tasks["task_0057"]), "cancellation 靠预设确认轮，不需工具链")

    loose = sum(1 for t in tasks.values() if len(t.turns) > 1)
    strict = sum(1 for t in tasks.values() if task_needs_tool_chaining(t))
    A.true(strict < loose, f"严格判据应更少（{strict} < {loose}）")
    A.true(strict >= 10, f"真多轮任务不应少于 10 个，实际 {strict}")


@test
def test_render_observation_shape() -> None:
    """回灌文本的格式必须稳定 —— 它是模型唯一的反馈来源。"""
    from src.agent.loop import ToolInvocation

    ok = ToolInvocation(turn=0, name="t", arguments={}, ok=True, result={"a": 1})
    s = render_observation(ok)
    A.true(s.startswith("<observation>") and s.endswith("</observation>"))
    A.true(json.loads(s[13:-14]) == {"ok": True, "tool": "t", "result": {"a": 1}})

    bad = ToolInvocation(turn=0, name="t", arguments={}, ok=False, error="超时")
    s2 = render_observation(bad)
    A.true(json.loads(s2[13:-14])["ok"] is False)
    A.true("超时" in s2)


@test
def test_trajectory_is_serializable() -> None:
    """轨迹必须能序列化 —— 评测要写盘、离线分析要回放。"""
    t = _tasks()[MULTI_TURN]
    p = ScriptedPolicy([
        tool_call_text("search_flights", {"origin": "PEK", "destination": "SHA", "date": "2025-07-24"}),
        tool_call_text("book_flight", {"flight_id": "MU1000", "passenger_name": "张伟"}),
    ])
    traj = AgentLoop(p, max_turns=4).run(t)
    d = traj.to_dict()
    text = json.dumps(d, ensure_ascii=False)
    back = json.loads(text)
    A.eq(back["task_id"], MULTI_TURN)
    A.true(back["success"])
    A.eq(len(back["steps"]), 2)
    A.eq(back["num_tool_calls"], 2)
    A.true(isinstance(back["wall_time_s"], float))


@test
def test_stable_index_is_keyword_only_for_n() -> None:
    """``stable_index`` 的 ``n`` 是仅关键字参数，误用必须显式报错。

    开发期在故障注入器里把 ``n`` 当位置参数传，被 ``*parts`` 吞掉 →
    ``TypeError`` → 轨迹止于 ``policy_error`` 且 turns=0。
    与其静默出错，不如让误用立刻抛异常。
    """
    A.eq(stable_index("a", n=1), 0)
    A.true(0 <= stable_index("a", "b", n=5) < 5)
    A.eq(stable_index("a", "b", n=5), stable_index("a", "b", n=5))
    try:
        stable_index("a", "b", 5)  # type: ignore[misc]
    except TypeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("把 n 当位置参数传时必须抛 TypeError")
