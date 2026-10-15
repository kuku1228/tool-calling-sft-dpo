"""agent 执行循环：把「模型决策」与「工具执行」真正接起来。

为什么需要这个模块
------------------
本项目的单轮评测（``scripts/eval/run_baseline.py``）走的是**捷径**：
它把任务里预计算的 ``Turn.observation`` 直接注入 prompt，于是模型是
「读现成答案」。以 ``task_0023`` 为例::

    turn0  user   帮我订一张2025-07-24 北京到上海的机票
    turn1  tool   obs={"ok": true, "result": {"flights": [{"flight_id": "MU1000", ...}]}}
    turn2  user   就订 MU1000 这一班吧        ← 模型「知道」MU1000 是因为答案被喂进来了
    expected      book_flight(flight_id="MU1000", passenger_name="张伟")

真实 agent 不能这样工作 —— 它必须**自己调用 ``search_flights`` 拿到航班列表**，
从返回结果里读出 ``MU1000``，再决定下单。这就是单轮评测与 agent 评测的本质差别：

    单轮：给答案 → 测「肯不肯调工具」
    agent：不给答案 → 测「会不会自己取信息、读结果、再决策」

本模块实现后者。它**复用现有的** ``execute_tool``（已实现 6 个工具的完整执行
语义与错误返回），只补上缺失的部分：循环、轨迹记录、终止判定、故障注入。

分层设计
--------
::

    AgentLoop            ← 编排：循环、终止、轨迹记录
      ├── ToolRuntime    ← 工具执行 + 故障注入 + 调用审计
      ├── Trajectory     ← 每一步的不可变记录（便于离线分析/回放）
      └── PolicyFn       ← 模型：给定 messages → 文本（可注入假模型便于测试）

``ToolRuntime`` 与 ``PolicyFn`` 都是可替换的，所以整个循环**不需要 GPU 就能测试**
（用脚本化假模型 + 真实工具执行即可）。这是刻意的：开发期多次因为「不可测试的
关键组件」而在云端才发现问题（见 ``train_monitor`` 与 ``SFTPairDataset`` 的教训）。

术语
----
- **turn**：一次「模型输出 → 工具执行 → 结果回灌」的完整往返
- **action**：模型一次输出里的工具调用（可能为空 = 直接回答）
- **observation**：工具执行的返回，回灌给模型的文本

用法::

    # 真实模型
    from src.agent.loop import AgentLoop
    loop = AgentLoop(policy_fn, max_turns=6)
    traj = loop.run(task)

    # 无需 GPU 的自测（脚本化假模型 + 真实工具）
    from src.agent.fake_policy import ScriptedPolicy
    loop = AgentLoop(ScriptedPolicy([...]))
    traj = loop.run(task)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

from src.data.task_manifest import Task
from src.env.tool_catalog import execute_tool

# ============================================================================
# 数据契约
# ============================================================================


@dataclass
class ToolInvocation:
    """一次工具调用的完整记录（含执行结果）。"""

    turn: int
    name: str
    arguments: dict[str, Any]
    ok: bool
    result: dict[str, Any] | None = None
    error: str = ""
    #: 是否由故障注入器强制失败（用于区分「模型自己的错」与「环境的错」）
    injected_fault: str = ""
    #: 执行耗时（秒）
    elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "name": self.name,
            "arguments": self.arguments,
            "ok": self.ok,
            "result": self.result,
            "error": self.error,
            "injected_fault": self.injected_fault,
            "elapsed_s": round(self.elapsed_s, 4),
        }


@dataclass
class AgentStep:
    """循环里的一步：模型说了什么、调了什么、拿到什么。"""

    turn: int
    #: 模型原始输出
    text: str
    #: 解析出的工具调用（可能多个；本项目约定一次一个）
    calls: list[ToolInvocation] = field(default_factory=list)
    #: 生成结束原因（stop / length / error）
    finish_reason: str = ""
    num_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "text": self.text,
            "finish_reason": self.finish_reason,
            "num_tokens": self.num_tokens,
            "calls": [c.to_dict() for c in self.calls],
        }


#: 轨迹的终止原因。每种都要能被明确区分 —— 「没做完」和「做错了」是两件事。
TERMINATION_FINAL_ANSWER = "final_answer"        # 模型不再调用工具，给出最终回答
TERMINATION_EXPECTED_REACHED = "expected_reached"  # 已完成期望动作（成功路径）
TERMINATION_MAX_TURNS = "max_turns"              # 达到轮数上限（未完成）
TERMINATION_ERROR = "error"                      # 循环内部异常
TERMINATION_POLICY_ERROR = "policy_error"        # 模型侧异常


@dataclass
class Trajectory:
    """一条完整轨迹。可序列化，便于离线分析与回放。"""

    task_id: str
    category: str = ""
    difficulty: str = ""
    steps: list[AgentStep] = field(default_factory=list)
    #: 全部工具调用（跨 step 展平，便于统计）
    invocations: list[ToolInvocation] = field(default_factory=list)
    terminated_by: str = ""
    #: 是否达成了任务的期望动作
    success: bool = False
    #: 期望动作是在第几轮达成的（None = 未达成）
    success_turn: int | None = None
    #: 最终动作的判定说明（与单轮评测同一套判据）
    verdict: str = ""
    error: str = ""
    wall_time_s: float = 0.0
    #: 该任务是否注入过故障
    fault_injected: bool = False
    #: 任务定义里期望的工具调用数（工具链长度）—— 用于判断
    #: 「模型是否走了应有的依赖链」，而不只看最终动作
    expected_tool_calls: int = 0
    #: 播进历史里的预设 assistant 轮数。这些是任务定义的参考文本，
    #: 而模型会生成自己的版本 —— 两者不一致时历史会轻微不连贯，
    #: 因此需要记录以便按此分组分析。
    assistant_turns_preloaded: int = 0

    # --- 派生指标（供评估层使用；不在这里重复口径）---

    @property
    def num_turns(self) -> int:
        return len(self.steps)

    @property
    def num_tool_calls(self) -> int:
        return len(self.invocations)

    @property
    def num_failed_calls(self) -> int:
        """执行失败的工具调用数（含注入故障与真实错误）。"""
        return sum(1 for c in self.invocations if not c.ok)

    @property
    def num_invalid_calls(self) -> int:
        """**模型自己**导致的无效调用（排除注入故障）。"""
        return sum(1 for c in self.invocations if not c.ok and not c.injected_fault)

    @property
    def recovered_from_fault(self) -> bool:
        """注入过故障，但最终仍然成功 —— 这是 agent 鲁棒性的核心指标。"""
        return self.fault_injected and self.success

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "category": self.category,
            "difficulty": self.difficulty,
            "terminated_by": self.terminated_by,
            "success": self.success,
            "success_turn": self.success_turn,
            "verdict": self.verdict,
            "error": self.error,
            "wall_time_s": round(self.wall_time_s, 3),
            "fault_injected": self.fault_injected,
            "expected_tool_calls": self.expected_tool_calls,
            "assistant_turns_preloaded": self.assistant_turns_preloaded,
            "num_turns": self.num_turns,
            "num_tool_calls": self.num_tool_calls,
            "num_failed_calls": self.num_failed_calls,
            "num_invalid_calls": self.num_invalid_calls,
            "recovered_from_fault": self.recovered_from_fault,
            "steps": [s.to_dict() for s in self.steps],
        }


# ============================================================================
# 策略接口
# ============================================================================


class PolicyFn(Protocol):
    """模型侧接口：给定消息列表 → (文本, finish_reason, token 数)。

    刻意定得极窄（只要消息进、文本出），这样：

    * 真实 HF 模型、mock 模型、脚本化假模型**都能实现它**
    * 循环本身**不依赖 transformers**，因此能在没有 GPU 的环境里测试

    ⚠️ 不要传 ``Task`` 进来。传消息列表是刻意的 —— 它保证循环无法「偷看」
    任务的期望答案（``task.expected``）。早期的单轮评测就是把 ``Task``
    整个传下去，导致很容易写出「用了不该用的信息」的代码。
    """

    def __call__(
        self, messages: Sequence[dict[str, str]]
    ) -> tuple[str, str, int]:  # pragma: no cover
        ...


# ============================================================================
# 工具运行时（执行 + 故障注入 + 审计）
# ============================================================================


@dataclass
class FaultSpec:
    """故障注入规格。用于评测 agent 的鲁棒性。

    为什么需要它：真实 agent 面对的工具**会失败**（超时、限流、服务异常）。
    一个只会「顺利路径」的 agent 是不合格的。而干净的基准里不会有故障，
    所以必须主动注入才能测出恢复能力。

    注入是**确定性**的（由 ``(task_id, turn, call_index)`` 派生），
    因此同一任务重跑得到相同故障 —— 否则指标不可复现。
    """

    #: 在第几轮注入（None = 任意轮）
    at_turn: int | None = None
    #: 对哪些工具注入（None = 任意工具）
    tools: tuple[str, ...] | None = None
    #: 注入概率（按 task+turn 确定性派生，不走随机数）
    probability: float = 1.0
    #: 故障文案。模拟真实错误语义，模型需要据此判断能否重试
    messages: tuple[str, ...] = (
        "服务暂时不可用，请稍后重试（错误码 503）",
        "请求超时（错误码 504）",
        "工具执行失败：上游服务异常",
    )
    #: 是否允许重试成功（True = 只失败一次，重试会成功）
    #:
    #: 真实系统多半是「瞬时故障」，重试可恢复。设为 True 时能测出
    #: 「模型会不会重试」这个具体行为，而不是只测「运气好不好」。
    transient: bool = True


def _stable_score(*parts: Any) -> float:
    """由任意键派生 [0,1) 的确定性分数。

    刻意用 ``hashlib`` 而不是 **内建哈希函数** —— 后者对字符串按进程随机化
    （PYTHONHASHSEED），会让故障注入不可复现。本项目已在偏好数据上踩过这个
    坑，见 ``src/utils/hashing.py``。那里也有静态检查防止复发。

    注意：本说明里不写出那个内建函数的调用形式，否则项目自带的
    「禁止内建哈希做种子」检查会把这段文档误报为违规代码。
    """
    from src.utils.hashing import stable_seed

    return stable_seed(*parts) / 0xFFFFFFFF


class ToolRuntime:
    """工具执行层：包住 ``execute_tool``，加上故障注入与审计。

    Args:
        fault: 故障注入规格；None 表示不注入。
        executor: 底层执行函数，默认 ``execute_tool``。
            可替换以便测试（注入一个总是失败的假执行器）。
    """

    def __init__(
        self,
        fault: FaultSpec | None = None,
        executor: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.fault = fault
        self.executor = executor or execute_tool
        #: 记录「哪些 (任务, 工具) 已经故障过一次」，用于 transient 语义。
        #:
        #: ⚠️ 键里**不能含 turn**。模型遇到故障后的重试必然发生在**下一轮**，
        #: 若键含 turn，重试时键不同 → 又注入一次故障 → 永远无法恢复。
        #: 开发期就是这么写的，导致「故障恢复」用例失败。
        #: 瞬时故障的正确定义是「同一任务里的同一工具只失败一次」，与轮次无关。
        self._faulted_once: set[tuple[str, str]] = set()
        #: 被注入过故障的任务集合（跨 turn 累积）
        self.fault_tasks: set[str] = set()

    def _should_fault(self, task_id: str, turn: int, tool: str) -> str:
        """决定本次调用是否注入故障，返回故障文案（空串 = 不注入）。"""
        f = self.fault
        if f is None:
            return ""
        if f.at_turn is not None and f.at_turn != turn:
            return ""
        if f.tools is not None and tool not in f.tools:
            return ""
        key = (task_id, tool)
        # transient 语义：同一 (任务, 工具) 只失败一次，重试会成功
        if f.transient and key in self._faulted_once:
            return ""
        if _stable_score(task_id, turn, tool, "fault") >= f.probability:
            return ""
        self._faulted_once.add(key)
        self.fault_tasks.add(task_id)
        # ⚠️ ``n`` 是**仅关键字**参数，必须写 ``n=...``。
        # 写成位置参数会被 ``*parts`` 吞掉并抛 TypeError
        # （表现为 policy_error 且 turns=0，很难一眼看出根因）。
        pick = stable_index(task_id, turn, tool, "msg", n=len(f.messages))
        return f.messages[pick]

    def call(self, task_id: str, turn: int, name: str, arguments: dict[str, Any]) -> ToolInvocation:
        """执行一次工具调用，返回完整记录。"""
        t0 = time.time()

        fault_msg = self._should_fault(task_id, turn, name)
        if fault_msg:
            return ToolInvocation(
                turn=turn,
                name=name,
                arguments=dict(arguments),
                ok=False,
                error=fault_msg,
                injected_fault=fault_msg,
                elapsed_s=time.time() - t0,
            )

        try:
            raw = self.executor(name, arguments)
        except Exception as exc:  # 执行器本身抛异常也不能毁掉整条轨迹
            return ToolInvocation(
                turn=turn,
                name=name,
                arguments=dict(arguments),
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                elapsed_s=time.time() - t0,
            )

        ok = bool(raw.get("ok", True)) if isinstance(raw, dict) else False
        return ToolInvocation(
            turn=turn,
            name=name,
            arguments=dict(arguments),
            ok=ok,
            result=raw.get("result") if isinstance(raw, dict) else None,
            error=str(raw.get("error", "")) if isinstance(raw, dict) else "返回值不是 dict",
            elapsed_s=time.time() - t0,
        )


def stable_index(*parts: Any, n: int) -> int:
    """确定性选取 [0, n) 的下标。

    ⚠️ ``n`` 是**仅关键字**参数。调用时必须写 ``stable_index(a, b, n=3)``
    —— 这正是它的设计目的（避免把 ``n`` 和 ``parts`` 混在一起）。
    但如果写错成把 ``n`` 当位置参数混进 ``parts``，就会抛 TypeError；
    开发期就在故障注入器里犯过这个错（见 ``ToolRuntime._should_fault``）。

    >>> 0 <= stable_index("a", "b", n=3) < 3
    True
    >>> stable_index("a", "b", n=3) == stable_index("a", "b", n=3)
    True
    >>> stable_index("a", n=1)
    0
    """
    if n <= 0:
        return 0
    return int(_stable_score(*parts) * n) % n


# ============================================================================
# 观察结果渲染
# ============================================================================


def render_observation(inv: ToolInvocation) -> str:
    """把工具执行结果渲染成回灌给模型的文本。

    格式与任务集里预计算的 ``Turn.observation`` 保持一致的风格
    （``<observation>...</observation>`` 包裹的 JSON），这样：

    * 模型看到的历史结构与训练/单轮评测一致，不会引入分布漂移
    * 失败结果也走同一通道 —— 真实的 agent 必须能读懂错误信息

    >>> inv = ToolInvocation(turn=1, name="get_weather", arguments={"city": "北京"},
    ...                      ok=True, result={"condition": "晴"})
    >>> render_observation(inv)
    '<observation>{"ok": true, "tool": "get_weather", "result": {"condition": "晴"}}</observation>'
    >>> inv2 = ToolInvocation(turn=1, name="x", arguments={}, ok=False, error="超时")
    >>> render_observation(inv2)
    '<observation>{"ok": false, "tool": "x", "error": "超时"}</observation>'
    """
    payload: dict[str, Any] = {"ok": inv.ok, "tool": inv.name}
    if inv.ok:
        payload["result"] = inv.result
    else:
        payload["error"] = inv.error
    return f"<observation>{json.dumps(payload, ensure_ascii=False)}</observation>"


# ============================================================================
# 成功判定
# ============================================================================


def judge_action(inv: ToolInvocation, task: Task) -> tuple[bool, str]:
    """判断一次工具调用是否达成了任务的**期望动作**。

    复用单轮的参数等价判据（:func:`src.eval.equivalence.match_arguments`），
    保证 agent 评测与单轮评测的口径一致 —— 否则两个数字不可比。

    Returns:
        ``(是否达成, 说明)``
    """
    from src.eval.equivalence import match_arguments

    exp = task.expected
    if exp.tool is None:
        # 该轮期望「不调用工具，先追问」。
        # 在 agent 语义下，达成方式是「结束时没有调用工具并在追问」——
        # 由循环在终止时判定，这里不认为是 success。
        return False, "该轮期望澄清而非调用工具（由循环终止时判定）"
    if not inv.ok:
        return False, f"工具执行失败：{inv.error}"
    if inv.name != exp.tool:
        return False, f"工具选错：期望 {exp.tool!r}，实际 {inv.name!r}"
    graded = exp.graded_args or {}
    m = match_arguments(exp.args, inv.arguments, graded_subset=graded or None)
    if m is None:  # pragma: no cover - match_arguments 从不返回 None
        return True, "工具名正确（无参数约束）"
    # ⚠️ 口径必须与 grader 一致：主指标用 ``lenient_matched``。
    # 字段名是 ``lenient_matched`` / ``strict_matched`` —— **不是** ``matched`` / ``ok``。
    # 开发期按猜的字段名写过一次，导致**所有轨迹都判失败**（模型明明调对了），
    # 白跑一轮才发现。数据类的属性名不可假设，先打印 dir()。
    ok = bool(getattr(m, "lenient_matched", False))
    if ok:
        return True, "工具名与参数均正确"
    # ⚠️ ``ArgMatchResult.failures`` 与 ``param_accuracy`` 是**方法**，不是属性！
    # 必须 ``m.failures()``。写成 ``m.failures`` 会得到一个 bound method，
    # 迭代它抛 ``TypeError: 'method' object is not subscriptable``，
    # 而该异常会被循环捕获成 ``policy_error`` —— 表现为「成功率莫名偏低」，
    # 且错误信息完全指不到根因。开发期就是因此让 2 条本可成功的轨迹被判失败。
    # 数据类的成员形态不可假设，先用 ``type()``/``inspect`` 确认。
    try:
        bad = list(m.failures())
    except TypeError:  # pragma: no cover - 兼容 failures 为属性的旧版本
        bad = list(getattr(m, "failures", []) or [])
    detail = "; ".join(
        f"{getattr(c, 'field', '?')}: 期望 {getattr(c, 'expected', '?')!r}"
        f" 实际 {getattr(c, 'actual', '?')!r}"
        f"（{getattr(c, 'reason', '')}）"
        for c in bad[:3]
    )
    return False, f"参数不对：{detail or '见 ArgMatchResult'}"


# ============================================================================
# 主循环
# ============================================================================


class AgentLoop:
    """多轮 agent 执行循环。

    每轮：模型输出 → 解析工具调用 → 执行 → 结果回灌 → 下一轮。
    直到：达成期望动作 / 模型给最终回答 / 达到轮数上限。

    Args:
        policy: 模型侧可调用对象（见 :class:`PolicyFn`）。
        max_turns: 轮数上限。必须设 —— 没有上限的 agent 会烧光预算。
        fault: 故障注入规格（评测鲁棒性时用）。
        system_prompt: 覆盖默认 system prompt（一般不用改）。
        stop_on_expected: 达成期望动作后是否立即停止（默认 True）。
        builder: 构造初始消息的函数，默认用项目的
            ``prompt_builder.build_messages``。可替换以便测试。
    """

    def __init__(
        self,
        policy: PolicyFn,
        *,
        max_turns: int = 6,
        fault: FaultSpec | None = None,
        stop_on_expected: bool = True,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns 必须 >= 1")
        self.policy = policy
        self.max_turns = max_turns
        self.runtime = ToolRuntime(fault=fault)
        self.stop_on_expected = stop_on_expected

    # --- 解析模型输出里的工具调用 ---

    @staticmethod
    def _parse_calls(text: str) -> tuple[list[tuple[str, dict[str, Any]]], str]:
        """从模型输出里解析工具调用。返回 ``(calls, finish_hint)``。"""
        from src.eval.tool_call_parser import extract_tool_calls

        parsed = extract_tool_calls(text)
        calls: list[tuple[str, dict[str, Any]]] = []
        for c in parsed.calls:
            name = getattr(c, "name", "") or ""
            args = getattr(c, "arguments", None) or {}
            if name:
                calls.append((name, dict(args)))
        return calls, (parsed.error or "")

    # --- 主入口 ---

    def run(self, task: Task) -> Trajectory:
        """跑一条完整轨迹。**不依赖任务里预计算的 observation**。"""
        from src.models.prompt_builder import build_messages, render_chat_text

        t0 = time.time()
        traj = Trajectory(
            task_id=task.task_id,
            category=task.category,
            difficulty=task.difficulty,
        )

        # --- 消息播种：只给「当前该回答的那一轮」，不给未来信息 ---
        #
        # ⚠️ 这一段的正确性极难凭直觉判断，开发期在同一个地方**栽了两次**，
        # 两次都让成功率剧烈偏移（94.87% → 53.85%）。所以把推理写清楚。
        #
        # 【评测发生在哪个时刻】
        #   单轮评测把 prompt 播种到 ``task.target_turn`` 之前，然后让模型输出
        #   一次动作。agent 评测必须**在同一个时刻**开始，否则两个数字不可比。
        #
        # 【为什么不能播后续 user 轮】
        #   以 ``task_0023`` 为例::
        #
        #       turn0 [user] 帮我订一张2025-07-24 北京到上海的机票
        #       turn1 [tool] （search_flights 的返回，含 flight_id=MU1000）
        #       turn2 [user] 就订 MU1000 这一班吧          ← 依赖 turn1 的信息
        #
        #   ``turn2`` 里的 "MU1000" **只有模型自己调 search_flights 才能知道**。
        #   如果把 turn2 也播进去，等于把答案提前泄露：模型直接 book_flight，
        #   跳过搜索 —— 实测 10 条 booking_flow **全部失败**（0%）。
        #
        # 【为什么不能播 assistant 预设轮】
        #   任务集里有 20 个 assistant 轮（如「我确认一下：您要取消 X，
        #   原因是「身体不适」，对吗？」）。它们是**目标动作的参考输出**，
        #   是「模型应该产出什么」而非「模型已经看到了什么」。
        #   播进去会制造「模型没说过那句话，用户却回应了它」的不连贯 ——
        #   实测 8 条 cancellation **全部失败**（0%）。
        #
        # 【结论】
        #   只播 ``turn0``（首个 user 轮）。后续 user 轮由循环在**模型每次
        #   成功调用工具之后**按需释放 —— 这才复现了真实的时序。
        #
        # 【唯一保留的 agent 特性】
        #   ``include_observations=False``：不把预计算的 observation 注入。
        #   工具结果只能由模型自己的调用换来。这是 agent 与「读现成答案」的分界。
        bundle = build_messages(task, include_observations=False)
        _TOOL_PLACEHOLDER = "（工具已返回结果）"
        seeded: list[dict[str, str]] = [
            m
            for m in bundle.messages
            if m["role"] in ("system", "user") and m["content"] != _TOOL_PLACEHOLDER
        ]
        first_user_idx = next(
            (i for i, m in enumerate(seeded) if m["role"] == "user"), None
        )
        if first_user_idx is None:
            traj.terminated_by = TERMINATION_ERROR
            traj.error = "任务里没有 user 轮"
            return traj
        deferred_user_turns = [
            m["content"] for m in seeded[first_user_idx + 1:] if m["role"] == "user"
        ]
        messages: list[dict[str, str]] = seeded[: first_user_idx + 1]

        # 记录任务定义里出现的工具轮，用于统计「期望的工具链长度」
        expected_tool_turns = [t for t in task.turns if t.role == "tool"]
        traj.expected_tool_calls = len(expected_tool_turns)
        # 注：assistant 预设轮不参与播种，因此该计数恒为 0。
        # 保留字段是为了向后兼容与便于确认「确实没有播种」。
        traj.assistant_turns_preloaded = 0

        try:
            for turn in range(self.max_turns):
                text, finish, ntok = self.policy(messages)
                step = AgentStep(turn=turn, text=text, finish_reason=finish, num_tokens=ntok)

                calls, parse_err = self._parse_calls(text)

                if not calls:
                    # 模型没有调用工具 → 视为最终回答，轨迹结束
                    traj.steps.append(step)
                    traj.terminated_by = TERMINATION_FINAL_ANSWER
                    traj.verdict = (
                        "该轮期望澄清（未调用工具），符合预期"
                        if task.expected.tool is None
                        else f"模型未调用工具，直接回答（该轮期望 {task.expected.tool}）"
                    )
                    traj.success = task.expected.tool is None
                    break

                # 执行本轮的调用（本项目约定一次一个，但支持多个）
                expected_hit = False
                for name, args in calls:
                    inv = self.runtime.call(task.task_id, turn, name, args)
                    step.calls.append(inv)
                    traj.invocations.append(inv)

                    ok, why = judge_action(inv, task)
                    # 无论成功失败都记录判据 —— 否则失败轨迹的 verdict 会被
                    # 「达到轮数上限」覆盖，真正的失败原因（参数写错 / 工具选错）
                    # 在报告里完全看不到，排查时只能去翻原始轨迹。
                    traj.verdict = why
                    if ok:
                        expected_hit = True
                        traj.success = True
                        traj.success_turn = turn

                    # 回灌观测结果（失败也要回灌 —— 模型需要据此纠正）
                    messages.append(
                        {"role": "assistant", "content": text}
                    )
                    messages.append(
                        {"role": "user", "content": render_observation(inv)}
                    )

                traj.steps.append(step)

                if expected_hit and self.stop_on_expected:
                    traj.terminated_by = TERMINATION_EXPECTED_REACHED
                    break

                # 按需释放下一个 user 轮。
                #
                # 时序依据：任务里后续的 user 轮（如「就订 MU1000 这一班吧」）
                # 在现实中是**用户看到工具结果之后**才说的话。所以它应该在
                # 模型产生了工具调用、并拿到 observation 之后才出现。
                if deferred_user_turns:
                    messages.append({"role": "user", "content": deferred_user_turns.pop(0)})
            else:
                # for-else：循环跑满 max_turns 仍未 break
                traj.terminated_by = TERMINATION_MAX_TURNS
                # 若之前已记录过失败判据（如「参数不对：...」），
                # 保留它并补上「未达成」的信息，而不是覆盖成一句笼统的上限提示。
                tail = f"（已用满 {self.max_turns} 轮未达成期望动作）"
                traj.verdict = (traj.verdict + tail) if traj.verdict else f"达到轮数上限{tail}"

        except Exception as exc:  # 模型侧或解析侧异常不能丢整条轨迹
            traj.terminated_by = TERMINATION_POLICY_ERROR
            traj.error = f"{type(exc).__name__}: {exc}"
            traj.verdict = traj.verdict or f"循环异常：{traj.error}"

        traj.fault_injected = task.task_id in self.runtime.fault_tasks
        traj.wall_time_s = time.time() - t0
        return traj


def task_needs_tool_chaining(task: Task) -> bool:
    """任务是否**真的需要多轮工具链**。

    ⚠️ 不要用 ``len(task.turns) > 1`` 判断「是不是多轮任务」——
    那个条件包含大量**并非真多轮**的任务：

    * ``recovery`` 类（如 ``task_0089``）的 ``turns`` 里含纠错轮，
      但目标动作在第 0 轮就能完成 —— agent 一轮即可
    * 有些任务只是数组里多了轮次，但**第一轮的信息已足够**

    真正需要多轮的是「后续轮次所需的信息只存在于前序工具返回里」——
    典型形态是 ``turns[1].role == "tool"``（有一个工具返回），
    且后续 user 轮依赖它。

    开发期用 ``len(turns) > 1`` 筛选，导致 39 个"多轮"任务里有 11 个
    一轮就完成，`avg_turns` 被稀释到 1.72，看起来像「agent 很高效」，
    实际是筛选口径错了。

    >>> from src.data.task_manifest import load_tasks
    >>> ts = {t.task_id: t for t in load_tasks()}
    >>> task_needs_tool_chaining(ts["task_0023"])   # 必须先 search 才能 know flight_id
    True
    >>> task_needs_tool_chaining(ts["task_0089"])   # 一轮即可
    False
    """
    # 条件：存在一个 tool 角色的轮次，且它后面还有 user 轮
    # （说明后续 user 的话依赖那个工具返回）
    for i, turn in enumerate(task.turns):
        if turn.role == "tool" and any(t.role == "user" for t in task.turns[i + 1:]):
            return True
    return False


# ============================================================================
# 便捷入口：把 HFPolicy 适配成 PolicyFn
# ============================================================================


def make_hf_policy_fn(
    policy: Any,
    *,
    tools: Sequence[dict[str, Any]] | None = None,
    system_prompt: str = "",
) -> PolicyFn:
    """把 ``HFPolicy`` 适配成循环需要的 ``PolicyFn``。

    ``HFPolicy`` 走的是 ``predict(task, sample_index)``，它内部会读任务里预计算的
    observation —— 那正是本模块要绕开的。这里改用
    :meth:`HFPolicy.generate_from_messages`，只依赖传入的消息历史。

    ``tools`` 与 ``system_prompt`` 由调用方从 ``build_messages(task)`` 取出后
    固定下来，保证 agent 推理与单轮推理的 system prompt / 工具定义**逐字一致**
    —— 否则会引入训练/推理分布不一致（本项目最忌讳的那类问题）。
    """

    def _fn(messages: Sequence[dict[str, str]]) -> tuple[str, str, int]:
        return policy.generate_from_messages(
            list(messages), tools=list(tools or []), system_prompt=system_prompt
        )

    return _fn
