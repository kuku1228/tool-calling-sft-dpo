"""无需 GPU 的脚本化假模型，用于测试 agent 循环。

为什么需要它
------------
agent 循环是本项目的关键组件，而它**最容易出的错都在编排层**，不在模型层：

* 工具执行后 observation 有没有正确回灌
* 达到期望动作后有没有正确终止
* 轮数上限有没有生效
* 模型给最终回答时有没有识别出来
* 故障注入后模型重试、轨迹是否标记了 ``recovered_from_fault``

这些用**假模型**就能完整测出来，而且**秒级完成**。相比之下用真模型测一次要
十几分钟、还要 GPU。开发期已经反复吃过「关键组件不可测试」的亏
（``SFTPairDataset`` 定义在函数内部无法 import、``train_monitor`` 的键名错误
要到云端才发现），所以这里刻意把模型侧做成**可脚本化的替身**。

设计
----
``ScriptedPolicy`` 按「第几次被调用」返回预设输出，因此可以精确构造：

* 先输出一个工具调用，再输出最终答案（正常两轮）
* 故意输出坏 JSON（测参数校验路径）
* 每次都调用工具（测轮数上限）
* 第一次调用失败、第二次成功（测故障恢复）

用法::

    from src.agent.fake_policy import ScriptedPolicy, tool_call_text
    policy = ScriptedPolicy([
        tool_call_text("search_flights", {"origin": "PEK", ...}),
        tool_call_text("book_flight", {"flight_id": "MU1000", ...}),
    ])
    traj = AgentLoop(policy).run(task)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Sequence


def tool_call_text(name: str, arguments: dict[str, Any]) -> str:
    """渲染一个规范的 ``<tool_call>`` 输出，供假模型使用。

    >>> tool_call_text("get_weather", {"city": "北京"})
    '<tool_call>{"name": "get_weather", "arguments": {"city": "北京"}}</tool_call>'
    """
    payload = {"name": name, "arguments": arguments}
    return f"<tool_call>{json.dumps(payload, ensure_ascii=False)}</tool_call>"


def broken_tool_call_text(name: str) -> str:
    """渲染一个**语法坏掉**的 ``<tool_call>``（键无引号），用于测解析失败路径。

    >>> "<tool_call>" in broken_tool_call_text("x")
    True
    """
    return f'<tool_call>{{"name": "{name}", "arguments": {{city: 北京}}}}</tool_call>'


def plain_text(text: str) -> str:
    """渲染一个不含工具调用的自然语言回答。"""
    return text


@dataclass
class ScriptedPolicy:
    """按调用序号返回预设输出的假模型。

    Args:
        outputs: 预设输出序列。**超出长度后重复最后一个** —— 这样「一直调用工具」
            这类测试不需要写很长的列表。
        record: 是否记录收到的消息历史（便于断言「observation 有没有回灌」）。
        token_len: 假装生成的 token 数（仅用于填充 ``num_tokens``）。
    """

    outputs: Sequence[str]
    record: bool = True
    token_len: int = 32
    #: 实际收到的每次调用的 messages 快照（浅拷贝列表，内容为原字典）
    calls: list[list[dict[str, str]]] = field(default_factory=list)
    #: 每个输出对应的 finish_reason
    finish_reason: str = "stop"

    def __post_init__(self) -> None:
        if not self.outputs:
            raise ValueError("outputs 不能为空")

    def __call__(self, messages: Sequence[dict[str, str]]) -> tuple[str, str, int]:
        idx = min(len(self.calls), len(self.outputs) - 1)
        if self.record:
            self.calls.append([dict(m) for m in messages])
        return self.outputs[idx], self.finish_reason, self.token_len

    # --- 便于测试的辅助 ---

    @property
    def num_calls(self) -> int:
        return len(self.calls)

    def last_messages(self) -> list[dict[str, str]]:
        """最近一次调用收到的消息历史。"""
        return self.calls[-1] if self.calls else []

    def saw_observation(self, needle: str = "<observation>") -> bool:
        """是否在某次调用中收到了 observation（验证回灌是否生效）。

        >>> p = ScriptedPolicy(["a"])
        >>> p([{"role": "user", "content": "x"}])
        ('a', 'stop', 32)
        >>> p.saw_observation()
        False
        """
        return any(
            needle in m.get("content", "")
            for call in self.calls
            for m in call
        )
