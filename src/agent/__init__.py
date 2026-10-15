"""agent 执行循环与评测。

本包把「模型决策」与「工具执行」真正接起来，用于评测模型在多轮工具调用中的
行为，而不只是单轮「肯不肯调工具」。

主要导出：
    AgentLoop          多轮执行循环
    Trajectory         轨迹记录（可序列化）
    FaultSpec          工具故障注入规格
    ToolRuntime        工具执行 + 故障注入 + 审计
    run_and_collect    批量跑任务并聚合指标（见 metrics 模块）
"""

from src.agent.loop import (
    AgentLoop,
    AgentStep,
    FaultSpec,
    PolicyFn,
    ToolInvocation,
    ToolRuntime,
    Trajectory,
    judge_action,
    make_hf_policy_fn,
    render_observation,
)

__all__ = [
    "AgentLoop",
    "AgentStep",
    "FaultSpec",
    "PolicyFn",
    "ToolInvocation",
    "ToolRuntime",
    "Trajectory",
    "judge_action",
    "make_hf_policy_fn",
    "render_observation",
]
