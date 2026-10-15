"""agent 轨迹级指标：从一批 :class:`Trajectory` 聚合出可写进报告的指标。

与单轮指标的分工
----------------
单轮评测（``run_baseline.py``）回答「**这一步**模型调对了吗」；
agent 评测回答「**整条轨迹**模型把事办成了吗」。

两者不可互相替代：一个模型可能每步都对但把流程走乱（轮数爆炸），
也可能某步调错工具却最终把事办成了。所以本项目两套指标都保留。

关键指标
--------
=========================  ====================================================
指标                        含义与“为什么重要”
=========================  ====================================================
``trajectory_success_rate``  整条轨迹是否达成期望动作 —— **agent 的主指标**
``avg_turns``               平均轮数 —— 效率；轮数多说明在绕路
``avg_turns_to_success``    成功轨迹的平均轮数 —— 越接近 1 越果断
``tool_call_precision``     有效工具调用占比 —— 反映决策质量
``fault_recovery_rate``     **工具失败后最终成功的比例 —— agent 鲁棒性核心**
``give_up_rate``            提前给最终回答（放弃）的比例
``max_turns_rate``          撞上限的比例 —— 说明它“卡住了”
=========================  ====================================================

``fault_recovery_rate`` 的分母是**注入过故障的轨迹数**，不是全部轨迹 ——
否则故障少的时候这个比率会被稀释得看不出问题。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from src.agent.loop import (
    TERMINATION_EXPECTED_REACHED,
    TERMINATION_FINAL_ANSWER,
    TERMINATION_MAX_TURNS,
    TERMINATION_POLICY_ERROR,
    Trajectory,
)


def _rate(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0


def _mean(xs: Sequence[float]) -> float:
    return round(sum(xs) / len(xs), 3) if xs else 0.0


@dataclass
class AgentMetrics:
    """一批轨迹的聚合指标。"""

    label: str = ""
    num_trajectories: int = 0
    #: 整条轨迹达成期望动作的数量
    num_success: int = 0
    #: 按终止原因计数
    terminations: dict[str, int] = field(default_factory=dict)
    #: 工具调用总数 / 失败数 / 模型自身无效数
    num_tool_calls: int = 0
    num_failed_calls: int = 0
    num_invalid_calls: int = 0
    #: 注入过故障的轨迹数，以及其中最终成功的数量
    num_faulted: int = 0
    num_recovered: int = 0
    #: 轮数统计
    turns_all: list[int] = field(default_factory=list)
    turns_success: list[int] = field(default_factory=list)
    #: 分组（难度 / 类别）明细
    by_difficulty: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_category: dict[str, dict[str, Any]] = field(default_factory=dict)
    wall_time_s: float = 0.0

    # --- 派生比率 ---

    @property
    def success_rate(self) -> float:
        """**主指标**：整条轨迹成功率。"""
        return _rate(self.num_success, self.num_trajectories)

    @property
    def avg_turns(self) -> float:
        return _mean(self.turns_all)

    @property
    def avg_turns_to_success(self) -> float:
        return _mean(self.turns_success)

    @property
    def tool_call_precision(self) -> float:
        """有效工具调用占比（1 − 无效/总数）。"""
        if not self.num_tool_calls:
            return 0.0
        return round(1.0 - self.num_invalid_calls / self.num_tool_calls, 4)

    @property
    def fault_recovery_rate(self) -> float:
        """**鲁棒性主指标**：故障后恢复的比例（分母是故障轨迹数）。"""
        return _rate(self.num_recovered, self.num_faulted)

    @property
    def give_up_rate(self) -> float:
        return _rate(self.terminations.get(TERMINATION_FINAL_ANSWER, 0), self.num_trajectories)

    @property
    def max_turns_rate(self) -> float:
        return _rate(self.terminations.get(TERMINATION_MAX_TURNS, 0), self.num_trajectories)

    @property
    def error_rate(self) -> float:
        return _rate(self.terminations.get(TERMINATION_POLICY_ERROR, 0), self.num_trajectories)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "num_trajectories": self.num_trajectories,
            "num_success": self.num_success,
            "success_rate": self.success_rate,
            "terminations": dict(self.terminations),
            "num_tool_calls": self.num_tool_calls,
            "num_failed_calls": self.num_failed_calls,
            "num_invalid_calls": self.num_invalid_calls,
            "tool_call_precision": self.tool_call_precision,
            "num_faulted": self.num_faulted,
            "num_recovered": self.num_recovered,
            "fault_recovery_rate": self.fault_recovery_rate,
            "avg_turns": self.avg_turns,
            "avg_turns_to_success": self.avg_turns_to_success,
            "give_up_rate": self.give_up_rate,
            "max_turns_rate": self.max_turns_rate,
            "error_rate": self.error_rate,
            "by_difficulty": self.by_difficulty,
            "by_category": self.by_category,
            "wall_time_s": round(self.wall_time_s, 2),
        }

    def headline_lines(self) -> list[str]:
        """终端展示用的行（与 eval_report 的风格对齐）。"""
        lines = [
            f"【{self.label}】 {self.num_trajectories} 条轨迹（agent 多轮）",
            "-" * 66,
            f"  轨迹成功率 success_rate     {self.success_rate:.2%}",
            f"  平均轮数   avg_turns        {self.avg_turns:.2f}",
            f"  成功轨迹平均轮数            {self.avg_turns_to_success:.2f}",
            f"  工具调用有效率 precision    {self.tool_call_precision:.2%}",
        ]
        if self.num_faulted:
            lines += [
                f"  故障恢复率 recovery_rate    {self.fault_recovery_rate:.2%}"
                f"  （{self.num_recovered}/{self.num_faulted} 条注入过故障）",
            ]
        lines += [
            f"  放弃率     give_up_rate     {self.give_up_rate:.2%}",
            f"  撞上限率   max_turns_rate   {self.max_turns_rate:.2%}",
            f"  循环异常率 error_rate       {self.error_rate:.2%}",
            "",
            "  终止原因分布:",
        ]
        for k, v in sorted(self.terminations.items(), key=lambda kv: -kv[1]):
            lines.append(f"      {k:<22} {v}")
        for key, title in (("by_difficulty", "按难度"), ("by_category", "按类别")):
            g = getattr(self, key)
            if g:
                lines.append("")
                lines.append(f"  {title}:")
                for name, v in sorted(g.items()):
                    lines.append(
                        f"      {name:<20} n={v['n']:<4} "
                        f"success={v['success_rate']:.1%}  avg_turns={v['avg_turns']:.2f}"
                    )
        return lines


def _group_metrics(trajs: Sequence[Trajectory], attr: str) -> dict[str, dict[str, Any]]:
    """按 ``attr``（difficulty / category）分组统计。"""
    buckets: dict[str, list[Trajectory]] = defaultdict(list)
    for t in trajs:
        key = getattr(t, attr, "") or "(未知)"
        buckets[key].append(t)
    out: dict[str, dict[str, Any]] = {}
    for name, items in buckets.items():
        succ = [t for t in items if t.success]
        out[name] = {
            "n": len(items),
            "num_success": len(succ),
            "success_rate": _rate(len(succ), len(items)),
            "avg_turns": _mean([t.num_turns for t in items]),
            "avg_turns_to_success": _mean([t.num_turns for t in succ]),
        }
    return out


def aggregate(trajs: Iterable[Trajectory], *, label: str = "") -> AgentMetrics:
    """把一批轨迹聚合成指标。

    >>> from src.agent.loop import Trajectory
    >>> a = Trajectory(task_id="t1", success=True, terminated_by="expected_reached")
    >>> a.steps = [object(), object()]
    >>> b = Trajectory(task_id="t2", success=False, terminated_by="max_turns")
    >>> b.steps = [object()]
    >>> m = aggregate([a, b], label="demo")
    >>> m.num_trajectories, m.num_success, m.success_rate
    (2, 1, 0.5)
    >>> m.avg_turns
    1.5
    >>> m.avg_turns_to_success
    2.0
    """
    items = list(trajs)
    m = AgentMetrics(label=label, num_trajectories=len(items))
    if not items:
        return m

    terms: Counter[str] = Counter()
    for t in items:
        m.num_tool_calls += t.num_tool_calls
        m.num_failed_calls += t.num_failed_calls
        m.num_invalid_calls += t.num_invalid_calls
        m.turns_all.append(t.num_turns)
        if t.success:
            m.num_success += 1
            m.turns_success.append(t.num_turns)
        if t.fault_injected:
            m.num_faulted += 1
            if t.recovered_from_fault:
                m.num_recovered += 1
        terms[t.terminated_by or "(未标记)"] += 1
        m.wall_time_s += t.wall_time_s

    m.terminations = dict(terms)
    m.by_difficulty = _group_metrics(items, "difficulty")
    m.by_category = _group_metrics(items, "category")
    return m


#: 写报告时按这个顺序对照，避免把不同口径的指标混在一张表里
METRIC_GLOSSARY: dict[str, str] = {
    "success_rate": "整条轨迹达成期望动作的比例（agent 主指标）",
    "avg_turns": "平均轮数（效率）",
    "avg_turns_to_success": "成功轨迹的平均轮数",
    "tool_call_precision": "有效工具调用占比（1 − 无效/总数）",
    "fault_recovery_rate": "工具注入故障后最终成功的比例（鲁棒性主指标）",
    "give_up_rate": "模型给出最终回答而停止（放弃）的比例",
    "max_turns_rate": "撞到轮数上限的比例（卡住）",
    "error_rate": "循环内部异常的比例（应为 0）",
}
