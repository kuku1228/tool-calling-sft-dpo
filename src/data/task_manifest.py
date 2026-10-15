"""评测任务集：100 个航空客服多轮工具调用任务。

任务集结构
----------
===============  ====  ==================================================
类别              数量  考察点
===============  ====  ==================================================
flight_search     22   城市名→机场码、相对日期→绝对日期
booking_flow      18   多步：先搜再订，flight_id 必须来自 observation
baggage           15   订单号格式、数值参数
cancellation      12   写操作前的确认、原因参数
weather_advisory  12   参数语义区分（天气用中文城市名，航班用三字码）
multi_tool         9   一轮内串起多个工具
recovery          12   用户纠错后的自我修正（Week 2 错误恢复偏好对的数据源）
===============  ====  ==================================================

难度与环境
----------
- ``easy``   单工具、参数直接可提取（或已给全）
- ``medium`` 需要一次推理变换（相对日期换算、城市名→三字码、从 observation 取 flight_id）
- ``hard``   多步依赖，或包含用户纠错轮

- ``visibility=seen``    5 个 epoch 训练时会用到（偏好数据构造源）
- ``visibility=unseen``  训练完全不接触，只用于评测 ——
  这是 ``pass^k`` 之外本项目的第二个泛化度量，对标项目的 uncovered/unseen 划分。

评测口径
--------
每次采样只对**目标轮**（``target_turn``）的 assistant 输出打分，
``target_turn_kind`` 说明这一轮是什么性质：

- ``initial``    首次用户请求后的第一轮
- ``correction`` 用户指出错误之后的一轮（考察错误恢复能力）

这样设计的原因：多轮全自动评测需要真实的用户模拟器（对标项目用了 72B），
在 1.5B 单模型 + 单卡 24G 的约束下不可行。把目标轮显式标注出来后，
评测变成「给定确定的多轮前缀，评这一轮」，既可控又保持了 pass^k 有意义。
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from typing import Any, Iterator

# ============================================================================
# 数据类
# ============================================================================


@dataclass
class Turn:
    """一个对话轮次。"""

    role: str  # user | assistant | tool
    content: str
    #: 工具轮的 observation（仅 role="tool" 时使用）
    observation: dict[str, Any] | None = None


@dataclass
class ExpectedCall:
    """目标轮期望的工具调用。

    Attributes:
        tool: 期望的工具名。若为 ``None`` 表示该轮不该调用工具
            （例如需要先向用户澄清信息），此时只要模型不幻觉工具即算对。
        args: 期望参数（直接匹配口径，必须全部命中且值等价）。
        args_subset: 「只要命中这些字段就算参数正确」的宽松口径。
            用于那些用户没给全信息、模型没义务猜对的字段。
            统计里同时报告严格与宽松两个准确率，主指标用宽松口径。
        note: 这个任务的考察意图，会写进诊断报告。
    """

    tool: str | None
    args: dict[str, Any] = field(default_factory=dict)
    args_subset: dict[str, Any] = field(default_factory=dict)
    note: str = ""

    @property
    def graded_args(self) -> dict[str, Any]:
        """主指标实际比对的参数字段。"""
        return self.args_subset or self.args

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class Task:
    """一个评测任务。"""

    task_id: str
    category: str
    difficulty: str  # easy | medium | hard
    visibility: str  # seen | unseen
    turns: list[Turn]
    #: 打分的目标轮在 ``turns`` 中的下标
    target_turn: int
    #: initial | correction
    target_turn_kind: str
    expected: ExpectedCall
    #: 纠错轮的上下文：模型在首轮犯的错是什么（仅 recovery 类任务有）
    mistake_hint: str = ""
    tags: tuple[str, ...] = ()

    # ---- 便捷视图 ----
    @property
    def target_prefix(self) -> list[Turn]:
        """目标轮之前的所有轮次（含目标轮的 user 输入）。"""
        return self.turns[: self.target_turn]

    @property
    def is_recovery(self) -> bool:
        return self.target_turn_kind == "correction"

    def to_messages(self) -> list[dict[str, str]]:
        """转成 chat 格式（不含目标 assistant 回复）。

        工具轮会被渲染成 user 角色的 observation 文本 —— 这是
        Hermes/Qwen 风格 tool calling 的常见做法：observation 作为
        新的 user message 注入，而不是单独的 tool role。
        理由：1.5B 模型 + 简单 chat template 下，tool role 的
        训练/推理一致性更难保证。
        """
        msgs: list[dict[str, str]] = []
        for t in self.target_prefix:
            if t.role == "tool":
                content = t.content
                if t.observation is not None:
                    content = f"<observation>{json.dumps(t.observation, ensure_ascii=False, sort_keys=True)}</observation>"
                msgs.append({"role": "user", "content": content})
            else:
                msgs.append({"role": t.role, "content": t.content})
        return msgs

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "category": self.category,
            "difficulty": self.difficulty,
            "visibility": self.visibility,
            "target_turn": self.target_turn,
            "target_turn_kind": self.target_turn_kind,
            "mistake_hint": self.mistake_hint,
            "tags": list(self.tags),
            "expected": self.expected.to_dict(),
            "turns": [
                {"role": t.role, "content": t.content, "observation": t.observation}
                for t in self.turns
            ],
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Task:
        return Task(
            task_id=d["task_id"],
            category=d["category"],
            difficulty=d["difficulty"],
            visibility=d["visibility"],
            turns=[Turn(t["role"], t["content"], t.get("observation")) for t in d["turns"]],
            target_turn=d["target_turn"],
            target_turn_kind=d["target_turn_kind"],
            expected=ExpectedCall(**d["expected"]),
            mistake_hint=d.get("mistake_hint", ""),
            tags=tuple(d.get("tags", ())),
        )


# ============================================================================
# 构造用的素材池
# ============================================================================

_CITY_PAIRS: list[tuple[str, str]] = [
    ("北京", "上海"),
    ("上海", "广州"),
    ("深圳", "成都"),
    ("杭州", "西安"),
    ("重庆", "昆明"),
    ("厦门", "南京"),
    ("武汉", "青岛"),
    ("长沙", "三亚"),
    ("哈尔滨", "天津"),
    ("乌鲁木齐", "拉萨"),
]

#: 固定基准日期。用固定日期而非 today()，保证任务集在任何时候跑
#: 都得到同一份 ground truth —— 否则 pass^k 的绝对值无法跨天比较。
BASE_DATE = date(2025, 7, 14)  # 周一

_ABSOLUTE_DATES: list[str] = [
    "2025-07-15",
    "2025-07-18",
    "2025-07-21",
    "2025-07-24",
    "2025-07-28",
    "2025-08-01",
    "2025-08-05",
    "2025-08-09",
]

#: 相对日期表达 → 相对 BASE_DATE 的偏移天数
_RELATIVE_DATES: list[tuple[str, int]] = [
    ("明天", 1),
    ("后天", 2),
    ("大后天", 3),
    ("本周五", 4),      # BASE_DATE 是周一 → 周五 = +4
    ("下周一", 7),
    ("下周三", 9),
    ("这周日", 6),
    ("三天后", 3),
]

_PASSENGERS: list[str] = [
    "张伟",
    "李娜",
    "王芳",
    "刘洋",
    "陈晓明",
    "赵敏",
    "周杰",
    "吴桐",
    "郑爽",
    "孙悦",
    "徐磊",
    "马晓东",
]

_BOOKING_IDS: list[str] = [
    "BK7F2A9C",
    "BK3D8E1B",
    "BK9A4C6F",
    "BK2B7D5E",
    "BK8C1F3A",
    "BK5E9B2D",
    "BK1A6C8F",
    "BK4F3D7B",
]

_CANCEL_REASONS: list[str] = [
    "行程变更",
    "身体不适",
    "会议取消",
    "找到更合适的航班",
    "天气原因",
]

_WEATHER_CITIES: list[str] = ["北京", "上海", "广州", "成都", "三亚", "哈尔滨"]

#: 城市名 → 三字码，从 tool_catalog 引用避免两处维护
from src.env.tool_catalog import AIRPORT_CODES  # noqa: E402


# ============================================================================
# 难度标定表（⚠️ 由实测数据得出，不是主观判断）
# ============================================================================
#: 类别 → 难度。
#:
#: **这张表的由来：** 各任务生成函数原先按「任务表面复杂度」
#: 主观标注难度：以为「相对日期换算 + 城市名→三字码」是中等难度，
#: 于是把 ``flight_search`` 的 14 个任务标成 ``easy``。
#:
#: 真实基线实测（Qwen2.5-1.5B，100 任务 × 4 采样）结果：
#:
#: ====================  =========  ============
#: 类别                   实测 acc   原标签
#: ====================  =========  ============
#: weather_advisory       100.0%     easy
#: baggage                 96.7%     easy
#: recovery                62.5%     **hard**  ← 差 30 个百分点
#: cancellation            35.4%     medium
#: multi_tool              33.3%     hard
#: flight_search           26.1%     **easy**  ← 差 60+ 个百分点
#: booking_flow            18.1%     **easy**  ← 差 60+ 个百分点
#: ====================  =========  ============
#:
#: 后果：分组指标出现「medium 18% 比 hard 58% 还难」这种自相矛盾的结论，
#: 使「按难度看提升」这个分析维度完全失效。
#:
#: 现在的标定规则（按实测准确率分档）：``easy ≥ 90%``、
#: ``medium 45%~90%``、``hard < 45%``。
#:
#: ⚠️ **这个标定是模型相关的**：它描述的是「对 Qwen2.5-1.5B 基线而言的难度」，
#: 不是任务的固有属性。如果换用 7B 模型重跑基线，这张表应当重新标定。
#: 报告里引用按难度的分组数据时，需要说明这一点。
CATEGORY_DIFFICULTY: dict[str, str] = {
    "weather_advisory": "easy",    # 100.0%
    "baggage": "easy",             #  96.7%
    "recovery": "medium",          #  62.5%
    "cancellation": "hard",        #  35.4%
    "multi_tool": "hard",          #  33.3%
    "flight_search": "hard",       #  26.1%
    "booking_flow": "hard",        #  18.1%
    # 由纠错语料转成的训练任务（不进评测集，给个中性值）
    "trainsyn": "medium",
}
for _cat in ("trainsyn_format_error", "trainsyn_hallucination", "trainsyn_param_error",
             "trainsyn_wrong_tool", "trainsyn_no_recovery"):
    CATEGORY_DIFFICULTY[_cat] = "medium"


def _code(city: str) -> str:
    return AIRPORT_CODES[city]


def _absolute_date_str(offset_days: int) -> str:
    return (BASE_DATE + timedelta(days=offset_days)).isoformat()


# ============================================================================
# 问句模板
# ============================================================================
# 说明：模板的作用不是「造数据」，而是控制提问的自然度与信息完整度。
# 任务集本身是按类别手工设计的，模板只负责把同一意图换成不同说法，
# 避免模型过拟合到某一句话的措辞。

_SEARCH_TEMPLATES: list[str] = [
    "帮我查一下{date}从{origin}到{dest}的航班",
    "{date}{origin}飞{dest}有什么航班？",
    "我想{date}去{dest}，从{origin}出发，看看有哪些航班",
    "请搜索{date} {origin}到{dest}的机票",
    "{date}我要从{origin}去{dest}，帮我看看航班",
]

_BOOK_TEMPLATES: list[str] = [
    "帮我订一张{date} {origin}到{dest}的机票，乘客是{name}，经济舱就行",
    "我要给{name}订{date}从{origin}到{dest}的票",
    "{name}需要{date}从{origin}飞{dest}，直接帮我订上",
]

_BAGGAGE_TEMPLATES: list[str] = [
    "帮我查一下订单{bid}的行李额度",
    "订单{bid}还能免费托运多少公斤？",
    "我想知道{bid}这个订单的行李限额",
]

_MODIFY_TEMPLATES: list[str] = [
    "把订单{bid}的行李额度改成{kg}公斤",
    "订单{bid}我要加到{kg}公斤行李，可以吗",
    "{bid}的行李额度改到{kg}公斤",
]

_CANCEL_TEMPLATES: list[str] = [
    "我要取消订单{bid}，原因是{reason}",
    "帮我把{bid}这个订单退掉，{reason}",
    "取消{bid}，因为{reason}",
]

_WEATHER_TEMPLATES: list[str] = [
    "{city}今天的天气怎么样？",
    "帮我看看{city}今天会不会下雨",
    "查一下{city}的天气",

]

_TOOL_RESULT_STUB = {
    "ok": True,
    "note": "（历史 observation，用于给模型提供可引用的实体）",
}


# ============================================================================
# 任务生成器
# ============================================================================


class _DuplicateTask(Exception):
    """``_Builder.add`` 在去重守卫打开时遇到重复首轮文本会抛这个。

    它是**控制流信号**而不是错误：调用方（扩集生成的 `_fill`）捕获后
    跳过该候选、换下一个 index 继续试。用它比返回布尔值更省事，
    因为生成器内部可能是一条调用链（例如 cancel 的 clarify 分支）。
    """


class _Builder:
    """顺序构造任务，自动分配 task_id 与 seen/unseen 划分。

    ``enable_dedup()`` 打开后，``add()`` 会拒绝首轮文本已存在的任务 ——
    扩集时的唯一性保障（见 ``add`` 里的说明）。默认关闭，因为冻结的
    100 个任务内部本身有 13 处重复文本（weather 6 个文本覆盖 12 个任务等）。
    """

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)
        self.tasks: list[Task] = []
        self._counter = 0
        #: 去重守卫的开关与已见文本集合
        self._dedup = False
        self._seen_texts: set[str] = set()

    def enable_dedup(self) -> None:
        """打开去重守卫，并把当前已有的文本登记为「已见」。

        这样扩集时新增任务不会与已生成的基础集重复。"""
        self._dedup = True
        self._seen_texts = {x.turns[0].content for x in self.tasks if x.turns}

    def add(
        self,
        category: str,
        difficulty: str,
        turns: list[Turn],
        target_turn: int,
        target_turn_kind: str,
        expected: ExpectedCall,
        mistake_hint: str = "",
        tags: tuple[str, ...] = (),
    ) -> Task:
        # ------------------------------------------------------------------
        # 可选去重守卫：拒绝**首轮文本已存在**的任务
        # ------------------------------------------------------------------
        #
        # 为什么需要它：扩集时「同一段文本不得出现两次」是硬要求（重复任务会让
        # bootstrap CI 虚假压窄 —— 相关样本被当成独立样本）。但各生成器各有
        # 各的 index 公式，逐个改公式去躲碰撞**极易漏**：
        #
        #   实测连续漏过 8 处 → 7 处 → 2 处 → 3 处（各出现 5 次）。
        #   每一轮我都是修「上一轮发现的那一类」，然后又冒出新的。
        #
        # 根因是「用 index 公式去保证文本唯一」这个思路本身不可靠：
        # 多个生成器共享同一个计数器与实体池，任何一处公式微调都会
        # 与别处产生新的周期对齐。
        #
        # 所以把唯一性检查放到**唯一的汇聚点**（这里），用文本本身判定，
        # 与 index 公式解耦。守卫默认关闭（基础 100 个内部本来有重复），
        # 扩集时显式打开（当前 v3 任务集不使用该路径）。
        if self._dedup and turns:
            _t0 = turns[0].content
            if _t0 in self._seen_texts:
                raise _DuplicateTask(_t0)
            self._seen_texts.add(_t0)

        self._counter += 1
        tid = f"task_{self._counter:04d}"
        # 划分策略：按类别轮转，保证每个类别在 seen/unseen 里都有分布，
        # 而不是「前 70 个 seen、后 30 个 unseen」这种会引入类别偏置的切法。
        visibility = "unseen" if self._counter % 3 == 0 else "seen"
        # ⚠️ 难度**统一由类别查表决定**（见 CATEGORY_DIFFICULTY），
        # 忽略调用方传入的 difficulty。原因见那张表的说明：
        # 各生成函数原先按「任务表面复杂度」主观标注，实测与真实难度严重不符。
        difficulty = CATEGORY_DIFFICULTY.get(category, difficulty)
        task = Task(
            task_id=tid,
            category=category,
            difficulty=difficulty,
            visibility=visibility,
            turns=turns,
            target_turn=target_turn,
            target_turn_kind=target_turn_kind,
            expected=expected,
            mistake_hint=mistake_hint,
            tags=tags,
        )
        self.tasks.append(task)
        return task


def _make_search_task(b: _Builder, idx: int, *, relative: bool, offset: int = 0,
                     templates=None, dates=None, rel_dates=None,
                     pairs=None) -> None:
    templates = templates if templates is not None else _SEARCH_TEMPLATES
    dates = dates if dates is not None else _ABSOLUTE_DATES
    rel_dates = rel_dates if rel_dates is not None else _RELATIVE_DATES
    pairs = pairs if pairs is not None else _CITY_PAIRS
    origin, dest = pairs[idx % len(pairs)]
    if relative:
        rel_expr, off = rel_dates[(idx + offset) % len(rel_dates)]
        date_expr = rel_expr
        date_value = _absolute_date_str(off)
        difficulty = "medium"
        tags = ("relative_date", "date_conversion")
    else:
        date_value = dates[(idx + offset) % len(dates)]
        date_expr = date_value
        difficulty = "easy"
        tags = ("absolute_date",)

    tmpl = templates[(idx + offset) % len(templates)]
    user_msg = tmpl.format(date=date_expr, origin=origin, dest=dest)
    b.add(
        category="flight_search",
        difficulty=difficulty,
        turns=[Turn("user", user_msg)],
        target_turn=1,
        target_turn_kind="initial",
        expected=ExpectedCall(
            tool="search_flights",
            args={
                "origin": _code(origin),
                "destination": _code(dest),
                "date": date_value,
            },
            note="城市名→三字码；相对日期→绝对日期" if relative else "参数直接提取",
        ),
        tags=tags,
    )


def _make_booking_task(b: _Builder, idx: int, *, two_step: bool, offset: int = 0,
                      templates=None, dates=None, names=None,
                      pairs=None) -> None:
    templates = templates if templates is not None else _BOOK_TEMPLATES
    dates = dates if dates is not None else _ABSOLUTE_DATES
    names = names if names is not None else _PASSENGERS
    pairs = pairs if pairs is not None else _CITY_PAIRS
    origin, dest = pairs[idx % len(pairs)]
    date_value = dates[(idx + 3 + offset) % len(dates)]
    name = names[(idx + offset) % len(names)]
    flight_id = f"MU{1000 + idx * 37 % 9000}"

    tmpl = templates[(idx + offset) % len(templates)]
    user_msg = tmpl.format(date=date_value, origin=origin, dest=dest, name=name)

    if not two_step:
        # 单步：上下文已给出 flight_id，模型应直接 book
        turns = [
            Turn("user", user_msg),
            Turn(
                "tool",
                "",
                observation={
                    "ok": True,
                    "tool": "search_flights",
                    "result": {
                        "count": 2,
                        "flights": [
                            {"flight_id": flight_id, "price_cny": 1280, "depart_time": "08:20"},
                            {"flight_id": f"{flight_id[:2]}{int(flight_id[2:]) + 5}", "price_cny": 1620, "depart_time": "19:40"},
                        ],
                    },
                },
            ),
            Turn("user", f"就订 {flight_id} 这一班吧"),
        ]
        b.add(
            category="booking_flow",
            difficulty="medium",
            turns=turns,
            target_turn=3,
            target_turn_kind="initial",
            expected=ExpectedCall(
                tool="book_flight",
                args={"flight_id": flight_id, "passenger_name": name},
                note="flight_id 必须取自 observation，不能编造",
            ),
            tags=("from_observation",),
        )
    else:
        # 两步：先 search，observation 是空的（评测第一轮只要求 search）
        turns = [Turn("user", user_msg)]
        b.add(
            category="booking_flow",
            difficulty="easy",
            turns=turns,
            target_turn=1,
            target_turn_kind="initial",
            expected=ExpectedCall(
                tool="search_flights",
                args={"origin": _code(origin), "destination": _code(dest), "date": date_value},
                note="订票前置步骤：必须先搜索",
            ),
            tags=("multi_step_prefix",),
        )


def _make_baggage_task(b: _Builder, idx: int, *, modify: bool, offset: int = 0,
                      check_templates=None, modify_templates=None,
                      bids=None) -> None:
    check_templates = (check_templates if check_templates is not None
                       else _BAGGAGE_TEMPLATES)
    modify_templates = (modify_templates if modify_templates is not None
                        else _MODIFY_TEMPLATES)
    bids = bids if bids is not None else _BOOKING_IDS
    bid = bids[(idx + offset) % len(bids)]
    if modify:
        kg = [25, 30, 35, 40, 45][(idx + offset) % 5]
        tmpl = modify_templates[(idx + offset) % len(modify_templates)]
        user_msg = tmpl.format(bid=bid, kg=kg)
        expected = ExpectedCall(
            tool="modify_baggage",
            args={"booking_id": bid, "new_baggage_kg": kg},
            note="写操作 + 数值参数类型",
        )
        tags = ("write_tool", "numeric_param")
    else:
        tmpl = check_templates[(idx + offset) % len(check_templates)]
        user_msg = tmpl.format(bid=bid)
        expected = ExpectedCall(
            tool="check_baggage",
            args={"booking_id": bid},
            note="订单号格式必须匹配 ^BK[A-Z0-9]{6}$",
        )
        tags = ("id_format",)

    b.add(
        category="baggage",
        difficulty="easy" if not modify else "medium",
        turns=[Turn("user", user_msg)],
        target_turn=1,
        target_turn_kind="initial",
        expected=expected,
        tags=tags,
    )


def _make_cancel_task(b: _Builder, idx: int, *, with_reason: bool, offset: int = 0,
                     templates=None, reasons=None, bids=None) -> None:
    templates = templates if templates is not None else _CANCEL_TEMPLATES
    reasons = reasons if reasons is not None else _CANCEL_REASONS
    bids = bids if bids is not None else _BOOKING_IDS
    bid = bids[(idx + 2 + offset) % len(bids)]
    reason = reasons[(idx + offset) % len(reasons)]

    if with_reason:
        tmpl = templates[(idx + offset) % len(templates)]
        user_msg = tmpl.format(bid=bid, reason=reason)
        # 用户已给原因 → 模型应直接取消
        turns = [
            Turn("user", user_msg),
            Turn("assistant", f"好的，我确认一下：您要取消订单 {bid}，原因是「{reason}」，对吗？"),
            Turn("user", "对，取消吧"),
        ]
        b.add(
            category="cancellation",
            difficulty="medium",
            turns=turns,
            target_turn=3,
            target_turn_kind="initial",
            expected=ExpectedCall(
                tool="cancel_booking",
                args={"booking_id": bid},
                args_subset={"booking_id": bid},
                note="reason 由用户给出，模型只要正确转写；宽松口径只比对 booking_id",
            ),
            tags=("write_tool", "confirmation"),
        )
    else:
        # 用户没给原因且直接要求取消 → 应先询问，不该贸然调用
        #
        # ⚠️ 「需澄清」分支的文本是 f"我不想要{bid}这个订单了" —— **不含模板、
        # 也不含 reason**，所以它的唯一性完全取决于 bid。
        #
        # ⚠️⚠️ 但**基础集（offset==0）必须保持原来的 bid 计算方式**，
        # 否则会改动冻结的 task_0065..0067 —— 我第一版改动正是如此，
        # 直接把三个冻结任务改掉了（测试立刻抓到）。
        # 冻结集用的是 ``bids[(idx + 2) % len]``（idx 步长为 1，不重复）。
        #
        # 只有新增任务（offset > 0）才需要打散：那里 idx 以 3 为步长，
        # ``(idx + 2 + offset)`` 的周期只有 ``len/gcd(3,len)`` ——
        # 36 个订单号下周期仅 18，15 个澄清任务只用到 6 个订单号，
        # 产生 9 处重复。用与池长互质的步长即可打散。
        if offset == 0:
            bid = bids[(idx + 2) % len(bids)]
        else:
            # ⚠️ 新增任务里 idx 与 offset **同步增长**（调用处传的就是 o=off+j），
            # 所以任何同时含 idx 与 offset 的线性式都会退化成「只随 j 变化」，
            # 周期被压缩。实测 (7*idx + 2 + offset) % 36 的周期只有 12，
            # 仍然产生 3 个重复文本（各出现 5 次）。
            # 只用 idx、且步长与池长互质，才能保证 36 的满周期。
            _n = len(bids)
            _step = 7 if _n % 7 else 5
            bid = bids[(_step * idx + 2) % _n]
        user_msg = f"我不想要{bid}这个订单了"
        b.add(
            category="cancellation",
            difficulty="hard",
            turns=[Turn("user", user_msg)],
            target_turn=1,
            target_turn_kind="initial",
            expected=ExpectedCall(
                tool=None,
                args={},
                note="信息不全时应先向用户追问原因，而不是直接调用写操作",
            ),
            tags=("clarify_first", "write_tool"),
        )


def _make_weather_task(b: _Builder, idx: int, *, offset: int = 0,
                      templates=None, cities=None) -> None:
    templates = templates if templates is not None else _WEATHER_TEMPLATES
    cities = cities if cities is not None else _WEATHER_CITIES
    city = cities[(idx + offset) % len(cities)]
    tmpl = templates[(idx + offset) % len(templates)]
    user_msg = tmpl.format(city=city)
    b.add(
        category="weather_advisory",
        difficulty="easy",
        turns=[Turn("user", user_msg)],
        target_turn=1,
        target_turn_kind="initial",
        expected=ExpectedCall(
            tool="get_weather",
            args={"city": city},
            note="注意：这个工具要中文城市名，不是三字码 —— 与其他工具相反",
        ),
        tags=("param_semantics",),
    )


#: multi_tool 的首轮句式。**必须有多样性**：早期版本只有一种句式，
#: 44 个新增任务的首轮全部以日期开头（前 6 字骨架只有 1 种），
#: 会让人怀疑模型是在匹配固定句式而不是在理解意图。
#: 基础集（前 100 个）用的**唯一**句式 —— 必须与历史版本逐字相同。
#: ⚠️ 我一度把下面 8 种句式直接设为默认值，结果**改掉了基础 9 个
#: multi_tool 任务的文本**（task_0081..0089），静默作废了历史结果。
#: 冻结集的默认值永远只能是**原来那一个**。
_MULTI_TOOL_BASE_TEMPLATE: str = "{date}我要从{origin}去{dest}，顺便帮我看看{origin}当天的天气"

#: 新增任务用的扩展句式。**不含**基础句式 —— 基础任务已经用过它，
#: 若扩展里也含，会立刻产生与冻结集完全相同的文本（去重会丢弃，
#: 白白浪费候选名额）。
_MULTI_TOOL_TEMPLATES: tuple[str, ...] = (
    "帮我查{date}{origin}到{dest}的航班，另外{origin}那天天气怎么样",
    "{date}去{dest}，从{origin}走；先看看{origin}天气如何",
    "我要{date}从{origin}飞{dest}，还想知道{origin}当天会不会下雨",
    "{origin}到{dest} {date}的航班帮我查一下，顺便看看{origin}的天气",
    "{date}从{origin}到{dest}，出发前想先了解{origin}的天气",
    "帮我安排{date}{origin}去{dest}，另外{origin}那天冷不冷",
    "{date}{origin}前往{dest}，同时查一下{origin}的天气情况",
)


def _make_multi_tool_task(b: _Builder, idx: int, *, offset: int = 0,
                          pairs=None, dates=None, templates=None) -> None:
    """一轮里要求连续调用两个工具，考察是否会把两个工具的 schema 弄混。"""
    pairs = pairs if pairs is not None else _CITY_PAIRS
    dates = dates if dates is not None else _ABSOLUTE_DATES
    templates = templates if templates is not None else _MULTI_TOOL_TEMPLATES
    origin, dest = pairs[(idx + 4 + offset) % len(pairs)]
    date_value = dates[(idx + 5 + offset) % len(dates)]
    city = origin
    # offset == 0 时用**基础句式**，保证冻结集逐字不变；
    # offset > 0（新增任务）才用扩展句式。
    tmpl = (_MULTI_TOOL_BASE_TEMPLATE if offset == 0
            else templates[(idx + offset) % len(templates)])

    turns = [
        Turn("user", tmpl.format(date=date_value, origin=origin, dest=dest)),
        Turn(
            "tool",
            "",
            observation={
                "ok": True,
                "tool": "get_weather",
                "result": {"city": city, "condition": "小雨", "temp_c": 26, "flight_impact": "可能出现延误"},
            },
        ),
        Turn("user", "天气我看到了，现在把航班查出来吧"),
    ]
    b.add(
        category="multi_tool",
        difficulty="hard",
        turns=turns,
        target_turn=3,
        target_turn_kind="initial",
        expected=ExpectedCall(
            tool="search_flights",
            args={"origin": _code(origin), "destination": _code(dest), "date": date_value},
            note="上下文已出现天气查询，模型需切换到航班工具",
        ),
        tags=("multi_tool", "context_switch"),
    )


# ---------------------------------------------------------------------------
# recovery 类任务的错误模式定义
# ---------------------------------------------------------------------------
# 每个模式描述「首轮模型会犯的错」以及「纠错轮的正确目标」。
# 这些模式直接对应 docs/diagnosis/baseline_diagnosis.md 里的错误分类，
# 也是 Week 2 错误恢复偏好对的数据骨架。

@dataclass(frozen=True)
class _RecoveryPattern:
    key: str
    #: 这个模式考察的错误族，**取值必须是 ``src/eval/error_taxonomy.py``
    #: 里的类别常量**（FORMAT_ERROR / HALLUCINATION / WRONG_TOOL /
    #: PARAM_ERROR / NO_RECOVERY）。
    #:
    #: 该字段会被写进任务的 ``tags``（见 ``_make_recovery_task``），
    #: 用于按错误族聚合分析。早期版本用的是小写短名
    #: （``"format"`` / ``"param_error"``），与 error_taxonomy 的常量
    #: 不一致，导致按错误族聚合时必须再做一次映射 —— 是个静默失配源。
    error_family: str
    #: 首轮用户请求
    first_user: str
    #: 生成首轮错误回复（模型的错），占位符已被填好
    wrong_assistant: str
    #: 用户的纠错话术
    correction_user: str
    #: 纠错轮的期望调用
    expected: ExpectedCall
    #: 这个模式考察什么
    note: str


def _recovery_patterns() -> list[_RecoveryPattern]:
    bid = _BOOKING_IDS[0]
    origin, dest = "北京", "上海"
    date_value = "2025-07-18"
    name = _PASSENGERS[0]

    return [
        _RecoveryPattern(
            key="format_ignore_correction",
            error_family="FORMAT_ERROR",
            first_user=f"查一下{date_value}{origin}到{dest}的航班",
            wrong_assistant="好的，我来帮您查询一下北京到上海的航班，请稍等。",
            correction_user="你没有真的调用工具，请用规定的 <tool_call> 格式重新输出。",
            expected=ExpectedCall(
                tool="search_flights",
                args={"origin": _code(origin), "destination": _code(dest), "date": date_value},
                note="用户指出格式问题后能否改成合法 tool_call",
            ),
            note="格式错误 → 纠错后修正",
        ),
        _RecoveryPattern(
            key="hallucinated_tool_corrected",
            error_family="HALLUCINATION",
            first_user=f"查一下我的订单{bid}还能托运多少行李",
            wrong_assistant=(
                '<tool_call>{"name": "get_order_details", "arguments": {"order_id": "'
                + bid
                + '"}}</tool_call>'
            ),
            correction_user=(
                "工具列表里没有 get_order_details 这个工具，请用正确的工具重新调用。"
            ),
            expected=ExpectedCall(
                tool="check_baggage",
                args={"booking_id": bid},
                note="用户指出工具幻觉后能否换成白名单里的工具",
            ),
            note="工具幻觉 → 纠错后修正",
        ),
        _RecoveryPattern(
            key="wrong_tool_corrected",
            error_family="WRONG_TOOL",
            first_user=f"把订单{bid}的行李额度改成 30 公斤",
            wrong_assistant=(
                '<tool_call>{"name": "check_baggage", "arguments": {"booking_id": "'
                + bid
                + '"}}</tool_call>'
            ),
            correction_user="我不是要查询，我是要改成 30 公斤，请用修改额度的工具。",
            expected=ExpectedCall(
                tool="modify_baggage",
                args={"booking_id": bid, "new_baggage_kg": 30},
                note="用户指出选错工具后能否换到 modify_baggage",
            ),
            note="工具选错 → 纠错后修正",
        ),
        _RecoveryPattern(
            key="param_placeholder_corrected",
            error_family="PARAM_ERROR",
            first_user=f"帮我查一下订单{bid}的行李额度",
            wrong_assistant=(
                '<tool_call>{"name": "check_baggage", "arguments": {"booking_id": "previous_booking"}}</tool_call>'
            ),
            correction_user=f"订单号是 {bid}，不要填 previous_booking。",
            expected=ExpectedCall(
                tool="check_baggage",
                args={"booking_id": bid},
                note="占位符参数被指出后能否填回真实 ID",
            ),
            note="参数占位符 → 纠错后修正",
        ),
        _RecoveryPattern(
            key="city_code_corrected",
            error_family="PARAM_ERROR",
            first_user=f"{date_value}从{origin}飞{dest}有哪些航班",
            wrong_assistant=(
                '<tool_call>{"name": "search_flights", "arguments": {"origin": "'
                + origin
                + '", "destination": "'
                + dest
                + '", "date": "'
                + date_value
                + '"}}</tool_call>'
            ),
            correction_user="origin 和 destination 要填机场三字码，不要填中文城市名。",
            expected=ExpectedCall(
                tool="search_flights",
                args={"origin": _code(origin), "destination": _code(dest), "date": date_value},
                note="参数格式被指出后能否改成三字码",
            ),
            note="参数格式错误 → 纠错后修正",
        ),
        _RecoveryPattern(
            key="repeated_same_error",
            error_family="NO_RECOVERY",
            first_user=f"给{name}订{date_value}从{origin}到{dest}的机票",
            wrong_assistant=(
                '<tool_call>{"name": "book_flight", "arguments": {"flight_id": "CA0000", '
                '"passenger_name": "' + name + '"}}</tool_call>'
            ),
            correction_user="CA0000 这个航班号不存在，你必须先搜索航班再下单。",
            expected=ExpectedCall(
                tool="search_flights",
                args={"origin": _code(origin), "destination": _code(dest), "date": date_value},
                note="被指出编造 flight_id 后，能否回到「先搜索」这一步",
            ),
            note="编造实体 → 纠错后回到正确前置步骤",
        ),
        _RecoveryPattern(
            key="missing_reason_corrected",
            error_family="PARAM_ERROR",
            first_user=f"帮我取消订单{bid}",
            wrong_assistant=(
                '<tool_call>{"name": "cancel_booking", "arguments": {"booking_id": "'
                + bid
                + '", "reason": "用户要求"}}</tool_call>'
            ),
            correction_user="取消原因不能写「用户要求」，请先问清楚我为什么取消。",
            expected=ExpectedCall(
                tool=None,
                args={},
                note="被指出原因编造后，能否改为先追问而不是再次硬填",
            ),
            note="参数编造 → 纠错后改为澄清",
        ),
        _RecoveryPattern(
            key="wrong_date_corrected",
            error_family="PARAM_ERROR",
            first_user="后天从广州到成都的航班有哪些",
            wrong_assistant=(
                '<tool_call>{"name": "search_flights", "arguments": {"origin": "CAN", '
                '"destination": "CTU", "date": "后天"}}</tool_call>'
            ),
            correction_user="date 必须是 YYYY-MM-DD 格式的绝对日期，不能写「后天」。",
            expected=ExpectedCall(
                tool="search_flights",
                args={
                    "origin": "CAN",
                    "destination": "CTU",
                    "date": _absolute_date_str(2),
                },
                note="相对日期被指出后能否换算成绝对日期",
            ),
            note="相对日期未换算 → 纠错后修正",
        ),
        _RecoveryPattern(
            key="overbook_corrected",
            error_family="WRONG_TOOL",
            first_user="深圳到成都 2025-07-21 的航班，直接订最便宜那班，乘客刘洋",
            wrong_assistant=(
                '<tool_call>{"name": "search_flights", "arguments": {"origin": "SZX", '
                '"destination": "CTU", "date": "2025-07-21", "max_price": 500}}</tool_call>'
            ),
            correction_user="你搜了航班但没下单，而且不该加 max_price。请完成订票。",
            expected=ExpectedCall(
                tool="book_flight",
                args={"passenger_name": "刘洋"},
                args_subset={"passenger_name": "刘洋"},
                # 这里 flight_id 未知（要从前一步 observation 取），
                # 所以宽松口径只检查 passenger_name 与工具名
                note="被指出停在中间步骤后，能否推进到写操作",
            ),
            note="步骤不完整 → 纠错后推进",
        ),
        _RecoveryPattern(
            key="wrong_city_param_corrected",
            error_family="PARAM_ERROR",
            first_user="杭州今天天气怎么样",
            wrong_assistant=(
                '<tool_call>{"name": "get_weather", "arguments": {"city": "HGH"}}</tool_call>'
            ),
            correction_user="city 参数要填中文城市名，不要填三字码。",
            expected=ExpectedCall(
                tool="get_weather",
                args={"city": "杭州"},
                note="参数语义被指出后能否改成中文城市名",
            ),
            note="参数语义错误 → 纠错后修正",
        ),
        _RecoveryPattern(
            key="hallucinated_weather_tool_corrected",
            error_family="HALLUCINATION",
            first_user="帮我看看三亚的天气",
            wrong_assistant=(
                '<tool_call>{"name": "get_travel_advisory", "arguments": {"city": "三亚"}}</tool_call>'
            ),
            correction_user="没有 get_travel_advisory 这个工具，请用正确的天气工具。",
            expected=ExpectedCall(
                tool="get_weather",
                args={"city": "三亚"},
                note="第二次幻觉检测：能否在纠错后落到真实工具",
            ),
            note="工具幻觉（第二次）→ 纠错后修正",
        ),
        _RecoveryPattern(
            key="format_json_broken_corrected",
            error_family="FORMAT_ERROR",
            first_user=f"查一下订单{bid}的行李额度",
            wrong_assistant='<tool_call>{"name": "check_baggage", "arguments": {booking_id: ' + bid + "}}</tool_call>",
            correction_user="你的 JSON 不合法（键没有引号），请输出合法 JSON。",
            expected=ExpectedCall(
                tool="check_baggage",
                args={"booking_id": bid},
                note="JSON 语法错误被指出后能否输出合法 JSON",
            ),
            note="JSON 语法错误 → 纠错后修正",
        ),
    ]


def _make_recovery_task(b: _Builder, idx: int, pattern: _RecoveryPattern) -> None:
    turns = [
        Turn("user", pattern.first_user),
        Turn("assistant", pattern.wrong_assistant),
        Turn("user", pattern.correction_user),
    ]
    b.add(
        category="recovery",
        difficulty="hard",
        turns=turns,
        target_turn=3,
        target_turn_kind="correction",
        expected=pattern.expected,
        mistake_hint=pattern.key,
        tags=("recovery", pattern.error_family, pattern.key),
    )


# ============================================================================
# 主入口
# ============================================================================



# ===========================================================================
# 扩展词汇表（v2 扩集用）
# ===========================================================================
#
# ⚠️ 为什么不能简单把 build_tasks 的循环次数 ×4：
#
# 各生成器用 ``idx % len(列表)`` 取模板，而现有 100 个任务**已经把这些列表
# 跑过了好几轮**（例如 weather 12 个任务只对应 6 个不同的 user 文本）。
# 单纯放大循环次数只会**重复已有任务**，那样：
#   * 不会带来新的信息量
#   * 任务之间高度相关，bootstrap CI 会被**虚假地压窄**
#     （这正是「同一任务的多次采样不能当独立样本」的同一条原则）
#
# 所以要扩集必须**真正扩大词汇表**。下面这些列表是原有列表的超集：
# 前 len(原列表) 项与原列表逐字相同，因此 offset=0 生成的结果**完全不变**，
# 新增任务落在后面的新组合上。
_SEARCH_TEMPLATES_V2: list[str] = _SEARCH_TEMPLATES + [
    '麻烦帮我看看{date}{origin}到{dest}有没有航班',
    '{date}想去{dest}，从{origin}走，有哪些航班可选',
    '查询{origin}到{dest} {date}的航班信息',
    '我{date}要从{origin}出发去{dest}，帮我找一下航班',
    '有没有{date}从{origin}飞往{dest}的航班？',
    '{date}{origin}—{dest}的航班有哪些',
    '帮我搜一下{date} {origin}出发到{dest}的飞机',
    '想订{date}{origin}去{dest}的票，先看看有什么航班',
    '麻烦查{date}从{origin}到{dest}的航班班次',
    '我需要在{date}从{origin}前往{dest}，有哪些航班',
    '{origin}到{dest}的机票，{date}的，帮我查查',
    '看看{date}{origin}飞{dest}的航班',
    '帮我找{date}从{origin}去{dest}的班机',
    '{date}的{origin}到{dest}航班麻烦查一下',
    '查一下{date} {origin}到{dest}有哪些班次',
]

_BOOK_TEMPLATES_V2: list[str] = _BOOK_TEMPLATES + [
    '帮我订{date}{origin}到{dest}的机票，乘客{name}',
    '{name}要去{dest}，{date}从{origin}出发，帮我订票',
    '订一张{date}{origin}飞{dest}的票，给{name}',
    '{name} {date}从{origin}到{dest}，直接订经济舱',
    '帮{name}订{date}{origin}到{dest}的机票',
    '{date}{origin}去{dest}，乘客是{name}，帮我下单',
    '给{name}订{date}{origin}到{dest}的航班',
    '我要订{date}{origin}飞{dest}的票，乘客{name}',
    '{name}的行程：{date}{origin}到{dest}，帮我订上',
    '帮我给{name}订一张{date}{origin}到{dest}的机票',
]

_BAGGAGE_TEMPLATES_V2: list[str] = _BAGGAGE_TEMPLATES + [
    '{bid}这个订单能免费托运多少行李？',
    '查一下{bid}的行李额度是多少',
    '订单{bid}的免费行李限额是多少公斤',
    '{bid}可以托运几公斤行李？',
    '帮我看看{bid}还能托运多少',
    '我想确认{bid}的行李规定',
    '{bid}的行李额度麻烦查一下',
    '订单号{bid}，查一下能带多少行李',
    '{bid}这个订单行李额度是多少',
]

_MODIFY_TEMPLATES_V2: list[str] = _MODIFY_TEMPLATES + [
    '把订单{bid}的行李额度改成{kg}公斤',
    '{bid}的行李额度帮我调到{kg}公斤',
    '订单{bid}要改成{kg}公斤行李',
    '帮我把{bid}的额度改为{kg}公斤',
    '{bid}的行李额改成{kg}公斤',
    '我想把{bid}的行李额度提到{kg}公斤',
]

_CANCEL_TEMPLATES_V2: list[str] = _CANCEL_TEMPLATES + [
    '请取消订单{bid}，{reason}',
    '订单{bid}帮我取消掉，{reason}',
    '{bid}这个订单退了吧，{reason}',
    '我要退掉{bid}，{reason}',
    '麻烦取消{bid}，{reason}',
    '{bid}取消一下，{reason}',
    '帮我把订单{bid}取消，{reason}',
    '{bid}不需要了，取消掉，{reason}',
    '{bid}留着也没用，帮我取消，{reason}',
    '麻烦帮我退订{bid}，{reason}',
    '{bid}这个订单我不想保留了，{reason}',
    '帮我把{bid}退了吧，{reason}',
    '取消掉订单{bid}，{reason}',
]

_WEATHER_TEMPLATES_V2: list[str] = _WEATHER_TEMPLATES + [
    '{city}的天气如何？',
    '{city}今天什么天气',
    '帮我查一下{city}今天的天气情况',
    '{city}今天热不热？',
    '{city}今天适合出门吗，天气怎么样',
    '我想知道{city}今天的天气',
    '{city}今天有雨吗？',
    '查一下{city}今天的天气状况',
]

#: 扩展后的日期词汇表（16 个；前 8 个与原列表逐字相同）
_ABSOLUTE_DATES_V2: list[str] = _ABSOLUTE_DATES + [
    '2025-08-12', '2025-08-15', '2025-08-19', '2025-08-22',
    '2025-08-26', '2025-09-02', '2025-09-05', '2025-09-09',
]

#: 扩展后的相对日期（前 8 个与原列表逐字相同）
_RELATIVE_DATES_V2: list[tuple[str, int]] = _RELATIVE_DATES + [
    ('下周五', 11), ('下周日', 13), ('两周后', 14), ('这周六', 5),
    ('这周四', 4), ('下周二', 8), ('下周四', 10), ('四天后', 4),
]

#: 扩展后的天气城市（全部 20 个城市；前 6 个与原列表逐字相同）
_WEATHER_CITIES_V2: list[str] = list(_WEATHER_CITIES) + [
    c for c in AIRPORT_CODES if c not in set(_WEATHER_CITIES)
]

#: 扩展后的乘客名（24 个；前 12 个与原列表逐字相同）
_PASSENGERS_V2: list[str] = list(_PASSENGERS) + [
    '许静', '邓超', '曾毅', '彭飞', '苏晴', '蒋磊',
    '韩雪', '唐宇', '曹阳', '袁莉', '雷军', '谢娜',
]

#: 扩展后的订单号（40 个；前 8 个与原列表逐字相同）
_BOOKING_IDS_V2: list[str] = list(_BOOKING_IDS) + [
    'BK%02X%02X%02X' % (i * 7 + 13, i * 11 + 3, i * 19 + 17) for i in range(8, 40)
]

#: 扩展后的取消原因（12 个；前 5 个与原列表逐字相同）
_CANCEL_REASONS_V2: list[str] = list(_CANCEL_REASONS) + [
    '临时有事去不了', '找到更便宜的航班了', '个人原因',
    '公司临时安排出差', '想改到其他日期', '家里有急事', '签证没办下来',
]

def build_tasks(seed: int = 42) -> list[Task]:
    """按固定配方生成全部任务（确定性，可复现）。

    配方（合计 100）::

        flight_search     22   （14 绝对日期 easy + 8 相对日期 medium）
        booking_flow      18   （10 单步 search + 8 两步 book）
        baggage           15   （9 check + 6 modify）
        cancellation      12   （8 带原因 + 4 需澄清）
        weather_advisory  12
        multi_tool         9
        recovery          12

    >>> tasks = build_tasks()
    >>> len(tasks)
    100
    >>> sorted({t.category for t in tasks})
    ['baggage', 'booking_flow', 'cancellation', 'flight_search', 'multi_tool', 'recovery', 'weather_advisory']
    """
    b = _Builder(seed=seed)

    for i in range(22):
        _make_search_task(b, i, relative=(i >= 14))

    for i in range(18):
        _make_booking_task(b, i, two_step=(i >= 10))

    for i in range(15):
        _make_baggage_task(b, i, modify=(i >= 9))

    for i in range(12):
        _make_cancel_task(b, i, with_reason=(i < 8))

    for i in range(12):
        _make_weather_task(b, i)

    for i in range(9):
        _make_multi_tool_task(b, i)

    patterns = _recovery_patterns()
    for i in range(12):
        _make_recovery_task(b, i, patterns[i % len(patterns)])

    assert len(b.tasks) == 100, f"任务数应为 100，实际 {len(b.tasks)}"
    return b.tasks



_CACHE: list[Task] | None = None


def load_tasks(seed: int = 42, use_cache: bool = True) -> list[Task]:
    """加载 v1 任务集（100 个，已冻结）。

    ⚠️ 这是**历史任务集**：baseline / sft_v2 / sft_v3 / dpo_* 的结果都跑在它上面。
    它内部有 13 处重复文本、日期全在过去、``multi_tool`` 名不符实 ——
    已知缺陷见 ``src/data/task_manifest_v2.py`` 的模块文档。
    新实验请用 ``load_tasks_by_version("v3")``。

    >>> len(load_tasks())
    100
    """
    global _CACHE
    if _CACHE is None or not use_cache:
        _CACHE = build_tasks(seed=seed)
    return _CACHE


#: 评测协议版本 → 任务集
#:
#: ⚠️ 这里只有 **v1**（旧 100 任务）与 **v3**（新高质量 100 任务）——
#: 中间那个「400 任务扩集」（v2）已被**否决并删除**，原因见
#: ``tests/test_methodology.py::test_deprecated_400_task_set_is_gone``：
#: 实测 400 个任务只有 208 种骨架、8 种签名，``multi_tool`` 更是
#: 53 个任务 8 种骨架 —— 有效独立样本不是 400，那个 CI 是虚假精度。
TASK_SET_SIZES: dict[str, int] = {"v1": 100, "v3": 106}


def load_tasks_by_version(version: str, seed: int = 42) -> list[Task]:
    """按协议版本取任务集。

    * ``v1`` = 旧 100 任务（历史结果跑在它上面，保留以便对照）
    * ``v3`` = 新高质量 100 任务（实体唯一、日期未来、骨架不重复）

    >>> len(load_tasks_by_version("v1"))
    100
    >>> # v3 是 106 个（为了让「应先追问」样本达 14 个，放弃了整 100）
    >>> len(load_tasks_by_version("v3"))
    106
    >>> load_tasks_by_version("v2")
    Traceback (most recent call last):
        ...
    ValueError: 未知的评测协议版本 'v2'，可选: ['v1', 'v3']
    """
    if version == "v1":
        return load_tasks(seed=seed)
    if version == "v3":
        from src.data.task_manifest_v2 import load_tasks_v3

        return load_tasks_v3(seed=seed)
    raise ValueError(
        "未知的评测协议版本 %r，可选: %s" % (version, sorted(TASK_SET_SIZES))
    )


def tasks_by_category(tasks: list[Task] | None = None) -> dict[str, list[Task]]:
    out: dict[str, list[Task]] = {}
    for t in tasks or load_tasks():
        out.setdefault(t.category, []).append(t)
    return out


def tasks_by_visibility(vis: str, tasks: list[Task] | None = None) -> list[Task]:
    """``vis="unseen"`` 取泛化集 —— 训练从未接触过的任务。"""
    return [t for t in (tasks or load_tasks()) if t.visibility == vis]


def tasks_by_difficulty(diff: str, tasks: list[Task] | None = None) -> list[Task]:
    return [t for t in (tasks or load_tasks()) if t.difficulty == diff]


def recovery_tasks(tasks: list[Task] | None = None) -> list[Task]:
    """只取纠错轮任务 —— Week 2 错误恢复偏好对的数据源。"""
    return [t for t in (tasks or load_tasks()) if t.is_recovery]


def error_tasks(tasks: list[Task] | None = None) -> list[Task]:
    """对标项目的 uncovered 集：纠错轮 + 需澄清类任务（模型天然容易错的）。"""
    return [
        t
        for t in (tasks or load_tasks())
        if t.is_recovery or "clarify_first" in t.tags
    ]


def iter_tasks(**filters: Any) -> Iterator[Task]:
    """按任意字段过滤迭代。``iter_tasks(difficulty="hard", visibility="seen")``"""
    for t in load_tasks():
        if all(getattr(t, k) == v for k, v in filters.items()):
            yield t


def write_manifest(path: str, tasks: list[Task] | None = None) -> str:
    """把任务集固化到 json，方便跨机器复用（避免两边生成不一致）。"""
    tasks = tasks or load_tasks()
    payload = {
        "version": "1.0",
        "base_date": BASE_DATE.isoformat(),
        "num_tasks": len(tasks),
        "tasks": [t.to_dict() for t in tasks],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    return path


def read_manifest(path: str) -> list[Task]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return [Task.from_dict(d) for d in payload["tasks"]]


def summary(tasks: list[Task] | None = None) -> str:
    """人类可读的任务集统计，用于日志。"""
    tasks = tasks or load_tasks()
    lines = [
        f"任务集总计 {len(tasks)} 个",
        "-" * 62,
        f"{'类别':<20}{'总数':>6}{'easy':>7}{'medium':>8}{'hard':>7}{'unseen':>8}",
        "-" * 62,
    ]
    for cat, group in sorted(tasks_by_category(tasks).items()):
        n = len(group)
        e = sum(t.difficulty == "easy" for t in group)
        m = sum(t.difficulty == "medium" for t in group)
        h = sum(t.difficulty == "hard" for t in group)
        u = sum(t.visibility == "unseen" for t in group)
        lines.append(f"{cat:<20}{n:>6}{e:>7}{m:>8}{h:>7}{u:>8}")
    lines.append("-" * 62)
    lines.append(
        f"seen={sum(t.visibility == 'seen' for t in tasks)}  "
        f"unseen={sum(t.visibility == 'unseen' for t in tasks)}  "
        f"recovery={len(recovery_tasks(tasks))}"
    )
    return "\n".join(lines)


# ============================================================================
# 自检
# ============================================================================

def _self_check() -> None:
    """生成后立刻做一致性检查 —— 任务集是所有实验的地基，错了代价很大。"""
    tasks = load_tasks()
    errors: list[str] = []

    # 1. 唯一 ID
    ids = [t.task_id for t in tasks]
    if len(set(ids)) != len(ids):
        errors.append("task_id 有重复")

    # 2. 目标轮下标合法，且该轮必须是 assistant 生成位置（即前缀最后一个是 user）
    for t in tasks:
        if not (0 < t.target_turn <= len(t.turns)):
            errors.append(f"{t.task_id}: target_turn={t.target_turn} 越界")
            continue
        last = t.turns[t.target_turn - 1]
        if last.role != "user":
            errors.append(f"{t.task_id}: 目标轮前一条不是 user（是 {last.role}）")

    # 3. 期望工具名必须在白名单（或 None）
    from src.env.tool_catalog import VALID_TOOL_NAMES

    for t in tasks:
        tool = t.expected.tool
        if tool is not None and tool not in VALID_TOOL_NAMES:
            errors.append(f"{t.task_id}: 期望工具 {tool!r} 不在白名单")

    # 4. 相对日期任务必须换算过（ground truth 里不能出现中文相对词）
    for t in tasks:
        for v in t.expected.graded_args.values():
            if isinstance(v, str) and any(k in v for k in ("明天", "后天", "本周", "下周", "这周", "天后")):
                errors.append(f"{t.task_id}: ground truth 参数含未换算的相对日期 {v!r}")

    # 5. recovery 任务必须真的有纠错话术
    for t in tasks:
        if t.is_recovery and not any(
            k in t.turns[t.target_turn - 1].content for k in ("请", "不要", "没有", "必须", "不能", "不该")
        ):
            errors.append(f"{t.task_id}: recovery 任务的纠错话术不含明确指令")

    if errors:
        raise AssertionError("任务集自检失败:\n  - " + "\n  - ".join(errors))


if __name__ == "__main__":  # pragma: no cover
    _self_check()
    print(summary())
    print()
    print("示例任务（recovery）:")
    print(json.dumps(recovery_tasks()[0].to_dict(), ensure_ascii=False, indent=2)[:1200])
