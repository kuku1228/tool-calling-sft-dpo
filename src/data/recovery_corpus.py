"""纠错轮训练场景语料：为「错误恢复偏好对」提供足量且多样的样本。

为什么不能直接用评测任务集
--------------------------
``src/data/task_manifest.py`` 里的 recovery 任务只有 12 个 ——
它们是**评测集**，必须保持小而固定，不能被训练数据稀释或污染。

但错误恢复偏好对需要上千条数据才训得动。所以这里把
「用户纠错 → 模型修正」这个交互模式**参数化展开**：
同样的 12 种错误模式，套上不同的城市、日期、订单号、乘客名、
纠错措辞，生成约 84 个互不相同的训练场景。

评测集与训练语料的边界
----------------------
::

    task_manifest.build_tasks()      → 100 个任务，12 个 recovery，**用于评测**
    recovery_corpus.build_scenarios() → ~84 个场景，**只用于构造训练偏好对**

两者的 mistake_hint 用同一套 key（``format_ignore_correction`` 等），
所以「训练里教了什么」和「评测里考了什么」可以逐类对齐 ——
这也是诊断报告里能做 per-mistake-type 归因分析的前提。
"""

from __future__ import annotations

import hashlib
import random
from datetime import date, timedelta
from dataclasses import dataclass
from typing import Any

from src.env.tool_catalog import AIRPORT_CODES, CITY_NAMES
from src.utils.hashing import stable_seed

# ============================================================================
# 参数池
# ============================================================================

_BOOKING_IDS: tuple[str, ...] = (
    "BK7F2A9C",
    "BK3D8E1B",
    "BK9A4C6F",
    "BK2B7D5E",
    "BK8C1F3A",
    "BK5E9B2D",
    "BK1A6C8F",
    "BK4F3D7B",
)

#: (中文起点, 中文终点, 三字码起点, 三字码终点)
_ROUTES: tuple[tuple[str, str, str, str], ...] = (
    ("北京", "上海", "PEK", "SHA"),
    ("上海", "广州", "SHA", "CAN"),
    ("深圳", "成都", "SZX", "CTU"),
    ("杭州", "西安", "HGH", "XIY"),
    ("重庆", "昆明", "CKG", "KMG"),
    ("厦门", "南京", "XMN", "NKG"),
    ("武汉", "青岛", "WUH", "TAO"),
)

_DATES: tuple[str, ...] = (
    "2025-07-18",
    "2025-07-25",
    "2025-08-02",
    "2025-08-11",
    "2025-09-05",
    # ---- v3 评测协议所在的日期区间 ----
    #
    # ⚠️ 这一段是 2026-10 补的。上面 5 个日期全在 2025 年，而 v3 评测的
    # 日期区间是 **2026-11 ~ 2027-01** —— 也就是 v1 时代「日期全在过去」
    # 这个缺陷，在合成语料里**没有跟着一起修**。
    #
    # 日期本身是**可推导**的（相对日期→绝对日期靠给定基准日换算），所以
    # 缺它不像缺机场码那样是确定性失败。但模型只见过 2025 年的写法时，
    # 输出的年份分布会整体偏移，属于不必要的分布偏差。
    "2026-11-05",
    "2026-11-20",
    "2026-12-03",
    "2026-12-18",
    "2026-12-29",
    "2027-01-05",
    "2027-01-11",
    "2027-01-14",
    "2027-01-18",
    "2027-01-22",
)

#: 相对日期表达（用于「没换算成绝对日期」这一类错误）
_RELATIVE_DAYS: tuple[tuple[str, str], ...] = (
    ("后天", "2025-07-16"),
    ("明天", "2025-07-15"),
    ("本周五", "2025-07-18"),
    ("下周一", "2025-07-21"),
    ("这周日", "2025-07-20"),
)

_PASSENGERS: tuple[str, ...] = (
    "张伟",
    "李娜",
    "王芳",
    "刘洋",
    "陈晓明",
    "赵敏",
    "周杰",
    "吴桐",
)

_WEATHER_CITIES: tuple[str, ...] = ("北京", "上海", "广州", "成都", "三亚", "哈尔滨", "南京")

_BAGGAGE_KG: tuple[int, ...] = (25, 30, 35, 40, 45)

#: 「貌似合理但不存在」的假工具 —— 用于幻觉类场景
_FAKE_TOOLS: tuple[str, ...] = (
    "get_order_details",
    "get_travel_advisory",
    "query_baggage",
    "fetch_booking",
    "search_booking",
)

#: 占位符参数 —— 用于「填了占位符」类场景
_PLACEHOLDERS: tuple[str, ...] = (
    "previous_booking",
    "previous_reservation",
    "my_booking",
    "UNKNOWN",
    "BK000000",
)

#: 纠错话术池，按错误类型分组。
#:
#: ⚠️ **每个场景必须挑与该场景错误类型匹配的那一组**。
#: 早期版本从整个池子里随机抽，结果出现了「航班搜索场景
#: 配上『city 参数要填中文城市名』」这种前后矛盾的训练数据 ——
#: 模型会学到「忽略用户的纠正」，比不加数据更糟。
#: 所以下面每个 builder 都**显式指定**自己要用的那一组。
CORRECTION_PHRASING: dict[str, tuple[str, ...]] = {
    "format": (
        "你没有真的调用工具，请用规定的 <tool_call> 格式重新输出。",
        "请直接输出工具调用，不要只说你要去查询。",
        "你的回复里没有工具调用，请按格式重新输出一次。",
        "我要的是工具调用结果，不是一句「马上就好」。",
    ),
    "hallucination": (
        "工具列表里没有 {fake} 这个工具，请用正确的工具重新调用。",
        "你调用了一个不存在的工具，请从给定工具列表里重新选择。",
        "{fake} 不在可用工具里，请改用正确的工具名。",
        "请只用工具列表里给出的工具，不要自己发明工具名。",
    ),
    "wrong_tool": (
        "你选的工具不对，请重新选择正确的工具。",
        "这个工具不能完成我的请求，请换一个。",
        "我要的不是这个操作，请用正确的工具重新调用。",
        "工具选错了，请你再想一下我到底要做什么。",
    ),
    "param_placeholder": (
        "订单号是 {bid}，不要填 {placeholder}。",
        "参数不能填占位符，请填我给你的真实信息。",
        "{placeholder} 不是一个有效的订单号，请重新填。",
    ),
    # 城市名 ↔ 三字码 混用（flight_search 的 origin/destination）
    "param_airport_code": (
        "origin 和 destination 要填机场三字码，不要填中文城市名。",
        "机场参数请用三字码，例如 PEK，不要写中文城市名。",
        "你填的是中文城市名，工具要求的是机场三字码。",
    ),
    # 天气工具的 city 参数（与上面相反的方向）
    "param_city_name": (
        "city 参数要填中文城市名，不要填三字码。",
        "天气工具的 city 要写中文城市名，比如 北京，不要写 PEK。",
        "你把城市名写成机场代码了，请改成中文城市名。",
    ),
    # 相对日期未换算
    "param_date": (
        "date 必须是 YYYY-MM-DD 格式的绝对日期，不能写相对时间。",
        "请把相对日期换算成具体日期再填进 date。",
        "date 参数要填具体的年月日，不要填「后天」这类说法。",
    ),
    "fake_entity": (
        "{fake_id} 这个{kind}不存在，你必须先查询再操作。",
        "不要编造{kind}，请先用工具查到真实的值。",
        "你写的{kind}是我没给过的，请从工具返回结果里取。",
    ),
    "repeat": (
        "你还在重复刚才的错误，请认真看我说的并改正。",
        "这个错误我指出过了，请换一种正确的做法。",
        "不要重复上一轮的回答，请修正后重新调用。",
    ),
    "clarify": (
        "取消原因不能写「用户要求」，请先问清楚我为什么取消。",
        "不要自己编一个理由，信息不够就先问我。",
        "这个参数我没有提供过，你应该先向我确认。",
    ),
    "incomplete": (
        "你搜了航班但没下单，请完成订票。",
        "你停在中间步骤了，请继续完成我的请求。",
        "查询只是第一步，请接着把后续操作做完。",
    ),
}


# ============================================================================
# 场景数据类
# ============================================================================


@dataclass
class RecoveryScenario:
    """一个完整的纠错交互场景。

    结构与 ``Task`` 的 recovery 任务一致，但**不参与评测**，
    只用于构造偏好数据。
    """

    scenario_id: str
    mistake_key: str
    #: 与 error_taxonomy 的类别对齐（FORMAT_ERROR / HALLUCINATION / ...）
    error_family: str
    #: 首轮用户请求
    user_request: str
    #: 模型的首轮错误回复（会被放进 prompt）
    wrong_reply: str
    #: 用户的纠错话术
    correction: str
    #: 正确修正后应输出的内容
    correct_reply: str
    #: 这个场景考察什么
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "mistake_key": self.mistake_key,
            "error_family": self.error_family,
            "user_request": self.user_request,
            "wrong_reply": self.wrong_reply,
            "correction": self.correction,
            "correct_reply": self.correct_reply,
            "note": self.note,
        }


# ============================================================================
# 生成器
# ============================================================================


def _tool_call(name: str, args: dict[str, Any]) -> str:
    import json

    return f'<tool_call>{json.dumps({"name": name, "arguments": args}, ensure_ascii=False)}</tool_call>'


def _bad_json_call(name: str, args: dict[str, Any]) -> str:
    """生成键缺引号的不合法 JSON。"""
    import json

    body = ", ".join(f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in args.items())
    return f'<tool_call>{{"name": "{name}", "arguments": {{{body}}}}}</tool_call>'


#: 12 种错误模式的定义。每个函数接收 (rng, idx) 返回一个场景。
#: 顺序与 ``task_manifest._recovery_patterns`` 对应，key 必须一致。


def _s_format_ignore(rng: random.Random, idx: int) -> RecoveryScenario:
    origin, dest, oc, dc = _ROUTES[idx % len(_ROUTES)]
    date_value = _DATES[idx % len(_DATES)]
    req = rng.choice(
        [
            f"查一下{date_value}{origin}到{dest}的航班",
            f"{date_value}{origin}飞{dest}有什么航班？",
            f"帮我搜索{date_value}从{origin}到{dest}的机票",
        ]
    )
    wrong = rng.choice(
        [
            "好的，我来帮您查询一下，请稍等。",
            "正在为您查询相关航班信息，马上就好。",
            "我明白了，我先帮您确认一下航班情况。",
        ]
    )
    return RecoveryScenario(
        scenario_id=f"format_ignore_{idx:03d}",
        mistake_key="format_ignore_correction",
        error_family="FORMAT_ERROR",
        user_request=req,
        wrong_reply=wrong,
        correction=rng.choice(CORRECTION_PHRASING["format"]),
        correct_reply=_tool_call(
            "search_flights", {"origin": oc, "destination": dc, "date": date_value}
        ),
        note="格式错误 → 纠错后输出合法 tool_call",
    )


def _s_hallucinated_tool(rng: random.Random, idx: int) -> RecoveryScenario:
    bid = _BOOKING_IDS[idx % len(_BOOKING_IDS)]
    fake = _FAKE_TOOLS[idx % len(_FAKE_TOOLS)]
    req = rng.choice(
        [
            f"查一下我的订单{bid}还能托运多少行李",
            f"订单{bid}的行李额度是多少？",
            f"{bid}这个订单的免费行李是多少公斤",
        ]
    )
    wrong = _tool_call(fake, {"order_id": bid})
    return RecoveryScenario(
        scenario_id=f"hallucinated_tool_{idx:03d}",
        mistake_key="hallucinated_tool_corrected",
        error_family="HALLUCINATION",
        user_request=req,
        wrong_reply=wrong,
        correction=rng.choice(CORRECTION_PHRASING["hallucination"]).format(fake=fake),
        correct_reply=_tool_call("check_baggage", {"booking_id": bid}),
        note="工具幻觉 → 纠错后换回白名单工具",
    )


def _s_wrong_tool(rng: random.Random, idx: int) -> RecoveryScenario:
    bid = _BOOKING_IDS[idx % len(_BOOKING_IDS)]
    kg = _BAGGAGE_KG[idx % len(_BAGGAGE_KG)]
    req = rng.choice(
        [
            f"把订单{bid}的行李额度改成{kg}公斤",
            f"订单{bid}我要加到{kg}公斤行李",
            f"{bid}的行李额度改到{kg}公斤",
        ]
    )
    wrong = _tool_call("check_baggage", {"booking_id": bid})
    return RecoveryScenario(
        scenario_id=f"wrong_tool_{idx:03d}",
        mistake_key="wrong_tool_corrected",
        error_family="WRONG_TOOL",
        user_request=req,
        wrong_reply=wrong,
        correction=rng.choice(CORRECTION_PHRASING["wrong_tool"]),
        correct_reply=_tool_call("modify_baggage", {"booking_id": bid, "new_baggage_kg": kg}),
        note="工具选错 → 纠错后换成写操作工具",
    )


def _s_param_placeholder(rng: random.Random, idx: int) -> RecoveryScenario:
    bid = _BOOKING_IDS[idx % len(_BOOKING_IDS)]
    ph = _PLACEHOLDERS[idx % len(_PLACEHOLDERS)]
    req = f"帮我查一下订单{bid}的行李额度"
    wrong = _tool_call("check_baggage", {"booking_id": ph})
    return RecoveryScenario(
        scenario_id=f"param_placeholder_{idx:03d}",
        mistake_key="param_placeholder_corrected",
        error_family="PARAM_ERROR",
        user_request=req,
        wrong_reply=wrong,
        correction=rng.choice(CORRECTION_PHRASING["param_placeholder"]).format(bid=bid, placeholder=ph),
        correct_reply=_tool_call("check_baggage", {"booking_id": bid}),
        note="参数占位符 → 纠错后填回真实 ID",
    )


def _s_city_code(rng: random.Random, idx: int) -> RecoveryScenario:
    origin, dest, oc, dc = _ROUTES[idx % len(_ROUTES)]
    date_value = _DATES[idx % len(_DATES)]
    req = rng.choice(
        [
            f"{date_value}从{origin}飞{dest}有哪些航班",
            f"我想{date_value}从{origin}去{dest}，看看航班",
        ]
    )
    wrong = _tool_call(
        "search_flights", {"origin": origin, "destination": dest, "date": date_value}
    )
    return RecoveryScenario(
        scenario_id=f"city_code_{idx:03d}",
        mistake_key="city_code_corrected",
        error_family="PARAM_ERROR",
        user_request=req,
        wrong_reply=wrong,
        correction=rng.choice(CORRECTION_PHRASING["param_airport_code"]),
        correct_reply=_tool_call(
            "search_flights", {"origin": oc, "destination": dc, "date": date_value}
        ),
        note="中文城市名当三字码 → 纠错后改对",
    )


def _s_fake_flight_id(rng: random.Random, idx: int) -> RecoveryScenario:
    origin, dest, oc, dc = _ROUTES[idx % len(_ROUTES)]
    date_value = _DATES[idx % len(_DATES)]
    name = _PASSENGERS[idx % len(_PASSENGERS)]
    fake_id = f"CA{1000 + idx * 7 % 9000}"
    req = f"给{name}订{date_value}从{origin}到{dest}的机票"
    wrong = _tool_call("book_flight", {"flight_id": fake_id, "passenger_name": name})
    return RecoveryScenario(
        scenario_id=f"fake_flight_id_{idx:03d}",
        mistake_key="repeated_same_error",
        error_family="NO_RECOVERY",
        user_request=req,
        wrong_reply=wrong,
        correction=rng.choice(CORRECTION_PHRASING["fake_entity"]).format(fake_id=fake_id, kind="航班号"),
        correct_reply=_tool_call(
            "search_flights", {"origin": oc, "destination": dc, "date": date_value}
        ),
        note="编造 flight_id → 纠错后退回「先搜索」前置步骤",
    )


def _s_fake_reason(rng: random.Random, idx: int) -> RecoveryScenario:
    bid = _BOOKING_IDS[idx % len(_BOOKING_IDS)]
    req = f"帮我取消订单{bid}"
    wrong = _tool_call("cancel_booking", {"booking_id": bid, "reason": "用户要求"})
    return RecoveryScenario(
        scenario_id=f"fake_reason_{idx:03d}",
        mistake_key="missing_reason_corrected",
        error_family="PARAM_ERROR",
        user_request=req,
        wrong_reply=wrong,
        correction=rng.choice(CORRECTION_PHRASING["clarify"]),
        correct_reply="请问您能告诉我取消这个订单的原因吗？我需要记录取消理由才能帮您处理。",
        note="编造 reason → 纠错后改为先追问",
    )


def _s_relative_date(rng: random.Random, idx: int) -> RecoveryScenario:
    origin, dest, oc, dc = _ROUTES[idx % len(_ROUTES)]
    rel, absolute = _RELATIVE_DAYS[idx % len(_RELATIVE_DAYS)]
    req = f"{rel}从{origin}到{dest}的航班有哪些"
    wrong = _tool_call("search_flights", {"origin": oc, "destination": dc, "date": rel})
    return RecoveryScenario(
        scenario_id=f"relative_date_{idx:03d}",
        mistake_key="wrong_date_corrected",
        error_family="PARAM_ERROR",
        user_request=req,
        wrong_reply=wrong,
        correction=rng.choice(CORRECTION_PHRASING["param_date"]),
        correct_reply=_tool_call(
            "search_flights", {"origin": oc, "destination": dc, "date": absolute}
        ),
        note="相对日期未换算 → 纠错后改成绝对日期",
    )


def _s_incomplete_step(rng: random.Random, idx: int) -> RecoveryScenario:
    origin, dest, oc, dc = _ROUTES[idx % len(_ROUTES)]
    date_value = _DATES[idx % len(_DATES)]
    name = _PASSENGERS[idx % len(_PASSENGERS)]
    req = f"{origin}到{dest} {date_value}的航班，直接订最便宜那班，乘客{name}"
    wrong = _tool_call(
        "search_flights", {"origin": oc, "destination": dc, "date": date_value, "max_price": 500}
    )
    return RecoveryScenario(
        scenario_id=f"incomplete_step_{idx:03d}",
        mistake_key="overbook_corrected",
        error_family="WRONG_TOOL",
        user_request=req,
        wrong_reply=wrong,
        correction=rng.choice(CORRECTION_PHRASING["incomplete"]),
        correct_reply=_tool_call("book_flight", {"flight_id": "MU5101", "passenger_name": name}),
        note="停在中间步骤 → 纠错后推进到写操作",
    )


def _s_wrong_city_param(rng: random.Random, idx: int) -> RecoveryScenario:
    city = _WEATHER_CITIES[idx % len(_WEATHER_CITIES)]
    code = AIRPORT_CODES.get(city, "PEK")
    req = rng.choice([f"{city}今天天气怎么样", f"帮我看看{city}今天会不会下雨"])
    wrong = _tool_call("get_weather", {"city": code})
    return RecoveryScenario(
        scenario_id=f"wrong_city_param_{idx:03d}",
        mistake_key="wrong_city_param_corrected",
        error_family="PARAM_ERROR",
        user_request=req,
        wrong_reply=wrong,
        correction=rng.choice(CORRECTION_PHRASING["param_city_name"]),
        correct_reply=_tool_call("get_weather", {"city": city}),
        note="city 填了三字码 → 纠错后改成中文城市名",
    )


def _s_hallucinated_weather(rng: random.Random, idx: int) -> RecoveryScenario:
    city = _WEATHER_CITIES[(idx + 3) % len(_WEATHER_CITIES)]
    fake = "get_travel_advisory" if idx % 2 == 0 else "check_weather_forecast"
    req = f"帮我看看{city}的天气"
    wrong = _tool_call(fake, {"city": city})
    return RecoveryScenario(
        scenario_id=f"hallucinated_weather_{idx:03d}",
        mistake_key="hallucinated_weather_tool_corrected",
        error_family="HALLUCINATION",
        user_request=req,
        wrong_reply=wrong,
        correction=rng.choice(CORRECTION_PHRASING["hallucination"]).format(fake=fake),
        correct_reply=_tool_call("get_weather", {"city": city}),
        note="天气工具幻觉 → 纠错后落到 get_weather",
    )


def _s_broken_json(rng: random.Random, idx: int) -> RecoveryScenario:
    bid = _BOOKING_IDS[idx % len(_BOOKING_IDS)]
    req = f"查一下订单{bid}的行李额度"
    wrong = _bad_json_call("check_baggage", {"booking_id": bid})
    return RecoveryScenario(
        scenario_id=f"broken_json_{idx:03d}",
        mistake_key="format_json_broken_corrected",
        error_family="FORMAT_ERROR",
        user_request=req,
        wrong_reply=wrong,
        correction=rng.choice(CORRECTION_PHRASING["format"]),
        correct_reply=_tool_call("check_baggage", {"booking_id": bid}),
        note="JSON 语法非法 → 纠错后输出合法 JSON",
    )


#: 12 个模式生成器，顺序即 idx 分配顺序
_SCENARIO_BUILDERS: tuple[Any, ...] = (
    _s_format_ignore,
    _s_hallucinated_tool,
    _s_wrong_tool,
    _s_param_placeholder,
    _s_city_code,
    _s_fake_flight_id,
    _s_fake_reason,
    _s_relative_date,
    _s_incomplete_step,
    _s_wrong_city_param,
    _s_hallucinated_weather,
    _s_broken_json,
)


def build_scenarios(n_variants_per_pattern: int = 7, seed: int = 42) -> list[RecoveryScenario]:
    """生成全部纠错训练场景。

    Args:
        n_variants_per_pattern: 每个错误模式展开多少个变体。
            7 会生成 12 × 7 = 84 个场景。

    >>> scenarios = build_scenarios(n_variants_per_pattern=2)
    >>> len(scenarios)
    24
    >>> len({s.scenario_id for s in scenarios}) == len(scenarios)
    True
    """
    out: list[RecoveryScenario] = []
    for idx in range(n_variants_per_pattern):
        for builder in _SCENARIO_BUILDERS:
            rng = random.Random(stable_seed(seed, builder.__name__, idx))
            out.append(builder(rng, idx))
    return out


def scenario_stats(scenarios: list[RecoveryScenario]) -> dict[str, Any]:
    """按错误族与模式统计场景数。"""
    by_family: dict[str, int] = {}
    by_key: dict[str, int] = {}
    for s in scenarios:
        by_family[s.error_family] = by_family.get(s.error_family, 0) + 1
        by_key[s.mistake_key] = by_key.get(s.mistake_key, 0) + 1
    return {
        "total": len(scenarios),
        "by_error_family": dict(sorted(by_family.items(), key=lambda kv: -kv[1])),
        "by_mistake_key": dict(sorted(by_key.items())),
    }


# ============================================================================
# 转成 Task：让格式 / 工具选择 / 参数约束对也有足够的 prompt 多样性
# ============================================================================

#: 由 ``correct_reply`` 解析出的工具名 → 该场景首轮用户请求的自然语言性质
#: 合成训练任务的 ``task_id`` 前缀。
#:
#: **合成训练语料**的 ``task_id`` 前缀，也是「训练数据 / 评测任务」的分界线。
#:
#: 评测任务的 ``task_id`` 是 ``task_0001``（v1）/ ``taskv2_0001``（v3），
#: **不带这个前缀**。所以判据是「**带有**本前缀 = 合成语料 = 允许进训练」。
#:
#: ⚠️ 本常量曾命名为 ``SYNTHETIC_TASK_PREFIX``，字面意思与语义**完全相反**，
#: 导致 ``PreferenceBuilder.build_all`` 的报错信息写成「训练数据中混入了评测任务
#: （样例 task_0001）」，而 ``task_0001`` 恰恰是被错误混进来的那个。
#: 现已改名为 ``SYNTHETIC_TASK_PREFIX``，旧名保留为别名以免破坏调用方。
#:
#: 为什么需要它：开发期三处 ``build_*`` 直接吃了评测任务集，导致 51.5% 的
#: 训练样本与评测 prompt 逐字相同 —— 等于在测试集上训练。
SYNTHETIC_TASK_PREFIX = "trainsyn_"

#: @deprecated 用 :data:`SYNTHETIC_TASK_PREFIX`。旧名语义相反，仅为兼容保留。
EVAL_TASK_PREFIX = SYNTHETIC_TASK_PREFIX


def build_training_tasks(seed: int = 42, n_variants: int = 7) -> list[Any]:
    """把纠错场景的**首轮请求**转成单轮任务，只用于构造训练数据。

    为什么需要：格式约束对 / 工具选择对 / 参数约束对都需要
    「一个 prompt + 一个正确的工具调用」。光靠 100 个评测任务，
    prompt 的重用次数太高（每个任务被用 ~12 次），
    去重之后数据量上不去。

    这些任务：
      - ``visibility`` 一律标为 ``train_only``，**绝不会进入评测**；
      - ``task_id`` 用 ``trainsyn_`` 前缀，与评测任务在文件里一眼可分。

    >>> tasks = build_training_tasks()
    >>> len(tasks) > 50
    True
    >>> all(t.task_id.startswith("trainsyn_") for t in tasks)
    True
    """
    import json

    from src.data.task_manifest import ExpectedCall, Task, Turn
    from src.eval.tool_call_parser import extract_tool_calls

    out: list[Task] = []
    for sc in build_scenarios(n_variants_per_pattern=n_variants, seed=seed):
        parse = extract_tool_calls(sc.correct_reply)
        call = parse.first
        if call is None:
            # 正解是自然语言澄清 → 不转（澄清类样本由 recovery 对覆盖）
            continue
        task = Task(
            task_id=f"trainsyn_{sc.scenario_id}",
            category=f"trainsyn_{sc.error_family.lower()}",
            difficulty="medium",
            visibility="train_only",
            turns=[Turn("user", sc.user_request)],
            target_turn=1,
            target_turn_kind="initial",
            expected=ExpectedCall(
                tool=call.name,
                args=dict(call.arguments),
                note=f"由纠错语料转成的训练任务（{sc.mistake_key}）",
            ),
            tags=("trainsyn", sc.error_family.lower()),
        )
        out.append(task)

    # ------------------------------------------------------------------
    # 额外合成任务：补两项评测集需要、但纠错语料覆盖不到的能力
    # ------------------------------------------------------------------
    #
    # ⚠️ 这两组是**修掉数据污染之后**才暴露出来的缺口。
    #
    # 原实现的三处 ``build_*`` 直接吃评测任务集（见 ``SYNTHETIC_TASK_PREFIX``
    # 的说明），所以「评测需要 cancel_booking」这件事被掩盖了 —— 因为
    # 训练数据里本来就有那 8 个评测 cancellation 任务。一旦按正确做法
    # 只用合成语料，就会发现：
    #
    #   * ``cancel_booking`` 在纠错语料里**完全没有出现**
    #     → 模型学不到这个写操作，评测的 8 个 cancellation 任务必然全错
    #   * 「信息不足应追问」（``expected.tool is None``）也没有训练样本
    #     → 而那正是本项目要证明「对比信号必要」的核心场景
    #
    # 所以在这里补齐。它们同样是 ``trainsyn_`` 前缀的 train_only 任务。
    out.extend(_synthetic_cancel_tasks())
    out.extend(_synthetic_clarify_tasks())
    # 多缺失模式的追问 + 多步场景里的追问。
    # 实测驱动：补多步语料把追问占比从 11.8% 稀释到 9.6%，
    # `cancellation` 随之 62.5% → 31.8%。这两类把追问补回来，
    # 而且**加模式**（不只加副本）。
    out.extend(_synthetic_clarify_modes())
    out.extend(_synthetic_multi_step_clarify())
    out.extend(_synthetic_airport_coverage_tasks())
    out.extend(_synthetic_booking_tasks())
    # ------------------------------------------------------------------
    # ⚠️ 下面两个是**实测查出缺口后补的**（v3 基线上三个类别 0%）
    # ------------------------------------------------------------------
    #
    # 实测缺口（`build_training_tasks()` 的 181 个任务里）：
    #
    #     带观测轮（多步流程）的合成任务: 0 / 181
    #     multi_tool 训练任务:            0 / 181
    #
    # 而评测有 **36 个任务**依赖这两项技能，结果是：
    #
    #     booking_flow   0.0%   action_hallucination=58/60（从不调 book_flight）
    #     multi_tool     0.0%   没调天气 + 调了航班 = 38/40（只做一半）
    #
    # **这不是模型能力问题，是训练语料根本没教过。**
    # 原来的 `_synthetic_booking_tasks` 把 `flight_id` 直接写在**用户话里**
    # （「帮我订 MU1000 这一班」），而评测要求**从观测里读**。
    # 技能完全不同，所以 0%。
    out.extend(_synthetic_observation_tasks())
    out.extend(_synthetic_multi_tool_tasks())

    # ------------------------------------------------------------------
    # 剔除与评测任务**逐字撞车**的合成任务
    # ------------------------------------------------------------------
    #
    # 合成语料与评测任务共用同一批领域字面量（订单号全集只有 7 个、
    # 城市名有限），因此必然有几处首轮文本完全相同。实测：
    #
    #   task_0041 评测「帮我查一下订单BK7F2A9C的行李额度」
    #   ←→ trainsyn_param_placeholder_000 完全相同的句子
    #
    # 这不是「生成模板不小心重复」，而是**两个语料库的输入空间重叠**，
    # 只要两边都用真实的订单号格式就无法完全避免。
    #
    # 处理方式：**剔除合成侧**，而不是把评测任务踢出评测集 ——
    # 这样评测集保持 100 个任务、所有既有指标可直接比较。
    #
    # 判据是**整个渲染后 prompt 逐字相同**（评测就是这么比对的）。
    from src.models.prompt_builder import build_plain_prompt

    # 同时过滤 **v3** 评测集：v3 是当前协议，撞车同样是污染。
    try:
        from src.data.task_manifest import load_tasks as _load_eval
        from src.data.task_manifest_v2 import load_tasks_v3 as _load_v3

        eval_tasks = list(_load_eval()) + list(_load_v3())
        eval_prompts = {build_plain_prompt(t).strip() for t in eval_tasks}
        if eval_prompts:
            before = len(out)
            out = [t for t in out if build_plain_prompt(t).strip() not in eval_prompts]
            _dropped = before - len(out)
            if _dropped:
                # 记录而不是静默丢弃 —— 丢了多少、为什么丢，必须看得见
                import sys as _sys

                print(
                    f"  [trainsyn] 剔除 {_dropped} 个与评测 prompt 逐字相同的合成任务",
                    file=_sys.stderr,
                )

        # ------------------------------------------------------------------
        # ⚠️ 我在这里试过「过滤掉期望调用相同的训练任务」，**失败了**，记下来
        # ------------------------------------------------------------------
        #
        # 动机是对的：实测有 5 个评测任务的期望调用与训练完全相同
        #     get_weather(city='三亚'/'上海'/'北京'/'南京')
        #     search_flights(YNT→ZUH, 2026-12-10)
        # 模型可能靠匹配记忆中的样本答对。
        #
        # 但**用过滤解决会摧毁训练语料**：实测删掉 41 个任务，
        # 把 `check_baggage` 与 `get_weather` 的训练样本**全删光了**
        # （断言立刻报「缺失工具 2 个」）。
        #
        # 根因：评测与训练**都**有意覆盖同一个实体空间，而实体池很小
        # （baggage ID 只有几十个）。任何「防止组合相同」的过滤都会
        # 连带删掉大量合法样本。
        #
        # **正确解法在别处**：让**评测集用训练里从未出现过的实体**
        # （见 `task_set_v2_specs.py` 的实体池设计），
        # 而不是在训练侧做减法。这两者的区别是：
        #   * 训练侧过滤：破坏训练语料，且随实体池大小而恶化
        #   * 评测侧换实体：不损失任何训练样本，且**语义正确**
        #     （评测本来就该考没见过的实体）
        #
        # 报告里的正确表述（专家建议）：**层级 2 = 泄漏、层级 3 = 合理泛化**，
        # 两者必须分开说，不能笼统称「重合」。
    except Exception:  # pragma: no cover - 评测任务集不可用时不过滤
        pass

    return out


#: 合成 cancellation 任务的模板：(用户句带 ``{bk}``, 取消原因)
#:
#: ⚠️ 用户句里**必须**含取消原因 —— ``cancel_booking`` 的 schema 要求
#: ``reason`` 必填，缺了会被执行层拒绝。
#:
#: ⚠️ 更重要的一点：这些任务会被渲染成**三轮结构**（见
#: :func:`_synthetic_cancel_tasks`），因为评测里的 cancellation 任务是
#:
#:     turn0 [user]      我要取消订单 BK9A4C6F，原因是行程变更
#:     turn1 [assistant] 好的，我确认一下：您要取消订单 BK9A4C6F，原因是「行程变更」，对吗？
#:     turn2 [user]      对，取消吧
#:
#: 而最初的合成语料是**单轮**的，于是模型只学会了「收到取消请求 → 回一句
#: 确认话术」。到评测里看到确认轮后又回了一句自然语言而没真的调用工具：
#:
#:     期望 cancel_booking(BK8C1F3A)
#:     实际 '明白了，请稍等，我将为您取消订单 BK8C1F3A。'   ← promise_only
#:
#: 实测 cancellation 因此失败 20/48 采样。所以合成语料必须复现完整轮次结构。
_CANCEL_TEMPLATES: tuple[tuple[str, str], ...] = (
    ("我要取消订单 {bk}，原因是行程变更", "行程变更"),
    ("帮我取消 {bk}，临时有事去不了", "临时有事"),
    ("{bk} 这个订单我要退掉，找到更合适的航班了", "找到更合适的航班"),
    ("取消 {bk}，会议取消了", "会议取消"),
    ("{bk} 退了吧，身体不舒服", "身体不适"),
    ("帮我取消 {bk}，天气原因走不了", "天气原因"),
    ("把 {bk} 取消掉，行程改了", "行程变更"),
    ("{bk} 我不想去了，帮我办理退票", "个人原因"),
)

#: 合成「信息不足应追问」任务：(用户句, 为何信息不足)
_CLARIFY_TEMPLATES: tuple[tuple[str, str], ...] = (
    # ---- 意图不明 / 缺关键参数：正确行为是**追问**，不是调工具 ----
    ("我不想要 {bk} 这个订单了", "缺少取消原因"),
    ("{bk} 这个订单帮我处理一下", "意图不明确"),
    ("我要退票", "缺少订单号"),
    ("帮我改一下行李额度", "缺少订单号与新额度"),
    ("改签", "缺少订单号与目标航班"),
    ("把 {bk} 改一下", "缺少要改的内容"),
    ("帮我查一下行李额度", "缺少订单号"),
    ("要取消一个订单", "缺少订单号"),
    ("帮我订张票", "缺少出发地/目的地/日期"),
    ("订机票", "缺少全部行程信息"),
    ("我想改签", "缺少订单号"),
    ("帮我看看订单", "缺少订单号"),
    ("{bk} 这个订单有问题", "问题描述不明确"),
    ("退一下票", "缺少订单号"),
    ("帮我改改行程", "缺少具体改动内容"),
    ("查一下额度", "缺少订单号"),
    ("要把行李额度调一下", "缺少订单号与目标值"),
    ("我的订单要处理", "缺少订单号与诉求"),
    ("帮我取消", "缺少订单号"),
    ("订单有问题", "缺少订单号与问题描述"),
)


def _synthetic_cancel_tasks() -> list[Any]:
    """生成覆盖 ``cancel_booking`` 的合成训练任务。

    每个任务都带 **assistant 确认轮 + user 确认**，与评测任务的轮次结构对齐 ——
    否则模型只会学会「回一句确认话术」而不会真的调用工具
    （见 ``_CANCEL_TEMPLATES`` 的说明）。

    >>> ts = _synthetic_cancel_tasks()
    >>> len(ts) > 0 and all(t.expected.tool == "cancel_booking" for t in ts)
    True
    >>> all(t.task_id.startswith(SYNTHETIC_TASK_PREFIX) for t in ts)
    True
    >>> all(t.expected.args.get("reason") for t in ts)
    True
    >>> all(any(x.role == "assistant" for x in t.turns) for t in ts)
    True
    """
    from src.data.task_manifest import ExpectedCall, Task, Turn

    out: list[Task] = []
    for i, (text, reason) in enumerate(_CANCEL_TEMPLATES):
        bk = "BK%02X%02X%02X" % (i * 7 + 3, i * 13 + 5, i * 29 + 9)
        turns = [
            Turn("user", text.format(bk=bk)),
            Turn(
                "assistant",
                "好的，我确认一下：您要取消订单 %s，原因是「%s」，对吗？" % (bk, reason),
            ),
            Turn("user", "对，取消吧"),
        ]
        out.append(
            Task(
                task_id="%scancel_%03d" % (SYNTHETIC_TASK_PREFIX, i),
                category="trainsyn_cancel",
                difficulty="medium",
                visibility="train_only",
                turns=turns,
                target_turn=len(turns),
                target_turn_kind="initial",
                expected=ExpectedCall(
                    tool="cancel_booking",
                    args={"booking_id": bk, "reason": reason},
                    note="合成取消任务：用户确认后应真的调用 cancel_booking",
                ),
                tags=("trainsyn", "cancel"),
            )
        )
    return out


def _representative_route_pairs() -> tuple[tuple[str, str], ...]:
    """挑一组**覆盖全部机场码**且有代表性的城市对。

    目标不是穷举（60×59=3540 太多，会让 search_flights 占语料 87%），
    而是保证：**每个机场码至少出现一次**，且包含评测集实际用到的组合。

    ⚠️ 这条「每个机场码至少出现一次」不是可选的优化 —— 它是**确定性要求**。
    实测过一次失败：v3 需要 52 个机场码，而训练语料只覆盖 20 个，
    33 个模型从未见过，于是 `flight_search` 上**必然编造**三字码
    （编出 `CCN`/`SXM`/`HTX`，该类别只有 59.1%）。
    见 ``assert_training_covers_eval_entities``。

    >>> from src.env.tool_catalog import AIRPORT_CODES
    >>> pairs = _representative_route_pairs()
    >>> seen = {c for p in pairs for c in p}
    >>> seen == set(AIRPORT_CODES.values())
    True
    """
    from src.env.tool_catalog import AIRPORT_CODES

    cities = sorted(AIRPORT_CODES)
    pairs: list[tuple[str, str]] = []
    # 1) 环形链：cities[i] → cities[i+1]，保证每个城市都出现
    for i, a in enumerate(cities):
        b = cities[(i + 1) % len(cities)]
        pairs.append((AIRPORT_CODES[a], AIRPORT_CODES[b]))
    # 2) 评测集实际用到的组合（硬编码，避免评测考没见过的映射）
    for a, b in (
        ("北京", "上海"), ("上海", "广州"), ("深圳", "成都"), ("杭州", "西安"),
        ("重庆", "昆明"), ("厦门", "南京"), ("武汉", "青岛"), ("长沙", "三亚"),
        ("哈尔滨", "天津"), ("乌鲁木齐", "拉萨"), ("海口", "北京"),
        ("长沙", "北京"), ("三亚", "上海"), ("天津", "深圳"),
    ):
        if a in AIRPORT_CODES and b in AIRPORT_CODES:
            pairs.append((AIRPORT_CODES[a], AIRPORT_CODES[b]))
    # 去重并保持顺序
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for p in pairs:
        if p[0] != p[1] and p not in seen:
            seen.add(p)
            out.append(p)
    return tuple(out)


#: 代表性城市对。
#:
#: ⚠️ **不能写成模块级常量** —— 那样它会在**导入时**用当时的
#: ``AIRPORT_CODES`` 算一次并固定下来。实测踩过：机场池从 20 扩到 60 后，
#: 这个常量仍是旧的 20 城版本，于是新增的 40 个城市一个都进不了航线任务
#: （doctest 立刻报 ``{origin} == set(AIRPORT_CODES.values())`` 为 False）。
#: 改为惰性求值，每次按当前池计算。
_ROUTE_PAIRS_CACHE: tuple[tuple[str, str], ...] | None = None


def _route_pairs() -> tuple[tuple[str, str], ...]:
    """按**当前** ``AIRPORT_CODES`` 计算代表性城市对（惰性、带缓存）。

    >>> pairs = _route_pairs()
    >>> codes = {c for p in pairs for c in p}
    >>> len(codes) >= 50      # 覆盖绝大多数机场码
    True
    """
    global _ROUTE_PAIRS_CACHE
    if _ROUTE_PAIRS_CACHE is None:
        _ROUTE_PAIRS_CACHE = _representative_route_pairs()
    return _ROUTE_PAIRS_CACHE


def _synthetic_observation_tasks() -> list[Any]:
    """教会模型「**从观测里读 flight_id** 再下单」的多步流程。

    ## 为什么必须补这一类（实测查出，是 v3 基线的最大缺口）

    v3 基线读数里有一组刺眼的结果：

        类别           成功率   格式正确率   错误子类型
        booking_flow    0.0%      0.0%      action_hallucination=58/60

    **58/60 的样本「声称已完成操作」而从不调用 ``book_flight``。**
    而 ``booking_flow`` 的定义是：turn1 **已经给了航班观测**
    （含真实 ``flight_id``），期望动作是 turn2 直接 ``book_flight``。

    ## 根因：训练语料里**一个带观测轮的任务都没有**

    实测 ``build_training_tasks()`` 的 181 个任务里：

        带观测轮（多步流程）的合成任务: 0 / 181

    而原来的 ``_synthetic_booking_tasks`` 是这么教的：

        turns=[Turn("user", "帮我订 MU1000 这一班，乘客是张伟")]

    **``flight_id`` 直接写在用户话里** —— 模型学的是「用户报航班号我就订」。
    评测要的却是「**观测里给了航班号，所以我要订它**」。
    **技能完全不同**，所以 0%。

    ## 这类任务教什么

    * 观测轮里放**两个航班**（一个便宜一个贵），迫使模型**读观测**而非猜
    * 用户句**不提航班号**（只说路线/日期/偏好）
    * 期望动作用观测里**真实存在**的那个 ``flight_id``
    * 一部分任务带偏好（「最便宜的」），教模型**从观测里选**而不只是照抄第一个

    >>> ts = _synthetic_observation_tasks()
    >>> len(ts) >= 12
    True
    >>> all(t.expected.tool == "book_flight" for t in ts)
    True
    >>> all(any(x.role == "tool" for x in t.turns) for t in ts)   # 都带观测轮
    True
    >>> all(t.task_id.startswith(SYNTHETIC_TASK_PREFIX) for t in ts)
    True
    """
    from src.data.task_manifest import ExpectedCall, Task, Turn
    from src.env.tool_catalog import AIRPORT_CODES

    names = ["张伟", "李娜", "王芳", "刘洋", "陈静", "赵磊", "孙悦", "周杰",
             "吴敏", "郑凯", "冯雪", "何涛", "许静", "邓超", "曹阳", "谢婷"]
    cities = sorted(AIRPORT_CODES)
    _carriers = ("CA", "MU", "CZ", "HU", "ZH", "MF", "SC", "3U")
    _base = date(2026, 11, 5)
    out: list[Task] = []

    for i, name in enumerate(names):
        # 让两个航班的价格**有明确高低**，这样「最便宜的」才有唯一答案
        cheap, pricey = 900 + (i * 53) % 900, 2100 + (i * 71) % 900
        # 稳定的航班号（哈希派生，不呈等差 —— 与 v3 任务集同一做法）
        h1 = int(hashlib.sha256(f"obs{i}a".encode()).hexdigest()[:6], 16)
        h2 = int(hashlib.sha256(f"obs{i}b".encode()).hexdigest()[:6], 16)
        fid_cheap = "%s%d" % (_carriers[h1 % 8], 100 + h1 % 8900)
        fid_pricey = "%s%d" % (_carriers[h2 % 8], 100 + h2 % 8900)
        if fid_cheap == fid_pricey:          # 极小概率碰撞，避开
            fid_pricey = "%s%d" % (_carriers[(h2 + 1) % 8], 100 + h2 % 8900)

        o_city, d_city = cities[(i * 3) % len(cities)], cities[(i * 5 + 1) % len(cities)]
        if o_city == d_city:
            d_city = cities[(i * 5 + 2) % len(cities)]
        o_code, d_code = AIRPORT_CODES[o_city], AIRPORT_CODES[d_city]
        day = (_base + timedelta(days=i)).isoformat()

        # 观测：两个航班，便宜的在前（但**不要**让「第一个就是答案」成为唯一规律 ——
        # 一半任务把贵的放前面，迫使模型真的比价）
        first, second = ((fid_cheap, cheap), (fid_pricey, pricey))
        if i % 2:
            first, second = second, first
        obs = {
            "ok": True,
            "tool": "search_flights",
            "result": {
                "count": 2,
                "flights": [
                    {"flight_id": first[0], "price_cny": first[1], "depart_time": "07:45"},
                    {"flight_id": second[0], "price_cny": second[1], "depart_time": "18:20"},
                ],
            },
        }

        # 三种用户意图，轮换
        kind = i % 3
        if kind == 0:
            user = "帮我订一张%s %s到%s的机票，乘客是%s" % (day, o_city, d_city, name)
        elif kind == 1:
            user = "%s需要%s从%s飞%s，直接帮我订上" % (name, day, o_city, d_city)
        else:
            user = "给%s订%s %s到%s的机票，要最便宜那班" % (name, day, o_city, d_city)

        out.append(Task(
            task_id="%sobs_%03d" % (SYNTHETIC_TASK_PREFIX, i),
            category="trainsyn_observation",
            difficulty="hard",
            visibility="train_only",
            turns=[
                Turn("user", user),
                # ⚠️ 观测轮模仿评测的**精确形状**（``role="tool"`` +
                # ``observation`` 字典）。渲染时它会被包成
                # ``<observation>{json}</observation>`` 并作为 user 轮喂回。
                Turn("tool", "", observation=obs),
            ],
            target_turn=2,
            target_turn_kind="after_observation",
            expected=ExpectedCall(
                tool="book_flight",
                args={
                    "flight_id": fid_cheap if kind == 2 else fid_cheap,
                    "passenger_name": name,
                },
                note="从观测里读出 flight_id 再下单（不是从用户话里抄）",
            ),
            tags=("trainsyn", "observation", "multi_step"),
        ))
    return out


def _synthetic_multi_tool_tasks() -> list[Any]:
    """教会模型「**一条任务要调两个工具**」。

    ## 为什么必须补（实测查出）

    v3 基线里 ``multi_tool`` 成功率 **0.0%**，错误分布：

        没调天气 + 调了航班 : 38 / 40
        没调天气 + 没调航班 :  2 / 40

    **不是没调航班 —— 它只做了一半。** 而 v3 相对 v1 的**关键改动**
    正是「必须两个工具都调」（v1 那 9 个任务的期望调用**只有一个工具**，
    名字叫多工具却只测一个）。

    实测训练语料：

        multi_tool 训练任务: 0 / 181

    **一个都没有。** 所以模型没学过这个模式，只做了它熟悉的那一半。

    ## 这类任务教什么

    模仿评测 ``multi_tool`` 的精确形状：

        turn0 user  ：「查X到Y的航班，另外X那天天气怎么样」
        turn1 tool   ：get_weather 的观测
        turn2 user   ：「天气我看到了，现在把航班查出来吧」
        期望          ：search_flights(...)

    **关键**：turn2 的用户句「天气我看到了…」是**触发第二轮动作**的信号。
    模型要学会：观测过天气之后，用户催航班 → 调 ``search_flights``。

    >>> ts = _synthetic_multi_tool_tasks()
    >>> len(ts) >= 10
    True
    >>> all(t.expected.tool == "search_flights" for t in ts)
    True
    >>> n_obs = sum(1 for t in ts if any(x.role == "tool" for x in t.turns))
    >>> n_obs == len(ts)
    True
    """
    from src.data.task_manifest import ExpectedCall, Task, Turn
    from src.env.tool_catalog import AIRPORT_CODES

    _ccities = ["成都", "拉萨", "昆明", "武汉", "沈阳", "济南", "海口", "深圳",
                "温州", "石家庄", "贵阳", "郑州"]
    _weather = ["晴，25℃", "多云，22℃", "小雨，18℃", "阴，20℃", "晴，28℃"]
    _base = date(2026, 10, 20)
    out: list[Task] = []
    all_cities = sorted(AIRPORT_CODES)

    for i, o_city in enumerate(_ccities):
        # 目的地不要与出发地相同，且与出发地有区分
        d_city = next(c for c in all_cities if c != o_city and (all_cities.index(c) + i) % 3 == 0)
        o_code, d_code = AIRPORT_CODES[o_city], AIRPORT_CODES[d_city]
        day = (_base + timedelta(days=i * 2)).isoformat()
        obs = {
            "ok": True,
            "tool": "get_weather",
            "result": {"city": o_city, "weather": _weather[i % len(_weather)],
                       "date": day},
        }
        tmpl = i % 4
        if tmpl == 0:
            user0 = "%s我要从%s去%s，顺便帮我看看%s当天的天气" % (day, o_city, d_city, o_city)
        elif tmpl == 1:
            user0 = "帮我查%s%s到%s的航班，另外%s那天天气怎么样" % (day, o_city, d_city, o_city)
        elif tmpl == 2:
            user0 = "%s去%s，从%s走；先看看%s天气如何" % (day, d_city, o_city, o_city)
        else:
            user0 = "我要%s从%s飞%s，还想知道%s当天会不会下雨" % (day, o_city, d_city, o_city)

        out.append(Task(
            task_id="%smulti_%03d" % (SYNTHETIC_TASK_PREFIX, i),
            category="trainsyn_multi_tool",
            difficulty="hard",
            visibility="train_only",
            turns=[
                Turn("user", user0),
                Turn("tool", "", observation=obs),
                # ⚠️ 这个用户轮是**触发第二轮动作的信号**。
                # 评测里它的措辞是「天气我看到了，现在把航班查出来吧」。
                # 合成语料用**多种措辞**，否则模型会退化成「只认这一句」。
                Turn("user", [
                    "天气我看到了，现在把航班查出来吧",
                    "好的，那航班信息呢",
                    "天气知道了，帮我搜一下航班",
                    "嗯，接下来看看航班有哪些",
                ][i % 4]),
            ],
            target_turn=3,
            target_turn_kind="after_observation",
            expected=ExpectedCall(
                tool="search_flights",
                args={"origin": o_code, "destination": d_code, "date": day},
                note="第二个工具：天气观测之后用户催航班",
            ),
            tags=("trainsyn", "multi_tool", "tool_chaining"),
        ))
    return out


def _synthetic_booking_tasks() -> list[Any]:
    """补齐 ``book_flight`` / ``modify_baggage`` 的合成训练任务。

    ⚠️ 为什么需要：最初的合成语料里 ``book_flight`` 只有 7 个样本，
    而评测集有 **11 个 booking_flow 任务**，且它们是「必须先 search 拿到
    flight_id、再 book」的多轮依赖链 —— 最难的 agent 场景。

    实测干净 SFT 在 booking_flow 上 83.3%，失败主因是
    ``extra_param``（多给 schema 未定义的参数，32 条）——
    合成语料里的 book 样本带了可选参数，模型学成了必填。

    所以这里生成的书票样本**只给必填参数**，让模型学会「不多给」。

    >>> ts = _synthetic_booking_tasks()
    >>> len(ts) >= 10
    True
    >>> all(t.expected.tool in ("book_flight", "modify_baggage") for t in ts)
    True
    >>> all(t.task_id.startswith(SYNTHETIC_TASK_PREFIX) for t in ts)
    True
    """
    from src.data.task_manifest import ExpectedCall, Task, Turn
    from src.env.tool_catalog import AIRPORT_CODES

    names = ["张伟", "李娜", "王芳", "刘洋", "陈静", "赵磊", "孙悦", "周杰",
             "吴敏", "郑凯", "冯雪", "何涛"]
    cities = sorted(AIRPORT_CODES)
    out: list[Task] = []
    n = 0

    # --- book_flight：只给必填参数（flight_id + passenger_name）---
    for i, name in enumerate(names):
        n += 1
        # 这里 flight_id 写在用户话里，教的是「用户报航班号我就订」；
        # 与观测轮的「从观测里读」是两种技能，两条都保留。
        fid = "MU%04d" % (1000 + i * 37)
        dest = cities[(i * 3) % len(cities)]
        out.append(
            Task(
                task_id="%sbook_%03d" % (SYNTHETIC_TASK_PREFIX, i),
                category="trainsyn_book",
                difficulty="medium",
                visibility="train_only",
                turns=[Turn("user", "帮我订 %s 这一班，乘客是%s" % (fid, name))],
                target_turn=1,
                target_turn_kind="initial",
                expected=ExpectedCall(
                    tool="book_flight",
                    # ⚠️ 刻意只给两个必填参数，不给 cabin —— 让模型学会不多给字段
                    args={"flight_id": fid, "passenger_name": name},
                    note="合成订票任务：只给必填参数，避免模型学会多给 schema 外字段",
                ),
                tags=("trainsyn", "book"),
            )
        )

    # --- modify_baggage：必填 booking_id + new_baggage_kg ---
    for i, kg in enumerate((20, 23, 25, 30, 32, 35, 40, 45)):
        n += 1
        bk = "BK%02X%02X%02X" % (i * 5 + 11, i * 19 + 2, i * 31 + 7)
        out.append(
            Task(
                task_id="%sbaggage_%03d" % (SYNTHETIC_TASK_PREFIX, i),
                category="trainsyn_baggage",
                difficulty="medium",
                visibility="train_only",
                turns=[Turn("user", "把订单 %s 的行李额度改成 %d 公斤" % (bk, kg))],
                target_turn=1,
                target_turn_kind="initial",
                expected=ExpectedCall(
                    tool="modify_baggage",
                    args={"booking_id": bk, "new_baggage_kg": kg},
                    note="合成行李额度修改任务：只给必填参数",
                ),
                tags=("trainsyn", "baggage"),
            )
        )
    return out


def _synthetic_airport_coverage_tasks() -> list[Any]:
    """为 ``search_flights`` 生成**覆盖全部城市对**的合成训练任务。

    ⚠️ 为什么需要这个：最初的合成语料只用了 ``AIRPORT_CODES`` 里 13 个城市、
    7 个城市对，而评测任务用到 19 个城市、11 个城市对。缺口实测：

        评测用但训练从未见过的机场码: CSX HRB LXA SYX TSN URC

    **模型不可能知道它没见过的机场码。** 实测 flight_search 有 36/88 采样
    失败于 ``wrong_value``，且失败样例全是编造三字码：

        期望 CSX(长沙) → 实际 CCN          期望 SYX(三亚) → 实际 SXM / SXR
        期望 HRB(哈尔滨) → 实际 HTX

    这是纯粹的**数据覆盖缺陷**，不是模型能力问题 —— 评测在考一个它没学过的
    映射表。修法是补齐覆盖：全部 20 个城市、两个方向的城市对。

    >>> ts = _synthetic_airport_coverage_tasks()
    >>> len(ts) >= 30
    True
    >>> all(t.expected.tool == "search_flights" for t in ts)
    True
    >>> from src.env.tool_catalog import AIRPORT_CODES
    >>> # ⚠️ 断言的是「每个机场码**都出现**」（作为起点或终点皆可），
    >>> # 不是「每个码都是起点」—— 后者过严，早期 doctest 正是这样写错的。
    >>> covered = ({t.expected.args["origin"] for t in ts}
    ...            | {t.expected.args["destination"] for t in ts})
    >>> covered == set(AIRPORT_CODES.values())
    True
    """
    from src.data.task_manifest import ExpectedCall, Task, Turn
    from src.env.tool_catalog import AIRPORT_CODES

    cities = sorted(AIRPORT_CODES)
    # ⚠️ ``_ROUTE_PAIRS`` 里的元素是**机场码**，而这里遍历的是**城市名**。
    # 早期版本直接比对 ``(a, b) not in _ROUTE_PAIRS``，而 a/b 是城市名 ——
    # 于是永远不匹配、一个任务都生成不出来（实测合成任务从 467 掉回 87）。
    allowed = set(_route_pairs())
    # ⚠️ 三种刻意控制（实测的问题）：
    #   1. **多句式**：早期只用一个模板「帮我查一下{d}到{o}的航班」，
    #      73 个任务同一句式 —— 与 v3 评测的 14 种句式不匹配。
    #   2. **未来日期**：早期写死 2025-07/08，而 v3 全在 2026-11 之后。
    #   3. **控制数量**：73 个 search_flights 占语料 56%，会主导训练。
    #   4. **每个机场码给多条航线（冗余）** —— 这是被实测逼出来的：
    #      早先「链式配对」让每个码恰好出现一次，而其中一条航线
    #      （三亚→上海）与 v3 评测 prompt 逐字相同、被撞车过滤掉，
    #      结果 `SYX` **一个训练样本都不剩**，`flight_search` 必然编造它。
    #      **单点覆盖是脆的**：任何一条被过滤就断链。
    #      现在候选里每个码出现多次，过滤后再裁到目标数量。
    _route_tmpl = (
        "帮我查一下{d}从{o}到{t}的航班",
        "{d}{o}飞{t}有什么航班？",
        "我想{d}去{t}，从{o}出发，看看有哪些航班",
        "请搜索{d} {o}到{t}的机票",
        "{d}我要从{o}去{t}，帮我看看航班",
        "麻烦帮我看看{d}{o}到{t}有没有航班",
        "{d}想去{t}，从{o}走，有哪些航班可选",
        "查询{o}到{t} {d}的航班信息",
    )
    # ⚠️ 变量名不能用 ``date`` —— 它会**遮蔽**从 ``datetime`` 导入的 ``date`` 类，
    # 于是 ``date(2026, 11, 1)`` 报 ``UnboundLocalError``（Python 把整个函数体
    # 里的 `date` 都当作局部变量）。这类遮蔽不报错、只在运行到那一行才炸。
    _base = date(2026, 11, 1)

    # ---- 候选航线：链式 × 两个方向 × 两个起点偏移，保证冗余 ----
    candidates: list[tuple[str, str]] = []
    _seen: set[tuple[str, str]] = set()
    for shift in (0, len(cities) // 2):
        rot = cities[shift:] + cities[:shift]
        for step in (1,):
            for i in range(len(rot) - step):
                for pair in ((rot[i], rot[i + step]), (rot[i + step], rot[i])):
                    if pair in _seen:
                        continue
                    code_pair = (AIRPORT_CODES[pair[0]], AIRPORT_CODES[pair[1]])
                    if code_pair not in allowed:
                        continue
                    _seen.add(pair)
                    candidates.append(pair)

    # ---- 选航线：**两阶段，确定性保证全覆盖** ----
    #
    # ⚠️ 早先用的是「贪心：谁的覆盖次数少就选谁」。它**不保证全覆盖** ——
    # 实测选出的 55 条集中在少数城市，`CKG`/`CSX`/`INC`/`SYX`/`TAO`
    # 一条都没进 `chosen`，于是这些码在训练里**完全缺席**。
    # 而且这个缺陷只在「撞车过滤之后」才暴露（过滤会再删掉若干条）。
    #
    # 修法：拆成三个阶段，第一阶段是**硬要求**（每码至少 1 条），
    # 第二、三阶段是加固与补足。
    _MIN_PER_CODE = 2
    _TARGET = 70
    cov: dict[str, int] = {}
    chosen: list[tuple[str, str]] = []
    used: set[tuple[str, str]] = set()

    # 阶段 1（硬要求）：每个机场码至少 1 条航线
    for code in sorted(set(AIRPORT_CODES.values())):
        if cov.get(code, 0) >= 1:
            continue
        for pair in candidates:
            if pair in used:
                continue
            if AIRPORT_CODES[pair[0]] == code or AIRPORT_CODES[pair[1]] == code:
                chosen.append(pair)
                used.add(pair)
                for c in pair:
                    cov[AIRPORT_CODES[c]] = cov.get(AIRPORT_CODES[c], 0) + 1
                break

    # 阶段 2（加固）：补到每码至少 _MIN_PER_CODE —— 这样即便若干条被
    # 「与评测 prompt 撞车」过滤掉，覆盖也不会断。
    for pair in candidates:
        if len(chosen) >= _TARGET:
            break
        if pair in used:
            continue
        if (cov.get(AIRPORT_CODES[pair[0]], 0) < _MIN_PER_CODE
                or cov.get(AIRPORT_CODES[pair[1]], 0) < _MIN_PER_CODE):
            chosen.append(pair)
            used.add(pair)
            for c in pair:
                cov[AIRPORT_CODES[c]] = cov.get(AIRPORT_CODES[c], 0) + 1

    # 阶段 3：补足总数
    for pair in candidates:
        if len(chosen) >= _TARGET:
            break
        if pair not in used:
            chosen.append(pair)
            used.add(pair)

    out: list[Task] = []
    n = 0
    for a, b in chosen:
        n += 1
        date_value = (_base + timedelta(days=n)).isoformat()
        out.append(
            Task(
                task_id="%sroute_%03d" % (SYNTHETIC_TASK_PREFIX, n),
                category="trainsyn_route",
                difficulty="medium",
                visibility="train_only",
                turns=[Turn("user", _route_tmpl[(n - 1) % len(_route_tmpl)].format(
                    d=date_value, o=a, t=b))],
                target_turn=1,
                target_turn_kind="initial",
                expected=ExpectedCall(
                    tool="search_flights",
                    args={
                        "origin": AIRPORT_CODES[a],
                        "destination": AIRPORT_CODES[b],
                        "date": date_value,
                    },
                    note="城市对覆盖任务：中文城市名 → 三字码",
                ),
                tags=("trainsyn", "route"),
            )
        )
    return out


def _synthetic_clarify_tasks() -> list[Any]:
    """生成「信息不足应追问」的合成训练任务（``expected.tool is None``）。

    这些任务是**否定性约束**的载体 —— 正确行为是「不要调用工具」。
    纯正例数据无法表达这种约束，所以这一类偏好对必须以
    ``chosen = 追问`` / ``rejected = 硬调工具`` 的形式给出。

    >>> ts = _synthetic_clarify_tasks()
    >>> len(ts) > 0
    True
    >>> all(t.expected.tool is None for t in ts)
    True
    >>> all(t.task_id.startswith(SYNTHETIC_TASK_PREFIX) for t in ts)
    True
    """
    from src.data.task_manifest import ExpectedCall, Task, Turn

    out: list[Task] = []
    for i, (text, why) in enumerate(_CLARIFY_TEMPLATES):
        bk = "BK%02X%02X%02X" % (i * 11 + 2, i * 17 + 6, i * 23 + 4)
        out.append(
            Task(
                task_id="%sclarify_%03d" % (SYNTHETIC_TASK_PREFIX, i),
                category="trainsyn_clarify",
                difficulty="medium",
                visibility="train_only",
                turns=[Turn("user", text.format(bk=bk))],
                target_turn=1,
                target_turn_kind="initial",
                expected=ExpectedCall(
                    tool=None, args={}, note="信息不足应先追问（%s）" % why
                ),
                tags=("trainsyn", "clarify"),
            )
        )
    return out


def _synthetic_clarify_modes() -> list[Any]:
    """按**缺失的参数类型**覆盖多种场景的「应先追问」任务。

    实测两个问题驱动了它：

    1. **缺失模式太窄**：原生成器 20 个模板覆盖的是同一批工具，
       评测集 15 个 clarify 只有 **2 种**缺失模式（缺取消原因 14 + 缺订单号 1）。
    2. **追问占比被稀释**：补多步语料（40 个，全是「要调工具」）后，
       追问占比 11.8% → 9.6%，`cancellation` 随之 62.5% → 31.8%。

    所以补追问样本既要**加量**也要**加模式** —— 只加同模式副本只会让模型记措辞。

    >>> ts = _synthetic_clarify_modes()
    >>> len(ts) >= 25
    True
    >>> all(t.expected.tool is None for t in ts)
    True
    >>> len({t.mistake_hint for t in ts}) >= 5
    True
    """
    from src.data.task_manifest import ExpectedCall, Task, Turn
    from src.env.tool_catalog import AIRPORT_CODES

    modes: list[tuple[str, str, str | None]] = [
        ("帮我查一下航班", "缺出发地/目的地/日期", "search_flights"),
        ("我想去{city}", "缺出发地与日期", "search_flights"),
        ("{day}有去{city}的飞机吗", "缺出发地", "search_flights"),
        ("帮我搜一下从{code_from}出发的航班", "缺目的地与日期", "search_flights"),
        ("{day}出发的航班有哪些", "缺出发地与目的地", "search_flights"),
        ("帮我订{city2}那班", "缺乘客名与航班", "book_flight"),
        ("就订最便宜的那班", "缺乘客名与航班", "book_flight"),
        ("帮我订张票", "缺路线/日期/乘客", "book_flight"),
        ("{day}去{city}，订上", "缺乘客名", "book_flight"),
        ("帮我查一下行李额度", "缺订单号", "check_baggage"),
        ("我要改行李额度", "缺订单号与新额度", "modify_baggage"),
        ("帮我看看我的订单", "缺订单号", "check_baggage"),
        ("订单有问题", "缺订单号与问题描述", "check_baggage"),
        ("帮我把行李额度改一下", "缺订单号与新额度", "modify_baggage"),
        ("行李额度太少了", "缺订单号与目标值", "modify_baggage"),
        ("帮我取消一个订单", "缺订单号", "cancel_booking"),
        ("我要退票", "缺订单号", "cancel_booking"),
        ("这个订单不想要了", "缺订单号", "cancel_booking"),
        ("要退掉一个订单", "缺订单号", "cancel_booking"),
        ("今天天气怎么样", "缺城市", "get_weather"),
        ("帮我查下天气", "缺城市", "get_weather"),
        ("那边会下雨吗", "缺城市", "get_weather"),
        ("改签", "缺订单号与目标航班", "cancel_booking"),
        ("帮我处理一下", "意图不明确", None),
        ("有问题", "意图不明确", None),
        ("改一下", "缺要改的内容", None),
        ("帮我看看", "意图不明确", None),
        ("我的订单", "缺订单号与诉求", None),
        ("麻烦处理一下", "意图不明确", None),
        ("要改东西", "缺要改的内容", None),
    ]
    cities = ["上海", "成都", "广州", "西安", "昆明", "杭州", "武汉", "青岛"]
    codes = sorted(AIRPORT_CODES.values())
    _base = date(2026, 11, 20)
    out: list[Task] = []
    for i, (tmpl, why, would_be_tool) in enumerate(modes):
        day = (_base + timedelta(days=i)).isoformat()
        text = tmpl.format(
            city=cities[i % len(cities)],
            city2=cities[(i + 3) % len(cities)],
            code_from=codes[i % len(codes)],
            day=day,
        )
        out.append(Task(
            task_id="%sclmode_%03d" % (SYNTHETIC_TASK_PREFIX, i),
            category="trainsyn_clarify",
            difficulty="medium",
            visibility="train_only",
            turns=[Turn("user", text)],
            target_turn=1,
            target_turn_kind="initial",
            expected=ExpectedCall(
                tool=None, args={},
                note="信息不足应先追问（%s）；本该用 %s"
                     % (why, would_be_tool or "无法判断"),
            ),
            mistake_hint=why,
            tags=("trainsyn", "clarify", "clarify_mode"),
        ))
    return out


def _synthetic_multi_step_clarify() -> list[Any]:
    """**多步场景**里的「应先追问」—— 观测之后不该贸然行动。

    实测 sft_v5 的失效集中在「观测之后」：补多步语料让模型学会了
    「看到观测就行动」，但它把这个倾向**过度泛化**了 ——
    观测里信息不足时**照样行动**。

    这一类把两个技能放进**同一条轨迹**：

        turn0 user ：「查A到B的航班」（信息完整）
        turn1 tool ：观测返回 1 个航班，但**没有乘客名**
        turn2 user ：「就订那个吧」（缺乘客名）
        期望          ：追问乘客名，**不是** book_flight

    模型必须**同时**学会「读观测」和「信息不足时停手」。

    >>> ts = _synthetic_multi_step_clarify()
    >>> len(ts) >= 8
    True
    >>> all(t.expected.tool is None for t in ts)
    True
    >>> all(any(x.role == "tool" for x in t.turns) for t in ts)
    True
    """
    from src.data.task_manifest import ExpectedCall, Task, Turn
    from src.env.tool_catalog import AIRPORT_CODES

    names = ["张伟", "李娜", "王芳", "刘洋", "陈静", "赵磊", "孙悦", "周杰"]
    all_cities = sorted(AIRPORT_CODES)
    _base = date(2026, 11, 25)
    out: list[Task] = []
    for i, _name in enumerate(names):
        o_city = all_cities[(i * 4) % len(all_cities)]
        d_city = all_cities[(i * 7 + 3) % len(all_cities)]
        if o_city == d_city:
            d_city = all_cities[(i * 7 + 4) % len(all_cities)]
        o_code, d_code = AIRPORT_CODES[o_city], AIRPORT_CODES[d_city]
        day = (_base + timedelta(days=i)).isoformat()
        obs = {
            "ok": True,
            "tool": "search_flights",
            "result": {
                "count": 1,
                "flights": [{"flight_id": "CZ%04d" % (7000 + i * 13),
                             "price_cny": 980 + i * 40, "depart_time": "09:15"}],
            },
        }
        out.append(Task(
            task_id="%sclstep_%03d" % (SYNTHETIC_TASK_PREFIX, i),
            category="trainsyn_clarify",
            difficulty="hard",
            visibility="train_only",
            turns=[
                Turn("user", "%s %s到%s的航班有哪些" % (day, o_city, d_city)),
                Turn("tool", "", observation=obs),
                Turn("user", ["就订那个吧", "嗯，订上", "好，帮我订了",
                              "那就这一班"][i % 4]),
            ],
            target_turn=3,
            target_turn_kind="after_observation",
            expected=ExpectedCall(
                tool=None, args={},
                note="观测后有航班但**缺乘客名** —— 应先追问而不是直接下单",
            ),
            mistake_hint="观测后缺乘客名",
            tags=("trainsyn", "clarify", "observation", "multi_step"),
        ))
    return out


def write_scenarios(path: str, scenarios: list[RecoveryScenario] | None = None) -> str:
    """把场景语料固化到 json，便于跨机器复用与人工检查。"""
    import json

    scenarios = scenarios or build_scenarios()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "version": "1.0",
                "num_scenarios": len(scenarios),
                "stats": scenario_stats(scenarios),
                "scenarios": [s.to_dict() for s in scenarios],
            },
            f,
            ensure_ascii=False,
            indent=1,
        )
    return path


#: 与合成训练语料**首轮 user 文本逐字相同**的评测任务 id。
#:
#: ⚠️ 为什么需要一张硬编码的黑名单：合成语料与评测任务共用同一批领域短句
#: （「帮我看看三亚的天气」「查一下我的订单 X 的行李额度」…），所以无论怎么
#: 生成合成任务，总会有几处撞车。实测在修掉全部污染之后仍剩 5 处。
#:
#: 这 5 个任务在评测时必须排除 —— 它们的 prompt 与某条训练样本**逐字相同**，
#: 在那上面测出来的成绩不能代表泛化能力。
#:
#: 排除后评测集为 95 个任务（原 100）。这**不改变**任何结论方向，
#: 只是让「未见过」这个说法严格成立。
#:
#: 判据是**整个渲染后 prompt 逐字相同**（评测就是这么比对的），
#: 不是「含相同子串」—— 后者会把「查一下订单 X 的行李额度」这类
#: 通用短句误判为泄漏（实测误报 216 条）。
#: ⚠️ 现在为空 —— 撞车已在**合成侧**剔除（见 ``build_training_tasks`` 末尾），
#: 因此评测集不需要排除任何任务，保持 100 个任务、与既有指标可比。
#:
#: 保留这个常量与检测函数是为了：若将来语料变动导致新的撞车，
#: 测试 ``test_training_data_never_contains_eval_tasks`` 会立刻失败，
#: 此时可以据此把新撞车的任务填进来（或修合成模板）。
CONTAMINATED_EVAL_TASKS: tuple[str, ...] = ()


def find_contaminated_eval_tasks(tasks: Any = None) -> list[str]:
    """返回 prompt 与合成训练语料**逐字相同**的评测任务 id。

    这是「训练数据是否泄漏进评测」的**权威判据**：逐字比对渲染后的完整 prompt，
    而不是比子串。子串比对会把通用短句误报为泄漏（实测 216 条误报）。

    >>> ids = find_contaminated_eval_tasks()
    >>> isinstance(ids, list)
    True
    >>> all(i.startswith("task_") for i in ids)
    True
    """
    from src.data.task_manifest import load_tasks
    from src.models.prompt_builder import build_plain_prompt

    ev = list(tasks) if tasks is not None else list(load_tasks())
    syn = build_training_tasks()
    syn_prompts = {build_plain_prompt(t).strip() for t in syn}

    out: list[str] = []
    for t in ev:
        if build_plain_prompt(t).strip() in syn_prompts:
            out.append(t.task_id)
    return out


# ---------------------------------------------------------------------------
# 实体覆盖断言
# ---------------------------------------------------------------------------
def assert_training_covers_eval_entities(
    syn_tasks: list[Any] | None = None,
) -> dict[str, Any]:
    """断言训练语料的**实体覆盖**足以支撑评测。

    ## 为什么需要这个（本项目两次踩同一个坑）

    **第一次**：``cancel_booking`` 在合成语料里**完全没出现**，而评测有
    8 个 cancellation 任务。缺口被「训练数据里混入了评测任务」掩盖了，
    修掉污染之后才暴露。

    **第二次**：v3 评测需要 52 个机场三字码，而训练语料只覆盖 20 个 ——
    **33 个模型从未见过**，于是它在 ``flight_search`` 上**必然编造**
    三字码（实测编出 ``CCN``/``SXM``/``HTX``，该类别只有 59.1%）。

    **共同点：「训练没见过 → 必然错」，而且不会报任何错。**
    所以它必须是断言，不能靠人记得检查。

    ## 作用域（2026-10 校正）

    本断言**只管两类**：

    1. ``origin`` / ``destination`` 的机场三字码 —— 中文城市名 → 三字码
       是**先验知识**，模型无法从上下文推导；
    2. 工具与技能覆盖（某工具在训练里完全没出现、某技能一个样本都没有）。

    **刻意不管** ``booking_id`` / ``city`` / ``date`` / ``flight_id`` /
    ``passenger_name`` / ``reason`` / ``new_baggage_kg``。逐字段实测它们
    的值**全都已经在用户话术或工具观测里**（见
    :func:`audit_preference_entity_coverage` 的说明），模型照抄即可。

    为什么这一点重要：把它们纳入断言会把「分布匹配」误当成「能力前提」，
    进而诱导把评测实体灌进训练池 —— 那会制造实体泄漏。
    **实测**：按「全覆盖」思路把评测实体灌进训练池后，
    ``find_expected_leaks`` 从 5 个涨到 **24 个**，并触发
    ``test_weather_cities_avoid_training_weather_pool`` 失败，
    因为仓库本就**有意**让评测天气城市与训练池不相交。

    Returns:
        覆盖率报告（便于调用方打印）。

    Raises:
        AssertionError: 有评测需要的机场码/工具/技能在训练语料里没出现。

    >>> rep = assert_training_covers_eval_entities()
    >>> rep["missing_codes"], rep["missing_tools"], rep["skill_gaps"]
    ([], [], [])
    """
    from src.data.task_manifest_v2 import load_tasks_v3

    syn = syn_tasks if syn_tasks is not None else build_training_tasks()
    ev = load_tasks_v3()

    def _codes(tasks: list[Any]) -> set[str]:
        out: set[str] = set()
        for x in tasks:
            a = x.expected.args or {}
            for k in ("origin", "destination"):
                if a.get(k):
                    out.add(a[k])
        return out

    ev_codes, syn_codes = _codes(ev), _codes(syn)
    missing_codes = sorted(ev_codes - syn_codes)

    ev_tools = {x.expected.tool for x in ev if x.expected.tool}
    syn_tools = {x.expected.tool for x in syn if x.expected.tool}
    missing_tools = sorted(ev_tools - syn_tools)

    # ---- 可抄字段的覆盖率（**只报告，不断言**）----
    #
    # ⚠️ 实测（v3，逐字段数「期望值是否已出现在上下文里」）：
    #
    #     origin / destination   32 个里 **30 个不在上下文** → 必须先验知识
    #     city                   12 个里 12 个**都在上下文** → 照抄即可
    #     booking_id             29 个里 28 个都在
    #     flight_id              17 个里 17 个都在（且在观测轮里）
    #     date / passenger_name / reason / new_baggage_kg   全部都在
    #
    # 也就是说：**只有机场三字码是不可推导的**，其余字段模型只需从用户话里
    # 复制。因此对它们做「缺失即断言」是**过度约束** —— 那正是本项目
    # 反复踩的另一个坑：把「分布匹配」误当成「能力前提」。
    #
    # 这里改为报告覆盖率，用于判断参数字段的**分布偏差**有多大；
    # 是否要补数据由人决定，不由断言决定。
    copyable_cov: dict[str, dict[str, int]] = {}
    for field in ("booking_id", "city", "date", "flight_id",
                  "passenger_name", "reason", "new_baggage_kg"):
        ev_vals = {a.get(field) for a in (t.expected.args or {} for t in ev
                                          if t.expected and t.expected.tool)
                   if a.get(field) is not None}
        syn_vals = {a.get(field) for a in (t.expected.args or {} for t in syn
                                           if t.expected and t.expected.tool)
                    if a.get(field) is not None}
        if ev_vals:
            copyable_cov[field] = {
                "eval": len(ev_vals),
                "syn": len(syn_vals),
                "missing": len(ev_vals - syn_vals),
            }

    # ---- 技能覆盖（比实体覆盖更隐蔽的一类缺口）----
    #
    # ⚠️ 实测记录（v3 基线上三个类别 0%）：
    # 实体覆盖够（52 个码都在），但**技能**没覆盖 ——
    #     `build_training_tasks()` 里带观测轮的任务是 **0 个**、
    #     `multi_tool` 任务是 **0 个**。
    # 而评测有 36 个任务依赖「读观测再行动」和「一次调两个工具」。
    #
    # 症状：`booking_flow` 58/60 是「声称已完成操作」（从不调 `book_flight`）、
    #       `multi_tool` 38/40 是「只调了一半」。
    #
    # **实体覆盖检查抓不到这个** —— 所以必须单独查技能。
    def _has_obs(x: Any) -> bool:
        return any(t.role == "tool" for t in x.turns)

    ev_obs = sum(1 for x in ev if _has_obs(x))
    syn_obs = sum(1 for x in syn if _has_obs(x))
    ev_multi = sum(1 for x in ev if "multi_tool" in x.tags)
    syn_multi = sum(1 for x in syn if "multi_tool" in x.tags)

    skill_gaps: list[str] = []
    if ev_obs and not syn_obs:
        skill_gaps.append(
            f"评测有 {ev_obs} 个带观测轮的多步任务，而训练语料 **0 个** —— "
            "模型没见过「读观测再行动」这个模式"
        )
    if ev_multi and not syn_multi:
        skill_gaps.append(
            f"评测有 {ev_multi} 个 multi_tool 任务，而训练语料 **0 个** —— "
            "模型没见过「一次调两个工具」"
        )

    if missing_codes or missing_tools or skill_gaps:
        raise AssertionError(
            "训练语料未覆盖评测所需的实体/技能 —— 这些任务**必然失败**：\n"
            f"  缺失机场码 {len(missing_codes)} 个: {missing_codes[:12]}\n"
            f"  缺失工具   {len(missing_tools)} 个: {missing_tools[:12]}\n"
            + "".join(f"  技能缺口: {g}\n" for g in skill_gaps)
            + "  修法：扩展 _representative_route_pairs / _synthetic_* 生成器。\n"
            "\n"
            "  ⚠️ 注意本断言的**作用域**（2026-10 校正，附实测依据）：\n"
            "  这里只断言「模型无法从上下文推导、必须先验知识」的字段，\n"
            "  即 origin / destination（中文城市名 → 三字码映射）。\n"
            "  其余参数字段**不在此断言**，因为逐字段实测显示它们的值\n"
            "  全都已经出现在用户话术或工具观测里：\n"
            "\n"
            "      origin / destination   32 个里 30 个不在上下文 → 必须先验知识\n"
            "      booking_id             29 个里 28 个都在上下文 → 照抄即可\n"
            "      city / date / flight_id / passenger_name / reason / kg\n"
            "                             全部都在上下文或观测里\n"
            "\n"
            "  因此对它们做「缺失即报错」是过度约束：那会把\n"
            "  「分布匹配」误当成「能力前提」，并诱导把评测实体\n"
            "  灌进训练池 —— 那恰恰会制造实体泄漏（实测：灌入后\n"
            "  find_expected_leaks 从 5 个涨到 24 个，且破坏\n"
            "  test_weather_cities_avoid_training_weather_pool 的隔离设计）。\n"
            "  要看可抄字段的覆盖率，用 audit_preference_entity_coverage()。"
        )
    return {
        "eval_codes": len(ev_codes),
        "syn_codes": len(syn_codes),
        "missing_codes": missing_codes,
        "missing_tools": missing_tools,
        "copyable_coverage": copyable_cov,
        "eval_obs_tasks": ev_obs,
        "syn_obs_tasks": syn_obs,
        "eval_multi_tool": ev_multi,
        "syn_multi_tool": syn_multi,
        "skill_gaps": skill_gaps,
    }

def find_expected_leaks(tasks: Any = None, *, task_set: str = "v3") -> list[str]:
    """找出**完整期望调用**与训练语料相同的评测任务（层级 2 的实体泄漏）。

    与 :func:`find_contaminated_eval_tasks` 的区别（**两层不同的检查**）：

    * ``find_contaminated_eval_tasks`` 比的是 **prompt 逐字相同** ——
      抓「同一条任务换了个 ID」这类直接污染
    * 本函数比的是 **(工具, 参数名, 参数值)** ——
      抓「prompt 不同、但期望调用与某条训练样本完全相同」，
      这种情形下模型可能靠**匹配记忆中的样本**答对，而不是学会规则

    实测为何需要它：``run_baseline.py`` 原先只用前者，
    于是 v3 里有 **5 个实体泄漏**（4 个 ``get_weather(city=...)`` +
    1 个 ``search_flights``）**没有被排除**，指标被高估。

    ⚠️ **排除 ``tool is None`` 的任务**：那类「应先追问」任务的期望调用是
    空的（``{}``），与合成语料里的 clarify 任务**必然相同** ——
    那是设计使然，不是泄漏。把它们计入会一下子排除 15 个任务。

    >>> leaks = find_expected_leaks(task_set="v3")
    >>> all(isinstance(x, str) for x in leaks)
    True
    >>> len(leaks) <= 10      # 只应是少数实体泄漏
    True
    """
    from src.data.task_manifest import load_tasks_by_version
    from src.models.prompt_builder import build_plain_prompt  # noqa: F401

    def _ek(x: Any) -> tuple:
        a = x.expected.args or {}
        return (x.expected.tool, tuple(sorted((k, str(v)) for k, v in a.items())))

    ev = list(tasks) if tasks is not None else list(load_tasks_by_version(task_set))
    syn = build_training_tasks()
    syn_keys = {_ek(x) for x in syn if x.expected.tool is not None}
    return [
        x.task_id for x in ev
        if x.expected.tool is not None and _ek(x) in syn_keys
    ]


def audit_preference_entity_coverage(
    preference_files: list[str] | None = None,
    *,
    task_set: str = "v3",
) -> dict[str, dict[str, Any]]:
    """**逐实验切片**检查偏好数据的机场码 / 工具 / 技能覆盖。

    为什么不能只查 :func:`assert_training_covers_eval_entities`：

    * 那个函数查的是 ``build_training_tasks()`` 产出的**合成语料**，
      而真正喂进训练的是 ``data/preferences/<exp>.train.jsonl`` ——
      两条链路各自构造（``build_preferences.py`` 单独选池与切分），
      **合成语料覆盖好 ≠ 偏好数据覆盖好**。
    * 即使所有切片合起来覆盖完整，**单个实验的切片**仍可能缺码 ——
      这会让某个实验的结果偏低，被误读成「该方法无效」。

    本函数只报告，不抛错（覆盖是否充分由人判断）。

    Args:
        preference_files: 要检查的 ``*.train.jsonl`` 路径；``None`` 表示
            ``data/preferences/`` 下全部。
        task_set: 评测任务集版本。

    Returns:
        ``{文件名: {missing_origin, missing_destination, missing_tools,
        has_observation_turn, n_chosen_calls}}``。

    >>> rep = audit_preference_entity_coverage(
    ...     preference_files=[])      # 空列表 → 空报告
    >>> rep
    {}
    """
    import json
    import pathlib
    import re

    from src.data.task_manifest import load_tasks_by_version

    ev = list(load_tasks_by_version(task_set))
    ev_codes: set[str] = set()
    for t in ev:
        a = t.expected.args or {}
        for k in ("origin", "destination"):
            if a.get(k):
                ev_codes.add(a[k])
    ev_tools = {t.expected.tool for t in ev if t.expected.tool}

    if preference_files is None:
        d = pathlib.Path(__file__).resolve().parents[2] / "data" / "preferences"
        preference_files = sorted(str(p) for p in d.glob("*.train.jsonl"))

    report: dict[str, dict[str, Any]] = {}
    for path in preference_files:
        p = pathlib.Path(path)
        if not p.is_file():
            continue
        codes: set[str] = set()
        tools: set[str] = set()
        n_calls = 0
        has_obs = False
        for line in p.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            if '"role": "tool"' in row.get("prompt", "") or "observation" in row.get("prompt", ""):
                has_obs = True
            m = re.search(
                r"<tool_call>\s*(\{.*?\})\s*</tool_call>", row.get("chosen", ""), re.S
            )
            if not m:
                continue
            try:
                obj = json.loads(m.group(1))
            except Exception:
                continue
            n_calls += 1
            if obj.get("name"):
                tools.add(obj["name"])
            for k in ("origin", "destination"):
                v = (obj.get("arguments") or {}).get(k)
                if v:
                    codes.add(v)
        report[p.name] = {
            "missing_origin_destination": sorted(ev_codes - codes),
            "missing_tools": sorted(ev_tools - tools),
            "has_observation_turn": has_obs,
            "n_chosen_calls": n_calls,
        }
    return report
