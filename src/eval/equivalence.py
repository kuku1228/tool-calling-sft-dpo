"""参数级语义等价判定。

为什么需要单独一个模块
----------------------
朴素的字符串精确匹配会把「模型答对了但写法不同」判成错误，产生大量假阳性：

===================  ==================  ==========  ==========================
ground truth         模型输出             精确匹配    语义等价
===================  ==================  ==========  ==========================
``2025-07-15``       ``2025/07/15``       ✗ 错         ✓ 对
``PEK``              ``pek``              ✗ 错         ✓ 对
``30``               ``"30"``             ✗ 错         ✓ 对
``北京``             ``北京市``            ✗ 错         ✓ 对
``张伟``             `` 张伟 ``            ✗ 错         ✓ 对
===================  ==================  ==========  ==========================

如果不做等价判定，DPO 的评测指标会被「格式偏好」污染 ——
模型只要学会输出标准写法就能刷分，而真实的任务完成能力没变。
这本身也是本项目要报告的一个发现（见 docs/diagnosis/baseline_diagnosis.md §5）。

同时，等价判定也是**偏好数据去噪**的关键：只有确认 chosen 与 rejected
在语义上真的不同，这条偏好对才有训练价值。

判定策略（由松到紧）
--------------------
1. ``normalize``：类型归一（数值字符串→数值）、空白、大小写、全角半角
2. 别名表：城市名 ↔ 三字码、日期分隔符
3. ``equivalent``：数值用相对容差；日期用同日判定（可选 ±1 天容差）；
   字符串做归一化后比对，辅以编辑距离阈值兜底
4. ``match_arguments``：字段级比对，支持容忍字段集与「多余字段」策略

设计取舍
--------
没有用 embedding 相似度作为默认路径：一是会引入不可复现的第三方模型依赖，
二是工具参数的正确性是**精确**的（订单号错一位就是错），模糊匹配反而有害。
embedding 只在 ``equivalence_soft`` 里作为「自然语言回复是否等价」的兜底，
用于错误分类时区分「同义改写」与「真的答错」。
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher
from typing import Any

from src.env.tool_catalog import CITY_NAMES

# ============================================================================
# 归一化
# ============================================================================

#: 中文城市名的常见后缀与别称，归一化时剥离
_CITY_SUFFIXES: tuple[str, ...] = ("市", "特别行政区", "自治区", "地区")

#: 城市别名 → 规范名
_CITY_ALIASES: dict[str, str] = {
    "帝都": "北京",
    "魔都": "上海",
    "羊城": "广州",
    "鹏城": "深圳",
    "蓉城": "成都",
    "春城": "昆明",
    "冰城": "哈尔滨",
    "乌鲁木齐市": "乌鲁木齐",
    "呼和浩特市": "呼和浩特",
}

#: 日期分隔符
_DATE_SEP_RE = re.compile(r"[./年月\-]")

#: 常见日期格式，按优先级尝试
_DATE_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y.%m.%d",
    "%Y年%m月%d日",
    "%Y%m%d",
    "%m-%d",       # 无年份，归属 BASE_DATE 年份
    "%Y-%m",
)

#: 数值型字符串
_NUMBER_RE = re.compile(r"^-?\d+(\.\d+)?$")

#: 布尔/空值的文本表示
_NULLISH: frozenset[str] = frozenset({"", "none", "null", "nil", "na", "n/a", "nan", "undefined", "无", "空"})


def to_halfwidth(text: str) -> str:
    """全角转半角（模型偶尔输出全角数字，如 ``３０``）。"""
    return unicodedata.normalize("NFKC", text)


def normalize_text(value: str) -> str:
    """文本归一化：全角→半角、去首尾空白、压缩内部空白、统一小写。

    >>> normalize_text("  ＰＥＫ  ")
    'pek'
    """
    s = to_halfwidth(str(value)).strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def is_nullish(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and normalize_text(value) in _NULLISH:
        return True
    return False


def coerce_number(value: Any) -> float | None:
    """尽量转成数值；失败返回 None。

    >>> coerce_number("30.0"), coerce_number("abc")
    (30.0, None)
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = normalize_text(value).replace(",", "").replace("公斤", "").replace("kg", "").replace("元", "")
        s = s.strip()
        if _NUMBER_RE.match(s):
            return float(s)
    return None


def normalize_date(value: Any, *, default_year: int = 2025) -> date | None:
    """把各种写法的日期解析成 ``datetime.date``；解析失败返回 None。

    >>> normalize_date("2025-07-15")
    datetime.date(2025, 7, 15)
    >>> normalize_date("2025年7月15日")
    datetime.date(2025, 7, 15)
    >>> normalize_date("2025/7/15") == normalize_date("2025-07-15")
    True
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value

    s = to_halfwidth(str(value)).strip()
    if not s:
        return None
    s = s.replace(" ", "")

    for fmt in _DATE_FORMATS:
        try:
            parsed = datetime.strptime(s, fmt)
        except ValueError:
            continue
        if "%Y" not in fmt:
            parsed = parsed.replace(year=default_year)
        return parsed.date()

    # 兜底：把中英文分隔符统一后按 3 段拆分
    parts = [p for p in _DATE_SEP_RE.split(s) if p]
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        y, m, d = (int(p) for p in parts)
        if y < 100:
            y += 2000
        try:
            return date(y, m, d)
        except ValueError:
            return None
    return None


def normalize_city(value: Any) -> str | None:
    """城市名归一化：剥离「市」等后缀、展开别名。

    ⚠️ 本函数**不做**三字码 ↔ 中文城市名的互换。那个映射属于**领域语义**，
    由 ``equivalent`` 按参数名决定是否启用（见该函数里对 ``city`` 与
    ``origin`` 的分流处理）。

    理由：如果把映射放在这里，归一化就不再是「语法层面」的操作，而是
    掺入了领域判断。后果是 ``city``（要求中文名）与 ``origin``（要求三字码）
    会变得无法区分 —— 而区分这两者是 ``weather_advisory`` 这一类任务的
    全部考察点。

    >>> normalize_city("北京市"), normalize_city("上海")
    ('北京', '上海')
    >>> normalize_city("PEK")          # 三字码只做大小写归一，不映射成中文名
    'PEK'
    >>> normalize_city(None) is None
    True
    """
    if is_nullish(value):
        return None
    s = to_halfwidth(str(value)).strip()
    for suf in _CITY_SUFFIXES:
        if s.endswith(suf) and len(s) > len(suf):
            s = s[: -len(suf)]
    s = _CITY_ALIASES.get(s, s)
    # 三字码统一成大写，保证 "pek" 与 "PEK" 等价
    if len(s) == 3 and s.isascii() and s.isalpha():
        return s.upper()
    return s


def normalize_value(value: Any, param_name: str = "") -> Any:
    """按参数名推断语义，输出用于比对的规范值。

    这是 ``equivalent`` 的核心：先看参数名，再看值本身。
    参数名比值的启发式更可靠 —— 例如同样是 ``"PEK"``，
    在 ``origin`` 里是三字码，在 ``reason`` 里就是普通文本。
    """
    if is_nullish(value):
        return None

    pname = param_name.lower()

    # --- 日期类参数 ---
    if pname in ("date", "depart_date", "return_date", "start_date", "end_date") or pname.endswith("_date"):
        d = normalize_date(value)
        if d is not None:
            return d.isoformat()

    # --- 城市 / 机场类参数 ---
    # 只做语法层归一（后缀、别名、大小写）。
    # 三字码 ↔ 中文城市名的互换是**领域语义**，在 equivalent() 里
    # 按参数名决定是否启用 —— 见 _city_equivalent 的说明。
    if pname in ("origin", "destination", "city", "from_city", "to_city", "arrival_city", "departure_city"):
        return normalize_city(value)

    # --- 数值类参数 ---
    if pname.endswith(("_kg", "_price", "_cny", "_count", "_min", "_num")) or pname in (
        "new_baggage_kg",
        "max_price",
        "amount",
    ):
        num = coerce_number(value)
        if num is not None:
            # 整数化：30 与 30.0 必须等价
            return int(num) if float(num).is_integer() else round(num, 4)

    # --- 默认：数值型字符串按数值处理 ---
    num = coerce_number(value)
    if num is not None and not isinstance(value, str):
        return int(num) if float(num).is_integer() else round(num, 4)
    if isinstance(value, str):
        s = to_halfwidth(value).strip()
        if _NUMBER_RE.match(s):
            n = float(s)
            return int(n) if float(n).is_integer() else round(n, 4)
        return normalize_text(s)
    if isinstance(value, bool):
        return value
    return value


# ============================================================================
# 等价比较
# ============================================================================

#: 中文城市名 ↔ 三字码的双向映射辅助（仅在机场参数上使用）
_CITY_TO_CODE: dict[str, str] = {v: k for k, v in CITY_NAMES.items()}


def _looks_like_airport_code(value: Any) -> bool:
    """是否是三字机场码形态（3 个 ASCII 字母）。

    >>> _looks_like_airport_code("PEK"), _looks_like_airport_code("北京")
    (True, False)
    """
    if not isinstance(value, str):
        return False
    s = to_halfwidth(value).strip()
    return len(s) == 3 and s.isascii() and s.isalpha()


def _to_canonical_city_name(value: Any) -> str | None:
    """把「中文城市名或三字码」统一成规范中文城市名；无法识别返回 None。

    这个映射**只用于机场参数**（origin / destination）。
    不要在通用归一化里使用它 —— 见 ``normalize_city`` 的说明。

    >>> _to_canonical_city_name("PEK"), _to_canonical_city_name("北京市")
    ('北京', '北京')
    >>> _to_canonical_city_name("XYZ") is None        # 三字码形态但不在已知机场表里
    True
    >>> _to_canonical_city_name("乌兰察布")            # 四字中文名：不是三字码形态，原样返回
    '乌兰察布'
    >>> _to_canonical_city_name(None) is None
    True
    """
    if is_nullish(value):
        return None
    s = to_halfwidth(str(value)).strip()
    if _looks_like_airport_code(s):
        # 三字码形态：必须在已知机场表里，否则无法判定为城市
        return CITY_NAMES.get(s.upper())
    return normalize_city(s)


#: 机场类参数（要求三字码，容忍模型写中文城市名）
_AIRPORT_PARAMS: frozenset[str] = frozenset(
    {"origin", "destination", "from_city", "to_city", "arrival_city", "departure_city"}
)


def _city_equivalent(expected: Any, actual: Any, pname: str) -> tuple[bool, str] | None:
    """城市类参数的领域语义判定。不适用时返回 ``None`` 交给通用逻辑。

    这是本模块最需要小心的一处分流：

    - ``origin`` / ``destination``：schema 要求**三字码**。
      用户本来就是用中文说的，模型写成中文城市名属于可原谅的格式偏差 → 判等价。
    - ``city``（``get_weather``）：schema 要求**中文城市名**。
      **不能**把三字码判为等价 —— 否则 ``weather_advisory`` 这一整类任务
      的考察点（区分「机场参数要三字码」与「天气参数要中文名」）
      就被评分器自己抹掉了。

    早期版本对两者一视同仁（在 ``normalize_city`` 里无条件把三字码映射成中文名），
    后果是 mock 注入的「city 填三字码」错误被判成正确，
    恢复率统计出现「成功率 100% 但恢复率 0%」这种自相矛盾的结果。

    >>> _city_equivalent("PEK", "北京", "origin")
    (True, '同一城市（三字码与中文名视为等价）')
    >>> _city_equivalent("杭州", "HGH", "city")[0]
    False
    >>> _city_equivalent("任务", "任务", "reason") is None
    True
    """
    if pname in _AIRPORT_PARAMS:
        a = _to_canonical_city_name(expected)
        b = _to_canonical_city_name(actual)
        if a is None or b is None:
            return None  # 有一方不是可识别的城市，交给通用文本比对
        if a == b:
            return True, "同一城市（三字码与中文名视为等价）"
        return False, f"城市不同：{a} vs {b}"

    if pname == "city":
        a = normalize_city(expected)
        b = normalize_city(actual)
        if a == b:
            return True, "归一化后城市名相同"
        if _looks_like_airport_code(actual) and not _looks_like_airport_code(expected):
            return False, f"city 参数要求中文城市名，实际给了三字码 {actual!r}"
        return False, f"城市名不同：{expected!r} vs {actual!r}"

    return None


@dataclass
class Comparison:
    """单字段比对结果。"""

    field: str
    expected: Any
    actual: Any
    matched: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "expected": self.expected,
            "actual": self.actual,
            "matched": self.matched,
            "reason": self.reason,
        }


def equivalent(
    expected: Any,
    actual: Any,
    *,
    param_name: str = "",
    date_tolerance_days: int = 0,
    numeric_rel_tol: float = 1e-6,
    text_similarity_threshold: float = 0.0,
) -> tuple[bool, str]:
    """判断两个值是否语义等价。

    Args:
        expected: ground truth。
        actual: 模型给出的值。
        param_name: 参数名，用于选择归一化策略。
        date_tolerance_days: 日期容差。0 表示必须同一天。
            跨时区/相对日期表述的任务可以给 1。
        numeric_rel_tol: 数值相对容差。
        text_similarity_threshold: 文本相似度兜底阈值。
            0 表示关闭（默认保守）；错误分类时可开到 0.9 用于识别同义改写。

    Returns:
        ``(是否等价, 说明)``

    >>> equivalent("2025-07-15", "2025/7/15", param_name="date")[0]
    True
    >>> equivalent("PEK", "北京", param_name="origin")[0]
    True
    >>> equivalent(30, "30公斤", param_name="new_baggage_kg")[0]
    True
    >>> equivalent("BK7F2A9C", "BK7F2A9D", param_name="booking_id")[0]
    False
    """
    if is_nullish(expected) and is_nullish(actual):
        return True, "两者都为空"
    if is_nullish(expected) != is_nullish(actual):
        return False, f"一方为空（expected={expected!r}, actual={actual!r}）"

    pname = param_name.lower()

    # --- 城市 / 机场类参数：走领域语义分流 ---
    # 必须在通用归一化之前处理，因为「三字码 vs 中文名是否等价」
    # 完全取决于参数名，通用逻辑无法判断。
    city_verdict = _city_equivalent(expected, actual, pname)
    if city_verdict is not None:
        return city_verdict

    exp_norm = normalize_value(expected, param_name)
    act_norm = normalize_value(actual, param_name)

    # --- 日期 ---
    if pname in ("date", "depart_date", "return_date", "start_date", "end_date") or pname.endswith("_date"):
        d1 = normalize_date(expected)
        d2 = normalize_date(actual)
        if d1 is not None and d2 is not None:
            delta = abs((d1 - d2).days)
            if delta == 0:
                return True, "同一日期"
            if delta <= date_tolerance_days:
                return True, f"日期相差 {delta} 天，在容差 {date_tolerance_days} 天内"
            return False, f"日期不同：{d1} vs {d2}（相差 {delta} 天）"
        if d1 is None and d2 is None:
            return (exp_norm == act_norm), "两者都无法解析为日期，退化为文本比对"

    # --- 数值 ---
    if isinstance(exp_norm, (int, float)) and isinstance(act_norm, (int, float)):
        e, a = float(exp_norm), float(act_norm)
        if e == a:
            return True, "数值相同"
        denom = max(abs(e), 1e-9)
        rel = abs(e - a) / denom
        if rel <= numeric_rel_tol:
            return True, f"数值相对误差 {rel:.2e} 在容差内"
        return False, f"数值不同：{expected!r} vs {actual!r}"

    # --- 类型不一致（一个是数值一个是文本）---
    if isinstance(exp_norm, (int, float)) != isinstance(act_norm, (int, float)):
        return False, f"类型不同：expected 归一化为 {exp_norm!r}，actual 归一化为 {act_norm!r}"

    # --- 文本 ---
    if exp_norm == act_norm:
        return True, "归一化后文本相同"

    if text_similarity_threshold > 0 and isinstance(exp_norm, str) and isinstance(act_norm, str):
        ratio = SequenceMatcher(None, exp_norm, act_norm).ratio()
        if ratio >= text_similarity_threshold:
            return True, f"文本相似度 {ratio:.3f} >= 阈值 {text_similarity_threshold}"

    return False, f"文本不同：{expected!r} vs {actual!r}"


def text_similarity(a: Any, b: Any) -> float:
    """归一化文本相似度，用于错误分类（同义改写 vs 真的答错）。"""
    sa = normalize_text(str(a)) if not is_nullish(a) else ""
    sb = normalize_text(str(b)) if not is_nullish(b) else ""
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return SequenceMatcher(None, sa, sb).ratio()


# ============================================================================
# 参数集合比对
# ============================================================================

#: 这些字段在宽松口径下可以缺失 —— 它们要么依赖 observation，
#: 要么用户没给（模型没义务猜对），严格比对会低估真实能力。
LENIENT_OPTIONAL_FIELDS: frozenset[str] = frozenset(
    {"flight_id", "booking_id", "reason", "max_price", "cabin", "new_baggage_kg"}
)


@dataclass
class ArgMatchResult:
    """一次工具调用的参数比对结果。"""

    tool_matched: bool
    args_matched: bool
    #: 严格口径（全部期望字段必须命中）
    strict_matched: bool
    #: 宽松口径（只比对 ``graded_fields``）
    lenient_matched: bool
    comparisons: list[Comparison] = field(default_factory=list)
    #: 期望有但实际没给的字段
    missing_fields: list[str] = field(default_factory=list)
    #: 实际给了、但**该工具 schema 里也没有**的字段（真正的编造参数）
    extra_fields: list[str] = field(default_factory=list)
    #: 实际给了、是该工具合法参数、但不在评分范围里的字段（不算错）
    ungraded_valid_fields: list[str] = field(default_factory=list)

    @property
    def graded_fields(self) -> list[str]:
        return [c.field for c in self.comparisons]

    def param_accuracy(self) -> float:
        """字段级准确率，用于「参数正确率」指标（比二元判定更细）。"""
        if not self.comparisons:
            return 1.0 if not self.missing_fields else 0.0
        hit = sum(c.matched for c in self.comparisons)
        total = len(self.comparisons) + len(self.missing_fields)
        return hit / total if total else 1.0

    def failures(self) -> list[Comparison]:
        return [c for c in self.comparisons if not c.matched]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_matched": self.tool_matched,
            "args_matched": self.args_matched,
            "strict_matched": self.strict_matched,
            "lenient_matched": self.lenient_matched,
            "param_accuracy": round(self.param_accuracy(), 4),
            "missing_fields": self.missing_fields,
            "extra_fields": self.extra_fields,
                    "ungraded_valid_fields": self.ungraded_valid_fields,
            "comparisons": [c.to_dict() for c in self.comparisons],
        }


def match_arguments(
    expected: dict[str, Any],
    actual: dict[str, Any],
    *,
    graded_subset: dict[str, Any] | None = None,
    allow_extra_fields: bool = True,
    date_tolerance_days: int = 0,
    schema_fields: set[str] | None = None,
) -> ArgMatchResult:
    """比对两组工具参数。

    Args:
        expected: 完整期望参数（严格口径用）。
        actual: 模型给出的参数。
        graded_subset: 主指标（宽松口径）实际比对的子集。
            None 表示与 ``expected`` 相同。
        allow_extra_fields: 是否容忍模型多给字段。
            默认 True —— 多给一个合理的可选参数不应算错。
        date_tolerance_days: 日期容差。

    >>> r = match_arguments({"booking_id": "BK7F2A9C"}, {"booking_id": "BK7F2A9C"})
    >>> r.strict_matched, r.lenient_matched
    (True, True)
    """
    actual = actual or {}
    graded = graded_subset if graded_subset is not None else expected

    comparisons: list[Comparison] = []
    missing: list[str] = []

    # 用完整期望的字段顺序做严格比对
    for fname, exp_val in expected.items():
        if fname not in actual:
            missing.append(fname)
            comparisons.append(
                Comparison(fname, exp_val, None, False, "期望字段缺失")
            )
            continue
        ok, reason = equivalent(
            exp_val, actual[fname], param_name=fname, date_tolerance_days=date_tolerance_days
        )
        comparisons.append(Comparison(fname, exp_val, actual[fname], ok, reason))

    # ⚠️ 「多给字段」必须区分两种情况，早期版本把它们混为一谈：
    #
    #   1. **有效但未评分的参数** —— 该工具 schema 里确实存在，只是任务的
    #      ``expected.args`` 没列它。**不该算错。**
    #   2. **编造的无效参数** —— schema 里根本没有。**该算错。**
    #
    # 混为一谈的后果实测（sft_v3，400 采样）：
    #
    #   32x  [cancellation] cancel_booking  多给=('reason',)   ← reason 是 schema 必填！
    #    4x  [recovery]     book_flight     多给=('flight_id',) ← flight_id 是 schema 必填！
    #
    # 即**模型给对了必填参数，反而被判成「多给了 schema 未定义的参数」**。
    # 这是评分器最严重的一处口径错误：它把 `extra_param` 抬成了头号失败原因
    # （42 条，占全部失败的 58.8%），而其中 36 条是误判。
    #
    # 根因：``expected.args`` 常常只是「答案的一部分」—— 有的任务有意不把
    # 需要模型自己获取的参数（如 flight_id）或自由文本（如 reason）列进去。
    # 但它仍是该工具的合法参数。
    from src.env.tool_catalog import TOOL_REGISTRY as _REG

    valid_fields = set(schema_fields) if schema_fields else set(expected)
    # 兜底：schema_fields 没给时，用全库同名工具的参数并集（避免误判）
    if not schema_fields:
        for _spec in _REG.values():
            valid_fields |= set(_spec.params)

    extra = [f for f in actual if f not in expected and f not in valid_fields]
    #: 模型给了、是该工具合法参数、但不在评分范围里的字段。
    #: 单列出来便于报告区分「模型多嘴」与「模型编造」。
    ungraded_valid = sorted(
        f for f in actual if f not in expected and f in valid_fields and f != ""
    )

    strict_ok = not missing and all(c.matched for c in comparisons) and (allow_extra_fields or not extra)

    # 宽松口径：只看 graded 里列出的字段
    lenient_comparisons = [c for c in comparisons if c.field in graded]
    lenient_missing = [f for f in graded if f not in actual]
    lenient_ok = not lenient_missing and all(c.matched for c in lenient_comparisons)

    return ArgMatchResult(
        tool_matched=True,  # 工具名是否匹配由调用方判断
        args_matched=strict_ok,
        strict_matched=strict_ok,
        lenient_matched=lenient_ok,
        comparisons=comparisons,
        missing_fields=missing + [f for f in lenient_missing if f not in missing],
        extra_fields=extra,
        ungraded_valid_fields=ungraded_valid,
    )


# ============================================================================
# 自然语言回复的软等价（用于错误分类，不用于打分）
# ============================================================================

#: 澄清类回复的特征词。模型在没有足够信息时应当追问，
#: 这类回复不该被判成「格式错误」。
CLARIFICATION_MARKERS: tuple[str, ...] = (
    "请问",
    "请提供",
    "能否告诉",
    "麻烦您",
    "您能提供",
    "需要您提供",
    "还需要",
    "缺",
    "没有提供",
    "请确认",
    "想确认",
    "哪个",
    "哪一种",
    "哪一班",
)

#: 「会去调用工具」的承诺式回复（但没有真的输出 tool_call）——
#: 这是小模型最高频的失败模式：嘴上说要查，实际没调。
PROMISE_MARKERS: tuple[str, ...] = (
    "我来帮您",
    "我帮您查",
    "正在查询",
    "请稍等",
    "马上为您",
    "我为您查询",
    "为您搜索",
    "我这就",
    "正在为您",
)


def looks_like_clarification(text: str) -> bool:
    """回复是否在向用户追问信息（而非答错）。"""
    t = normalize_text(text)
    return any(m in t for m in CLARIFICATION_MARKERS)


def looks_like_promise(text: str) -> bool:
    """回复是否只是「承诺要调用工具」却没真调。"""
    t = normalize_text(text)
    return any(m in t for m in PROMISE_MARKERS)


def parse_json_loose(text: str) -> tuple[Any, str]:
    """尽量从文本里抠出 JSON 对象。

    处理模型常见的几种坏格式：

    - 前后有解释文字：``这是工具调用 {"name": ...}``
    - Markdown 代码围栏：`` ```json\\n{...}\\n``` ``
    - 单引号键：``{'name': 'get_weather'}``
    - 尾随逗号：``{"a": 1,}``

    Returns:
        ``(解析结果或 None, 说明)``

    >>> parse_json_loose('blah {"a": 1} tail')[0]
    {'a': 1}
    """
    s = to_halfwidth(str(text)).strip()
    if not s:
        return None, "空字符串"

    # 去掉 markdown 围栏
    fence = re.search(r"```(?:json)?\s*(.+?)\s*```", s, flags=re.S)
    if fence:
        s = fence.group(1).strip()

    # 直接尝试
    try:
        return json.loads(s), "直接解析成功"
    except json.JSONDecodeError:
        pass

    # 抠出最外层 {...}
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        cand = s[start : end + 1]
        for attempt, label in (
            (cand, "截取花括号后解析成功"),
            (re.sub(r",\s*([}\]])", r"\1", cand), "去掉尾随逗号后解析成功"),
            (re.sub(r"'([^']*?)'(\s*:)", r'"\1"\2', cand), "单引号键转双引号后解析成功"),
        ):
            try:
                return json.loads(attempt), label
            except json.JSONDecodeError:
                continue

    return None, "无法解析出合法 JSON"
