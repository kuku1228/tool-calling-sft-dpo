"""错误分类体系。

为什么要有独立的分类体系
------------------------
「准确率 40%」这句话对改进没有指导意义。「40% 里有多少是 JSON 语法错误、
多少是编造工具名、多少是格式对但参数错」才是能直接决定方案设计的信息。

本项目的分类体系直接对标 ``docs/ablation/ablation_plan.md`` 的实验矩阵：
每一类错误都对应一个偏好数据构造策略。

分类体系
--------
::

    SUCCESS                     工具名与（宽松口径）参数全对
    ├─ 失败类型（互斥，优先级从高到低）
    ├─ FORMAT_ERROR            没输出可解析的 tool_call（含语法坏、承诺式回复）
    ├─ HALLUCINATION           工具名不在白名单
    ├─ WRONG_TOOL              工具名合法但不是期望的那个
    ├─ PARAM_ERROR             工具名对，但参数不对（缺失/占位符/值错）
    ├─ REFUSED                 明确拒绝或放弃任务
    └─ UNKNOWN                 其他

优先级说明
----------
「格式坏」优先于「工具名错」：如果 JSON 根本解析不出来，
就没法讨论工具名对不对。这个顺序决定了诊断报告里各类错误占比的可加性
（所有类别之和 = 失败总数，不会重复计数）。

为什么把「承诺式回复」单独拎出来
--------------------------------
`我来帮您查询一下航班` 这类回复在小模型上是最高频的失败模式，
它既不是拒绝也不是语法错，而是「意图正确但没落到工具调用」。
错误恢复偏好对里有一整类样本专门针对它（``promise_to_action``）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

# ============================================================================
# 类别常量
# ============================================================================

SUCCESS = "SUCCESS"
FORMAT_ERROR = "FORMAT_ERROR"
HALLUCINATION = "HALLUCINATION"
WRONG_TOOL = "WRONG_TOOL"
PARAM_ERROR = "PARAM_ERROR"
REFUSED = "REFUSED"
UNKNOWN = "UNKNOWN"

ALL_CATEGORIES: tuple[str, ...] = (
    SUCCESS,
    FORMAT_ERROR,
    HALLUCINATION,
    WRONG_TOOL,
    PARAM_ERROR,
    REFUSED,
    UNKNOWN,
)

#: 失败类型（SUCCESS 除外）
FAILURE_CATEGORIES: tuple[str, ...] = tuple(c for c in ALL_CATEGORIES if c != SUCCESS)

#: 中文名，用于报告与图表
CATEGORY_LABELS_ZH: dict[str, str] = {
    SUCCESS: "成功",
    FORMAT_ERROR: "格式错误",
    HALLUCINATION: "工具幻觉",
    WRONG_TOOL: "工具选错",
    PARAM_ERROR: "参数错误",
    REFUSED: "拒绝作答",
    UNKNOWN: "未知错误",
}

#: 每类错误对应的改进手段（写进诊断报告，让「分类」直接连到「方案」）
CATEGORY_REMEDIES_ZH: dict[str, str] = {
    FORMAT_ERROR: "格式约束偏好对：chosen 是合法 tool_call，rejected 是纯文本/坏 JSON",
    HALLUCINATION: "工具选择偏好对：chosen 用白名单工具，rejected 用貌似合理的假工具",
    WRONG_TOOL: "工具选择偏好对：同一请求下正确工具 vs 相邻语义工具",
    PARAM_ERROR: "参数约束偏好对：chosen 是真实实体，rejected 是占位符/错格式/编造值",
    REFUSED: "错误恢复偏好对 + 澄清类样本：示范「信息不足时追问」而非放弃",
    UNKNOWN: "人工抽检，补充新类别",
}

#: 详细子类型。用于诊断报告下钻，也用于偏好数据的样本筛选。
SUBTYPE_LABELS_ZH: dict[str, str] = {
    "no_tool_call": "完全没有 tool_call 标签",
    "action_hallucination": "声称已完成操作（编造执行结果）",
    "empty": "输出为空",
    "promise_only": "只承诺要查，没真的调用",
    "json_broken": "JSON 语法非法",
    "json_not_object": "解析出的 JSON 不是对象",
    "missing_name": "JSON 里没有 name 字段",
    "missing_arguments": "JSON 里没有 arguments 字段",
    "arguments_not_object": "arguments 不是对象",
    "truncated": "输出被 max_new_tokens 截断",
    "unknown_tool": "工具名不在白名单",
    "plausible_fake_tool": "工具名是「貌似合理但不存在」的名字",
    "adjacent_tool": "工具名属于同域相邻工具",
    "should_clarify": "信息不足时应先追问，却直接调用了工具",
    "missing_required": "缺少必填参数",
    "placeholder_value": "参数填了占位符",
    "wrong_value": "参数值不对",
    "no_arg_check": "未做参数校验",
    "wrong_format": "参数格式不合 schema",
    "out_of_range": "参数超出取值范围",
    "extra_param": "多给了 schema 未定义的参数",
    "refusal": "模型明确拒绝",
    "cannot_help": "模型表示无法完成",
    "clarify_instead": "该调用但只做了澄清",
    "partial": "多个 tool_call 中只有部分可解析",
    "unclosed_tag": "tool_call 标签未闭合（疑似被截断）",
    "bare_json": "没有标签，直接输出裸 JSON",
    "whole_text_json": "整段输出就是一个 JSON 对象",
}


# ============================================================================
# 数据结构
# ============================================================================


@dataclass
class ErrorDetail:
    """一条错误的详细证据。"""

    category: str
    subtype: str = ""
    #: 人类可读的一句话解释
    message: str = ""
    #: 涉及的字段（参数错误时用）
    fields: list[str] = field(default_factory=list)
    #: 原始证据片段（截断后入库，避免 json 膨胀）
    evidence: str = ""

    @property
    def subtype_zh(self) -> str:
        return SUBTYPE_LABELS_ZH.get(self.subtype, self.subtype)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "category_zh": CATEGORY_LABELS_ZH.get(self.category, self.category),
            "subtype": self.subtype,
            "subtype_zh": self.subtype_zh,
            "message": self.message,
            "fields": self.fields,
            "evidence": self.evidence,
        }


@dataclass
class Classification:
    """对一次采样输出的完整分类结论。"""

    category: str
    detail: ErrorDetail
    #: 是否存在「先错后改」的恢复行为（纠错轮任务用）
    recovered_from: str | None = None
    #: 是否属于参数格式对但值错（reward hacking 的典型信号）
    format_ok_value_wrong: bool = False

    @property
    def is_success(self) -> bool:
        return self.category == SUCCESS

    @property
    def category_zh(self) -> str:
        return CATEGORY_LABELS_ZH.get(self.category, self.category)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "category_zh": self.category_zh,
            "detail": self.detail.to_dict(),
            "recovered_from": self.recovered_from,
            "format_ok_value_wrong": self.format_ok_value_wrong,
        }


# ============================================================================
# 拒绝检测
# ============================================================================

_REFUSAL_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.I)
    for p in (
        r"抱歉[，,]?\s*(我)?(无法|不能|没法)",
        r"我(无法|不能|没法)(帮|为)(您|你)",
        r"作为(一个)?(AI|人工智能|语言模型)",
        r"超出(了)?我的(能力|权限|职能)",
        r"建议您(联系|拨打|咨询)(人工|客服|官方)",
        r"i('m| am) (sorry|unable)",
        r"i cannot (help|assist)",
        r"i can't (help|assist)",
    )
)


def detect_refusal(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    return any(p.search(t) for p in _REFUSAL_PATTERNS)


#: 「已经办成了」的强标记。单独出现即可判定为编造执行结果。
#:
#: 这类词本身就是完成时态（「预订了」「取消成功」），不可能出现在合理追问里。
_COMPLETION_STRONG: tuple[str, ...] = (
    "已为您预订", "已经为您预订", "为您预订了", "已为您订", "订好了", "预订成功",
    "已成功预订", "已经订好", "已为您取消", "已经取消", "取消成功",
    "已为您修改", "已经修改", "修改成功", "已为您更改为", "已更改为",
    "已为您办理", "办理成功", "已完成", "请查收",
    "祝您旅途愉快", "出票成功", "付款成功", "支付成功", "已提交",
)

#: 「报了具体标识符」的模式。必须**带上一个值**才算，
#: 否则「请问您的订单号是多少？」这种合理追问会被误判。
#:
#: 实际遇到过的假阳性：早期版本把 ``订单号`` 也当成标记，于是
#: 「请问您的订单号是多少？」被误判为「编造执行结果」，
#: 让诊断报告的错误分布虚高。修正方式是要求后面跟具体值。
_ID_REPORT_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        r"订单号\s*(?:为|是|：|:)\s*[A-Za-z0-9]{4,}",
        r"航班号\s*(?:为|是|：|:)\s*[A-Za-z0-9]{3,}",
        r"票价\s*(?:为|是|：|:)?\s*\d+",
        r"价格\s*(?:为|是|：|:)?\s*\d+",
        r"金额\s*(?:为|是|：|:)?\s*\d+",
        r"总价\s*(?:为|是|：|:)?\s*\d+",
        r"费用\s*(?:为|是|：|:)?\s*\d+",
    )
)


def _claims_completed_action(text: str) -> bool:
    """模型是否在**没有工具调用**的情况下声称已完成某个操作。

    判定规则（任一命中）：

    1. 出现完成时态的强标记（``预订了`` / ``取消成功`` / ``请查收`` …）
    2. 报出了具体的编造标识符（``订单号为 BK123456`` / ``票价 1280`` …）

    第 2 条**必须要求后面跟实际的值**。早期版本只匹配关键词，
    于是「请问您的订单号是多少？」这类合理追问被误判成编造结果。

    >>> _claims_completed_action("好的，我为您预订了 MU1000 航班，订单号为 BK7F2A9C。")
    True
    >>> _claims_completed_action("好的，请问您是经济舱还是商务舱？")
    False
    >>> _claims_completed_action("请问您的订单号是多少？")
    False
    >>> _claims_completed_action("")
    False
    """
    t = (text or "").strip()
    if not t:
        return False
    if any(m in t for m in _COMPLETION_STRONG):
        return True
    return any(p.search(t) for p in _ID_REPORT_PATTERNS)


# ============================================================================
# 分类
# ============================================================================


def classify(
    *,
    text: str,
    calls: list[Any],
    parse_error: str = "",
    expected_tool: str | None,
    arg_match: Any | None,
    valid_tool_names: Iterable[str],
    plausible_fake_tools: Iterable[str] = (),
    finish_reason: str = "",
    max_new_tokens_hit: bool = False,
    had_tag: bool = False,
) -> Classification:
    """把一次采样结果分类。

    Args:
        text: 模型原始输出。
        calls: 解析出的工具调用列表（``ToolCall`` 对象，含 ``name`` / ``arguments``）。
        parse_error: 解析失败的原因（若有）。
        expected_tool: 期望工具名；None 表示该轮不该调用工具。
        arg_match: ``ArgMatchResult``（工具名匹配时才有意义）。
        valid_tool_names: 工具白名单。
        plausible_fake_tools: 「貌似合理但不存在」的工具名池，用于细化幻觉子类。
        finish_reason: 生成结束原因。
        max_new_tokens_hit: 是否因达到 token 上限而截断。
        had_tag: 文本里**是否出现过** ``<tool_call>`` 标签。
            用于区分「有标签但 JSON 坏了」与「压根没打算调工具」——
            这两者的偏好数据设计方向完全相反，混为一谈会让错误分类失真。
            默认 False 是为了兼容旧调用；``grader.grade`` 会显式传入。

    Returns:
        ``Classification``
    """
    valid = {str(n) for n in valid_tool_names}
    fakes = {str(n) for n in plausible_fake_tools}
    raw = (text or "").strip()
    evidence = raw[:400]

    # ---------------------------------------------------------------
    # 1. 没有任何可用的工具调用 → FORMAT_ERROR 或 SUCCESS（若本就该澄清）
    # ---------------------------------------------------------------
    if not calls:
        if expected_tool is None:
            # 该轮期望「不调用工具，先澄清」。此时纯文本回复是正确的，
            # 但如果模型拒绝作答，仍要单独标注。
            #
            # ⚠️⚠️ **空输出必须判失败**（这一条是实测查出的严重漏洞）
            #
            # 原来这里无条件返回 SUCCESS，于是**空字符串也算「正确追问」**。
            # 实测后果（v3 基线）：
            #
            #     未训练的基座模型在 15 个「应先追问」任务上通过 **98.2%**
            #     （55/56），其中 26/56 的输出是「其他文本」、甚至有空输出。
            #
            # 那个 98.2% 是**免费的** —— 任何「几乎不调用工具」的模型都能拿到，
            # 因为这类任务的判据只是「没有 tool_call」。
            # 它让 baseline 的总体准确率虚高约 **14 个百分点**
            # （15/106 = 14.2% 的任务 × 98.2%）。
            #
            # 修法：空输出单独判失败。**注意不能要求「必须有问号」** ——
            # 合理的追问也可能不带问号（「请提供订单号」），
            # 所以判据只排除「什么都没说」。
            if not raw:
                return Classification(
                    category=FORMAT_ERROR,
                    detail=ErrorDetail(
                        category=FORMAT_ERROR,
                        subtype="empty",
                        message="该轮应追问信息，但输出为空",
                        evidence=evidence,
                    ),
                )
            if detect_refusal(raw):
                return Classification(
                    category=REFUSED,
                    detail=ErrorDetail(
                        category=REFUSED,
                        subtype="refusal",
                        message="该轮应追问信息，但模型直接拒绝作答",
                        evidence=evidence,
                    ),
                )
            return Classification(
                category=SUCCESS,
                detail=ErrorDetail(
                    category=SUCCESS,
                    subtype="clarify_instead",
                    message="正确地向用户追问信息，未贸然调用工具",
                    evidence=evidence,
                ),
            )

        # 期望调用工具但没调 → 细分。
        #
        # ⚠️ 判定顺序很重要，早期版本在这里出过一个真实的误标 bug：
        # 当时只要 ``parse_error`` 非空就归成 ``json_broken``，**没有先检查
        # 文本里到底有没有出现过 ``<tool_call>`` 标签**。于是「模型压根没打算
        # 调工具、只回了一句话」这种最常见的失败被标成了「JSON 语法非法」。
        #
        # 实测后果：一份 400 样本的真实基线报告显示 ``json_broken: 100``，
        # 逐条核查后发现**一条 JSON 语法错误都没有**，100 条全是纯文本回复。
        # 这会让「错误分类」这个项目核心产出完全失真，并误导偏好数据的设计方向
        # （去造「坏 JSON vs 好 JSON」的偏好对，而真正该造的是
        #  「纯文本/编造动作 vs 真实工具调用」）。
        #
        # 所以现在的顺序是：先判「有没有标签」，再判 JSON 是否合法。
        if not raw:
            subtype, msg = "empty", "模型输出为空"
        elif max_new_tokens_hit or finish_reason == "length":
            subtype, msg = "truncated", "输出因达到 max_new_tokens 被截断，tool_call 不完整"
        elif had_tag:
            # 有标签但解析不出调用 → 这才是真正的 JSON/结构问题
            subtype = "json_broken"
            msg = f"检测到 tool_call 标签但 {parse_error or '内容无法解析'}"
        else:
            # 没有标签 → 模型选择了「说话」而不是「调工具」
            from src.eval.equivalence import looks_like_clarification, looks_like_promise

            if _claims_completed_action(raw):
                # 最危险的一类：声称已经完成操作（订好了/已取消/已修改），
                # 但一次工具都没调用。用户会以为事情办成了。
                subtype = "action_hallucination"
                msg = "声称已完成操作（但没有任何工具调用），属于编造执行结果"
            elif looks_like_promise(raw):
                subtype, msg = "promise_only", "只承诺要查询，没有真的输出 tool_call"
            elif looks_like_clarification(raw):
                subtype, msg = "clarify_instead", "该轮应调用工具，模型却只做了澄清追问"
            else:
                subtype, msg = "no_tool_call", "输出里没有 <tool_call> 标签"

        if detect_refusal(raw):
            return Classification(
                category=REFUSED,
                detail=ErrorDetail(
                    category=REFUSED,
                    subtype="cannot_help",
                    message="模型表示无法完成该任务",
                    evidence=evidence,
                ),
            )
        return Classification(
            category=FORMAT_ERROR,
            detail=ErrorDetail(category=FORMAT_ERROR, subtype=subtype, message=msg, evidence=evidence),
        )

    # ---------------------------------------------------------------
    # 2. 取第一个调用作为主判据
    #    说明：多调用场景（multi_tool 的 later turn）里第一个调用就是
    #    目标调用；把「多调了一次」记在 subtypes 里而不是判错。
    # ---------------------------------------------------------------
    call = calls[0]
    tool = getattr(call, "name", None)
    args = getattr(call, "arguments", None) or {}

    if not isinstance(args, dict):
        return Classification(
            category=FORMAT_ERROR,
            detail=ErrorDetail(
                category=FORMAT_ERROR,
                subtype="arguments_not_object",
                message=f"arguments 必须是对象，实际是 {type(args).__name__}",
                evidence=evidence,
            ),
        )

    # ---------------------------------------------------------------
    # 3. 该轮期望不调用工具，但模型调了
    # ---------------------------------------------------------------
    if expected_tool is None:
        return Classification(
            category=WRONG_TOOL,
            detail=ErrorDetail(
                category=WRONG_TOOL,
                subtype="should_clarify",
                message=f"信息不足时应先追问，模型却直接调用了 {tool}",
                fields=["tool"],
                evidence=evidence,
            ),
        )

    # ---------------------------------------------------------------
    # 4. 工具名比对
    # ---------------------------------------------------------------
    if tool not in valid:
        if tool in fakes:
            subtype = "plausible_fake_tool"
            msg = f"调用了貌似合理但不存在于工具列表的工具 {tool!r}"
        else:
            subtype = "unknown_tool"
            msg = f"调用了不存在的工具 {tool!r}"
        return Classification(
            category=HALLUCINATION,
            detail=ErrorDetail(
                category=HALLUCINATION,
                subtype=subtype,
                message=msg,
                fields=["tool"],
                evidence=evidence,
            ),
        )

    if tool != expected_tool:
        return Classification(
            category=WRONG_TOOL,
            detail=ErrorDetail(
                category=WRONG_TOOL,
                subtype="adjacent_tool",
                message=f"工具选错：期望 {expected_tool!r}，实际 {tool!r}",
                fields=["tool"],
                evidence=evidence,
            ),
        )

    # ---------------------------------------------------------------
    # 5. 工具名正确 → 参数比对
    # ---------------------------------------------------------------
    if arg_match is None:
        return Classification(
            category=UNKNOWN,
            detail=ErrorDetail(
                category=UNKNOWN, subtype="no_arg_check", message="未提供参数比对结果", evidence=evidence
            ),
        )

    if arg_match.lenient_matched:
        # 严格口径也没过 → 记为成功但标注「宽松命中」，便于报告里区分
        extra = getattr(arg_match, "extra_fields", [])
        subtype = "extra_param" if extra else ""
        msg = "工具与关键参数正确"
        if extra:
            msg += f"（多给了未定义参数 {extra}）"
        return Classification(
            category=SUCCESS,
            detail=ErrorDetail(category=SUCCESS, subtype=subtype, message=msg, evidence=evidence),
        )

    # 参数错 → 细分失败原因
    failures = arg_match.failures() if hasattr(arg_match, "failures") else []
    missing = getattr(arg_match, "missing_fields", [])

    placeholders = [
        c.field
        for c in failures
        if getattr(c, "actual", None) is not None
        and _is_placeholder_like(getattr(c, "actual", None))
    ]
    if placeholders:
        subtype, msg = "placeholder_value", f"参数填了占位符：{placeholders}"
    elif missing:
        subtype, msg = "missing_required", f"缺少必填参数：{missing}"
    elif failures:
        fields = [c.field for c in failures]
        subtype = "wrong_value"
        msg = "参数值不对：" + "; ".join(
            f"{c.field} 期望 {c.expected!r} 实际 {c.actual!r}" for c in failures
        )
        # 判断是不是「格式对但值错」—— reward hacking 的信号
        format_ok_value_wrong = all(
            _same_shape(c.expected, c.actual) for c in failures
        )
        return Classification(
            category=PARAM_ERROR,
            detail=ErrorDetail(
                category=PARAM_ERROR,
                subtype=subtype,
                message=msg,
                fields=fields,
                evidence=evidence,
            ),
            format_ok_value_wrong=format_ok_value_wrong,
        )
    else:
        subtype, msg = "wrong_value", "参数不匹配，但未定位到具体字段"

    return Classification(
        category=PARAM_ERROR,
        detail=ErrorDetail(
            category=PARAM_ERROR,
            subtype=subtype,
            message=msg,
            fields=[c.field for c in failures] or missing,
            evidence=evidence,
        ),
    )


def _is_placeholder_like(value: Any) -> bool:
    from src.env.tool_catalog import looks_like_placeholder

    try:
        return looks_like_placeholder(value)
    except Exception:  # pragma: no cover - 防御性
        return False


def _same_shape(expected: Any, actual: Any) -> bool:
    """两个值的「形状」是否一致 —— 都像日期、都是数字等等。

    形状一致但值不同 = 格式学会了但实体填错，这是 DPO 最需要监控的退化模式。
    """
    from src.eval.equivalence import coerce_number, normalize_date

    if normalize_date(expected) is not None and normalize_date(actual) is not None:
        return True
    e_num = coerce_number(expected)
    a_num = coerce_number(actual)
    if e_num is not None and a_num is not None:
        return True
    if isinstance(expected, str) and isinstance(actual, str):
        # 长度接近且字符集相同（都像订单号 / 都像三字码）
        if len(expected) == len(actual):
            return True
    return False


# ============================================================================
# 分布统计
# ============================================================================


@dataclass
class ErrorDistribution:
    """一组分类结果的分布统计。"""

    total: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    subtype_counts: dict[str, int] = field(default_factory=dict)
    #: 记录若干条各类错误的具体样本（task_id + 证据），供报告引用
    exemplars: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    max_exemplars: int = 3

    def add(self, classification: Classification, *, task_id: str = "", meta: dict[str, Any] | None = None) -> None:
        self.total += 1
        cat = classification.category
        self.counts[cat] = self.counts.get(cat, 0) + 1
        st = classification.detail.subtype or "-"
        self.subtype_counts[st] = self.subtype_counts.get(st, 0) + 1
        if cat != SUCCESS and st != "clarify_instead":
            bucket = self.exemplars.setdefault(cat, [])
            if len(bucket) < self.max_exemplars:
                rec = {
                    "task_id": task_id,
                    "subtype": st,
                    "message": classification.detail.message,
                    "evidence": classification.detail.evidence[:300],
                }
                if meta:
                    rec.update(meta)
                bucket.append(rec)

    @property
    def success_count(self) -> int:
        return self.counts.get(SUCCESS, 0)

    @property
    def success_rate(self) -> float:
        return self.success_count / self.total if self.total else 0.0

    def rate(self, category: str) -> float:
        """该类别占**总样本**的比例。各类别比例之和 = 1。"""
        return self.counts.get(category, 0) / self.total if self.total else 0.0

    def failure_share(self, category: str) -> float:
        """该类别占**失败样本**的比例。"""
        fails = self.total - self.success_count
        return self.counts.get(category, 0) / fails if fails else 0.0

    def to_dict(self) -> dict[str, Any]:
        # 子类型统计里剔除成功样本 —— 否则「成功」会以 `-` 的形式
        # 混进错误子类型排行榜，把真正要看的失败模式挤下去。
        failure_subtypes = {
            k: v
            for k, v in self.subtype_counts.items()
            if k not in ("-", "clarify_instead", "extra_param")
        }
        return {
            "total": self.total,
            "success_count": self.success_count,
            "success_rate": round(self.success_rate, 6),
            "counts": dict(sorted(self.counts.items())),
            "counts_zh": {CATEGORY_LABELS_ZH.get(k, k): v for k, v in sorted(self.counts.items())},
            "rates": {k: round(self.rate(k), 6) for k in ALL_CATEGORIES},
            "failure_share": {
                k: round(self.failure_share(k), 6) for k in FAILURE_CATEGORIES
            },
            "subtype_counts": dict(sorted(self.subtype_counts.items(), key=lambda kv: -kv[1])),
            "failure_subtype_counts": dict(
                sorted(failure_subtypes.items(), key=lambda kv: -kv[1])
            ),
            "subtype_counts_zh": {
                SUBTYPE_LABELS_ZH.get(k, k): v
                for k, v in sorted(failure_subtypes.items(), key=lambda kv: -kv[1])
            },
            "exemplars": self.exemplars,
        }

    def summary_lines(self, top_subtypes: int = 8) -> list[str]:
        lines = [
            f"样本总数 {self.total}，成功 {self.success_count}（{self.success_rate:.1%}）",
            "-" * 66,
            f"{'错误类别':<14}{'数量':>6}{'占总样本':>10}{'占失败':>10}  说明",
            "-" * 66,
        ]
        for cat in FAILURE_CATEGORIES:
            n = self.counts.get(cat, 0)
            if n == 0:
                continue
            lines.append(
                f"{CATEGORY_LABELS_ZH[cat]:<14}{n:>6}{self.rate(cat):>10.1%}"
                f"{self.failure_share(cat):>10.1%}  {CATEGORY_LABELS_ZH[cat]}"
            )
        lines.append("-" * 66)
        if self.subtype_counts:
            lines.append("详细子类型 Top:")
            for st, n in sorted(self.subtype_counts.items(), key=lambda kv: -kv[1])[:top_subtypes]:
                lines.append(f"    {SUBTYPE_LABELS_ZH.get(st, st):<24}{n:>6}")
        return lines
