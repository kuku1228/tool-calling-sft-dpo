"""工具调用输出的解析器。

解析是整条评测链路的入口，也是最容易出错的一环 ——
如果解析器把「模型其实答对了」的输出判成格式错误，
后面的错误分类和偏好数据构造会被系统性污染。

因此本模块只做一件事：**尽最大努力抽取意图，并如实报告失败原因**。
不做任何「猜测补全」（例如不要把 ``name: get_weather`` 这种
缺引号的片段猜成一个合法调用）—— 猜错比不猜更危险。

支持的格式
----------
1. Hermes / Qwen 标准格式::

       <tool_call>{"name": "search_flights", "arguments": {...}}</tool_call>

2. 顶层平铺（有些微调把 arguments 展平到顶层）::

       <tool_call>{"name": "search_flights", "origin": "PEK", ...}</tool_call>

3. 无标签的裸 JSON（只含 name/arguments 时接受）

4. 多个 tool_call 标签（取全部，主判据用第一个）

明确**不**接受的形式（会被判成 FORMAT_ERROR，并给出子类型）::

   我来帮您查询           ← 承诺式，没有调用
   ```json ... ```         ← 只有围栏没有标签时会尝试解析，失败则记 json_broken
   <tool_call>{bad json}   ← JSON 非法
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from src.env.tool_catalog import TOOL_ARGS_KEY, TOOL_CALL_CLOSE, TOOL_CALL_OPEN, TOOL_NAME_KEY
from src.eval.equivalence import parse_json_loose

# ============================================================================
# 数据结构
# ============================================================================


@dataclass
class ToolCall:
    """一次解析出来的工具调用。"""

    name: str | None
    arguments: dict[str, Any] = field(default_factory=dict)
    #: 这个调用来自哪种解析路径，用于诊断解析器本身的质量
    source: str = "tool_call_tag"
    #: 原始 JSON 文本（截断）
    raw: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments, "source": self.source}

    def pretty(self) -> str:
        return json.dumps({"name": self.name, "arguments": self.arguments}, ensure_ascii=False)


@dataclass
class ParseResult:
    """解析结果。"""

    calls: list[ToolCall]
    error: str = ""
    subtype: str = ""
    #: 文本里是否出现了 tool_call 标签（用于区分「没尝试」与「尝试了但坏了」）
    had_tag: bool = False

    @property
    def ok(self) -> bool:
        return bool(self.calls) and not self.error

    @property
    def first(self) -> ToolCall | None:
        return self.calls[0] if self.calls else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_calls": len(self.calls),
            "calls": [c.to_dict() for c in self.calls],
            "error": self.error,
            "subtype": self.subtype,
            "had_tag": self.had_tag,
        }


# ============================================================================
# 正则
# ============================================================================

_TAG_RE = re.compile(
    re.escape(TOOL_CALL_OPEN) + r"(.*?)" + re.escape(TOOL_CALL_CLOSE),
    flags=re.S | re.I,
)

#: 未闭合的 tool_call（被截断的典型形态）
_UNCLOSED_TAG_RE = re.compile(re.escape(TOOL_CALL_OPEN) + r"(.*)$", flags=re.S | re.I)

#: 裸 JSON 对象里出现 name/arguments 键的特征
_BARE_CALL_RE = re.compile(
    r"\{[^{}]*\"(?:name|tool|function)\"\s*:\s*\"[^\"]+\"[^{}]*\}",
    flags=re.S,
)

#: 顶层平铺时，这些键属于调用元信息而非参数
_META_KEYS = frozenset({"name", "tool", "function", "arguments", "parameters", "args", "tool_name"})


# ============================================================================
# 主体
# ============================================================================


def extract_tool_calls(text: str) -> ParseResult:
    """从模型输出中抽取工具调用。

    >>> r = extract_tool_calls('<tool_call>{"name": "get_weather", "arguments": {"city": "北京"}}</tool_call>')
    >>> r.ok, r.first.name, r.first.arguments
    (True, 'get_weather', {'city': '北京'})

    >>> r2 = extract_tool_calls("我来帮您查询一下航班")
    >>> r2.ok, r2.subtype
    (False, 'no_tag')

    >>> r3 = extract_tool_calls('<tool_call>{"name": "get_weather", "arguments": {city: 北京}}</tool_call>')
    >>> r3.ok, r3.subtype
    (False, 'json_broken')
    """
    raw = text or ""
    if not raw.strip():
        return ParseResult(calls=[], error="输出为空", subtype="empty", had_tag=False)

    # ---------- 路径 1：完整的 tool_call 标签 ----------
    blocks = _TAG_RE.findall(raw)
    had_tag = bool(blocks)

    if not had_tag:
        # 未闭合的标签 → 通常是被 max_new_tokens 截断
        m = _UNCLOSED_TAG_RE.search(raw)
        if m:
            block = m.group(1).strip()
            call, err = _parse_block(block)
            if call is not None:
                return ParseResult(
                    calls=[call], error="tool_call 标签未闭合（疑似生成被截断）", subtype="unclosed_tag", had_tag=True
                )
            return ParseResult(
                calls=[], error=f"tool_call 标签未闭合且内容无法解析：{err}", subtype="json_broken", had_tag=True
            )

    if blocks:
        calls: list[ToolCall] = []
        errors: list[str] = []
        for block in blocks:
            call, err = _parse_block(block.strip())
            if call is not None:
                calls.append(call)
            else:
                errors.append(err)
        if calls:
            # 部分成功也算成功，但把错误记录下来
            return ParseResult(calls=calls, error="; ".join(errors), subtype="partial", had_tag=True)
        return ParseResult(calls=[], error="; ".join(errors) or "tool_call 内容无法解析", subtype="json_broken", had_tag=True)

    # ---------- 路径 2：裸 JSON 里的 name ----------
    for m in _BARE_CALL_RE.finditer(raw):
        call, err = _parse_block(m.group(0))
        if call is not None:
            return ParseResult(calls=[call], error="", subtype="bare_json", had_tag=False)

    # ---------- 路径 3：整段就是一个 JSON 对象 ----------
    obj, _ = parse_json_loose(raw)
    if isinstance(obj, dict) and _looks_like_call(obj):
        call, err = _call_from_dict(obj, source="whole_text_json")
        if call is not None:
            return ParseResult(calls=[call], error="", subtype="whole_text_json", had_tag=False)

    # ---------- 没找到 ----------
    if TOOL_CALL_OPEN.lower() in raw.lower():
        return ParseResult(calls=[], error="检测到 <tool_call> 但没有闭合标签", subtype="unclosed_tag", had_tag=True)
    return ParseResult(calls=[], error="输出中没有工具调用", subtype="no_tag", had_tag=False)


def _looks_like_call(obj: dict[str, Any]) -> bool:
    return any(k in obj for k in _META_KEYS)


def _parse_block(block: str) -> tuple[ToolCall | None, str]:
    """解析一个 tool_call 标签内部的文本。"""
    if not block.strip():
        return None, "tool_call 标签内为空"

    obj, note = parse_json_loose(block)
    if obj is None:
        return None, f"JSON 解析失败（{note}）"
    if not isinstance(obj, dict):
        return None, f"解析结果不是对象而是 {type(obj).__name__}"
    return _call_from_dict(obj, source="tool_call_tag", raw=block[:500])


def _call_from_dict(
    obj: dict[str, Any], *, source: str, raw: str = ""
) -> tuple[ToolCall | None, str]:
    """把解析出的 dict 规范成 ToolCall。

    处理三种形态：

    1. ``{"name": X, "arguments": {...}}``            —— 标准
    2. ``{"name": X, "parameters": {...}}``           —— OpenAI 旧版字段名
    3. ``{"name": X, "origin": "PEK", "date": ...}``  —— 参数平铺到顶层
    4. ``{"function": {"name": X, "arguments": {...}}}`` —— OpenAI 包裹一层
    """
    # 形态 4
    if "function" in obj and isinstance(obj["function"], dict):
        inner = obj["function"]
        name = inner.get("name") or inner.get("tool_name")
        args = inner.get("arguments", inner.get("parameters", {}))
        if isinstance(args, str):
            parsed, _ = parse_json_loose(args)
            args = parsed if isinstance(parsed, dict) else {}
        if name:
            return ToolCall(str(name), args if isinstance(args, dict) else {}, source, raw), ""
        return None, "function 包裹层里没有 name"

    name = obj.get("name") or obj.get("tool") or obj.get("tool_name")
    if not name:
        return None, "缺少 name 字段"

    # 参数容器
    if TOOL_ARGS_KEY in obj:
        args = obj[TOOL_ARGS_KEY]
    elif "parameters" in obj:
        args = obj["parameters"]
    elif "args" in obj:
        args = obj["args"]
    else:
        # 平铺形态
        args = {k: v for k, v in obj.items() if k not in _META_KEYS}
        return ToolCall(str(name), args, source + "+flattened" if source else "flattened", raw), ""

    if isinstance(args, str):
        parsed, note = parse_json_loose(args)
        if parsed is None:
            return None, f"arguments 是字符串但无法解析成 JSON（{note}）"
        args = parsed

    if not isinstance(args, dict):
        return None, f"arguments 必须是对象，实际是 {type(args).__name__}"

    return ToolCall(str(name), args, source, raw), ""


# ============================================================================
# 输出格式校验（用于训练时判断「格式是否合法」）
# ============================================================================

_STANDARD_CALL_RE = re.compile(
    re.escape(TOOL_CALL_OPEN)
    + r"\s*\{\s*\"name\"\s*:\s*\"[^\"]+\"\s*,\s*\"arguments\"\s*:\s*\{.*?\}\s*\}\s*"
    + re.escape(TOOL_CALL_CLOSE),
    flags=re.S,
)


def is_canonical_format(text: str) -> bool:
    """输出是否「长得像一条规范的 tool_call」。

    ⚠️ **这不是 JSON 合法性检查，而是形态检查。**

    实现用正则匹配 ``<tool_call>{"name": ..., "arguments": {...}}</tool_call>``，
    只保证：

    - 恰好有一处 ``tool_call`` 标签对
    - 标签内是 ``{"name": "...", "arguments": {...}}`` 这个**形状**
    - 标签外没有多余文字

    它**不**验证 ``arguments`` 内层是合法 JSON。实测::

        <tool_call>{"name": "check_baggage", "arguments": {booking_id: BK7F2A9C}}</tool_call>
        # 键缺引号 → JSON 非法，但 is_canonical_format() 返回 True

    为什么这样设计：``format_accuracy`` 这个指标要回答的问题是
    「模型有没有把意图落成一条工具调用」，而不是「JSON 有没有语法错」。
    后者由 ``extract_tool_calls`` 的解析结果负责（它会给出 ``json_broken`` 子类型），
    两者是**互补**的两层：

    - ``is_canonical_format``：形态层（用于 format_accuracy 指标）
    - ``extract_tool_calls().ok``：语义层（用于判任务是否成功）

    因此报告里不要用 ``format_accuracy`` 去推断「JSON 合法率」——
    想表达后者请用 ``1 − FORMAT_ERROR/json_broken 占比``。

    >>> is_canonical_format('<tool_call>{"name": "get_weather", "arguments": {"city": "北京"}}</tool_call>')
    True
    >>> is_canonical_format('好的，我帮您查一下')
    False
    >>> # 形态正确但 JSON 非法 —— 这正是「形态 ≠ 合法」的证据
    >>> is_canonical_format('<tool_call>{"name": "x", "arguments": {k: 1}}</tool_call>')
    True
    """
    if not text or not text.strip():
        return False
    matches = _STANDARD_CALL_RE.findall(text.strip())
    if len(matches) != 1:
        return False
    return text.strip() == matches[0].strip()


def format_defects(text: str) -> list[str]:
    """列出一个输出的格式缺陷清单，用于诊断报告下钻。

    >>> format_defects('我来帮您查询')
    ['missing_tool_call_tag']
    """
    defects: list[str] = []
    t = text or ""
    if not t.strip():
        return ["empty_output"]

    has_open = TOOL_CALL_OPEN.lower() in t.lower()
    has_close = TOOL_CALL_CLOSE.lower() in t.lower()
    if not has_open:
        defects.append("missing_tool_call_tag")
    elif not has_close:
        defects.append("unclosed_tool_call_tag")
    if not has_open and "```" in t:
        defects.append("markdown_fence_instead_of_tag")
    if re.search(r"\{\s*'", t) or re.search(r"'\s*:", t):
        defects.append("single_quoted_keys")
    if re.search(r",\s*[}\]]", t):
        defects.append("trailing_comma")
    if has_open and not _STANDARD_CALL_RE.search(t):
        # 有标签但不是标准结构
        if TOOL_ARGS_KEY not in t:
            defects.append("arguments_flattened_or_renamed")
    if re.search(r"[\u4e00-\u9fff]\s*:", t) and has_open:
        defects.append("unquoted_chinese_keys")
    return defects


def format_accuracy(texts: list[str]) -> float:
    """一批输出的格式正确率 —— 报告里的「格式正确率」指标。

    与「任务成功率」的区别：格式对不代表工具选对、参数填对。
    DPO 常见的退化模式就是格式率飙升而任务成功率不动（reward hacking），
    所以两个指标必须分开报告。

    ⚠️ **本函数不看任务，因此对「应先追问」的任务会误判**。
    那类任务的正确输出是**自然语言追问**（本来就没有 ``tool_call``），
    而本函数只认 canonical ``tool_call`` 格式 → 判为格式错误。

    实测后果：v3 基线里 `cancellation` 的任务成功率 **62.5%**
    而格式正确率 **0.0%** —— 那个 0.0% 是**测量产物，不是模型问题**
    （该类别 22 个任务里 14 个是「应先追问」）。

    需要按任务判定时请用 :func:`format_accuracy_by_task`。
    """
    if not texts:
        return 0.0
    return sum(1 for t in texts if is_canonical_format(t)) / len(texts)


def expected_format_ok(*, is_clarify_task: bool, text: str) -> bool:
    """按**任务类型**判断格式是否正确。

    * 要调工具的任务（``is_clarify_task=False``）：必须有 canonical ``tool_call``
    * 应先追问的任务（``is_clarify_task=True``）：**不应**有 ``tool_call``，
      而应给出非空的自然语言回复

    这是「格式正确率」的正确口径。旧口径对所有任务都要求 ``tool_call``，
    于是把追问类任务的**正确答案**判成格式错误。

    >>> expected_format_ok(is_clarify_task=False,
    ...     text='<tool_call>{"name":"f","arguments":{}}</tool_call>')
    True
    >>> expected_format_ok(is_clarify_task=False, text="我不会")
    False
    >>> expected_format_ok(is_clarify_task=True, text="请问您取消的原因是什么？")
    True
    >>> expected_format_ok(is_clarify_task=True, text="")
    False
    """
    if is_clarify_task:
        # 追问类：不该有 tool_call，且必须有实际内容
        # ⚠️ 若模型**仍然调了工具**，也不算格式错误（那是「该追问却调了」，
        # 属于任务成功率的问题，不是格式问题）。但那种情形下
        # ``is_clarify_task`` 的口径会把它记为「格式对」——
        # 这是有意的：格式指标只回答「输出的**形式**合不合规」。
        return bool(text.strip())
    return is_canonical_format(text)


def format_accuracy_by_task(
    pairs: list[tuple[bool, str]],
) -> float:
    """按任务类型算格式正确率。``pairs`` 是 ``[(是否应先追问, 输出文本)]``。

    >>> format_accuracy_by_task([(False, '<tool_call>{"name":"f","arguments":{}}</tool_call>'),
    ...                          (True, "请问订单号是多少？")])
    1.0
    >>> format_accuracy_by_task([(True, "")])
    0.0
    """
    if not pairs:
        return 0.0
    return sum(
        1 for is_clarify, text in pairs
        if expected_format_ok(is_clarify_task=is_clarify, text=text)
    ) / len(pairs)


def canonical_call(name: str, arguments: dict[str, Any]) -> str:
    """生成一个 canonical 格式的工具调用文本（偏好数据构造用）。

    >>> canonical_call("get_weather", {"city": "北京"})
    '<tool_call>{"name": "get_weather", "arguments": {"city": "北京"}}</tool_call>'
    """
    body = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
    return f"{TOOL_CALL_OPEN}{body}{TOOL_CALL_CLOSE}"
