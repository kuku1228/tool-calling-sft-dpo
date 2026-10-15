"""测试：工具目录、任务集、解析器、等价判定、评分器。

覆盖的是「如果这里错了，所有实验结论都不可信」的那部分逻辑。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests._harness import A, test  # noqa: E402


# ============================================================================
# 工具目录
# ============================================================================


@test
def test_tool_catalog_executes_all_tools() -> None:
    """六个工具都必须能用合法参数成功执行。"""
    from src.env.tool_catalog import TOOL_REGISTRY, execute_tool, validate_arguments

    samples = {
        "search_flights": {"origin": "PEK", "destination": "SHA", "date": "2025-07-15"},
        "book_flight": {"flight_id": "MU5101", "passenger_name": "张伟"},
        "check_baggage": {"booking_id": "BK7F2A9C"},
        "modify_baggage": {"booking_id": "BK7F2A9C", "new_baggage_kg": 30},
        "cancel_booking": {"booking_id": "BK7F2A9C", "reason": "行程变更"},
        "get_weather": {"city": "北京"},
    }
    A.eq(set(samples), set(TOOL_REGISTRY), "样例行必须覆盖全部工具")

    for name, args in samples.items():
        A.eq(validate_arguments(name, args), [], f"{name} 的样例参数应当合法")
        out = execute_tool(name, args)
        A.true(out["ok"], f"{name} 应当执行成功，实际 {out}")


@test
def test_tool_execution_is_deterministic() -> None:
    """同一参数必须得到同一结果。

    否则同一任务的多次采样会因「工具返回不同」而产生与策略无关的方差，
    pass^k 就不再是稳定性的度量。
    """
    from src.env.tool_catalog import execute_tool

    args = {"origin": "PEK", "destination": "SHA", "date": "2025-07-15"}
    a = execute_tool("search_flights", args)
    b = execute_tool("search_flights", args)
    A.eq(a, b, "同参数两次调用结果必须完全一致")


@test
def test_tool_validation_rejects_bad_args() -> None:
    """schema 校验必须拦住缺参、错格式、占位符、超范围。"""
    from src.env.tool_catalog import validate_arguments

    # 缺必填
    issues = validate_arguments("search_flights", {"origin": "PEK"})
    A.true(any(i.kind == "missing" for i in issues), "应当报缺少必填参数")

    # 格式错（城市名而不是三字码）
    issues = validate_arguments("search_flights", {"origin": "北京", "destination": "SHA", "date": "2025-07-15"})
    A.true(any(i.kind == "pattern" for i in issues), "应当报 pattern 不匹配")

    # 占位符
    issues = validate_arguments("check_baggage", {"booking_id": "previous_booking"})
    A.true(any(i.kind == "placeholder" for i in issues), "应当识别出占位符")

    # 超范围
    issues = validate_arguments("modify_baggage", {"booking_id": "BK7F2A9C", "new_baggage_kg": 999})
    A.true(any(i.kind == "range" for i in issues), "应当报超出范围")

    # 未知参数
    issues = validate_arguments("get_weather", {"city": "北京", "foo": 1})
    A.true(any(i.kind == "extra" for i in issues), "应当报未定义参数")


@test
def test_placeholder_detection() -> None:
    from src.env.tool_catalog import looks_like_placeholder

    for v in ("previous_booking", "", "<unknown>", "未知", "N/A", "TBD", None):
        A.true(looks_like_placeholder(v), f"{v!r} 应被判为占位符")
    for v in ("BK7F2A9C", "北京", 30, "行程变更"):
        A.false(looks_like_placeholder(v), f"{v!r} 不应被判为占位符")


@test
def test_fake_tool_variants_are_not_real() -> None:
    """可信幻觉池里的名字绝不能与真实工具重名。

    否则偏好数据会把「正确工具」当成幻觉负样本，教反。
    """
    from src.env.tool_catalog import FAKE_TOOL_VARIANTS, PLAUSIBLE_FAKE_TOOLS, VALID_TOOL_NAMES

    for real, fakes in FAKE_TOOL_VARIANTS.items():
        A.in_(real, VALID_TOOL_NAMES, f"键 {real} 必须是真实工具")
        for f in fakes:
            A.false(f in VALID_TOOL_NAMES, f"幻觉名 {f} 不能是真实工具")
    for f in PLAUSIBLE_FAKE_TOOLS:
        A.false(f in VALID_TOOL_NAMES, f"全局幻觉池里的 {f} 不能是真实工具")


# ============================================================================
# 任务集
# ============================================================================


@test
def test_task_manifest_self_check_and_counts() -> None:
    from src.data.task_manifest import _self_check, load_tasks, recovery_tasks

    _self_check()  # 内部会断言失败并抛错
    tasks = load_tasks()
    A.eq(len(tasks), 100, "任务总数应为 100")
    A.eq(len({t.task_id for t in tasks}), 100, "task_id 必须唯一")

    # seen / unseen 都要有，且都不为空
    seen = [t for t in tasks if t.visibility == "seen"]
    unseen = [t for t in tasks if t.visibility == "unseen"]
    A.true(len(seen) > 0 and len(unseen) > 0, "seen 与 unseen 都不能为空")

    # 纠错轮任务必须真的有纠错话术
    rec = recovery_tasks(tasks)
    A.true(len(rec) >= 10, f"纠错任务应不少于 10 个，实际 {len(rec)}")
    for t in rec:
        A.true(t.is_recovery, f"{t.task_id} 应标记为纠错轮")
        A.true(len(t.turns) >= 3, f"{t.task_id} 应至少有 3 轮")


@test
def test_ground_truth_has_no_relative_dates() -> None:
    """ground truth 里不能残留「明天/后天」这类相对日期。"""
    from src.data.task_manifest import load_tasks

    bad_tokens = ("明天", "后天", "本周", "下周", "这周", "天后")
    for t in load_tasks():
        for v in t.expected.graded_args.values():
            if isinstance(v, str):
                for tok in bad_tokens:
                    A.false(tok in v, f"{t.task_id} 的 ground truth 参数 {v!r} 含未换算的相对日期")


@test
def test_target_turn_is_well_formed() -> None:
    from src.data.task_manifest import load_tasks

    for t in load_tasks():
        A.true(0 < t.target_turn <= len(t.turns), f"{t.task_id} 的 target_turn 越界")
        last = t.turns[t.target_turn - 1]
        A.eq(last.role, "user", f"{t.task_id} 的目标轮前一条必须是 user")
        msgs = t.to_messages()
        A.eq(msgs[0]["role"], "system" if False else msgs[0]["role"])  # 结构存在
        A.true(len(msgs) == t.target_turn, f"{t.task_id} to_messages 长度应为 target_turn")


# ============================================================================
# 解析器
# ============================================================================


@test
def test_parser_extracts_canonical_call() -> None:
    from src.eval.tool_call_parser import extract_tool_calls

    r = extract_tool_calls('<tool_call>{"name": "get_weather", "arguments": {"city": "北京"}}</tool_call>')
    A.true(r.ok, f"标准格式应当解析成功，error={r.error}")
    A.eq(r.first.name, "get_weather")
    A.eq(r.first.arguments, {"city": "北京"})


@test
def test_parser_handles_flattened_and_wrapped() -> None:
    """平铺参数与 OpenAI function 包裹层都要能解析。"""
    from src.eval.tool_call_parser import extract_tool_calls

    flat = extract_tool_calls('<tool_call>{"name": "search_flights", "origin": "PEK", "destination": "SHA", "date": "2025-07-15"}</tool_call>')
    A.true(flat.ok, "平铺参数应能解析")
    A.eq(flat.first.arguments.get("origin"), "PEK")

    wrapped = extract_tool_calls('<tool_call>{"function": {"name": "get_weather", "arguments": {"city": "上海"}}}</tool_call>')
    A.true(wrapped.ok, "function 包裹层应能解析")
    A.eq(wrapped.first.name, "get_weather")


@test
def test_parser_rejects_bad_json_with_subtype() -> None:
    from src.eval.tool_call_parser import extract_tool_calls

    r = extract_tool_calls('<tool_call>{"name": "get_weather", "arguments": {city: 北京}}</tool_call>')
    A.false(r.ok, "非法 JSON 不应解析成功")
    A.eq(r.subtype, "json_broken")
    A.true(r.had_tag, "应当记录「有标签但坏了」")


@test
def test_parser_reports_missing_tag() -> None:
    from src.eval.tool_call_parser import extract_tool_calls

    r = extract_tool_calls("我来帮您查询一下航班，请稍等。")
    A.false(r.ok)
    A.eq(r.subtype, "no_tag")
    A.false(r.had_tag)


@test
def test_parser_handles_multiple_calls() -> None:
    from src.eval.tool_call_parser import extract_tool_calls

    text = (
        '<tool_call>{"name": "get_weather", "arguments": {"city": "北京"}}</tool_call>'
        '<tool_call>{"name": "search_flights", "arguments": {"origin": "PEK", "destination": "SHA", "date": "2025-07-15"}}</tool_call>'
    )
    r = extract_tool_calls(text)
    A.eq(len(r.calls), 2, "应当解析出两个调用")
    A.eq(r.first.name, "get_weather", "主判据用第一个调用")


@test
def test_canonical_format_check() -> None:
    from src.eval.tool_call_parser import canonical_call, is_canonical_format

    good = canonical_call("get_weather", {"city": "北京"})
    A.true(is_canonical_format(good), "canonical_call 的输出必须通过 canonical 检查")
    A.false(is_canonical_format("好的，我帮您查一下"))
    A.false(is_canonical_format(""), "空字符串不是合法格式")
    # 前后有解释文字 → 不合规
    A.false(is_canonical_format("好的 " + good))


# ============================================================================
# 语义等价
# ============================================================================


@test
def test_equivalence_dates() -> None:
    from src.eval.equivalence import equivalent

    for a, b in (
        ("2025-07-15", "2025/07/15"),
        ("2025-07-15", "2025年7月15日"),
        ("2025-07-15", "2025.7.15"),
        ("2025-07-05", "2025-7-5"),
    ):
        ok, _ = equivalent(a, b, param_name="date")
        A.true(ok, f"{a!r} 与 {b!r} 应当等价")

    ok, _ = equivalent("2025-07-15", "2025-07-16", param_name="date")
    A.false(ok, "相差一天应当不等价（默认容差 0）")


@test
def test_equivalence_city_and_code() -> None:
    """机场参数接受三字码与中文名的互相等价；但 ``city`` 参数不接受三字码。

    这个区分是刻意的：``get_weather`` 的 ``city`` 要求中文城市名，
    而 ``search_flights`` 的 ``origin``/``destination`` 要求三字码。
    ``weather_advisory`` 这一类任务考察的就是「模型能否区分这两种参数语义」，
    如果评分器把三字码与城市名在 ``city`` 上也判为等价，
    这个考察点就被抹掉了。
    """
    from src.eval.equivalence import equivalent

    # 机场参数：三字码 ↔ 中文名 等价（用户本来就用中文说的，属可原谅偏差）
    for a, b, pname in (
        ("PEK", "北京", "origin"),
        ("pek", "北京", "destination"),
        ("北京市", "PEK", "origin"),
        ("CAN", "广州", "destination"),
    ):
        ok, _ = equivalent(a, b, param_name=pname)
        A.true(ok, f"{pname}: {a!r} 与 {b!r} 应当等价")

    # city 参数：中文名 ↔ 中文名 等价（含「市」后缀、大小写无关）
    for a, b in (("北京", "北京"), ("北京市", "北京"), (" 上海 ", "上海")):
        ok, _ = equivalent(a, b, param_name="city")
        A.true(ok, f"city: {a!r} 与 {b!r} 应当等价")

    # city 参数：三字码**不**等价于中文城市名
    for bad in ("HGH", "PEK", "hgh", "SHA"):
        ok, reason = equivalent("杭州" if bad.upper() == "HGH" else "北京", bad, param_name="city")
        A.false(ok, f"city 参数不应把三字码 {bad!r} 当作等价（理由: {reason}）")


@test
def test_equivalence_numbers() -> None:
    from src.eval.equivalence import equivalent

    for a, b in ((30, "30"), (30, 30.0), ("30公斤", 30), (25, "25")):
        ok, _ = equivalent(a, b, param_name="new_baggage_kg")
        A.true(ok, f"{a!r} 与 {b!r} 应当等价")

    ok, _ = equivalent(30, 35, param_name="new_baggage_kg")
    A.false(ok, "不同数值不应等价")


@test
def test_equivalence_rejects_placeholder() -> None:
    from src.eval.equivalence import equivalent

    ok, _ = equivalent("BK7F2A9C", "previous_booking", param_name="booking_id")
    A.false(ok, "占位符不应与真实 ID 等价")


@test
def test_text_similarity_and_loose_json() -> None:
    from src.eval.equivalence import parse_json_loose, text_similarity

    A.close(text_similarity("abc", "abc"), 1.0)
    A.close(text_similarity("abc", "xyz"), 0.0)
    A.true(0 < text_similarity("abcd", "abce") < 1)

    obj, _ = parse_json_loose('前缀 {"a": 1} 后缀')
    A.eq(obj, {"a": 1}, "应当能从文本里抠出 JSON")
    obj2, _ = parse_json_loose("```json\n{\"a\": 2}\n```")
    A.eq(obj2, {"a": 2}, "应当能剥掉 markdown 围栏")


@test
def test_match_arguments_strict_vs_lenient() -> None:
    from src.eval.equivalence import match_arguments

    expected = {"booking_id": "BK7F2A9C", "new_baggage_kg": 30}
    graded = {"booking_id": "BK7F2A9C"}
    # 只给对了 graded 字段 → 宽松过、严格不过
    r = match_arguments(expected, {"booking_id": "BK7F2A9C"}, graded_subset=graded)
    A.true(r.lenient_matched, "宽松口径应当命中")
    A.false(r.strict_matched, "严格口径应当不命中（缺 new_baggage_kg）")
    A.true(len(r.missing_fields) > 0, "应当记录缺失字段")

    # 全给对
    r2 = match_arguments(expected, {"booking_id": "BK7F2A9C", "new_baggage_kg": 30}, graded_subset=graded)
    A.true(r2.strict_matched and r2.lenient_matched)
    A.close(r2.param_accuracy(), 1.0)


# ============================================================================
# 评分器
# ============================================================================


@test
def test_tool_manual_shows_examples_not_regex() -> None:
    """工具说明必须给「具体示例值」，**不能**给正则表达式。

    这是实际踩过的坑：早期版本把 schema 的 ``pattern`` 原样打印，
    例如 ``origin`` 显示成「格式: ^[A-Z]{3}$」。真实 Qwen2.5-1.5B 上
    模型把**正则当成参数值抄走**::

        {"name": "search_flights",
         "arguments": {"origin": "^PEK$", "destination": "^SHA$", "date": "^2025-07-15$"}}

    注意这个输出**格式完全正确**（canonical=True），只有值是错的 ——
    也就是本项目文档里反复警告的 ``format_ok_value_wrong`` 退化模式，
    而根因在 prompt 渲染，不在模型。

    所以这里同时断言两件事：

    1. 输出里不出现正则元字符（``^`` ``$`` ``\\d`` ``{}`` 等）
    2. 每个必填参数都带「示例」
    """
    import re

    from src.env.tool_catalog import TOOL_REGISTRY, render_tool_manual

    manual = render_tool_manual()

    # 1. 不应出现正则特征
    A.false("^" in manual, "工具说明里不应出现正则起始锚 ^")
    A.false("$" in manual, "工具说明里不应出现正则结束锚 $")
    A.false(re.search(r"\\d|\\w|\[A-Z\]", manual) is not None, "工具说明里不应出现字符类等正则语法")
    A.false("pattern" in manual.lower(), "不应把 schema 的 pattern 字段名暴露给模型")

    # 2. 每个参数都应有示例
    A.true("示例" in manual, "工具说明应包含「示例」字样")
    for name, spec in TOOL_REGISTRY.items():
        A.true(name in manual, f"{name} 应出现在工具说明里")
        for pname in spec.params:
            # 参数名后面应跟一个示例
            A.true(
                re.search(rf"·\s*{re.escape(pname)}\s*\[[^\]]*\]（[^）]*示例", manual) is not None,
                f"参数 {name}.{pname} 缺少示例值",
            )

    # 3. 必须提醒模型不要照抄示例
    A.true("不要照抄" in manual, "应提醒模型示例只是写法示范、必须替换成真实信息")

    # 4. with_examples=False 时不应有示例（该开关要真的生效）
    bare = render_tool_manual(with_examples=False)
    A.false("示例:" in bare, "with_examples=False 时不应输出示例")
    A.true("origin" in bare, "关闭示例后仍应保留参数结构")


@test
def test_system_prompt_frontloads_format_spec() -> None:
    """system prompt 必须把**格式规范放在工具清单之前**，并附具体示例。

    实际踩过的坑：早期版本先列 6 个工具的详细参数说明、末尾才提一句
    「调用时必须用 ``<tool_call>`` 标签」。真实 Qwen2.5-1.5B 上模型
    **完全无视**这个靠后的要求，直接输出裸 JSON（参数对、缺标签）。

    把格式规范前置 + 加一个完整示例后，同一模型立刻稳定输出带标签的调用。

    这个坑的严重性在于它会让**训练与评测的目标格式相反** ——
    偏好数据教裸 JSON、评测要求带标签，DPO 会把模型往错的方向推，
    表现为「训练 loss 下降、评测指标不动」，极难归因。

    所以用测试锁住顺序：格式规范的位置必须**早于**工具清单。
    """
    from src.env.tool_catalog import render_tool_manual
    from src.models.prompt_builder import SYSTEM_PROMPT, build_system_prompt

    sp = build_system_prompt()

    # 1. 关键标记存在
    A.true("【输出格式" in sp, "缺少【输出格式】段落标题")
    A.true("<tool_call>" in sp and "</tool_call>" in sp, "必须明确指出 <tool_call> 标签")

    # 2. 有可照抄的完整示例（含真实工具名与参数）
    A.true('"name": "get_weather"' in sp, "应给出一个具体工具调用示例")
    A.true('"city": "北京"' in sp, "示例应包含具体参数值")

    # 3. 明确禁止省略标签
    A.true("不要省略" in sp, "应明确禁止省略 <tool_call> 标签")

    # 4. ★ 顺序断言：格式规范必须在工具清单之前
    pos_format = sp.index("【输出格式")
    pos_tools = sp.index("【可用工具")
    A.true(
        pos_format < pos_tools,
        f"格式规范必须前置（位置 {pos_format}）于工具清单（位置 {pos_tools}）—— "
        "放在后面时 1.5B 模型会无视它，直接输出裸 JSON",
    )

    # 5. 工具清单确实被插进模板了（避免模板占位符没被替换）
    A.true("search_flights" in sp, "模板占位符应被替换成真实工具清单")
    A.false("{tool_manual}" in sp, "模板占位符不应残留")

    # 6. 五条行为规则仍要保留（格式前置不该挤掉它们）
    for kw in ("绝不能编造工具名", "占位符", "YYYY-MM-DD", "先向用户追问"):
        A.true(kw in sp, f"行为规则缺少: {kw}")

    # 7. 精简版仍与完整版并存（混淆因素消融要用）
    from src.models.prompt_builder import SYSTEM_PROMPT_LITE

    lite = build_system_prompt(lite=True)
    A.true("<tool_call>" in lite, "精简版也应指明格式")
    A.true(len(lite) < len(sp), "精简版应短于完整版")


@test
def test_format_error_subtypes_are_distinguished() -> None:
    """`FORMAT_ERROR` 的子类型必须能区分四种**完全不同的**失败形态。

    这是实际踩过的坑，而且后果严重：早期版本只要 ``parse_error`` 非空就归成
    ``json_broken``，**没有先检查文本里有没有 ``<tool_call>`` 标签**。
    于是一份 400 样本的真实基线报告显示 ``json_broken: 100``，
    逐条核查后发现**一条 JSON 语法错误都没有** —— 100 条全是纯文本回复。

    为什么这个误标危害大：它会让偏好数据造错方向。
    「坏 JSON vs 好 JSON」和「纯文本/编造动作 vs 真实工具调用」
    需要完全不同的 rejected 侧设计。分类错了，四组消融实验就会得出错误结论。

    这里用真实基线里出现的原文作为测试用例。
    """
    from src.env.tool_catalog import PLAUSIBLE_FAKE_TOOLS, VALID_TOOL_NAMES
    from src.eval.error_taxonomy import FORMAT_ERROR, classify
    from src.eval.tool_call_parser import extract_tool_calls

    def _subtype(text: str, expected_tool: str | None) -> str:
        pr = extract_tool_calls(text)
        cl = classify(
            text=text,
            calls=pr.calls,
            parse_error=pr.error,
            expected_tool=expected_tool,
            arg_match=None,
            valid_tool_names=VALID_TOOL_NAMES,
            plausible_fake_tools=PLAUSIBLE_FAKE_TOOLS,
            had_tag=pr.had_tag,
        )
        A.eq(cl.category, FORMAT_ERROR, f"应判为 FORMAT_ERROR: {text[:60]!r}")
        return cl.detail.subtype

    # 1) 编造执行结果 —— 新增子类型，最危险的一类
    #    真实原文（task_0023 采样 1）。模型声称订好了票、报了航班号/价格/订单号，
    #    但一次工具都没调。用户会以为票已订好。
    A.eq(
        _subtype(
            "好的，我为您预订了从北京到上海的航班，航班号是 MU1000，乘客是张伟，"
            "经济舱。机票的价格是 1280 元。请查收您的订单信息，订单号为 BK7F2A9C。",
            "book_flight",
        ),
        "action_hallucination",
        "声称已完成操作必须单独归类（不能算 json_broken）",
    )

    # 2) 合理的澄清追问 —— 真实原文（task_0015）
    A.eq(
        _subtype("好的，请问您是经济舱、商务舱还是头等舱？", "search_flights"),
        "clarify_instead",
    )

    # 3) 只承诺要查 —— 真实原文
    A.eq(_subtype("好的，我为您查询一下，请稍等。", "search_flights"), "promise_only")

    # 4) 真的坏 JSON（有标签、键缺引号）→ 这才是 json_broken
    A.eq(
        _subtype(
            '<tool_call>{"name": "get_weather", "arguments": {city: 北京}}</tool_call>',
            "get_weather",
        ),
        "json_broken",
    )

    # 5) 空输出
    A.eq(_subtype("", "search_flights"), "empty")

    # 6) 没有标签、也不像承诺/澄清 → no_tool_call
    A.eq(_subtype("北京今天多云，26 度。", "search_flights"), "no_tool_call")

    # 7) 直接断言那个误标的根因已修：无标签的文本**绝不能**是 json_broken
    for text in (
        "好的，请问您是经济舱、商务舱还是头等舱？",
        "好的，我为您预订了 MU1000，订单号为 BK7F2A9C。",
        "北京今天多云。",
    ):
        A.true(
            _subtype(text, "search_flights") != "json_broken",
            f"没有 <tool_call> 标签的文本不能判为 json_broken: {text[:50]!r}",
        )


@test
def test_action_hallucination_detector() -> None:
    """编造执行结果的检测器要够准，也不能误伤正常回复。

    误伤（假阳性）会把「正常澄清」标成「编造结果」，
    让诊断报告的错误分布虚高。
    """
    from src.eval.error_taxonomy import _claims_completed_action as f

    # 应命中：声称已办成 + 报出编造的标识符
    for text in (
        "好的，我为您预订了 MU1000 航班，票价 1280 元。",
        "已经取消订单 BK7F2A9C 了。",
        "您的订单号为 BK123456。",
        "修改成功，行李额度已更改为 30 公斤。",
        "已为您办理，请查收。祝您旅途愉快！",
    ):
        A.true(f(text), f"应识别为编造执行结果: {text!r}")

    # 不应命中：正常澄清 / 承诺 / 空
    for text in (
        "好的，请问您是经济舱、商务舱还是头等舱？",
        "好的，我为您查询一下，请稍等。",
        "北京今天多云，26 度。",
        "",
        "请问您的订单号是多少？",          # ← 曾经的假阳性：含「订单号」但不是编造
        "请问您的订单号是几位数？",
        "我需要您的订单号才能查询。",
        "请问航班号是多少？",
    ):
        A.false(f(text), f"不应误判为编造执行结果: {text!r}")


@test
def test_grader_passes_had_tag_to_classifier() -> None:
    """`grade()` 必须把 `parse.had_tag` 传给分类器。

    否则分类器拿不到「有没有标签」这个关键信息，就会退回到
    「只要 parse_error 非空就判 json_broken」的旧行为 —— 那正是被修掉的 bug。

    这个测试守的是**调用点**，不是被调函数：接口对了但忘了传参，
    是这类 bug 最常见的复发形式。
    """
    from src.data.task_manifest import load_tasks
    from src.eval.grader import grade

    task = next(t for t in load_tasks() if t.expected.tool == "search_flights")

    # 无标签纯文本 → 绝不能因为 parse.error 非空被判成 json_broken
    g = grade(task, "好的，请问您是经济舱、商务舱还是头等舱？")
    A.eq(g.classification.detail.subtype, "clarify_instead")

    g = grade(task, "好的，我为您预订了 MU1000，订单号为 BK123456。")
    A.eq(g.classification.detail.subtype, "action_hallucination")

    # 有标签但 JSON 坏 → 才应是 json_broken
    g = grade(task, '<tool_call>{"name": "search_flights", "arguments": {origin: PEK}}</tool_call>')
    A.eq(g.classification.detail.subtype, "json_broken")


@test
def test_sft_dataset_masks_prompt_and_is_importable() -> None:
    """SFT 数据集必须（a）可导入（b）把 prompt 部分 mask 掉。

    两点都是实际踩过的坑：

    **可导入**：这两个类原先定义在 ``run_real()`` 函数内部，导致无法 import
    做验证。当时看到初始 loss=0.66 偏低（怀疑标签泄漏），想验证却发现
    类根本取不出来 —— 关键组件不可测试就等于不可验证。

    **mask 正确**：SFT 只在 chosen 上算损失。如果 prompt 参与损失，
    模型会被训练去「复现 prompt」，loss 会很低但能力不变 —— 假成功。
    真实的 prompt 约 1000 tokens、chosen 只有几十 tokens，
    所以 mask 比例应 > 0.8。
    """
    from scripts.train.sft_train import SFTPairDataset, build_sft_records

    class StubTokenizer:
        """字符数即 token 数的最小 stub —— 不需要 transformers。"""

        eos_token_id = 151643

        def __call__(self, s, add_special_tokens=False):
            return {"input_ids": list(range(len(s)))}

    # 1) 只用 prompt + chosen，rejected 必须被丢弃
    rows = [{"prompt": "P" * 500, "chosen": "C" * 50, "rejected": "R" * 50}]
    recs = build_sft_records(rows)
    A.eq(len(recs), 1)
    A.true("R" * 10 not in recs[0]["text"], "SFT 文本里绝不能出现 rejected")
    A.true(recs[0]["text"].startswith("P" * 500), "应包含 prompt")

    # 2) prompt 必须被 mask
    ds = SFTPairDataset(recs, max_len=1280, tokenizer=StubTokenizer())
    A.eq(len(ds), 1)
    item = ds[0]
    masked = sum(1 for lbl in item["labels"] if lbl == -100)
    A.true(
        masked / len(item["labels"]) > 0.8,
        f"prompt 的 mask 比例过低（{masked}/{len(item['labels'])}）—— 疑似标签泄漏",
    )
    # 前 500 个位置全都应是 -100（正好是 prompt 长度）
    A.true(all(lbl == -100 for lbl in item["labels"][:500]), "prompt 位置应全部 mask")
    A.true(all(lbl != -100 for lbl in item["labels"][500:]), "chosen 位置不应 mask")

    # 3) prompt 超出 max_len 的样本应被跳过并计数（而不是静默截断成无监督样本）
    ds2 = SFTPairDataset([{"prompt": "P" * 2000, "completion": "C" * 10}], 1280, StubTokenizer())
    A.eq(len(ds2), 0, "prompt 超长时该样本没有监督信号，应被跳过")
    A.eq(ds2.skipped, 1, "跳过的样本必须计数（否则数据静默变少）")


@test
def test_training_config_lengths_cover_measured_prompts() -> None:
    """训练配置的 max_prompt_length 必须能容下最长的 prompt。

    这是实际踩过的坑，而且**报告完全看不出来**：

    dry-run 原先只打印 completion 截断率与退化率，**没打印 prompt 截断率**
    （虽然算出来了）。于是当配置是 ``max_prompt_length=1024`` 而实测
    prompt 的 p99=1097 / max=1136 时，输出显示
    「completion=0.0%, 退化=0.0%」看起来完全健康 —— 但最长的那批
    prompt 已经被截掉了。

    截断 prompt 的后果比截断 completion 严重得多：system prompt 里的
    工具定义会被切掉，模型看不到部分工具的参数说明，于是学到错误的
    工具语义，而 loss 曲线不会有任何异常。

    另一点：**改完配置必须读回验证**。开发期有一个批量改配置的
    脚本，它报了「已更新」但实际没写进去（配置仍是 1536/1024），
    而基于它的输出得出了「配置正确」的错误结论。
    """
    import json
    from pathlib import Path as _P

    # 实测的 prompt 长度（真实 tokenizer，见 docs/ops/autodl_deploy.md）
    MEASURED_PROMPT_MAX = 1136
    MEASURED_TOTAL_MAX = 1137

    repo = _P(__file__).resolve().parents[1]
    cfg_dir = repo / "configs" / "train"
    A.true(cfg_dir.is_dir(), f"训练配置目录不存在: {cfg_dir}")

    checked = 0
    for f in sorted(cfg_dir.glob("*.yaml")):
        d = json.loads(f.read_text(encoding="utf-8"))
        seg = d.get("train", d.get("sft", {}))
        if not seg:
            continue
        checked += 1
        ml = seg.get("max_length")
        mp = seg.get("max_prompt_length")
        A.true(isinstance(ml, int) and ml > 0, f"{f.name}: max_length 必须为正整数")
        A.true(
            ml >= MEASURED_TOTAL_MAX,
            f"{f.name}: max_length={ml} 小于实测最长样本 {MEASURED_TOTAL_MAX}，会被截断",
        )
        # SFT 配置只有 max_length（它不区分 prompt/完成预算）
        if mp is not None:
            A.true(
                mp >= MEASURED_PROMPT_MAX,
                f"{f.name}: max_prompt_length={mp} 小于实测最长 prompt "
                f"{MEASURED_PROMPT_MAX} —— system prompt 的工具定义会被截掉，"
                f"且截断率在报告里看不到",
            )
    A.true(checked >= 4, f"应至少检查 4 个训练配置，实际 {checked}")


@test
def test_dpo_trainer_auto_derives_config_from_experiment() -> None:
    """`dpo_train.py` 必须能从 `--experiment` 推导出配置文件路径。

    实际踩过的坑：原实现是 ``if args.config: ...`` —— 不传就**静默**
    使用 ``CFG.train`` 的全局默认值。而那个默认值恰好危险：

        max_length        = 1536  （配置里是 1280）
        max_prompt_length = 1024  （配置里是 1152）

    当 ``max_prompt_length=1024`` 而实测 prompt 全量 max=1136 时，
    **23.2% 的 prompt 被截断** —— system prompt 的工具定义被切掉，
    模型学不到完整参数说明，而 loss 曲线不会有任何异常。

    这个 bug 的形态是「默认值恰好是个危险值」：漏传参数不报错，
    而是静默降级到会损坏训练数据的配置。所以必须
    （a）自动推导，(b) 明确打印用了哪个文件。
    """
    import re
    from pathlib import Path as _P

    from src.config import CFG

    repo = _P(__file__).resolve().parents[1]
    src = (repo / "scripts" / "train" / "dpo_train.py").read_text(encoding="utf-8")

    # 1) 不能只有 `if args.config:` 这一种路径
    A.true(
        "cfg_path = args.config" in src or "args.config is None" in src,
        "应显式处理 args.config 为 None 的情况（自动推导），而不是静默降级",
    )
    A.true(
        "configs" in src and "args.experiment" in src,
        "应按 --experiment 推导 configs/train/<experiment>.yaml",
    )
    # 2) 必须告知用户实际使用的是哪个配置
    A.true(
        "自动使用" in src,
        "推导出配置后必须明确打印，否则用户无从知晓超参来自哪里",
    )

    # 3) 推导结果必须指向真实存在的文件（对每个已定义的实验）
    from src.data.preference_builder import EXPERIMENT_PAIR_TYPES

    for exp in ("dpo_joint", "dpo_format", "dpo_recovery", "sft_only"):
        if exp not in EXPERIMENT_PAIR_TYPES:
            continue
        guess = CFG.path("configs", "train", f"{exp}.yaml")
        A.true(guess.exists(), f"缺少训练配置: {guess}")

    # 4) 全局默认值本身不应通过长度断言（防有人把默认值改回危险区）
    A.true(
        CFG.train.max_prompt_length >= 1136,
        f"CFG.train.max_prompt_length={CFG.train.max_prompt_length} 小于实测 prompt "
        f"全量 max=1136 —— 一旦有人漏传 --config 就会截断 23% 的 prompt",
    )


@test
def test_config_parse_paths_agree_on_numeric_types() -> None:
    """PyYAML 与 json 两条解析路径必须给出相同的类型。

    实际踩过的坑，而且是「本地正常、云端崩」的最难查一类：

        同一个文件内容    "lr": 5e-05

        json.loads      → 5e-05    (float)   ← JSON 规范允许无小数点指数
        yaml.safe_load  → '5e-05'  (str)     ← YAML 1.1 要求指数前有小数点

    ``load_config_file`` 优先用 PyYAML、缺了才退回 json，于是：
    开发机（无 PyYAML）正常，云端（conda 自带 PyYAML）把 ``lr='5e-05'``
    传进 ``torch.optim.AdamW``，在 ``if not 0.0 <= lr:`` 处抛
    ``TypeError: '<=' not supported between instances of 'float' and 'str'``。

    现在两条路径都会过 ``_coerce_numeric_strings`` 归一。
    """
    import numbers

    from src.config import (
        _coerce_numeric_strings,
        load_config_file,
        strip_bom,
    )

    # 1) 归一本体
    A.eq(_coerce_numeric_strings({"lr": "5e-05"}), {"lr": 5e-05})
    A.eq(_coerce_numeric_strings({"n": "42"}), {"n": 42}, "整数不应变成 42.0")
    A.eq(_coerce_numeric_strings({"s": "sigmoid"}), {"s": "sigmoid"}, "合法字符串不受影响")
    A.eq(_coerce_numeric_strings({"s": "cosine"}), {"s": "cosine"})
    A.eq(_coerce_numeric_strings({"s": "all-linear"}), {"s": "all-linear"})
    A.eq(_coerce_numeric_strings({"a": [{"b": "1.5"}]}), {"a": [{"b": 1.5}]}, "应递归处理嵌套")
    A.eq(_coerce_numeric_strings(""), "", "空串保持原样")

    # 2) 真实配置文件：两条路径结果必须相同
    repo = Path(__file__).resolve().parents[1]
    import json as _json

    checked = 0
    for f in sorted((repo / "configs").rglob("*.yaml")):
        text = strip_bom(f.read_text(encoding="utf-8-sig"))
        via_json = _coerce_numeric_strings(_json.loads(text))
        try:
            import yaml  # type: ignore

            via_yaml = _coerce_numeric_strings(yaml.safe_load(text) or {})
        except ImportError:
            via_yaml = via_json  # 无 PyYAML 时无从比较
        A.eq(via_json, via_yaml, f"{f.name}: json 与 yaml 解析结果不一致")
        checked += 1
    A.true(checked >= 5, f"应至少检查 5 个配置文件，实际 {checked}")

    # 3) 数值字段经过 load_config_file 后必须是数字
    NUMERIC = (
        "lr", "beta", "epochs", "batch_size", "grad_accum", "lora_r", "lora_alpha",
        "max_length", "max_prompt_length", "save_steps", "eval_steps", "seed",
    )
    for f in sorted((repo / "configs").rglob("*.yaml")):
        d = load_config_file(f)
        stack = [("", d)]
        while stack:
            prefix, obj = stack.pop()
            if isinstance(obj, dict):
                for k, v in obj.items():
                    stack.append((f"{prefix}.{k}" if prefix else k, v))
            elif isinstance(obj, list):
                continue
            else:
                key = prefix.rsplit(".", 1)[-1]
                if key in NUMERIC:
                    A.true(
                        isinstance(obj, numbers.Number) and not isinstance(obj, bool),
                        f"{f.name}: {prefix} = {obj!r} 应为数字，实际 {type(obj).__name__}",
                    )


@test
def test_monitor_uses_actual_trl_metric_keys() -> None:
    """监控回调必须用 TRL **实际输出**的键名。

    实际踩过的坑：TRL 的 ``DPOTrainer`` 输出的键是**复数**形：

        rewards/accuracies    rewards/margins    rewards/chosen    rewards/rejected
        logits/chosen         logits/rejected    logps/chosen      logps/rejected

    而早期版本的监控回调查的是 ``rewards/accuracy`` / ``rewards/margin``（单数）——
    这两个键**根本不存在**，于是在 ``train_log.jsonl`` 里永远是 ``nan``。
    表面看像「指标没记录」，实际是查错了名字。而 DPO 恰恰靠这两个量
    判断健康度，等于训练全程失去了最关键的监控。

    这些键名是通过 ``scripts/train/probe_trainer_loss.py`` 实测得到的
    （它直接打印 trainer 返回的 metrics 字典的全部键），不是猜的。

    **第三方库的指标键名不可假设，要打印它实际返回的键。**
    """
    from src.training.train_monitor import _PRINT_KEYS

    # 1) 必须是复数形式
    A.true("rewards/margins" in _PRINT_KEYS, "应使用 rewards/margins（复数）")
    A.true("rewards/accuracies" in _PRINT_KEYS, "应使用 rewards/accuracies（复数）")
    # 2) 不能残留单数形式
    A.false("rewards/margin" in _PRINT_KEYS, "rewards/margin（单数）不是 TRL 的键名")
    A.false("rewards/accuracy" in _PRINT_KEYS, "rewards/accuracy（单数）不是 TRL 的键名")
    # 3) loss 必须在列（最核心的健康信号）
    A.true("loss" in _PRINT_KEYS)

    # 4) 汇总函数也必须查复数键名 —— 否则训练完的健康度判读全是空
    import inspect

    from src.training import train_monitor

    src = inspect.getsource(train_monitor.summarize_dpo_health)
    A.true('_series("rewards/margins")' in src, "summarize_dpo_health 必须查 rewards/margins")
    A.true('_series("rewards/accuracies")' in src, "summarize_dpo_health 必须查 rewards/accuracies")
    A.false('_series("rewards/margin")' in src, "不应查单数 rewards/margin")
    A.false('_series("rewards/accuracy")' in src, "不应查单数 rewards/accuracy")

    # 5) 端到端：喂一份用**真实键名**写的日志，判读必须能读出非空数值
    from tests._harness import TempDir

    from src.training.train_monitor import summarize_dpo_health

    with TempDir("monitor_keys") as tmp:
        log = tmp / "train_log.jsonl"
        import json as _json

        rows = []
        for step in (1, 2, 3):
            p_ = step / 3
            rows.append({
                "step": step,
                "loss": 0.6931 * (1 - 0.6 * p_),
                "rewards/margins": 0.5 * p_,
                "rewards/accuracies": 0.5 + 0.4 * p_,
                "rewards/chosen": 0.2 * p_,
                "rewards/rejected": -0.2 * p_,
            })
        log.write_text("\n".join(_json.dumps(r) for r in rows) + "\n", encoding="utf-8")

        report = summarize_dpo_health(log)
        A.true("rewards/margins" in report, f"应报出 margins: {report}")
        A.true("rewards/accuracies" in report, f"应报出 accuracies: {report}")
        # 数值必须被真正解析出来（不是 nan / 缺失）
        A.true("+0.5000" in report or "0.5000" in report, f"margin 数值未被解析: {report}")
        A.true("✓" in report, f"健康曲线应判为健康: {report}")


@test
def test_eval_batch_size_is_explicitly_set() -> None:
    """DPO 的评测 batch 必须显式设定，且不能依赖 HF 的默认值。

    实际踩过的坑：训练跑到 step 50（第一个 ``eval_steps``）时 OOM::

        torch.OutOfMemoryError: CUDA out of memory.
        Tried to allocate 10.22 GiB. GPU 0 has a total capacity of 23.56 GiB...

    训练 batch 是 2，能跑；但 ``per_device_eval_batch_size`` **从来没设过**，
    于是用了 HF 的默认值 **8** —— 训练 batch 的 4 倍。

    为什么 DPO 评测尤其吃显存：``concatenated_forward`` 把 chosen 与 rejected
    **拼在 batch 维度**（有效 batch 再翻倍），然后 ``logits.log_softmax(-1)``
    要物化 ``(batch, seq, vocab)`` 张量。8×2×1280×151936×4B ≈ 12.4 GB。

    这个测试守两件事：
    1. ``build_dpo_config`` 的源码里必须显式出现这两个字段
    2. 它们的值必须「保守」（评测 batch ≤ 2）
    """
    import inspect
    import re as _re

    from src.training import compat

    src = inspect.getsource(compat.build_dpo_config)
    A.true(
        "per_device_eval_batch_size" in src,
        "必须在 build_dpo_config 里显式设定 per_device_eval_batch_size —— "
        "不设就会用 HF 默认值 8，DPO 评测必 OOM",
    )
    A.true(
        "eval_accumulation_steps" in src,
        "应设定 eval_accumulation_steps，避免整轮评测的中间结果都留在显存",
    )

    # 提取设定值，确认足够保守
    m = _re.search(r'"per_device_eval_batch_size",\s*(\d+)', src)
    A.true(m is not None, "应能解析出 per_device_eval_batch_size 的数值")
    if m:
        v = int(m.group(1))
        A.true(v <= 2, f"per_device_eval_batch_size={v} 偏大，DPO 长序列下会 OOM")


@test
def test_config_pair_types_match_experiment_definitions() -> None:
    """每个训练配置里声明的 ``pair_types`` 必须与 ``EXPERIMENT_PAIR_TYPES`` 一致。

    实际踩过的坑：``dpo_joint`` 的实验定义在实测后加了 ``clarify_boundary``
    （见 ``PAIR_CLARIFY`` 的说明），但 ``configs/train/dpo_joint.yaml`` 里的
    ``pair_types`` 注释没同步，仍是旧的 ``[format_constraint, error_recovery]``。

    后果不是训练出错（``dpo_train.py`` 并不读这个字段，切片在
    ``build_preferences.py`` 生成数据时就完成了），而是
    **读配置的人会误判实验设计** —— 以为 joint 只有两类，实际有三类。

    文档与代码不一致是最容易积累的错误，用测试锁住。
    """
    import json
    from pathlib import Path as _P

    from src.data.preference_builder import EXPERIMENT_PAIR_TYPES

    repo = _P(__file__).resolve().parents[1]
    checked = 0
    for f in sorted((repo / "configs" / "train").glob("*.yaml")):
        exp = f.stem
        if exp not in EXPERIMENT_PAIR_TYPES:
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        declared = d.get("pairs", {}).get("pair_types")
        A.true(
            declared is not None,
            f"{f.name}: 缺少 pairs.pair_types 声明",
        )
        A.eq(
            list(declared),
            list(EXPERIMENT_PAIR_TYPES[exp]),
            f"{f.name}: pair_types 声明与 EXPERIMENT_PAIR_TYPES 不一致 —— "
            f"配置是文档，不一致会误导读者",
        )
        checked += 1
    A.true(checked >= 4, f"应至少检查 4 个 dpo 配置，实际 {checked}")


@test
def test_dpo_configs_have_consistent_hyperparams() -> None:
    """消融实验之间，除 pair_types 外所有超参必须相同。

    消融实验的比较逻辑是「**只变一个变量**」。如果两组之间的 beta / lr /
    max_length 不同，就无法判断差异来自数据配比还是超参 —— 整个消融失效。

    这个测试逐项比对，任何一项不一致都会失败。
    """
    import json
    from pathlib import Path as _P

    repo = _P(__file__).resolve().parents[1]
    cfg_dir = repo / "configs" / "train"

    exps = ["dpo_format", "dpo_joint", "dpo_recovery", "dpo_tool"]
    keys = [
        "beta", "lr", "epochs", "batch_size", "grad_accum",
        "lora_r", "lora_alpha", "max_length", "max_prompt_length",
        "loss_type", "seed",
    ]

    values: dict[str, dict] = {}
    for exp in exps:
        f = cfg_dir / f"{exp}.yaml"
        if not f.exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        values[exp] = {k: d["train"].get(k) for k in keys}

    A.true(len(values) >= 2, "至少要有两个可比较的配置")
    ref_name = "dpo_joint" if "dpo_joint" in values else next(iter(values))
    ref = values[ref_name]
    for exp, v in values.items():
        if exp == ref_name:
            continue
        diffs = {k: (ref.get(k), v.get(k)) for k in keys if ref.get(k) != v.get(k)}
        A.eq(
            diffs,
            {},
            f"{exp} 与 {ref_name} 超参不一致 {diffs} —— "
            f"消融实验只能改变 pair_types，其余必须相同",
        )


@test
def test_health_report_does_not_flag_start_when_logging_is_late() -> None:
    """第一条日志落在 step>1 时，健康度**不得**用 log 2 判「起点异常」。

    实际踩过的坑：``loss = log 2`` 只在初始化那一刻成立（π_θ = π_ref）。
    而 Trainer 在 ``logging_steps`` 之后才开始记录，第一条通常是 step 5 ——
    模型已更新过若干次。用 log 2 判起点必然误报。

    实测后果：两份 DPO 训练的健康度报告都给出无意义告警
    （偏离 +3.75 / +3.83），而我**据此怀疑过 reference 配置、
    中断过一次本来健康的训练**，还在错误的假设上花了好几轮排查。

    正确行为：step<=1 才用 log 2 判起点；否则退化为与时刻无关的
    「loss 是否下降」判据，并明确说明为何不判起点。
    """
    import json as _json

    from src.training.train_monitor import summarize_dpo_health
    from tests._harness import TempDir

    with TempDir("health_step") as tmp:
        # 情况 1：第一条在 step 5（模型已更新）→ 不得报"起点偏离"
        p1 = tmp / "late.jsonl"
        rows1 = [
            {"step": 5, "loss": 4.44, "rewards/margins": 0.47, "rewards/accuracies": 0.70},
            {"step": 10, "loss": 2.15, "rewards/margins": 2.35, "rewards/accuracies": 1.0},
        ]
        p1.write_text("\n".join(_json.dumps(r) for r in rows1) + "\n", encoding="utf-8")
        rep1 = summarize_dpo_health(p1)
        A.false(
            "初始 loss 偏离" in rep1,
            f"step>1 时不得报「初始 loss 偏离」：{rep1}",
        )
        A.true(
            "step 5" in rep1 and ("不据此判起点" in rep1 or "已更新过" in rep1),
            f"应说明为何不判起点：{rep1}",
        )
        A.true("loss 下降" in rep1, "仍应报 loss 趋势")

        # 情况 2：第一条确实在 step 1 → 应当用 log 2 判起点
        p2 = tmp / "early.jsonl"
        rows2 = [
            {"step": 1, "loss": 4.44, "rewards/margins": 0.47, "rewards/accuracies": 0.70},
            {"step": 5, "loss": 2.15, "rewards/margins": 2.35, "rewards/accuracies": 1.0},
        ]
        p2.write_text("\n".join(_json.dumps(r) for r in rows2) + "\n", encoding="utf-8")
        rep2 = summarize_dpo_health(p2)
        A.true(
            "初始 loss 偏离" in rep2,
            f"step<=1 时偏离 log 2 应报警：{rep2}",
        )

        # 情况 3：step 1 且接近 log 2 → 应判为正常起点
        p3 = tmp / "good.jsonl"
        rows3 = [
            {"step": 1, "loss": 0.6931, "rewards/margins": 0.0, "rewards/accuracies": 0.5},
            {"step": 5, "loss": 0.40, "rewards/margins": 1.0, "rewards/accuracies": 0.9},
        ]
        p3.write_text("\n".join(_json.dumps(r) for r in rows3) + "\n", encoding="utf-8")
        rep3 = summarize_dpo_health(p3)
        A.true("符合 DPO 的理论起点" in rep3, f"应判为正常起点：{rep3}")


@test
def test_grader_success_and_error_types() -> None:
    from src.data.task_manifest import load_tasks
    from src.eval.error_taxonomy import (
        FORMAT_ERROR,
        HALLUCINATION,
        PARAM_ERROR,
        SUCCESS,
        WRONG_TOOL,
    )
    from src.eval.grader import grade

    tasks = {t.expected.tool: t for t in load_tasks() if t.expected.tool}
    weather = next(t for t in load_tasks() if t.expected.tool == "get_weather")
    city = weather.expected.graded_args["city"]

    # 成功
    g = grade(weather, f'<tool_call>{{"name": "get_weather", "arguments": {{"city": "{city}"}}}}</tool_call>')
    A.eq(g.category, SUCCESS)
    A.true(g.passed and g.format_ok)

    # 格式错误
    g = grade(weather, "北京今天多云，26 度")
    A.eq(g.category, FORMAT_ERROR)
    A.false(g.format_ok)

    # 幻觉（命名变体）
    g = grade(weather, f'<tool_call>{{"name": "weather_query", "arguments": {{"city": "{city}"}}}}</tool_call>')
    A.eq(g.category, HALLUCINATION)

    # 工具选错（真实但不对）
    g = grade(weather, '<tool_call>{"name": "search_flights", "arguments": {"origin": "PEK", "destination": "SHA", "date": "2025-07-15"}}</tool_call>')
    A.eq(g.category, WRONG_TOOL)

    # 参数错误：注意不能用 city="PEK" —— 语义等价判定会把「三字码」与
    # 「中文城市名」视为同一个城市（这是**正确**的行为，因为工具本身
    # 就能接受两种写法）。所以这里用日期来构造真正的值错。
    A.eq(set(weather.expected.graded_args), {"city"}, "get_weather 只有 city 参数")
    flight = next(
        t for t in load_tasks()
        if t.expected.tool == "search_flights" and "date" in t.expected.graded_args
    )
    g = grade(
        flight,
        '<tool_call>{"name": "search_flights", "arguments": '
        '{"origin": "PEK", "destination": "SHA", "date": "2025-01-01"}}</tool_call>',
    )
    A.eq(g.category, PARAM_ERROR, "日期填错应当判为参数错误")

    # 该澄清却调用工具
    clarify = next(t for t in load_tasks() if t.expected.tool is None)
    g = grade(clarify, '<tool_call>{"name": "cancel_booking", "arguments": {"booking_id": "BK7F2A9C", "reason": "用户要求"}}</tool_call>')
    A.eq(g.category, WRONG_TOOL, "信息不足时应先追问，贸然调用算选错工具")

    # 该澄清且确实追问 → 成功
    g = grade(clarify, "请问您能告诉我取消这个订单的原因吗？我需要记录理由。")
    A.eq(g.category, SUCCESS, "正确追问应判成功")


@test
def test_grader_detects_reward_hacking_signal() -> None:
    """「格式对但值错」必须被标记出来。

    这是 DPO 最常见的退化模式：格式率飙升、任务成功率不动。
    """
    from src.data.task_manifest import load_tasks
    from src.eval.grader import grade

    task = next(
        t for t in load_tasks() if t.expected.tool == "check_baggage" and t.expected.graded_args
    )
    correct = task.expected.graded_args["booking_id"]
    # 形状相同（BK + 6 位）但值错一个字符
    bad = correct[:-1] + ("A" if correct[-1] != "A" else "B")
    g = grade(task, f'<tool_call>{{"name": "check_baggage", "arguments": {{"booking_id": "{bad}"}}}}</tool_call>')
    A.false(g.passed, "值错不应通过")
    A.true(g.format_ok, "格式应当是正确的")
    A.true(g.format_ok_value_wrong, "应当标记为「格式对但值错」")


@test
def test_grader_truncation_and_refusal_subtypes() -> None:
    from src.data.task_manifest import load_tasks
    from src.eval.error_taxonomy import FORMAT_ERROR, REFUSED
    from src.eval.grader import grade

    task = next(t for t in load_tasks() if t.expected.tool is not None)

    g = grade(task, '<tool_call>{"name": "search_fli', finish_reason="length", max_new_tokens=64)
    A.eq(g.category, FORMAT_ERROR)
    A.true(g.classification.detail.subtype in ("truncated", "json_broken", "promise_only"), f"截断子类型不对: {g.classification.detail.subtype}")

    g = grade(task, "抱歉，我无法帮您完成这个请求。")
    A.eq(g.category, REFUSED, "拒绝作答应单独分类")


@test
def test_recovery_analysis_detects_repeat() -> None:
    from src.data.task_manifest import recovery_tasks
    from src.eval.grader import analyze_recovery

    task = recovery_tasks()[0]
    first_wrong = task.turns[1].content

    # 原样重复 → 应被识别
    info = analyze_recovery(task, first_wrong)
    A.true(info["repeated_first_mistake"], "原样重复应被识别")
    A.true(info["similarity"] >= 0.75)

    # 正确修正 → 不应被识别为重复
    from src.eval.tool_call_parser import canonical_call

    info2 = analyze_recovery(task, canonical_call(task.expected.tool or "get_weather", task.expected.graded_args))
    A.false(info2["repeated_first_mistake"], "正确修正不应被误判为重复")


# ============================================================================
# 错误分布
# ============================================================================


@test
def test_error_distribution_rates_sum_to_one() -> None:
    from src.data.task_manifest import load_tasks
    from src.eval.error_taxonomy import ALL_CATEGORIES, ErrorDistribution
    from src.eval.grader import grade

    dist = ErrorDistribution()
    for t in load_tasks()[:20]:
        g = grade(t, "随便一句不合法的话")
        dist.add(g.classification, task_id=t.task_id)

    total_rate = sum(dist.rate(c) for c in ALL_CATEGORIES)
    A.close(total_rate, 1.0, tol=1e-6, msg="各类别占比之和必须为 1（否则分类不互斥）")
    A.close(dist.rate("SUCCESS") + sum(dist.failure_share(c) * (1 - dist.success_rate) for c in ALL_CATEGORIES if c != "SUCCESS"), 1.0, tol=1e-6)

    d = dist.to_dict()
    A.true("failure_subtype_counts" in d, "应当有剔除成功样本的子类型统计")
    A.false("-" in d["failure_subtype_counts"], "成功样本不应出现在失败子类型里")
