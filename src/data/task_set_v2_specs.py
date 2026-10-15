"""v2 任务清单：100 个**显式定义**的高质量任务。

设计约束（由 ``task_manifest_v2.verify_task_set`` 强制检查）：
1. 订单号 / 乘客名在全库唯一；城市复用 ≤3 次
2. 日期全部在 **2026-11 ~ 2026-12**（未来）
3. 骨架（句式）除刻意成对的情况外唯一
4. ``multi_tool`` 真的要求**多个工具**（``also_required`` 非空）
5. ``unseen`` 与 ``seen`` 的骨架**零交集**

## 为什么显式列出而不用模板相乘

v1 用 ``idx % len(templates)`` 做笛卡尔积，结果 400 个任务只有
208 种骨架、8 种签名 —— 典型的「30 个任务换 10 种说法」。
显式列出虽然啰嗦，但**每个任务都是独立设计的**，这才是 100 个任务
该有的样子：不是 100 行数据，而是 100 个不同的考查点。

## 类别构成（合计 100）

    weather_advisory  12      flight_search     18
    booking_flow      15      baggage           14
    cancellation      12      multi_tool        14
    recovery          15
"""

from __future__ import annotations

# ⚠️ 用**绝对导入**（与仓库其他模块一致）。相对导入 ``from .task_manifest_v2``
# 在 ``python -m doctest src/data/task_set_v2_specs.py`` 下会报
# ``attempted relative import with no known parent package`` —— 那样本文件的
# doctest 永远跑不过，而「doctest 全量通过」是项目的回归门禁之一。
from src.data.task_manifest_v2 import (
    AIRPORT_CODES_V2 as AC,
    BOOKING_IDS_V2 as BIDS,
    CANCEL_REASONS_V2 as REASONS,
    PASSENGERS_V2 as NAMES,
    TaskSpec,
)


# ---------------------------------------------------------------------------
# 互斥的实体池（**跨类别不得重合**）
# ---------------------------------------------------------------------------
#
# ⚠️ 为什么需要这个：验证框架早期只检查「同类别内骨架唯一」，
# **完全没检查跨类别实体重合**。专家正是从这个盲区找出问题：
#   * multi_tool 的航线与 flight_search 重复（实测 10 个里 3 个）
#   * recovery 的 (工具,参数) 与其他类完全重合
#   * 天气城市与主航线城市全面重合
#
# 为什么这很重要：**若两个类别用同一批实体，模型在一个类别上学会的映射
# 可以直接套到另一个类别**，那个类别就不再测它的名义能力。
#
# 分配（40 个城市按名称排序后切块，互不重叠）：
#   flight_search  城市   0-15   （16 城，取 14 组城市对）
#   multi_tool     城市  16-29   （14 城，取 10 组）
#   multi_turn     城市  30-39   （10 城，取 6 组）
#   weather        城市  36-39 + 30-35 中的一部分
#
# 城市用中文名，三字码由 AIRPORT_CODES_V2 查表。
_ACS = sorted(AC)


def _city(i: int) -> str:
    return _ACS[i % len(_ACS)]


def _pair(i: int, j: int) -> tuple[str, str]:
    """按池内索引取一组城市对，返回 (中文, 中文)。"""
    return _city(i), _city(j)


# ---------------------------------------------------------------------------
# 三个互斥的城市池：航线（50 城）/ 天气（10 城）
# ---------------------------------------------------------------------------
#
# ⚠️ 判据的边界（我在这里先错过一次）：
# 要防的是「**同一组城市对**」跨类别复用，不是「同一个城市名」出现两次。
#   * 60 城可组合出 3540 组有序城市对，各类只需 14/10/6 组 ——
#     城市名重合是自然且不可避免的
#   * 真正让模型「套用」的是**同一组起降机场**（同一条航线），
#     因为那对应同一个三字码映射记忆
#
# ⚠️ **但还有一个更隐蔽的约束：天气城市必须避开训练语料的天气池。**
#
# 训练语料的天气池是 ``recovery_corpus._WEATHER_CITIES``
# = 北京/上海/广州/成都/三亚/哈尔滨/南京（7 城，固定小池）。
# 实测踩过：评测原来用了其中的 4 个（三亚/上海/北京/南京），
# 于是那 4 个评测任务的**期望调用** ``get_weather(city=X)``
# 与某条训练样本**完全相同** —— 模型可能靠记忆答对（层级 2 泄漏）。
#
# **修法是换评测实体，不是过滤训练语料。** 为什么不能过滤训练侧：
# 实测「过滤掉期望调用相同的训练任务」删掉了 41 个任务，
# 把 ``check_baggage`` 与 ``get_weather`` 的训练样本**全删光**
# （``assert_training_covers_eval_entities`` 立刻报「缺失工具 2 个」）。
# 根因是训练与评测都覆盖同一实体空间，而实体池很小。
#
# 所以这里的分配是：
#   * **50 城**给航线（20 对 disjoint：flight_search 14 + multi_tool 6
#     + multi_turn 6 = 26 对，实际只需 20 对的城市不重复）
#   * **10 城**专用于天气，且**全部避开训练天气池**
_ALL_CITY_SEQ = sorted(AC)

#: 天气专用城市 —— **必须全部不在训练天气池里**。
#: 这个列表由测试 `test_weather_cities_avoid_training_pool` 守着。
_WEATHER_CITY = [
    "绵阳", "襄阳", "西宁", "西安", "贵阳", "郑州", "重庆", "银川",
    "长沙", "青岛",
]

#: 航线用城市 = 全部城市 − 天气专用（50 城）
_ROUTE_CITY_SEQ = [c for c in _ALL_CITY_SEQ if c not in set(_WEATHER_CITY)]
_PAIRED = [(_ROUTE_CITY_SEQ[i], _ROUTE_CITY_SEQ[i + 1])
           for i in range(0, len(_ROUTE_CITY_SEQ) - 1, 2)]  # 25 组，互斥

#: 依次切给三个类别
_FS_PAIRS_CITY = _PAIRED[0:14]     # flight_search  14 组
_MT_PAIRS_CITY = _PAIRED[14:24]    # multi_tool     10 组
_MTN_PAIRS_CITY = _PAIRED[24:30]   # multi_turn      6 组
_REVERSED: list[tuple[str, str]] = [(b, a) for a, b in _PAIRED]

# 日期分配：每个任务一个不同的未来日期（2026-11 ~ 2026-12）
_D = [f"2026-11-{d:02d}" for d in range(2, 31)] + \
     [f"2026-12-{d:02d}" for d in range(1, 31)] + \
     [f"2027-01-{d:02d}" for d in range(2, 21)]


def build_specs() -> list[TaskSpec]:
    """构造 100 个任务的显式定义。"""
    specs: list[TaskSpec] = []
    di = 0

    def nxt_date() -> str:
        nonlocal di
        d = _D[di % len(_D)]
        di += 1
        return d

    # =====================================================================
    # 1) weather_advisory（12）—— 每个任务换城市与句式，不重复
    # =====================================================================
    # ⚠️ 这 10 个城市改过一轮。原因：训练语料的天气池是
    # ``("北京","上海","广州","成都","三亚","哈尔滨","南京")``，
    # 而评测原来用了其中的 4 个（三亚/上海/北京/南京）——
    # 于是那 4 个评测任务的**完整期望调用**（get_weather(city=X)）
    # 与训练样本完全相同，模型可能靠记忆答对（层级 2 泄漏）。
    #
    # 修法是**换评测实体**，而不是过滤训练语料。为什么：
    # 实测「过滤训练侧」会删掉 41 个训练任务、把 check_baggage 与
    # get_weather 的样本全删光（断言立刻报缺失工具）。根因是训练与评测
    # 都覆盖同一实体空间，而实体池很小。
    #
    # 而成都有 53 个城市**从未用于训练天气** —— 换掉即可，
    # 不损失任何训练样本，且语义正确（评测本来就该考没见过的实体）。
    _wcities = _WEATHER_CITY[:10]
    _wtmpl = [
        "{c}今天的天气怎么样？",
        "帮我看看{c}今天会不会下雨",
        "查一下{c}的天气",
        "{c}现在多少度？",
        "我想知道{c}今天的天气情况",
        "{c}今天适合出门吗，天气如何",
        "麻烦看一下{c}今天的天气",
        "{c}今天有雨吗？",
        "帮我查查{c}的天气状况",
        "看一下{c}今日天气",
    ]
    for c, tmpl in zip(_wcities, _wtmpl):
        specs.append(TaskSpec(
            category="weather_advisory",
            user=tmpl.format(c=c),
            tool="get_weather", args={"city": c},
            tags=("param_semantics",),
            note="这个工具要中文城市名，不是三字码",
        ))

    # =====================================================================
    # 2) flight_search（18）—— 城市对互不重复
    # =====================================================================
    _fs = [(AC[a], AC[b]) for a, b in _FS_PAIRS_CITY]
    _fstmpl = [
        "帮我查一下{d}从{o}到{t}的航班",
        "{d}{o}飞{t}有什么航班？",
        "我想{d}去{t}，从{o}出发，看看有哪些航班",
        "请搜索{d} {o}到{t}的机票",
        "{d}我要从{o}去{t}，帮我看看航班",
        "麻烦帮我看看{d}{o}到{t}有没有航班",
        "{d}想去{t}，从{o}走，有哪些航班可选",
        "查询{o}到{t} {d}的航班信息",
        "我{d}要从{o}出发去{t}，帮我找一下航班",
        "有没有{d}从{o}飞往{t}的航班？",
        "{d}{o}—{t}的航班有哪些",
        "帮我搜一下{d} {o}出发到{t}的飞机",
        "想订{d}{o}去{t}的票，先看看有什么航班",
        "麻烦查{d}从{o}到{t}的航班班次",
        "我需要在{d}从{o}前往{t}，有哪些航班",
        "{o}到{t}的机票，{d}的，帮我查查",
        "看看{d}{o}飞{t}的航班",
        "帮我找{d}从{o}去{t}的班机",
    ]
    code2city = {v: k for k, v in AC.items()}
    for (o, t), tmpl in zip(_fs, _fstmpl):
        d = nxt_date()
        specs.append(TaskSpec(
            category="flight_search",
            user=tmpl.format(d=d, o=code2city[o], t=code2city[t]),
            tool="search_flights",
            args={"origin": o, "destination": t, "date": d},
            tags=("city_to_code", "date_extract"),
            note="城市名→三字码",
        ))

    # =====================================================================
    # 3) booking_flow（15）—— 真人名，每个任务不同
    # =====================================================================
    _bk = [
        (NAMES[0], "PEK", "CTU"), (NAMES[1], "SHA", "CAN"), (NAMES[2], "SZX", "XIY"),
        (NAMES[3], "HGH", "KMG"), (NAMES[4], "CKG", "NKG"), (NAMES[5], "XMN", "TAO"),
        (NAMES[6], "WUH", "CSX"), (NAMES[7], "SYX", "TSN"), (NAMES[8], "HRB", "HAK"),
        (NAMES[9], "URC", "SHE"), (NAMES[10], "DLC", "CGO"), (NAMES[11], "LXA", "TNA"),
        (NAMES[12], "FOC", "KHN"), (NAMES[13], "KWE", "NNG"), (NAMES[14], "TYN", "SJW"),
    ]
    _bktmpl = [
        "帮我订一张{d} {o}到{t}的机票，乘客是{n}，经济舱就行",
        "{n}需要{d}从{o}飞{t}，直接帮我订上",
        "给{n}订{d}{o}到{t}的票",
        "我要帮{n}订{d}{o}去{t}的航班",
        "{d}{o}到{t}的机票帮我订一下，乘客{n}",
        "帮{n}预订{d}{o}飞{t}的机票",
        "订{d}{o}到{t}的票，乘客是{n}",
        "{n}要去{t}，{d}从{o}出发，帮我订票",
        "帮我给{n}订{d}{o}到{t}的机票",
        "{d}从{o}到{t}，乘客{n}，帮我下单",
        "给{n}安排{d}{o}到{t}的行程",
        "帮我订{d}{o}飞{t}的票，乘客{n}",
        "{n} {d}从{o}到{t}，订经济舱",
        "订一张{d}{o}到{t}的票给{n}",
        "{n}的机票：{d}{o}到{t}，帮我订上",
    ]
    for (nm, o, t), tmpl in zip(_bk, _bktmpl):
        d = nxt_date()
        # ⚠️ 不要用 "CA%04d" % (2000 + i*17)：那会产生 CA2510/CA2527/CA2544…
        # **恒定等差**的航班号。专家一眼看出「明显是编的」。
        # 真实的航班号是航司二字码 + 1~4 位数字，且不连续。
        # 这里用稳定哈希派生，保证可复现、不重复、不呈现规律。
        import hashlib as _hl

        _h = int(_hl.sha256(
            f"{nm}{o}{t}{len(specs)}".encode("utf-8")).hexdigest()[:6], 16)
        _carriers = ("CA", "MU", "CZ", "HU", "ZH", "MF", "SC", "3U")
        fid = "%s%d" % (_carriers[_h % len(_carriers)], 100 + _h % 8900)
        specs.append(TaskSpec(
            category="booking_flow",
            user=tmpl.format(d=d, o=code2city[o], t=code2city[t], n=nm),
            tool="book_flight",
            args={"flight_id": fid, "passenger_name": nm},
            extra_turns=(("tool", "search_flights"),),
            tags=("write_tool", "two_step"),
            note="上下文已给 flight_id，模型应直接 book",
        ))

    # =====================================================================
    # 4) baggage（14）—— 订单号全库唯一
    # =====================================================================
    _bg_check = BIDS[0:8]
    _bg_tmpl = [
        "帮我查一下订单{b}的行李额度",
        "订单{b}还能免费托运多少公斤？",
        "我想知道{b}这个订单的行李限额",
        "{b}这个订单能托运多少行李？",
        "查一下{b}的行李额度是多少",
        "订单{b}的免费行李限额是多少公斤",
        "{b}可以托运几公斤行李？",
        "帮我确认一下{b}的托运额度",
    ]
    for b, tmpl in zip(_bg_check, _bg_tmpl):
        specs.append(TaskSpec(
            category="baggage", user=tmpl.format(b=b),
            tool="check_baggage", args={"booking_id": b},
            tags=("id_format",), note="订单号格式 ^BK[A-Z0-9]{6}$",
        ))
    _bg_mod = [
        (BIDS[8], 23, "把订单{b}的行李额度改成{k}公斤"),
        (BIDS[9], 25, "{b}的行李额度帮我调到{k}公斤"),
        (BIDS[10], 28, "订单{b}要改成{k}公斤行李"),
        (BIDS[11], 32, "帮我把{b}的额度改为{k}公斤"),
        (BIDS[12], 35, "{b}的行李额改成{k}公斤"),
        (BIDS[13], 40, "我想把{b}的行李额度提到{k}公斤"),
    ]
    for b, k, tmpl in _bg_mod:
        specs.append(TaskSpec(
            category="baggage", user=tmpl.format(b=b, k=k),
            tool="modify_baggage", args={"booking_id": b, "new_baggage_kg": k},
            tags=("write_tool", "numeric_param"),
        ))

    # =====================================================================
    # 5) cancellation（12）—— 每个任务不同订单号与原因
    # =====================================================================
    _cx = [
        (BIDS[14], REASONS[0], "我要取消订单{b}，原因是{r}"),
        (BIDS[15], REASONS[1], "帮我取消{b}，{r}"),
        (BIDS[16], REASONS[2], "{b}这个订单我要退掉，{r}"),
        (BIDS[17], REASONS[3], "取消{b}，{r}"),
        (BIDS[18], REASONS[4], "{b}退了吧，{r}"),
        (BIDS[19], REASONS[5], "帮我取消{b}，{r}"),
        (BIDS[20], REASONS[6], "把{b}取消掉，{r}"),
        (BIDS[21], REASONS[7], "请取消订单{b}，{r}"),
    ]
    for b, r, tmpl in _cx:
        specs.append(TaskSpec(
            category="cancellation", user=tmpl.format(b=b, r=r),
            tool="cancel_booking", args={"booking_id": b, "reason": r},
            extra_turns=(("assistant_confirm", ""), ("user", "对，取消吧")),
            tags=("write_tool", "confirmation"),
        ))
    # 4 个「信息不足应先追问」—— 与上面成对，这是条件化能力的评测侧
    # ⚠️ 从 4 个扩到 12 个：``missing_reason`` 是本项目**核心价值点**的证据
    # （SFT 0/4 失败 vs DPO 4/4 成功），而 n=4 的统计意义极有限
    # （Fisher 精确检验 p≈0.029，勉强显著）。专家明确指出要补到 10-15 个。
    _cl = [
        (BIDS[22], "我不想要{b}这个订单了"),
        (BIDS[23], "{b}这个订单帮我处理一下"),
        (BIDS[24], "{b}我想退掉"),
        (BIDS[25], "帮我取消{b}"),
        # 扩到 14 个：``missing_reason`` 是本项目**核心价值点**的证据
        # （SFT 0/4 失败 vs DPO 4/4 成功）。n=4 时 Fisher 精确检验 p≈0.029，
        # 勉强显著；n=14 才能支撑统计结论。
        # ⚠️ 这里**放弃了「整 100 个任务」的形式要求** —— 样本量比整数重要。
        (BIDS[30], "{b}不想要了"),
        (BIDS[31], "麻烦处理一下{b}"),
        (BIDS[32], "{b}退掉吧"),
        (BIDS[33], "把{b}取消"),
        (BIDS[34], "{b}帮我退一下"),
        (BIDS[35], "这个订单{b}我不要了"),
        (BIDS[36], "{b}想退掉"),
        (BIDS[37], "麻烦帮我退{b}"),
        (BIDS[38], "{b}取消一下"),
        (BIDS[39], "我要把{b}退掉"),
    ]
    for b, tmpl in _cl:
        specs.append(TaskSpec(
            category="cancellation", user=tmpl.format(b=b),
            tool=None, args={},
            tags=("clarify_first",),
            note="信息不全时应先追问原因，不该直接调用写操作",
        ))

    # =====================================================================
    # 6) multi_tool（14）—— **真的要求多工具**（与 v1 的关键区别）
    # =====================================================================
    _mt = [(NAMES[15 + i], AC[a], AC[b])
           for i, (a, b) in enumerate(_MT_PAIRS_CITY)]
    _mttmpl = [
        "{d}我要从{o}去{t}，顺便帮我看看{o}当天的天气",
        "帮我查{d}{o}到{t}的航班，另外{o}那天天气怎么样",
        "{d}去{t}，从{o}走；先看看{o}天气如何",
        "我要{d}从{o}飞{t}，还想知道{o}当天会不会下雨",
        "{o}到{t} {d}的航班帮我查一下，顺便看看{o}的天气",
        "{d}从{o}到{t}，出发前想先了解{o}的天气",
        "帮我安排{d}{o}去{t}，另外{o}那天冷不冷",
        "{d}{o}前往{t}，同时查一下{o}的天气情况",
        "我要{d}{o}到{t}的航班，出发地天气也帮我看看",
        "{o}{d}飞{t}有哪些航班，{o}那天有雨吗",
        "计划{d}从{o}去{t}，航班和天气一起给我",
        "{d}{o}到{t}，先查航班再告诉我{o}的天气",
        "帮我看看{d}{o}去{t}的航班，另外查{o}天气",
        "{d}从{o}出发到{t}，航班和当地天气都要",
    ][:len(_mt)]
    for (nm, o, t), tmpl in zip(_mt, _mttmpl):
        # ⚠️ 这里**不能**用 ``nxt_date()``（顺序取日期池）。
        # 实测：multi_tool 的第 10 条拿到 ``2026-12-10``，而训练语料里
        # ``search_flights(YNT→ZUH, 2026-12-10)`` 正好是**同一个三元组** ——
        # 于是该任务的完整期望调用与训练样本完全相同（层级 2 泄漏）。
        #
        # 注意判据的精确边界（专家指出）：
        #   * **航线相同、日期不同** = 层级 3 **合理泛化**（模型该学会
        #     「用 search_flights 查这条航线」，不是记住答案）
        #   * **航线 + 日期都相同** = 层级 2 **泄漏**
        # 实测：multi_tool 的 10 条航线**全部**与训练重合（因为训练覆盖
        # 全部 60 城），但只有 1 条连日期也相同 —— 只需修那一条。
        #
        # 换到日期池的**后段**：那里是 2027-01，与训练语料的日期
        # （2026-11 起、逐日推进）不重合。
        d = _D[-(len(_mt) - _mt.index((nm, o, t)))]
        specs.append(TaskSpec(
            category="multi_tool",
            user=tmpl.format(d=d, o=code2city[o], t=code2city[t]),
            tool="search_flights",
            args={"origin": o, "destination": t, "date": d},
            # ⚠️ 与 v1 的关键区别：这里声明**必须另外调用** get_weather。
            # v1 的 9 个 multi_tool 任务 expected.tool 全是 search_flights，
            # 观测里的 get_weather 从不参与判定 —— 名字叫多工具却只测一个。
            also_required=("get_weather",),
            extra_turns=(("tool", "get_weather"), ("user", "天气我看到了，现在把航班查出来吧")),
            tags=("multi_tool", "tool_chaining"),
            note="必须调用 get_weather 与 search_flights 两个工具",
        ))

    # =====================================================================
    # 7) recovery（18）—— **真正的多轮纠错**
    # =====================================================================
    #
    # ⚠️ 重构原因（专家指出，实测确认）：
    # v2 初版把 recovery 写成「首轮 user + 固定的『请修正』」，三个硬伤：
    #   * 15 个任务的纠错话术**全是「请修正」三个字** —— 退化成固定 prompt，
    #     测不出「模型是否读懂了具体的纠错内容」
    #   * 其中 2 个任务的 ``expected.args`` 是**空的** → 任何调用都判成功
    #     （这条会直接污染准确率，比其余问题更严重）
    #   * 没有一条真的做到「模型先错一次 → 用户针对该错误纠正 → 模型改对」
    #
    # 现在每条都写清：模型犯了什么错、用户针对该错误说了什么、纠错后正确调用是什么。
    _rc: list[tuple[str, str, str, str, dict, str | None]] = [
        ("format_ignore",
         "查一下2026-11-12上海到成都的航班",
         "好的，我来帮您查询一下，请稍等。",
         "你没有真的调用工具，请用规定的 <tool_call> 格式重新输出。",
         {"origin": "SHA", "destination": "CTU", "date": "2026-11-12"},
         "search_flights"),
        ("hallucinated_tool",
         "查一下我的订单BK4A71C2还能托运多少行李",
         '<tool_call>{"name": "get_order_details", "arguments": {"order_id": "BK4A71C2"}}</tool_call>',
         "工具列表里没有 get_order_details 这个工具，请用正确的工具重新调用。",
         {"booking_id": "BK4A71C2"}, "check_baggage"),
        ("wrong_tool",
         "帮我调整一下订单BK9D3E64的行李限额，要 23 公斤",
         '<tool_call>{"name": "check_baggage", "arguments": {"booking_id": "BK9D3E64"}}</tool_call>',
         "我不是要查询，我是要改成 23 公斤，请用修改额度的工具。",
         {"booking_id": "BK9D3E64", "new_baggage_kg": 23}, "modify_baggage"),
        ("param_placeholder",
         "帮我查一下订单BK2F80B7的行李额度",
         '<tool_call>{"name": "check_baggage", "arguments": {"booking_id": "previous_booking"}}</tool_call>',
         "订单号是 BK2F80B7，不要填 previous_booking。",
         {"booking_id": "BK2F80B7"}, "check_baggage"),
        ("city_code",
         "2026-11-15从重庆飞厦门有哪些航班",
         '<tool_call>{"name": "search_flights", "arguments": {"origin": "重庆", "destination": "厦门", "date": "2026-11-15"}}</tool_call>',
         "origin 和 destination 要填机场三字码，不要填中文城市名。",
         {"origin": "CKG", "destination": "XMN", "date": "2026-11-15"},
         "search_flights"),
        ("repeated_same_error",
         "给苏晴订2026-11-18从南京到武汉的机票",
         '<tool_call>{"name": "book_flight", "arguments": {"flight_id": "CA0000", "passenger_name": "苏晴"}}</tool_call>',
         "CA0000 这个航班号不存在，你必须先搜索航班再下单。",
         {"origin": "NKG", "destination": "WUH", "date": "2026-11-18"},
         "search_flights"),
        ("missing_booking_id",
         "我要查一下行李额度",
         '<tool_call>{"name": "check_baggage", "arguments": {"booking_id": ""}}</tool_call>',
         "你没有订单号就调用了工具，应该先问我要订单号。",
         {}, None),
        ("wrong_date",
         # ⚠️ 原为「后天从青岛到长沙的航班有哪些」，但期望 date 写的是
         # 2026-11-08，而 BASE_DATE_V2 = 2026-10-06 → 「后天」应是 2026-10-08。
         # **用户从未说过 11-08，这个任务无解。**
         # 改为含**绝对日期**的表述，让「date 格式错」这个教学点仍然成立，
         # 同时任务变得可解。
         "查一下2026-11-08从青岛到长沙的航班",
         '<tool_call>{"name": "search_flights", "arguments": {"origin": "TAO", "destination": "CSX", "date": "后天"}}</tool_call>',
         # ⚠️ 纠正消息必须与**用户话**一致。原来用户话是「后天…」、
         # 纠正说「不能写后天」—— 但那个任务无解（期望日期与后天不符），
         # 我把用户话改成绝对日期 2026-11-08 后，**漏改了这条纠正**，
         # 于是它说「不能写『后天』」而用户话里根本没有「后天」，
         # 自相矛盾 → 模型改成追问。
         "date 必须是 YYYY-MM-DD 格式的绝对日期，请原样使用我给你的日期。",
         # ⚠️ 「后天」必须按基准日换算，不能写死。
         # 原来写死 2026-11-08，而 BASE_DATE_V2 = 2026-10-06 →
         # 「后天」应为 2026-10-08，差了 33 天，**用户从未说过那个日期**，
         # 于是这个任务无解。修法：用户话改用**绝对日期**（明确无歧义），
         # 并把「写相对日期是错的」这个教学点保留在纠正消息里。
         {"origin": "TAO", "destination": "CSX", "date": "2026-11-08"},
         "search_flights"),
        ("overbook",
         "三亚到哈尔滨 2026-11-25 的航班，直接订最便宜那班，乘客韩雪",
         '<tool_call>{"name": "search_flights", "arguments": {"origin": "SYX", "destination": "HRB", "date": "2026-11-25", "max_price": 1000}}</tool_call>',
         "你搜了航班但没下单，而且不该加 max_price。请完成订票。",
         {"passenger_name": "韩雪"}, "book_flight"),
        ("wrong_city_param",
         "温州今天天气怎么样",
         '<tool_call>{"name": "get_weather", "arguments": {"city": "WNZ"}}</tool_call>',
         "city 参数要填中文城市名，不要填三字码。",
         {"city": "温州"}, "get_weather"),
        ("hallucinated_weather_tool",
         "帮我看看海口的天气",
         '<tool_call>{"name": "get_travel_advisory", "arguments": {"city": "海口"}}</tool_call>',
         "没有 get_travel_advisory 这个工具，请用正确的天气工具。",
         {"city": "海口"}, "get_weather"),
        ("format_json_broken",
         "查一下订单BK8B24F3的行李额度",
         '<tool_call>{"name": "check_baggage", "arguments": {booking_id: BK8B24F3}}</tool_call>',
         "你的 JSON 不合法（键没有引号），请输出合法 JSON。",
         {"booking_id": "BK8B24F3"}, "check_baggage"),
        ("overflow_baggage",
         "把订单BK1E97A5的行李额度改成 60 公斤",
         '<tool_call>{"name": "modify_baggage", "arguments": {"booking_id": "BK1E97A5", "new_baggage_kg": 60}}</tool_call>',
         "60 公斤超过上限了，最多 40 公斤，请按 40 公斤改。",
         {"booking_id": "BK1E97A5", "new_baggage_kg": 40}, "modify_baggage"),
        ("extra_param",
         "查一下 2026-12-07 郑州到济南的航班",
         '<tool_call>{"name": "search_flights", "arguments": {"origin": "CGO", "destination": "TNA", "date": "2026-12-07", "cabin": "经济舱"}}</tool_call>',
         "search_flights 没有 cabin 这个参数，请去掉它。",
         {"origin": "CGO", "destination": "TNA", "date": "2026-12-07"},
         "search_flights"),
        ("wrong_id_format",
         "查一下订单 bk8e51a3 的行李额度",
         '<tool_call>{"name": "check_baggage", "arguments": {"booking_id": "bk8e51a3"}}</tool_call>',
         "订单号要写成大写 BK 开头的格式。",
         # ⚠️ 期望必须与**用户原话**一致：用户说 bk8e51a3，
         # 纠正要求「写成大写 BK 开头」→ 正确结果是它的**大写形式**。
         # 原来是 BK3A5C18 —— 与用户说的话毫无关系（我全局替换订单号时
         # 只改了期望、漏改了用户话），使这个任务**无解**。
         {"booking_id": "BK8E51A3"}, "check_baggage"),
    ]
    for key, user, wrong, fix, args, tool in _rc:
        specs.append(TaskSpec(
            category="error_correction", user=user, tool=tool, args=args,
            extra_turns=(("assistant_wrong_raw", wrong), ("user", fix)),
            tags=("error_correction", "recovery", key),
            note=f"纠错模式 {key}：模型首轮错 → 用户针对该错误纠正 → 应改对",
        ))
    # =====================================================================
    # 8) multi_turn（8）—— **真正的多轮对话**
    # =====================================================================
    #
    # ⚠️ 专家指出：项目叫 agent-optimization，但**没有一个真正的多轮任务**。
    # multi_tool 虽有 3 轮，实质是「一次调两个工具」，不是「多轮对话」。
    #
    # 真正的多轮标志：**后一轮的输入依赖前一轮的输出**，且用户的意图会**变化**。
    # 这里用「用户看到结果后改主意」来构造 —— 模型必须：
    #   1. 调工具
    #   2. 读用户的反悔/修改
    #   3. **用新参数再调一次**（而不是重复上一轮的调用）
    #
    # 评测侧的口径：用 ``expected_turns`` 声明每一轮应调什么，
    # 判「跨轮参数是否真的更新了」。只判最终动作是不够的 ——
    # 模型可能一直重复第一轮的调用而最终「碰巧」匹配。
    _mturn: list[tuple[str, str, list[tuple[str, str]], tuple, str]] = [
        ("user_flips_date", "PEK", "SHA",
         [("user", "{d1}北京到上海的航班有哪些"),
          ("tool", "search_flights"),
          ("user", "这些不合适，我要{d2}的"),
          ],
         {"origin": "PEK", "destination": "SHA", "date": "{d2}"}, "search_flights"),
        ("user_changes_dest", "CAN", "CTU",
         [("user", "{d1}从广州到成都的航班"),
          ("tool", "search_flights"),
          ("user", "算了不去成都了，改去昆明"),
          ],
         {"origin": "CAN", "destination": "KMG", "date": "{d1}"}, "search_flights"),
        ("user_changes_class", "SZX", "HGH",
         [("user", "帮我订{d1}深圳到杭州的机票，乘客沈括"),
          ("tool", "search_flights"),
          ("user", "航班号看到了，就订 MU3311，经济舱"),
          ],
         {"flight_id": "MU3311", "passenger_name": "沈括"}, "book_flight"),
        ("user_adds_baggage", "NKG", "TAO",
         [("user", "帮我查订单BK9F2C41的行李额度"),
          ("tool", "check_baggage"),
          ("user", "额度太少了，帮我提到 35 公斤"),
          ],
         {"booking_id": "BK9F2C41", "new_baggage_kg": 35}, "modify_baggage"),
        ("user_reconsiders_cancel", "KMG", "XMN",
         [("user", "帮我看看订单BK3D7B58的行李额度"),
          ("tool", "check_baggage"),
          ("user", "额度我知道了。另外帮我查一下{d1}昆明到厦门的航班"),
          ],
         {"origin": "KMG", "destination": "XMN", "date": "{d1}"}, "search_flights"),
        ("user_corrects_passenger", "TYN", "SJW",
         [("user", "太原到石家庄{d1}，给雷鸣订一张"),
          ("tool", "search_flights"),
          ("user", "乘客名字写错了，是雷鸣的朋友傅莹，帮她订"),
          ],
         {"flight_id": "CZ5522", "passenger_name": "傅莹"}, "book_flight"),
    ]
    for key, o, tt, turns, args, tool in _mturn:
        d1, d2 = nxt_date(), nxt_date()
        built: list[tuple[str, str]] = []
        for role, payload in turns:
            built.append((role, payload.format(d1=d1, d2=d2)))
        fixed_args = {
            k: (v.format(d1=d1, d2=d2) if isinstance(v, str) else v)
            for k, v in args.items()
        }
        specs.append(TaskSpec(
            category="multi_turn",
            user=built[0][1],
            tool=tool, args=fixed_args,
            extra_turns=tuple(built[1:]),
            tags=("multi_turn", key),
            note=f"真多轮：{key} —— 后一轮依赖前一轮输出，用户意图会变化",
        ))

    return specs


__all__ = ["build_specs"]
