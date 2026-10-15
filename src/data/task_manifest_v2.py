"""v2 高质量任务集：**把「高质量」变成可检查的约束**，而不是靠感觉。

## 为什么要重建

v1（当前 100 任务）经实测有五处硬伤（详见 ``verify_task_set`` 的输出）：

1. **实体复用严重**：订单号池只有 8 个，``BK7F2A9C`` 出现 7 次；
   机场码 ``SHA`` 出现 13 次；人名只有 6 个。
2. **日期全是过去**：全部落在 ``2025-07/08``，而当前已是 2026-10。
3. **``multi_tool`` 不是多工具**：9 个任务的 ``expected.tool`` 全部是
   ``search_flights`` —— 名字与实际测的东西不符，观测里的 ``get_weather``
   从来不参与判定。
4. **``seen``/``unseen`` 骨架共享 26%** —— 两组测的是同一批任务类型的
   不同说法，「泛化」这个词不成立。
5. **骨架/任务比极低**：``multi_tool`` 0.11、``weather`` 0.25、``baggage`` 0.40。
   即「53 个任务」可能只有 8 种真正的任务。

## 设计原则

* **一实体一任务**：订单号、乘客名在全库唯一；机场码复用上限 3 次。
  这样「模型答对了」不是因为它记住了 ``BK7F2A9C``。
* **日期全部在未来**（2026-11 / 2026-12），相对日期按 ``BASE_DATE`` 换算。
* **``multi_tool`` 真的要求多工具**：用 ``expected.also_required`` 声明
  必须**另外**调用的工具，评测侧据此判「是否真的调了多个」。
* **骨架唯一**：每类任务的句式骨架不重复，``skeleton_ratio`` 作为硬指标。
* **``unseen`` 真的没出现过**：按骨架分配给 seen/unseen，保证两组骨架无交集。

## 本模块只负责「定义与校验」

任务清单在 ``TASKS_V2`` 里**显式列出**（不再用 ``idx % len`` 模板相乘）——
因为模板相乘正是 v1 退化成「30 个任务换 10 种说法」的原因。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

#: 所有相对日期都从这个基准日换算。选 2026-10-06 是为了让
#: 「明天 / 下周一 / 三天后」等表达落在 **2026-11 ~ 2026-12** 的未来区间。
BASE_DATE_V2 = date(2026, 10, 6)

#: 未来日期池（2026-11 与 2026-12）。daily 粒度，避免复用。
_FUTURE_DATES: list[str] = [
    (BASE_DATE_V2 + timedelta(days=d)).isoformat()
    for d in range(30, 90)
]


# ---------------------------------------------------------------------------
# 实体池
# ---------------------------------------------------------------------------

#: 城市 → 三字码。
#:
#: ⚠️ **不要在这里独立维护一份表** —— 早期版本在 ``tool_catalog`` 与这里
#: 各存了一份（20 城 vs 60 城），于是训练语料（读 tool_catalog）
#: 与评测任务集（读本模块）**静默漂移**：
#: v3 需要 52 个机场码，而训练语料只覆盖 20 个，**33 个模型从未见过**，
#: `flight_search` 因此必然编造三字码（实测只有 59.1%）。
#:
#: 现在统一从 ``src.env.tool_catalog`` 取 —— 那里是运行时工具定义的来源，
#: 也应当是机场码的唯一来源。
from src.env.tool_catalog import AIRPORT_CODES as AIRPORT_CODES_V2  # noqa: E402

#: 乘客名池（**40 个**，每个任务用不同的人）
PASSENGERS_V2: list[str] = [
    "张伟", "李娜", "王芳", "刘洋", "陈静", "赵磊", "孙悦", "周杰",
    "吴敏", "郑凯", "冯雪", "何涛", "许静", "邓超", "曾毅", "彭飞",
    "苏晴", "蒋磊", "韩雪", "唐宇", "曹阳", "袁莉", "谢楠", "雷鸣",
    "傅莹", "沈括", "范冰", "毕昇", "邢昭", "章邯", "陆游", "荣格",
    "阎肃", "巫哲", "宁浩", "崔健", "晏殊", "晁盖", "莫言", "程颐",
]

#: 订单号池（**60 个**，确定性生成，避免与 v1 的 8 个重合）
BOOKING_IDS_V2: list[str] = [
    "BK%02X%02X%02X" % (((i * 37 + 11) % 256), ((i * 91 + 23) % 256),
                        ((i * 53 + 7) % 256))
    for i in range(60)
]
# v1 用过的那 8 个，v2 一律避开
_V1_BOOKING_IDS = {"BK7F2A9C", "BK3D8E1B", "BK9A4C6F", "BK2B7D5E",
                   "BK8C1F3A", "BK5E9B2D", "BK1A6C8F", "BK4F3D7B"}
BOOKING_IDS_V2 = [b for b in BOOKING_IDS_V2 if b not in _V1_BOOKING_IDS]

#: 取消原因（每个任务用不同的说法）
CANCEL_REASONS_V2: list[str] = [
    "行程变更", "身体不适", "会议取消", "找到更合适的航班", "天气原因",
    "临时有事去不了", "个人原因", "公司临时安排出差", "想改到其他日期",
    "家里有急事", "签证没办下来", "朋友改期了", "预算超了", "想换直飞",
    "目的地改了他处", "需要提前返回", "单位调整了假期", "同行人不去了",
    "买错日期了", "想改签别家航司",
]


# ---------------------------------------------------------------------------
# 任务定义（显式列出每个任务，不再用模板相乘）
# ---------------------------------------------------------------------------
@dataclass
class TaskSpec:
    """一个任务的**显式**定义。

    显式列出的原因：v1 用 ``idx % len(templates)`` 做笛卡尔积，结果
    「400 个任务」实际只有 208 种骨架、8 种签名。显式定义可以保证
    每个任务在实体与句式上都不重复。
    """

    category: str
    user: str
    tool: str | None
    args: dict[str, Any]
    #: 该任务还**必须**额外调用的工具（用于 multi_tool）
    also_required: tuple[str, ...] = ()
    #: 额外轮次（role, content 或 observation）
    extra_turns: tuple[tuple[str, str], ...] = ()
    tags: tuple[str, ...] = ()
    note: str = ""


#: 骨架化：把实体抽象掉，用于检查「是否只是换了说法」
def skeleton(text: str) -> str:
    """把实体替换成占位符，得到「任务骨架」。

    >>> skeleton("帮我查2026-11-05北京到上海的航班")
    '帮我查<DATE><CITY>到<CITY>的航班'
    >>> skeleton("取消BK7F2A9C")
    '取消<BID>'
    """
    s = text
    s = re.sub(r"BK[0-9A-F]{6}", "<BID>", s)
    s = re.sub(r"\d{4}-\d{2}-\d{2}", "<DATE>", s)
    s = re.sub(r"MU\d+", "<FLIGHT>", s)
    s = re.sub(r"\d+\s*(公斤|kg)", "<KG>", s)
    for city in sorted(AIRPORT_CODES_V2, key=len, reverse=True):
        s = s.replace(city, "<CITY>")
    for nm in sorted(PASSENGERS_V2, key=len, reverse=True):
        s = s.replace(nm, "<NAME>")
    s = re.sub(r"\d+", "<N>", s)
    return s


def signature(tool: str | None, args: dict[str, Any]) -> tuple:
    """(工具, 参数名集合) —— 任务的信息量上限。

    >>> signature("search_flights", {"origin": "PEK", "destination": "SHA", "date": "2026-11-01"})
    ('search_flights', ('date', 'destination', 'origin'))
    >>> signature(None, {})
    (None, ())
    """
    return (tool, tuple(sorted((args or {}).keys())))


def verify_task_set(specs: list[TaskSpec]) -> dict[str, Any]:
    """对任务集执行全部硬性检查，返回指标与违规清单。

    这是本模块存在的**核心理由**：把「高质量」变成可执行的断言，
    而不是每轮靠人眼看。v1 的五个缺陷都能被这里的检查抓到。
    """
    report: dict[str, Any] = {"n": len(specs), "violations": []}
    v = report["violations"]

    # ---- 1) 实体唯一性 ----
    bks = [m for s in specs for m in re.findall(r"BK[0-9A-F]{6}", s.user)]
    dup_bk = {k: c for k, c in
              __import__("collections").Counter(bks).items() if c > 1}
    if dup_bk:
        v.append(f"订单号复用: {dup_bk}")

    names = [nm for s in specs for nm in PASSENGERS_V2 if nm in s.user]
    dup_nm = {k: c for k, c in
              __import__("collections").Counter(names).items() if c > 1}
    if dup_nm:
        v.append(f"乘客名复用: {dup_nm}")

    cities = [c for s in specs for c in AIRPORT_CODES_V2 if c in s.user]
    city_cnt = __import__("collections").Counter(cities)
    # ⚠️ 判据的边界：**同一类别内**不得复用同一个城市（那才是「换皮」），
    # 但跨类别复用是合理的 —— 「北京」同时出现在 flight_search 与
    # booking_flow 里不构成缺陷（真实世界城市就是有限的）。
    # 早期我把阈值设成「全库 ≤3」，结果把一个合理现象报成违规。
    same_cat: dict[tuple[str, str], int] = __import__("collections").Counter(
        (s.category, c) for s in specs for c in AIRPORT_CODES_V2 if c in s.user
    )
    reuse_in_cat = {f"{k[0]}/{k[1]}": v for k, v in same_cat.items() if v > 1}
    if reuse_in_cat:
        v.append(f"同类别内城市复用: {reuse_in_cat}")
    hot = {k: c for k, c in city_cnt.items() if c > 6}
    if hot:
        v.append(f"城市全库复用超过 6 次: {hot}")

    # ---- 2) 日期必须都是未来 ----
    dates = [d for s in specs for d in re.findall(r"\d{4}-\d{2}-\d{2}", s.user)]
    past = [d for d in dates if d < BASE_DATE_V2.isoformat()]
    if past:
        v.append(f"含过去日期: {sorted(set(past))[:5]}")

    # ---- 3) 骨架唯一性 ----
    #
    # ⚠️ 判据的边界：**同一类别内**不得有重复骨架（那才是「换皮」）。
    # **跨类别**的骨架重复是允许的、有时还是刻意的 —— 例如 recovery 的
    # ``param_placeholder`` 模式就要求与 baggage 的 check 任务**同一句话**，
    # 只在订单号上做手脚。早期把「全库骨架唯一」当硬约束，会把这种
    # 刻意设计报成违规。
    sk = [skeleton(s.user) for s in specs]
    per_cat_sk = __import__("collections").defaultdict(list)
    for s, k in zip(specs, sk):
        per_cat_sk[s.category].append(k)
    dup_in_cat = {}
    for c, ks in per_cat_sk.items():
        dd = {k: n for k, n in __import__("collections").Counter(ks).items() if n > 1}
        if dd:
            dup_in_cat[c] = dd
    if dup_in_cat:
        v.append(f"同类别内骨架重复: {dup_in_cat}")

    # 跨类别重复只记录、不算违规（便于人工判断是否刻意）
    global_cnt = __import__("collections").Counter(sk)
    cross = {k: n for k, n in global_cnt.items() if n > 1}
    report["cross_category_skeleton_reuse"] = cross

    # ---- 4) multi_tool 必须真的要求多工具 ----
    mt = [s for s in specs if s.category == "multi_tool"]
    bad_mt = [s.user[:30] for s in mt if not s.also_required]
    if bad_mt:
        v.append(f"multi_tool 未声明额外工具: {bad_mt[:3]}")

    # ---- 5) **跨类别实体不得重合** ----
    #
    # ⚠️ 这是验证框架早期的**盲区**：我只检查了「同类别内骨架唯一」，
    # 完全没检查跨类别的实体（航线 / 订单号 / 城市 / 人名）是否重合。
    # 专家正是从这个盲区里找出问题：
    #   * multi_tool 的航线与 flight_search 重复（实测 10 个里 3 个）
    #   * recovery 的 (工具, 参数) 与其他类完全重合（实测 15 个里 2 个）
    #
    # 为什么这很重要：**若两个类别用同一批实体，模型在一个类别上学会的
    # 映射可以直接套到另一个类别**，那个类别就不再测它的名义能力。
    # 例如 multi_tool 若复用 flight_search 的「北京→上海」，
    # 它测的就只是「多调一个天气工具」，而航线理解那部分已经被训过了。
    #
    # 判据：
    #   * **航线**（origin,destination 对）：flight_search / booking_flow /
    #     multi_tool / multi_turn 之间不得重合
    #   * **订单号**：baggage / cancellation / recovery 之间不得重合
    #   * **城市**（天气用）：weather_advisory 与其他类别不得重合
    #   * recovery 例外：它天然建立在其他域的能力之上（要纠错就得先有个
    #     「错」可纠），所以只要求它与其他类的**(工具, 参数)签名不完全相同**，
    #     即它必须是「会先出错再被纠正」的独立场景。
    def _routes(cat: str) -> set[tuple]:
        out = set()
        for s in specs:
            if s.category != cat:
                continue
            a = s.args or {}
            if "origin" in a and "destination" in a:
                out.add((a["origin"], a["destination"]))
        return out

    route_cats = ["flight_search", "booking_flow", "multi_tool", "multi_turn"]
    for i, c1 in enumerate(route_cats):
        for c2 in route_cats[i + 1:]:
            shared = _routes(c1) & _routes(c2)
            if shared:
                v.append(f"航线跨类别重合 {c1} ∩ {c2} = {sorted(shared)}")

    def _bids(cat: str) -> set[str]:
        return {s.args["booking_id"] for s in specs
                if s.category == cat and (s.args or {}).get("booking_id")}

    bid_cats = ["baggage", "cancellation", "recovery"]
    for i, c1 in enumerate(bid_cats):
        for c2 in bid_cats[i + 1:]:
            shared = _bids(c1) & _bids(c2)
            if shared:
                v.append(f"订单号跨类别重合 {c1} ∩ {c2} = {sorted(shared)}")

    # ⚠️ 这里我一度检查「天气城市是否与其他类别的城市重合」，并把它报成违规。
    # 那是**判据错误**：防的是「同一组城市对（航线）」跨类别复用，
    # 不是「同一个城市名」出现两次。理由：
    #   * 40 个城市可组合出 1560 组有序城市对，各类只需 14/10/6 组 ——
    #     城市名的重合是自然且不可避免的；
    #   * 真正会让模型「套用」的是**同一组起降机场**（同一条航线），
    #     因为那对应同一个三字码映射记忆；
    #   * 天气用的参数是城市名、航班用的是三字码，两者本就不是同一个映射。
    # 所以城市名重合不算违规；只检查航线（已在上面的 route 检查里）。

    # recovery 必须与其他类别在「同一工具 + 同一参数值」上完全区分。
    #
    # ⚠️ 排除 ``tool is None`` 的情形：那表示「应先追问」，
    # 而追问类任务天然是多类别的（取消要追问、查额度也要追问），
    # 它们的 ``(None, ())`` 签名必然相同 —— 用签名判定会把它们全部误报。
    # 真正该判的是：**有具体工具调用的** recovery 任务是否与其他类同形。
    non_rc_sig = {
        (s.tool, tuple(sorted((s.args or {}).items())))
        for s in specs
        if s.category != "recovery" and s.tool is not None
    }
    rc_dup = [
        s.user[:34] for s in specs
        if s.category == "recovery"
        and s.tool is not None
        and (s.tool, tuple(sorted((s.args or {}).items()))) in non_rc_sig
    ]
    if rc_dup:
        v.append(f"recovery 与其他类实体完全重合（等于换皮）: {rc_dup}")

    # ---- 6) 期望调用不得「空参数」----
    #
    # ⚠️ 这是一条**会污染准确率**的检查：``expected.args == {}`` 时
    # 任何工具调用都会匹配成功（宽松口径下 missing/extra 都为空），
    # 于是该任务**永远判成功**。实测 v2 有 2 个这样的任务
    # （taskv2_0086 / 0092，都是 recovery 里 search_flights 的 args 没填）。
    # 它不会报错，只会让准确率虚高。
    empty_args = [
        s.user[:36] for s in specs
        if s.tool is not None and not (s.args or {})
    ]
    if empty_args:
        v.append(f"期望调用为空参数（将永远判成功）: {empty_args}")

    # ---- 6) 纠错话术不得全部相同 ----
    #
    # ⚠️ recovery 类要求「用户纠正」。若所有任务的纠正话术都是同一句
    # （实测 15 个任务全是「请修正」），那它退化成了一句固定 prompt，
    # 测不出「模型能否理解具体的纠错内容」。
    corr = [s.user for s in specs if s.category == "recovery"]
    if corr:
        fixes = []
        for s in specs:
            for role, payload in s.extra_turns:
                if role == "user" and s.category == "recovery":
                    fixes.append(payload)
        if fixes and len(set(fixes)) == 1:
            v.append(f"recovery 的纠错话术全部相同（{fixes[0]!r}），"
                     "退化为固定 prompt")

    # ---- 7) 指标 ----
    import collections as _c

    by_cat = _c.defaultdict(list)
    for s, k in zip(specs, sk):
        by_cat[s.category].append(k)
    report["skeleton_total"] = len(set(sk))
    report["skeleton_ratio"] = round(len(set(sk)) / max(1, len(specs)), 3)
    report["by_category"] = {
        c: {
            "n": len(ks),
            "skeletons": len(set(ks)),
            "ratio": round(len(set(ks)) / max(1, len(ks)), 2),
            "signatures": len({signature(s.tool, s.args)
                               for s in specs if s.category == c}),
        }
        for c, ks in sorted(by_cat.items())
    }
    return report


__all__ = [
    "compile_specs", "load_tasks_v3", "DIFFICULTY_V2",
    "BASE_DATE_V2", "AIRPORT_CODES_V2", "PASSENGERS_V2", "BOOKING_IDS_V2",
    "CANCEL_REASONS_V2", "TaskSpec", "skeleton", "signature", "verify_task_set",
]

# ---------------------------------------------------------------------------
# 把 TaskSpec 编译成 Task
# ---------------------------------------------------------------------------
def compile_specs(specs: list[TaskSpec], *, seed: int = 42) -> list[Any]:
    """把显式定义编译成 ``Task`` 对象。

    要点：
    * ``seen`` / ``unseen`` 按**骨架**分配（不是按序号），保证两组骨架无交集。
      这修正了 v1 的缺陷：v1 按 ``counter % 3`` 划分，实测两组共享 26% 的骨架，
      于是「unseen 泛化指标」实际测的是同一批任务类型的另一种说法。
    * ``multi_tool`` 与 ``booking_flow`` 会补上真实的 observation 轮，
      让任务真的需要多步完成。

    >>> specs = [TaskSpec("weather_advisory", "北京今天天气怎么样？",
    ...                   "get_weather", {"city": "北京"})]
    >>> ts = compile_specs(specs)
    >>> len(ts), ts[0].category, ts[0].expected.tool
    (1, 'weather_advisory', 'get_weather')
    >>> ts[0].visibility in ("seen", "unseen")
    True
    """
    import hashlib

    from src.data.task_manifest import ExpectedCall, Task, Turn

    # ---- 按骨架分配 seen/unseen，保证两组骨架不重叠 ----
    skels = [skeleton(s.user) for s in specs]
    uniq = sorted(set(skels))
    # 用稳定哈希决定归属：同一骨架永远落在同一组（与 specs 顺序无关）
    def _is_unseen(k: str) -> bool:
        h = int(hashlib.sha256(k.encode("utf-8")).hexdigest()[:8], 16)
        return h % 3 == 0

    skel_vis = {k: ("unseen" if _is_unseen(k) else "seen") for k in uniq}

    out: list[Task] = []
    for i, sp in enumerate(specs, start=1):
        turns = [Turn("user", sp.user)]

        # recovery：补「模型首轮答错 + 用户纠错」
        for role, payload in sp.extra_turns:
            if role == "assistant_wrong_raw":
                # 直接把「模型首轮的错误回复」写进任务定义。
                # 比按 key 生成更可靠：错误内容与用户的纠错话术必须**对应**，
                # 而 key 生成无法保证这一点（实测初版 15 条纠错话术全是
                # 「请修正」，与错误内容毫无对应关系）。
                turns.append(Turn("assistant", payload))
            elif role == "assistant_wrong":
                turns.append(Turn("assistant", _wrong_reply(payload, sp)))
            elif role == "assistant_confirm":
                bid = (sp.args or {}).get("booking_id", "")
                reason = (sp.args or {}).get("reason", "")
                turns.append(Turn(
                    "assistant",
                    f"好的，我确认一下：您要取消订单 {bid}，原因是「{reason}」，对吗？",
                ))
            elif role == "tool":
                turns.append(Turn("tool", "", observation=_observation_for(payload, sp)))
            else:
                turns.append(Turn(role, payload))

        out.append(Task(
            task_id=f"taskv2_{i:04d}",
            category=sp.category,
            difficulty=DIFFICULTY_V2.get(sp.category, "medium"),
            visibility=skel_vis[skeleton(sp.user)],
            turns=turns,
            target_turn=len(turns),
            target_turn_kind="initial",
            expected=ExpectedCall(
                tool=sp.tool, args=dict(sp.args or {}),
                args_subset=dict(sp.args or {}) or None,
                note=sp.note,
            ),
            tags=tuple(sp.tags) + tuple(sp.also_required),
            mistake_hint=sp.tags[0] if sp.tags else "",
        ))
    return out


#: v2 的难度标注（按类别，与 v1 的 CATEGORY_DIFFICULTY 口径一致）
DIFFICULTY_V2: dict[str, str] = {
    "weather_advisory": "easy",
    "baggage": "easy",
    "flight_search": "hard",
    "booking_flow": "hard",
    "cancellation": "hard",
    "multi_tool": "hard",
    "multi_turn": "hard",
    "error_correction": "medium",
    "recovery": "medium",
}


def _observation_for(tool: str, sp: TaskSpec) -> dict[str, Any]:
    """为多步任务构造真实的工具返回观测。"""
    a = sp.args or {}
    if tool == "get_weather":
        city = a.get("city", "")
        return {"ok": True, "tool": "get_weather",
                "result": {"city": city, "condition": "小雨", "temp_c": 19,
                           "flight_impact": "可能出现延误"}}
    if tool == "search_flights":
        fid = a.get("flight_id") or "CA7788"
        return {"ok": True, "tool": "search_flights",
                "result": {"count": 2, "flights": [
                    {"flight_id": fid, "price_cny": 1180, "depart_time": "07:45"},
                    {"flight_id": "CA9900", "price_cny": 1560, "depart_time": "18:20"},
                ]}}
    if tool == "check_baggage":
        return {"ok": True, "tool": "check_baggage",
                "result": {"booking_id": a.get("booking_id", ""), "free_baggage_kg": 20}}
    return {"ok": True, "tool": tool, "result": {}}


def _wrong_reply(key: str, sp: TaskSpec) -> str:
    """为 recovery 任务生成「模型首轮答错」的内容（按纠错模式）。"""
    a = sp.args or {}
    if key.startswith("format_ignore"):
        return "好的，我来帮您查询一下，请稍等。"
    if key.startswith("hallucinated_tool"):
        return ('<tool_call>{"name": "get_order_details", "arguments": '
                '{"order_id": "%s"}}</tool_call>' % a.get("booking_id", ""))
    if key.startswith("hallucinated_weather_tool"):
        return ('<tool_call>{"name": "get_travel_advisory", "arguments": '
                '{"city": "%s"}}</tool_call>' % a.get("city", ""))
    if key.startswith("wrong_tool"):
        return ('<tool_call>{"name": "check_baggage", "arguments": '
                '{"booking_id": "%s"}}</tool_call>' % a.get("booking_id", ""))
    if key.startswith("param_placeholder"):
        return ('<tool_call>{"name": "check_baggage", "arguments": '
                '{"booking_id": "previous_booking"}}</tool_call>')
    if key.startswith("city_code"):
        return ('<tool_call>{"name": "search_flights", "arguments": '
                '{"origin": "北京", "destination": "上海", "date": "明天"}}</tool_call>')
    if key.startswith("repeated_same_error"):
        return ('<tool_call>{"name": "book_flight", "arguments": '
                '{"flight_id": "CA0000", "passenger_name": "%s"}}</tool_call>'
                % a.get("passenger_name", ""))
    if key.startswith("missing_reason"):
        return ('<tool_call>{"name": "cancel_booking", "arguments": '
                '{"booking_id": "%s", "reason": "用户要求"}}</tool_call>'
                % a.get("booking_id", ""))
    if key.startswith("wrong_date"):
        return ('<tool_call>{"name": "search_flights", "arguments": '
                '{"origin": "%s", "destination": "%s", "date": "后天"}}</tool_call>'
                % (a.get("origin", ""), a.get("destination", "")))
    if key.startswith("overbook"):
        return ('<tool_call>{"name": "search_flights", "arguments": '
                '{"origin": "%s", "destination": "%s", "date": "%s", '
                '"max_price": 1000}}</tool_call>'
                % (a.get("origin", ""), a.get("destination", ""), a.get("date", "")))
    if key.startswith("wrong_city_param"):
        return ('<tool_call>{"name": "get_weather", "arguments": '
                '{"city": "WNZ"}}</tool_call>')
    if key.startswith("format_json_broken"):
        return ('<tool_call>{"name": "check_baggage", "arguments": '
                '{booking_id: %s}}</tool_call>' % a.get("booking_id", ""))
    return "好的，我来帮您处理。"


_V2_TASK_CACHE: list[Any] | None = None


def load_tasks_v3(seed: int = 42, use_cache: bool = True) -> list[Any]:
    """加载 v2 高质量 100 任务集（显式定义、约束校验通过）。

    >>> ts = load_tasks_v3()
    >>> # 106 个：为让「应先追问」样本达 14 个，放弃了整 100 的形式要求
    >>> len(ts)
    106
    >>> len({t.turns[0].content for t in ts}) >= 105
    True
    >>> len({t.visibility for t in ts})
    2
    """
    global _V2_TASK_CACHE
    if _V2_TASK_CACHE is None or not use_cache:
        from src.data.task_set_v2_specs import build_specs

        _V2_TASK_CACHE = compile_specs(build_specs(), seed=seed)
    return _V2_TASK_CACHE

def verify_visibility_is_meaningful(tasks: list[Any] | None = None) -> dict:
    """检查 ``visibility``（seen/unseen）标签**是否真的有含义**。

    ## 为什么需要它

    ``visibility`` 的赋值**从不读取训练数据**：

    * v1：``"unseen" if counter % 3 == 0``（按序号轮转）
    * v2/v3：``"unseen" if sha256(skeleton) % 3 == 0``（按骨架哈希）

    它假设「合成训练语料只覆盖评测分布的一部分」。而我为了修
    「33 个机场码模型从未见过」，把合成语料扩到**覆盖全部 60 个机场码** ——
    于是评测里每个任务的签名（工具 + 参数名）都在训练里出现过。

    实测后果（这是一条**会让泛化结论作废**的缺陷）：

        现有 seen   里「签名训练见过」: 67/68
        现有 unseen 里「签名训练见过」: 38/38   ← unseen 也 100% 见过

    **「seen vs unseen 的差值是泛化差距」这个说法因此不成立** ——
    那个差值测的是哈希分组的噪声。

    Returns:
        诊断报告。``meaningful=False`` 表示该标签不可用于泛化结论。

    >>> rep = verify_visibility_is_meaningful()
    >>> rep["total"] > 0
    True
    >>> "unseen_all_covered" in rep
    True
    """
    from src.data.recovery_corpus import build_training_tasks

    ts = list(tasks) if tasks is not None else load_tasks_v3()
    syn = build_training_tasks()
    syn_sig = {
        (x.expected.tool, tuple(sorted((x.expected.args or {}).keys())))
        for x in syn
    }

    def _covered(x: Any) -> bool:
        return (x.expected.tool,
                tuple(sorted((x.expected.args or {}).keys()))) in syn_sig

    seen = [x for x in ts if x.visibility == "seen"]
    unseen = [x for x in ts if x.visibility == "unseen"]
    seen_cov = sum(1 for x in seen if _covered(x))
    unseen_cov = sum(1 for x in unseen if _covered(x))

    # 「有含义」的最低要求：unseen 组里必须有相当一部分是训练**没**覆盖的
    meaningful = bool(unseen) and unseen_cov < len(unseen)
    return {
        "total": len(ts),
        "seen": len(seen),
        "unseen": len(unseen),
        "seen_signature_covered": seen_cov,
        "unseen_signature_covered": unseen_cov,
        "seen_coverage_rate": seen_cov / len(seen) if seen else 0.0,
        "unseen_coverage_rate": unseen_cov / len(unseen) if unseen else 0.0,
        "unseen_all_covered": bool(unseen) and unseen_cov == len(unseen),
        "meaningful": meaningful,
    }


def report_visibility_warning() -> str:
    """返回一句可直接打印的警告；标签有效时返回空串。

    >>> isinstance(report_visibility_warning(), str)
    True
    """
    rep = verify_visibility_is_meaningful()
    if rep["meaningful"]:
        return ""
    return (
        "⚠️ **visibility（seen/unseen）标签在当前配置下没有含义**：\n"
        f"   seen   组签名被训练覆盖: {rep['seen_signature_covered']}/{rep['seen']}"
        f" ({rep['seen_coverage_rate']:.0%})\n"
        f"   unseen 组签名被训练覆盖: {rep['unseen_signature_covered']}/{rep['unseen']}"
        f" ({rep['unseen_coverage_rate']:.0%})\n"
        "   原因：合成训练语料**故意覆盖了整个评测分布**（为修「机场码未见」）。\n"
        "   **不要**把 seen 与 unseen 的差值解释为泛化差距 —— 那是哈希分组的噪声。\n"
        "   应当改用「重合层级」分层报告（见 scripts/eval/audit_eval_overlap.py）。"
    )
