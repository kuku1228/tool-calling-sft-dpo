"""偏好对构造器：把诊断结果变成 DPO 能吃的 (prompt, chosen, rejected) 三元组。

四类偏好对的设计依据
--------------------
每一类都针对 ``docs/diagnosis/baseline_diagnosis.md`` 里的一类高发错误，
而不是「随便造点数据」。这是本项目与「拿现成 preference dataset 直接训」
的核心区别。

======================  ==========================  ============================
类型                     针对的错误                    chosen ↔ rejected 的对比点
======================  ==========================  ============================
``format_constraint``   FORMAT_ERROR（占比最高）      合法 ``<tool_call>`` ↔ 纯文本承诺
``error_recovery``      NO_RECOVERY / 纠错轮失败      **纠错后正确修正** ↔ 原样重复错误
``tool_selection``      HALLUCINATION / WRONG_TOOL    白名单工具 ↔ 貌似合理的假工具
``param_constraint``    PARAM_ERROR                  真实实体 ↔ 占位符 / 错格式
======================  ==========================  ============================

为什么 ``error_recovery`` 是原创点
----------------------------------
SFT 的 chosen 里只会有「正确答案」，模型学不到「先错后改」这个模式。
而真实的 agent 交互里，用户纠错是高频事件。所以恢复类偏好对的
prompt 里**显式包含**「模型上一轮的错误回复 + 用户的纠错话术」，
chosen 是修正后的回复，rejected 是原样重复 —— 这是普通 instruction
tuning 数据里拿不到的信号。

硬负样本（hard negative）与简单负样本
-------------------------------------
``rejected`` 分两档：

- **简单负样本**：明显的坏输出（纯文本承诺、坏 JSON）。
  模型很容易区分 → 梯度贡献小但稳定。
- **硬负样本**：看起来完全合理、格式也合法，只是**实体错**。
  例如把 ``previous_booking`` 当订单号、把三字码写成中文城市名。
  这是 DPO 的主要学习信号。

``hard_negative_ratio`` 控制两者配比（默认 0.35）。全用硬负样本
会让训练不稳（正负样本差异过小），全用简单负样本则学不到实质能力。

质量门禁
--------
每条偏好对生成后要过 ``_passes_quality_gate``：

1. chosen 必须是 canonical 格式；
2. chosen 与 rejected **语义必须真的不同**（用 ``equivalent`` 判定），
   否则会教模型去区分两个其实等价的东西 —— 这是偏好数据最常见的污染源；
3. chosen 的参数必须通过工具 schema 校验；
4. prompt 不能为空。

未通过的数据被丢弃并计数，最终写入 ``build_stats.json``，
让「构造了多少条」和「用了多少条」都是可审计的。
"""

from __future__ import annotations

import copy
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.config import CFG, PreferenceCfg
from src.data.task_manifest import Task, load_tasks, recovery_tasks
from src.env.tool_catalog import (
    AIRPORT_CODES,
    CITY_NAMES,
    PLAUSIBLE_FAKE_TOOLS,
    TOOL_REGISTRY,
    looks_like_placeholder,
    plausible_fakes_for,
    validate_arguments,
)
from src.eval.equivalence import equivalent, match_arguments
from src.eval.tool_call_parser import canonical_call, extract_tool_calls, is_canonical_format

# ============================================================================
# 常量
# ============================================================================

PAIR_FORMAT = "format_constraint"
PAIR_RECOVERY = "error_recovery"
PAIR_TOOL_SELECTION = "tool_selection"
PAIR_PARAM = "param_constraint"

#: 「该追问 vs 硬调工具」边界对。
#:
#: ⚠️ **这一类是真实基线数据逼出来的，不是预先设计的。**
#:
#: 原方案的四类偏好对（格式/恢复/工具选择/参数）全部假设失败来自
#: 「调错工具或参数写错」。真实基线（Qwen2.5-1.5B，400 样本）显示
#: 失败集中在「该不该调工具」这个决策上，而 SFT 之后剩下的**唯一**
#: 失败类型是：
#:
#:     信息不足时应先追问，却直接调用了工具    20 条（占失败 100%）
#:
#: 形态很具体：用户说「我不想要 BK9A4C6F 这个订单了」——
#: ``cancel_booking`` 需要 ``reason``，但用户没说原因。
#: 正确行为是追问「请问您取消的原因是什么？」，
#: 而模型会硬填 ``reason="用户要求"`` 然后直接调用写操作。
#:
#: SFT 修不了这个：它的数据全是「正确调用」的正例，
#: 没有任何「此处不该调用」的对比信号，于是把模型推成了「无脑调工具」。
#: 这正是需要 DPO 对比信号的地方。
PAIR_CLARIFY = "clarify_boundary"

PAIR_TYPES: tuple[str, ...] = (
    PAIR_FORMAT,
    PAIR_RECOVERY,
    PAIR_TOOL_SELECTION,
    PAIR_PARAM,
    PAIR_CLARIFY,
)

PAIR_TYPE_LABELS_ZH: dict[str, str] = {
    PAIR_FORMAT: "格式约束对",
    PAIR_RECOVERY: "错误恢复对",
    PAIR_TOOL_SELECTION: "工具选择对",
    PAIR_PARAM: "参数约束对",
    PAIR_CLARIFY: "该追问边界对",
}

#: 用户纠错话术的**唯一定义处**在 ``src/data/recovery_corpus.py`` 的
#: ``CORRECTION_PHRASING``（按错误类型分为 11 组，如 ``format`` /
#: ``param_airport_code`` / ``param_date`` …）。
#:
#: 本文件**不再**定义第二套话术模板。历史上这里曾有一份
#: ``CORRECTION_TEMPLATES``（5 组，键名用错误类别常量
#: ``FORMAT_ERROR`` 等），与 recovery_corpus 的那套**键名不一致且无人引用** ——
#: 改它的人会以为改了训练数据，实际毫无影响。已经删除。
#:
#: 需要改纠错话术时请改 ``recovery_corpus.CORRECTION_PHRASING``，
#: 并遵守那里的约束：**话术里提到的参数必须真的出现在该场景首轮错误的
#: 调用参数里**（由下面的 ``_recovery_coherence`` 门禁强制检查）。

#: 「简单负样本」池：明显的坏输出
_SIMPLE_NEGATIVES: tuple[str, ...] = (
    "好的，我来帮您查询一下，请稍等。",
    "正在为您查询，马上就好。",
    "我明白了，我先帮您确认一下。",
    "好的，我这就为您处理。",
    "稍等片刻，我为您查一下相关信息。",
)


#: 「信息不足该追问」的**合成场景**兜底池。
#:
#: 为什么需要兜底：任务集里 ``expected.tool is None`` 的任务只有 5 个
#: （全是取消订单场景）。如果只靠它们，这一类偏好对的多样性太差
#: （模型可能学到「看到『不想要了』就追问」这种表层模式，
#: 而不是「缺必填参数就该追问」的一般规则）。
#:
#: 这里的每个场景都遵循同一个结构：
#:   - 用户提供了**部分**信息（known）
#:   - 但**缺**某个必填参数（fabricated 里的键）
#:   - 正确行为 = 追问；错误行为 = 编造该参数后照常调用
#:
#: ``prompt`` 用项目的统一模板渲染，保证与评测分布一致。
_SYNTHETIC_CLARIFY: tuple[dict[str, Any], ...] = (
    {
        "user": "我不想要 BK9A4C6F 这个订单了",
        "tool": "cancel_booking",
        "known": {"booking_id": "BK9A4C6F"},
        "fabricated": {"reason": ["用户要求", "行程变更", "不需要了", "个人原因"]},
        "clarify": [
            "好的，为了帮您取消订单，请问您取消的原因是什么？",
            "请问您方便告诉我取消的原因吗？我需要填写后才能为您办理。",
            "了解，请问取消这笔订单的原因是什么？",
        ],
    },
    {
        "user": "把 BK3D8E1B 的行李额度改一下",
        "tool": "modify_baggage",
        "known": {"booking_id": "BK3D8E1B"},
        "fabricated": {"new_baggage_kg": ["20", "23", "30"]},
        "clarify": [
            "好的，请问您想把行李额度改成多少公斤？",
            "请问新的行李额度是多少公斤呢？",
            "好的，请告诉我您需要改成的公斤数。",
        ],
    },
    {
        "user": "帮我把 BK5F2C7A 这个订单取消掉，我临时有事",
        "tool": "cancel_booking",
        # 用户说了 reason（临时有事），但确认信息不完整时仍需明确化；
        # 这里用「缺 reason 的精确措辞」构造，避免与已知信息矛盾。
        "known": {"booking_id": "BK5F2C7A"},
        "fabricated": {"reason": ["用户要求", "个人原因"]},
        "clarify": [
            "好的，请确认一下取消原因是否为「临时有事」？我需要按实际原因提交。",
            "了解，我将按「临时有事」提交取消，可以吗？",
        ],
    },
    {
        "user": "我要退票",
        "tool": "cancel_booking",
        "known": {},
        "fabricated": {"booking_id": ["BK000000", "BK111111"], "reason": ["用户要求"]},
        "clarify": [
            "好的，请问您要退的是哪个订单？请提供订单号。",
            "请问需要退票的订单号是多少？",
            "请提供您的订单号，我好为您办理退票。",
        ],
    },
    {
        "user": "帮我改一下航班",
        "tool": "modify_baggage",
        "known": {},
        "fabricated": {"booking_id": ["BK000000"], "new_baggage_kg": ["20"]},
        "clarify": [
            "好的，请问您要修改哪个订单，具体想改成什么？",
            "请问需要修改的订单号以及您想改成的行李额度是多少？",
        ],
    },
)


def _normalize_loose(text: str) -> str:
    """极宽松的文本归一化，只用于「两侧是不是同一句话」的判断。"""
    from src.eval.equivalence import normalize_text

    return normalize_text(text)


# ============================================================================
# 数据结构
# ============================================================================


@dataclass
class PreferencePair:
    """一条偏好对。

    字段名与 ``datasets`` / TRL 的约定对齐（``prompt`` / ``chosen`` / ``rejected``），
    但**不依赖 datasets** —— 本模块只产出 jsonl，加载交给训练脚本，
    这样在没有装 datasets 的机器上也能构造数据。
    """

    prompt: str
    chosen: str
    rejected: str
    pair_type: str
    #: 产生这条数据的任务
    task_id: str = ""
    category: str = ""
    difficulty: str = ""
    #: 针对的错误类型（error_taxonomy 的类别常量或 NO_RECOVERY）
    target_error: str = ""
    #: hard | simple | recovery
    negative_kind: str = "simple"
    #: 数据用途：train 会用于训练，eval 会被留出
    split: str = "train"
    #: 工具 schema（部分训练配置需要把 tools 放进 prompt 之外单独传）
    tools: list[dict[str, Any]] = field(default_factory=list)
    #: 一句话说明这条数据的教学意图，便于抽样检查
    intent: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "pair_type": self.pair_type,
            "task_id": self.task_id,
            "category": self.category,
            "difficulty": self.difficulty,
            "target_error": self.target_error,
            "negative_kind": self.negative_kind,
            "split": self.split,
            "intent": self.intent,
        }
        if self.tools:
            d["tools"] = self.tools
        return d

    @property
    def key(self) -> str:
        """去重键。

        ⚠️ 必须把 ``rejected`` 也纳入键。只用 ``prompt + chosen`` 会让
        「同一个任务、同一个正确回复，但换了不同错误类型作为负样本」
        的多条对互相顶掉 —— 那会白扔掉大量有价值的数据
        （实测会丢掉 97% 的候选）。
        """
        return f"{self.pair_type}|{self.prompt}|{self.chosen}|{self.rejected}"


# ============================================================================
# Prompt 构造
# ============================================================================
# ⚠️ 系统提示词与渲染函数都从 src/models/prompt_builder.py 取，
# **不在本模块重复定义**。
#
# 训练与推理的 prompt 必须逐字一致，否则 DPO 学到的是
# 「在训练那版 prompt 下输出 tool_call」，换了 prompt 就不激活 ——
# 表现为训练 loss 下降但评测指标不动。这是偏好对齐最隐蔽的坑，
# 用「两份模板各自维护」的方式几乎必然踩到。

from src.models.prompt_builder import (  # noqa: E402
    SYSTEM_PROMPT,
    build_system_prompt,
    build_plain_prompt,
)

#: 向后兼容别名（早期版本叫 SYSTEM_PROMPT_TEMPLATE）
SYSTEM_PROMPT_TEMPLATE = SYSTEM_PROMPT


def render_prompt(task: Task, *, system_prompt: str | None = None, extra_turns: Sequence[Any] = ()) -> str:
    """把一个任务的对话前缀渲染成训练用的 prompt 文本。

    用纯文本而不是 chat template 的原因：构造偏好数据时不依赖
    tokenizer（可能是远程机器上才有的），训练脚本会在加载时
    再套 chat template。这样偏好数据的生成与模型无关、可复现。

    格式（与 Qwen/Hermes 风格对齐）::

        <|im_start|>system
        ...
        <|im_end|>
        <|im_start|>user
        ...
        <|im_end|>
        <|im_start|>assistant
    """
    sp = system_prompt if system_prompt is not None else build_system_prompt()
    parts: list[str] = [f"<|im_start|>system\n{sp}<|im_end|>"]
    for msg in task.to_messages():
        role = msg["role"]
        parts.append(f"<|im_start|>{role}\n{msg['content']}<|im_end|>")
    for msg in extra_turns:
        role = getattr(msg, "role", "user")
        content = getattr(msg, "content", str(msg))
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    # 结尾的 assistant 头，表示轮到模型生成
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def render_user_prompt(user_text: str, *, system_prompt: str | None = None) -> str:
    """把（系统提示 + 单轮用户消息）渲染成训练用 prompt。

    给**合成场景**用 —— 那些场景没有 ``Task`` 对象，但对 prompt 格式的
    要求与真实任务完全相同（否则又会引入训练/评测分布不一致）。

    >>> p = render_user_prompt("我要退票", system_prompt="SYS")
    >>> p.startswith("<|im_start|>system\\nSYS<|im_end|>")
    True
    >>> p.endswith("<|im_start|>assistant\\n")
    True
    >>> "<|im_start|>user\\n我要退票<|im_end|>" in p
    True
    """
    sp = system_prompt if system_prompt is not None else build_system_prompt()
    return (
        f"<|im_start|>system\n{sp}<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


#: 订单号形如 ``BK9A4C6F``（BK + 6 位十六进制）。
#:
#: ⚠️ **必须带 ``re.ASCII``**，否则 ``\b`` 在中文旁会失效。
#: 这是个不直观但影响很大的坑：
#:
#:     >>> import re
#:     >>> re.findall(r"\\bBK[0-9A-F]{6}\\b", "我不想要BK9A4C6F这个订单了")
#:     []
#:     >>> re.findall(r"\\bBK[0-9A-F]{6}\\b", "我不想要BK9A4C6F这个订单了", re.ASCII)
#:     ['BK9A4C6F']
#:
#: 原因：Python 3 的 ``\b`` 默认按 **Unicode** 语义判断单词字符，
#: 中文「我」「这」都算单词字符，于是 ``我B`` 与 ``F这`` 之间
#: **不存在边界**，``\b`` 匹配失败、返回空列表。
#: 加上 ``re.ASCII`` 后 ``\b`` 只认 ASCII 单词字符，才能正确切出订单号。
#:
#: 这个 bug 的实际后果很隐蔽：提取失败会静默回退到占位值 ``BK000000``，
#: 于是所有 clarify 偏好对的 rejected 都变成同一个字符串，
#: 被去重门禁压成**一条** —— 数据量看起来正常，实际只剩一个样本。
_BOOKING_ID_RE = re.compile(r"\bBK[0-9A-F]{6}\b", re.ASCII)


def extract_booking_id(text: str) -> str:
    """从用户话里抽订单号，抽不到返回空串。

    >>> extract_booking_id("我不想要BK9A4C6F这个订单了")
    'BK9A4C6F'
    >>> extract_booking_id("帮我取消订单BK7F2A9C")
    'BK7F2A9C'
    >>> extract_booking_id("我要退票")
    ''
    """
    m = _BOOKING_ID_RE.search(text or "")
    return m.group(0) if m else ""


# ============================================================================
# 构造器
# ============================================================================


class PreferenceBuilder:
    """按四类策略构造偏好数据（格式 / 恢复 / 工具选择 / 参数约束）。

    Args:
        cfg: 构造参数（条数、硬负样本比例、种子）。
        seed: 覆盖 cfg.seed。
    """

    def __init__(self, cfg: PreferenceCfg | None = None, seed: int | None = None) -> None:
        self.cfg = cfg or CFG.preference
        self.seed = seed if seed is not None else self.cfg.seed
        self.rng = random.Random(self.seed)
        self.system_prompt = build_system_prompt()
        #: 统计信息，写入 build_stats.json
        self.stats: dict[str, Any] = {
            "generated": {},
            "rejected_by_gate": {},
            "gate_reasons": {},
            "by_pair_type": {},
            "by_negative_kind": {},
            "by_difficulty": {},
            "by_split": {},
        }
        self._seen_keys: set[str] = set()

    # ------------------------------------------------------------------
    # 质量门禁
    # ------------------------------------------------------------------
    def _passes_quality_gate(self, pair: PreferencePair) -> tuple[bool, str]:
        if not pair.prompt.strip():
            return False, "empty_prompt"
        if not pair.chosen.strip() or not pair.rejected.strip():
            return False, "empty_side"
        if pair.chosen.strip() == pair.rejected.strip():
            return False, "identical_sides"

        # --- chosen 的格式与参数校验 ---
        #
        # 三种情况：
        #   1. canonical 格式   → 直接过格式检查
        #   2. 能解析出调用但不是 canonical 形态
        #      → **允许**（下游偏好数据里有大量这种样本：chosen 是规范调用，
        #        而某些恢复类样本的 chosen 依赖上下文才能判断形态），
        #        但仍必须过参数 schema 校验
        #   3. 解析不出任何调用 → chosen 是自然语言（例如「信息不足时应先追问」）。
        #        这类对完全合法，只要 rejected 与它语义不同即可 ——
        #        硬性拒绝会把「学会反问而不是硬填」这类最有价值的恢复样本丢掉。
        chosen_parse = extract_tool_calls(pair.chosen)
        rejected_parse = extract_tool_calls(pair.rejected)

        if not is_canonical_format(pair.chosen) and not chosen_parse.calls:
            # 情况 3：chosen 是自然语言
            if not rejected_parse.calls and _normalize_loose(pair.chosen) == _normalize_loose(pair.rejected):
                return False, "both_sides_natural_language_equal"

        # chosen 的参数必须通过 schema 校验（情况 2 与 1 都要过）
        #
        # ⚠️ 这里曾经有一段 `elif not is_canonical_format(pair.chosen): return False`
        # 的死分支：它嵌套在 `if not is_canonical_format(...)` 内部，
        # 条件恒为真，于是「能解析出调用但不是 canonical」的样本会被
        # **直接拒绝**，永远走不到下面这段参数校验 —— 既丢数据，
        # 也让 `chosen_invalid_args` 这条门禁对这类样本形同虚设。
        if chosen_parse.calls:
            call = chosen_parse.first
            issues = validate_arguments(call.name or "", call.arguments)
            if issues:
                return False, f"chosen_invalid_args:{issues[0].kind}"

        # ★ 最关键的一条：chosen 与 rejected 必须语义不同
        if chosen_parse.calls and rejected_parse.calls:
            c, r = chosen_parse.first, rejected_parse.first
            if c.name == r.name:
                m = match_arguments(c.arguments, r.arguments, graded_subset=c.arguments)
                if m.lenient_matched:
                    return False, "sides_semantically_equal"
        return True, ""

    def _recovery_coherence(self, pair: PreferencePair) -> tuple[bool, str]:
        """检查「用户纠错话术」是否与「首轮错误」对得上。

        为什么需要这个门禁：最初版本从整个纠错话术池里随机抽句子，
        结果生成了「航班搜索场景 + 『city 参数要填中文城市名』」这样
        前后矛盾的训练数据。这种数据比没有数据更糟 ——
        它教模型「用户的纠正可以忽略」。

        规则：纠错话术里提到的**参数名**必须真的出现在首轮错误调用的参数里。
        例如提到 ``city`` 就必须有一个叫 city 的参数；
        提到 ``origin``/``destination`` 同理。
        """
        import re

        prompt = pair.prompt
        # 取出 prompt 里 assistant 那段（首轮错误回复）
        asst_blocks = re.findall(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", prompt, flags=re.S)
        correction = re.findall(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", prompt, flags=re.S)
        if not asst_blocks or not correction:
            return True, ""  # 结构不对的话由其他门禁处理

        wrong_call = extract_tool_calls(asst_blocks[-1])
        wrong_args = set(wrong_call.first.arguments) if wrong_call.first else set()
        last_correction = correction[-1]

        # 纠错话术里显式点名的参数
        mentioned = set(re.findall(r"\b(city|origin|destination|date|booking_id|flight_id|reason|new_baggage_kg)\b", last_correction))
        # ``origin 和 destination`` 这类并列写法会被上面全部捕获
        missing = {m for m in mentioned if m not in wrong_args}
        if missing:
            return False, f"correction_mentions_absent_param:{','.join(sorted(missing))}"

        # 幻觉类纠错必须点名那个假工具，且假工具不能是真工具
        if pair.negative_kind in ("simple", "recovery") and "工具" in last_correction and wrong_call.first:
            from src.env.tool_catalog import VALID_TOOL_NAMES

            if wrong_call.first.name not in VALID_TOOL_NAMES:
                if wrong_call.first.name not in last_correction:
                    return False, "hallucination_correction_does_not_name_fake_tool"
        return True, ""

    def _add(self, pair: PreferencePair, bucket: str) -> bool:
        ok, reason = self._passes_quality_gate(pair)
        if not ok:
            self.stats["rejected_by_gate"][bucket] = self.stats["rejected_by_gate"].get(bucket, 0) + 1
            self.stats["gate_reasons"][reason] = self.stats["gate_reasons"].get(reason, 0) + 1
            return False
        if pair.pair_type == PAIR_RECOVERY:
            ok2, reason2 = self._recovery_coherence(pair)
            if not ok2:
                self.stats["rejected_by_gate"][bucket] = self.stats["rejected_by_gate"].get(bucket, 0) + 1
                self.stats["gate_reasons"][reason2] = self.stats["gate_reasons"].get(reason2, 0) + 1
                return False
        if pair.key in self._seen_keys:
            return False
        self._seen_keys.add(pair.key)

        for key, val in (
            ("by_pair_type", pair.pair_type),
            ("by_negative_kind", pair.negative_kind),
            ("by_difficulty", pair.difficulty or "unknown"),
            ("by_split", pair.split),
        ):
            self.stats[key][val] = self.stats[key].get(val, 0) + 1
        self.stats["generated"][bucket] = self.stats["generated"].get(bucket, 0) + 1
        return True

    # ------------------------------------------------------------------
    # 类型 1：格式约束对
    # ------------------------------------------------------------------
    def build_format_pairs(self, tasks: Sequence[Task], n: int) -> list[PreferencePair]:
        """chosen = 合法 tool_call；rejected = 纯文本/坏 JSON。

        这类对在教「把意图落成工具调用」这件事，是所有改进的前提：
        格式都错了，后面的工具选择和参数正确性无从谈起。
        """
        out: list[PreferencePair] = []
        candidates = [t for t in tasks if t.expected.tool is not None]
        if not candidates:
            return out
        i = 0
        guard = 0
        while len(out) < n and guard < n * 20:
            guard += 1
            task = candidates[i % len(candidates)]
            i += 1
            chosen = canonical_call(task.expected.tool, task.expected.graded_args)

            mode = self.rng.random()
            if mode < 0.45:
                rejected = self.rng.choice(_SIMPLE_NEGATIVES)
                kind, intent = "simple", "模型只是承诺要查，没有真的输出 tool_call"
            elif mode < 0.75:
                # 坏 JSON：键缺引号
                args = ", ".join(
                    f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in task.expected.graded_args.items()
                )
                rejected = f'<tool_call>{{"name": "{task.expected.tool}", "arguments": {{{args}}}}}</tool_call>'
                kind, intent = "hard", "有 tool_call 标签但 JSON 语法非法"
            else:
                # 用 markdown 围栏而不是标签
                body = json.dumps(
                    {"name": task.expected.tool, "arguments": task.expected.graded_args}, ensure_ascii=False
                )
                rejected = f"```json\n{body}\n```"
                kind, intent = "hard", "用 markdown 代码块代替规定的 tool_call 标签"

            pair = PreferencePair(
                prompt=render_prompt(task, system_prompt=self.system_prompt),
                chosen=chosen,
                rejected=rejected,
                pair_type=PAIR_FORMAT,
                task_id=task.task_id,
                category=task.category,
                difficulty=task.difficulty,
                target_error="FORMAT_ERROR",
                negative_kind=kind,
                tools=[],  # 工具说明已写进 system prompt
                intent=intent,
            )
            if self._add(pair, PAIR_FORMAT):
                out.append(pair)
        return out

    # ------------------------------------------------------------------
    # 类型 2：错误恢复对
    # ------------------------------------------------------------------
    def build_recovery_pairs(
        self, tasks: Sequence[Task], n: int, *, scenarios: Sequence[Any] | None = None
    ) -> list[PreferencePair]:
        """chosen = 纠错后正确修正；rejected = 原样重复错误/换一个错。

        prompt 里包含「模型的首轮错误回复 + 用户纠错话术 + 正确修正」，
        这是 SFT 数据里不可能出现的信号 —— 也是本项目相对
        「直接拿开源 preference dataset」的核心差异。

        数据源说明：不用评测集里的 12 个 recovery 任务（它们要保持固定、
        不被训练接触），而用 ``src/data/recovery_corpus.py`` 展开出的
        ~84 个参数化场景。评测与训练因此完全分离。
        """
        out: list[PreferencePair] = []

        if scenarios is None:
            from src.data.recovery_corpus import build_scenarios

            scenarios = build_scenarios()
        pool = list(scenarios)
        if not pool:
            return out

        i = 0
        guard = 0
        while len(out) < n and guard < n * 30:
            guard += 1
            sc = pool[i % len(pool)]
            i += 1

            chosen = sc.correct_reply
            correction = sc.correction

            mode = self.rng.random()
            if mode < 0.50:
                # 原样重复首轮错误 —— 真实观察到的最常见失败模式
                rejected = sc.wrong_reply
                kind = "recovery"
                intent = "被指出错误后原样重复，没有听进用户的纠正"
            elif mode < 0.75:
                # 退化成纯文本回复
                rejected = self.rng.choice(_SIMPLE_NEGATIVES)
                kind = "simple"
                intent = "被指出错误后反而退化成纯文本，不再输出工具调用"
            elif mode < 0.9:
                # 换了个工具但换错
                wrong_tool = self._pick_adjacent_tool_for_scenario(sc)
                rejected = canonical_call(wrong_tool, self._args_for_scenario(sc, wrong_tool))
                kind = "hard"
                intent = "被指出错误后换了工具，但换到了另一个同样错误的工具"
            else:
                # 格式对了但参数还是错的（最难的一档）
                rejected = self._same_tool_wrong_args(sc)
                kind = "hard"
                intent = "被指出错误后工具改对了，但关键参数仍然是错的"

            # 用一个临时的 Task-like 对象渲染 prompt
            prompt = self._render_scenario_prompt(sc, correction)
            pair = PreferencePair(
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
                pair_type=PAIR_RECOVERY,
                task_id=sc.scenario_id,
                category=f"recovery_{sc.error_family.lower()}",
                difficulty="hard",
                target_error=sc.error_family,
                negative_kind=kind,
                intent=intent,
            )
            if self._add(pair, PAIR_RECOVERY):
                out.append(pair)
        return out

    def _render_scenario_prompt(self, sc: Any, correction: str) -> str:
        """把纠错场景渲染成带 system prompt 的多轮 prompt。"""
        parts = [
            f"<|im_start|>system\n{self.system_prompt}<|im_end|>",
            f"<|im_start|>user\n{sc.user_request}<|im_end|>",
            f"<|im_start|>assistant\n{sc.wrong_reply}<|im_end|>",
            f"<|im_start|>user\n{correction}<|im_end|>",
            "<|im_start|>assistant\n",
        ]
        return "\n".join(parts)

    def _pick_adjacent_tool_for_scenario(self, sc: Any) -> str:
        """从场景的正确回复里解析出工具名，再挑一个相邻的错误工具。"""
        parse = extract_tool_calls(sc.correct_reply)
        correct = parse.first.name if parse.first else None
        return self._pick_adjacent_tool(correct)

    def _args_for_scenario(self, sc: Any, tool: str) -> dict[str, Any]:
        spec = TOOL_REGISTRY.get(tool)
        if spec is None:
            return {}
        parse = extract_tool_calls(sc.correct_reply)
        src = parse.first.arguments if parse.first else {}
        out: dict[str, Any] = {}
        for pname, pspec in spec.params.items():
            if not pspec.get("required"):
                continue
            if pname in src:
                out[pname] = src[pname]
            elif pname in ("origin", "destination"):
                out[pname] = self.rng.choice(list(AIRPORT_CODES.values()))
            elif pname == "city":
                out[pname] = self.rng.choice(list(AIRPORT_CODES))
            elif pname == "date":
                out[pname] = "2025-07-18"
            elif pname == "booking_id":
                out[pname] = "BK7F2A9C"
            elif pname == "flight_id":
                out[pname] = "MU5101"
            elif pname == "passenger_name":
                out[pname] = "张伟"
            elif pname == "reason":
                out[pname] = "行程变更"
            elif pname == "new_baggage_kg":
                out[pname] = 30
            else:
                out[pname] = "未知"
        return out

    def _same_tool_wrong_args(self, sc: Any) -> str:
        """工具正确、但把其中一个关键参数改错 —— 最难区分的负样本。"""
        parse = extract_tool_calls(sc.correct_reply)
        if not parse.first:
            return sc.wrong_reply
        call = parse.first
        if not call.arguments:
            return sc.wrong_reply
        bad = copy.deepcopy(call.arguments)
        field_name = self.rng.choice(list(bad))
        bad[field_name] = self._perturb(bad[field_name], field_name)
        return canonical_call(call.name or "", bad)

    # ------------------------------------------------------------------
    # 类型 3：工具选择对
    # ------------------------------------------------------------------
    def build_tool_selection_pairs(self, tasks: Sequence[Task], n: int) -> list[PreferencePair]:
        """chosen = 白名单里的正确工具；rejected = 假工具或语义相邻工具。

        针对 HALLUCINATION 与 WRONG_TOOL 两类错误。
        negative_kind=hard 的对里 rejected 是一个**真实存在**的相邻工具，
        模型必须理解工具语义才能区分 —— 这是最有价值的一类信号。
        """
        out: list[PreferencePair] = []
        candidates = [t for t in tasks if t.expected.tool is not None]
        if not candidates:
            return out

        i = 0
        guard = 0
        while len(out) < n and guard < n * 20:
            guard += 1
            task = candidates[i % len(candidates)]
            i += 1
            tool = task.expected.tool
            assert tool is not None
            chosen = canonical_call(tool, task.expected.graded_args)

            if self.rng.random() < self.cfg.hard_negative_ratio:
                wrong = self._pick_adjacent_tool(tool)
                rejected = canonical_call(wrong, self._args_for_tool(wrong, task))
                kind = "hard"
                intent = f"真实存在的相邻工具 {wrong}，语义上不能完成该请求"
            else:
                # 用「正确工具命名的变体」而不是随机假工具 ——
                # 见 tool_catalog.FAKE_TOOL_VARIANTS 的说明。
                # 注意变量名不要用 candidates —— 它已经是外层任务池了。
                fake_candidates = [f for f in plausible_fakes_for(tool) if f != tool]
                fake = self.rng.choice(fake_candidates)
                rejected = canonical_call(fake, dict(task.expected.graded_args) or {"query": "航空相关"})
                kind = "simple"
                intent = f"编造了与正确工具命名相近但并不存在的工具 {fake}"

            pair = PreferencePair(
                prompt=render_prompt(task, system_prompt=self.system_prompt),
                chosen=chosen,
                rejected=rejected,
                pair_type=PAIR_TOOL_SELECTION,
                task_id=task.task_id,
                category=task.category,
                difficulty=task.difficulty,
                target_error="HALLUCINATION" if kind == "simple" else "WRONG_TOOL",
                negative_kind=kind,
                intent=intent,
            )
            if self._add(pair, PAIR_TOOL_SELECTION):
                out.append(pair)
        return out

    # ------------------------------------------------------------------
    # 类型 4：参数约束对
    # ------------------------------------------------------------------
    def build_param_pairs(self, tasks: Sequence[Task], n: int) -> list[PreferencePair]:
        """chosen = 真实实体参数；rejected = 占位符/错格式/换形字符。

        针对 PARAM_ERROR。硬负样本是「形状完全正确、只是值错一个字符」
        （例如 ``BK7F2A9C`` → ``BK7F2A9D``）——
        这类对直接对抗 reward hacking（模型学会格式但乱填实体）。
        """
        out: list[PreferencePair] = []
        candidates = [t for t in tasks if t.expected.graded_args]
        if not candidates:
            return out

        i = 0
        guard = 0
        while len(out) < n and guard < n * 20:
            guard += 1
            task = candidates[i % len(candidates)]
            i += 1
            tool = task.expected.tool
            if tool is None:
                continue
            args = task.expected.graded_args
            chosen = canonical_call(tool, args)

            field_name = self.rng.choice(list(args))
            bad = copy.deepcopy(args)
            kind = "hard" if self.rng.random() < self.cfg.hard_negative_ratio else "simple"

            if kind == "simple":
                bad[field_name] = self._placeholder_for(field_name)
                intent = f"把 {field_name} 填成了占位符而不是真实实体"
            else:
                bad[field_name] = self._perturb(args[field_name], field_name)
                intent = f"{field_name} 的格式完全正确，但值错了一个字符/单位"

            rejected = canonical_call(tool, bad)
            pair = PreferencePair(
                prompt=render_prompt(task, system_prompt=self.system_prompt),
                chosen=chosen,
                rejected=rejected,
                pair_type=PAIR_PARAM,
                task_id=task.task_id,
                category=task.category,
                difficulty=task.difficulty,
                target_error="PARAM_ERROR",
                negative_kind=kind,
                intent=intent,
            )
            if self._add(pair, PAIR_PARAM):
                out.append(pair)
        return out

    # ------------------------------------------------------------------

    def build_clarify_pairs(self, tasks: Sequence[Task], n: int) -> list[PreferencePair]:
        """构造「该追问 vs 硬调工具」边界对。

        这一类**不是预先设计的**，是真实数据逼出来的：SFT 之后剩下的
        唯一失败类型就是「信息不足时应先追问，却直接调用了工具」（20 条，
        占失败 100%）。而原四类偏好对里没有任何一类覆盖这个决策边界 ——
        SFT 的数据全是「正确调用」的正例，模型自然被推成「无脑调工具」。

        构造逻辑：

        * ``chosen``：以「追问」收尾的自然语言回复（不调用工具）
        * ``rejected``：**把缺失的必填参数用编造值补上后照常调用**
          —— 这精确复现了实测到的模型行为，例如
          ``cancel_booking(booking_id="BK9A4C6F", reason="用户要求")``
          里 ``reason`` 是编的、用户从没说过。

        重点在于 rejected **不是乱码或明显错误**，而是一个格式完全合法、
        看起来很像样子的工具调用。模型必须从「参数值有没有依据」
        这个层面才能区分两侧 —— 这正是我们想教的判别能力。

        若任务集里没有 ``expected.tool is None`` 的任务，则用内置的
        合成场景兜底（保证这一类数据始终可构造）。

        >>> b = PreferenceBuilder(seed=7)
        >>> tasks = [t for t in load_tasks() if t.expected.tool is None]
        >>> pairs = b.build_clarify_pairs(tasks, 12)
        >>> len(pairs) > 0
        True
        >>> all(p.pair_type == PAIR_CLARIFY for p in pairs)
        True
        >>> all("<tool_call>" not in p.chosen for p in pairs)
        True
        >>> from src.eval.tool_call_parser import extract_tool_calls
        >>> all(extract_tool_calls(p.rejected).calls for p in pairs)
        True
        """
        out: list[PreferencePair] = []
        candidates = [t for t in tasks if t.expected.tool is None]

        i = 0
        guard = 0
        while len(out) < n and guard < n * 30:
            guard += 1

            if candidates:
                task = candidates[i % len(candidates)]
                i += 1
                template = self._clarify_scenario_for(task)
            else:
                template = dict(_SYNTHETIC_CLARIFY[i % len(_SYNTHETIC_CLARIFY)])
                i += 1
                # 合成场景没有 Task 对象，但 prompt 必须走同一套渲染，
                # 否则又会引入训练/评测分布不一致。
                template["prompt"] = render_user_prompt(
                    template["user"], system_prompt=self.system_prompt
                )
                template.setdefault("task_id", "")
                template.setdefault("category", "trainsyn")
                template.setdefault("difficulty", "medium")

            chosen = self.rng.choice(template["clarify"])
            # 硬调的形态：把缺失的必填参数用编造值补上
            missing = {k: self.rng.choice(v) for k, v in template["fabricated"].items()}
            rejected = canonical_call(template["tool"], {**template["known"], **missing})
            fabricated_fields = "、".join(missing)

            pair = PreferencePair(
                prompt=template["prompt"],
                chosen=chosen,
                rejected=rejected,
                pair_type=PAIR_CLARIFY,
                task_id=template.get("task_id", ""),
                category=template.get("category", "trainsyn"),
                difficulty=template.get("difficulty", "medium"),
                target_error="WRONG_TOOL",
                negative_kind="hard",
                intent=(
                    f"信息不足时应先追问，却直接调用了 {template['tool']}"
                    f"（其中 {fabricated_fields} 是编造的，用户从未提供）"
                ),
            )
            if self._add(pair, PAIR_CLARIFY):
                out.append(pair)
        return out

    def _clarify_scenario_for(self, task: Task) -> dict[str, Any]:
        """把 ``expected.tool is None`` 的任务转成合成场景模板。

        ``cancel_booking`` 需要 ``reason``，而用户只说「我不想要了」——
        这是缺参数→该追问的典型。

        订单号从**整个对话历史**里找（不一定在第一轮），
        并用 ``extract_booking_id`` 做 ASCII 边界匹配 ——
        直接用 ``\\b`` 会在中文旁失效，导致所有样本退化成同一个占位值。
        """
        user_text = task.turns[0].content
        booking_id = extract_booking_id(user_text)
        if not booking_id:
            for turn in task.turns:
                booking_id = extract_booking_id(turn.content)
                if booking_id:
                    break
        # 仍然找不到才用占位：此时该任务确实没有订单号可引用
        if not booking_id:
            booking_id = "BK000000"

        return {
            "prompt": render_prompt(task, system_prompt=self.system_prompt),
            "tool": "cancel_booking",
            "known": {"booking_id": booking_id},
            "fabricated": {
                "reason": ["用户要求", "行程变更", "不需要了", "个人原因"],
            },
            "clarify": [
                "好的，为了帮您取消订单，请问您取消的原因是什么？",
                "请问您方便告诉我取消的原因吗？我需要填写后才能为您办理。",
                "了解，请问取消这笔订单的原因是什么？",
                "好的，取消前我需要确认一下原因，方便告诉我吗？",
            ],
            "task_id": task.task_id,
            "category": task.category,
            "difficulty": task.difficulty,
        }

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------
    def _pick_adjacent_tool(self, tool: str | None) -> str:
        prefer = {
            "search_flights": ["get_weather", "check_baggage"],
            "book_flight": ["search_flights", "check_baggage"],
            "check_baggage": ["modify_baggage", "book_flight"],
            "modify_baggage": ["check_baggage", "cancel_booking"],
            "cancel_booking": ["modify_baggage", "check_baggage"],
            "get_weather": ["search_flights"],
        }.get(tool or "", [])
        pool = [t for t in prefer if t in TOOL_REGISTRY] or [t for t in TOOL_REGISTRY if t != tool]
        return self.rng.choice(pool)

    def _args_for_tool(self, tool: str, task: Task) -> dict[str, Any]:
        spec = TOOL_REGISTRY.get(tool)
        if spec is None:
            return {}
        out: dict[str, Any] = {}
        src = task.expected.graded_args
        for pname, pspec in spec.params.items():
            if not pspec.get("required"):
                continue
            if pname in src:
                out[pname] = src[pname]
            elif pname in ("origin", "destination"):
                out[pname] = self.rng.choice(list(AIRPORT_CODES.values()))
            elif pname == "city":
                out[pname] = self.rng.choice(list(AIRPORT_CODES))
            elif pname == "date":
                out[pname] = "2025-07-18"
            elif pname == "booking_id":
                out[pname] = "BK7F2A9C"
            elif pname == "flight_id":
                out[pname] = "MU5101"
            elif pname == "passenger_name":
                out[pname] = "张伟"
            elif pname == "reason":
                out[pname] = "行程变更"
            elif pname == "new_baggage_kg":
                out[pname] = 30
            else:
                out[pname] = "未知"
        return out

    def _placeholder_for(self, field_name: str) -> Any:
        if field_name in ("booking_id",):
            return self.rng.choice(["previous_booking", "my_booking", "BK000000", "UNKNOWN"])
        if field_name == "flight_id":
            return self.rng.choice(["CA0000", "previous_flight", "FLIGHT_ID"])
        if "date" in field_name:
            return self.rng.choice(["后天", "明天", "TBD", "2025-13-45"])
        if field_name in ("origin", "destination"):
            return self.rng.choice(["北京", "北京首都", "PEK机场"])
        if field_name == "city":
            return self.rng.choice(["PEK", "SHA", "未知"])
        if field_name == "passenger_name":
            return self.rng.choice(["乘客", "本人", "张三"])
        if field_name == "reason":
            return self.rng.choice(["用户要求", "无", "待定"])
        if isinstance(field_name, str) and field_name.endswith("_kg"):
            return self.rng.choice([0, -5, 999, "很多"])
        return "未知"

    def _perturb(self, value: Any, field_name: str) -> Any:
        """把值改成一个「形状一模一样、内容错一点」的值（硬负样本）。"""
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            delta = self.rng.choice([-5, -2, 2, 5, 10])
            new = value + delta
            return max(0, new) if field_name.endswith("_kg") else new
        if isinstance(value, str):
            if value.isdigit():
                return str(int(value) + self.rng.choice([-3, 3, 1]))
            if len(value) >= 2:
                idx = self.rng.randrange(len(value))
                repl = self.rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789")
                return value[:idx] + repl + value[idx + 1 :]
        return str(value) + "?"

    # ------------------------------------------------------------------
    # 总入口
    # ------------------------------------------------------------------
    def build_all(
        self,
        tasks: Sequence[Task] | None = None,
        *,
        n_format: int | None = None,
        n_recovery: int | None = None,
        n_tool_selection: int | None = None,
        n_param: int | None = None,
        n_clarify: int | None = None,
        include_param: bool = True,
        include_clarify: bool = True,
    ) -> list[PreferencePair]:
        """构造全部偏好对。

        Args:
            include_param: 是否包含参数约束对。
                项目方案的 Exp 1-3 只声明了格式 + 恢复两类；
                参数约束类是**额外提出的一类**（见 docs 里的说明），
                单独开关便于把它作为一组独立消融。
            include_clarify: 是否包含「该追问」边界对。
                这一类是**真实基线数据逼出来的**（SFT 后剩下的唯一失败类型），
                不是预先设计的。见 ``PAIR_CLARIFY`` 的说明。

        Returns:
            全部偏好对（已过质量门禁、已去重）。
        """
        # 默认**不**吃评测任务集：训练数据一律来自合成语料。
        # 传 None 时只建合成任务池；显式传入的 tasks 会被门禁检查。
        tasks = list(tasks) if tasks is not None else []
        n_format = self.cfg.n_format if n_format is None else n_format
        n_recovery = self.cfg.n_recovery if n_recovery is None else n_recovery
        n_tool = self.cfg.n_tool_selection if n_tool_selection is None else n_tool_selection

        # 把纠错语料转成的训练任务也加进候选池。
        # 关键约束：这些任务的 visibility 是 train_only，
        # 永远不会出现在评测集里（见 recovery_corpus.build_training_tasks）。
        from src.data.recovery_corpus import (
            SYNTHETIC_TASK_PREFIX,
            build_scenarios,
            build_training_tasks,
        )

        scenarios = build_scenarios(seed=self.seed)
        self.stats["num_recovery_scenarios"] = len(scenarios)
        syn_tasks = build_training_tasks(seed=self.seed)
        self.stats["num_trainsyn_tasks"] = len(syn_tasks)

        # ------------------------------------------------------------------
        # ⚠️⚠️ 硬门禁：训练数据绝不能来自评测任务集 ⚠️⚠️
        # ------------------------------------------------------------------
        #
        # 开发期在这里犯过一个**会作废全部实验结论**的错误：
        # ``build_format_pairs`` / ``build_param_pairs`` /
        # ``build_tool_selection_pairs`` 都直接吃 ``pool = tasks + syn_tasks``，
        # 而 ``tasks`` 就是那 100 个评测任务。后果实测：
        #
        #     sft_only 的 2509 条训练样本里，1292 条（51.5%）的 prompt
        #     与评测 prompt **逐字相同**，且 chosen 就是该任务的正确答案。
        #
        # 这等于**在测试集上做梯度下降**：SFT 的 95.0% 准确率因此虚高，
        # 与泛化能力无关。
        #
        # 更隐蔽的是：``build_recovery_pairs`` 的注释明确写了
        # 「不用评测集里的 recovery 任务…评测与训练因此完全分离」——
        # 但**只有它做到了**，另外三处没有。**注释说的和代码做的不一致**，
        # 是这类错误能长期潜伏的根本原因。
        #
        # 现在把「只用合成语料」变成硬约束，越界即抛错。
        #
        # 判据方向：**带有** SYNTHETIC_TASK_PREFIX 前缀的才是合成语料（允许），
        # 不带该前缀的是评测任务（禁止）。本行原先写成
        # ``if not t.task_id.startswith(EVAL_TASK_PREFIX)``，配合那个语义相反的
        # 常量名读起来像「不带前缀的才是合法的」，极易被后人"修正"成
        # ``startswith(...)`` 从而**静默失效**。
        leaked = [t for t in tasks if not t.task_id.startswith(SYNTHETIC_TASK_PREFIX)]
        if leaked:
            raise ValueError(
                "训练数据中混入了评测任务，这会让全部实验结论作废：\n"
                f"  越界任务数: {len(leaked)}"
                f"（样例 {[t.task_id for t in leaked[:5]]}）\n"
                "  训练只能用 build_training_tasks() 产出的合成任务"
                f"（task_id 前缀 {SYNTHETIC_TASK_PREFIX!r}）。"
            )

        pool = tasks + syn_tasks

        pairs: list[PreferencePair] = []
        pairs += self.build_format_pairs(pool, n_format)
        pairs += self.build_recovery_pairs(pool, n_recovery, scenarios=scenarios)
        pairs += self.build_tool_selection_pairs(pool, n_tool)
        if include_param:
            pairs += self.build_param_pairs(pool, n_param if n_param is not None else n_format)
        if include_clarify:
            # 用**评测任务集**（tasks）而非 pool：该边界对要精确复现评测时的
            # 「信息不足」场景，合成纠错任务里没有这种轮次结构。
            pairs += self.build_clarify_pairs(
                tasks, n_clarify if n_clarify is not None else self.cfg.n_clarify
            )

        # 划分 train / dev。**按 task_id 划分**而不是按样本随机划分 ——
        # 同一任务的多个偏好对必须在同一侧，否则 dev 会因为
        # 见过同一 prompt 的其他变体而虚高，起不到早停的作用。
        task_ids = sorted({p.task_id for p in pairs})
        rng = random.Random(self.seed + 999)
        rng.shuffle(task_ids)
        n_dev = max(1, int(len(task_ids) * 0.1))
        dev_ids = set(task_ids[:n_dev])
        for p in pairs:
            p.split = "dev" if p.task_id in dev_ids else "train"

        # 重新统计 split（_add 时还不知道最终划分）
        self.stats["by_split"] = {}
        for p in pairs:
            self.stats["by_split"][p.split] = self.stats["by_split"].get(p.split, 0) + 1

        self.rng.shuffle(pairs)
        return pairs


# ============================================================================
# 落盘
# ============================================================================


def write_jsonl(pairs: Iterable[PreferencePair], path: str | Path) -> int:
    """写 jsonl，返回条数。

    使用 ``ensure_ascii=False`` 保留中文原文 —— 这是训练数据，
    转义后会让真实 token 数与预期不符。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


#: 各实验需要哪些类型的偏好对 —— 这是消融矩阵的可执行定义。
#:
#: ⚠️ **这个配比表已被真实基线数据修正过一次，不要照抄原方案。**
#:
#: 原方案（基于 mock 假设值）给 ``tool_selection`` 配了 600 条，
#: 针对的是「模型编造工具名」。真实基线实测（Qwen2.5-1.5B，400 样本）
#: **工具幻觉率 = 0%** —— 这个假设不成立，那 600 条预期收益接近零。
#:
#: 同时真实数据暴露了一个原方案完全没覆盖的失败类型：
#: 「信息不足时应先追问，却直接调用了工具」（SFT 后占失败 100%），
#: 于是新增 ``PAIR_CLARIFY``。
#:
#: ``sft_only`` 的说明：它是对照组，验证「提升来自**对比信号**，
#: 还是仅仅来自更多同分布监督」。SFT 只用 chosen 做监督，
#: 因此它需要覆盖「要学什么」的那几类（格式 / 工具名 / 参数），
#: 但**不包含** ``error_recovery`` —— 恢复能力无法由「只有正确答案」
#: 的数据教出来，这正是本项目要证明的核心论点。
#: 它同样**不包含** ``clarify_boundary`` —— 这一类只有在对比信号下
#: 才能学会（要看到「硬调」这个反面例子），正例无法表达「此处不该调用」。
EXPERIMENT_PAIR_TYPES: dict[str, tuple[str, ...]] = {
    "dpo_format": (PAIR_FORMAT,),
    "dpo_recovery": (PAIR_RECOVERY,),
    # joint = 格式 + 恢复 + 追问边界（原为 format+recovery，实测后加 clarify）
    "dpo_joint": (PAIR_FORMAT, PAIR_RECOVERY, PAIR_CLARIFY),
    "dpo_joint_param": (PAIR_FORMAT, PAIR_RECOVERY, PAIR_PARAM),
    "dpo_tool": (PAIR_TOOL_SELECTION,),
    "dpo_all": (PAIR_FORMAT, PAIR_RECOVERY, PAIR_TOOL_SELECTION, PAIR_PARAM, PAIR_CLARIFY),
    "dpo_clarify": (PAIR_CLARIFY,),
    # 对照：SFT 用同样的 chosen，但没有对比信号，也没有恢复类/追问边界样本
    "sft_only": (PAIR_FORMAT, PAIR_TOOL_SELECTION, PAIR_PARAM),
    # ⚠️ **实验变体**：在 sft_only 基础上加「追问边界」的 chosen。
    #
    # 动机（实测查出）：`sft_only` 的 2635 条里，`chosen` **不含** `tool_call`
    # 的是 **0 条** —— 纯正例监督在结构上无法表达「此处不该调用工具」。
    # 后果：v3 基线上 60/89 的失败是「信息不足时应先追问，却直接调用了工具」，
    # 而补追问**任务**（语料 20→58）完全没用 —— 因为那些任务
    # **从没进过 SFT 数据**（pair_type 组合里没有 clarify_boundary）。
    #
    # 加 364 条 clarify 偏好对后，不含 tool_call 的占 **12.1%**，
    # 而评测里「应追问」任务占 **14.2%** —— 比例接近，不构成偏置。
    #
    # ⚠️ **风险**：若模型学不到「条件规则」而只学到「总是追问」，
    # 准确率会从 78.5% 掉到 ~14%（因为 91/106 的任务要调工具）。
    # 所以这个实验的价值在于**判定模型能否学到条件规则**，
    # 而不是「加数据就能提升」。
    "sft_clarify": (PAIR_FORMAT, PAIR_TOOL_SELECTION, PAIR_PARAM, PAIR_CLARIFY),
    # ⚠️ `sft_all` 原来只在 `SFT_EXPERIMENTS` 里、**没有 pair_type 定义**
    # （孤儿条目：能选中实验名，但没有数据生成规则）。
    # 补全它是为了消除「列表里有、但跑不了」这种不一致 ——
    # 那类不一致的危害是：`--experiment sft_all` 会静默退回声明文件，
    # 读到别的实验的数据。
    "sft_all": (PAIR_FORMAT, PAIR_RECOVERY, PAIR_TOOL_SELECTION, PAIR_PARAM,
                PAIR_CLARIFY),
    # 只加工具选择对 → 单独定位幻觉的贡献
    "dpo_tool_param": (PAIR_TOOL_SELECTION, PAIR_PARAM),
}


def build_experiment_dataset(
    experiment: str,
    all_pairs: Sequence[PreferencePair],
    *,
    split: str = "train",
    balance: bool = False,
) -> list[PreferencePair]:
    """从全量偏好对里筛出某个实验需要的子集。

    Args:
        experiment: ``EXPERIMENT_PAIR_TYPES`` 的键。
        split: ``train`` / ``dev`` / ``all``。
        balance: 是否让各类型条数相等（默认按自然配比）。
            类型不均衡时，条数多的类型会主导梯度；
            本项目默认保持格式:恢复 = 1:1（两者的目标错误率量级相当），
            因此不需要额外平衡。

    >>> build_experiment_dataset("dpo_format", [])
    []
    """
    if experiment not in EXPERIMENT_PAIR_TYPES:
        raise ValueError(f"未知实验 {experiment!r}，可选: {sorted(EXPERIMENT_PAIR_TYPES)}")
    wanted = set(EXPERIMENT_PAIR_TYPES[experiment])
    out = [
        p
        for p in all_pairs
        if p.pair_type in wanted and (split == "all" or p.split == split)
    ]
    if balance and out:
        by_type: dict[str, list[PreferencePair]] = {}
        for p in out:
            by_type.setdefault(p.pair_type, []).append(p)
        k = min(len(v) for v in by_type.values())
        out = [p for v in by_type.values() for p in v[:k]]
    return out


def summarize_pairs(pairs: Sequence[PreferencePair]) -> str:
    """人类可读的偏好数据统计，用于日志与报告。"""
    if not pairs:
        return "偏好数据为空"

    def _count(key: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for p in pairs:
            v = getattr(p, key) or "unknown"
            out[str(v)] = out.get(str(v), 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    lines = [
        f"偏好对总计 {len(pairs)} 条",
        "-" * 58,
        "按类型:",
    ]
    for k, v in _count("pair_type").items():
        lines.append(f"    {PAIR_TYPE_LABELS_ZH.get(k, k):<14}{v:>6}")
    lines.append("按负样本难度:")
    for k, v in _count("negative_kind").items():
        lines.append(f"    {k:<14}{v:>6}")
    lines.append("按目标错误类型:")
    for k, v in _count("target_error").items():
        lines.append(f"    {k:<14}{v:>6}")
    lines.append("按难度:")
    for k, v in _count("difficulty").items():
        lines.append(f"    {k:<14}{v:>6}")
    lines.append("按划分:")
    for k, v in _count("split").items():
        lines.append(f"    {k:<14}{v:>6}")

    # 长度统计（决定 max_length 该怎么设）
    lens = [len(p.prompt) + len(p.chosen) + len(p.rejected) for p in pairs]
    lens.sort()
    if lens:
        lines.append(
            f"字符长度: min={lens[0]} p50={lens[len(lens) // 2]} "
            f"p90={lens[int(len(lens) * 0.9)]} max={lens[-1]}"
        )
        # 粗估 token 数（中文约 1.5 字符/token，英文约 4）
        est = [x / 1.6 for x in lens]
        lines.append(
            f"预估 token（粗估）: p50={est[len(est) // 2]:.0f} "
            f"p90={est[int(len(est) * 0.9)]:.0f} max={est[-1]:.0f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    b = PreferenceBuilder()
    pairs = b.build_all(include_param=True)
    print(summarize_pairs(pairs))
    print()
    print("质量门禁统计:")
    print(json.dumps(b.stats, ensure_ascii=False, indent=2))
