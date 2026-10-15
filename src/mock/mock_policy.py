"""Mock 策略：在一个**没有 GPU 的机器**上真实地跑完整条实验流水线。

它是什么，不是什么
------------------
**是**：一个确定性的、参数化的「合成策略」，它按可调的错误率掷骰子，
产出与真实模型输出**格式完全一致**的预测记录。因此
``评测 → 解析 → 错误分类 → 偏好对构造 → 训练 → 再评测 → 画图``
这条链路的每一环都会被真实执行，代码 bug 会被真实暴露。

**不是**：任何关于 Qwen2.5-1.5B 或 DPO 效果的结论来源。
Mock 的数字是我们**喂进去的假设**，不是实验发现。

所以本项目把产物严格分开：
::

    experiments/<exp>/eval_report.mock.json    ← mock 产出，仅供自检
    experiments/<exp>/eval_report.json         ← 真实 GPU 产出，报告只引用这个

``scripts/eval/collect_results.py`` 默认只读 ``eval_report.json``，
mock 结果要显式加 ``--include-mock`` 才会被纳入，并且会在报告里标注。

错误注入模型
------------
对每个任务按难度给出「首轮就成功」的概率，剩下的概率质量按
``MockPolicyCfg.mitigation_factors`` 分配给五类错误。
不同实验改变的是这张乘性系数表：

======================  ==========================================================
实验                     注入的假设
======================  ==========================================================
``baseline``            未训练，五类错误都按基线概率出现
``sft_only``            只用 chosen 做监督：格式改善明显，恢复能力几乎不变
``dpo_format``          只有格式偏好对：格式错误大降，其余几乎不动
``dpo_recovery``        只有恢复偏好对：恢复能力大增，首轮错误率降幅有限
``dpo_joint``           两者叠加：格式 + 幻觉 + 恢复都改善
======================  ==========================================================

这个「哪个实验改善哪一类错误」的映射，就是
``docs/ablation/ablation_plan.md`` §1.3 实验矩阵的可执行版本。

纠错轮任务的特殊处理
--------------------
纠错轮任务（``task.is_recovery``）考察的是「被指出错误后能否改对」，
所以错误注入的对象是**恢复概率**而不是首轮错误率：
以 ``1 − recovery_rate`` 的概率重放首轮那段错误输出（模拟「没听进去」）。
这和真实观察到的失败模式一致 —— 小模型在纠错轮最常见的失败是
**原样重复上一轮的错误**，而不是犯一个全新的错。
"""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass, field
from typing import Any

from src.config import CFG, MockPolicyCfg
from src.data.task_manifest import Task
from src.env.tool_catalog import (
    AIRPORT_CODES,
    CITY_NAMES,
    PLAUSIBLE_FAKE_TOOLS,
    TOOL_REGISTRY,
    looks_like_placeholder,
    plausible_fakes_for,
)
from src.eval.error_taxonomy import FORMAT_ERROR, HALLUCINATION, PARAM_ERROR, SUCCESS, WRONG_TOOL
from src.eval.grader import TaskPrediction
from src.eval.tool_call_parser import canonical_call, extract_tool_calls
from src.utils.hashing import stable_seed

# ============================================================================
# 实验矩阵
# ============================================================================

#: 实验名 → 默认的 mitigation 系数键（用于查 MockPolicyCfg）
#: 这几个名字同时是 configs/train/*.yaml 的文件名、outputs/ 的子目录名、
#: experiments/ 的子目录名 —— 全仓库统一用这一套标识。
EXPERIMENTS: tuple[str, ...] = (
    "baseline",
    "sft_only",
    "dpo_format",
    "dpo_recovery",
    "dpo_joint",
)

EXPERIMENT_LABELS_ZH: dict[str, str] = {
    "baseline": "基线（未训练）",
    "sft_only": "对照：仅 SFT（只用 chosen）",
    "dpo_format": "Exp 1：格式约束偏好对",
    "dpo_recovery": "Exp 2：错误恢复偏好对",
    "dpo_joint": "Exp 3：联合方案（格式 + 恢复）",
}


#: 参数错误用到的错误类型键（与 error_taxonomy 的类别常量对齐）
NO_RECOVERY = "NO_RECOVERY"

#: ``CATEGORY_REMEDIES_ZH`` 之类展示层用的中文名由 error_taxonomy 提供
_ERROR_KEYS_INJECTABLE: tuple[str, ...] = (
    FORMAT_ERROR,
    HALLUCINATION,
    WRONG_TOOL,
    PARAM_ERROR,
)


# ============================================================================
# 需要额外参数的参数错误（必须从任务上下文里取，所以单独处理）
# ============================================================================

#: 当这些参数出错时，只能走「填占位符」这一条路 —— 编造一个合法格式的
#: 值反而会被判成另一类错误（形状对但值错），两者诊断含义不同。
_CONTEXT_ONLY_PARAMS: frozenset[str] = frozenset({"flight_id", "booking_id"})

#: 订单号占位符池
_BOOKING_PLACEHOLDERS: tuple[str, ...] = (
    "previous_booking",
    "previous_reservation",
    "BK000000",
    "my_booking",
    "UNKNOWN",
    "订单号",
)

#: 相对日期占位符池（模拟「没把后天换算成绝对日期」）
_RELATIVE_DATE_ECHOES: tuple[str, ...] = ("后天", "明天", "本周五", "下周一", "TBD", "未知")


def _sigmoid(x: float) -> float:
    """1/(1+exp(-x))，数值稳定实现。保留给概率校准脚本使用。"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


# ============================================================================
# Mock 策略
# ============================================================================


@dataclass
class MockPolicy:
    """按配置好的错误率合成的策略。

    Args:
        experiment: ``baseline`` / ``sft_only`` / ``dpo_format`` /
            ``dpo_recovery`` / ``dpo_joint``。
        seed: 采样种子。**逐次采样必须用不同种子**（这里用
            ``seed + task_index * 1000 + sample_index``），否则同一任务的
            4 次采样会得到完全相同的结果，pass^k 就恒等于 accuracy，
            「稳定性」这个维度会被静默抹掉。
        cfg: 错误率配置，默认取全局 ``CFG.mock``。
        enabled_errors: 只注入这些错误类型（消融实验用）。
            None 表示全部启用。
        recovery_multiplier: 对恢复概率的额外乘子，用于扫描。
    """

    experiment: str = "baseline"
    seed: int = 42
    cfg: MockPolicyCfg = field(default_factory=lambda: CFG.mock)
    enabled_errors: tuple[str, ...] | None = None
    recovery_multiplier: float = 1.0

    #: 运行时统计（每类错误实际注入了多少次），用于自检「注入生效了吗」
    stats: dict[str, int] = field(default_factory=dict)
    #: 逐任务固定的困难度偏移，缓存起来保证同一任务跨采样一致
    _task_offsets: dict[str, float] = field(default_factory=dict)
    _call_count: int = 0

    #: 任务级难度偏移的标准差（取正数则模型在该任务上整体偏强）。
    #:
    #: 为什么必须有这一项：如果每次采样都独立掷骰子，
    #: 那么每任务 4 次采样近似 i.i.d.，pass^k ≈ accuracy^k，
    #: 而 accuracy 接近 0.5 时 0.5^4 = 6%，各实验几乎无差异 ——
    #: 「稳定性」这个维度会被静默抹平，pass^k 就失去了区分力。
    #: 真实模型的行为是：对某些任务稳定做对，对另一些稳定做错，
    #: 只有中间地带才是随机的。这个偏移项就是在建模这种**任务级异质性**。
    difficulty_sigma: float = 1.0
    #: 同一任务内逐次采样的抖动
    within_task_sigma: float = 0.45

    def __post_init__(self) -> None:
        if self.experiment not in self.cfg.mitigation_factors:
            raise ValueError(
                f"未知实验 {self.experiment!r}，可选: {sorted(self.cfg.mitigation_factors)}"
            )

    def _task_offset(self, task_id: str) -> float:
        """逐任务固定的困难度偏移（正态，均值 0）。

        用 hash 派生而不是 random：同一 task_id 在任何进程、任何顺序下
        都得到同一个偏移，这让 mock 结果完全可复现。
        """
        if task_id not in self._task_offsets:
            rng = random.Random(stable_seed(self.seed, "offset", task_id))
            self._task_offsets[task_id] = rng.gauss(0.0, self.difficulty_sigma)
        return self._task_offsets[task_id]

    # ------------------------------------------------------------------
    # 概率模型
    # ------------------------------------------------------------------
    def _probabilities(self, task: Task, sample_index: int = 0) -> dict[str, float]:
        """第 ``sample_index`` 次采样时，各类错误的发生概率与成功率。

        结构（这是 mock 能区分各实验的关键）::

            p_err(e)   = baseline_error_rates[e] × mitigation_factors[exp][e]
            task_scale = 任务难度/类别修正（>1 表示这个任务更难）
            p_err(e)  *= task_scale
            p_success  = 1 − Σ p_err(e)              ← 成功是补集，不是独立设定

        「成功 = 1 − Σ错误」保证了改善一类错误**必然**转化为准确率提升，
        因此各实验的 accuracy / pass^k 会真实产生差异。

        任务级异质性与 sampling 抖动都在 logit 空间施加，
        以保证概率始终落在 (0,1) 且首次采样不加抖动（均值可校准）。
        """
        factors = self.cfg.mitigation_factors[self.experiment]
        base = self.cfg.baseline_error_rates

        # 逐类缩减
        err = {k: base.get(k, 0.0) * factors.get(k, 1.0) for k in base}

        # 该类错误对这个任务是否有意义
        if not task.is_recovery:
            err[NO_RECOVERY] = 0.0
        if task.expected.tool is None:
            # 该轮本该澄清而不调用工具：格式错误（没输出 tool_call）反而是对的，
            # 这里的错误形态是「贸然调用」，归到 wrong_tool。
            err[FORMAT_ERROR] = 0.0
            err[HALLUCINATION] = 0.0

        # --- 任务难度 / 类别修正 ---
        scale = 1.0
        if task.difficulty == "easy":
            scale -= 0.25
        elif task.difficulty == "hard":
            scale += 0.20
        if task.category == "booking_flow":
            scale += 0.10
        if task.category == "weather_advisory":
            scale -= 0.15
        if task.expected.tool is None:
            scale += 0.35  # 「知道不该调用」本身就难
        scale = max(scale, 0.1)

        err = {k: v * scale for k, v in err.items()}

        # --- 任务级异质性（逐任务固定）---
        sigma = self._task_offset(task.task_id) * 0.35
        err = {k: v * math.exp(sigma) for k, v in err.items()}

        # --- 采样抖动（首次为 0）---
        if sample_index > 0:
            rng = random.Random(stable_seed(self.seed, "jitter", task.task_id, sample_index))
            jitter = (rng.random() - 0.5) * 2 * self.within_task_sigma * 0.3
            err = {k: v * math.exp(jitter) for k, v in err.items()}

        # --- 数值安全：错误概率总和必须 < 1，超了就等比压回去 ---
        total = sum(err.values())
        cap = 0.95
        if total > cap:
            shrink = cap / total
            err = {k: v * shrink for k, v in err.items()}
            total = cap

        # --- 消融开关：只保留指定的错误类型 ---
        if self.enabled_errors is not None:
            err = {k: (v if k in self.enabled_errors else 0.0) for k, v in err.items()}
            total = sum(err.values())

        err["_success"] = 1.0 - total
        return err

    def _pick_error_type(self, task: Task, probs: dict[str, float], rng: random.Random) -> str:
        """在「已经确定会失败」的前提下，按错误类型的相对占比抽一个。"""
        keys = [k for k in probs if not k.startswith("_") and probs[k] > 0]
        vals = [probs[k] for k in keys]
        total = sum(vals)
        if total <= 0:
            return FORMAT_ERROR
        r = rng.random() * total
        acc = 0.0
        for k, v in zip(keys, vals):
            acc += v
            if r <= acc:
                return k
        return keys[-1]

    def _recovery_rate(self, task: Task) -> float:
        """纠错轮任务被纠正成功的概率。"""
        base = 0.30
        if task.difficulty == "easy":
            base += 0.10
        rate = base * self.cfg.recovery_gain.get(self.experiment, 1.0) * self.recovery_multiplier
        return min(max(rate, 0.02), 0.97)

    # ------------------------------------------------------------------
    # 生成「错误输出」
    # ------------------------------------------------------------------
    def _correct_text(self, task: Task) -> str:
        expected = task.expected
        if expected.tool is None:
            # 该轮不该调用工具 → 正解是一句追问
            return (
                "请问您能提供一下取消的原因吗？我需要记录取消理由才能帮您处理这个订单。"
            )
        return canonical_call(expected.tool, expected.graded_args)

    def _format_error_text(self, task: Task, rng: random.Random) -> str:
        """格式错误的几种典型形态。"""
        variants = [
            "好的，我来帮您查询一下，请稍等。",
            "正在为您查询相关信息，马上就好。",
            "我明白了，我先帮您确认一下。",
            "好的，我这就为您处理这个问题。",
            "稍等片刻，我为您查一下。",
        ]
        if rng.random() < 0.25 and task.expected.tool is not None:
            # 坏 JSON 形态
            tool = task.expected.tool
            args = ", ".join(
                f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in task.expected.graded_args.items()
            )
            return f'<tool_call>{{"name": "{tool}", "arguments": {{{args}}}}}</tool_call>'
        # 拒绝作答只在「格式约束尚未改善」的实验里按低概率出现。
        # 这是真实小模型的形态之一，但比例不宜高，否则会被评分器
        # 单独判成 REFUSED，让注入一致性自检失真。
        if rng.random() < 0.05 * self.cfg.mitigation_factors[self.experiment].get(FORMAT_ERROR, 1.0):
            return "抱歉，我无法完成这个请求。"
        return rng.choice(variants)

    def _hallucination_text(self, task: Task, rng: random.Random) -> str:
        # 用「正确工具命名的变体」当幻觉名，而不是随机挑一个完全不相干的工具。
        # 理由：``send_confirmation_email`` 配上 search_flights 的参数，
        # 一眼就是假的，模型不需要理解工具语义就能排除 —— 没有诊断价值。
        # 而 ``find_flights`` 与 ``search_flights`` 语义相同、只差命名，
        # 这才是 1.5B 模型真实的幻觉形态。见 tool_catalog.FAKE_TOOL_VARIANTS。
        pool = [t for t in plausible_fakes_for(task.expected.tool) if t != task.expected.tool]
        fake = rng.choice(pool)
        # 参数从任务的期望参数抄过来，模拟「工具名编错但参数看着合理」
        args = dict(task.expected.graded_args) if task.expected.graded_args else {"query": "航空相关"}
        return canonical_call(fake, args)

    def _wrong_tool_text(self, task: Task, rng: random.Random) -> str:
        real_tools = [n for n in TOOL_REGISTRY if n != task.expected.tool]
        # 优先选「语义相邻」的工具 —— 这比随机选更有诊断价值
        prefer = {
            "search_flights": ["get_weather", "check_baggage"],
            "book_flight": ["search_flights", "check_baggage"],
            "check_baggage": ["modify_baggage", "book_flight"],
            "modify_baggage": ["check_baggage", "cancel_booking"],
            "cancel_booking": ["modify_baggage", "check_baggage"],
            "get_weather": ["search_flights"],
        }.get(task.expected.tool or "", [])
        pool = [t for t in prefer if t in real_tools] or real_tools
        tool = rng.choice(pool)
        args = self._args_for_tool(tool, task, rng)
        return canonical_call(tool, args)

    def _param_error_text(self, task: Task, rng: random.Random) -> str:
        """参数错误的几种形态：占位符、格式错、相对日期未换算。"""
        tool = task.expected.tool
        if tool is None:
            # 该轮本该澄清却调用了工具 → 归到 wrong_tool 更准确，这里兜底
            return self._wrong_tool_text(task, rng)

        args = copy.deepcopy(task.expected.graded_args)
        spec = TOOL_REGISTRY[tool]
        if not args:
            return self._wrong_tool_text(task, rng)

        mode = rng.random()
        field = rng.choice(list(args))

        if mode < 0.40:
            # 占位符
            if field in _CONTEXT_ONLY_PARAMS:
                args[field] = rng.choice(_BOOKING_PLACEHOLDERS)
            elif field == "new_baggage_kg":
                args[field] = -5
            else:
                args[field] = rng.choice(["previous_value", "my_info", "未知", "N/A", "TBD"])
        elif mode < 0.70:
            # 格式错：中文城市名 / 三字码互换 / 日期分隔符 / 数值当字符串
            pspec = spec.params.get(field, {})
            pattern = pspec.get("pattern", "")
            val = args[field]
            if field in ("origin", "destination") and isinstance(val, str) and val in CITY_NAMES:
                args[field] = CITY_NAMES[val]  # 三字码 → 中文名
            elif field == "city" and isinstance(val, str) and val in AIRPORT_CODES:
                args[field] = AIRPORT_CODES[val]  # 中文名 → 三字码
            elif "date" in pattern:
                args[field] = rng.choice(_RELATIVE_DATE_ECHOES)
            elif pspec.get("type") == "number":
                args[field] = str(val) + "公斤"
            elif pattern.startswith("^BK"):
                args[field] = str(val).lower()
            else:
                args[field] = str(val) + "?"
        else:
            # 值错：形状对但实体不同（reward hacking 的典型信号）
            val = args[field]
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                args[field] = val + rng.choice([-5, 5, 10, -10])
            elif isinstance(val, str):
                if val.isdigit():
                    args[field] = str(int(val) + rng.choice([-3, 3]))
                elif len(val) >= 2:
                    # 换一个同形字符
                    idx = rng.randrange(len(val))
                    repl = rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789")
                    args[field] = val[:idx] + repl + val[idx + 1 :]
        return canonical_call(tool, args)

    def _args_for_tool(self, tool: str, task: Task, rng: random.Random) -> dict[str, Any]:
        """给一个非期望的工具凑一组「看起来合理」的参数。"""
        spec = TOOL_REGISTRY.get(tool)
        if spec is None:
            return {}
        expected_args = task.expected.graded_args
        out: dict[str, Any] = {}
        for pname, pspec in spec.params.items():
            if not pspec.get("required"):
                continue
            if pname in expected_args:
                out[pname] = expected_args[pname]
            elif pname in ("origin", "destination"):
                out[pname] = rng.choice(list(AIRPORT_CODES.values()))
            elif pname == "city":
                out[pname] = rng.choice(list(AIRPORT_CODES))
            elif pname == "date":
                out[pname] = "2025-07-18"
            elif pname == "booking_id":
                out[pname] = rng.choice(["BK7F2A9C", "BK3D8E1B"])
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

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def sample(self, task: Task, sample_index: int, *, task_index: int = 0) -> TaskPrediction:
        """产出一条预测记录。

        逐次采样用不同的派生种子，保证 pass^k 有真实方差。
        """
        rng = random.Random(stable_seed(self.seed, task.task_id, sample_index))
        self._call_count += 1

        # ---------- 纠错轮任务：注入「恢复 / 没恢复」 ----------
        if task.is_recovery:
            rate = self._recovery_rate(task)
            recovered = rng.random() < rate
            if recovered:
                text = self._correct_text(task)
                self._bump(SUCCESS)
            else:
                # 原样重放首轮的错误输出
                text = task.turns[1].content if len(task.turns) > 1 else ""
                self._bump(NO_RECOVERY)
            return TaskPrediction(
                task_id=task.task_id,
                sample_index=sample_index,
                text=text,
                finish_reason="stop",
                num_tokens=len(text) // 3,
                policy=self.experiment,
                meta={"mock": True, "injected": "recovery" if recovered else NO_RECOVERY},
            )

        # ---------- 普通任务：先定成功/失败，再定错误类型 ----------
        probs = self._probabilities(task, sample_index)
        if rng.random() < probs["_success"]:
            text = self._correct_text(task)
            self._bump(SUCCESS)
            return TaskPrediction(
                task_id=task.task_id,
                sample_index=sample_index,
                text=text,
                finish_reason="stop",
                num_tokens=len(text) // 3,
                policy=self.experiment,
                meta={"mock": True, "injected": SUCCESS},
            )

        etype = self._pick_error_type(task, probs, rng)
        if etype == FORMAT_ERROR:
            text = self._format_error_text(task, rng)
            finish = "length" if rng.random() < 0.1 else "stop"
        elif etype == HALLUCINATION:
            text = self._hallucination_text(task, rng)
            finish = "stop"
        elif etype == WRONG_TOOL:
            text = self._wrong_tool_text(task, rng)
            finish = "stop"
        elif etype == PARAM_ERROR:
            text = self._param_error_text(task, rng)
            finish = "stop"
        else:  # no_recovery 在非纠错轮不会出现，兜底给格式错
            text = self._format_error_text(task, rng)
            finish = "stop"

        self._bump(etype)
        return TaskPrediction(
            task_id=task.task_id,
            sample_index=sample_index,
            text=text,
            finish_reason=finish,
            num_tokens=len(text) // 3,
            policy=self.experiment,
            meta={"mock": True, "injected": etype},
        )

    def _bump(self, key: str) -> None:
        self.stats[key] = self.stats.get(key, 0) + 1

    # ------------------------------------------------------------------
    # 自检
    # ------------------------------------------------------------------
    def verify_injection(self, tasks: list[Task], num_samples: int = 4) -> dict[str, Any]:
        """检查「注入的错误类型」与「评分器判出的类别」是否一致。

        这个自检非常重要：如果 mock 想注入 PARAM_ERROR 但生成的文本
        被评分器判成 FORMAT_ERROR，说明 mock 构造器与评分器的口径不一致，
        整个 mock 链路就是在自欺欺人。一致性高（>90%）才说明链路可信。
        """
        from src.eval.grader import grade

        agree = 0
        total = 0
        confusion: dict[str, dict[str, int]] = {}
        for ti, task in enumerate(tasks):
            for si in range(num_samples):
                pred = self.sample(task, si, task_index=ti)
                injected = pred.meta.get("injected", "?")
                judged = grade(task, pred.text, finish_reason=pred.finish_reason).category
                # 纠错轮注入的是恢复行为，不与类别直接对应
                if task.is_recovery:
                    continue
                total += 1
                injected_cat = injected if injected in (SUCCESS, FORMAT_ERROR, HALLUCINATION, WRONG_TOOL, PARAM_ERROR) else "?"
                if injected_cat == judged:
                    agree += 1
                confusion.setdefault(injected_cat, {}).setdefault(judged, 0)
                confusion[injected_cat][judged] += 1
        return {
            "consistency": round(agree / total, 4) if total else 0.0,
            "total_checked": total,
            "confusion": confusion,
        }

    def as_policy_fn(self, tasks: list[Task] | None = None):
        """返回 ``run_evaluation`` 需要的 ``(task, sample_index) -> TaskPrediction`` 函数。"""
        index = {t.task_id: i for i, t in enumerate(tasks or [])}

        def _fn(task: Task, sample_index: int) -> TaskPrediction:
            return self.sample(task, sample_index, task_index=index.get(task.task_id, 0))

        return _fn


def make_policy(experiment: str, seed: int | None = None, **kwargs: Any) -> MockPolicy:
    """便捷构造器。

    >>> p = make_policy("dpo_joint", seed=1)
    >>> p.experiment
    'dpo_joint'
    """
    return MockPolicy(experiment=experiment, seed=seed if seed is not None else CFG.seed, **kwargs)
