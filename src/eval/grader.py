"""评分器：把一条策略输出判成 SUCCESS 或某一类错误。

调用链::

    模型输出文本
        │
        ├─ extract_tool_calls()        解析（src/eval/tool_call_parser.py）
        │
        ├─ match_arguments()           参数语义等价比对（src/eval/equivalence.py）
        │
        └─ classify()                  错误分类（src/eval/error_taxonomy.py）
                                        │
                                        └─ GradeResult   ← 本模块的输出

为什么不直接比对字符串
----------------------
见 ``src/eval/equivalence.py`` 的模块文档。这里只补充一点：
评分器的判定口径直接决定了 DPO 的训练目标是否合理。
如果评分器把「同义但写法不同」判成错，偏好数据里就会出现
「chosen 与 rejected 其实等价」的垃圾对，DPO 会去拟合噪声。

两套口径同时报告
----------------
- **严格口径**（strict）：期望参数必须全部命中且值等价，不容忍多余字段。
  这是「能不能一次做对」的天花板。
- **宽松口径**（lenient）：只比对关键字段（``args_subset``）。
  这是主指标 —— 对那些「模型没义务猜对」的字段（如需要从 observation 取
  的 flight_id）不做要求。

两个都报告的原因：如果宽松涨了而严格没涨，说明模型学会了「关键字段对、
细节不管」的捷径 —— 这正是需要提前发现并写进报告的退化模式。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from src.data.task_manifest import Task
from src.env.tool_catalog import PLAUSIBLE_FAKE_TOOLS, VALID_TOOL_NAMES
from src.eval.error_taxonomy import (
    SUCCESS,
    Classification,
    ErrorDistribution,
    classify,
)
from src.eval.equivalence import match_arguments, text_similarity
from src.eval.metrics import EvalReport, SampledTaskResult
from src.eval.tool_call_parser import ToolCall, extract_tool_calls, format_defects, is_canonical_format

# ============================================================================
# 数据结构
# ============================================================================


@dataclass
class GradeResult:
    """一次采样的完整评分结论。"""

    task_id: str
    passed: bool
    classification: Classification
    #: 模型输出是否满足 canonical 格式（与「任务是否成功」独立）
    format_ok: bool
    #: 解析出的调用（可能为空）
    calls: list[ToolCall] = field(default_factory=list)
    #: 严格口径是否命中
    strict_ok: bool = False
    #: 宽松口径是否命中（= passed，保留字段便于对照）
    lenient_ok: bool = False
    #: 参数级准确率（0~1）
    param_accuracy: float = 0.0
    #: 格式缺陷清单
    defects: list[str] = field(default_factory=list)
    #: 是否「格式对但值错」—— reward hacking 的信号
    format_ok_value_wrong: bool = False

    @property
    def category(self) -> str:
        return self.classification.category

    def to_dict(self, include_text: str = "") -> dict[str, Any]:
        d = {
            "task_id": self.task_id,
            "passed": self.passed,
            "category": self.category,
            "classification": self.classification.to_dict(),
            "format_ok": self.format_ok,
            "strict_ok": self.strict_ok,
            "lenient_ok": self.lenient_ok,
            "param_accuracy": round(self.param_accuracy, 4),
            "defects": self.defects,
            "format_ok_value_wrong": self.format_ok_value_wrong,
            "predicted": [c.to_dict() for c in self.calls],
        }
        if include_text:
            d["text"] = include_text[:2000]
        return d


@dataclass
class TaskPrediction:
    """一次采样的原始预测记录（会落到 jsonl，作为一切分析的输入）。"""

    task_id: str
    sample_index: int
    text: str
    finish_reason: str = ""
    #: 生成耗时（秒），用于报告延迟
    latency_s: float = 0.0
    #: 生成的 token 数（若可得）
    num_tokens: int = 0
    #: 策略名（baseline / dpo_joint ...）
    policy: str = ""
    #: 额外信息（例如 truncated 标志）
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "sample_index": self.sample_index,
            "text": self.text,
            "finish_reason": self.finish_reason,
            "latency_s": round(self.latency_s, 4),
            "num_tokens": self.num_tokens,
            "policy": self.policy,
            "meta": self.meta,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> TaskPrediction:
        return TaskPrediction(
            task_id=d["task_id"],
            sample_index=d.get("sample_index", 0),
            text=d.get("text", ""),
            finish_reason=d.get("finish_reason", ""),
            latency_s=d.get("latency_s", 0.0),
            num_tokens=d.get("num_tokens", 0),
            policy=d.get("policy", ""),
            meta=d.get("meta", {}),
        )


# ============================================================================
# 单条评分
# ============================================================================



def _tool_schema_fields(tool_name: str | None) -> set[str]:
    """返回该工具 schema 里声明的全部参数名。

    用于区分「模型给了有效但未评分的参数」（不算错）与「模型编造了
    不存在的参数」（算错）。早期版本没有这个区分，导致 42 条
    ``extra_param`` 里 36 条是误判 —— 其中包括 ``cancel_booking.reason``
    这种 **schema 必填**参数。

    >>> sorted(_tool_schema_fields("cancel_booking"))
    ['booking_id', 'reason']
    >>> _tool_schema_fields("不存在的工具")
    set()
    >>> _tool_schema_fields(None)
    set()

    工具不在库里时返回空集 —— 此时 scorer 退化为「与 expected 比对」的
    保守行为，不会因为查不到 schema 而放过真正多余的字段。
    """
    if not tool_name:
        return set()
    try:
        from src.env.tool_catalog import TOOL_REGISTRY

        spec = TOOL_REGISTRY.get(tool_name)
        return set(spec.params) if spec else set()
    except Exception:  # pragma: no cover
        return set()


def grade(task: Task, text: str, *, finish_reason: str = "", max_new_tokens: int = 0) -> GradeResult:
    """对一条输出打分。

    >>> from src.data.task_manifest import load_tasks
    >>> t = next(x for x in load_tasks() if x.expected.tool == "get_weather")
    >>> r = grade(t, '<tool_call>{"name": "get_weather", "arguments": {"city": "北京"}}</tool_call>')
    >>> r.passed, r.category, r.format_ok
    (True, 'SUCCESS', True)

    >>> r2 = grade(t, '北京今天多云，26 度')
    >>> r2.passed, r2.category
    (False, 'FORMAT_ERROR')

    >>> t2 = next(x for x in load_tasks() if x.expected.tool == "get_weather")
    >>> r3 = grade(t2, '<tool_call>{"name": "get_travel_advisory", "arguments": {"city": "北京"}}</tool_call>')
    >>> r3.passed, r3.category
    (False, 'HALLUCINATION')
    """
    parse = extract_tool_calls(text)
    expected = task.expected

    truncated = bool(max_new_tokens) and finish_reason == "length"

    # ---- 参数比对（工具名匹配时才有意义） ----
    arg_match = None
    if parse.calls and parse.calls[0].name == expected.tool:
        # ⚠️ 必须把工具 schema 的合法参数传进去，否则「模型给了有效但未评分的
        # 参数」会被误判成「多给了 schema 未定义的参数」。实测该误判占全部
        # 失败的 58.8%（42 条 extra_param 里 36 条是误判）：
        #   cancel_booking 的 reason（**schema 必填**）
        #   book_flight 的 flight_id（**schema 必填**）、cabin（schema 合法可选）
        # 即模型给对了必填参数反而被罚 —— 这是评分器最严重的一处口径错误。
        arg_match = match_arguments(
            expected.args,
            parse.calls[0].arguments,
            graded_subset=expected.graded_args,
            allow_extra_fields=True,
            date_tolerance_days=0,
            schema_fields=_tool_schema_fields(expected.tool),
        )

    classification = classify(
        text=text,
        calls=parse.calls,
        parse_error=parse.error,
        expected_tool=expected.tool,
        arg_match=arg_match,
        valid_tool_names=VALID_TOOL_NAMES,
        plausible_fake_tools=PLAUSIBLE_FAKE_TOOLS,
        finish_reason=finish_reason,
        max_new_tokens_hit=truncated,
        # 必须传 had_tag：否则「有标签但 JSON 坏」与「压根没输出标签」
        # 会被混为一谈（见 error_taxonomy.classify 的说明与那个真实误标 bug）
        had_tag=parse.had_tag,
    )

    passed = classification.category == SUCCESS
    format_ok = is_canonical_format(text)
    strict_ok = bool(arg_match and arg_match.strict_matched) if arg_match else passed
    lenient_ok = bool(arg_match and arg_match.lenient_matched) if arg_match else passed
    param_acc = arg_match.param_accuracy() if arg_match else (1.0 if passed else 0.0)

    return GradeResult(
        task_id=task.task_id,
        passed=passed,
        classification=classification,
        format_ok=format_ok,
        calls=parse.calls,
        strict_ok=strict_ok,
        lenient_ok=lenient_ok,
        param_accuracy=param_acc,
        defects=format_defects(text),
        format_ok_value_wrong=classification.format_ok_value_wrong,
    )


# ============================================================================
# 恢复行为分析（纠错轮任务专用）
# ============================================================================


def analyze_recovery(task: Task, text: str) -> dict[str, Any]:
    """分析纠错轮里模型「是否重复了首轮的错误」。

    首轮错误文本是 ``task.turns[1].content``（assistant 的错误回复）。
    这里比对模型在纠错轮的输出与首轮错误输出的相似度：
    高相似 = 没有听进用户的纠正（NO_RECOVERY 的核心表现）。

    Returns:
        dict，含 ``repeated_first_mistake`` / ``similarity`` / ``switched_tool``
    """
    if not task.is_recovery or len(task.turns) < 2:
        return {"is_recovery": False}

    first_wrong = task.turns[1].content if task.turns[1].role == "assistant" else ""
    parse = extract_tool_calls(text)
    prev_parse = extract_tool_calls(first_wrong)

    sim = text_similarity(first_wrong, text)
    prev_tool = prev_parse.first.name if prev_parse.first else None
    now_tool = parse.first.name if parse.first else None

    return {
        "is_recovery": True,
        "mistake_key": task.mistake_hint,
        "similarity": round(sim, 4),
        # 阈值 0.75：实证上重复同一句错误回复的相似度通常在 0.9 以上，
        # 而正确修正后的回复相似度一般低于 0.5。
        "repeated_first_mistake": sim >= 0.75,
        "switched_tool": prev_tool is not None and now_tool is not None and prev_tool != now_tool,
        "prev_tool": prev_tool,
        "now_tool": now_tool,
        "had_prev_call": prev_parse.first is not None,
        "has_now_call": parse.first is not None,
    }


# ============================================================================
# 批次评分与报告
# ============================================================================


def grade_batch(
    tasks: Sequence[Task],
    predictions: Sequence[TaskPrediction],
    *,
    max_new_tokens: int = 0,
) -> tuple[list[GradeResult], ErrorDistribution]:
    """批量评分，返回逐条结果与错误分布。"""
    by_id = {t.task_id: t for t in tasks}
    results: list[GradeResult] = []
    dist = ErrorDistribution()

    for pred in predictions:
        task = by_id.get(pred.task_id)
        if task is None:
            continue
        r = grade(task, pred.text, finish_reason=pred.finish_reason, max_new_tokens=max_new_tokens)
        results.append(r)
        dist.add(
            r.classification,
            task_id=task.task_id,
            meta={
                "category": task.category,
                "difficulty": task.difficulty,
                "visibility": task.visibility,
                "sample_index": pred.sample_index,
            },
        )
    return results, dist


def build_sampled_results(
    tasks: Sequence[Task],
    predictions: Sequence[TaskPrediction],
    *,
    max_new_tokens: int = 0,
) -> list[SampledTaskResult]:
    """把逐次采样聚合成「每任务多次采样」的结构，供 pass^k 计算。"""
    by_id = {t.task_id: t for t in tasks}
    grouped: dict[str, list[TaskPrediction]] = {}
    for p in predictions:
        grouped.setdefault(p.task_id, []).append(p)

    out: list[SampledTaskResult] = []
    for tid, preds in grouped.items():
        task = by_id.get(tid)
        if task is None:
            continue
        preds = sorted(preds, key=lambda p: p.sample_index)
        grades = [
            grade(task, p.text, finish_reason=p.finish_reason, max_new_tokens=max_new_tokens)
            for p in preds
        ]
        out.append(
            SampledTaskResult(
                task_id=tid,
                outcomes=[g.passed for g in grades],
                category=task.category,
                difficulty=task.difficulty,
                visibility=task.visibility,
                categories=[g.category for g in grades],
                texts=[p.text for p in preds],
                predictions=[g.to_dict() for g in grades],
            )
        )
    out.sort(key=lambda r: r.task_id)
    return out


def build_eval_report(
    label: str,
    tasks: Sequence[Task],
    predictions: Sequence[TaskPrediction],
    *,
    k_values: Iterable[int] = (1, 2, 4),
    primary_k: int = 4,
    metadata: dict[str, Any] | None = None,
    max_new_tokens: int = 0,
) -> EvalReport:
    """端到端：预测记录 → 完整评测报告。"""
    sampled = build_sampled_results(tasks, predictions, max_new_tokens=max_new_tokens)
    n_per = max((r.n for r in sampled), default=0)
    meta = dict(metadata or {})
    # 把恢复行为的分布也塞进 metadata，报告里可直接引用
    if any(t.is_recovery for t in tasks):
        meta["recovery_analysis"] = recovery_summary(tasks, predictions)
    meta["num_predictions"] = len(predictions)
    return EvalReport(
        label=label,
        num_tasks=len(sampled),
        num_samples_per_task=n_per,
        results=sampled,
        k_values=tuple(k for k in k_values if k <= n_per) or (1,),
        primary_k=min(primary_k, n_per) if n_per else 1,
        metadata=meta,
    )


def recovery_summary(
    tasks: Sequence[Task], predictions: Sequence[TaskPrediction]
) -> dict[str, Any]:
    """纠错轮任务的**通过率**摘要（写进 ``eval_report.json`` 的 metadata）。

    ⚠️ **本函数有意不使用 ``recovery_rate`` 这个名字。**

    历史上仓库里出现过三个同名的 ``recovery_rate``，含义各不相同，极易引用错：

    ==========================================  ==========================================
    来源                                         实际含义
    ==========================================  ==========================================
    ``experiments/<exp>/recovery_metrics.json``  Σrecoveries / Σerrors —— **权威定义**
    ``error_analysis.json``                       count(成功) / n —— 其实是**通过率**
    ``eval_report.json → metadata``（本函数）      count(成功) / n —— 同样是**通过率**
    ==========================================  ==========================================

    真正的「错误恢复率」必须是「改对次数 / 犯错次数」，只有
    ``scripts/eval/eval_recovery.py`` 走的**真实多轮**流程才数得出来
    （它知道每次对话里到底错了几次）。而本函数站在**单轮评分**的视角，
    每一轮只知道「对/错」，不知道「这次错是不是上次那个错的延续」，
    因此它只能给出通过率。

    结论：**引用恢复率时只认 ``recovery_metrics.json``**。
    本函数返回的键名都带 ``_rate`` 后缀且语义明确为「占样本比例」，
    避免与权威定义混淆。
    """
    by_id = {t.task_id: t for t in tasks}
    rec_tasks = {t.task_id: t for t in tasks if t.is_recovery}
    if not rec_tasks:
        return {"num_recovery_samples": 0}

    total = 0
    recovered = 0
    repeated = 0
    per_key: dict[str, dict[str, int]] = {}

    for pred in predictions:
        task = by_id.get(pred.task_id)
        if task is None or not task.is_recovery:
            continue
        info = analyze_recovery(task, pred.text)
        g = grade(task, pred.text, finish_reason=pred.finish_reason)
        total += 1
        recovered += int(g.passed)
        repeated += int(bool(info.get("repeated_first_mistake")))
        bucket = per_key.setdefault(task.mistake_hint or "unknown", {"total": 0, "recovered": 0, "repeated": 0})
        bucket["total"] += 1
        bucket["recovered"] += int(g.passed)
        bucket["repeated"] += int(bool(info.get("repeated_first_mistake")))

    return {
        "_definition_note": (
            "本节的比率均为「占纠错轮样本数的比例」（通过率），"
            "不是权威的 recovery_rate（=改对次数/犯错次数）。"
            "权威指标见 experiments/<exp>/recovery_metrics.json。"
        ),
        "num_recovery_samples": total,
        "recovered": recovered,
        # 通过率：成功样本 / 纠错轮样本数
        "recovered_rate": round(recovered / total, 6) if total else 0.0,
        # 重复率：原样重复首轮错误 / 纠错轮样本数（这个定义与 eval_recovery 一致，可放心引用）
        "repeat_rate": round(repeated / total, 6) if total else 0.0,
        "by_mistake_key": {
            k: {
                **v,
                "recovered_rate": round(v["recovered"] / v["total"], 4) if v["total"] else 0.0,
                "repeat_rate": round(v["repeated"] / v["total"], 4) if v["total"] else 0.0,
            }
            for k, v in sorted(per_key.items())
        },
    }


# ============================================================================
# 运行一轮评测（策略无关）
# ============================================================================

#: 策略签名：接收 (task, sample_index) 返回 TaskPrediction
PolicyFn = Callable[[Task, int], TaskPrediction]


def run_evaluation(
    policy_fn: PolicyFn,
    tasks: Sequence[Task],
    *,
    label: str,
    num_samples_per_task: int = 4,
    k_values: Iterable[int] = (1, 2, 4),
    primary_k: int = 4,
    seed: int = 42,
    metadata: dict[str, Any] | None = None,
    max_new_tokens: int = 0,
    progress: bool = True,
    stream_path: str | Path | None = None,
    progress_every: int = 10,
) -> tuple[list[TaskPrediction], EvalReport]:
    """跑一整轮评测。

    Args:
        policy_fn: ``(task, sample_index) -> TaskPrediction``。
            真实模型与 mock 策略共用这个接口 —— 这是「同一套评测协议
            评所有方案」的工程保证。
        progress: 是否打印进度（长跑时开，单元测试关）。
        stream_path: 传入文件路径时**边跑边追加写入 jsonl**。
        progress_every: 每多少个任务刷一次进度与日志。

    ⚠️ 为什么需要 ``stream_path``：

    早期版本把全部预测攒在内存里，**跑完才落盘**。在云端跑 100 任务 × 4 采样
    （约 30 分钟）时有三个后果：

    1. **看不到任何进度** —— 输出文件在结束前一直是空文件，无法判断跑到哪了
    2. **崩溃即全丢** —— 跑到 95% 时 OOM 或断线，前 95% 的结果全部作废
    3. 日志还有缓冲问题，两者叠加后完全无法观测

    加上流式写入后，任何时刻都能用 ``wc -l`` 看进度，
    中途崩溃也保留已完成的部分（后续可用 ``read_predictions`` 续算报告）。

    进度行同时带上**速率与预计剩余时间**（ETA），便于判断要不要等。

    Returns:
        ``(predictions, report)``
    """
    predictions: list[TaskPrediction] = []
    t_start = time.time()
    stream_fh = None
    n_tasks = len(tasks)
    total_samples = n_tasks * num_samples_per_task

    if stream_path is not None:
        sp = Path(stream_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        # 新开一轮评测时截断旧文件，避免与上次运行的结果混在一起
        stream_fh = sp.open("w", encoding="utf-8")

    def _emit(pred: TaskPrediction) -> None:
        if stream_fh is not None:
            stream_fh.write(json.dumps(pred.to_dict(), ensure_ascii=False) + "\n")
            stream_fh.flush()  # 立即落盘，崩溃也不丢

    try:
        for ti, task in enumerate(tasks):
            for si in range(num_samples_per_task):
                try:
                    pred = policy_fn(task, si)
                except Exception as exc:  # 单次失败不能毁掉整轮评测
                    pred = TaskPrediction(
                        task_id=task.task_id,
                        sample_index=si,
                        text="",
                        finish_reason="error",
                        policy=label,
                        meta={"exception": f"{type(exc).__name__}: {exc}"},
                    )
                if not pred.policy:
                    pred.policy = label
                predictions.append(pred)
                _emit(pred)

            if progress and ((ti + 1) % progress_every == 0 or ti + 1 == n_tasks):
                done = ti + 1
                elapsed = time.time() - t_start
                rate = done / elapsed if elapsed > 0 else 0.0
                eta = (n_tasks - done) / rate if rate > 0 else 0.0
                print(
                    f"  [{label}] {done}/{n_tasks} 任务"
                    f"（{done * num_samples_per_task}/{total_samples} 采样）  "
                    f"{elapsed / 60:.1f} 分已用  "
                    f"{rate * 60:.1f} 任务/分  "
                    f"预计剩余 {eta / 60:.1f} 分",
                    flush=True,  # 云上重定向到文件时防缓冲
                )
    finally:
        if stream_fh is not None:
            stream_fh.close()

    report = build_eval_report(
        label,
        tasks,
        predictions,
        k_values=k_values,
        primary_k=primary_k,
        metadata={**(metadata or {}), "seed": seed, "wall_time_s": round(time.time() - t_start, 2)},
        max_new_tokens=max_new_tokens,
    )
    return predictions, report


# ============================================================================
# 读写
# ============================================================================


def write_predictions(predictions: Sequence[TaskPrediction], path: str | Path) -> str:
    """逐行写 jsonl —— 原始预测必须落盘，因为它是所有后续分析的唯一来源。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred.to_dict(), ensure_ascii=False) + "\n")
    return str(p)


def read_predictions(path: str | Path) -> list[TaskPrediction]:
    out: list[TaskPrediction] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(TaskPrediction.from_dict(json.loads(line)))
    return out
