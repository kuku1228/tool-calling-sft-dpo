"""DPO 数据集：加载偏好 jsonl、统计真实 token 长度、校验 prompt 一致性。

三个必须由本模块回答的问题
--------------------------
1. **``max_length`` 该设多少？** → ``token_length_stats()``
   用真实 tokenizer 量出 prompt / prompt+completion 的分位数。
   靠猜会导致大量样本被截断（指标虚低）或显存浪费。

2. **训练与推理的 prompt 是否一致？** → ``verify_prompt_consistency()``
   同一条数据分别经 ``preference_builder`` 与 ``prompt_builder``
   渲染，逐字比对。不一致就直接报错，不要等到训练完才发现指标不动。

3. **多少样本会被截断？** → ``truncation_report()``
   在训练前给出「p99 长度 / max_length」的比值。
   如果 p99 超过 max_length，说明有 1% 以上的样本被砍，
   偏好对的 chosen/rejected 可能被砍成一样 → 变成噪声。

数据格式
--------
每行一个 JSON 对象::

    {
      "prompt":   "<|im_start|>system\\n...<|im_end|>\\n<|im_start|>assistant\\n",
      "chosen":   "<tool_call>{...}</tool_call>",
      "rejected": "<tool_call>{...}</tool_call>",
      "pair_type": "format_constraint",
      ...
    }

``prompt`` 已经是套好 chat template 的纯文本，DPOTrainer 会把它
当普通 prompt 处理。这样做的理由见
``docs/ops/prompt_contract.md``（为什么工具说明写在 system prompt 里
而不是通过 ``apply_chat_template(tools=...)`` 注入）：tools 通道的渲染结果
与我们训练的 ``<tool_call>`` 格式不保证一致，
而 prompt 文本自己带格式则完全可控。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

# ============================================================================
# 加载
# ============================================================================

#: DPOTrainer 需要的字段
REQUIRED_FIELDS: tuple[str, ...] = ("prompt", "chosen", "rejected")


def load_preference_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """读偏好 jsonl。

    Raises:
        ValueError: 缺少必需字段或文件为空。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"偏好数据不存在: {p}")
    rows: list[dict[str, Any]] = []
    with p.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{p}:{lineno} JSON 解析失败: {exc}") from exc
            missing = [k for k in REQUIRED_FIELDS if not str(obj.get(k, "")).strip()]
            if missing:
                raise ValueError(f"{p}:{lineno} 缺少字段 {missing}")
            rows.append(obj)
    if not rows:
        raise ValueError(f"偏好数据为空: {p}")
    return rows


def to_dpo_dataset(rows: Sequence[dict[str, Any]], tokenizer: Any = None) -> Any:
    """转成 ``datasets.Dataset``。

    Args:
        rows: ``load_preference_jsonl`` 的输出。
        tokenizer: 传入时会把 prompt/chosen/rejected 预先 tokenize 化？
            **不**。这里不做 tokenize —— 交给 DPOTrainer 自己的
            collator 处理，避免两处 tokenize 逻辑不一致。
            tokenizer 参数保留是为了未来可能的 chat-template 路径。

    Raises:
        ImportError: 没有安装 datasets。
    """
    try:
        from datasets import Dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "训练需要 datasets 库：pip install datasets\n"
            "若只想验证数据，用 scripts/prepare/inspect_preferences.py"
        ) from exc
    # 只保留 DPOTrainer 认得的列，其余列会让 remove_unused_columns 出问题
    keep = list(REQUIRED_FIELDS)
    cleaned = [{k: r[k] for k in keep} for r in rows]
    return Dataset.from_list(cleaned)


# ============================================================================
# 长度统计
# ============================================================================


@dataclass
class TokenLengthStats:
    """token 长度分布。"""

    num_samples: int = 0
    prompt_p50: int = 0
    prompt_p90: int = 0
    prompt_p99: int = 0
    prompt_max: int = 0
    total_p50: int = 0
    total_p90: int = 0
    total_p99: int = 0
    total_max: int = 0
    completion_p50: int = 0
    completion_p99: int = 0
    completion_max: int = 0
    tokenizer_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_samples": self.num_samples,
            "tokenizer_name": self.tokenizer_name,
            "prompt_tokens": {
                "p50": self.prompt_p50,
                "p90": self.prompt_p90,
                "p99": self.prompt_p99,
                "max": self.prompt_max,
            },
            "prompt_plus_completion_tokens": {
                "p50": self.total_p50,
                "p90": self.total_p90,
                "p99": self.total_p99,
                "max": self.total_max,
            },
            "completion_tokens": {
                "p50": self.completion_p50,
                "p99": self.completion_p99,
                "max": self.completion_max,
            },
        }

    def recommend_max_length(self, safety: float = 1.02, round_to: int = 128) -> dict[str, int]:
        """按 p99 推荐 ``max_length`` / ``max_prompt_length``。

        用 p99 而不是 max：单个异常长样本会把 max_length 顶到很高，
        浪费显存。用 p99 意味着约 1% 的样本会被截断 —— 这个代价
        远小于「为了 1 个样本把 max_length 翻倍」。
        截断比例由 ``truncation_report`` 显式报出，不做隐瞒。

        >>> s = TokenLengthStats(prompt_p99=900, total_p99=1100)
        >>> s.recommend_max_length()["max_length"]
        1152
        """
        import math

        def _ceil(v: float) -> int:
            return int(math.ceil(v * safety / round_to) * round_to)

        return {
            "max_length": max(_ceil(self.total_p99), 256),
            "max_prompt_length": max(_ceil(self.prompt_p99), 128),
        }


def token_length_stats(
    rows: Sequence[dict[str, Any]],
    tokenizer: Any,
    *,
    sample_limit: int = 400,
    seed: int = 42,
) -> TokenLengthStats:
    """用真实 tokenizer 统计长度。

    默认只测前 ``sample_limit`` 条（打乱后取样）——
    完整测 4000 条要几分钟，而分位数在几百条上已经稳定。
    """
    import random

    from src.eval.metrics import percentile

    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    idx = idx[:sample_limit]

    prompt_lens: list[int] = []
    total_lens: list[int] = []
    comp_lens: list[int] = []

    for i in idx:
        r = rows[i]
        p = tokenizer(r["prompt"], add_special_tokens=False)["input_ids"]
        c = tokenizer(r["chosen"], add_special_tokens=False)["input_ids"]
        j = tokenizer(r["rejected"], add_special_tokens=False)["input_ids"]
        prompt_lens.append(len(p))
        comp_lens.append(max(len(c), len(j)))
        # 保守估计：prompt + 较长的那个 completion
        total_lens.append(len(p) + max(len(c), len(j)))

    name = getattr(tokenizer, "name_or_path", "unknown")
    return TokenLengthStats(
        num_samples=len(idx),
        prompt_p50=int(percentile(prompt_lens, 50)),
        prompt_p90=int(percentile(prompt_lens, 90)),
        prompt_p99=int(percentile(prompt_lens, 99)),
        prompt_max=max(prompt_lens) if prompt_lens else 0,
        total_p50=int(percentile(total_lens, 50)),
        total_p90=int(percentile(total_lens, 90)),
        total_p99=int(percentile(total_lens, 99)),
        total_max=max(total_lens) if total_lens else 0,
        completion_p50=int(percentile(comp_lens, 50)),
        completion_p99=int(percentile(comp_lens, 99)),
        completion_max=max(comp_lens) if comp_lens else 0,
        tokenizer_name=str(name),
    )


def truncation_report(
    rows: Sequence[dict[str, Any]],
    tokenizer: Any,
    max_length: int,
    max_prompt_length: int,
    *,
    sample_limit: int = 400,
) -> dict[str, Any]:
    """估算被截断的样本比例，以及「截断后 chosen 与 rejected 是否变得相同」。

    最后这一项很关键：如果截断后两侧文本一样，这条样本对 DPO
    来说就是纯噪声（loss 恒为 log 2，梯度无信息）。
    数量多的话必须调大 max_length 或过滤长样本。
    """
    import random

    rng = random.Random(42)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    idx = idx[:sample_limit]

    prompt_truncated = 0
    total_truncated = 0
    degenerate = 0  # 截断后 chosen == rejected

    for i in idx:
        r = rows[i]
        p = tokenizer(r["prompt"], add_special_tokens=False)["input_ids"]
        c = tokenizer(r["chosen"], add_special_tokens=False)["input_ids"]
        j = tokenizer(r["rejected"], add_special_tokens=False)["input_ids"]

        p_trunc = p[:max_prompt_length]
        if len(p) > max_prompt_length:
            prompt_truncated += 1

        budget = max(1, max_length - len(p_trunc))
        c_trunc = c[:budget]
        j_trunc = j[:budget]
        if len(c) > budget or len(j) > budget:
            total_truncated += 1
        if c_trunc == j_trunc:
            degenerate += 1

    n = max(1, len(idx))
    return {
        "num_sampled": len(idx),
        "max_length": max_length,
        "max_prompt_length": max_prompt_length,
        "prompt_truncation_rate": round(prompt_truncated / n, 4),
        "completion_truncation_rate": round(total_truncated / n, 4),
        "degenerate_after_truncation_rate": round(degenerate / n, 4),
        "note": (
            "degenerate_after_truncation_rate 是「截断后 chosen 与 rejected 变得完全相同」"
            "的比例 —— 这些样本对 DPO 是纯噪声。若 > 5% 请调大 max_length 或过滤长样本。"
        ),
    }


# ============================================================================
# prompt 一致性校验
# ============================================================================


def verify_prompt_consistency(rows: Sequence[dict[str, Any]], tasks: Sequence[Any]) -> dict[str, Any]:
    """校验偏好数据里的 prompt 与评测时构造的 prompt 同源。

    ⚠️ **这个函数原先有两个会给出「假通过」的缺陷，已修：**

    1. **不带 ``task_id`` 的样本被完全跳过。** 原实现是
       ``if task is None: continue`` —— 于是一份**一条 task_id 都没有**的
       偏好数据会被判为「一致」，而它实际上**一条都没被检查**。
       实测中 ``sft_only.train.jsonl`` 的 2435 条全都没有 task_id，
       报告却打印「检查 164 条 ✓ 一致」—— 那 164 条是另一批数据。
       现在改为：无法用 task_id 比对的样本，退化为
       **「system prompt 是否与评测一致」** 的结构检查，并把数量单独报告。

    2. **只比对 ``[:400]``，覆盖不到工具清单。** 原实现拿评测 prompt 的
       前 400 字符当锚点。可 system prompt 现在约 900+ 字符，
       **工具定义部分完全在 400 字符之后，从来没被校验过**。
       而工具清单正是模型判断该调哪个工具的依据 —— 这里不一致会让
       模型学到错误的工具语义。现在改为比对**整个 system prompt**
       （用 ``<|im_end|>`` 定位边界），这才是真正需要同源的部分。

    做法：对每个样本，取出它的 system prompt 段，与评测时
    ``build_plain_prompt`` 渲染出的 system prompt 段比对。

    Returns:
        ``{"checked", "num_mismatch", "consistent", "num_uncheckable", "examples"}``
    """
    from src.models.prompt_builder import build_plain_prompt

    by_id = {t.task_id: t for t in tasks}
    checked = 0
    num_uncheckable = 0
    num_mismatch = 0
    mismatches: list[dict[str, Any]] = []

    # 评测侧的 system prompt 是唯一的（不含 task 相关内容），
    # 取任意一个任务渲染出来作为基准即可。这里取第一个任务，
    # 同时保留一个「按 task_id 精确比对」的通道。
    ref_system = ""
    if tasks:
        ref_system = _extract_system_segment(build_plain_prompt(tasks[0]))

    for r in rows:
        prompt = r.get("prompt", "")
        actual_system = _extract_system_segment(prompt)

        tid = r.get("task_id", "")
        task = by_id.get(tid)

        if task is not None:
            # 有 task_id：用该任务的完整 system prompt 比对
            expected_system = _extract_system_segment(build_plain_prompt(task))
            ok = actual_system == expected_system
        elif ref_system:
            # 没有 task_id：退化为「system prompt 是否与评测一致」
            expected_system = ref_system
            ok = actual_system == expected_system
        else:
            num_uncheckable += 1
            continue

        checked += 1
        if not ok:
            num_mismatch += 1
            if len(mismatches) < 5:
                # 找到第一个差异位置，便于定位
                a, b = actual_system, expected_system
                diff_at = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
                mismatches.append(
                    {
                        "task_id": tid or "(无 task_id)",
                        "pair_type": r.get("pair_type", ""),
                        "diff_at": diff_at,
                        "expected_head": b[max(0, diff_at - 60):diff_at + 100],
                        "actual_head": a[max(0, diff_at - 60):diff_at + 100],
                    }
                )

    return {
        "checked": checked,
        "num_mismatch": num_mismatch,
        "consistent": num_mismatch == 0,
        "num_uncheckable": num_uncheckable,
        "examples": mismatches,
    }


def _extract_system_segment(prompt: str) -> str:
    """从渲染好的 prompt 里取出 system 段的内容。

    渲染格式固定为 ``<|im_start|>system\\n{内容}<|im_end|>``，
    这里把 ``{内容}`` 取出来。取不到时返回空串（会被判为不一致）。

    >>> _extract_system_segment("<|im_start|>system\\nSYS<|im_end|>\\n<|im_start|>user\\nU<|im_end|>")
    'SYS'
    >>> _extract_system_segment("格式不对的字符串")
    ''
    """
    marker = "<|im_start|>system\n"
    if not prompt.startswith(marker):
        return ""
    rest = prompt[len(marker):]
    end = rest.find("<|im_end|>")
    return rest[:end] if end >= 0 else ""


# ============================================================================
# 数据集统计
# ============================================================================


@dataclass
class PreferenceDataStats:
    """偏好数据的构成统计。"""

    total: int = 0
    by_pair_type: dict[str, int] = field(default_factory=dict)
    by_negative_kind: dict[str, int] = field(default_factory=dict)
    by_split: dict[str, int] = field(default_factory=dict)
    by_target_error: dict[str, int] = field(default_factory=dict)
    #: chosen/rejected 的平均字符长度
    avg_chosen_chars: float = 0.0
    avg_rejected_chars: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "by_pair_type": self.by_pair_type,
            "by_negative_kind": self.by_negative_kind,
            "by_split": self.by_split,
            "by_target_error": self.by_target_error,
            "avg_chosen_chars": round(self.avg_chosen_chars, 1),
            "avg_rejected_chars": round(self.avg_rejected_chars, 1),
        }

    def summary_lines(self) -> list[str]:
        lines = [f"偏好数据 {self.total} 条"]
        for label, d in (
            ("类型", self.by_pair_type),
            ("负样本", self.by_negative_kind),
            ("划分", self.by_split),
            ("目标错误", self.by_target_error),
        ):
            lines.append(f"  按{label}: " + ", ".join(f"{k}={v}" for k, v in sorted(d.items())))
        lines.append(f"  平均长度: chosen={self.avg_chosen_chars:.0f} 字符, rejected={self.avg_rejected_chars:.0f} 字符")
        return lines


def preference_data_stats(rows: Iterable[dict[str, Any]]) -> PreferenceDataStats:
    rows = list(rows)
    st = PreferenceDataStats(total=len(rows))
    for r in rows:
        for field_name, bucket in (
            ("pair_type", st.by_pair_type),
            ("negative_kind", st.by_negative_kind),
            ("split", st.by_split),
            ("target_error", st.by_target_error),
        ):
            v = str(r.get(field_name, "unknown"))
            bucket[v] = bucket.get(v, 0) + 1
    if rows:
        st.avg_chosen_chars = sum(len(r["chosen"]) for r in rows) / len(rows)
        st.avg_rejected_chars = sum(len(r["rejected"]) for r in rows) / len(rows)
    return st
