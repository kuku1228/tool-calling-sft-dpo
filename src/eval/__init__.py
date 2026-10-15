"""评测层：解析、等价判定、错误分类、评分、指标聚合。

对外主入口
----------
>>> from src.eval.grader import grade, run_evaluation, build_eval_report
>>> from src.eval.metrics import EvalReport, pass_at_k_unbiased, pass_hat_k_unbiased
"""

from __future__ import annotations

from src.eval.equivalence import equivalent, match_arguments
from src.eval.error_taxonomy import (
    ALL_CATEGORIES,
    CATEGORY_LABELS_ZH,
    FAILURE_CATEGORIES,
    Classification,
    ErrorDistribution,
    classify,
)
from src.eval.metrics import (
    EvalReport,
    MetricValue,
    SampledTaskResult,
    cohens_h,
    pass_at_k_unbiased,
    pass_hat_k_unbiased,
    required_sample_size,
    wilson_interval,
)
from src.eval.tool_call_parser import (
    ToolCall,
    canonical_call,
    extract_tool_calls,
    format_accuracy,
    is_canonical_format,
)

__all__ = [
    "ALL_CATEGORIES",
    "CATEGORY_LABELS_ZH",
    "FAILURE_CATEGORIES",
    "Classification",
    "ErrorDistribution",
    "EvalReport",
    "MetricValue",
    "SampledTaskResult",
    "ToolCall",
    "canonical_call",
    "classify",
    "cohens_h",
    "equivalent",
    "extract_tool_calls",
    "format_accuracy",
    "is_canonical_format",
    "match_arguments",
    "pass_at_k_unbiased",
    "pass_hat_k_unbiased",
    "required_sample_size",
    "wilson_interval",
]
