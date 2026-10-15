"""Mock 层：无需 GPU 即可端到端验证流水线。"""

from __future__ import annotations

from src.mock.mock_policy import (
    EXPERIMENT_LABELS_ZH,
    EXPERIMENTS,
    MockPolicy,
    make_policy,
)

__all__ = ["EXPERIMENTS", "EXPERIMENT_LABELS_ZH", "MockPolicy", "make_policy"]
