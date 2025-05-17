from typing import Any, Optional

import rapidfuzz.fuzz

from .base import Metric
from .core import register_metric


class TokenOverlapMetric(Metric):
    name = "token_overlap"
    higher_is_better = True

    def __init__(self):
        pass

    def score(
        self, response: str, reference: Optional[str] = None, **kwargs: Any
    ) -> float:
        if reference is None:
            raise ValueError("TokenOverlapMetric requires a reference string.")

        if not response and not reference:
            return 1.0

        return rapidfuzz.fuzz.token_set_ratio(response, reference) / 100.0


register_metric(TokenOverlapMetric)
