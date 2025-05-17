from abc import ABC, abstractmethod
from typing import Any, Optional


class Metric(ABC):
    name: str
    higher_is_better: bool

    @abstractmethod
    def score(
        self, response: str, reference: Optional[str] = None, **kwargs: Any
    ) -> float:
        pass
