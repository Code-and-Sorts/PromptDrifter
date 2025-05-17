from typing import Dict, Type

from .base import Metric

_METRIC_REGISTRY: Dict[str, Type[Metric]] = {}


def register_metric(metric_class: Type[Metric]):
    if not issubclass(metric_class, Metric):
        raise TypeError("Registered metric must be a subclass of Metric.")
    _METRIC_REGISTRY[metric_class.name] = metric_class


def get_metric(name: str) -> Type[Metric]:
    metric_class = _METRIC_REGISTRY.get(name)
    if not metric_class:
        raise ValueError(
            f"Metric '{name}' not found. Available: {list(_METRIC_REGISTRY.keys())}"
        )
    return metric_class


def get_all_metrics() -> Dict[str, Type[Metric]]:
    return _METRIC_REGISTRY.copy()
