from typing import Any, Optional

import pytest

from promptdrifter.metrics.base import Metric
from promptdrifter.metrics.core import (
    _METRIC_REGISTRY,
    get_all_metrics,
    get_metric,
    register_metric,
)


class MockMetric(Metric):
    name = "mock_metric"
    higher_is_better = True

    def score(self, response: str, reference: Optional[str] = None, **kwargs: Any) -> float:
        return 1.0


class AnotherMockMetric(Metric):
    name = "another_mock_metric"
    higher_is_better = False

    def score(self, response: str, reference: Optional[str] = None, **kwargs: Any) -> float:
        return 0.5


class DuplicateNameMetric(Metric):
    name = "mock_metric"
    higher_is_better = False

    def score(self, response: str, reference: Optional[str] = None, **kwargs: Any) -> float:
        return 0.75


class NotAMetric:
    name = "not_a_metric"


@pytest.fixture
def clean_registry():
    original_registry = _METRIC_REGISTRY.copy()
    _METRIC_REGISTRY.clear()
    yield
    _METRIC_REGISTRY.clear()
    _METRIC_REGISTRY.update(original_registry)


def test_register_metric(clean_registry):
    assert len(_METRIC_REGISTRY) == 0

    register_metric(MockMetric)
    assert len(_METRIC_REGISTRY) == 1
    assert "mock_metric" in _METRIC_REGISTRY
    assert _METRIC_REGISTRY["mock_metric"] == MockMetric


def test_register_multiple_metrics(clean_registry):
    register_metric(MockMetric)
    register_metric(AnotherMockMetric)

    assert len(_METRIC_REGISTRY) == 2
    assert "mock_metric" in _METRIC_REGISTRY
    assert "another_mock_metric" in _METRIC_REGISTRY


def test_register_metric_overwrites_existing(clean_registry):
    register_metric(MockMetric)
    assert _METRIC_REGISTRY["mock_metric"] == MockMetric

    register_metric(DuplicateNameMetric)

    assert len(_METRIC_REGISTRY) == 1
    assert _METRIC_REGISTRY["mock_metric"] == DuplicateNameMetric


def test_register_non_metric_class(clean_registry):
    with pytest.raises(TypeError, match="Registered metric must be a subclass of Metric"):
        register_metric(NotAMetric)


def test_get_metric(clean_registry):
    register_metric(MockMetric)
    register_metric(AnotherMockMetric)

    mock_metric_class = get_metric("mock_metric")
    assert mock_metric_class == MockMetric

    another_metric_class = get_metric("another_mock_metric")
    assert another_metric_class == AnotherMockMetric


def test_get_nonexistent_metric(clean_registry):
    with pytest.raises(ValueError, match="Metric 'nonexistent' not found"):
        get_metric("nonexistent")


def test_get_nonexistent_metric_shows_available(clean_registry):
    register_metric(MockMetric)

    with pytest.raises(ValueError) as excinfo:
        get_metric("nonexistent")

    assert "mock_metric" in str(excinfo.value)


def test_get_all_metrics(clean_registry):
    register_metric(MockMetric)
    register_metric(AnotherMockMetric)

    all_metrics = get_all_metrics()
    assert len(all_metrics) == 2
    assert "mock_metric" in all_metrics
    assert "another_mock_metric" in all_metrics
    assert all_metrics["mock_metric"] == MockMetric
    assert all_metrics["another_mock_metric"] == AnotherMockMetric


def test_get_all_metrics_empty(clean_registry):
    all_metrics = get_all_metrics()
    assert isinstance(all_metrics, dict)
    assert len(all_metrics) == 0


def test_get_all_metrics_is_copy(clean_registry):
    register_metric(MockMetric)

    all_metrics = get_all_metrics()
    assert all_metrics is not _METRIC_REGISTRY

    all_metrics.clear()
    assert len(_METRIC_REGISTRY) == 1
    assert "mock_metric" in _METRIC_REGISTRY
