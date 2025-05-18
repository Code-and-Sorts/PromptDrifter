from unittest.mock import patch

import pytest

from promptdrifter.metrics.token_overlap import TokenOverlapMetric


@pytest.fixture
def metric_instance():
    return TokenOverlapMetric()


MODULE_UNDER_TEST = "promptdrifter.metrics.token_overlap"


@patch(f"{MODULE_UNDER_TEST}.rapidfuzz.fuzz.token_set_ratio")
def test_score_rapidfuzz_identical(mock_rf_token_set_ratio, metric_instance):
    mock_rf_token_set_ratio.return_value = 100.0
    assert metric_instance.score("hello world", "hello world") == 1.0
    mock_rf_token_set_ratio.assert_called_once_with("hello world", "hello world")
    mock_rf_token_set_ratio.reset_mock()
    assert metric_instance.score("", "") == 1.0
    mock_rf_token_set_ratio.assert_not_called()


@patch(f"{MODULE_UNDER_TEST}.rapidfuzz.fuzz.token_set_ratio")
def test_score_rapidfuzz_disjoint(mock_rf_token_set_ratio, metric_instance):
    mock_rf_token_set_ratio.return_value = 0.0
    assert metric_instance.score("hello world", "goodbye moon") == 0.0
    mock_rf_token_set_ratio.assert_called_once_with("hello world", "goodbye moon")
    mock_rf_token_set_ratio.reset_mock()

    mock_rf_token_set_ratio.return_value = 0.0
    assert metric_instance.score("", "world") == 0.0
    mock_rf_token_set_ratio.assert_called_once_with("", "world")
    mock_rf_token_set_ratio.reset_mock()

    mock_rf_token_set_ratio.return_value = 0.0
    assert metric_instance.score("hello", "") == 0.0
    mock_rf_token_set_ratio.assert_called_once_with("hello", "")


@patch(f"{MODULE_UNDER_TEST}.rapidfuzz.fuzz.token_set_ratio")
def test_score_rapidfuzz_partial_overlap(mock_rf_token_set_ratio, metric_instance):
    mock_rf_token_set_ratio.return_value = 33.0
    assert metric_instance.score("hello world", "hello moon") == pytest.approx(0.33)
    mock_rf_token_set_ratio.assert_called_once_with("hello world", "hello moon")
    mock_rf_token_set_ratio.reset_mock()

    mock_rf_token_set_ratio.return_value = 25.0
    assert metric_instance.score("apple banana cherry", "apple grape") == pytest.approx(
        0.25
    )
    mock_rf_token_set_ratio.assert_called_once_with(
        "apple banana cherry", "apple grape"
    )


@patch(f"{MODULE_UNDER_TEST}.rapidfuzz.fuzz.token_set_ratio")
def test_score_value_error_no_reference(mock_rf_token_set_ratio, metric_instance):
    with pytest.raises(
        ValueError, match="TokenOverlapMetric requires a reference string."
    ):
        metric_instance.score("hello world")
    mock_rf_token_set_ratio.assert_not_called()


@patch(f"{MODULE_UNDER_TEST}.rapidfuzz.fuzz.token_set_ratio")
def test_score_rapidfuzz_handles_preprocessing(
    mock_rf_token_set_ratio, metric_instance
):
    mock_rf_token_set_ratio.return_value = 100.0
    response_str = "Hello, World!"
    reference_str = "hello world"
    assert metric_instance.score(response_str, reference_str) == 1.0
    mock_rf_token_set_ratio.assert_called_once_with(response_str, reference_str)
