from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch  # For extensive mocking

import pytest
from rich.console import Console  # To potentially capture or mock console output

from promptdrifter.adapters.base import Adapter
from promptdrifter.cache import PromptCache
from promptdrifter.runner import ADAPTER_REGISTRY, Runner
from promptdrifter.yaml_loader import YamlFileLoader

# Mark all tests in this file as asyncio
pytestmark = pytest.mark.asyncio

# --- Mocks and Fixtures ---


@pytest.fixture
def mock_yaml_loader(mocker) -> MagicMock:
    return mocker.MagicMock(spec=YamlFileLoader)


@pytest.fixture
def mock_cache(mocker) -> MagicMock:
    mock = mocker.MagicMock(spec=PromptCache)
    mock.get = MagicMock(return_value=None)  # Default to cache miss
    mock.put = MagicMock()
    return mock


@pytest.fixture
def mock_console(mocker) -> MagicMock:
    return mocker.MagicMock(spec=Console)


@pytest.fixture
def mock_adapter_instance() -> AsyncMock:  # Returns an instance of a mock adapter
    adapter_mock = AsyncMock(spec=Adapter)
    adapter_mock.execute = AsyncMock()
    adapter_mock.close = AsyncMock()
    return adapter_mock


@pytest.fixture
def runner_dependencies_setup(
    mocker, mock_yaml_loader, mock_cache, mock_console, mock_adapter_instance
):
    """Fixture to provide all mocked dependencies for the Runner, replacing class methods/constructors."""
    mocker.patch("promptdrifter.runner.YamlFileLoader", return_value=mock_yaml_loader)
    mocker.patch("promptdrifter.runner.PromptCache", return_value=mock_cache)
    mocker.patch("promptdrifter.runner.Console", return_value=mock_console)

    # This will be the mock object for the _get_adapter_instance method
    mock_get_adapter_method = MagicMock(return_value=mock_adapter_instance)

    return {
        "yaml_loader": mock_yaml_loader,
        "cache": mock_cache,
        "console": mock_console,
        "adapter_instance": mock_adapter_instance,
        "get_adapter_method_mock": mock_get_adapter_method,  # Provide the function mock itself
    }


@pytest.fixture
def test_runner(runner_dependencies_setup, tmp_path) -> Runner:
    runner = Runner(config_dir=tmp_path, use_cache=True)
    # Directly replace the _get_adapter_instance method on this specific runner instance
    runner._get_adapter_instance = runner_dependencies_setup["get_adapter_method_mock"]
    return runner


# --- Test Cases ---


async def test_run_single_test_case_pass_exact_match(
    test_runner: Runner, runner_dependencies_setup
):
    mock_adapter = runner_dependencies_setup["adapter_instance"]
    mock_cache = runner_dependencies_setup["cache"]

    test_file = Path("test_exact.yaml")
    test_data = {
        "id": "exact-pass-001",
        "prompt": "Say hello",
        "adapter": [{
            "type": "mocked_adapter",
            "model": "test_model"
        }],
        "expect_exact": "Hello there",
    }
    mock_adapter.execute.return_value = {
        "text_response": "Hello there",
        "raw_response": {},
    }
    mock_cache.get.return_value = None  # Cache miss

    result = await test_runner._run_single_test_case(test_file, test_data)

    assert result["status"] == "PASS"
    test_runner._get_adapter_instance.assert_called_with("mocked_adapter")
    mock_adapter.execute.assert_called_once_with("Say hello", model="test_model")
    mock_cache.put.assert_called_once()
    mock_adapter.close.assert_called_once()


async def test_run_single_test_case_fail_exact_match(
    test_runner: Runner, runner_dependencies_setup
):
    mock_adapter = runner_dependencies_setup["adapter_instance"]
    mock_adapter.execute.return_value = {"text_response": "Goodbye", "raw_response": {}}

    test_file = Path("test_exact_fail.yaml")
    test_data = {
        "id": "exact-fail-001",
        "prompt": "Say hello",
        "adapter": [{
            "type": "mocked_adapter"
        }],
        "expect_exact": "Hello",
    }

    result = await test_runner._run_single_test_case(test_file, test_data)
    assert result["status"] == "FAIL"
    assert "Exact match failed" in result["reason"]
    assert test_runner.overall_success is False
    mock_adapter.close.assert_called_once()


async def test_run_single_test_case_pass_regex_match(
    test_runner: Runner, runner_dependencies_setup
):
    mock_adapter = runner_dependencies_setup["adapter_instance"]
    mock_adapter.execute.return_value = {
        "text_response": "The number is 42.",
        "raw_response": {},
    }

    test_file = Path("test_regex.yaml")
    test_data = {
        "id": "regex-pass-001",
        "prompt": "What number?",
        "adapter": [{
            "type": "mocked_adapter"
        }],
        "expect_regex": r"number is \d+",
    }

    result = await test_runner._run_single_test_case(test_file, test_data)
    assert result["status"] == "PASS"
    mock_adapter.close.assert_called_once()


async def test_run_single_test_case_cache_hit(
    test_runner: Runner, runner_dependencies_setup
):
    mock_cache = runner_dependencies_setup["cache"]
    mock_adapter = runner_dependencies_setup["adapter_instance"]

    test_file = Path("test_cache_hit.yaml")
    test_data = {
        "id": "cache-hit-001",
        "prompt": "Cached prompt",
        "adapter": [{
            "type": "cached_adapter"
        }],
        "expect_exact": "Cached response",
    }
    cached_llm_response = {
        "text_response": "Cached response",
        "raw_response": {"from_cache": True},
    }

    mock_cache.get.return_value = cached_llm_response

    result = await test_runner._run_single_test_case(test_file, test_data)

    assert result["status"] == "PASS"
    assert result["cache_status"] == "HIT"
    mock_adapter.execute.assert_not_called()
    mock_cache.get.assert_called_once()
    mock_adapter.close.assert_called_once()


async def test_run_single_test_case_adapter_error(
    test_runner: Runner, runner_dependencies_setup
):
    mock_adapter = runner_dependencies_setup["adapter_instance"]
    mock_adapter.execute.return_value = {
        "error": "Something went wrong with LLM",
        "raw_response": {},
    }

    test_file = Path("test_adapter_err.yaml")
    test_data = {
        "id": "adapter-err-001",
        "prompt": "Trigger error",
        "adapter": [{
            "type": "error_adapter"
        }],
        "expect_exact": "N/A",
    }

    result = await test_runner._run_single_test_case(test_file, test_data)
    assert result["status"] == "ERROR"
    assert "Adapter error: Something went wrong with LLM" in result["reason"]
    mock_adapter.close.assert_called_once()


async def test_run_single_test_case_no_text_response(
    test_runner: Runner, runner_dependencies_setup
):
    mock_adapter = runner_dependencies_setup["adapter_instance"]
    mock_adapter.execute.return_value = {"text_response": None, "raw_response": {}}

    test_file = Path("test_no_text.yaml")
    test_data = {
        "id": "no-text-001",
        "prompt": "No text",
        "adapter": [{
            "type": "no_text_adapter"
        }],
        "expect_exact": "Something",
    }

    result = await test_runner._run_single_test_case(test_file, test_data)
    assert result["status"] == "FAIL"
    assert "Adapter returned no text_response" in result["reason"]
    mock_adapter.close.assert_called_once()


async def test_run_single_test_case_skipped_no_prompt(
    test_runner: Runner, runner_dependencies_setup
):
    test_file = Path("test_skip_prompt.yaml")
    test_data = {"id": "skip-prompt", "adapter": [{"type": "any"}], "expect_exact": "any"}
    result = await test_runner._run_single_test_case(test_file, test_data)
    assert result["status"] == "SKIPPED"
    assert result["reason"] == "No prompt defined."


async def test_run_single_test_case_skipped_no_adapter(
    test_runner: Runner, runner_dependencies_setup
):
    test_file = Path("test_skip_adapter.yaml")
    test_data = {"id": "skip-adapter", "prompt": "A prompt", "expect_exact": "any"}
    result = await test_runner._run_single_test_case(test_file, test_data)
    assert result["status"] == "SKIPPED"
    assert result["reason"] == "No adapter specified."


async def test_run_single_test_case_skipped_no_assertion(
    test_runner: Runner, runner_dependencies_setup
):
    test_file = Path("test_skip_assertion.yaml")
    test_data = {"id": "skip-assertion", "prompt": "A prompt", "adapter": [{"type": "any"}]}
    result = await test_runner._run_single_test_case(test_file, test_data)
    assert result["status"] == "SKIPPED"
    assert result["reason"] == (
        "No assertion (expect_exact, expect_regex, expect_substring, or expect_substring_case_insensitive) defined."
    )


async def test_run_single_test_case_unknown_adapter(
    test_runner: Runner, runner_dependencies_setup
):
    runner_dependencies_setup["get_adapter_method_mock"].return_value = None
    test_file = Path("test_unknown_adapter.yaml")
    test_data = {
        "id": "unknown-adapter",
        "prompt": "Hello",
        "adapter": [{
            "type": "non_existent_adapter"
        }],
        "expect_exact": "Hi",
    }
    result = await test_runner._run_single_test_case(test_file, test_data)
    assert result["status"] == "SKIPPED"
    assert (
        "Adapter 'non_existent_adapter' could not be initialized or found."
        in result["reason"]
    )
    test_runner._get_adapter_instance.assert_called_with("non_existent_adapter")


async def test_run_suite_overall_success_and_failure(
    test_runner: Runner, runner_dependencies_setup, tmp_path
):
    mock_yaml_loader = runner_dependencies_setup["yaml_loader"]
    mock_adapter = runner_dependencies_setup["adapter_instance"]
    mock_console = runner_dependencies_setup["console"]

    test_file_2 = tmp_path / "test2.yaml"
    test_file_2.write_text(
        "id: suite-fail-1\nprompt: P2\nadapter:\n  - type: mocked\nexpect_exact: R2-expected"
    )
    test_data_2 = {
        "id": "suite-fail-1",
        "prompt": "P2",
        "adapter": [{
            "type": "mocked"
        }],
        "expect_exact": "R2-expected",
    }

    mock_yaml_loader.load_and_validate_yaml.return_value = test_data_2
    mock_yaml_loader.load_and_validate_yaml.side_effect = None

    mock_adapter.execute.return_value = {
        "text_response": "R2-actual",
        "raw_response": {},
    }
    mock_adapter.execute.side_effect = None

    success = await test_runner.run_suite([test_file_2])

    assert not success
    assert len(test_runner.results) == 1
    assert test_runner.results[0]["status"] == "FAIL"
    mock_console.print.assert_any_call(
        "\n[bold red]Some tests failed or encountered errors.[/bold red]"
    )
    assert mock_adapter.close.call_count == 1


async def test_run_suite_yaml_error(
    test_runner: Runner, runner_dependencies_setup, tmp_path
):
    mock_yaml_loader = runner_dependencies_setup["yaml_loader"]
    mock_console = runner_dependencies_setup["console"]

    test_file_err = tmp_path / "error.yaml"
    test_file_err.write_text("invalid_yaml_content_for_test_purpose_only")
    mock_yaml_loader.load_and_validate_yaml.side_effect = ValueError("Bad YAML format")

    success = await test_runner.run_suite([test_file_err])
    assert not success
    mock_console.print.assert_any_call(
        f"[bold red]Error processing {test_file_err}: Bad YAML format[/bold red]"
    )
