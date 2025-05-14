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

    results_list = await test_runner._run_single_test_case(test_file, test_data)
    assert len(results_list) == 1
    result = results_list[0]

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

    results_list = await test_runner._run_single_test_case(test_file, test_data)
    assert len(results_list) == 1
    result = results_list[0]

    assert result["status"] == "FAIL"
    assert "Exact match failed" in result["reason"]
    assert test_runner.overall_success is False
    mock_adapter.close.assert_called_once()


async def test_run_single_test_case_pass_regex_match(
    test_runner: Runner, runner_dependencies_setup
):
    mock_adapter = runner_dependencies_setup["adapter_instance"]
    test_runner.overall_success = True 
    mock_adapter.execute.return_value = {
        "text_response": "The number is 42.",
        "raw_response": {},
    }

    test_file = Path("test_regex.yaml")
    test_data = {
        "id": "regex-pass-001",
        "prompt": "What number?",
        "adapter": [{
            "type": "mocked_adapter", "model": "regex_model"
        }],
        "expect_regex": "number is \\d+", # Corrected: No r prefix here, just the string for regex engine
    }

    results_list = await test_runner._run_single_test_case(test_file, test_data)
    assert len(results_list) == 1
    result = results_list[0]

    assert result["status"] == "PASS" # This will fail if the regex issue persists
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
            "type": "cached_adapter", "model": "cached_model"
        }],
        "expect_exact": "Cached response",
    }
    cached_llm_response = {
        "text_response": "Cached response",
        "raw_response": {"from_cache": True},
    }

    mock_cache.get.return_value = cached_llm_response
    # Define expected cache key for adapter_options
    # In _run_single_test_case, adapter_options for this test would be an empty dict
    # as temperature and max_tokens are None and no other kwargs.
    expected_cache_options_key = frozenset({})


    results_list = await test_runner._run_single_test_case(test_file, test_data)
    assert len(results_list) == 1
    result = results_list[0]


    assert result["status"] == "PASS"
    assert result["cache_status"] == "HIT"
    mock_adapter.execute.assert_not_called()
    mock_cache.get.assert_called_once_with("Cached prompt", "cached_adapter", "cached_model", expected_cache_options_key)
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
        "expect_exact": "This won't be checked",
    }

    results_list = await test_runner._run_single_test_case(test_file, test_data)
    assert len(results_list) == 1
    result = results_list[0]

    assert result["status"] == "ERROR"
    assert "Adapter error: Something went wrong with LLM" in result["reason"]
    assert test_runner.overall_success is False
    mock_adapter.close.assert_called_once()


async def test_run_single_test_case_execution_exception(
    test_runner: Runner, runner_dependencies_setup
):
    mock_adapter = runner_dependencies_setup["adapter_instance"]
    mock_adapter.execute.side_effect = Exception("Network issue")

    test_file = Path("test_exec_exception.yaml")
    test_data = {
        "id": "exec-exception-001",
        "prompt": "Causes exception",
        "adapter": [{
            "type": "exception_adapter"
        }],
        "expect_exact": "Not checked",
    }

    results_list = await test_runner._run_single_test_case(test_file, test_data)
    assert len(results_list) == 1
    result = results_list[0]

    assert result["status"] == "ERROR"
    assert "Adapter execution error: Network issue" in result["reason"]
    assert test_runner.overall_success is False
    mock_adapter.close.assert_called_once()


async def test_run_single_test_case_no_text_response(
    test_runner: Runner, runner_dependencies_setup
):
    mock_adapter = runner_dependencies_setup["adapter_instance"]
    mock_adapter.execute.return_value = {"raw_response": {}} # No text_response

    test_file = Path("test_no_text.yaml")
    test_data = {
        "id": "no-text-001",
        "prompt": "No text",
        "adapter": [{
            "type": "no_text_adapter"
        }],
        "expect_exact": "Not checked",
    }

    results_list = await test_runner._run_single_test_case(test_file, test_data)
    assert len(results_list) == 1
    result = results_list[0]

    assert result["status"] == "FAIL"
    assert "Adapter returned no text_response" in result["reason"]
    assert test_runner.overall_success is False
    mock_adapter.close.assert_called_once()


async def test_run_single_test_case_skipped_no_prompt(
    test_runner: Runner, runner_dependencies_setup
):
    test_file = Path("test_skip_no_prompt.yaml")
    test_data = {"id": "skip-no-prompt", "adapter": [{"type": "any"}]} # No prompt

    results_list = await test_runner._run_single_test_case(test_file, test_data)
    assert len(results_list) == 1
    result = results_list[0]
    assert result["status"] == "SKIPPED"
    assert "No prompt defined" in result["reason"]


async def test_run_single_test_case_skipped_no_adapter(
    test_runner: Runner, runner_dependencies_setup
):
    test_file = Path("test_skip_no_adapter.yaml")
    test_data = {"id": "skip-no-adapter", "prompt": "A prompt"} # No adapter list

    results_list = await test_runner._run_single_test_case(test_file, test_data)
    assert len(results_list) == 1
    result = results_list[0]
    assert result["status"] == "SKIPPED"
    assert "No adapter specified" in result["reason"]


async def test_run_single_test_case_skipped_no_assertion(
    test_runner: Runner, runner_dependencies_setup
):
    test_file = Path("test_skip_no_assertion.yaml")
    test_data = { # No expect_* key
        "id": "skip-no-assertion",
        "prompt": "A prompt",
        "adapter": [{"type": "any_adapter"}]
    }
    results_list = await test_runner._run_single_test_case(test_file, test_data)
    assert len(results_list) == 1
    result = results_list[0]
    assert result["status"] == "SKIPPED"
    assert "No assertion defined" in result["reason"]


async def test_run_single_test_case_unknown_adapter(
    test_runner: Runner, runner_dependencies_setup
):
    # Ensure _get_adapter_instance returns None for this specific adapter name
    get_adapter_mock = runner_dependencies_setup["get_adapter_method_mock"]
    
    def side_effect_for_get_adapter(adapter_name_called):
        if adapter_name_called == "unknown_adapter_type":
            return None # Simulate adapter not found/failed init
        return runner_dependencies_setup["adapter_instance"] # Return standard mock for other calls if any

    get_adapter_mock.side_effect = side_effect_for_get_adapter

    test_file = Path("test_unknown_adapter.yaml")
    test_data = {
        "id": "unknown-adapter-001",
        "prompt": "Test prompt",
        "adapter": [{"type": "unknown_adapter_type", "model": "some_model"}],
        "expect_exact": "anything"
    }

    results_list = await test_runner._run_single_test_case(test_file, test_data)
    assert len(results_list) == 1
    result = results_list[0]

    assert result["status"] == "ERROR"
    assert "Adapter 'unknown_adapter_type' could not be initialized or found" in result["reason"]
    assert test_runner.overall_success is False
    # Reset side effect if other tests might use the same mock from fixture
    get_adapter_mock.side_effect = None 
    get_adapter_mock.return_value = runner_dependencies_setup["adapter_instance"]


async def test_run_single_test_case_with_adapter_options(
    test_runner: Runner, runner_dependencies_setup
):
    mock_adapter = runner_dependencies_setup["adapter_instance"]
    test_file = Path("test_adapter_options.yaml")
    test_data = {
        "id": "adapter-options-001",
        "prompt": "Prompt with options",
        "adapter": [{
            "type": "options_adapter",
            "model": "options_model",
            "temperature": 0.77,
            "max_tokens": 123,
            "custom_param": "custom_value"
        }],
        "expect_exact": "Response",
    }
    mock_adapter.execute.return_value = {"text_response": "Response", "raw_response": {}}
    expected_cache_options_key = frozenset({
        ("temperature", 0.77),
        ("max_tokens", 123),
        ("custom_param", "custom_value")
    })


    results_list = await test_runner._run_single_test_case(test_file, test_data)
    assert len(results_list) == 1
    result = results_list[0]

    assert result["status"] == "PASS"
    mock_adapter.execute.assert_called_once_with(
        "Prompt with options",
        model="options_model",
        temperature=0.77,
        max_tokens=123,
        custom_param="custom_value"
    )
    # Check cache put call arguments
    runner_dependencies_setup["cache"].put.assert_called_once()
    _, cache_put_args, _ = runner_dependencies_setup["cache"].put.mock_calls[0] # (name, args, kwargs)
    assert cache_put_args[0] == "Prompt with options"
    assert cache_put_args[1] == "options_adapter"
    assert cache_put_args[2] == "options_model"
    assert cache_put_args[3] == expected_cache_options_key
    assert cache_put_args[4] == {"text_response": "Response", "raw_response": {}} # llm_response_data


async def test_run_suite_overall_success_and_failure(
    test_runner: Runner, # Uses the fixture that has _get_adapter_instance mocked
    runner_dependencies_setup, # To access mock_yaml_loader and mock_console
    tmp_path
):
    mock_yaml_loader = runner_dependencies_setup["yaml_loader"]
    mock_console = runner_dependencies_setup["console"]
    
    # We need _run_single_test_case to be an async mock for await, as it's called by run_suite
    # The test_runner fixture already mocks _get_adapter_instance, which is used by the real _run_single_test_case
    # So, we mock _run_single_test_case itself on the instance for more direct control in this suite test.
    test_runner._run_single_test_case = AsyncMock()

    # Setup: Create two dummy YAML files
    test_file1_path = tmp_path / "suite_test1.yaml"
    test_file1_path.write_text("id: test1\nprompt: P1\nadapter: [{type: a1}]\nexpect_exact: E1")
    test_case_data1 = {"id": "test1", "prompt": "P1", "adapter": [{"type": "a1"}], "expect_exact": "E1"}

    test_file2_path = tmp_path / "suite_test2.yaml"
    test_file2_path.write_text("id: test2\nprompt: P2\nadapter: [{type: a2}]\nexpect_exact: E2")
    test_case_data2 = {"id": "test2", "prompt": "P2", "adapter": [{"type": "a2"}], "expect_exact": "E2"}

    # --- Scenario 1: All tests pass (one file, one adapter run) ---
    mock_yaml_loader.load_and_validate_yaml.side_effect = [test_case_data1]
    test_runner._run_single_test_case.return_value = [
        {"status": "PASS", "file": "suite_test1.yaml", "id": "test1", "adapter": "a1", "model": "m1"}
    ]
    test_runner.results = []
    test_runner.overall_success = True # Reset for this scenario

    success1 = await test_runner.run_suite([test_file1_path])
    assert success1 is True
    assert len(test_runner.results) == 1
    assert test_runner.results[0]["status"] == "PASS"
    assert test_runner.overall_success is True
    test_runner._run_single_test_case.assert_called_once_with(test_file1_path, test_case_data1)
    mock_console.print.assert_any_call("\n[bold green]All tests passed successfully![/bold green]")

    # --- Scenario 2: One test fails (two files, results combined) ---
    mock_yaml_loader.load_and_validate_yaml.side_effect = [test_case_data1, test_case_data2]
    
    # Mock _run_single_test_case to also set overall_success on the runner instance if a FAIL occurs
    async def mock_rrtc_side_effect(test_path, test_data_arg):
        if test_path == test_file1_path:
            # test_runner.overall_success might be set by previous calls if not reset, but for this effect, it's fine.
            return [{ "status": "PASS", "file": str(test_path), "id": test_data_arg.get("id"), "adapter": "a1", "model": "m1" }]
        elif test_path == test_file2_path:
            test_runner.overall_success = False # Simulate what the real method would do upon failure
            return [{ "status": "FAIL", "file": str(test_path), "id": test_data_arg.get("id"), "adapter": "a2", "model": "m2" }]
        return []

    test_runner._run_single_test_case.side_effect = mock_rrtc_side_effect

    test_runner.results = []
    test_runner.overall_success = True # Reset for this scenario
    test_runner._run_single_test_case.reset_mock() 
    mock_console.reset_mock()

    success2 = await test_runner.run_suite([test_file1_path, test_file2_path])
    assert success2 is False # Overall success from run_suite should be False
    assert len(test_runner.results) == 2
    assert test_runner.results[0]["status"] == "PASS"
    assert test_runner.results[1]["status"] == "FAIL"
    # overall_success is set by _run_single_test_case or if an exception occurs in run_suite directly.
    # If _run_single_test_case indicated a failure (by setting self.overall_success = False during its execution),
    # and run_suite completes, success2 (which is self.overall_success) should be False.
    assert test_runner.overall_success is False

    calls = [
        call(test_file1_path, test_case_data1),
        call(test_file2_path, test_case_data2),
    ]
    test_runner._run_single_test_case.assert_has_calls(calls, any_order=False)
    mock_console.print.assert_any_call("\n[bold red]Some tests failed or encountered errors.[/bold red]")


async def test_run_suite_yaml_error(
    test_runner: Runner, runner_dependencies_setup, tmp_path
):
    mock_yaml_loader = runner_dependencies_setup["yaml_loader"]
    mock_console = runner_dependencies_setup["console"]

    test_file_path = tmp_path / "error.yaml"
    test_file_path.write_text("invalid_yaml_content: this is not right") # File exists

    mock_yaml_loader.load_and_validate_yaml.side_effect = ValueError("YAML parsing failed")

    success = await test_runner.run_suite([test_file_path])

    assert success is False
    assert len(test_runner.results) == 1
    result = test_runner.results[0]
    assert result["status"] == "ERROR"
    assert "YAML parsing failed" in result["reason"]
    assert result["file"] == "error.yaml"
    mock_console.print.assert_any_call(
        f"[bold red]Error processing {test_file_path}: YAML parsing failed[/bold red]"
    )
    assert test_runner.overall_success is False


async def test_run_suite_empty_or_non_yaml_files(
    test_runner: Runner, runner_dependencies_setup, tmp_path
):
    mock_console = runner_dependencies_setup["console"]
    mock_yaml_loader = runner_dependencies_setup["yaml_loader"]

    non_yaml_file = tmp_path / "test.txt"
    non_yaml_file.write_text("hello")
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()

    success = await test_runner.run_suite([non_yaml_file, empty_dir / "not_a_file.yaml"])

    assert success is True # Should be true if no actual tests run or fail
    assert len(test_runner.results) == 0
    mock_yaml_loader.load_and_validate_yaml.assert_not_called()
    mock_console.print.assert_any_call(f"[yellow]Skipping non-YAML file: {non_yaml_file}[/yellow]")
    # The second file (not_a_file.yaml) also won't be processed by load_and_validate_yaml because it won't pass the .is_file() check
    # or will be skipped by the extension check if it did exist. We primarily check no tests were attempted.



async def test_run_single_test_case_multiple_adapters(
    test_runner: Runner, runner_dependencies_setup
):
    mock_adapter_1_instance = AsyncMock(spec=Adapter)
    mock_adapter_1_instance.execute = AsyncMock(return_value={"text_response": "Adapter 1 says PASS"})
    mock_adapter_1_instance.close = AsyncMock()

    mock_adapter_2_instance = AsyncMock(spec=Adapter)
    mock_adapter_2_instance.execute = AsyncMock(return_value={"text_response": "Adapter 2 says FAIL"})
    mock_adapter_2_instance.close = AsyncMock()

    # Control what _get_adapter_instance returns based on adapter name
    def get_adapter_side_effect(adapter_name):
        if adapter_name == "adapter1":
            return mock_adapter_1_instance
        elif adapter_name == "adapter2":
            return mock_adapter_2_instance
        return None
    runner_dependencies_setup["get_adapter_method_mock"].side_effect = get_adapter_side_effect

    test_file = Path("multi_adapter_test.yaml")
    test_data = {
        "id": "multi-adapter-001",
        "prompt": "Test all adapters",
        "adapter": [
            {"type": "adapter1", "model": "model1"},
            {"type": "adapter2", "model": "model2"}
        ],
        "expect_substring": "PASS" # Adapter1 should pass, Adapter2 should fail
    }

    results_list = await test_runner._run_single_test_case(test_file, test_data)

    assert len(results_list) == 2

    # Check result for adapter1
    result_adapter1 = next(r for r in results_list if r["adapter"] == "adapter1")
    assert result_adapter1["status"] == "PASS"
    assert result_adapter1["model"] == "model1"
    mock_adapter_1_instance.execute.assert_called_once_with("Test all adapters", model="model1")
    mock_adapter_1_instance.close.assert_called_once()

    # Check result for adapter2
    result_adapter2 = next(r for r in results_list if r["adapter"] == "adapter2")
    assert result_adapter2["status"] == "FAIL"
    assert result_adapter2["model"] == "model2"
    assert "Substring match failed" in result_adapter2["reason"]
    mock_adapter_2_instance.execute.assert_called_once_with("Test all adapters", model="model2")
    mock_adapter_2_instance.close.assert_called_once()

    # Check overall_success flag on the runner instance
    # If any adapter run within a single test case definition fails, overall_success should be False.
    assert test_runner.overall_success is False

    # Reset side effect for other tests
    runner_dependencies_setup["get_adapter_method_mock"].side_effect = None
    runner_dependencies_setup["get_adapter_method_mock"].return_value = runner_dependencies_setup["adapter_instance"]



# Example of how ADAPTER_REGISTRY is used internally (not a test of Runner directly but related)
@patch.dict(ADAPTER_REGISTRY, {"test_real_adapter": MagicMock(spec=Adapter)}, clear=True)
async def test_get_adapter_instance_success(tmp_path):
    runner = Runner(config_dir=tmp_path, use_cache=False)
    mock_adapter_class = ADAPTER_REGISTRY["test_real_adapter"]
    mock_adapter_instance = mock_adapter_class.return_value

    # Call the original method on the local runner instance
    adapter = runner._get_adapter_instance("test_real_adapter")
    assert adapter is mock_adapter_instance
    mock_adapter_class.assert_called_once()
    # No finally block needed.

# This test should also be async and use a local runner if testing original method,
# or be removed if test_runner_get_adapter_instance_unknown_original_method covers it.
# For now, let's assume it's covered by test_runner_get_adapter_instance_unknown_original_method
# and the warning comes from an older version. If it's still present, it should be refactored or removed.
# For safety, I will make it async and assume it's distinct for now.
async def test_get_adapter_instance_unknown(tmp_path, runner_dependencies_setup):
    runner = Runner(config_dir=tmp_path, use_cache=False)
    runner.console = runner_dependencies_setup["console"]

    adapter = runner._get_adapter_instance("completely_unknown_adapter_type_2")
    assert adapter is None
    runner.console.print.assert_any_call("[bold red]Unknown adapter: completely_unknown_adapter_type_2[/bold red]")


# Restore the original regex for test_run_single_test_case_pass_regex_match
# as the simplified version passed, indicating the core logic is okay.
# The issue might be subtle or in how the runner processes it.
# For now, let's revert it and if it fails, we'll debug regex_match directly.
async def test_run_single_test_case_pass_regex_match(
    test_runner: Runner, runner_dependencies_setup
):
    mock_adapter = runner_dependencies_setup["adapter_instance"]
    test_runner.overall_success = True 
    mock_adapter.execute.return_value = {
        "text_response": "The number is 42.",
        "raw_response": {},
    }

    test_file = Path("test_regex.yaml")
    test_data = {
        "id": "regex-pass-001",
        "prompt": "What number?",
        "adapter": [{
            "type": "mocked_adapter", "model": "regex_model"
        }],
        "expect_regex": "number is \\d+", # Corrected: No r prefix here, just the string for regex engine
    }

    results_list = await test_runner._run_single_test_case(test_file, test_data)
    assert len(results_list) == 1
    result = results_list[0]

    assert result["status"] == "PASS" # This will fail if the regex issue persists
    mock_adapter.close.assert_called_once()


# Tests for the original _get_adapter_instance method
@patch.dict(ADAPTER_REGISTRY, {"real_adapter_type": MagicMock(spec=Adapter)}, clear=True)
async def test_runner_get_adapter_instance_success_original_method(tmp_path):
    runner = Runner(config_dir=tmp_path, use_cache=False)
    mock_adapter_class = ADAPTER_REGISTRY["real_adapter_type"]
    mock_adapter_instance = mock_adapter_class.return_value

    adapter = runner._get_adapter_instance("real_adapter_type")
    assert adapter is mock_adapter_instance
    mock_adapter_class.assert_called_once()
    # No finally block, runner is local.

async def test_runner_get_adapter_instance_unknown_original_method(tmp_path, runner_dependencies_setup):
    runner = Runner(config_dir=tmp_path, use_cache=False)
    runner.console = runner_dependencies_setup["console"]

    adapter = runner._get_adapter_instance("completely_unknown_adapter_type")
    assert adapter is None
    runner.console.print.assert_any_call("[bold red]Unknown adapter: completely_unknown_adapter_type[/bold red]")
    # No finally block, runner is local.

async def test_runner_get_adapter_instance_init_error_original_method(tmp_path, runner_dependencies_setup):
    runner = Runner(config_dir=tmp_path, use_cache=False)
    runner.console = runner_dependencies_setup["console"]

    mock_failing_adapter_class = MagicMock(spec=Adapter)
    mock_failing_adapter_class.side_effect = ValueError("Failed to init adapter")

    with patch.dict(ADAPTER_REGISTRY, {"failing_adapter": mock_failing_adapter_class}, clear=True):
        adapter = runner._get_adapter_instance("failing_adapter")
        assert adapter is None
        mock_failing_adapter_class.assert_called_once()
        runner.console.print.assert_any_call("[bold red]Error initializing adapter 'failing_adapter': Failed to init adapter[/bold red]")
        # No finally block, runner is local.

# Previous placeholder tests are removed as they are now properly implemented above.
