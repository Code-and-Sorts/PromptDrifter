from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console

from promptdrifter.adapters.base import Adapter
from promptdrifter.cache import PromptCache
from promptdrifter.models.config import PromptDrifterConfig, TestCase
from promptdrifter.runner import ADAPTER_REGISTRY, Runner
from promptdrifter.yaml_loader import YamlFileLoader

pytestmark = pytest.mark.asyncio


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

    mock_get_adapter_method = MagicMock(return_value=mock_adapter_instance)

    return {
        "yaml_loader": mock_yaml_loader,
        "cache": mock_cache,
        "console": mock_console,
        "adapter_instance": mock_adapter_instance,
        "get_adapter_method_mock": mock_get_adapter_method,
    }


@pytest.fixture
def test_runner(runner_dependencies_setup, tmp_path) -> Runner:
    runner = Runner(config_dir=tmp_path, use_cache=True)
    runner._get_adapter_instance = runner_dependencies_setup["get_adapter_method_mock"]
    return runner


# --- Test Cases ---


async def test_run_single_test_case_pass_exact_match(
    test_runner: Runner, runner_dependencies_setup
):
    mock_adapter = runner_dependencies_setup["adapter_instance"]
    mock_cache = runner_dependencies_setup["cache"]

    test_file = Path("test_exact.yaml")
    test_data_dict = {
        "id": "exact-pass-001",
        "prompt": "Say hello",
        "adapter": [{"type": "openai", "model": "test_model"}],
        "expect_exact": "Hello there",
    }
    test_case_model = TestCase(**test_data_dict)
    mock_adapter.execute.return_value = {
        "text_response": "Hello there",
        "raw_response": {},
    }
    mock_cache.get.return_value = None

    results_list = await test_runner._run_single_test_case(test_file, test_case_model)
    assert len(results_list) == 1
    result = results_list[0]

    assert result["status"] == "PASS"
    test_runner._get_adapter_instance.assert_called_with("openai")
    mock_adapter.execute.assert_called_once_with("Say hello", model="test_model")
    mock_cache.put.assert_called_once()
    mock_adapter.close.assert_called_once()


async def test_run_single_test_case_fail_exact_match(
    test_runner: Runner, runner_dependencies_setup
):
    mock_adapter = runner_dependencies_setup["adapter_instance"]
    mock_adapter.execute.return_value = {"text_response": "Goodbye", "raw_response": {}}

    test_file = Path("test_exact_fail.yaml")
    test_data_dict = {
        "id": "exact-fail-001",
        "prompt": "Say hello",
        "adapter": [{"type": "openai", "model": "test_model_fail"}],
        "expect_exact": "Hello",
    }
    test_case_model = TestCase(**test_data_dict)

    results_list = await test_runner._run_single_test_case(test_file, test_case_model)
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
    test_data_dict = {
        "id": "regex-pass-001",
        "prompt": "What number?",
        "adapter": [{"type": "openai", "model": "regex_model"}],
        "expect_regex": r"number is \d+",
    }
    test_case_model = TestCase(**test_data_dict)

    results_list = await test_runner._run_single_test_case(test_file, test_case_model)
    assert len(results_list) == 1
    result = results_list[0]

    assert result["status"] == "PASS"
    mock_adapter.close.assert_called_once()


async def test_run_single_test_case_cache_hit(
    test_runner: Runner, runner_dependencies_setup
):
    mock_cache = runner_dependencies_setup["cache"]
    mock_adapter = runner_dependencies_setup["adapter_instance"]

    test_file = Path("test_cache_hit.yaml")
    test_data_dict = {
        "id": "cache-hit-001",
        "prompt": "Cached prompt",
        "adapter": [{"type": "openai", "model": "cached_model"}],
        "expect_exact": "Cached response",
    }
    test_case_model = TestCase(**test_data_dict)
    cached_llm_response = {
        "text_response": "Cached response",
        "raw_response": {"from_cache": True},
    }

    mock_cache.get.return_value = cached_llm_response
    expected_cache_options_key = frozenset({
        ("_assertion_type", "exact"),
        ("_assertion_value", "Cached response"),
    })

    results_list = await test_runner._run_single_test_case(test_file, test_case_model)
    assert len(results_list) == 1
    result = results_list[0]

    assert result["status"] == "PASS"
    assert result["cache_status"] == "HIT"
    mock_adapter.execute.assert_not_called()
    mock_cache.get.assert_called_once_with(
        "Cached prompt", "openai", "cached_model", expected_cache_options_key
    )
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
    test_data_dict = {
        "id": "adapter-err-001",
        "prompt": "Trigger error",
        "adapter": [{"type": "openai", "model": "error_model"}],
        "expect_exact": "This won't be checked",
    }
    test_case_model = TestCase(**test_data_dict)

    results_list = await test_runner._run_single_test_case(test_file, test_case_model)
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
    test_data_dict = {
        "id": "exec-exception-001",
        "prompt": "Causes exception",
        "adapter": [{"type": "openai", "model": "exception_model"}],
        "expect_exact": "Not checked",
    }
    test_case_model = TestCase(**test_data_dict)

    results_list = await test_runner._run_single_test_case(test_file, test_case_model)
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
    mock_adapter.execute.return_value = {"raw_response": {}}  # No text_response

    test_file = Path("test_no_text.yaml")
    test_data_dict = {
        "id": "no-text-001",
        "prompt": "No text",
        "adapter": [{"type": "openai", "model": "no_text_model"}],
        "expect_exact": "Not checked",
    }
    test_case_model = TestCase(**test_data_dict)

    results_list = await test_runner._run_single_test_case(test_file, test_case_model)
    assert len(results_list) == 1
    result = results_list[0]

    assert result["status"] == "FAIL"
    assert "Adapter returned no text_response." in result["reason"]
    assert test_runner.overall_success is False
    mock_adapter.close.assert_called_once()


async def test_run_single_test_case_prompt_templating_error(
    test_runner: Runner, runner_dependencies_setup
):
    test_file = Path("test_templating_error.yaml")
    test_data_dict = {
        "id": "templating-err-001",
        "prompt": "Hello {{name",
        "inputs": {"name": "Test"},
        "adapter": [{"type": "openai", "model": "any_model"}],
        "expect_exact": "Should not run",
    }
    test_case_model = TestCase(**test_data_dict)

    results_list = await test_runner._run_single_test_case(test_file, test_case_model)
    assert len(results_list) == 1
    result = results_list[0]
    assert result["status"] == "ERROR"
    assert "Prompt templating error" in result["reason"]
    assert result["id"] == "templating-err-001"
    runner_dependencies_setup["adapter_instance"].execute.assert_not_called()


async def test_run_single_test_case_no_adapter_configs(
    test_runner: Runner, runner_dependencies_setup
):
    test_file = Path("test_no_adapter_configs.yaml")
    test_case_model = TestCase(
        id="no-adapter-configs-test",
        prompt="A prompt",
        adapter_configurations=[],
        expect_exact="anything",
    )
    results_list = await test_runner._run_single_test_case(test_file, test_case_model)
    assert len(results_list) == 0


async def test_run_single_test_case_skipped_no_assertion(
    test_runner: Runner, runner_dependencies_setup
):
    mock_adapter = runner_dependencies_setup["adapter_instance"]
    mock_adapter.execute.return_value = {
        "text_response": "Some response",
        "raw_response": {},
    }
    test_file = Path("test_skip_no_assertion.yaml")
    test_data_dict = {
        "id": "skip-no-assertion",
        "prompt": "A prompt",
        "adapter": [{"type": "openai", "model": "any_model"}],
    }
    test_case_model = TestCase(**test_data_dict)
    results_list = await test_runner._run_single_test_case(test_file, test_case_model)
    assert len(results_list) == 1
    result = results_list[0]
    assert result["status"] == "FAIL"
    assert result["reason"] == ""
    assert test_runner.overall_success is False


async def test_run_single_test_case_unknown_adapter(
    test_runner: Runner, runner_dependencies_setup
):
    get_adapter_mock = runner_dependencies_setup["get_adapter_method_mock"]

    def side_effect_for_get_adapter(adapter_name_called):
        if adapter_name_called == "gemini":
            return None
        return runner_dependencies_setup["adapter_instance"]

    get_adapter_mock.side_effect = side_effect_for_get_adapter

    test_file = Path("test_unknown_adapter.yaml")
    test_data_dict = {
        "id": "unknown-adapter-001",
        "prompt": "Test prompt",
        "adapter": [{"type": "gemini", "model": "some_model"}],
        "expect_exact": "anything",
    }
    test_case_model = TestCase(**test_data_dict)

    results_list = await test_runner._run_single_test_case(test_file, test_case_model)
    assert len(results_list) == 1
    result = results_list[0]

    assert result["status"] == "ERROR"
    assert "Adapter 'gemini' not found or failed to initialize." in result["reason"]
    assert test_runner.overall_success is False


async def test_run_single_test_case_with_adapter_options(
    test_runner: Runner, runner_dependencies_setup
):
    mock_adapter = runner_dependencies_setup["adapter_instance"]
    test_file = Path("test_adapter_options.yaml")
    test_data_dict = {
        "id": "adapter-options-001",
        "prompt": "Prompt with options",
        "adapter": [
            {
                "type": "openai",
                "model": "options_model",
                "temperature": 0.77,
                "max_tokens": 123,
                "custom_param": "custom_value",
            }
        ],
        "expect_exact": "Response",
    }
    test_case_model = TestCase(**test_data_dict)
    mock_adapter.execute.return_value = {
        "text_response": "Response",
        "raw_response": {},
    }

    expected_options_to_execute = {
        "temperature": 0.77,
        "max_tokens": 123,
        "custom_param": "custom_value",
    }
    expected_cache_options_key_for_put = frozenset(
        list(expected_options_to_execute.items()) +
        [
            ("_assertion_type", "exact"),
            ("_assertion_value", "Response"),
        ]
    )

    results_list = await test_runner._run_single_test_case(test_file, test_case_model)
    assert len(results_list) == 1
    result = results_list[0]

    assert result["status"] == "PASS"
    mock_adapter.execute.assert_called_once_with(
        "Prompt with options", model="options_model", **expected_options_to_execute
    )
    runner_dependencies_setup["cache"].put.assert_called_once()
    _, cache_put_args, _ = runner_dependencies_setup["cache"].put.mock_calls[0]
    assert cache_put_args[0] == "Prompt with options"
    assert cache_put_args[1] == "openai"
    assert cache_put_args[2] == "options_model"
    assert cache_put_args[3] == expected_cache_options_key_for_put
    assert cache_put_args[4] == {"text_response": "Response", "raw_response": {}}


async def test_run_single_test_case_multiple_adapters(
    test_runner: Runner, runner_dependencies_setup
):
    mock_adapter_1_instance = AsyncMock(spec=Adapter)
    mock_adapter_1_instance.execute = AsyncMock(
        return_value={"text_response": "Adapter 1 says PASS"}
    )
    mock_adapter_1_instance.close = AsyncMock()

    mock_adapter_2_instance = AsyncMock(spec=Adapter)
    mock_adapter_2_instance.execute = AsyncMock(
        return_value={"text_response": "Adapter 2 says FAIL"}
    )
    mock_adapter_2_instance.close = AsyncMock()

    def get_adapter_side_effect(adapter_name):
        if adapter_name == "openai":
            return mock_adapter_1_instance
        elif adapter_name == "gemini":
            return mock_adapter_2_instance
        return None

    runner_dependencies_setup[
        "get_adapter_method_mock"
    ].side_effect = get_adapter_side_effect

    test_file = Path("multi_adapter_test.yaml")
    test_data_dict = {
        "id": "multi-adapter-001",
        "prompt": "Test all adapters",
        "adapter": [
            {"type": "openai", "model": "model1"},
            {"type": "gemini", "model": "model2"},
        ],
        "expect_substring": "PASS",
    }
    test_case_model = TestCase(**test_data_dict)

    results_list = await test_runner._run_single_test_case(test_file, test_case_model)

    assert len(results_list) == 2

    result_adapter1 = next(r for r in results_list if r["adapter"] == "openai")
    assert result_adapter1["status"] == "PASS"
    assert result_adapter1["model"] == "model1"
    mock_adapter_1_instance.execute.assert_called_once_with(
        "Test all adapters", model="model1"
    )
    mock_adapter_1_instance.close.assert_called_once()

    result_adapter2 = next(r for r in results_list if r["adapter"] == "gemini")
    assert result_adapter2["status"] == "FAIL"
    assert result_adapter2["model"] == "model2"
    assert "Substring match failed" in result_adapter2["reason"]
    mock_adapter_2_instance.execute.assert_called_once_with(
        "Test all adapters", model="model2"
    )
    mock_adapter_2_instance.close.assert_called_once()

    assert test_runner.overall_success is False

    runner_dependencies_setup["get_adapter_method_mock"].side_effect = None
    runner_dependencies_setup[
        "get_adapter_method_mock"
    ].return_value = runner_dependencies_setup["adapter_instance"]


async def test_run_suite_overall_success_and_failure(
    test_runner: Runner, runner_dependencies_setup, tmp_path
):
    mock_yaml_loader = runner_dependencies_setup["yaml_loader"]
    run_single_mock = AsyncMock()
    test_runner._run_single_test_case = run_single_mock

    test_case_data1_dict = {
        "id": "t1-valid-id",
        "prompt": "P1",
        "adapter": [{"type": "openai", "model": "m1"}],
        "expect_exact": "E1",
    }
    config_model1 = PromptDrifterConfig(
        version="0.1", tests=[TestCase(**test_case_data1_dict)]
    )
    test_file1_path = tmp_path / "suite_test1.yaml"
    test_file1_path.write_text(
        f'version: "0.1"\\nadapters:\\n  - id: {test_case_data1_dict["id"]}\\n    prompt: "{test_case_data1_dict["prompt"]}"\\n    expect_exact: "{test_case_data1_dict["expect_exact"]}"\\n    adapter:\\n      - type: {test_case_data1_dict["adapter"][0]["type"]}\\n        model: {test_case_data1_dict["adapter"][0]["model"]}'
    )

    test_case_data2_dict = {
        "id": "t2-valid-id",
        "prompt": "P2",
        "adapter": [{"type": "gemini", "model": "m2"}],
        "expect_exact": "E2",
    }
    config_model2 = PromptDrifterConfig(
        version="0.1", tests=[TestCase(**test_case_data2_dict)]
    )
    test_file2_path = tmp_path / "suite_test2.yaml"
    test_file2_path.write_text(
        f'version: "0.1"\\nadapters:\\n  - id: {test_case_data2_dict["id"]}\\n    prompt: "{test_case_data2_dict["prompt"]}"\\n    expect_exact: "{test_case_data2_dict["expect_exact"]}"\\n    adapter:\\n      - type: {test_case_data2_dict["adapter"][0]["type"]}\\n        model: {test_case_data2_dict["adapter"][0]["model"]}'
    )

    mock_yaml_loader.load_and_validate_yaml.side_effect = [config_model1]
    run_single_mock.return_value = [
        {
            "status": "PASS",
            "file": "suite_test1.yaml",
            "id": "t1",
            "adapter": "a1",
            "model": "m1",
        }
    ]
    test_runner.results = []
    test_runner.overall_success = True

    success1 = await test_runner.run_suite([test_file1_path])
    assert success1 is True
    assert test_runner.overall_success is True
    assert len(test_runner.results) == 1
    mock_yaml_loader.load_and_validate_yaml.assert_called_with(test_file1_path)
    run_single_mock.assert_called_with(test_file1_path, config_model1.tests[0])

    mock_yaml_loader.load_and_validate_yaml.side_effect = [config_model2]

    async def mock_run_single_fail_side_effect(*args, **kwargs):
        test_runner.overall_success = False
        return [
            {
                "status": "FAIL",
                "file": str(args[0].name),
                "id": args[1].id,
                "adapter": "gemini",
                "model": "m2",
                "reason": "Failed",
            }
        ]

    run_single_mock.side_effect = mock_run_single_fail_side_effect
    test_runner.results = []
    test_runner.overall_success = True

    success2 = await test_runner.run_suite([test_file2_path])
    assert success2 is False
    assert test_runner.overall_success is False
    assert len(test_runner.results) == 1
    assert test_runner.results[0]["status"] == "FAIL"
    run_single_mock.assert_called_with(test_file2_path, config_model2.tests[0])


async def test_run_suite_yaml_error(
    test_runner: Runner, runner_dependencies_setup, tmp_path
):
    mock_yaml_loader = runner_dependencies_setup["yaml_loader"]
    mock_console = runner_dependencies_setup["console"]
    test_file_path = tmp_path / "error.yaml"
    test_file_path.write_text("invalid_yaml_content: this is not right")

    error_message = "YAML parsing failed via Pydantic"
    mock_yaml_loader.load_and_validate_yaml.side_effect = ValueError(error_message)

    success = await test_runner.run_suite([test_file_path])
    assert success is False
    assert len(test_runner.results) == 1
    result = test_runner.results[0]
    assert result["status"] == "ERROR"
    assert error_message in result["reason"]
    assert result["id"] == "YAML_LOAD_ERROR"
    assert result["file"] == "error.yaml"
    mock_console.print.assert_any_call(error_message)
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

    success = await test_runner.run_suite(
        [non_yaml_file, empty_dir / "not_a_file.yaml"]
    )
    assert success is True
    assert len(test_runner.results) == 0
    mock_yaml_loader.load_and_validate_yaml.assert_not_called()
    mock_console.print.assert_any_call(
        f"[yellow]Skipping non-YAML file: {non_yaml_file}[/yellow]"
    )


@patch.dict(
    ADAPTER_REGISTRY, {"test_real_adapter": MagicMock(spec=Adapter)}, clear=True
)
async def test_get_adapter_instance_success(tmp_path):
    runner = Runner(config_dir=tmp_path)
    ADAPTER_REGISTRY["test_real_adapter"].return_value = MagicMock(spec=Adapter)
    adapter_instance = runner._get_adapter_instance("test_real_adapter")
    assert adapter_instance is not None
    ADAPTER_REGISTRY["test_real_adapter"].assert_called_once()


async def test_get_adapter_instance_unknown(tmp_path, runner_dependencies_setup):
    runner = Runner(config_dir=tmp_path)
    runner.console = runner_dependencies_setup["console"]
    adapter_instance = runner._get_adapter_instance("completely_unknown_adapter")
    assert adapter_instance is None
    runner.console.print.assert_called_with(
        "[bold red]Unknown adapter: completely_unknown_adapter[/bold red]"
    )


@patch.dict(
    ADAPTER_REGISTRY, {"real_adapter_type": MagicMock(spec=Adapter)}, clear=True
)
async def test_runner_get_adapter_instance_success_original_method(tmp_path):
    runner = Runner(config_dir=tmp_path)
    MockAdapterClass = ADAPTER_REGISTRY["real_adapter_type"]
    mock_adapter_instance = MagicMock(spec=Adapter)
    MockAdapterClass.return_value = mock_adapter_instance
    instance = runner._get_adapter_instance("real_adapter_type")
    assert instance == mock_adapter_instance
    MockAdapterClass.assert_called_once()


async def test_runner_get_adapter_instance_unknown_original_method(
    tmp_path, runner_dependencies_setup
):
    runner = Runner(config_dir=tmp_path)
    runner.console = runner_dependencies_setup["console"]
    instance = runner._get_adapter_instance("non_existent_adapter_for_real")
    assert instance is None
    runner.console.print.assert_called_with(
        "[bold red]Unknown adapter: non_existent_adapter_for_real[/bold red]"
    )


async def test_runner_get_adapter_instance_init_error_original_method(
    tmp_path, runner_dependencies_setup
):
    MockAdapterWithError = MagicMock(spec=Adapter)
    MockAdapterWithError.side_effect = Exception("Init failed")
    with patch.dict(ADAPTER_REGISTRY, {"error_adapter": MockAdapterWithError}):
        runner = Runner(config_dir=tmp_path)
        runner.console = runner_dependencies_setup["console"]
        instance = runner._get_adapter_instance("error_adapter")
        assert instance is None
        runner.console.print.assert_called_with(
            "[bold red]Error initializing adapter 'error_adapter': Init failed[/bold red]"
        )


async def test_run_single_test_case_prompt_templating(
    test_runner: Runner, runner_dependencies_setup
):
    mock_adapter = runner_dependencies_setup["adapter_instance"]

    test_file = Path("test_templating.yaml")
    test_data_dict = {
        "id": "templating-001",
        "prompt": "Hello {{name}} from {{place}}!",
        "inputs": {"name": "Test User", "place": "Pytest"},
        "adapter": [{"type": "openai", "model": "template_model"}],
        "expect_exact": "Hello Test User from Pytest!",
    }
    test_case_model = TestCase(**test_data_dict)
    mock_adapter.execute.return_value = {
        "text_response": "Hello Test User from Pytest!",
        "raw_response": {},
    }

    results_list = await test_runner._run_single_test_case(test_file, test_case_model)

    assert len(results_list) == 1
    result = results_list[0]
    assert result["status"] == "PASS"

    mock_adapter.execute.assert_called_once_with(
        "Hello Test User from Pytest!", model="template_model"
    )
