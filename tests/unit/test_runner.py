from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console

from promptdrifter.adapters.base import Adapter
from promptdrifter.cache import PromptCache
from promptdrifter.models.config import PromptDrifterConfig, TestCase
from promptdrifter.runner import Runner
from promptdrifter.yaml_loader import YamlFileLoader

pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_yaml_loader(mocker) -> MagicMock:
    return mocker.MagicMock(spec=YamlFileLoader)


@pytest.fixture
def mock_cache(mocker) -> MagicMock:
    mock = mocker.MagicMock(spec=PromptCache)
    mock.get = MagicMock(return_value=None)
    mock.put = MagicMock()
    return mock


@pytest.fixture
def mock_console(mocker) -> MagicMock:
    return mocker.MagicMock(spec=Console)


@pytest.fixture
def mock_adapter_instance() -> AsyncMock:
    adapter_mock = AsyncMock(spec=Adapter)
    adapter_mock.execute = AsyncMock()
    adapter_mock.close = AsyncMock()

    mock_config = MagicMock()
    mock_config_class = MagicMock()
    mock_config_class.return_value = mock_config
    adapter_mock.config = mock_config
    adapter_mock.config.__class__ = mock_config_class

    return adapter_mock


@pytest.fixture
def runner_dependencies_setup(
    mocker, mock_yaml_loader, mock_cache, mock_console, mock_adapter_instance
):
    """Fixture to provide all mocked dependencies for the Runner, replacing class methods/constructors."""
    mocker.patch("promptdrifter.runner.YamlFileLoader", return_value=mock_yaml_loader)
    mocker.patch("promptdrifter.runner.PromptCache", return_value=mock_cache)
    mocker.patch("promptdrifter.runner.Console", return_value=mock_console)

    mock_get_adapter_method = AsyncMock(return_value=mock_adapter_instance)

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
        "tags": ["test-tag", "example"],
    }
    test_case_model = TestCase(**test_data_dict)

    # Create a mock response object with attributes
    mock_response = MagicMock()
    mock_response.text_response = "Hello there"
    mock_response.raw_response = {}
    mock_response.error = None
    mock_adapter.execute.return_value = mock_response

    mock_cache.get.return_value = None

    results_list = await test_runner._run_single_test_case(test_file, test_case_model)
    assert len(results_list) == 1
    result = results_list[0]

    assert result["status"] == "PASS"
    test_runner._get_adapter_instance.assert_called_with("openai", None)
    mock_adapter.execute.assert_called_once()
    call_args = mock_adapter.execute.call_args
    assert call_args[0][0] == "Say hello"  # First positional arg is prompt
    assert call_args[1]["config_override"].default_model == "test_model"  # Check config override
    mock_cache.put.assert_called_once()


async def test_run_single_test_case_fail_exact_match(
    test_runner: Runner, runner_dependencies_setup
):
    mock_adapter = runner_dependencies_setup["adapter_instance"]

    # Create a mock response object with attributes
    mock_response = MagicMock()
    mock_response.text_response = "Goodbye"
    mock_response.raw_response = {}
    mock_response.error = None
    mock_adapter.execute.return_value = mock_response

    test_file = Path("test_exact_fail.yaml")
    test_data_dict = {
        "id": "exact-fail-001",
        "prompt": "Say hello",
        "adapter": [{"type": "openai", "model": "test_model_fail"}],
        "expect_exact": "Hello",
        "tags": ["fail-test"],
    }
    test_case_model = TestCase(**test_data_dict)

    results_list = await test_runner._run_single_test_case(test_file, test_case_model)
    assert len(results_list) == 1
    result = results_list[0]

    assert result["status"] == "FAIL"
    assert "Exact match failed" in result["reason"]
    assert test_runner.overall_success is False


async def test_run_single_test_case_pass_regex_match(
    test_runner: Runner, runner_dependencies_setup
):
    mock_adapter = runner_dependencies_setup["adapter_instance"]
    test_runner.overall_success = True

    mock_response = MagicMock()
    mock_response.text_response = "The number is 42."
    mock_response.raw_response = {}
    mock_response.error = None
    mock_adapter.execute.return_value = mock_response

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
        "tags": ["cached", "test"],
    }
    test_case_model = TestCase(**test_data_dict)
    cached_llm_response = {
        "text_response": "Cached response",
        "raw_response": {"from_cache": True},
    }

    mock_cache.get.return_value = cached_llm_response
    expected_cache_options_key = frozenset(
        {
            ("_assertion_type", "exact"),
            ("_assertion_value", "Cached response"),
        }
    )

    results_list = await test_runner._run_single_test_case(test_file, test_case_model)
    assert len(results_list) == 1
    result = results_list[0]

    assert result["status"] == "PASS"
    assert result["cache_status"] == "HIT"
    mock_adapter.execute.assert_not_called()
    mock_cache.get.assert_called_once_with(
        "Cached prompt", "openai", "cached_model", expected_cache_options_key
    )
    # With the concurrent adapter implementation, adapter.close() is no longer called
    # for cached responses since we don't add the adapter to adapter_instances list


async def test_run_single_test_case_adapter_error(
    test_runner: Runner, runner_dependencies_setup
):
    mock_adapter = runner_dependencies_setup["adapter_instance"]

    # Create a mock response object with attributes
    mock_response = MagicMock()
    mock_response.text_response = None
    mock_response.raw_response = {}
    mock_response.error = "Something went wrong with LLM"
    mock_adapter.execute.return_value = mock_response

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


async def test_run_single_test_case_no_text_response(
    test_runner: Runner, runner_dependencies_setup
):
    mock_adapter = runner_dependencies_setup["adapter_instance"]

    mock_response = MagicMock()
    mock_response.text_response = None
    mock_response.raw_response = {}
    mock_response.error = None
    mock_adapter.execute.return_value = mock_response

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

    mock_response = MagicMock()
    mock_response.text_response = "Some response"
    mock_response.raw_response = {}
    mock_response.error = None
    mock_adapter.execute.return_value = mock_response

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

    async def side_effect_for_get_adapter(adapter_name_called, base_url=None):
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

    mock_response = MagicMock()
    mock_response.text_response = "Response"
    mock_response.raw_response = {}
    mock_response.error = None
    mock_adapter.execute.return_value = mock_response

    results_list = await test_runner._run_single_test_case(test_file, test_case_model)
    assert len(results_list) == 1
    result = results_list[0]

    assert result["status"] == "PASS"
    mock_adapter.execute.assert_called_once()
    call_args = mock_adapter.execute.call_args
    assert call_args[0][0] == "Prompt with options"
    config_override = call_args[1]["config_override"]
    assert config_override.default_model == "options_model"
    assert config_override.temperature == 0.77
    assert config_override.max_tokens == 123
    assert config_override.custom_param == "custom_value"
    runner_dependencies_setup["cache"].put.assert_called_once()


async def test_run_single_test_case_multiple_adapters(
    test_runner: Runner, runner_dependencies_setup
):
    mock_adapter_1_instance = AsyncMock(spec=Adapter)

    mock_response_1 = MagicMock()
    mock_response_1.text_response = "Adapter 1 says PASS"
    mock_response_1.raw_response = {}
    mock_response_1.error = None
    mock_adapter_1_instance.execute = AsyncMock(return_value=mock_response_1)
    mock_adapter_1_instance.close = AsyncMock()

    mock_config_1 = MagicMock()
    mock_config_class_1 = MagicMock()
    mock_config_class_1.return_value = mock_config_1
    mock_adapter_1_instance.config = mock_config_1
    mock_adapter_1_instance.config.__class__ = mock_config_class_1

    mock_adapter_2_instance = AsyncMock(spec=Adapter)

    mock_response_2 = MagicMock()
    mock_response_2.text_response = "Adapter 2 says FAIL"
    mock_response_2.raw_response = {}
    mock_response_2.error = None
    mock_adapter_2_instance.execute = AsyncMock(return_value=mock_response_2)
    mock_adapter_2_instance.close = AsyncMock()

    mock_config_2 = MagicMock()
    mock_config_class_2 = MagicMock()
    mock_config_class_2.return_value = mock_config_2
    mock_adapter_2_instance.config = mock_config_2
    mock_adapter_2_instance.config.__class__ = mock_config_class_2

    async def get_adapter_side_effect(adapter_name, base_url=None):
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
    mock_adapter_1_instance.execute.assert_called_once()
    call_args_1 = mock_adapter_1_instance.execute.call_args
    assert call_args_1[0][0] == "Test all adapters"
    assert call_args_1[1]["config_override"].default_model == "model1"

    result_adapter2 = next(r for r in results_list if r["adapter"] == "gemini")
    assert result_adapter2["status"] == "FAIL"
    assert result_adapter2["model"] == "model2"
    assert "Substring match failed" in result_adapter2["reason"]
    mock_adapter_2_instance.execute.assert_called_once()
    call_args_2 = mock_adapter_2_instance.execute.call_args
    assert call_args_2[0][0] == "Test all adapters"
    assert call_args_2[1]["config_override"].default_model == "model2"

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


async def test_get_adapter_instance_success(tmp_path):
    runner = Runner(config_dir=tmp_path)

    with patch.object(runner.adapter_manager, 'get_adapter', new_callable=AsyncMock) as mock_get_adapter:
        mock_adapter = MagicMock(spec=Adapter)
        mock_get_adapter.return_value = mock_adapter

        adapter_instance = await runner._get_adapter_instance("openai")

        assert adapter_instance is not None
        mock_get_adapter.assert_called_once_with(
            adapter_type="openai",
            api_key=None,
            base_url=None
        )


async def test_get_adapter_instance_unknown(tmp_path, runner_dependencies_setup):
    runner = Runner(config_dir=tmp_path)
    runner.console = runner_dependencies_setup["console"]

    with patch.object(runner.adapter_manager, 'get_adapter', new_callable=AsyncMock) as mock_get_adapter:
        mock_get_adapter.return_value = None

        adapter_instance = await runner._get_adapter_instance("completely_unknown_adapter")
        assert adapter_instance is None
        runner.console.print.assert_called_with(
            "[bold red]Unknown adapter: completely_unknown_adapter[/bold red]"
        )


async def test_runner_get_adapter_instance_success_with_api_key(tmp_path):
    runner = Runner(config_dir=tmp_path, openai_api_key="test-key")

    with patch.object(runner.adapter_manager, 'get_adapter', new_callable=AsyncMock) as mock_get_adapter:
        mock_adapter = MagicMock(spec=Adapter)
        mock_get_adapter.return_value = mock_adapter

        instance = await runner._get_adapter_instance("openai")
        assert instance == mock_adapter
        mock_get_adapter.assert_called_once_with(
            adapter_type="openai",
            api_key="test-key",
            base_url=None
        )


async def test_runner_get_adapter_instance_error_handling(tmp_path, runner_dependencies_setup):
    runner = Runner(config_dir=tmp_path)
    runner.console = runner_dependencies_setup["console"]

    with patch.object(runner.adapter_manager, 'get_adapter', new_callable=AsyncMock) as mock_get_adapter:
        mock_get_adapter.side_effect = Exception("Test error")

        instance = await runner._get_adapter_instance("openai")
        assert instance is None
        runner.console.print.assert_called_with(
            "[bold red]Error initializing adapter 'openai': Test error[/bold red]"
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

    mock_response = MagicMock()
    mock_response.text_response = "Hello Test User from Pytest!"
    mock_response.raw_response = {}
    mock_response.error = None
    mock_adapter.execute.return_value = mock_response

    results_list = await test_runner._run_single_test_case(test_file, test_case_model)

    assert len(results_list) == 1
    result = results_list[0]
    assert result["status"] == "PASS"

    mock_adapter.execute.assert_called_once()
    call_args = mock_adapter.execute.call_args
    assert call_args[0][0] == "Hello Test User from Pytest!"
    assert call_args[1]["config_override"].default_model == "template_model"
