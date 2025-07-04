import re
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import typer
from typer.testing import CliRunner

from promptdrifter.cli import _run_async, app

cli_runner = CliRunner()


def strip_ansi(text: str) -> str:
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


@pytest.mark.asyncio
async def test_cli_run_logic_success(mocker, tmp_path):
    test_file = tmp_path / "test.yaml"
    test_file.write_text("version: 0.1\nid: test\nprompt: hello")
    mock_instance = AsyncMock()
    mock_instance.run_suite = AsyncMock(return_value=True)
    mock_instance.close_cache_connection = AsyncMock()
    mock_class = mocker.patch("promptdrifter.cli.Runner", return_value=mock_instance)
    await _run_async(
        files=[Path(str(test_file))],
        no_cache=False,
        cache_db=None,
        config_dir=Path("."),
        max_concurrent_prompt_tests=10,
        openai_api_key=None,
        gemini_api_key=None,
        qwen_api_key=None,
        claude_api_key=None,
        grok_api_key=None,
        deepseek_api_key=None,
        mistral_api_key=None,
        # llama_api_key=None,
    )
    mock_class.assert_called_once_with(
        config_dir=Path("."),
        cache_db_path=None,
        use_cache=True,
        max_concurrent_prompt_tests=10,
        openai_api_key=None,
        gemini_api_key=None,
        qwen_api_key=None,
        claude_api_key=None,
        grok_api_key=None,
        deepseek_api_key=None,
        mistral_api_key=None,
        # llama_api_key=None,
    )
    mock_instance.run_suite.assert_called_once_with([str(test_file)])
    mock_instance.close_cache_connection.assert_called_once()


@pytest.mark.asyncio
async def test_cli_run_logic_failure_from_suite(mocker, tmp_path):
    test_file = tmp_path / "test.yaml"
    test_file.write_text("version: 0.1\nid: test\nprompt: hello")
    mock_instance = AsyncMock()
    mock_instance.run_suite = AsyncMock(return_value=False)
    mock_instance.close_cache_connection = AsyncMock()
    mock_class = mocker.patch("promptdrifter.cli.Runner", return_value=mock_instance)
    with pytest.raises(typer.Exit) as exc_info:
        await _run_async(
            files=[Path(str(test_file))],
            no_cache=False,
            cache_db=None,
            config_dir=Path("."),
            max_concurrent_prompt_tests=10,
            openai_api_key=None,
            gemini_api_key=None,
            qwen_api_key=None,
            claude_api_key=None,
            grok_api_key=None,
            deepseek_api_key=None,
            mistral_api_key=None,
        )
    assert exc_info.value.exit_code == 1
    mock_class.assert_called_once_with(
        config_dir=Path("."),
        cache_db_path=None,
        use_cache=True,
        max_concurrent_prompt_tests=10,
        openai_api_key=None,
        gemini_api_key=None,
        qwen_api_key=None,
        claude_api_key=None,
        grok_api_key=None,
        deepseek_api_key=None,
        mistral_api_key=None,
    )
    mock_instance.run_suite.assert_called_once_with([str(test_file)])
    mock_instance.close_cache_connection.assert_called_once()


@pytest.mark.asyncio
async def test_cli_run_logic_runner_init_exception(mocker, tmp_path, capsys):
    test_file = tmp_path / "test.yaml"
    test_file.write_text("version: 0.1\nid: test\nprompt: hello")
    mock_class_raising = mocker.patch(
        "promptdrifter.cli.Runner", side_effect=RuntimeError("Runner init boom!")
    )
    with pytest.raises(typer.Exit) as exc_info:
        await _run_async(
            files=[Path(str(test_file))],
            no_cache=False,
            cache_db=None,
            config_dir=Path("."),
            max_concurrent_prompt_tests=10,
            openai_api_key=None,
            gemini_api_key=None,
            qwen_api_key=None,
            claude_api_key=None,
            grok_api_key=None,
            deepseek_api_key=None,
            mistral_api_key=None,
            # llama_api_key=None,
        )
    assert exc_info.value.exit_code == 1
    mock_class_raising.assert_called_once()
    captured = capsys.readouterr()
    assert "Runner init boom!" in captured.out


@pytest.mark.asyncio
async def test_cli_run_logic_suite_exception(mocker, tmp_path, capsys):
    test_file = tmp_path / "test.yaml"
    test_file.write_text("version: 0.1\nid: test\nprompt: hello")
    mock_instance = AsyncMock()
    mock_instance.run_suite = AsyncMock(side_effect=Exception("Runner suite boom!"))
    mock_instance.close_cache_connection = AsyncMock()
    mock_class = mocker.patch("promptdrifter.cli.Runner", return_value=mock_instance)
    with pytest.raises(typer.Exit) as exc_info:
        await _run_async(
            files=[Path(str(test_file))],
            no_cache=False,
            cache_db=None,
            config_dir=Path("."),
            max_concurrent_prompt_tests=10,
            openai_api_key=None,
            gemini_api_key=None,
            qwen_api_key=None,
            claude_api_key=None,
            grok_api_key=None,
            deepseek_api_key=None,
            mistral_api_key=None,
            # llama_api_key=None,
        )
    assert exc_info.value.exit_code == 1
    mock_instance.run_suite.assert_called_once_with([str(test_file)])
    mock_instance.close_cache_connection.assert_called_once()
    mock_class.assert_called_once()
    captured = capsys.readouterr()
    assert "Runner suite boom!" in captured.out


@pytest.mark.asyncio
async def test_cli_run_logic_multiple_files(mocker, tmp_path):
    file1 = tmp_path / "test1.yaml"
    file1.write_text("version: 0.1\nid: test1\nprompt: hello")
    file2 = tmp_path / "test2.yaml"
    file2.write_text("version: 0.1\nid: test2\nprompt: world")
    mock_instance = AsyncMock()
    mock_instance.run_suite = AsyncMock(return_value=True)
    mock_instance.close_cache_connection = AsyncMock()
    mock_class = mocker.patch("promptdrifter.cli.Runner", return_value=mock_instance)
    await _run_async(
        files=[Path(str(file1)), Path(str(file2))],
        no_cache=False,
        cache_db=None,
        config_dir=Path("."),
        max_concurrent_prompt_tests=10,
        openai_api_key=None,
        gemini_api_key=None,
        qwen_api_key=None,
        claude_api_key=None,
        grok_api_key=None,
        deepseek_api_key=None,
        mistral_api_key=None,
        # llama_api_key=None,
    )
    mock_class.assert_called_once_with(
        config_dir=Path("."),
        cache_db_path=None,
        use_cache=True,
        max_concurrent_prompt_tests=10,
        openai_api_key=None,
        gemini_api_key=None,
        qwen_api_key=None,
        claude_api_key=None,
        grok_api_key=None,
        deepseek_api_key=None,
        mistral_api_key=None,
        # llama_api_key=None,
    )
    mock_instance.run_suite.assert_called_once_with([str(file1), str(file2)])
    mock_instance.close_cache_connection.assert_called_once()


@pytest.mark.asyncio
async def test_cli_run_logic_file_not_found(mocker, capsys):
    mock_class_for_safety = mocker.patch("promptdrifter.cli.Runner")
    with pytest.raises(typer.Exit) as exc_info:
        await _run_async(
            files=[Path("non_existent_file.yaml")],
            no_cache=False,
            cache_db=None,
            config_dir=Path("."),
            max_concurrent_prompt_tests=10,
            openai_api_key=None,
            gemini_api_key=None,
            qwen_api_key=None,
            claude_api_key=None,
            grok_api_key=None,
            deepseek_api_key=None,
            mistral_api_key=None,
            # llama_api_key=None,
        )
    assert exc_info.value.exit_code == 1
    mock_class_for_safety.assert_not_called()
    captured = capsys.readouterr()
    assert "Invalid file(s) provided:" in captured.out
    assert "File not found" in captured.out


@pytest.mark.asyncio
async def test_cli_run_logic_no_files_provided(mocker, capsys):
    mock_class_for_safety = mocker.patch("promptdrifter.cli.Runner")
    with pytest.raises(typer.Exit) as exc_info:
        await _run_async(
            files=[],
            no_cache=False,
            cache_db=None,
            config_dir=Path("."),
            max_concurrent_prompt_tests=10,
            openai_api_key=None,
            gemini_api_key=None,
            qwen_api_key=None,
            claude_api_key=None,
            grok_api_key=None,
            deepseek_api_key=None,
            mistral_api_key=None,
            # llama_api_key=None,
        )
    assert exc_info.value.exit_code == 1
    mock_class_for_safety.assert_not_called()
    captured = capsys.readouterr()
    assert "No YAML files provided" in captured.out


def test_init_command_default_path(mocker, tmp_path):
    """Test init command in a temporary directory (simulating default '.' behavior)."""
    config_file = tmp_path / "promptdrifter.yaml"
    if config_file.exists():
        config_file.unlink()

    mocker.patch.object(Path, "resolve", return_value=tmp_path)
    mock_files = mocker.patch("importlib.resources.files")
    mock_files.return_value.joinpath.return_value.read_text.return_value = (
        "version: '0.1'\\nsample_content_for_default_init: true"
    )

    result = cli_runner.invoke(app, ["init", str(tmp_path)])

    assert result.exit_code == 0, strip_ansi(result.stdout)
    normalized_stdout = " ".join(strip_ansi(result.stdout).replace("\n", " ").split())
    assert "Successfully created sample configuration:" in normalized_stdout
    assert config_file.name in normalized_stdout
    assert (
        "You can now edit this file and run 'promptdrifter run'." in normalized_stdout
    )
    assert config_file.exists()
    assert (
        config_file.read_text()
        == "version: '0.1'\\nsample_content_for_default_init: true"
    )
    if config_file.exists():  # Clean up
        config_file.unlink()

def test_init_new_directory_success(mocker, tmp_path):
    """Test init command when target directory needs to be created."""
    base_dir = tmp_path / "new_project_dir"
    config_file = base_dir / "promptdrifter.yaml"

    assert not base_dir.exists()

    mocker.patch.object(Path, "resolve", return_value=base_dir)
    mock_files = mocker.patch("importlib.resources.files")
    mock_files.return_value.joinpath.return_value.read_text.return_value = (
        "version: '0.1'\\nsample_content: true"
    )

    result = cli_runner.invoke(app, ["init", str(base_dir)])

    assert result.exit_code == 0, strip_ansi(result.stdout)
    normalized_stdout = " ".join(strip_ansi(result.stdout).replace("\n", " ").split())
    assert "Created directory:" in normalized_stdout
    assert base_dir.name in normalized_stdout
    assert "Successfully created sample configuration:" in normalized_stdout
    assert config_file.name in normalized_stdout
    assert base_dir.exists()
    assert base_dir.is_dir()
    assert config_file.exists()
    assert config_file.read_text() == "version: '0.1'\\nsample_content: true"


def test_init_sample_config_not_found(mocker, tmp_path):
    """Test init command when the sample config file is not found by importlib."""
    target_dir = tmp_path / "project_dir"
    target_dir.mkdir()

    mocker.patch.object(Path, "resolve", return_value=target_dir)
    mock_files = mocker.patch("importlib.resources.files")
    mock_files.return_value.joinpath.return_value.read_text.side_effect = (
        FileNotFoundError("Sample not found!")
    )

    result = cli_runner.invoke(app, ["init", str(target_dir)])

    assert result.exit_code == 1, strip_ansi(result.stdout)
    assert "Error: Sample configuration file not found in the package." in strip_ansi(
        result.stdout
    )
    assert not (target_dir / "promptdrifter.yaml").exists()


def test_init_target_path_is_file(mocker, tmp_path):
    """Test init command when the target path exists but is a file."""
    target_file_path = tmp_path / "iam_a_file.txt"
    target_file_path.write_text("I am a file, not a directory.")

    mocker.patch.object(Path, "resolve", return_value=target_file_path)

    result = cli_runner.invoke(app, ["init", str(target_file_path)])

    assert result.exit_code == 1, strip_ansi(result.stdout)
    normalized_stdout = " ".join(strip_ansi(result.stdout).replace("\n", " ").split())
    assert "Error: Target path" in normalized_stdout
    assert target_file_path.name in normalized_stdout
    assert "exists but is not a directory." in normalized_stdout


def test_init_config_already_exists(mocker, tmp_path):
    """Test init command when promptdrifter.yaml already exists."""
    target_dir = tmp_path / "existing_project"
    target_dir.mkdir()
    config_file = target_dir / "promptdrifter.yaml"
    original_content = "version: '0.1'\\niam_already_here: true"
    config_file.write_text(original_content)

    mocker.patch.object(Path, "resolve", return_value=target_dir)

    result = cli_runner.invoke(app, ["init", str(target_dir)])

    assert result.exit_code == 0, strip_ansi(result.stdout)
    normalized_stdout = " ".join(strip_ansi(result.stdout).replace("\n", " ").split())
    assert "Warning: Configuration file" in normalized_stdout
    assert config_file.name in normalized_stdout
    assert "already exists. Skipping." in normalized_stdout
    assert config_file.read_text() == original_content


def test_init_io_error_writing_config(mocker, tmp_path):
    """Test init command when there's an IOError writing the config file."""
    target_dir = tmp_path / "write_fail_dir"

    mocker.patch.object(Path, "resolve", return_value=target_dir)
    mock_files = mocker.patch("importlib.resources.files")
    mock_files.return_value.joinpath.return_value.read_text.return_value = (
        "version: '0.1'\\nsample_content: true"
    )
    mocker.patch(
        "builtins.open", side_effect=IOError("Disk full or something terrible")
    )

    result = cli_runner.invoke(app, ["init", str(target_dir)])

    assert result.exit_code == 1, strip_ansi(result.stdout)
    normalized_stdout = " ".join(strip_ansi(result.stdout).replace("\n", " ").split())
    assert "Created directory:" in normalized_stdout
    assert target_dir.name in normalized_stdout
    assert "Error writing configuration file to" in normalized_stdout
    assert (target_dir / "promptdrifter.yaml").name in normalized_stdout
    assert "Disk full or something terrible" in normalized_stdout
    assert target_dir.exists()


@pytest.mark.asyncio
async def test_cli_run_prints_security_warning(mocker, tmp_path, capsys):
    test_file = tmp_path / "test.yaml"
    test_file.write_text("version: 0.1\\nid: test\\nprompt: hello")
    mock_instance = AsyncMock()
    mock_instance.run_suite = AsyncMock(return_value=True)
    mock_instance.close_cache_connection = AsyncMock()
    mocker.patch("promptdrifter.cli.Runner", return_value=mock_instance)

    await _run_async(
        files=[Path(str(test_file))],
        no_cache=False,
        cache_db=None,
        config_dir=Path("."),
        max_concurrent_prompt_tests=10,
        openai_api_key="cli_key_here",
        gemini_api_key=None,
        qwen_api_key=None,
        claude_api_key=None,
        grok_api_key=None,
        deepseek_api_key=None,
        mistral_api_key=None,
        # llama_api_key=None,
    )
    captured = capsys.readouterr()
    assert "SECURITY WARNING" in captured.out
    assert "shell history" in captured.out
    mock_instance.close_cache_connection.assert_called_once()
    mock_instance.close_cache_connection.reset_mock()

    await _run_async(
        files=[Path(str(test_file))],
        no_cache=False,
        cache_db=None,
        config_dir=Path("."),
        max_concurrent_prompt_tests=10,
        openai_api_key=None,
        gemini_api_key=None,
        qwen_api_key=None,
        claude_api_key=None,
        grok_api_key=None,
        deepseek_api_key=None,
        mistral_api_key=None,
        # llama_api_key=None,
    )
    captured = capsys.readouterr()
    assert "Warning" not in captured.out
    assert "SECURITY WARNING:" not in captured.out
    mock_instance.close_cache_connection.assert_called_once()


def test_run_command_with_api_keys(mocker):
    mock_run_async = mocker.patch("promptdrifter.cli._run_async")
    test_file_path = Path("dummy.yaml")
    dummy_cache_db = Path("dummy_cache.db")
    dummy_config_dir = Path("dummy_config")

    result = cli_runner.invoke(
        app,
        [
            "run",
            str(test_file_path),
            "--openai-api-key",
            "key1",
            "--gemini-api-key",
            "key2",
            "--qwen-api-key",
            "test_qwen_key",
            "--claude-api-key",
            "claude_key",
            "--grok-api-key",
            "grok_key",
            "--deepseek-api-key",
            "deepseek_key",
            "--no-cache",
            "--cache-db",
            str(dummy_cache_db),
            "--config-dir",
            str(dummy_config_dir),
        ],
    )
    assert result.exit_code == 0, strip_ansi(result.stdout)

    mock_run_async.assert_called_once_with(
        [test_file_path],  # files (positional)
        True,  # no_cache (positional)
        dummy_cache_db,  # cache_db (positional)
        dummy_config_dir,  # config_dir (positional)
        10,  # max_concurrent_prompt_tests (positional)
        "key1",  # openai_api_key (positional)
        "key2",  # gemini_api_key (positional)
        "test_qwen_key",  # qwen_api_key (positional)
        "claude_key",  # claude_api_key (positional)
        "grok_key",  # grok_api_key (positional)
        "deepseek_key",  # deepseek_api_key (positional)
        None,  # mistral_api_key (positional)
        # None,  # llama_api_key (positional)
    )
    mock_run_async.reset_mock()

    result = cli_runner.invoke(
        app,
        [
            "run",
            str(test_file_path),
            "--no-cache",
            "--cache-db",
            str(dummy_cache_db),
            "--config-dir",
            str(dummy_config_dir),
        ],
    )
    assert result.exit_code == 0, strip_ansi(result.stdout)

    mock_run_async.assert_called_once_with(
        [test_file_path],
        True,
        dummy_cache_db,
        dummy_config_dir,
        10,  # max_concurrent_prompt_tests
        None,  # No OpenAI key
        None,  # No Gemini key
        None,  # No Qwen key
        None,  # No Claude key
        None,  # No Grok key
        None,  # No DeepSeek key
        None,  # No Mistral key
        # None,  # No Llama key
    )
    mock_run_async.reset_mock()

    result = cli_runner.invoke(
        app,
        [
            "run",
            str(test_file_path),
            "--gemini-api-key",
            "just_gemini_key",
            "--no-cache",
            "--cache-db",
            str(dummy_cache_db),
            "--config-dir",
            str(dummy_config_dir),
        ],
    )
    assert result.exit_code == 0, strip_ansi(result.stdout)

    mock_run_async.assert_called_once_with(
        [test_file_path],
        True,
        dummy_cache_db,
        dummy_config_dir,
        10,  # max_concurrent_prompt_tests
        None,  # No OpenAI key
        "just_gemini_key",  # Only Gemini key provided
        None,  # No Qwen key
        None,  # No Claude key
        None,  # No Grok key
        None,  # No DeepSeek key
        None,  # No Mistral key
        # None,  # No Llama key
    )


def test_assertion_exact_match_true(mocker):
    """
    Test the drift-type command with exact_match assertion that returns True.
    """
    result = cli_runner.invoke(app, ["test-drift-type", "exact_match", "hello", "hello"])
    assert result.exit_code == 0
    assert "Assertion: exact_match" in strip_ansi(result.stdout)
    assert "Result: True" in strip_ansi(result.stdout)


def test_assertion_exact_match_false(mocker):
    """
    Test the drift-type command with exact_match assertion that returns False.
    """
    result = cli_runner.invoke(app, ["test-drift-type", "exact_match", "hello", "world"])
    assert result.exit_code == 0
    assert "Assertion: exact_match" in strip_ansi(result.stdout)
    assert "Result: False" in strip_ansi(result.stdout)


def test_assertion_regex_match(mocker):
    """
    Test the drift-type command with regex_match assertion.
    """
    result = cli_runner.invoke(app, ["test-drift-type", "regex_match", "he.*o", "hello"])
    assert result.exit_code == 0
    assert "Assertion: regex_match" in strip_ansi(result.stdout)
    assert "Result: True" in strip_ansi(result.stdout)


def test_assertion_substring(mocker):
    """
    Test the drift-type command with expect_substring assertion.
    """
    result = cli_runner.invoke(app, ["test-drift-type", "expect_substring", "he", "hello"])
    assert result.exit_code == 0
    assert "Assertion: expect_substring" in strip_ansi(result.stdout)
    assert "Result: True" in strip_ansi(result.stdout)


def test_assertion_invalid_type(mocker):
    """
    Test the drift-type command with an invalid assertion type.
    """
    result = cli_runner.invoke(app, ["test-drift-type", "invalid_type", "hello", "hello"])
    assert result.exit_code == 1
    assert "Error: Invalid assertion type 'invalid_type'" in strip_ansi(result.stdout)
    assert "Valid assertion types:" in strip_ansi(result.stdout)
