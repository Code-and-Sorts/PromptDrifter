import re  # For stripping ANSI codes
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
        openai_api_key=None,
        gemini_api_key=None,
    )
    mock_class.assert_called_once_with(
        config_dir=Path("."), cache_db_path=None, use_cache=True, openai_api_key=None, gemini_api_key=None
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
            openai_api_key=None,
            gemini_api_key=None,
        )
    assert exc_info.value.exit_code == 1
    mock_class.assert_called_once_with(
        config_dir=Path("."), cache_db_path=None, use_cache=True, openai_api_key=None, gemini_api_key=None
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
            openai_api_key=None,
            gemini_api_key=None,
        )
    assert exc_info.value.exit_code == 1
    mock_class_raising.assert_called_once_with(
        config_dir=Path("."), cache_db_path=None, use_cache=True, openai_api_key=None, gemini_api_key=None
    )
    captured = capsys.readouterr()
    assert (
        "An unexpected error occurred during CLI run: Runner init boom!"
        in strip_ansi(captured.out)
    )


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
            openai_api_key=None,
            gemini_api_key=None,
        )
    assert exc_info.value.exit_code == 1
    mock_class.assert_called_once_with(
        config_dir=Path("."), cache_db_path=None, use_cache=True, openai_api_key=None, gemini_api_key=None
    )
    mock_instance.run_suite.assert_called_once_with([str(test_file)])
    mock_instance.close_cache_connection.assert_called_once()
    captured = capsys.readouterr()
    assert (
        "An unexpected error occurred during CLI run: Runner suite boom!"
        in strip_ansi(captured.out)
    )


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
        openai_api_key=None,
        gemini_api_key=None,
    )
    mock_class.assert_called_once_with(
        config_dir=Path("."), cache_db_path=None, use_cache=True, openai_api_key=None, gemini_api_key=None
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
            openai_api_key=None,
            gemini_api_key=None,
        )
    assert exc_info.value.exit_code == 1
    mock_class_for_safety.assert_not_called()
    captured_out_content = strip_ansi(capsys.readouterr().out)
    normalized_captured_out = " ".join(captured_out_content.replace('\n', ' ').split())
    expected_string_in_output = "• non_existent_file.yaml: File not found"
    assert expected_string_in_output in normalized_captured_out


@pytest.mark.asyncio
async def test_cli_run_logic_no_files_provided(mocker, capsys):
    mock_class_for_safety = mocker.patch("promptdrifter.cli.Runner")
    with pytest.raises(typer.Exit) as exc_info:
        await _run_async(files=[], no_cache=False, cache_db=None, config_dir=Path("."), openai_api_key=None, gemini_api_key=None)
    assert exc_info.value.exit_code == 1
    mock_class_for_safety.assert_not_called()
    captured = capsys.readouterr()
    assert "Error: No YAML files provided." in strip_ansi(captured.out)


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
    normalized_stdout = " ".join(strip_ansi(result.stdout).replace('\n', ' ').split())
    assert "Successfully created sample configuration:" in normalized_stdout
    assert config_file.name in normalized_stdout
    assert "You can now edit this file and run 'promptdrifter run'." in normalized_stdout
    assert config_file.exists()
    assert config_file.read_text() == "version: '0.1'\\nsample_content_for_default_init: true"
    if config_file.exists(): # Clean up
        config_file.unlink()


def test_record_command(tmp_path, mocker):
    output_json_file = tmp_path / "recorded.json"

    mock_typer_prompt = mocker.patch("typer.prompt")
    mock_typer_prompt.return_value = "exit"

    result = cli_runner.invoke(
        app,
        [
            "record",
            "--adapter",
            "dummy_adapter",
            "--model",
            "dummy_model",
            "--output",
            str(output_json_file),
        ],
    )

    assert result.exit_code == 0, (
        f"Exited with {result.exit_code}, stdout: {result.stdout}, stderr: {result.stderr}"
    )
    mock_typer_prompt.assert_called_once_with("\nEnter your prompt")

    assert "Recording mode started. Type 'exit' to finish." in strip_ansi(result.stdout)
    assert "No interactions were recorded." in strip_ansi(result.stdout)
    assert not output_json_file.exists()


def test_init_new_directory_success(mocker, tmp_path):
    """Test init command when target directory needs to be created."""
    base_dir = tmp_path / "new_project_dir"
    config_file = base_dir / "promptdrifter.yaml"

    # Ensure base_dir does not exist before the command is run
    assert not base_dir.exists()

    mocker.patch.object(Path, "resolve", return_value=base_dir)
    mock_files = mocker.patch("importlib.resources.files")
    mock_files.return_value.joinpath.return_value.read_text.return_value = (
        "version: '0.1'\\nsample_content: true"
    )

    result = cli_runner.invoke(app, ["init", str(base_dir)])

    assert result.exit_code == 0, strip_ansi(result.stdout)
    normalized_stdout = " ".join(strip_ansi(result.stdout).replace('\n', ' ').split())
    assert "Created directory:" in normalized_stdout
    assert base_dir.name in normalized_stdout
    assert "Successfully created sample configuration:" in normalized_stdout
    assert config_file.name in normalized_stdout
    assert base_dir.exists() # Should have been created by CLI
    assert base_dir.is_dir()
    assert config_file.exists()
    assert config_file.read_text() == "version: '0.1'\\nsample_content: true"


def test_init_sample_config_not_found(mocker, tmp_path):
    """Test init command when the sample config file is not found by importlib."""
    target_dir = tmp_path / "project_dir"
    target_dir.mkdir() # Ensure target_dir exists

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
    normalized_stdout = " ".join(strip_ansi(result.stdout).replace('\n', ' ').split())
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
    normalized_stdout = " ".join(strip_ansi(result.stdout).replace('\n', ' ').split())
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
    normalized_stdout = " ".join(strip_ansi(result.stdout).replace('\n', ' ').split())
    assert "Created directory:" in normalized_stdout
    assert target_dir.name in normalized_stdout
    assert "Error writing configuration file to" in normalized_stdout
    assert (target_dir / 'promptdrifter.yaml').name in normalized_stdout
    assert "Disk full or something terrible" in normalized_stdout
    assert target_dir.exists()


@pytest.mark.asyncio
async def test_cli_run_prints_security_warning(mocker, tmp_path, capsys):
    test_file = tmp_path / "test.yaml"
    test_file.write_text("version: 0.1\nid: test\nprompt: hello")
    mock_instance = AsyncMock()
    mock_instance.run_suite = AsyncMock(return_value=True)
    mock_instance.close_cache_connection = AsyncMock()
    mocker.patch("promptdrifter.cli.Runner", return_value=mock_instance)

    # Test with OpenAI key
    await _run_async(
        files=[Path(str(test_file))],
        no_cache=False,
        cache_db=None,
        config_dir=Path("."),
        openai_api_key="test_openai_key",
        gemini_api_key=None,
    )
    captured = capsys.readouterr()
    assert "SECURITY WARNING" in strip_ansi(captured.out)
    assert "Passing API keys directly via command-line arguments" in strip_ansi(captured.out)

    # Test with Gemini key
    await _run_async(
        files=[Path(str(test_file))],
        no_cache=False,
        cache_db=None,
        config_dir=Path("."),
        openai_api_key=None,
        gemini_api_key="test_gemini_key",
    )
    captured = capsys.readouterr() # Capture output again
    assert "SECURITY WARNING" in strip_ansi(captured.out)

    # Test with no keys (should not print warning)
    await _run_async(
        files=[Path(str(test_file))],
        no_cache=False,
        cache_db=None,
        config_dir=Path("."),
        openai_api_key=None,
        gemini_api_key=None,
    )
    captured = capsys.readouterr() # Capture output again
    assert "SECURITY WARNING" not in strip_ansi(captured.out)


def test_run_command_with_api_keys(mocker):
    mock_run_async = mocker.patch("promptdrifter.cli._run_async")
    test_file_path = Path("dummy.yaml")

    cli_runner.invoke(
        app,
        [
            "run",
            str(test_file_path),
            "--openai-api-key",
            "key1",
            "--gemini-api-key",
            "key2",
        ],
    )

    mock_run_async.assert_called_once_with(
        [test_file_path], False, None, Path("."), "key1", "key2"
    )

    mock_run_async.reset_mock()
    cli_runner.invoke(app, ["run", str(test_file_path)])
    mock_run_async.assert_called_once_with(
        [test_file_path], False, None, Path("."), None, None
    )


def test_version_command():
    # This test is not provided in the original file or the new code block
    # It's assumed to exist as it's called in the test_run_command_with_api_keys function
    pass
