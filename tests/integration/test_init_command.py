from pathlib import Path

from .test_cli_base import (
    CLICommands,
    execute_cli_command,
    verify_cli_failure,
    verify_cli_success,
)


def test_init_creates_valid_config_file(temp_dir: Path):
    result = execute_cli_command(
        ["promptdrifter", CLICommands.INIT, str(temp_dir)],
        working_directory=temp_dir
    )
    verify_cli_success(result, "Successfully created sample configuration")

    config_file = temp_dir / "promptdrifter.yaml"
    assert config_file.exists()

    validate_result = execute_cli_command(["promptdrifter", CLICommands.VALIDATE, str(config_file)])
    verify_cli_success(validate_result, "Valid")


def test_init_current_directory_creates_config(temp_dir: Path):
    result = execute_cli_command(
        ["promptdrifter", CLICommands.INIT],
        working_directory=temp_dir
    )
    verify_cli_success(result, "Successfully created sample configuration")

    config_file = temp_dir / "promptdrifter.yaml"
    assert config_file.exists()


def test_init_nonexistent_directory_creates_path(temp_dir: Path):
    new_project_dir = temp_dir / "new_project"
    result = execute_cli_command(["promptdrifter", CLICommands.INIT, str(new_project_dir)])
    verify_cli_success(result, "Created directory")
    verify_cli_success(result, "Successfully created sample configuration")

    assert new_project_dir.exists()
    assert (new_project_dir / "promptdrifter.yaml").exists()


def test_init_existing_config_file_skips_creation(temp_dir: Path):
    config_file = temp_dir / "promptdrifter.yaml"
    config_file.write_text("existing config")

    result = execute_cli_command(
        ["promptdrifter", CLICommands.INIT, str(temp_dir)],
        working_directory=temp_dir
    )
    verify_cli_success(result, "already exists. Skipping")

    assert config_file.read_text() == "existing config"


def test_init_target_path_is_file_fails(temp_dir: Path):
    existing_file = temp_dir / "existing.txt"
    existing_file.write_text("not a directory")

    result = execute_cli_command(["promptdrifter", CLICommands.INIT, str(existing_file)])
    verify_cli_failure(result, expected_error_contains="exists but is not a directory")


def test_init_creates_nested_directory_structure(temp_dir: Path):
    nested_dir = temp_dir / "projects" / "my_tests" / "prompt_tests"
    result = execute_cli_command(["promptdrifter", CLICommands.INIT, str(nested_dir)])
    verify_cli_success(result, "Created directory")
    verify_cli_success(result, "Successfully created sample configuration")

    assert nested_dir.exists()
    assert (nested_dir / "promptdrifter.yaml").exists()


def test_init_command_help():
    result = execute_cli_command(["promptdrifter", CLICommands.INIT, "--help"])
    verify_cli_success(result, "Initialize a new promptdrifter project")
