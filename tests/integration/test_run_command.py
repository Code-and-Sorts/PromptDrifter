import os
from pathlib import Path

from .test_cli_base import (
    CLICommands,
    execute_cli_command,
    verify_cli_failure,
    verify_cli_success,
)


def test_run_no_files_provided():
    result = execute_cli_command(["promptdrifter", CLICommands.RUN])
    verify_cli_failure(result, expected_exit_code=2, expected_error_contains="Missing argument")


def test_run_nonexistent_file(nonexistent_yaml_file: Path):
    result = execute_cli_command(["promptdrifter", CLICommands.RUN, str(nonexistent_yaml_file)])
    verify_cli_failure(result, expected_error_contains="File not found")


def test_run_non_yaml_file(non_yaml_file: Path):
    result = execute_cli_command(["promptdrifter", CLICommands.RUN, str(non_yaml_file)])
    verify_cli_failure(result, expected_error_contains="Not a YAML file")


def test_run_malformed_yaml_file(malformed_yaml_file: Path):
    result = execute_cli_command(["promptdrifter", CLICommands.RUN, str(malformed_yaml_file)])
    verify_cli_failure(result)


def test_run_invalid_yaml_schema(invalid_yaml_file: Path):
    result = execute_cli_command(["promptdrifter", CLICommands.RUN, str(invalid_yaml_file)])
    verify_cli_failure(result)


def test_run_valid_yaml_no_api_keys(valid_yaml_file: Path):
    clean_env = {k: v for k, v in os.environ.items()
    if not k.endswith('_API_KEY')}

    result = execute_cli_command(
        ["promptdrifter", CLICommands.RUN, str(valid_yaml_file)],
        environment_vars=clean_env
    )
    verify_cli_failure(result)


def test_run_with_cache_db_option(temp_dir: Path, valid_yaml_file: Path):
    cache_db_path = temp_dir / "test_cache.db"
    clean_env = {k: v for k, v in os.environ.items()
    if not k.endswith('_API_KEY')}

    result = execute_cli_command([
        "promptdrifter", CLICommands.RUN,
        "--cache-db", str(cache_db_path),
        str(valid_yaml_file)
    ], environment_vars=clean_env)
    verify_cli_failure(result)


def test_run_with_no_cache_option(valid_yaml_file: Path):
    clean_env = {k: v for k, v in os.environ.items()
    if not k.endswith('_API_KEY')}

    result = execute_cli_command([
        "promptdrifter", CLICommands.RUN,
        "--no-cache",
        str(valid_yaml_file)
    ], environment_vars=clean_env)
    verify_cli_failure(result)


def test_run_with_config_dir_option(temp_config_dir: Path, valid_yaml_file: Path):
    clean_env = {k: v for k, v in os.environ.items()
    if not k.endswith('_API_KEY')}

    result = execute_cli_command([
        "promptdrifter", CLICommands.RUN,
        "--config-dir", str(temp_config_dir),
        str(valid_yaml_file)
    ], environment_vars=clean_env)
    verify_cli_failure(result)


def test_run_multiple_yaml_files_one_invalid(valid_yaml_file: Path, nonexistent_yaml_file: Path):
    result = execute_cli_command([
        "promptdrifter", CLICommands.RUN,
        str(valid_yaml_file), str(nonexistent_yaml_file)
    ])
    verify_cli_failure(result, expected_error_contains="File not found")


def test_run_multiple_valid_yaml_files_no_api_keys(valid_yaml_file: Path, multiple_adapters_yaml_file: Path):
    clean_env = {k: v for k, v in os.environ.items()
    if not k.endswith('_API_KEY')}

    result = execute_cli_command([
        "promptdrifter", CLICommands.RUN,
        str(valid_yaml_file), str(multiple_adapters_yaml_file)
    ], environment_vars=clean_env)
    verify_cli_failure(result)


def test_run_shows_security_warning_with_cli_api_key():
    result = execute_cli_command([
        "promptdrifter", CLICommands.RUN,
        "--openai-api-key", "fake-key-for-test",
        "/nonexistent/file.yaml"
    ])
    verify_cli_failure(result, expected_error_contains="SECURITY WARNING")


def test_run_api_key_from_env_no_warning(valid_yaml_file: Path):
    env_with_key = dict(os.environ)
    env_with_key["OPENAI_API_KEY"] = "fake-key-from-env"

    result = execute_cli_command([
        "promptdrifter", CLICommands.RUN, str(valid_yaml_file)
    ], environment_vars=env_with_key)
    verify_cli_failure(result)
    assert "SECURITY WARNING" not in result.stdout


def test_run_directory_instead_of_file(temp_dir: Path):
    result = execute_cli_command(["promptdrifter", CLICommands.RUN, str(temp_dir)])
    verify_cli_failure(result, expected_error_contains="Path is not a")


def test_run_empty_yaml_file(empty_yaml_file: Path):
    result = execute_cli_command(["promptdrifter", CLICommands.RUN, str(empty_yaml_file)])
    verify_cli_failure(result)


def test_run_command_help():
    result = execute_cli_command(["promptdrifter", CLICommands.RUN, "--help"])
    verify_cli_success(result, "Run a suite of prompt tests")


def test_run_shows_usage_examples_on_no_files():
    result = execute_cli_command(["promptdrifter", CLICommands.RUN])
    verify_cli_failure(result, expected_exit_code=2, expected_error_contains="promptdrifter run")
    verify_cli_failure(result, expected_exit_code=2, expected_error_contains="Missing argument")
