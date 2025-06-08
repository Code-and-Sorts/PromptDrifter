from pathlib import Path

from .test_cli_base import (
    CLICommands,
    execute_cli_command,
    verify_cli_failure,
    verify_cli_success,
)


def test_validate_valid_yaml_file(valid_yaml_file: Path):
    result = execute_cli_command(["promptdrifter", CLICommands.VALIDATE, str(valid_yaml_file)])
    verify_cli_success(result, "Valid")


def test_validate_invalid_yaml_missing_fields(invalid_yaml_file: Path):
    result = execute_cli_command(["promptdrifter", CLICommands.VALIDATE, str(invalid_yaml_file)])
    verify_cli_failure(result, expected_error_contains="Validation failed")


def test_validate_malformed_yaml_syntax(malformed_yaml_file: Path):
    result = execute_cli_command(["promptdrifter", CLICommands.VALIDATE, str(malformed_yaml_file)])
    verify_cli_failure(result, expected_error_contains="Validation failed")


def test_validate_empty_yaml_file(empty_yaml_file: Path):
    result = execute_cli_command(["promptdrifter", CLICommands.VALIDATE, str(empty_yaml_file)])
    verify_cli_failure(result, expected_error_contains="Validation failed")


def test_validate_nonexistent_file(nonexistent_yaml_file: Path):
    result = execute_cli_command(["promptdrifter", CLICommands.VALIDATE, str(nonexistent_yaml_file)])
    verify_cli_failure(result, expected_error_contains="File not found")


def test_validate_non_yaml_file(non_yaml_file: Path):
    result = execute_cli_command(["promptdrifter", CLICommands.VALIDATE, str(non_yaml_file)])
    verify_cli_failure(result, expected_error_contains="Validation failed")


def test_validate_multiple_files_mixed_validity(valid_yaml_file: Path, invalid_yaml_file: Path):
    result = execute_cli_command([
        "promptdrifter", CLICommands.VALIDATE,
        str(valid_yaml_file), str(invalid_yaml_file)
    ])
    verify_cli_failure(result, expected_error_contains="Validation failed")


def test_validate_multiple_valid_files(valid_yaml_file: Path, multiple_adapters_yaml_file: Path):
    result = execute_cli_command([
        "promptdrifter", CLICommands.VALIDATE,
        str(valid_yaml_file), str(multiple_adapters_yaml_file)
    ])
    verify_cli_success(result, "All configuration files are valid")


def test_validate_invalid_version_yaml_file(invalid_version_yaml_file: Path):
    result = execute_cli_command(["promptdrifter", CLICommands.VALIDATE, str(invalid_version_yaml_file)])
    verify_cli_failure(result, expected_error_contains="Validation failed")


def test_validate_command_help():
    result = execute_cli_command(["promptdrifter", CLICommands.VALIDATE, "--help"])
    verify_cli_success(result, "Validate YAML configuration files")
