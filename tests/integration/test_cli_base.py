import re
import subprocess
from pathlib import Path
from typing import List, Optional

import pytest


class CLICommandResult:
    def __init__(self, exit_code: int, stdout: str, stderr: str):
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr

    def __repr__(self):
        return f"CLICommandResult(exit_code={self.exit_code}, stdout='{self.stdout[:50]}...', stderr='{self.stderr[:50]}...')"


def execute_cli_command(
    command: List[str],
    working_directory: Optional[Path] = None,
    environment_vars: Optional[dict] = None,
    stdin_input: Optional[str] = None
) -> CLICommandResult:
    try:
        result = subprocess.run(
            command,
            cwd=working_directory,
            env=environment_vars,
            input=stdin_input,
            capture_output=True,
            text=True,
            timeout=30
        )
        return CLICommandResult(
            exit_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"Command timed out: {' '.join(command)}")
    except Exception as e:
        pytest.fail(f"Command execution failed: {e}")


def strip_ansi_codes(text: str) -> str:
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


def verify_cli_success(result: CLICommandResult, expected_output_contains: Optional[str] = None):
    assert result.exit_code == 0, f"Command failed with exit code {result.exit_code}. stderr: {result.stderr}"
    if expected_output_contains:
        clean_stdout = strip_ansi_codes(result.stdout)
        assert expected_output_contains in clean_stdout, f"Expected '{expected_output_contains}' in output: {clean_stdout}"


def verify_cli_failure(result: CLICommandResult, expected_exit_code: int = 1, expected_error_contains: Optional[str] = None):
    assert result.exit_code == expected_exit_code, f"Expected exit code {expected_exit_code}, got {result.exit_code}"
    if expected_error_contains:
        combined_output = result.stderr + result.stdout
        assert expected_error_contains in combined_output, f"Expected '{expected_error_contains}' in error output: {combined_output}"


class CLICommands:
    INIT = "init"
    VALIDATE = "validate"
    MIGRATE = "migrate"
    TEST_DRIFT_TYPE = "test-drift-type"
    VERSION = "--version"
    HELP = "--help"
    RUN = "run"


class DriftTypes:
    EXACT_MATCH = "exact_match"
    REGEX_MATCH = "regex_match"
    EXPECT_SUBSTRING = "expect_substring"
    EXPECT_SUBSTRING_CASE_INSENSITIVE = "expect_substring_case_insensitive"
    TEXT_SIMILARITY = "text_similarity"
