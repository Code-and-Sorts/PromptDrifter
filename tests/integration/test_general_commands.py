from .test_cli_base import execute_cli_command, normalize_output, verify_cli_success


def test_version_flag_shows_version():
    result = execute_cli_command(["promptdrifter", "--version"])
    verify_cli_success(result, "Version:")


def test_version_short_flag_shows_version():
    result = execute_cli_command(["promptdrifter", "-v"])
    verify_cli_success(result, "Version:")


def test_main_help_command():
    result = execute_cli_command(["promptdrifter", "--help"])
    verify_cli_success(result, "Usage:")
    verify_cli_success(result, "╭─ Commands")


def test_no_args_shows_help():
    result = execute_cli_command(["promptdrifter"])
    assert result.exit_code in (0, 2), (
        f"Expected exit code 0 or 2, got {result.exit_code}. stderr: {result.stderr}"
    )
    clean_output = normalize_output(result.stdout + result.stderr)
    assert "Usage:" in clean_output, f"Expected 'Usage:' in output: {clean_output}"


def test_invalid_command():
    result = execute_cli_command(["promptdrifter", "nonexistent-command"])
    assert result.exit_code != 0


def test_version_shows_ascii_logo():
    result = execute_cli_command(["promptdrifter", "--version"])
    verify_cli_success(result, "▒")


def test_version_shows_package_version():
    result = execute_cli_command(["promptdrifter", "--version"])
    verify_cli_success(result, "Version:")
    assert any(char.isdigit() for char in result.stdout)
