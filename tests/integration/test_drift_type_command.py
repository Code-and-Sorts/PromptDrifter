import pytest

from .test_cli_base import (
    CLICommands,
    DriftTypes,
    execute_cli_command,
    verify_cli_failure,
    verify_cli_success,
)


def test_exact_match_assertion_true():
    result = execute_cli_command([
        "promptdrifter", CLICommands.TEST_DRIFT_TYPE,
        DriftTypes.EXACT_MATCH, "hello world", "hello world"
    ])
    verify_cli_success(result, "True")


def test_exact_match_assertion_false():
    result = execute_cli_command([
        "promptdrifter", CLICommands.TEST_DRIFT_TYPE,
        DriftTypes.EXACT_MATCH, "hello world", "Hello World"
    ])
    verify_cli_success(result, "False")


def test_regex_match_assertion_true():
    result = execute_cli_command([
        "promptdrifter", CLICommands.TEST_DRIFT_TYPE,
        DriftTypes.REGEX_MATCH, "^Hello.*world$", "Hello beautiful world"
    ])
    verify_cli_success(result, "True")


def test_regex_match_assertion_false():
    result = execute_cli_command([
        "promptdrifter", CLICommands.TEST_DRIFT_TYPE,
        DriftTypes.REGEX_MATCH, "^Hello.*world$", "Hi beautiful world"
    ])
    verify_cli_success(result, "False")


def test_expect_substring_assertion_true():
    result = execute_cli_command([
        "promptdrifter", CLICommands.TEST_DRIFT_TYPE,
        DriftTypes.EXPECT_SUBSTRING, "beautiful", "Hello beautiful world"
    ])
    verify_cli_success(result, "True")


def test_expect_substring_assertion_false():
    result = execute_cli_command([
        "promptdrifter", CLICommands.TEST_DRIFT_TYPE,
        DriftTypes.EXPECT_SUBSTRING, "amazing", "Hello beautiful world"
    ])
    verify_cli_success(result, "False")


def test_expect_substring_case_insensitive_true():
    result = execute_cli_command([
        "promptdrifter", CLICommands.TEST_DRIFT_TYPE,
        DriftTypes.EXPECT_SUBSTRING_CASE_INSENSITIVE, "BEAUTIFUL", "Hello beautiful world"
    ])
    verify_cli_success(result, "True")


def test_expect_substring_case_insensitive_false():
    result = execute_cli_command([
        "promptdrifter", CLICommands.TEST_DRIFT_TYPE,
        DriftTypes.EXPECT_SUBSTRING_CASE_INSENSITIVE, "AMAZING", "Hello beautiful world"
    ])
    verify_cli_success(result, "False")


def test_text_similarity_high_score():
    result = execute_cli_command([
        "promptdrifter", CLICommands.TEST_DRIFT_TYPE,
        DriftTypes.TEXT_SIMILARITY, "The quick brown fox", "The quick brown fox"
    ])
    verify_cli_success(result)
    assert "1.0000" in result.stdout


def test_text_similarity_medium_score():
    result = execute_cli_command([
        "promptdrifter", CLICommands.TEST_DRIFT_TYPE,
        DriftTypes.TEXT_SIMILARITY, "The quick brown fox", "The fast brown fox"
    ])
    verify_cli_success(result)
    assert any(score in result.stdout for score in ["0.8", "0.9"])


def test_text_similarity_low_score():
    result = execute_cli_command([
        "promptdrifter", CLICommands.TEST_DRIFT_TYPE,
        DriftTypes.TEXT_SIMILARITY, "The quick brown fox", "Completely different text"
    ])
    verify_cli_success(result)


def test_invalid_drift_type():
    result = execute_cli_command([
        "promptdrifter", CLICommands.TEST_DRIFT_TYPE,
        "invalid_type", "expected", "actual"
    ])
    verify_cli_failure(result, expected_error_contains="Invalid assertion type")


def test_regex_match_invalid_pattern():
    result = execute_cli_command([
        "promptdrifter", CLICommands.TEST_DRIFT_TYPE,
        DriftTypes.REGEX_MATCH, "[invalid regex", "some text"
    ])
    verify_cli_success(result)


def test_drift_type_command_shows_assertion_details():
    result = execute_cli_command([
        "promptdrifter", CLICommands.TEST_DRIFT_TYPE,
        DriftTypes.EXACT_MATCH, "test input", "test input"
    ])
    verify_cli_success(result, "Assertion: exact_match")
    verify_cli_success(result, "Expected: test input")
    verify_cli_success(result, "Actual: test input")


def test_drift_type_command_help():
    result = execute_cli_command([
        "promptdrifter", CLICommands.TEST_DRIFT_TYPE, "--help"
    ])
    verify_cli_success(result, "Test a drift type with the provided inputs")


@pytest.mark.parametrize("drift_type,expected,actual,should_pass", [
    (DriftTypes.EXACT_MATCH, "hello", "hello", True),
    (DriftTypes.EXACT_MATCH, "hello", "Hello", False),
    (DriftTypes.REGEX_MATCH, "^test.*", "test123", True),
    (DriftTypes.REGEX_MATCH, "^test.*", "nottest", False),
    (DriftTypes.EXPECT_SUBSTRING, "sub", "substring", True),
    (DriftTypes.EXPECT_SUBSTRING, "sub", "nothing", False),
    (DriftTypes.EXPECT_SUBSTRING_CASE_INSENSITIVE, "SUB", "substring", True),
    (DriftTypes.EXPECT_SUBSTRING_CASE_INSENSITIVE, "SUB", "nothing", False),
])
def test_all_drift_types_comprehensive(drift_type: str, expected: str, actual: str, should_pass: bool):
    result = execute_cli_command([
        "promptdrifter", CLICommands.TEST_DRIFT_TYPE,
        drift_type, expected, actual
    ])
    verify_cli_success(result)

    if should_pass:
        assert "True" in result.stdout
    else:
        assert "False" in result.stdout
