import re


def exact_match(expected_output: str, actual_output: str) -> bool:
    return expected_output == actual_output


def regex_match(regex_pattern: str, actual_output: str) -> bool:
    try:
        compiled_regex = re.compile(regex_pattern)
        return bool(compiled_regex.search(actual_output))
    except re.error:
        return False


def expect_substring(substring: str, actual_output: str) -> bool:
    return substring in actual_output


def expect_substring_case_insensitive(substring: str, actual_output: str) -> bool:
    return substring.lower() in actual_output.lower()
