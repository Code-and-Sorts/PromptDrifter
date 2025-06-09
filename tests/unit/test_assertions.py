import pytest

from promptdrifter.drift_types import (
    exact_match,
    expect_substring,
    expect_substring_case_insensitive,
    regex_match,
)


# Tests for exact_match
def test_exact_match_positive():
    assert exact_match("hello world", "hello world") is True


def test_exact_match_negative():
    assert exact_match("hello world", "hello there") is False


def test_exact_match_case_sensitive():
    assert exact_match("Hello World", "hello world") is False


def test_exact_match_empty_strings():
    assert exact_match("", "") is True


def test_exact_match_one_empty():
    assert exact_match("abc", "") is False
    assert exact_match("", "abc") is False


# Tests for regex_match
def test_regex_match_positive():
    assert regex_match(r"^hello\s+world$", "hello    world") is True


def test_regex_match_negative():
    assert regex_match(r"^hello\s+world$", "hello there") is False


def test_regex_match_case_sensitive_default():
    # Default re.search is case sensitive
    assert regex_match(r"Hello", "hello") is False


def test_regex_match_case_insensitive_flag():
    # To test case insensitivity, the pattern itself must include the flag
    assert regex_match(r"(?i)hello", "HELLO") is True


def test_regex_match_empty_string_pattern():
    # An empty pattern typically matches, but let's see re.compile behavior
    # re.compile('') is valid and will match any string (including empty) if used with re.match or re.search
    assert regex_match(r"", "abc") is True
    assert regex_match(r"", "") is True


def test_regex_match_empty_string_input():
    assert regex_match(r"^abc$", "") is False


def test_regex_match_invalid_pattern():
    # The function should handle re.error and return False
    assert regex_match(r"[invalid(", "some string") is False


def test_regex_match_partial_match():
    # regex_match uses re.search, so partial matches are fine
    assert regex_match(r"world", "hello world example") is True


def test_regex_match_full_string_anchor():
    assert regex_match(r"^world$", "hello world example") is False
    assert regex_match(r"^hello world example$", "hello world example") is True


# Tests for expect_substring
def test_expect_substring_positive():
    assert expect_substring("hello", "hello world") is True


def test_expect_substring_negative():
    assert expect_substring("hello", "world") is False


# Tests for expect_substring_case_insensitive
def test_expect_substring_case_insensitive_positive():
    assert expect_substring_case_insensitive("hello", "Hello World") is True


def test_expect_substring_case_insensitive_negative():
    assert expect_substring_case_insensitive("hello", "world") is False
