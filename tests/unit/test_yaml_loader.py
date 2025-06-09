from pathlib import Path

import pytest
import yaml

from promptdrifter.models.config import AdapterConfig, PromptDrifterConfig, TestCase
from promptdrifter.yaml_loader import YamlFileLoader

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "yaml"


@pytest.fixture
def loader() -> YamlFileLoader:
    return YamlFileLoader()


def test_load_valid_yaml(loader: YamlFileLoader):
    valid_file = FIXTURES_DIR / "valid_schema_compliant.yaml"
    config = loader.load_and_validate_yaml(valid_file)

    assert isinstance(config, PromptDrifterConfig)
    assert config.version == "0.1"
    assert len(config.tests) == 1

    test_case = config.tests[0]
    assert isinstance(test_case, TestCase)
    assert test_case.id == "test-example-valid"
    assert test_case.prompt == "This is a valid prompt for {{subject}}."
    assert test_case.inputs == {"subject": "testing"}
    assert test_case.expect_exact == "This is the exact expected output."
    assert test_case.tags == ["smoke", "validation"]
    assert test_case.expect_regex is None

    assert len(test_case.adapter_configurations) == 1
    adapter_conf = test_case.adapter_configurations[0]
    assert isinstance(adapter_conf, AdapterConfig)
    assert adapter_conf.adapter_type == "openai"
    assert adapter_conf.model == "gpt-3.5-turbo"
    assert adapter_conf.temperature == 0.7
    assert adapter_conf.max_tokens == 100
    assert adapter_conf.model_extra == {}


def test_load_yaml_with_extra_adapter_params(loader: YamlFileLoader):
    yaml_content_with_extra = """
version: "0.1"
adapters:
  - id: "test-with-extra"
    prompt: "Test prompt"
    expect_exact: "Expected result"
    adapter:
      - type: "openai"
        model: "gpt-4"
        custom_param: "value1"
        another_custom: 123
"""
    test_file = FIXTURES_DIR / "temp_extra_params.yaml"
    test_file.write_text(yaml_content_with_extra)

    config = loader.load_and_validate_yaml(test_file)

    adapter_config = config.tests[0].adapter_configurations[0].model_dump(exclude_unset=True)
    assert "custom_param" in adapter_config
    assert adapter_config["custom_param"] == "value1"
    assert "another_custom" in adapter_config
    assert adapter_config["another_custom"] == 123

    test_file.unlink()


def test_load_invalid_missing_required_field_in_test_case(loader: YamlFileLoader):
    invalid_content = """
version: "0.1"
adapters:
  - id: "missing-prompt"
    # prompt: "is missing"
    adapter:
      - type: "openai"
        model: "gpt-3.5-turbo"
"""
    invalid_file = FIXTURES_DIR / "temp_missing_prompt.yaml"
    invalid_file.write_text(invalid_content)

    with pytest.raises(ValueError) as excinfo:
        loader.load_and_validate_yaml(invalid_file)

    error_str = str(excinfo.value)
    assert "Configuration Error" in error_str
    assert f"in '{invalid_file}'" in error_str
    assert "'prompt' is a required property" in error_str

    invalid_file.unlink()


def test_load_invalid_wrong_type_for_field(loader: YamlFileLoader):
    invalid_content = """
version: "0.1"
adapters:
  - id: "wrong-type-test"
    prompt: "Test prompt"
    expect_exact: "Expected result"
    adapter:
      - type: "openai"
        model: "gpt-3.5-turbo"
        max_tokens: "not-an-integer"
"""
    invalid_file = FIXTURES_DIR / "temp_wrong_type.yaml"
    invalid_file.write_text(invalid_content)

    with pytest.raises(ValueError) as excinfo:
        loader.load_and_validate_yaml(invalid_file)

    error_str = str(excinfo.value)
    assert "Configuration Error" in error_str
    assert f"in '{invalid_file}'" in error_str
    assert "JSON Schema validation failed" in error_str
    assert any(term in error_str for term in ["max_tokens", "not-an-integer"])

    invalid_file.unlink()


def test_load_invalid_version(loader: YamlFileLoader):
    invalid_content = 'version: "0.2"\nadapters: []'
    invalid_file = FIXTURES_DIR / "temp_invalid_version.yaml"
    invalid_file.write_text(invalid_content)
    with pytest.raises(ValueError) as excinfo:
        loader.load_and_validate_yaml(invalid_file)
    error_str = str(excinfo.value)
    assert "Configuration Error" in error_str
    assert "Schema version error" in error_str
    assert "Unsupported schema version: 0.2" in error_str
    invalid_file.unlink()


def test_load_multiple_expectations_error(loader: YamlFileLoader):
    yaml_content = """
version: "0.1"
adapters:
  - id: "multi-expect"
    prompt: "Test prompt"
    expect_exact: "exact string"
    expect_regex: "regex_pattern"
    adapter:
      - type: "openai"
        model: "gpt-3.5-turbo"
"""
    test_file = FIXTURES_DIR / "temp_multi_expect.yaml"
    test_file.write_text(yaml_content)
    with pytest.raises(ValueError) as excinfo:
        loader.load_and_validate_yaml(test_file)
    error_str = str(excinfo.value)
    assert "Configuration Error" in error_str
    assert "JSON Schema validation failed" in error_str
    assert "expect_exact" in error_str
    assert "expect_regex" in error_str
    test_file.unlink()


def test_load_empty_yaml(loader: YamlFileLoader):
    empty_file = FIXTURES_DIR / "empty.yaml"
    empty_file.write_text("")

    with pytest.raises(ValueError) as excinfo:
        loader.load_and_validate_yaml(empty_file)

    error_str = str(excinfo.value).lower()
    assert (
        "input should be a dictionary" in error_str
        or "input should be an object" in error_str
        or "data (got none from yaml)" in error_str
        or "field required" in error_str
        or "version\n  field required" in error_str
        or "tests\n  field required" in error_str
    ), f"Unexpected error message for empty YAML: {excinfo.value}"
    assert f"in '{str(empty_file)}'" in str(excinfo.value)


def test_load_malformed_yaml(loader: YamlFileLoader):
    malformed_file = FIXTURES_DIR / "malformed.yaml"
    malformed_file.write_text(
        "key: value\n another_key: \n  - subkey_no_value_then_bad_indent\n key3 :val3"
    )

    with pytest.raises(ValueError) as excinfo:
        loader.load_and_validate_yaml(malformed_file)
    assert "Error parsing YAML file" in str(excinfo.value)
    assert str(malformed_file) in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, yaml.YAMLError)


def test_load_non_existent_yaml(loader: YamlFileLoader):
    non_existent_file = FIXTURES_DIR / "i_do_not_exist.yaml"
    if non_existent_file.exists():
        non_existent_file.unlink()

    with pytest.raises(FileNotFoundError) as excinfo:
        loader.load_and_validate_yaml(non_existent_file)
    assert str(non_existent_file) in str(excinfo.value)
