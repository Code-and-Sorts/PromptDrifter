from pathlib import Path

import pytest

from promptdrifter.yaml_loader import YamlFileLoader

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "yaml"
SCHEMA_PATH_FROM_ROOT = Path("src/promptdrifter/schema/v0.1.json")


@pytest.fixture
def loader():
    return YamlFileLoader(schema_path=SCHEMA_PATH_FROM_ROOT)


def test_load_valid_yaml(loader):
    valid_file = FIXTURES_DIR / "valid_schema_compliant.yaml"
    data = loader.load_and_validate_yaml(valid_file)
    assert data["version"] == "0.1"
    assert len(data["adapters"]) == 1
    test_def = data["adapters"][0]
    assert test_def["id"] == "test-example-valid"
    assert test_def["prompt"] == "This is a valid prompt for {{subject}}."
    assert test_def["inputs"] == {"subject": "testing"}
    assert test_def["expect_exact"] == "This is the exact expected output."
    assert test_def["tags"] == ["smoke", "validation"]
    assert len(test_def["adapter"]) == 1
    adapter_config = test_def["adapter"][0]
    assert adapter_config["type"] == "openai"
    assert adapter_config["model"] == "gpt-3.5-turbo"
    assert adapter_config["temperature"] == 0.7
    assert adapter_config["max_tokens"] == 100


def test_load_invalid_missing_required(loader):
    invalid_file = FIXTURES_DIR / "invalid_missing_required.yaml"
    with pytest.raises(ValueError) as excinfo:
        loader.load_and_validate_yaml(invalid_file)
    error_str = str(excinfo.value)
    assert "Schema Validation Error" in error_str
    assert f"in '{invalid_file}'" in error_str
    assert "[red]'prompt' is a required property[/red]" in error_str
    assert "Location: At 'adapters -> 0' in your YAML." in error_str


def test_load_invalid_wrong_type(loader):
    invalid_file = FIXTURES_DIR / "invalid_wrong_type.yaml"
    with pytest.raises(ValueError) as excinfo:
        loader.load_and_validate_yaml(invalid_file)
    error_str = str(excinfo.value)
    assert "Schema Validation Error" in error_str
    assert f"in '{invalid_file}'" in error_str
    assert "[red]'not-an-integer' is not of type 'integer'[/red]" in error_str
    assert "Location: At 'adapters -> 0 -> adapter -> 0 -> max_tokens' in your YAML." in error_str


def test_load_empty_yaml(loader):
    empty_file = FIXTURES_DIR / "empty.yaml"
    with pytest.raises(ValueError) as excinfo:
        loader.load_and_validate_yaml(empty_file)
    error_str = str(excinfo.value)
    assert "Schema Validation Error" in error_str
    assert f"in '{empty_file}'" in error_str
    assert "[red]None is not of type 'object'[/red]" in error_str
    assert "Location: At the root of your YAML." in error_str


def test_load_malformed_yaml(loader):
    malformed_file = FIXTURES_DIR / "malformed.yaml"
    with pytest.raises(ValueError) as excinfo:
        loader.load_and_validate_yaml(malformed_file)
    assert "Error parsing YAML file" in str(excinfo.value)
    assert str(malformed_file) in str(excinfo.value)


def test_load_non_existent_yaml(loader):
    non_existent_file = FIXTURES_DIR / "i_do_not_exist.yaml"
    with pytest.raises(FileNotFoundError) as excinfo:
        loader.load_and_validate_yaml(non_existent_file)
    assert str(non_existent_file) in str(excinfo.value)


def test_schema_not_found():
    with pytest.raises(FileNotFoundError) as excinfo:
        YamlFileLoader(schema_path=Path("non_existent_schema_dir/schema.json"))
    assert "Schema file not found" in str(excinfo.value)
