import importlib.resources
import json
from pathlib import Path

import pytest

from promptdrifter.yaml_loader import YamlFileLoader

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "yaml"
SCHEMA_PATH_FROM_ROOT = Path("src/promptdrifter/schema/v0.1.json")


# Helper function to create a schema file
def _create_schema_file(dir_path: Path, filename: str, content: dict | str):
    schema_file = dir_path / filename
    schema_file.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, dict):
        schema_file.write_text(json.dumps(content))
    else: # Assume string for malformed JSON
        schema_file.write_text(content)
    return schema_file

VALID_SCHEMA_CONTENT = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Test Schema",
    "type": "object",
    "properties": {"version": {"type": "string"}},
    "required": ["version"],
}

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
    schema_input_path = Path("non_existent_schema_dir/schema.json")
    with pytest.raises(FileNotFoundError) as excinfo:
        YamlFileLoader(schema_path=schema_input_path)
    assert "Schema file not found" in str(excinfo.value)
    assert str(schema_input_path) in str(excinfo.value)


def test_auto_detect_latest_schema_single_file(mocker, tmp_path):
    mock_schema_root_dir = tmp_path / "promptdrifter_pkg"
    mock_schema_dir = mock_schema_root_dir / "schema"

    mocker.patch.object(importlib.resources, "files", return_value=mock_schema_root_dir)

    expected_schema_path = _create_schema_file(mock_schema_dir, "v0.1.json", VALID_SCHEMA_CONTENT)

    loader_instance = YamlFileLoader() # No schema_path provided
    assert loader_instance.schema_path == expected_schema_path
    assert loader_instance._schema == VALID_SCHEMA_CONTENT

def test_auto_detect_latest_schema_multiple_files(mocker, tmp_path):
    mock_schema_root_dir = tmp_path / "promptdrifter_pkg"
    mock_schema_dir = mock_schema_root_dir / "schema"
    mocker.patch.object(importlib.resources, "files", return_value=mock_schema_root_dir)

    _create_schema_file(mock_schema_dir, "v0.1.json", {"version": "0.1"})
    _create_schema_file(mock_schema_dir, "v0.10.0.json", {"version": "0.10.0"}) # latest
    _create_schema_file(mock_schema_dir, "v0.2.1.json", {"version": "0.2.1"})
    _create_schema_file(mock_schema_dir, "v0.0.1alpha.json", {"version": "0.0.1alpha"}) # invalid name part

    expected_schema_path = mock_schema_dir / "v0.10.0.json"

    loader_instance = YamlFileLoader()
    assert loader_instance.schema_path == expected_schema_path

def test_auto_detect_skips_invalid_and_non_json_files(mocker, tmp_path):
    mock_schema_root_dir = tmp_path / "promptdrifter_pkg"
    mock_schema_dir = mock_schema_root_dir / "schema"
    mocker.patch.object(importlib.resources, "files", return_value=mock_schema_root_dir)

    expected_schema_path = _create_schema_file(mock_schema_dir, "v1.0.json", VALID_SCHEMA_CONTENT)
    _create_schema_file(mock_schema_dir, "v_alpha.json", VALID_SCHEMA_CONTENT) # Invalid version format
    _create_schema_file(mock_schema_dir, "v.json", VALID_SCHEMA_CONTENT) # Invalid version format
    _create_schema_file(mock_schema_dir, "v1.beta-1.json", VALID_SCHEMA_CONTENT) # Invalid version format
    (mock_schema_dir / "not_a_schema.txt").write_text("hello")
    (mock_schema_dir / "schema.json").write_text(json.dumps(VALID_SCHEMA_CONTENT)) # Not starting with v

    loader_instance = YamlFileLoader()
    assert loader_instance.schema_path == expected_schema_path

def test_auto_detect_schema_dir_is_file(mocker, tmp_path):
    mock_schema_root_dir = tmp_path / "promptdrifter_pkg"
    mock_schema_dir_as_file = mock_schema_root_dir / "schema" # This will be a file
    mock_schema_dir_as_file.parent.mkdir(parents=True, exist_ok=True)
    mock_schema_dir_as_file.write_text("I am a file, not a directory.")

    mocker.patch.object(importlib.resources, "files", return_value=mock_schema_root_dir)

    with pytest.raises(RuntimeError) as excinfo:
        YamlFileLoader()
    assert "Failed to automatically determine the latest schema" in str(excinfo.value)
    assert "Schema directory not found or is not a directory" in str(excinfo.value)
    assert str(mock_schema_dir_as_file) in str(excinfo.value)


def test_auto_detect_no_valid_schema_files_in_dir(mocker, tmp_path):
    mock_schema_root_dir = tmp_path / "promptdrifter_pkg"
    mock_schema_dir = mock_schema_root_dir / "schema"
    mock_schema_dir.mkdir(parents=True, exist_ok=True) # Exists but empty or no valid files

    mocker.patch.object(importlib.resources, "files", return_value=mock_schema_root_dir)

    # Test 1: Empty schema directory
    with pytest.raises(RuntimeError) as excinfo:
        YamlFileLoader()
    assert "Failed to automatically determine the latest schema" in str(excinfo.value)
    assert f"No valid schema files (e.g., vX.Y.json) found in schema directory: {mock_schema_dir}" in str(excinfo.value)

    # Test 2: Directory with only non-matching files
    (mock_schema_dir / "readme.txt").write_text("info")
    _create_schema_file(mock_schema_dir, "my_schema.json", VALID_SCHEMA_CONTENT) # Doesn't start with 'v'
    with pytest.raises(RuntimeError) as excinfo:
        YamlFileLoader()
    assert "Failed to automatically determine the latest schema" in str(excinfo.value)
    assert f"No valid schema files (e.g., vX.Y.json) found in schema directory: {mock_schema_dir}" in str(excinfo.value)


def test_auto_detect_general_exception_during_discovery(mocker):
    # Mock importlib.resources.files to raise an unexpected error
    mocker.patch.object(importlib.resources, "files", side_effect=PermissionError("Access denied"))

    with pytest.raises(RuntimeError) as excinfo:
        YamlFileLoader()
    assert "Failed to automatically determine the latest schema: Access denied" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, PermissionError)

def test_load_schema_invalid_json_explicit_path(tmp_path):
    malformed_schema_file = _create_schema_file(tmp_path, "invalid_v0.1.json", "{'key': 'value',,}") # Malformed

    with pytest.raises(ValueError) as excinfo:
        YamlFileLoader(schema_path=malformed_schema_file)
    assert "Error decoding schema JSON" in str(excinfo.value)
    assert str(malformed_schema_file) in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, json.JSONDecodeError)

def test_load_schema_invalid_json_auto_detected(mocker, tmp_path):
    mock_schema_root_dir = tmp_path / "promptdrifter_pkg"
    mock_schema_dir = mock_schema_root_dir / "schema"
    mocker.patch.object(importlib.resources, "files", return_value=mock_schema_root_dir)

    _create_schema_file(mock_schema_dir, "v0.1.json", "{'key': 'value',,}") # Malformed

    with pytest.raises(ValueError) as excinfo:
        YamlFileLoader() # Auto-detection
    assert "Error decoding schema JSON" in str(excinfo.value)
    assert "v0.1.json" in str(excinfo.value) # Check if the path contains the filename
    assert isinstance(excinfo.value.__cause__, json.JSONDecodeError)

def test_yaml_loader_uses_explicit_path_when_provided(mocker, tmp_path):
    # This test ensures that if schema_path is provided, auto-detection is skipped.
    mock_explicit_schema_file = tmp_path / "my_schemas" / "explicit_v1.0.json"
    _create_schema_file(mock_explicit_schema_file.parent, mock_explicit_schema_file.name, VALID_SCHEMA_CONTENT)

    # Mock importlib.resources.files to raise an error if called.
    # If schema_path is respected, this mock should not be triggered.
    mock_files = mocker.patch.object(importlib.resources, "files", side_effect=AssertionError("Auto-detection should not run!"))

    loader_instance = YamlFileLoader(schema_path=mock_explicit_schema_file)
    assert loader_instance.schema_path == mock_explicit_schema_file
    assert loader_instance._schema == VALID_SCHEMA_CONTENT
    mock_files.assert_not_called() # Verify auto-detection path was indeed not taken.

# Test that the original FileNotFoundError in __init__ for schema_path (if provided) still works
def test_init_explicit_schema_not_found_directly():
    non_existent_path = Path("path/to/non_existent_schema.json")
    with pytest.raises(FileNotFoundError) as excinfo:
        YamlFileLoader(schema_path=non_existent_path)
    # This error comes from _load_schema, but is triggered by __init__ providing the bad path
    assert f"Schema file not found at {non_existent_path}" in str(excinfo.value)

def test_auto_detect_continue_on_version_parse_error(mocker, tmp_path):
    """Ensures the loop continues if a schema version string is unparsable."""
    mock_schema_root_dir = tmp_path / "promptdrifter_pkg"
    mock_schema_dir = mock_schema_root_dir / "schema"
    mocker.patch.object(importlib.resources, "files", return_value=mock_schema_root_dir)

    # This file should cause a ValueError during version parsing and be skipped
    _create_schema_file(mock_schema_dir, "vINVALID.json", {"title": "Invalid Version Schema"})
    # This valid schema should be picked up after the invalid one is skipped
    expected_schema_path = _create_schema_file(mock_schema_dir, "v0.1.json", VALID_SCHEMA_CONTENT)

    loader_instance = YamlFileLoader()
    assert loader_instance.schema_path == expected_schema_path
    assert loader_instance._schema == VALID_SCHEMA_CONTENT
