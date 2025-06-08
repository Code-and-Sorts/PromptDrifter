from unittest import mock

import jsonschema
import pytest

from promptdrifter.schema.constants import SCHEMA_VERSIONS
from promptdrifter.schema.validator import get_schema_path, load_schema, validate_config


def test_get_schema_path_current_version():
    path = get_schema_path()
    assert path.exists()
    assert path.name == "schema.json"
    assert path.parent.name == f"v{SCHEMA_VERSIONS.current_version}"


def test_get_schema_path_specific_version():
    version = "0.1"
    path = get_schema_path(version)
    assert path.exists()
    assert path.name == "schema.json"
    assert path.parent.name == f"v{version}"


def test_get_schema_path_invalid_version():
    invalid_version = "999.999"
    with pytest.raises(ValueError) as excinfo:
        get_schema_path(invalid_version)

    error_msg = str(excinfo.value)
    assert "Unsupported schema version" in error_msg
    assert invalid_version in error_msg
    assert ", ".join(SCHEMA_VERSIONS.supported_versions) in error_msg


def test_load_schema_current_version():
    schema = load_schema()
    assert isinstance(schema, dict)
    assert "$schema" in schema
    assert "title" in schema
    assert "properties" in schema
    assert "version" in schema["properties"]
    assert "adapters" in schema["properties"]


def test_load_schema_specific_version():
    version = "0.1"
    schema = load_schema(version)
    assert isinstance(schema, dict)
    assert "$schema" in schema
    assert "title" in schema
    assert "properties" in schema
    assert "version" in schema["properties"]
    assert "const" in schema["properties"]["version"]
    assert schema["properties"]["version"]["const"] == version


def test_load_schema_nonexistent_file():
    with mock.patch("pathlib.Path.exists", return_value=True):
        with mock.patch("builtins.open", mock.mock_open()) as m:
            m.side_effect = FileNotFoundError("No such file")
            with pytest.raises(FileNotFoundError):
                load_schema("0.1")


def test_validate_config_valid():
    valid_config = {
        "version": "0.1",
        "adapters": [
            {
                "id": "test-adapter",
                "prompt": "Test prompt",
                "expect_exact": "Expected result",
                "adapter": [
                    {
                        "type": "openai",
                        "model": "gpt-3.5-turbo"
                    }
                ]
            }
        ]
    }

    validate_config(valid_config)


def test_validate_config_invalid_structure():
    invalid_config = {
        "version": "0.1",
        "adapters": [
            {
                "id": "test-adapter",
                "adapter": [
                    {
                        "type": "openai",
                        "model": "gpt-3.5-turbo"
                    }
                ]
            }
        ]
    }

    with pytest.raises(jsonschema.exceptions.ValidationError) as excinfo:
        validate_config(invalid_config)

    assert "validation failed" in str(excinfo.value)
    assert "'prompt' is a required property" in str(excinfo.value)


def test_validate_config_invalid_version():
    invalid_config = {
        "version": "9999.9999",
        "adapters": []
    }

    with pytest.raises(ValueError) as excinfo:
        validate_config(invalid_config)

    assert "Unsupported schema version" in str(excinfo.value)


def test_validate_config_no_version():
    no_version_config = {
        "adapters": []
    }

    with pytest.raises(ValueError) as excinfo:
        validate_config(no_version_config)

    assert "No version specified" in str(excinfo.value)


@mock.patch("promptdrifter.schema.validator.load_schema")
def test_validate_config_explicit_version(mock_load_schema):
    mock_schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "Test Schema",
        "type": "object",
        "properties": {
            "version": {"type": "string"},
            "adapters": {"type": "array"}
        },
        "required": ["version", "adapters"]
    }
    mock_load_schema.return_value = mock_schema

    config = {
        "version": "invalid",
        "adapters": [
            {
                "id": "test-adapter",
                "prompt": "Test prompt",
                "expect_exact": "Expected result",
                "adapter": [
                    {
                        "type": "openai",
                        "model": "gpt-3.5-turbo"
                    }
                ]
            }
        ]
    }

    validate_config(config, version="0.1")

    mock_load_schema.assert_called_once_with("0.1")
