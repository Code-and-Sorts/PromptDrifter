from unittest import mock

import pytest
import yaml
from typer.testing import CliRunner

from promptdrifter.cli import app


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def valid_config_file(tmp_path):
    config = {
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

    config_file = tmp_path / "valid_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config, f)

    return config_file


@pytest.fixture
def invalid_config_file(tmp_path):
    config = {
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

    config_file = tmp_path / "invalid_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config, f)

    return config_file


@mock.patch("promptdrifter.yaml_loader.YamlFileLoader.load_and_validate_yaml")
def test_validate_command_success(mock_load_validate, runner, valid_config_file):
    mock_load_validate.return_value = None

    result = runner.invoke(app, ["validate", str(valid_config_file)])

    assert result.exit_code == 0
    assert "Valid" in result.stdout
    assert "All configuration files are valid" in result.stdout
    mock_load_validate.assert_called_once()


@mock.patch("promptdrifter.yaml_loader.YamlFileLoader.load_and_validate_yaml",
            side_effect=ValueError("Validation failed"))
def test_validate_command_validation_error(mock_load_validate, runner, valid_config_file):
    result = runner.invoke(app, ["validate", str(valid_config_file)])

    assert result.exit_code != 0
    assert "Validation failed" in result.stdout
    assert "Validation failed for one or more files" in result.stdout
    mock_load_validate.assert_called_once()


def test_validate_command_file_not_found(runner):
    result = runner.invoke(app, ["validate", "nonexistent_file.yaml"])

    assert result.exit_code != 0
    assert "File not found" in result.stdout
    assert "Validation failed for one or more files" in result.stdout


def test_validate_command_multiple_files(runner, valid_config_file, invalid_config_file):
    result = runner.invoke(app, ["validate", str(valid_config_file), str(invalid_config_file)])

    assert result.exit_code != 0
    assert "Valid" in result.stdout
    assert "Validation failed" in result.stdout
    assert "Validation failed for one or more files" in result.stdout
