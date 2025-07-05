from unittest import mock

import pytest
from jsonschema.exceptions import ValidationError

from promptdrifter.schema.migration import get_migration_function, migrate_config


def test_get_migration_function_same_version():
    result = get_migration_function("0.1", "0.1")
    assert result is None


def test_get_migration_function_unsupported_path():
    with pytest.raises(ValueError) as excinfo:
        get_migration_function("0.1", "0.2")

    error_msg = str(excinfo.value)
    assert "No migration path available" in error_msg
    assert "0.1 to 0.2" in error_msg


def test_migrate_config_no_version():
    config = {
        "adapters": []
    }

    with pytest.raises(ValueError) as excinfo:
        migrate_config(config)

    assert "No version specified" in str(excinfo.value)


def test_migrate_config_unsupported_target():
    config = {
        "version": "0.1",
        "adapters": []
    }

    with pytest.raises(ValueError) as excinfo:
        migrate_config(config, target_version="9999.9999")

    assert "No migration path available" in str(excinfo.value)


def test_migrate_config_invalid_source():
    with mock.patch("promptdrifter.schema.migration.validate_config") as mock_validate:
        mock_validate.side_effect = ValidationError("Invalid config")

        config = {
            "version": "0.1",
            "adapters": [{}]
        }

        with pytest.raises(ValidationError):
            migrate_config(config, target_version="0.2")

def test_migrate_config_with_migration_function():
    config = {
        "version": "0.1",
        "adapters": []
    }

    expected_result = {
        "version": "0.2",
        "adapters": []
    }

    with mock.patch("promptdrifter.schema.migration.validate_config") as mock_validate:
        with mock.patch("promptdrifter.schema.migration.get_migration_function") as mock_get_migration:
            mock_migration_func = mock.Mock(return_value=expected_result)
            mock_get_migration.return_value = mock_migration_func

            result = migrate_config(config, target_version="0.2")

            mock_migration_func.assert_called_once_with(config)

            assert mock_validate.call_count == 2

            assert result == expected_result


@mock.patch("promptdrifter.schema.validator.validate_config")
def test_migrate_config_same_version(mock_validate):
    config = {
        "version": "0.1",
        "adapters": []
    }

    result = migrate_config(config)

    assert result is config
    assert result == config

    mock_validate.assert_not_called()
