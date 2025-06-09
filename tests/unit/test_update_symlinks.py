from unittest import mock

import pytest

from promptdrifter.schema.constants import SCHEMA_VERSIONS
from promptdrifter.schema.update_symlinks import update_symlinks


@mock.patch("os.symlink")
@mock.patch("pathlib.Path.is_symlink")
@mock.patch("pathlib.Path.unlink")
def test_update_symlinks_existing_symlink(mock_unlink, mock_is_symlink, mock_symlink):
    mock_is_symlink.return_value = True

    update_symlinks()

    mock_is_symlink.assert_called_once()
    mock_unlink.assert_called_once()
    mock_symlink.assert_called_once_with(f"v{SCHEMA_VERSIONS.current_version}", mock.ANY)


@mock.patch("os.symlink")
@mock.patch("shutil.rmtree")
@mock.patch("pathlib.Path.is_symlink")
@mock.patch("pathlib.Path.exists")
def test_update_symlinks_existing_directory(mock_exists, mock_is_symlink, mock_rmtree, mock_symlink):
    mock_is_symlink.return_value = False
    mock_exists.return_value = True

    update_symlinks()

    mock_is_symlink.assert_called_once()
    mock_exists.assert_called_once()
    mock_rmtree.assert_called_once()
    mock_symlink.assert_called_once_with(f"v{SCHEMA_VERSIONS.current_version}", mock.ANY)


@mock.patch("os.symlink")
@mock.patch("pathlib.Path.is_symlink")
@mock.patch("pathlib.Path.exists")
def test_update_symlinks_no_existing_path(mock_exists, mock_is_symlink, mock_symlink):
    mock_is_symlink.return_value = False
    mock_exists.return_value = False

    update_symlinks()

    mock_is_symlink.assert_called_once()
    mock_exists.assert_called_once()
    mock_symlink.assert_called_once_with(f"v{SCHEMA_VERSIONS.current_version}", mock.ANY)


@mock.patch("builtins.print")
@mock.patch("os.symlink")
@mock.patch("pathlib.Path.is_symlink")
@mock.patch("pathlib.Path.exists")
def test_update_symlinks_output_message(mock_exists, mock_is_symlink, mock_symlink, mock_print):
    mock_is_symlink.return_value = False
    mock_exists.return_value = False

    update_symlinks()

    mock_print.assert_called_with(f"Updated 'latest' symlink to point to 'v{SCHEMA_VERSIONS.current_version}'")


@mock.patch("os.symlink")
@mock.patch("pathlib.Path.is_symlink")
@mock.patch("pathlib.Path.exists")
def test_update_symlinks_permission_error(mock_exists, mock_is_symlink, mock_symlink):
    mock_is_symlink.return_value = False
    mock_exists.return_value = False
    mock_symlink.side_effect = OSError("Permission denied")

    with pytest.raises(OSError) as excinfo:
        update_symlinks()

    assert "Permission denied" in str(excinfo.value)
