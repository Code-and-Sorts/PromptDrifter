#!/usr/bin/env python3
"""Script to update schema symlinks based on current version settings."""

import os
import pathlib
import shutil

from promptdrifter.schema.constants import SCHEMA_VERSIONS


def update_symlinks() -> None:
    """
    Update the 'latest' symlink to point to the current version directory.

    This should be run whenever the SCHEMA_VERSIONS.current_version is updated.
    """
    schema_dir = pathlib.Path(__file__).parent
    latest_link = schema_dir / "latest"
    target_dir = f"v{SCHEMA_VERSIONS.current_version}"

    # Remove existing symlink or directory if it exists
    if latest_link.is_symlink():
        latest_link.unlink()
    elif latest_link.exists():
        # If it's a directory, remove it
        shutil.rmtree(latest_link)
        print("Removed existing 'latest' directory")

    # Create new symlink
    os.symlink(target_dir, latest_link)
    print(f"Updated 'latest' symlink to point to '{target_dir}'")


if __name__ == "__main__":
    update_symlinks()
