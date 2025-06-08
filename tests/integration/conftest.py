import shutil
import tempfile
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def temp_config_dir(temp_dir: Path) -> Path:
    config_dir = temp_dir / "config"
    config_dir.mkdir()
    return config_dir


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def valid_yaml_file(temp_dir: Path, fixtures_dir: Path) -> Path:
    source_file = fixtures_dir / "valid_test_config.yaml"
    target_file = temp_dir / "valid_test.yaml"
    shutil.copy2(source_file, target_file)
    return target_file


@pytest.fixture
def invalid_yaml_file(temp_dir: Path, fixtures_dir: Path) -> Path:
    source_file = fixtures_dir / "invalid_missing_fields.yaml"
    target_file = temp_dir / "invalid_test.yaml"
    shutil.copy2(source_file, target_file)
    return target_file


@pytest.fixture
def malformed_yaml_file(temp_dir: Path, fixtures_dir: Path) -> Path:
    source_file = fixtures_dir / "malformed_syntax.yaml"
    target_file = temp_dir / "malformed_test.yaml"
    shutil.copy2(source_file, target_file)
    return target_file


@pytest.fixture
def multiple_adapters_yaml_file(temp_dir: Path, fixtures_dir: Path) -> Path:
    source_file = fixtures_dir / "multiple_adapters.yaml"
    target_file = temp_dir / "multiple_adapters_test.yaml"
    shutil.copy2(source_file, target_file)
    return target_file


@pytest.fixture
def empty_yaml_file(temp_dir: Path, fixtures_dir: Path) -> Path:
    source_file = fixtures_dir / "empty_file.yaml"
    target_file = temp_dir / "empty_test.yaml"
    shutil.copy2(source_file, target_file)
    return target_file


@pytest.fixture
def nonexistent_yaml_file(temp_dir: Path) -> Path:
    return temp_dir / "nonexistent.yaml"


@pytest.fixture
def non_yaml_file(temp_dir: Path) -> Path:
    txt_file = temp_dir / "not_yaml.txt"
    txt_file.write_text("This is not a YAML file")
    return txt_file


@pytest.fixture
def invalid_version_yaml_file(temp_dir: Path, fixtures_dir: Path) -> Path:
    source_file = fixtures_dir / "invalid_version.yaml"
    target_file = temp_dir / "invalid_version_test.yaml"
    shutil.copy2(source_file, target_file)
    return target_file
