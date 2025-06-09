import json
import pathlib
from typing import Any, Dict, Optional

import jsonschema
from jsonschema import ValidationError

from promptdrifter.schema.constants import SCHEMA_VERSIONS


def get_schema_path(version: str = SCHEMA_VERSIONS.current_version) -> pathlib.Path:
    if version not in SCHEMA_VERSIONS.supported_versions:
        raise ValueError(
            f"Unsupported schema version: {version}. "
            f"Supported versions: {', '.join(SCHEMA_VERSIONS.supported_versions)}"
        )

    schema_dir = pathlib.Path(__file__).parent
    version_path = schema_dir / f"v{version}" / "schema.json"

    # Fallback to 'latest' if version directory doesn't exist (for packaged installs)
    if not version_path.exists():
        latest_path = schema_dir / "latest" / "schema.json"
        if latest_path.exists():
            return latest_path

    return version_path


def load_schema(version: str = SCHEMA_VERSIONS.current_version) -> Dict[str, Any]:
    schema_path = get_schema_path(version)
    with open(schema_path, "r") as f:
        return json.load(f)


def validate_config(config_data: Dict[str, Any], version: Optional[str] = None) -> None:
    if version is None:
        version = config_data.get("version")
        if not version:
            raise ValueError("No version specified in config data")

    try:
        schema = load_schema(version)
        jsonschema.validate(instance=config_data, schema=schema)
    except FileNotFoundError:
        raise ValueError(f"Schema version {version} not found")
    except ValidationError as e:
        raise ValidationError(
            f"Configuration validation failed for version {version}: {e.message}",
            path=e.path, schema_path=e.schema_path, schema=e.schema,
            instance=e.instance, context=e.context
        )
