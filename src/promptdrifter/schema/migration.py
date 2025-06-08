import copy
from typing import Any, Dict, Optional

from promptdrifter.schema.constants import SCHEMA_VERSIONS
from promptdrifter.schema.validator import validate_config


def get_migration_function(from_version: str, to_version: str):
    if from_version == to_version:
        return None

    migration_map = {
    }

    migration_key = f"{from_version}-{to_version}"
    if migration_key in migration_map:
        return migration_map[migration_key]

    raise ValueError(
        f"No migration path available from version {from_version} to {to_version}. "
        f"Supported migrations: {', '.join(migration_map.keys())}"
    )


def migrate_config(
    config_data: Dict[str, Any], target_version: Optional[str] = None
) -> Dict[str, Any]:
    source_version = config_data.get("version")
    if not source_version:
        raise ValueError("No version specified in config data")

    if target_version is None:
        target_version = SCHEMA_VERSIONS.latest_version

    if source_version == target_version:
        return config_data
    validate_config(config_data, version=source_version)

    migrate_func = get_migration_function(source_version, target_version)
    if migrate_func is None:
        if source_version != target_version:
            new_data = copy.deepcopy(config_data)
            new_data["version"] = target_version
            return new_data
        return config_data

    new_data = migrate_func(config_data)
    validate_config(new_data, version=target_version)

    return new_data
