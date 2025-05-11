import importlib.resources
import json
from pathlib import Path

import yaml
from jsonschema import validate
from jsonschema.exceptions import ValidationError


class YamlFileLoader:
    def __init__(self, schema_path: Path = None):
        determined_schema_path: Path

        if schema_path is None:
            try:
                # Use Path() to ensure we are working with Path objects for iteration
                schema_dir = Path(importlib.resources.files("promptdrifter") / "schema")

                if not schema_dir.is_dir():
                    raise FileNotFoundError(
                        f"Schema directory not found or is not a directory: {schema_dir}"
                    )

                available_schemas = []
                for item_path in schema_dir.iterdir():
                    if (
                        item_path.is_file()
                        and item_path.name.startswith("v")
                        and item_path.name.endswith(".json")
                    ):
                        version_str = item_path.name[
                            1:-5
                        ]  # Remove "v" prefix and ".json" suffix
                        try:
                            # Convert version parts to integers for correct comparison (e.g., 0.10 > 0.2)
                            version_parts = tuple(map(int, version_str.split(".")))
                            if not version_parts:  # Handles cases like "v.json"
                                raise ValueError(
                                    "Empty version string after stripping."
                                )
                            available_schemas.append((version_parts, item_path))
                        except ValueError:
                            # Optionally, log a warning here for files that match the pattern
                            # but have unparsable version numbers. For now, we just skip them.
                            continue
                if not available_schemas:
                    raise FileNotFoundError(
                        f"No valid schema files (e.g., vX.Y.json) found in schema directory: {schema_dir}"
                    )

                # Sort by version tuple (e.g., (0, 10) > (0, 2)) in descending order
                available_schemas.sort(key=lambda x: x[0], reverse=True)

                # The path to the latest schema file
                determined_schema_path = available_schemas[0][1]

            except Exception as e:
                # Wrap any exception during auto-detection in a RuntimeError
                raise RuntimeError(
                    f"Failed to automatically determine the latest schema: {e}"
                ) from e
        else:
            # If schema_path is provided, use it directly
            determined_schema_path = schema_path

        self.schema_path = determined_schema_path
        self._schema = self._load_schema()

    def _load_schema(self) -> dict:
        try:
            with open(self.schema_path, "r") as schema_file:
                return json.load(schema_file)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Schema file not found at {self.schema_path}. "
                "Please ensure the schema file exists in the correct location."
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding schema JSON from {self.schema_path}: {e}")

    def load_and_validate_yaml(self, yaml_path: Path) -> dict:
        if not yaml_path.is_file():
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")

        try:
            with open(yaml_path, "r") as yaml_file:
                yaml_data = yaml.safe_load(yaml_file)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file {yaml_path}: {e}")

        try:
            validate(instance=yaml_data, schema=self._schema)
        except ValidationError as e:
            error_path = " -> ".join(map(str, e.path))
            if error_path:
                error_location_message = f"  Location: At '{error_path}' in your YAML."
            else:
                error_location_message = "  Location: At the root of your YAML."

            user_message = (
                f"[bold red]Schema Validation Error in '{yaml_path}':[/bold red]\\n"
                f"  [red]{e.message}[/red]\\n"
                f"{error_location_message}"
            )
            raise ValueError(user_message) from e

        return yaml_data
