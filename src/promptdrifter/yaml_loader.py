from pathlib import Path

import yaml
from jsonschema import ValidationError as JsonSchemaValidationError
from pydantic import ValidationError as PydanticValidationError

from promptdrifter.models.config import PromptDrifterConfig
from promptdrifter.schema.validator import validate_config


class YamlFileLoader:
    def __init__(self):
        pass

    def load_and_validate_yaml(self, yaml_path: Path) -> PromptDrifterConfig:
        if not yaml_path.is_file():
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")

        try:
            with open(yaml_path, "r") as yaml_file:
                yaml_data = yaml.safe_load(yaml_file)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file {yaml_path}: {e}") from e

        if not isinstance(yaml_data, dict):
            expected_type = (
                "a dictionary"
                if yaml_data is not None
                else "data (got None from YAML)"
            )
            user_message = (
                f"[bold red]Configuration Error in '{yaml_path}':[/bold red]\n"
                f"Validation failed:\n"
                f"  - At 'root': Input should be {expected_type}, received {type(yaml_data).__name__}."
            )
            raise ValueError(user_message)

        try:
            validate_config(yaml_data)
        except JsonSchemaValidationError as e:
            user_message = (
                f"[bold red]Configuration Error in '{yaml_path}':[/bold red]\n"
                f"JSON Schema validation failed:\n"
                f"  - {e.message}"
            )
            raise ValueError(user_message) from e
        except ValueError as e:
            user_message = (
                f"[bold red]Configuration Error in '{yaml_path}':[/bold red]\n"
                f"Schema version error:\n"
                f"  - {str(e)}"
            )
            raise ValueError(user_message) from e

        try:
            config_model = PromptDrifterConfig(**yaml_data)
        except PydanticValidationError as e:
            error_details = e.errors()
            formatted_errors = []
            for error in error_details:
                loc = " -> ".join(map(str, error["loc"]))
                msg = error["msg"]
                formatted_errors.append(f"  - At '{loc}': {msg}")

            user_message = (
                f"[bold red]Configuration Error in '{yaml_path}':[/bold red]\n"
                f"Pydantic validation failed:\n" + "\n".join(formatted_errors)
            )
            raise ValueError(user_message) from e
        except Exception as e:
            raise ValueError(
                f"Unexpected error processing YAML data from '{yaml_path}' with Pydantic: {type(e).__name__} - {e}"
            ) from e

        return config_model
