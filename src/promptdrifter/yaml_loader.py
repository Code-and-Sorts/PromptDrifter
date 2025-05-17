from pathlib import Path

import yaml
from pydantic import ValidationError as PydanticValidationError

from promptdrifter.models.config import PromptDrifterConfig


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
            raise ValueError(f"Error parsing YAML file {yaml_path}: {e}")

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
                f"Pydantic validation failed:\n"
                + "\n".join(formatted_errors)
            )
            raise ValueError(user_message) from e
        except Exception as e:
            raise ValueError(f"Unexpected error processing YAML data from '{yaml_path}' with Pydantic: {e}") from e

        return config_model
