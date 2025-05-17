from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, model_validator

class AdapterConfig(BaseModel):
    adapter_type: Literal["openai", "ollama", "gemini"] = Field(..., alias="type")
    model: str
    temperature: Optional[float] = Field(default=None, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    extra_params: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        pass


class TestCase(BaseModel):
    id: str = Field(..., pattern="^[a-z0-9][a-z0-9\\-]{1,48}[a-z0-9]$")
    prompt: str = Field(..., min_length=1)
    inputs: Optional[Dict[str, str]] = Field(default_factory=dict)
    expect_exact: Optional[str] = None
    expect_regex: Optional[str] = None
    expect_substring: Optional[str] = None
    expect_substring_case_insensitive: Optional[str] = None
    tags: Optional[List[str]] = Field(default_factory=list)
    adapter_configurations: List[AdapterConfig] = Field(..., alias="adapter")

    @model_validator(mode='after')
    def check_one_expectation(self) -> 'TestCase':
        expect_fields = [
            "expect_exact",
            "expect_regex",
            "expect_substring",
            "expect_substring_case_insensitive",
        ]

        provided_expects_count = 0
        if self.expect_exact is not None:
            provided_expects_count += 1
        if self.expect_regex is not None:
            provided_expects_count += 1
        if self.expect_substring is not None:
            provided_expects_count += 1
        if self.expect_substring_case_insensitive is not None:
            provided_expects_count += 1

        if provided_expects_count > 1:
            raise ValueError(f"Only one of {', '.join(expect_fields)} can be provided.")
        return self

class PromptDrifterConfig(BaseModel):
    version: Literal["0.1"]
    tests: List[TestCase] = Field(..., alias="adapters")

    class Config:
        allow_population_by_field_name = True

# Example of how to load (parser would be elsewhere): # TODO: Add parser
# import yaml
# from pydantic import ValidationError
#
# def load_config(config_path: str) -> Optional[PromptDrifterConfig]:
#     try:
#         with open(config_path, 'r') as f:
#             data = yaml.safe_load(f)
#         return PromptDrifterConfig(**data)
#     except FileNotFoundError:
#         print(f"Error: Config file not found at {config_path}")
#         return None
#     except ValidationError as e:
#         print(f"Error validating config file {config_path}:\n{e}")
#         return None
#     except Exception as e:
#         print(f"An unexpected error occurred while loading {config_path}: {e}")
#         return None
#
# if __name__ == '__main__':
#     # Create a dummy promptdrifter.yaml for testing
#     dummy_yaml_content = """
# tests:
#   - id: sample-test-1
#     prompt: "What is the capital of France?"
#     expected: "Paris"
#     adapters:
#       - adapter: gemini
#         model: "gemini-pro"
#       - adapter: openai
#         model: "gpt-3.5-turbo"
#         params:
#           temperature: 0.7
#   - id: sample-test-2
#     prompt: "Translate to Spanish: Hello"
#     context:
#       text_to_translate: "Hello"
#     adapters:
#       - adapter: ollama
#         model: "llama2"
# """
#     with open("promptdrifter.yaml", "w") as f:
#         f.write(dummy_yaml_content)
#
#     config = load_config("promptdrifter.yaml")
#     if config:
#         for test_case in config.tests:
#             print(f"Test ID: {test_case.id}")
#             print(f"  Prompt: {test_case.prompt}")
#             if test_case.context:
#                 print(f"  Context: {test_case.context}")
#             for adapter_conf in test_case.adapters:
#                 print(f"    Adapter: {adapter_conf.adapter}, Model: {adapter_conf.model}")
#                 if adapter_conf.params:
#                     print(f"      Params: {adapter_conf.params}") 