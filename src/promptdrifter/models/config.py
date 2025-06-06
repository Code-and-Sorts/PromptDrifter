from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AdapterConfig(BaseModel):
    model_config = ConfigDict(
        extra="allow", validate_assignment=True, populate_by_name=True
    )

    adapter_type: Literal[
        "openai", "ollama", "gemini", "qwen", "claude", "grok", "deepseek", "llama", "mistral"
    ] = Field(
        ..., alias="type"
    )
    model: str
    base_url: Optional[str] = None
    temperature: Optional[float] = Field(default=None, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    skip: bool = Field(default=False, description="When true, this adapter will be skipped during test execution.")


class TestCase(BaseModel):
    __test__ = False
    model_config = ConfigDict(validate_assignment=True, populate_by_name=True)

    id: str = Field(..., pattern="^[a-z0-9][a-z0-9\\-]{1,48}[a-z0-9]$")
    prompt: str = Field(..., min_length=1)
    inputs: Optional[Dict[str, str]] = Field(default_factory=dict)
    expect_exact: Optional[str] = None
    expect_regex: Optional[str] = None
    expect_substring: Optional[str] = None
    expect_substring_case_insensitive: Optional[str] = None
    tags: Optional[List[str]] = Field(default_factory=list)
    adapter_configurations: List[AdapterConfig] = Field(..., alias="adapter")

    @model_validator(mode="after")
    def check_one_expectation(self) -> "TestCase":
        expect_fields = [
            self.expect_exact,
            self.expect_regex,
            self.expect_substring,
            self.expect_substring_case_insensitive,
        ]
        provided_expects_count = sum(1 for f in expect_fields if f is not None)

        if provided_expects_count > 1:
            field_names = [
                "expect_exact",
                "expect_regex",
                "expect_substring",
                "expect_substring_case_insensitive",
            ]
            raise ValueError(f"Only one of {', '.join(field_names)} can be provided.")
        return self


class PromptDrifterConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True, populate_by_name=True)

    version: Literal["0.1"]
    tests: List[TestCase] = Field(..., alias="adapters")
