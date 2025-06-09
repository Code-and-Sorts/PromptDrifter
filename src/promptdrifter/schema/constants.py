"""Constants for schema versioning."""

from typing import List

from pydantic import BaseModel, Field, computed_field


class SchemaVersions(BaseModel):
    """Schema version configuration."""

    current_version: str = Field("0.1", description="Current schema version")
    supported_versions: List[str] = Field(["0.1"], description="List of all supported versions, in chronological order")

    @computed_field
    def latest_version(self) -> str:
        """Latest supported version (always the last in the list)."""
        return self.supported_versions[-1]


SCHEMA_VERSIONS = SchemaVersions()
