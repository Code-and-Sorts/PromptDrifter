from pydantic import BaseModel, Field

from .openai_models import StandardResponse

__all__ = [
    "AzureOpenAIHeaders",
    "StandardResponse",
]


class AzureOpenAIHeaders(BaseModel):
    api_key: str = Field(alias="api-key")
    content_type: str = Field(alias="Content-Type", default="application/json")
