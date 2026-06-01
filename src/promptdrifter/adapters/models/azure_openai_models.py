from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .openai_models import StandardResponse

__all__ = [
    "AzureOpenAIHeaders",
    "AzureOpenAIMessage",
    "AzureOpenAIPayload",
    "AzureOpenAIChoice",
    "AzureOpenAIResponse",
    "StandardResponse",
]


class AzureOpenAIHeaders(BaseModel):
    api_key: str = Field(alias="api-key")
    content_type: str = Field(alias="Content-Type", default="application/json")


class AzureOpenAIMessage(BaseModel):
    role: str
    content: str


class AzureOpenAIPayload(BaseModel):
    messages: List[AzureOpenAIMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


class AzureOpenAIChoice(BaseModel):
    message: Dict[str, Any]
    finish_reason: Optional[str] = None


class AzureOpenAIResponse(BaseModel):
    choices: List[AzureOpenAIChoice]
    usage: Optional[Dict[str, Any]] = None
