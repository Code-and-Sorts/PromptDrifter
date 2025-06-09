from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class OpenAIHeaders(BaseModel):
    authorization: str = Field(alias="Authorization")
    content_type: str = Field(alias="Content-Type", default="application/json")


class OpenAIMessage(BaseModel):
    role: str
    content: str


class OpenAIPayload(BaseModel):
    model: str
    messages: List[OpenAIMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


class OpenAIChoice(BaseModel):
    message: Dict[str, Any]
    finish_reason: Optional[str] = None


class OpenAIResponse(BaseModel):
    choices: List[OpenAIChoice]
    usage: Optional[Dict[str, Any]] = None


class StandardResponse(BaseModel):
    text_response: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None
    model_name: str
    finish_reason: Optional[str] = None
    error: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
