from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QwenHeaders(BaseModel):
    authorization: str = Field(alias="Authorization")
    content_type: str = Field(alias="Content-Type", default="application/json")


class QwenMessage(BaseModel):
    role: str
    content: str


class QwenPayload(BaseModel):
    model: str
    messages: List[QwenMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class QwenChoice(BaseModel):
    message: QwenMessage
    finish_reason: Optional[str] = None


class QwenError(BaseModel):
    message: str
    type: Optional[str] = None
    code: Optional[str] = None


class QwenResponse(BaseModel):
    choices: List[QwenChoice]
    usage: Optional[Dict[str, Any]] = None
    error: Optional[QwenError] = None
    code: Optional[str] = None
    message: Optional[str] = None


class QwenStandardResponse(BaseModel):
    text_response: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None
    model_name: str
    finish_reason: Optional[str] = None
    error: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
