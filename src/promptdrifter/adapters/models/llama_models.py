from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class LlamaHeaders(BaseModel):
    authorization: str = Field(alias="Authorization")
    content_type: str = Field(alias="Content-Type", default="application/json")
    model_config = ConfigDict(populate_by_name=True)

    def model_dump(self, **kwargs) -> Dict[str, str]:
        return super().model_dump(by_alias=True, **kwargs)

class LlamaMessage(BaseModel):
    role: str = Field(default="user")
    content: str

class LlamaPayload(BaseModel):
    model: str
    max_tokens: int
    messages: List[LlamaMessage]
    temperature: Optional[float] = None
    system: Optional[str] = None

class LlamaResponse(BaseModel):
    text_response: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None
    model_name: str
    finish_reason: Optional[str] = None
    error: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None

class LlamaErrorResponse(BaseModel):
    error: Optional[Dict[str, Any]] = None
    detail: Optional[str] = None

    def get_error_message(self) -> str:
        if self.error and isinstance(self.error, dict):
            return self.error.get("message", str(self.error))
        if self.detail:
            return str(self.detail)
        return str(self.model_dump())

class LlamaRawResponse(BaseModel):
    id: Optional[str] = None
    object: Optional[str] = None
    created: Optional[int] = None
    model: Optional[str] = None
    choices: Optional[List[Dict[str, Any]]] = None
    usage: Optional[Dict[str, Any]] = None
    stop_reason: Optional[str] = Field(default=None)
    stop_sequence: Optional[str] = Field(default=None)
    model_config = ConfigDict(populate_by_name=True)

    def to_standard_response(self, model_name: str) -> LlamaResponse:
        text_response = None
        finish_reason = None
        if self.choices and len(self.choices) > 0:
            choice = self.choices[0]
            message = choice.get("message", {})
            if message and isinstance(message, dict):
                text_response = message.get("content")
            finish_reason = choice.get("finish_reason")
        return LlamaResponse(
            model_name=model_name,
            text_response=text_response,
            raw_response=self.model_dump(exclude_none=True),
            finish_reason=finish_reason or self.stop_reason or "unknown",
            usage=self.usage
        )
