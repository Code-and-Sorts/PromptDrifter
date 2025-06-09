from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class MistralHeaders(BaseModel):
    authorization: str = Field(alias="Authorization")
    content_type: str = Field(alias="Content-Type", default="application/json")
    model_config = ConfigDict(populate_by_name=True)
    def model_dump(self, *args, **kwargs):
        return super().model_dump(*args, by_alias=True, **kwargs)

class MistralMessage(BaseModel):
    role: str
    content: str

class MistralPayload(BaseModel):
    model: str
    messages: List[MistralMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None

class MistralErrorResponse(BaseModel):
    error: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    detail: Optional[Any] = None
    def get_error_message(self) -> str:
        if self.error and isinstance(self.error, dict):
            return self.error.get("message", str(self.error))
        if self.message:
            return self.message
        if self.detail:
            return str(self.detail)
        return "Unknown error"

class MistralRawResponse(BaseModel):
    choices: Optional[List[Dict[str, Any]]] = None
    usage: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    def to_standard_response(self, model_name: str) -> "MistralResponse":
        text_response = None
        finish_reason = None
        if self.choices and len(self.choices) > 0:
            first_choice = self.choices[0]
            finish_reason = first_choice.get("finish_reason")
            message = first_choice.get("message", {})
            if message and isinstance(message, dict):
                text_response = message.get("content")
        return MistralResponse(
            text_response=text_response,
            raw_response=self.model_dump(exclude_none=True),
            model_name=model_name,
            finish_reason=finish_reason,
            error=None if text_response else "No text content found in successful response.",
            usage=self.usage,
        )

class MistralResponse(BaseModel):
    text_response: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None
    model_name: str
    finish_reason: Optional[str] = None
    error: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None

class MistralChoice(BaseModel):
    message: Dict[str, Any]
    finish_reason: Optional[str] = None

class StandardResponse(BaseModel):
    text_response: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None
    model_name: str
    finish_reason: Optional[str] = None
    error: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
