from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class DeepSeekHeaders(BaseModel):
    authorization: str = Field(alias="Authorization")
    content_type: str = Field(alias="Content-Type", default="application/json")

    model_config = ConfigDict(populate_by_name=True)

    def model_dump(self, **kwargs) -> Dict[str, str]:
        return super().model_dump(by_alias=True, **kwargs)

class DeepSeekResponse(BaseModel):
    text_response: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None
    model_name: str
    finish_reason: Optional[str] = None
    error: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None

class DeepSeekMessage(BaseModel):
    role: str
    content: str

class DeepSeekPayload(BaseModel):
    model: str
    messages: List[DeepSeekMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None

class DeepSeekErrorResponse(BaseModel):
    error: Optional[Dict[str, Any]] = None
    message: Optional[str] = None

    def get_error_message(self) -> str:
        if self.error and isinstance(self.error, dict):
            return self.error.get("message", str(self.error))
        return self.message or "Unknown error"

class DeepSeekRawResponse(BaseModel):
    id: Optional[str] = None
    object: Optional[str] = None
    created: Optional[int] = None
    model: Optional[str] = None
    choices: List[Dict[str, Any]]
    usage: Optional[Dict[str, Any]] = None

    def to_standard_response(self, model_name: str) -> DeepSeekResponse:
        text_response = None
        finish_reason = None
        if self.choices and len(self.choices) > 0:
            choice = self.choices[0]
            message = choice.get("message", {})
            if message and isinstance(message, dict):
                text_response = message.get("content")
            finish_reason = choice.get("finish_reason")

        return DeepSeekResponse(
            text_response=text_response,
            raw_response=self.model_dump(exclude_none=True),
            model_name=model_name,
            finish_reason=finish_reason,
            usage=self.usage
        )

class DeepSeekChoice(BaseModel):
    message: Dict[str, Any]
    finish_reason: Optional[str] = None

class StandardResponse(BaseModel):
    text_response: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None
    model_name: str
    finish_reason: Optional[str] = None
    error: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
