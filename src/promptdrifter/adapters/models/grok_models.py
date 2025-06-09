from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class GrokHeaders(BaseModel):
    authorization: str = Field(alias="Authorization")
    content_type: str = Field(alias="Content-Type", default="application/json")
    model_config = ConfigDict(populate_by_name=True)

    def model_dump(self, **kwargs) -> Dict[str, str]:
        kwargs.setdefault('by_alias', True)
        return super().model_dump(**kwargs)

class GrokMessage(BaseModel):
    role: str
    content: str

class GrokPayload(BaseModel):
    model: str
    messages: List[GrokMessage]
    max_tokens: int
    temperature: Optional[float] = None
    model_config = ConfigDict(populate_by_name=True)

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        kwargs.setdefault('by_alias', True)
        return super().model_dump(**kwargs)

class GrokResponse(BaseModel):
    text_response: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None
    model_name: str
    finish_reason: Optional[str] = None
    error: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None

class GrokErrorResponse(BaseModel):
    error: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    def get_error_message(self) -> str:
        if self.error and isinstance(self.error, dict):
            return self.error.get("message", str(self.error))
        return self.message or "Unknown error"

class GrokRawResponse(BaseModel):
    choices: Optional[List[Dict[str, Any]]] = None
    usage: Optional[Dict[str, Any]] = None
    model_config = ConfigDict(populate_by_name=True)
    def to_standard_response(self, model_name: str) -> GrokResponse:
        text_response = None
        finish_reason = None
        error = None
        if self.choices and len(self.choices) > 0:
            choice = self.choices[0]
            message = choice.get("message", {})
            if message and isinstance(message, dict):
                text_response = message.get("content")
            finish_reason = choice.get("finish_reason")
            if not text_response:
                error = "No text content found in successful response."
        else:
            error = "Unexpected response structure for 200 OK."
        return GrokResponse(
            text_response=text_response,
            raw_response=self.model_dump(exclude_none=True, by_alias=True),
            model_name=model_name,
            finish_reason=finish_reason,
            error=error,
            usage=self.usage
        )

class GrokChoice(BaseModel):
    message: Dict[str, Any]
    finish_reason: Optional[str] = None

class StandardResponse(BaseModel):
    text_response: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None
    model_name: str
    finish_reason: Optional[str] = None
    error: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
