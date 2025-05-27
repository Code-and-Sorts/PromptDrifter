from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ClaudeResponse(BaseModel):
    text_response: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None
    model_name: str
    finish_reason: Optional[str] = None
    error: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None


class ClaudeMessage(BaseModel):
    role: str = Field(default="user")
    content: str


class ClaudePayload(BaseModel):
    model: str
    max_tokens: int
    messages: List[ClaudeMessage]
    temperature: Optional[float] = None
    system: Optional[str] = None


class ClaudeHeaders(BaseModel):
    anthropic_version: str = Field(alias="anthropic-version", default="2023-06-01")
    x_api_key: str = Field(alias="x-api-key")
    content_type: str = Field(alias="Content-Type", default="application/json")

    model_config = ConfigDict(populate_by_name=True)

    def model_dump(self, **kwargs) -> Dict[str, str]:
        return super().model_dump(by_alias=True, **kwargs)


class ClaudeContentBlock(BaseModel):
    type: str
    text: Optional[str] = None


class ClaudeErrorResponse(BaseModel):
    error: Optional[Dict[str, Any]] = None
    detail: Optional[str] = None

    def get_error_message(self) -> str:
        if self.error and isinstance(self.error, dict):
            return self.error.get("message", str(self.error))
        if self.detail:
            return str(self.detail)
        return str(self.model_dump())


class ClaudeRawResponse(BaseModel):
    id: Optional[str] = None
    type: Optional[str] = None
    role: Optional[str] = None
    content: List[ClaudeContentBlock]
    model: Optional[str] = None
    stop_reason: Optional[str] = Field(alias="stop_reason")
    stop_sequence: Optional[str] = Field(alias="stop_sequence", default=None)
    usage: Optional[Dict[str, Any]] = None

    def to_standard_response(self, model_name: str) -> ClaudeResponse:
        text_blocks = [block for block in self.content if block.type == "text" and block.text]
        return ClaudeResponse(
            model_name=model_name,
            text_response=text_blocks[0].text if text_blocks else None,
            raw_response=self.model_dump(exclude_none=True),
            finish_reason=self.stop_reason or "unknown",
            usage=self.usage
        )
