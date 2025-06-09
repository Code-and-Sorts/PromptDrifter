from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class GeminiHeaders(BaseModel):
    content_type: str = Field(alias="Content-Type", default="application/json")

    model_config = ConfigDict(populate_by_name=True)

    def model_dump(self, **kwargs) -> Dict[str, str]:
        return super().model_dump(by_alias=True, **kwargs)

class GeminiPart(BaseModel):
    text: str

class GeminiContent(BaseModel):
    parts: List[GeminiPart]

class GeminiGenerationConfig(BaseModel):
    temperature: Optional[float] = None
    maxOutputTokens: Optional[int] = None

class GeminiPayload(BaseModel):
    contents: List[GeminiContent]
    generationConfig: Optional[GeminiGenerationConfig] = None

class GeminiPromptFeedback(BaseModel):
    promptTokenCount: Optional[int] = None
    blockReason: Optional[str] = None
    blockReasonMessage: Optional[str] = None

class GeminiCandidate(BaseModel):
    content: GeminiContent
    finishReason: Optional[str] = None
    tokenCount: Optional[int] = None

class GeminiResponse(BaseModel):
    candidates: List[GeminiCandidate]
    promptFeedback: Optional[GeminiPromptFeedback] = None

class StandardResponse(BaseModel):
    text_response: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None
    model_name: str
    finish_reason: Optional[str] = None
    error: Optional[str] = None
    usage: Optional[Dict[str, Optional[int]]] = None

class GeminiErrorResponse(BaseModel):
    error: Optional[Dict[str, Any]] = None
    message: Optional[str] = None

    def get_error_message(self) -> str:
        if self.error and isinstance(self.error, dict):
            return self.error.get("message", str(self.error))
        return self.message or "Unknown error"

class GeminiRawResponse(BaseModel):
    candidates: Optional[List[GeminiCandidate]] = None
    prompt_feedback: Optional[Dict[str, Any]] = Field(alias="promptFeedback", default=None)
    usage_metadata: Optional[Dict[str, Any]] = Field(alias="usageMetadata", default=None)

    model_config = ConfigDict(populate_by_name=True)

    def to_standard_response(self, model_name: str) -> GeminiResponse:
        finish_reason = None
        error = None

        if self.candidates and len(self.candidates) > 0:
            candidate = self.candidates[0]
            if candidate.content and candidate.content.parts:
                pass
            finish_reason = candidate.finishReason
        elif self.prompt_feedback:
            finish_reason = self.prompt_feedback.get("blockReason")
            block_reason_message = self.prompt_feedback.get("blockReasonMessage")
            error = f"Prompt blocked: {finish_reason}. {block_reason_message if block_reason_message else ''}"

        return GeminiResponse(
            candidates=self.candidates,
            promptFeedback=self.prompt_feedback,
            model_name=model_name,
            finish_reason=finish_reason,
            error=error,
            usage=self.usage_metadata
        )
