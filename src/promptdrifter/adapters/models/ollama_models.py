from typing import Any, Dict, Optional

from pydantic import BaseModel


class OllamaHeaders(BaseModel):
    """Headers for Ollama API requests."""
    pass

class OllamaMessage(BaseModel):
    """Message for Ollama API requests."""
    role: str = "user"
    content: str

class OllamaPayload(BaseModel):
    """Payload for Ollama API requests."""
    model: str
    prompt: str
    stream: bool = False
    options: Optional[Dict[str, Any]] = None

class OllamaErrorResponse(BaseModel):
    """Error response from Ollama API."""
    error: Optional[str] = None

    def get_error_message(self) -> str:
        return self.error or "Unknown error"

class OllamaRawResponse(BaseModel):
    """Raw response from Ollama API."""
    response: Optional[str] = None
    done: Optional[bool] = None
    prompt_eval_count: Optional[int] = None
    eval_count: Optional[int] = None

    def to_standard_response(self, model_name: str) -> "OllamaResponse":
        return OllamaResponse(
            text_response=self.response,
            raw_response=self.model_dump(),
            model_name=model_name,
            finish_reason="completed" if self.done else "unknown",
            usage={
                "prompt_tokens": self.prompt_eval_count or 0,
                "completion_tokens": self.eval_count or 0,
                "total_tokens": (self.prompt_eval_count or 0) + (self.eval_count or 0)
            },
            error=None
        )

class OllamaResponse(BaseModel):
    """Standardized response from Ollama API."""
    text_response: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None
    model_name: str
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    error: Optional[str] = None
