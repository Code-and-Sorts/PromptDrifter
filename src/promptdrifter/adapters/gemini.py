import os
from typing import Any, Dict, Optional

import httpx
from pydantic import Field, model_validator
from rich.console import Console

from ..config.adapter_settings import (
    API_KEY_ENV_VAR_GEMINI,
    DEFAULT_GEMINI_MODEL,
    GEMINI_API_BASE_URL,
)
from ..http_client_manager import get_shared_client
from .base import Adapter, BaseAdapterConfig
from .models.gemini_models import (
    GeminiContent,
    GeminiGenerationConfig,
    GeminiHeaders,
    GeminiPart,
    GeminiPayload,
    GeminiResponse,
    StandardResponse,
)

console = Console()


class GeminiAdapterConfig(BaseAdapterConfig):
    base_url: str = GEMINI_API_BASE_URL
    default_model: str = DEFAULT_GEMINI_MODEL
    api_key: Optional[str] = Field(default=None, validate_default=True)
    max_tokens: Optional[int] = Field(default=2048, validate_default=True)
    temperature: Optional[float] = None
    system_prompt: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def load_api_key(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values.get("api_key"):
            return values
        api_key_from_env = os.getenv(API_KEY_ENV_VAR_GEMINI)
        if api_key_from_env:
            values["api_key"] = api_key_from_env
        return values

    @model_validator(mode="after")
    def check_api_key_present(self) -> "GeminiAdapterConfig":
        if not self.api_key:
            raise ValueError(
                f"Gemini API key not provided. Set the {API_KEY_ENV_VAR_GEMINI} environment variable, "
                f"or pass 'api_key' to the adapter or its config."
            )
        return self

    def get_headers(self) -> Dict[str, str]:
        return GeminiHeaders().model_dump()

    def get_payload(
            self,
            prompt: str,
            config_override: Optional["GeminiAdapterConfig"] = None
        ) -> Dict[str, Any]:
        effective_config = config_override or self
        generation_config = None
        if effective_config.temperature is not None or effective_config.max_tokens is not None:
            generation_config = GeminiGenerationConfig(
                temperature=effective_config.temperature,
                maxOutputTokens=effective_config.max_tokens
            )

        payload = GeminiPayload(
            contents=[GeminiContent(parts=[GeminiPart(text=prompt)])],
            generationConfig=generation_config
        )
        payload_dict = payload.model_dump(exclude_none=True)
        if effective_config.system_prompt:
            payload_dict["system_instruction"] = {"parts": [{"text": effective_config.system_prompt}]}
        return payload_dict


class GeminiAdapter(Adapter):
    """Adapter for interacting with Google Gemini API via REST using httpx."""

    def __init__(
        self,
        config: Optional[GeminiAdapterConfig] = None,
    ):
        self.config = config or GeminiAdapterConfig()
        self._client = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get shared HTTP client with connection pooling."""
        if self._client is None:
            self._client = await get_shared_client(
                base_url=self.config.base_url,
                headers=self.config.get_headers()
            )
        return self._client

    async def execute(
        self,
        prompt: str,
        config_override: Optional[GeminiAdapterConfig] = None,
    ) -> StandardResponse:
        """Makes a REST request to the Google Gemini API."""
        effective_model = config_override.default_model if config_override else self.config.default_model
        endpoint = f"/models/{effective_model}:generateContent"
        params = {"key": self.config.api_key}
        payload = self.config.get_payload(prompt, config_override)

        response = StandardResponse(model_name=effective_model)

        try:
            client = await self._get_client()
            api_response = await client.post(
                endpoint, params=params, json=payload, timeout=60.0
            )
            api_response.raise_for_status()
            gemini_response = GeminiResponse.model_validate(api_response.json())
            response.raw_response = gemini_response.model_dump()

            if gemini_response.candidates:
                first_candidate = gemini_response.candidates[0]
                if first_candidate.content.parts:
                    response.text_response = first_candidate.content.parts[0].text
                response.finish_reason = first_candidate.finishReason

            if gemini_response.promptFeedback:
                response.usage = {
                    "prompt_tokens": gemini_response.promptFeedback.promptTokenCount,
                    "completion_tokens": gemini_response.candidates[0].tokenCount if gemini_response.candidates else None,
                }
                # If no candidates, treat as prompt blocked
                if not gemini_response.candidates:
                    block_reason = gemini_response.promptFeedback.blockReason
                    block_message = gemini_response.promptFeedback.blockReasonMessage
                    response.error = f"Prompt blocked: {block_reason}. {block_message if block_message else ''}"
                    response.finish_reason = block_reason

        except httpx.HTTPStatusError as e:
            error_detail = "Unknown error"
            try:
                error_data = e.response.json()
                error_detail = error_data.get("error", {}).get("message", error_detail)
            except Exception:
                error_detail = e.response.text
            response.error = f"API Error (HTTP {e.response.status_code}): {error_detail}"
            response.raw_response = {"error_detail": error_detail}
            response.finish_reason = "error"

        except httpx.RequestError as e:
            response.error = f"HTTP Client Error: RequestError - {str(e)}"
            response.raw_response = {"error_detail": str(e)}
            response.finish_reason = "error"

        except Exception as e:
            console.print_exception()
            response.error = f"An unexpected error occurred: {str(e)}"
            response.raw_response = {"error_detail": str(e)}
            response.finish_reason = "error"

        return response

    async def close(self):
        """Close method - HTTP connections managed by shared client manager."""
        self._client = None
