import json
import os
from typing import Any, Dict, Optional

import httpx
from pydantic import Field, model_validator
from rich.console import Console

from ..config.adapter_settings import (
    API_KEY_ENV_VAR_CLAUDE,
    CLAUDE_API_BASE_URL,
    CLAUDE_API_VERSION,
    DEFAULT_CLAUDE_MODEL,
)
from ..http_client_manager import get_shared_client
from .base import Adapter, BaseAdapterConfig
from .models import (
    ClaudeErrorResponse,
    ClaudeHeaders,
    ClaudeMessage,
    ClaudePayload,
    ClaudeRawResponse,
    ClaudeResponse,
)

console = Console()


class ClaudeAdapterConfig(BaseAdapterConfig):
    """Configuration for Claude API adapter."""
    base_url: str = CLAUDE_API_BASE_URL
    default_model: str = DEFAULT_CLAUDE_MODEL
    api_version: str = CLAUDE_API_VERSION
    api_key: Optional[str] = Field(default=None, validate_default=True)
    max_tokens: Optional[int] = Field(default=1024, validate_default=True)
    temperature: Optional[float] = Field(default=1.0, validate_default=True)
    system_prompt: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def load_api_key(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values.get("api_key"):
            return values

        api_key_from_env = os.getenv(API_KEY_ENV_VAR_CLAUDE)
        if api_key_from_env:
            values["api_key"] = api_key_from_env
        return values

    @model_validator(mode="after")
    def check_api_key_present(self) -> "ClaudeAdapterConfig":
        if not self.api_key:
            raise ValueError(
                f"Claude API key not provided. Set the {API_KEY_ENV_VAR_CLAUDE} environment variable, "
                f"or pass 'api_key' to the adapter or its config."
            )
        return self

    def get_headers(self) -> Dict[str, str]:
        headers = ClaudeHeaders(
            x_api_key=self.api_key,
            anthropic_version=self.api_version
        )
        return headers.model_dump()

    def get_payload(
            self,
            prompt: str,
            config_override: Optional["ClaudeAdapterConfig"] = None
        ) -> Dict[str, Any]:
        selected_model = config_override.default_model if config_override else self.default_model
        selected_max_tokens = config_override.max_tokens if config_override else self.max_tokens
        selected_temperature = config_override.temperature if config_override else self.temperature
        selected_system = config_override.system_prompt if config_override else None

        payload = ClaudePayload(
            model=selected_model,
            max_tokens=selected_max_tokens,
            messages=[ClaudeMessage(content=prompt)],
            temperature=selected_temperature,
            system=selected_system
        )
        return payload.model_dump(exclude_none=True)


class ClaudeAdapter(Adapter):
    """Adapter for interacting with Anthropic's Claude API."""

    def __init__(
        self,
        config: Optional[ClaudeAdapterConfig] = None,
    ):
        self.config = config or ClaudeAdapterConfig()
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
        config_override: Optional[ClaudeAdapterConfig] = None,
    ) -> ClaudeResponse:
        """
        Execute the prompt against the specified Claude model.
        See: https://docs.anthropic.com/claude/reference/messages_post
        """
        selected_model = config_override.default_model if config_override else self.config.default_model
        payload = self.config.get_payload(prompt, config_override)
        endpoint = "/messages"

        response = ClaudeResponse(model_name=selected_model)

        try:
            client = await self._get_client()
            http_response = await client.post(
                endpoint,
                json=payload,
                timeout=60.0
            )
            http_response.raise_for_status()

            raw_response_content = http_response.json()
            raw_response = ClaudeRawResponse.model_validate(raw_response_content)
            response = raw_response.to_standard_response(selected_model)

        except httpx.HTTPStatusError as e:
            error_message = self._extract_error_message(e.response)
            response.error = f"API Error (HTTP {e.response.status_code}): {error_message}"
            try:
                response.raw_response = e.response.json()
            except json.JSONDecodeError:
                response.raw_response = {"error_detail": e.response.text}
            response.text_response = None
            response.finish_reason = "error"

        except httpx.RequestError as e:
            response.error = f"HTTP Client Error: {type(e).__name__} - {e}"
            response.raw_response = {"error_detail": str(e)}
            response.text_response = None
            response.finish_reason = "error"

        except Exception as e:
            console.print_exception()
            response.error = f"An unexpected error occurred: {e}"
            response.finish_reason = "error"
            response.raw_response = {"error": str(e)}

        return response

    async def close(self):
        """Close method - HTTP connections managed by shared client manager."""
        self._client = None

    def _extract_error_message(self, response) -> str:
        try:
            raw_response_content = response.json()
            error_response = ClaudeErrorResponse.model_validate(raw_response_content)
            return error_response.get_error_message()
        except Exception:
            return response.text
