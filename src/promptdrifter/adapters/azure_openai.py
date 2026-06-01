import os
from typing import Any, Dict, Optional

import httpx
from pydantic import Field, model_validator
from rich.console import Console

from ..config.adapter_settings import (
    API_KEY_ENV_VAR_AZURE_OPENAI,
    API_VERSION_ENV_VAR_AZURE_OPENAI,
    AZURE_OPENAI_API_BASE_URL,
    DEFAULT_AZURE_OPENAI_API_VERSION,
    DEFAULT_AZURE_OPENAI_MODEL,
    ENDPOINT_ENV_VAR_AZURE_OPENAI,
)
from ..http_client_manager import get_shared_client
from .base import Adapter, BaseAdapterConfig
from .models.azure_openai_models import AzureOpenAIHeaders, StandardResponse

console = Console()


class AzureOpenAIAdapterConfig(BaseAdapterConfig):
    base_url: str = AZURE_OPENAI_API_BASE_URL
    default_model: str = DEFAULT_AZURE_OPENAI_MODEL
    api_version: str = DEFAULT_AZURE_OPENAI_API_VERSION
    api_key: Optional[str] = Field(default=None, validate_default=True)
    max_tokens: Optional[int] = Field(default=2048, validate_default=True)
    temperature: Optional[float] = None
    system_prompt: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def load_from_env(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if not values.get("api_key"):
            api_key_from_env = os.getenv(API_KEY_ENV_VAR_AZURE_OPENAI)
            if api_key_from_env:
                values["api_key"] = api_key_from_env
        if not values.get("base_url"):
            endpoint_from_env = os.getenv(ENDPOINT_ENV_VAR_AZURE_OPENAI)
            if endpoint_from_env:
                values["base_url"] = endpoint_from_env
        if not values.get("api_version"):
            api_version_from_env = os.getenv(API_VERSION_ENV_VAR_AZURE_OPENAI)
            if api_version_from_env:
                values["api_version"] = api_version_from_env
        return values

    @model_validator(mode="after")
    def check_api_key_present(self) -> "AzureOpenAIAdapterConfig":
        if not self.api_key:
            raise ValueError(
                f"Azure OpenAI API key not provided. Set the {API_KEY_ENV_VAR_AZURE_OPENAI} environment variable, "
                f"or pass 'api_key' to the adapter or its config."
            )
        return self

    def get_headers(self) -> Dict[str, str]:
        return AzureOpenAIHeaders(**{"api-key": self.api_key}).model_dump(by_alias=True)

    def get_chat_completions_path(
        self,
        config_override: Optional["AzureOpenAIAdapterConfig"] = None,
    ) -> str:
        deployment = (config_override or self).default_model
        return f"/openai/deployments/{deployment}/chat/completions"

    def get_payload(
            self,
            prompt: str,
            config_override: Optional["AzureOpenAIAdapterConfig"] = None
        ) -> Dict[str, Any]:
        effective_config = config_override or self
        messages = []
        if effective_config.system_prompt:
            messages.append({"role": "system", "content": effective_config.system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {"messages": messages}
        if effective_config.temperature is not None:
            payload["temperature"] = effective_config.temperature
        if effective_config.max_tokens is not None:
            payload["max_tokens"] = effective_config.max_tokens
        return payload


class AzureOpenAIAdapter(Adapter):
    """Adapter for interacting with the Azure OpenAI Service."""

    def __init__(
        self,
        config: Optional[AzureOpenAIAdapterConfig] = None,
    ):
        self.config = config or AzureOpenAIAdapterConfig()
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
        config_override: Optional[AzureOpenAIAdapterConfig] = None,
    ) -> StandardResponse:
        """Makes a request to the Azure OpenAI Chat Completions API."""
        payload = self.config.get_payload(prompt, config_override)
        effective_config = config_override or self.config
        model_name = effective_config.default_model
        path = self.config.get_chat_completions_path(config_override)
        params = {"api-version": effective_config.api_version}
        response = StandardResponse(model_name=model_name)
        try:
            client = await self._get_client()
            api_response = await client.post(
                path, json=payload, params=params, timeout=60.0
            )
            api_response.raise_for_status()
            raw_response_content = api_response.json()
            response.raw_response = raw_response_content
            if (
                raw_response_content.get("choices")
                and isinstance(raw_response_content["choices"], list)
                and len(raw_response_content["choices"]) > 0
            ):
                first_choice = raw_response_content["choices"][0]
                if (
                    isinstance(first_choice, dict)
                    and first_choice.get("message")
                    and isinstance(first_choice["message"], dict)
                ):
                    response.text_response = first_choice["message"].get("content")
                response.finish_reason = first_choice.get("finish_reason")
            response.usage = raw_response_content.get("usage")
        except httpx.HTTPStatusError as e:
            error_content = e.response.text
            response.error = f"API Error (HTTP {e.response.status_code}): {error_content}"
            response.raw_response = {"error_detail": error_content}
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
