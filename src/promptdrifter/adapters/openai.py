import os
from typing import Any, Dict, Optional

import httpx
from pydantic import BaseModel, Field, model_validator

from ..config.adapter_settings import (
    API_KEY_ENV_VAR_OPENAI,
    DEFAULT_OPENAI_MODEL,
    OPENAI_API_BASE_URL,
)
from .base import Adapter


class OpenAIAdapterConfig(BaseModel):
    api_key: Optional[str] = Field(default=None, validate_default=True)
    base_url: str = OPENAI_API_BASE_URL
    default_model: str = DEFAULT_OPENAI_MODEL

    @model_validator(mode="before")
    @classmethod
    def load_api_key(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values.get("api_key"):
            return values

        api_key_from_env = os.getenv(API_KEY_ENV_VAR_OPENAI)
        if api_key_from_env:
            values["api_key"] = api_key_from_env
        return values

    @model_validator(mode="after")
    def check_api_key_present(self) -> "OpenAIAdapterConfig":
        if not self.api_key:
            raise ValueError(
                f"OpenAI API key not provided. Set the {API_KEY_ENV_VAR_OPENAI} environment variable, "
                f"or pass 'api_key' to the adapter or its config."
            )
        return self


class OpenAIAdapter(Adapter):
    """Adapter for interacting with OpenAI API."""

    def __init__(
        self,
        config: Optional[OpenAIAdapterConfig] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        if config:
            self.config = config
        else:
            config_data = {}
            if api_key:
                config_data["api_key"] = api_key
            if base_url:
                config_data["base_url"] = base_url
            self.config = OpenAIAdapterConfig(**config_data)

        self.client = httpx.AsyncClient(
            base_url=self.config.base_url,
            headers={"Authorization": f"Bearer {self.config.api_key}"},
        )

    async def execute(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Makes a request to the OpenAI Chat Completions API."""
        effective_model = model or self.config.default_model

        payload = {
            "model": effective_model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        payload.update(kwargs)

        try:
            response = await self.client.post(
                "/chat/completions", json=payload, timeout=60.0
            )
            response.raise_for_status()
            response_data = response.json()

            text_response = None
            if (
                response_data.get("choices")
                and isinstance(response_data["choices"], list)
                and len(response_data["choices"]) > 0
            ):
                first_choice = response_data["choices"][0]
                if (
                    isinstance(first_choice, dict)
                    and first_choice.get("message")
                    and isinstance(first_choice["message"], dict)
                ):
                    text_response = first_choice["message"].get("content")

            return {
                "raw_response": response_data,
                "text_response": text_response,
                "model_used": effective_model,
            }
        except httpx.HTTPStatusError as e:
            error_content = e.response.text
            return {
                "error": f"HTTP error {e.response.status_code} from OpenAI: {error_content}",
                "raw_response": None,
                "text_response": None,
                "model_used": effective_model,
            }
        except httpx.RequestError as e:
            return {
                "error": f"Request error connecting to OpenAI: {e}",
                "raw_response": None,
                "text_response": None,
                "model_used": effective_model,
            }
        except Exception as e:
            return {
                "error": f"An unexpected error occurred: {e}",
                "raw_response": None,
                "text_response": None,
                "model_used": effective_model,
            }

    async def close(self):
        """Close the underlying HTTPX client."""
        await self.client.aclose()
