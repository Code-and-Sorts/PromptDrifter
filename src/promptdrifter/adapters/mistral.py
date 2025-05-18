import os
from typing import Any, Dict, Optional

import httpx
from pydantic import BaseModel, Field, model_validator
from rich.console import Console

from ..config.adapter_settings import (
    API_KEY_ENV_VAR_MISTRAL,
    DEFAULT_MISTRAL_MODEL,
    MISTRAL_API_BASE_URL,
)
from .base import Adapter

console = Console()


class MistralAdapterConfig(BaseModel):
    api_key: Optional[str] = Field(default=None, validate_default=True)
    base_url: str = MISTRAL_API_BASE_URL
    default_model: str = DEFAULT_MISTRAL_MODEL

    @model_validator(mode="before")
    @classmethod
    def load_api_key(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values.get("api_key"):
            return values

        api_key_from_env = os.getenv(API_KEY_ENV_VAR_MISTRAL)
        if api_key_from_env:
            values["api_key"] = api_key_from_env
        return values

    @model_validator(mode="after")
    def check_api_key_present(self) -> "MistralAdapterConfig":
        if not self.api_key:
            raise ValueError(
                f"Mistral API key not provided. Set the {API_KEY_ENV_VAR_MISTRAL} environment variable, "
                f"or pass 'api_key' to the adapter or its config."
            )
        return self


class MistralAdapter(Adapter):
    """Adapter for interacting with Mistral AI API."""

    def __init__(
        self,
        config: Optional[MistralAdapterConfig] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
    ):
        if config:
            self.config = config
        else:
            config_data = {}
            if api_key:
                config_data["api_key"] = api_key
            if base_url:
                config_data["base_url"] = base_url
            if default_model:
                config_data["default_model"] = default_model
            self.config = MistralAdapterConfig(**config_data)

        self.headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        self.client = httpx.AsyncClient(
            base_url=self.config.base_url,
            headers=self.headers,
        )

    async def execute(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Makes a request to the Mistral Chat Completions API."""
        if not self.config.api_key:
            return {
                "error": "Mistral API key not configured.",
                "raw_response": None,
                "text_response": None,
                "model_name": model or self.config.default_model,
                "finish_reason": "error",
            }

        effective_model = model or self.config.default_model

        # Build the messages array based on the inputs
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Override messages if provided directly via kwargs
        if "messages" in kwargs:
            messages = kwargs.pop("messages")

        payload = {
            "model": effective_model,
            "messages": messages,
        }

        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        # Add any additional parameters from kwargs
        for key, value in kwargs.items():
            payload[key] = value

        try:
            response = await self.client.post(
                "/chat/completions", json=payload, timeout=60.0
            )
            response.raise_for_status()
            response_data = response.json()

            text_response = None
            finish_reason = None
            usage = None

            # Extract response info
            try:
                choices = response_data.get("choices", [])
                if choices and len(choices) > 0:
                    first_choice = choices[0]
                    finish_reason = first_choice.get("finish_reason")

                    message = first_choice.get("message", {})
                    if message and isinstance(message, dict):
                        text_response = message.get("content")
                    else:
                        return {
                            "error": "No text content found in successful response.",
                            "raw_response": response_data,
                            "text_response": None,
                            "model_name": effective_model,
                            "finish_reason": finish_reason,
                            "usage": response_data.get("usage"),
                        }
                else:
                    return {
                        "error": "Unexpected response structure for 200 OK.",
                        "raw_response": response_data,
                        "text_response": None,
                        "model_name": effective_model,
                    }

                usage = response_data.get("usage")
            except Exception as e:
                console.print_exception()
                return {
                    "error": f"Error parsing successful response: {e}",
                    "raw_response": response_data,
                    "text_response": None,
                    "model_name": effective_model,
                }

            return {
                "raw_response": response_data,
                "text_response": text_response,
                "model_name": effective_model,
                "finish_reason": finish_reason,
                "usage": usage,
                "error": None,
            }

        except httpx.HTTPStatusError as e:
            error_detail = "Unknown error"
            try:
                error_json = e.response.json()
                if "error" in error_json:
                    error_msg = error_json["error"].get("message", str(error_json["error"]))
                    error_detail = error_msg
                else:
                    error_detail = str(error_json)
            except Exception:
                error_detail = e.response.text

            return {
                "error": f"API Error (HTTP {e.response.status_code}): {error_detail}",
                "raw_response": error_detail,
                "text_response": None,
                "model_name": effective_model,
                "finish_reason": "error",
            }
        except httpx.RequestError as e:
            return {
                "error": f"HTTP Client Error: {e}",
                "raw_response": None,
                "text_response": None,
                "model_name": effective_model,
                "finish_reason": "error",
            }
        except Exception as e:
            console.print_exception()
            return {
                "error": f"An unexpected error occurred: {e}",
                "raw_response": None,
                "text_response": None,
                "model_name": effective_model,
                "finish_reason": "error",
            }

    async def close(self):
        """Close the underlying HTTPX client."""
        await self.client.aclose()
