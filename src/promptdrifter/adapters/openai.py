import os
from typing import Any, Dict, Optional

import httpx

from ..config.adapter_settings import (
    API_KEY_ENV_VAR_OPENAI,
    DEFAULT_OPENAI_MODEL,
    OPENAI_API_BASE_URL,
)
from .base import Adapter


class OpenAIAdapter(Adapter):
    """Adapter for interacting with OpenAI API."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or os.getenv(API_KEY_ENV_VAR_OPENAI)
        if not self.api_key:
            raise ValueError(
                f"OpenAI API key not provided. Set the {API_KEY_ENV_VAR_OPENAI} environment variable "
                f"or pass it to the adapter constructor."
            )
        self.base_url = base_url or OPENAI_API_BASE_URL
        self.client = httpx.AsyncClient(
            base_url=self.base_url, headers={"Authorization": f"Bearer {self.api_key}"}
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
        effective_model = model or DEFAULT_OPENAI_MODEL

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
