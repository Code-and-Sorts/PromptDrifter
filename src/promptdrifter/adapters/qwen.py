import os
from typing import Any, Dict, Optional

import httpx
from pydantic import BaseModel, Field, model_validator

from ..config.adapter_settings import (
    API_KEY_ENV_VAR_QWEN,
    DEFAULT_QWEN_MODEL,
    QWEN_API_BASE_URL,
)
from .base import Adapter


class QwenAdapterConfig(BaseModel):
    api_key: Optional[str] = Field(default=None, validate_default=True)
    base_url: str = QWEN_API_BASE_URL
    default_model: str = DEFAULT_QWEN_MODEL

    @model_validator(mode='before')
    @classmethod
    def load_api_key(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values.get('api_key'):
            return values

        api_key_from_env = os.getenv(API_KEY_ENV_VAR_QWEN)
        if api_key_from_env:
            values['api_key'] = api_key_from_env
            return values

        api_key_from_env = os.getenv("DASHSCOPE_API_KEY")
        if api_key_from_env:
            values['api_key'] = api_key_from_env
            return values

        return values

    @model_validator(mode='after')
    def check_api_key_present(self) -> 'QwenAdapterConfig':
        if not self.api_key:
            raise ValueError(
                f"Qwen API key not provided. Set the {API_KEY_ENV_VAR_QWEN} or DASHSCOPE_API_KEY environment variable, "
                f"or pass 'api_key' to the adapter or its config."
            )
        return self


class QwenAdapter(Adapter):
    """Adapter for interacting with Alibaba Cloud Qwen (Tongyi Qianwen) API via DashScope."""

    def __init__(
        self,
        config: Optional[QwenAdapterConfig] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
    ):
        if config:
            self.config = config
        else:
            config_data = {}
            if api_key:
                config_data['api_key'] = api_key
            if base_url:
                config_data['base_url'] = base_url
            if default_model:
                config_data['default_model'] = default_model
            self.config = QwenAdapterConfig(**config_data)

        self.client = httpx.AsyncClient(
            base_url=self.config.base_url,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            }
        )

    async def execute(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Makes a request to the Qwen API via DashScope."""
        effective_model = model or self.config.default_model
        endpoint = "/api/v1/services/aigc/text-generation/generation"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": effective_model,
            "input": {
                "messages": messages
            },
            "parameters": {
                "result_format": "message"
            }
        }

        if temperature is not None:
            payload["parameters"]["temperature"] = temperature

        if kwargs:
            payload["parameters"].update(kwargs)

        if not payload["parameters"]:
             del payload["parameters"]


        try:
            response = await self.client.post(endpoint, json=payload, timeout=180.0)

            try:
                response_data = await response.json()
            except Exception:
                return {
                    "error": f"Failed to decode JSON response from Qwen API (status {response.status_code}): {response.text}",
                    "raw_response": response.text,
                    "text_response": None,
                    "model_used": effective_model,
                }

            response.raise_for_status()

            if response_data.get("code"):
                return {
                    "error": f"Qwen API error (code: {response_data.get('code')}): {response_data.get('message')}",
                    "raw_response": response_data,
                    "text_response": None,
                    "model_used": effective_model,
                }

            output = response_data.get("output")
            text_response = None
            finish_reason = None

            if output and isinstance(output, dict):
                if output.get("message") and isinstance(output["message"], dict):
                     text_response = output["message"].get("content")
                finish_reason = output.get("finish_reason")


            return {
                "raw_response": response_data,
                "text_response": text_response,
                "model_used": effective_model,
                "finish_reason": finish_reason,
                "usage": response_data.get("usage")
            }

        except httpx.HTTPStatusError as e:
            error_content = "Unknown error content"
            try:
                error_content = await e.response.json()
            except Exception:
                error_content = e.response.text

            return {
                "error": f"HTTP error {e.response.status_code} from Qwen API: {error_content}",
                "raw_response_error": error_content,
                "text_response": None,
                "model_used": effective_model,
            }
        except httpx.RequestError as e:
            return {
                "error": f"Request error connecting to Qwen API: {e}",
                "raw_response": None,
                "text_response": None,
                "model_used": effective_model,
            }
        except Exception as e:
            return {
                "error": f"An unexpected error occurred with QwenAdapter: {e}",
                "raw_response": None,
                "text_response": None,
                "model_used": effective_model,
            }

    async def close(self):
        """Close the underlying HTTPX client."""
        if hasattr(self, 'client'):
            await self.client.aclose()
