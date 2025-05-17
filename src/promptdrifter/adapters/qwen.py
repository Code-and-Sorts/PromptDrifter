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

    @model_validator(mode="before")
    @classmethod
    def load_api_key(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values.get("api_key"):
            return values

        api_key_from_env = os.getenv(API_KEY_ENV_VAR_QWEN)
        if api_key_from_env:
            values["api_key"] = api_key_from_env
            return values

        api_key_from_env = os.getenv("DASHSCOPE_API_KEY")
        if api_key_from_env:
            values["api_key"] = api_key_from_env
            return values

        return values

    @model_validator(mode="after")
    def check_api_key_present(self) -> "QwenAdapterConfig":
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
                config_data["api_key"] = api_key
            if base_url:
                config_data["base_url"] = base_url
            if default_model:
                config_data["default_model"] = default_model
            self.config = QwenAdapterConfig(**config_data)

        self.client = httpx.AsyncClient(
            base_url=self.config.base_url,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
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
        """Makes a request to the Qwen API (OpenAI compatible chat completions)."""
        effective_model = model or self.config.default_model
        endpoint = "/chat/completions"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": effective_model,
            "messages": messages,
        }

        if temperature is not None:
            payload["temperature"] = temperature

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        if kwargs:
            payload.update(kwargs)

        try:
            response = await self.client.post(endpoint, json=payload, timeout=180.0)
            response.raise_for_status()  # Check for HTTP errors first

            try:
                response_data = response.json()
            except Exception as json_decode_exception:
                return {
                    "error": f"Failed to decode JSON response from Qwen API (status {response.status_code}). Parser error: [{type(json_decode_exception).__name__}: {str(json_decode_exception)}]. Response text: {response.text}",
                    "raw_response": response.text,
                    "text_response": None,
                    "model_used": effective_model,
                }

            if "error" in response_data and isinstance(response_data["error"], dict):
                err_dict = response_data["error"]
                return {
                    "error": f"Qwen API error (type: {err_dict.get('type')}, code: {err_dict.get('code')}): {err_dict.get('message')}",
                    "raw_response": response_data,
                    "text_response": None,
                    "model_used": effective_model,
                }
            elif response_data.get("code") and response_data.get("message"):
                return {
                    "error": f"Qwen API error (code: {response_data.get('code')}): {response_data.get('message')}",
                    "raw_response": response_data,
                    "text_response": None,
                    "model_used": effective_model,
                }

            text_response = None
            finish_reason = None
            usage_data = None

            if response_data.get("choices") and isinstance(response_data["choices"], list) and len(response_data["choices"]) > 0:
                first_choice = response_data["choices"][0]
                if isinstance(first_choice, dict):
                    if first_choice.get("message") and isinstance(first_choice["message"], dict):
                        text_response = first_choice["message"].get("content")
                    finish_reason = first_choice.get("finish_reason")

            if response_data.get("usage") and isinstance(response_data["usage"], dict):
                usage_data = response_data["usage"]

            return {
                "raw_response": response_data,
                "text_response": text_response,
                "model_used": effective_model,
                "finish_reason": finish_reason,
                "usage": usage_data
            }

        except httpx.HTTPStatusError as e:
            error_content = "Unknown error content"
            error_detail_for_message = ""
            try:
                json_error_content = e.response.json()
                error_content = json_error_content
                if isinstance(json_error_content.get("error"), dict):
                    err_dict = json_error_content["error"]
                    api_type = err_dict.get("type")
                    api_code = err_dict.get("code")
                    api_message = err_dict.get("message")
                    if api_message:
                         error_detail_for_message = f" (type: {api_type}, code: {api_code}): {api_message}"
                    else:
                        error_detail_for_message = f": {json_error_content}"
                elif json_error_content.get("code") and json_error_content.get("message"):
                    api_code = json_error_content.get("code")
                    api_message = json_error_content.get("message")
                    error_detail_for_message = f" (code: {api_code}): {api_message}"
                else:
                    error_detail_for_message = f": {json_error_content}"
            except Exception:
                error_content = e.response.text
                error_detail_for_message = f": {e.response.text}"

            return {
                "error": f"HTTP error {e.response.status_code} from Qwen API{error_detail_for_message}",
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
        if hasattr(self, "client"):
            await self.client.aclose()
