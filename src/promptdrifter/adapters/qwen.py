import os
from typing import Any, Dict, Optional

import httpx
from pydantic import Field, model_validator
from rich.console import Console

from ..config.adapter_settings import (
    API_KEY_ENV_VAR_QWEN,
    DEFAULT_QWEN_MODEL,
    QWEN_API_BASE_URL,
)
from ..http_client_manager import get_shared_client
from .base import Adapter, BaseAdapterConfig
from .models.qwen_models import (
    QwenError,
    QwenHeaders,
    QwenMessage,
    QwenPayload,
    QwenResponse,
    QwenStandardResponse,
)

console = Console()


class QwenAdapterConfig(BaseAdapterConfig):
    base_url: str = QWEN_API_BASE_URL
    default_model: str = DEFAULT_QWEN_MODEL
    api_key: Optional[str] = Field(default=None, validate_default=True)
    max_tokens: Optional[int] = Field(default=1536, validate_default=True)
    temperature: Optional[float] = None
    system_prompt: Optional[str] = None

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

    def get_headers(self) -> Dict[str, str]:
        return QwenHeaders(
            Authorization=f"Bearer {self.api_key}"
        ).model_dump(by_alias=True)

    def get_payload(
            self,
            prompt: str,
            config_override: Optional["QwenAdapterConfig"] = None
        ) -> Dict[str, Any]:
        effective_config = config_override or self
        messages = []
        if effective_config.system_prompt:
            messages.append(QwenMessage(role="system", content=effective_config.system_prompt))
        messages.append(QwenMessage(role="user", content=prompt))

        payload = QwenPayload(
            model=effective_config.default_model,
            messages=messages,
            temperature=effective_config.temperature,
            max_tokens=effective_config.max_tokens
        )
        return payload.model_dump(exclude_none=True)


class QwenAdapter(Adapter):
    """Adapter for interacting with Alibaba Cloud Qwen (Tongyi Qianwen) API via DashScope."""

    def __init__(
        self,
        config: Optional[QwenAdapterConfig] = None,
    ):
        self.config = config or QwenAdapterConfig()
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
        config_override: Optional[QwenAdapterConfig] = None,
    ) -> QwenStandardResponse:
        """Makes a request to the Qwen API (OpenAI compatible chat completions)."""
        effective_config = config_override or self.config
        endpoint = "/chat/completions"
        payload = self.config.get_payload(prompt, effective_config)

        response = QwenStandardResponse(model_name=effective_config.default_model)

        try:
            client = await self._get_client()
            http_response = await client.post(endpoint, json=payload, timeout=180.0)
            http_response.raise_for_status()
            raw_response_content = http_response.json()
            response.raw_response = raw_response_content

            if "error" in raw_response_content:
                error_data = raw_response_content["error"]
                if isinstance(error_data, dict):
                    qwen_error = QwenError.model_validate(error_data)
                    response.error = f"Qwen API error (type: {qwen_error.type}, code: {qwen_error.code}): {qwen_error.message}"
                else:
                    response.error = f"Qwen API error: {error_data}"
                response.finish_reason = "error"
                return response

            qwen_response = QwenResponse.model_validate(raw_response_content)
            if qwen_response.code and qwen_response.code != "Success":
                response.error = f"Qwen API error (code: {qwen_response.code}): {qwen_response.message}"
                response.finish_reason = "error"
                return response

            if qwen_response.choices:
                first_choice = qwen_response.choices[0]
                response.text_response = first_choice.message.content
                response.finish_reason = first_choice.finish_reason
            else:
                response.error = "No choices found in Qwen response."

            if qwen_response.usage:
                response.usage = qwen_response.usage

        except httpx.HTTPStatusError as e:
            error_detail = "Unknown error"
            raw_error_content_text = e.response.text
            try:
                raw_error_data = e.response.json()
                response.raw_response = raw_error_data
                if isinstance(raw_error_data, dict):
                    if "error" in raw_error_data and isinstance(raw_error_data["error"], dict):
                        err_dict = raw_error_data["error"]
                        error_detail = f"(type: {err_dict.get('type')}, code: {err_dict.get('code')}) {err_dict.get('message')}"
                    elif raw_error_data.get("code") and raw_error_data.get("message"):
                        error_detail = f"(code: {raw_error_data.get('code')}) {raw_error_data.get('message')}"
                    elif "message" in raw_error_data:
                        error_detail = raw_error_data["message"]
                    else:
                        error_detail = str(raw_error_data)
                else:
                    error_detail = raw_error_content_text
            except Exception:
                response.raw_response = {"error_detail": raw_error_content_text}
                error_detail = raw_error_content_text
            response.error = f"HTTP error {e.response.status_code}: {error_detail}"
            response.raw_response = {"error_detail": e.response.text}
            response.text_response = None
            response.model_name = effective_config.default_model
            response.finish_reason = "error"

        except httpx.RequestError as e:
            response.error = f"Request error connecting to Qwen API: {e}"
            response.raw_response = {"error_detail": str(e)}
            response.finish_reason = "error"
        except Exception as e:
            console.print_exception()
            response.error = f"An unexpected error occurred with QwenAdapter: {type(e).__name__} - {e}"
            response.raw_response = {"error_detail": str(e)}
            response.finish_reason = "error"

        return response

    async def close(self):
        """Close method - HTTP connections managed by shared client manager."""
        if hasattr(self, "_client") and self._client is not None:
            self._client = None
