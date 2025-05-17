import os
from typing import Any, Dict, Optional

import httpx
from pydantic import BaseModel, Field, model_validator

from ..config.adapter_settings import (
    API_KEY_ENV_VAR_GEMINI,
    DEFAULT_GEMINI_MODEL,
    GEMINI_API_BASE_URL,
)
from .base import Adapter


class GeminiAdapterConfig(BaseModel):
    api_key: Optional[str] = Field(default=None, validate_default=True)
    base_url: str = GEMINI_API_BASE_URL
    default_model: str = DEFAULT_GEMINI_MODEL

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


class GeminiAdapter(Adapter):
    """Adapter for interacting with Google Gemini API via REST using httpx."""

    def __init__(
        self,
        config: Optional[GeminiAdapterConfig] = None,
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
            self.config = GeminiAdapterConfig(**config_data)

        self.client = httpx.AsyncClient(base_url=self.config.base_url)

    async def execute(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Makes a REST request to the Google Gemini API."""
        effective_model = model or self.config.default_model
        endpoint = f"/models/{effective_model}:generateContent"
        params = {"key": self.config.api_key}

        payload = {"contents": [{"parts": [{"text": prompt}]}]}

        generation_config: Dict[str, Any] = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        if max_tokens is not None:
            generation_config["maxOutputTokens"] = max_tokens

        generation_config.update(kwargs.get("generation_config", {}))

        if generation_config:
            payload["generationConfig"] = generation_config

        if "safetySettings" in kwargs:
            payload["safetySettings"] = kwargs["safetySettings"]

        try:
            response = await self.client.post(
                endpoint, params=params, json=payload, timeout=60.0
            )
            response.raise_for_status()
            response_data = response.json()

            text_response = None
            finish_reason = None
            usage_metadata = response_data.get("usageMetadata")
            safety_ratings = None

            parsing_error_message = None

            try:
                candidates = response_data.get("candidates")
                if candidates and isinstance(candidates, list) and len(candidates) > 0:
                    first_candidate = candidates[0]
                    finish_reason = first_candidate.get("finishReason")
                    safety_ratings = first_candidate.get("safetyRatings")

                    content = first_candidate.get("content")
                    if content and isinstance(content, dict):
                        parts = content.get("parts")
                        if parts and isinstance(parts, list) and len(parts) > 0:
                            first_part = parts[0]
                            if isinstance(first_part, dict) and "text" in first_part:
                                text_response = first_part["text"]
                            else:
                                parsing_error_message = "Text not found in the first part of the first candidate."
                        else:
                            parsing_error_message = "Parts not found or empty in the first candidate's content."
                    else:
                        parsing_error_message = (
                            "Content not found in the first candidate."
                        )
                else:
                    parsing_error_message = (
                        "Candidates not found, empty, or not a list in the response."
                    )

            except (KeyError, IndexError, TypeError) as e:
                parsing_error_message = f"Error parsing Gemini response structure: {e}"

            if text_response is None and parsing_error_message:
                return {
                    "error": parsing_error_message,
                    "raw_response": response_data,
                    "text_response": None,
                    "model_used": effective_model,
                }

            return {
                "raw_response": response_data,
                "text_response": text_response,
                "model_used": effective_model,
                "finish_reason": finish_reason,
                "usage_metadata": usage_metadata,
                "safety_ratings": safety_ratings,
            }
        except httpx.HTTPStatusError as e:
            error_detail = "Unknown error"
            try:
                error_data = e.response.json()
                error_detail = error_data.get("error", {}).get(
                    "message", e.response.text
                )
            except Exception:
                error_detail = e.response.text
            return {
                "error": f"HTTP error {e.response.status_code} from Gemini API: {error_detail}",
                "raw_response": error_detail,
                "text_response": None,
                "model_used": effective_model,
            }
        except httpx.RequestError as e:
            return {
                "error": f"Request error connecting to Gemini API: {e}",
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
        """Close the underlying httpx client."""
        await self.client.aclose()
