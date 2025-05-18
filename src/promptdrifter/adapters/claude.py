import asyncio
import os
from typing import Any, Dict, Optional

import httpx
from rich.console import Console

from promptdrifter.adapters.base import Adapter
from promptdrifter.config.adapter_settings import (
    API_KEY_ENV_VAR_CLAUDE,
    CLAUDE_API_BASE_URL,
    CLAUDE_API_VERSION,
    DEFAULT_CLAUDE_MODEL,
)

console = Console()


class ClaudeAdapter(Adapter):
    """Adapter for Anthropic Claude models."""

    DEFAULT_MODEL = DEFAULT_CLAUDE_MODEL
    API_ENDPOINT = CLAUDE_API_BASE_URL
    DEFAULT_MAX_TOKENS = 1024

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv(API_KEY_ENV_VAR_CLAUDE)
        if not self.api_key:
            raise ValueError(
                f"Claude API key not provided. Please set {API_KEY_ENV_VAR_CLAUDE} environment variable."
            )
        self.headers = {
            "x-api-key": self.api_key,
            "anthropic-version": CLAUDE_API_VERSION,
            "content-type": "application/json",
        }

    async def execute(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Execute the prompt against the specified Claude model.
        See: https://docs.anthropic.com/claude/reference/messages_post
        """
        if not self.api_key:
            return {
                "error": "Anthropic API key not configured.",
                "text_response": None,
                "raw_response": {"error": "API key missing after init"},
                "model_name": model or self.DEFAULT_MODEL,
                "finish_reason": "error",
            }

        selected_model = model or self.DEFAULT_MODEL
        selected_max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS
        endpoint = base_url or self.API_ENDPOINT

        payload: Dict[str, Any] = {
            "model": selected_model,
            "max_tokens": selected_max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }

        if temperature is not None:
            payload["temperature"] = temperature
        if system_prompt:
            payload["system"] = system_prompt

        raw_response_content: Optional[Dict[str, Any]] = None
        text_response: Optional[str] = None
        finish_reason: Optional[str] = None
        error_message: Optional[str] = None

        response_dict = {
            "text_response": None,
            "raw_response": None,
            "model_name": selected_model,
            "finish_reason": None,
            "error": None,
            "usage": None,
        }

        client = httpx.AsyncClient()
        try:
            response = await client.post(
                endpoint, headers=self.headers, json=payload, timeout=60.0
            )
            response.raise_for_status()

            raw_response_content = response.json()
            if (
                raw_response_content
                and isinstance(raw_response_content.get("content"), list)
                and len(raw_response_content["content"]) > 0
            ):
                for block in raw_response_content["content"]:
                    if block.get("type") == "text":
                        text_response = block.get("text")
                        break
                if not text_response:
                    error_message = (
                        "No text content found in successful response."
                    )
            else:
                error_message = "Unexpected response structure for 200 OK."

            finish_reason = (
                raw_response_content.get("stop_reason")
                if raw_response_content
                else "unknown"
            )
            response_dict["usage"] = (
                raw_response_content.get("usage")
                if raw_response_content
                else None
            )

        except httpx.HTTPStatusError as e:
            error_detail = "Unknown error"
            try:
                raw_response_content = e.response.json()
                if raw_response_content and isinstance(
                    raw_response_content.get("error"), dict
                ):
                    error_detail = raw_response_content["error"].get(
                        "message", str(raw_response_content["error"])
                    )
                elif (
                    isinstance(raw_response_content, dict)
                    and "detail" in raw_response_content
                ):
                    error_detail = str(raw_response_content.get("detail"))
                else:
                    error_detail = str(raw_response_content)
                error_message = f"API Error (HTTP {e.response.status_code}): {error_detail}"
            except Exception:
                error_detail = e.response.text
                error_message = f"API Error (HTTP {e.response.status_code}): {e.response.reason_phrase or error_detail}"

            finish_reason = "error"

        except httpx.RequestError as e:
            error_message = f"HTTP Client Error: {e}"
            finish_reason = "error"
            raw_response_content = {"error": str(e)}

        except asyncio.TimeoutError:
            error_message = "Request timed out."
            finish_reason = "error"
            raw_response_content = {"error": "Timeout"}

        except Exception as e:
            console.print_exception()
            error_message = f"An unexpected error occurred: {e}"
            finish_reason = "error"
            raw_response_content = {"error": str(e)}

        finally:
            await client.aclose()

        response_dict["text_response"] = text_response
        response_dict["raw_response"] = raw_response_content
        response_dict["finish_reason"] = finish_reason
        if error_message:
            response_dict["error"] = error_message
            if not text_response:
                response_dict["text_response"] = None

        return response_dict
