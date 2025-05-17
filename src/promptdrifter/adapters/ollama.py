import json
from typing import Any, AsyncGenerator, Dict, Optional

import httpx
from pydantic import BaseModel

from ..config.adapter_settings import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
)
from .base import Adapter


class OllamaAdapterConfig(BaseModel):
    base_url: str = DEFAULT_OLLAMA_BASE_URL
    default_model: str = DEFAULT_OLLAMA_MODEL


class OllamaAdapter(Adapter):
    """Adapter for interacting with a local Ollama API."""

    def __init__(
        self,
        config: Optional[OllamaAdapterConfig] = None,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None
    ):
        if config:
            self.config = config
        else:
            config_data = {}
            if base_url:
                config_data['base_url'] = base_url
            if default_model:
                config_data['default_model'] = default_model
            self.config = OllamaAdapterConfig(**config_data)

        self.client = httpx.AsyncClient(base_url=self.config.base_url)

    async def _stream_response(
        self, response: httpx.Response
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Helper to stream and parse JSON objects from the response."""
        buffer = ""
        async for chunk in response.aiter_bytes():
            buffer += chunk.decode("utf-8", errors="replace")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        pass
        if buffer.strip():
            try:
                yield json.loads(buffer)
            except json.JSONDecodeError:
                pass

    async def execute(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Makes a request to the Ollama /api/generate endpoint."""
        effective_model = model or self.config.default_model

        payload = {
            "model": effective_model,
            "prompt": prompt,
            "stream": stream,
            "options": {},
        }

        if temperature is not None:
            payload["options"]["temperature"] = temperature
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens

        for key, value in kwargs.items():
            if key not in ["model", "prompt", "stream", "temperature", "max_tokens"]:
                payload["options"][key] = value

        if not payload["options"]:
            del payload["options"]

        full_response_text = ""
        raw_response_parts = []

        try:
            if payload.get("stream") is True:
                async with self.client.stream(
                    "POST", "/api/generate", json=payload, timeout=120.0
                ) as response:
                    response.raise_for_status()
                    async for part in self._stream_response(response):
                        raw_response_parts.append(part)
                        if part.get("response"):
                            full_response_text += part["response"]
                        if part.get("done") and part.get("done") is True:
                            pass
                final_context = (
                    raw_response_parts[-1]
                    if raw_response_parts and raw_response_parts[-1].get("done")
                    else {}
                )
                return {
                    "raw_response": {
                        "parts": raw_response_parts,
                        "final_context": final_context,
                    },
                    "text_response": full_response_text.strip()
                    if full_response_text
                    else None,
                    "model_used": effective_model,
                }
            else:
                response = await self.client.post(
                    "/api/generate", json=payload, timeout=120.0
                )
                response.raise_for_status()
                response_data = response.json()
                return {
                    "raw_response": response_data,
                    "text_response": response_data.get("response", "").strip()
                    if response_data
                    else None,
                    "model_used": effective_model,
                }

        except httpx.HTTPStatusError as e:
            error_content = e.response.text
            return {
                "error": f"HTTP error {e.response.status_code} from Ollama: {error_content}",
                "raw_response": None,
                "text_response": None,
                "model_used": effective_model,
            }
        except httpx.RequestError as e:
            return {
                "error": f"Request error connecting to Ollama: {e}",
                "raw_response": None,
                "text_response": None,
                "model_used": effective_model,
            }
        except json.JSONDecodeError as e:
            return {
                "error": f"Failed to decode JSON response from Ollama: {e}",
                "raw_response": None,
                "text_response": None,
                "model_used": effective_model,
            }
        except Exception as e:
            return {
                "error": f"An unexpected error occurred with Ollama: {e}",
                "raw_response": None,
                "text_response": None,
                "model_used": effective_model,
            }

    async def close(self):
        """Close the underlying HTTPX client."""
        await self.client.aclose()
