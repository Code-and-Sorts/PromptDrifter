import json
from typing import Any, AsyncGenerator, Dict, Optional

import httpx
from rich.console import Console

from ..config.adapter_settings import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
)
from ..http_client_manager import get_shared_client
from .base import Adapter, BaseAdapterConfig
from .models import (
    OllamaErrorResponse,
    OllamaHeaders,
    OllamaPayload,
    OllamaRawResponse,
    OllamaResponse,
)

console = Console()


class OllamaAdapterConfig(BaseAdapterConfig):
    """Configuration for Ollama API adapter."""
    base_url: str = DEFAULT_OLLAMA_BASE_URL
    default_model: str = DEFAULT_OLLAMA_MODEL
    # Ollama doesn't require an API key since it's locally hosted
    api_key: Optional[str] = None
    max_tokens: Optional[int] = 1024 # Standard default, can be overridden

    def get_headers(self) -> Dict[str, str]:
        return OllamaHeaders().model_dump()

    def get_payload(
            self,
            prompt: str,
            config_override: Optional["OllamaAdapterConfig"] = None,
            stream: bool = False
        ) -> Dict[str, Any]:
        effective_model = config_override.default_model if config_override else self.default_model
        effective_max_tokens = config_override.max_tokens if config_override else self.max_tokens
        # Assuming temperature and other options might be added to OllamaAdapterConfig later
        options = {}
        if effective_max_tokens is not None:
            options["num_predict"] = effective_max_tokens

        # Allow direct override of options if provided in config_override
        if config_override and hasattr(config_override, 'options') and config_override.options:
            options.update(config_override.options)

        payload = OllamaPayload(
            model=effective_model,
            prompt=prompt,
            stream=stream, # Pass stream parameter here
            options=options if options else None
        )
        return payload.model_dump(exclude_none=True)


class OllamaAdapter(Adapter):
    """Adapter for interacting with a local Ollama API."""

    def __init__(
        self,
        config: Optional[OllamaAdapterConfig] = None,
    ):
        self.config = config or OllamaAdapterConfig()
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
        config_override: Optional[OllamaAdapterConfig] = None,
        stream: bool = False # Add stream parameter here
    ) -> OllamaResponse:
        """Makes a request to the Ollama /api/generate endpoint."""
        effective_model = config_override.default_model if config_override else self.config.default_model
        # Pass the stream argument to get_payload
        payload = self.config.get_payload(prompt, config_override, stream=stream)

        response = OllamaResponse(model_name=effective_model)

        try:
            if stream:
                client = await self._get_client()
                async with client.stream(
                    "POST", "/api/generate", json=payload, timeout=120.0
                ) as http_response:
                    http_response.raise_for_status()
                    full_response_text = ""
                    raw_response_parts = []
                    final_part_data = {}
                    async for part_str in self._stream_response_lines(http_response):
                        try:
                            part_data = json.loads(part_str)
                            raw_response_parts.append(part_data)
                            if part_data.get("response"):
                                full_response_text += part_data["response"]
                            if part_data.get("done"):
                                final_part_data = part_data # Capture the final part with context
                        except json.JSONDecodeError:
                            # Handle cases where a line is not valid JSON (e.g. empty lines)
                            pass

                    response.text_response = full_response_text.strip() if full_response_text else None
                    # Use final_part_data for usage and finish_reason if available
                    if final_part_data:
                        response.raw_response = {"parts": raw_response_parts, "final_context": final_part_data}
                        response.finish_reason = "completed" if final_part_data.get("done") else "unknown"
                        response.usage = {
                            "prompt_tokens": final_part_data.get("prompt_eval_count", 0),
                            "completion_tokens": final_part_data.get("eval_count", 0),
                            "total_tokens": final_part_data.get("prompt_eval_count", 0) + final_part_data.get("eval_count", 0)
                        }
                    else:
                        response.raw_response = {"parts": raw_response_parts}
                        response.finish_reason = "stream_ended_without_done_flag" # Or some other indicator

            else: # Non-streaming case
                client = await self._get_client()
                http_response = await client.post(
                    "/api/generate", json=payload, timeout=120.0
                )
                http_response.raise_for_status()
                response_data = http_response.json()
                raw_response_model = OllamaRawResponse.model_validate(response_data)
                response = raw_response_model.to_standard_response(effective_model)

        except httpx.HTTPStatusError as e:
            error_content = self._extract_ollama_error_message(e.response)
            response.error = f"HTTP error {e.response.status_code} from Ollama: {error_content}"
            try:
                response.raw_response = e.response.json()
            except json.JSONDecodeError:
                response.raw_response = {"error_detail": e.response.text}
            response.text_response = None
            response.finish_reason = "error"
        except httpx.RequestError as e:
            response.error = f"Request error connecting to Ollama: {e}"
            response.raw_response = {"error_detail": str(e)}
            response.text_response = None
            response.finish_reason = "error"
        except json.JSONDecodeError as e:
            response.error = f"Failed to decode JSON response from Ollama: {e}"
            response.raw_response = {"error_detail": str(e)}
            response.text_response = None
            response.finish_reason = "error"
        except Exception as e:
            console.print_exception()
            response.error = f"An unexpected error occurred with Ollama: {e}"
            response.raw_response = {"error_detail": str(e)}
            response.text_response = None
            response.finish_reason = "error"
        return response

    async def close(self):
        """Close method - HTTP connections managed by shared client manager."""
        self._client = None

    def _extract_ollama_error_message(self, response: httpx.Response) -> str:
        try:
            error_data = response.json()
            error_response = OllamaErrorResponse.model_validate(error_data)
            return error_response.get_error_message()
        except json.JSONDecodeError:
            return response.text

    async def _stream_response_lines(
        self, response: httpx.Response
    ) -> AsyncGenerator[str, None]:
        """Helper to stream lines from the response."""
        buffer = ""
        async for chunk in response.aiter_bytes():
            buffer += chunk.decode("utf-8", errors="replace")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                if line.strip():
                    yield line
        if buffer.strip():
            yield buffer
