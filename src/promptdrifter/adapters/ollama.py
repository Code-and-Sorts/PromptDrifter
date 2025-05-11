import json  # For parsing streaming JSON responses
from typing import Any, AsyncGenerator, Dict, Optional

import httpx

from .base import Adapter

# Default Ollama API endpoint
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
# Default model if not specified (user should specify a model they have pulled)
DEFAULT_OLLAMA_MODEL = "llama3"


class OllamaAdapter(Adapter):
    """Adapter for interacting with a local Ollama API."""

    def __init__(
        self, base_url: Optional[str] = None, default_model: Optional[str] = None
    ):
        self.base_url = base_url or DEFAULT_OLLAMA_BASE_URL
        self.default_model = default_model or DEFAULT_OLLAMA_MODEL
        self.client = httpx.AsyncClient(base_url=self.base_url)

    async def _stream_response(
        self, response: httpx.Response
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Helper to stream and parse JSON objects from the response."""
        buffer = ""
        async for chunk in response.aiter_bytes():
            buffer += chunk.decode("utf-8", errors="replace")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                if line.strip():  # Ensure line is not empty
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        # Potentially log this error or handle incomplete JSON lines
                        # For now, we skip lines that are not valid JSON
                        # print(f"Skipping non-JSON line: {line}")
                        pass
        if buffer.strip():  # Process any remaining part of the buffer
            try:
                yield json.loads(buffer)
            except json.JSONDecodeError:
                # print(f"Skipping non-JSON line from remaining buffer: {buffer}")
                pass

    async def execute(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,  # Ollama uses 'temperature' in options
        max_tokens: Optional[int] = None,  # Ollama uses 'num_predict' in options
        stream: bool = False,  # Added stream parameter with default
        # Ollama specific options can be passed via kwargs, e.g., stop sequences
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Makes a request to the Ollama /api/generate endpoint."""
        effective_model = model or self.default_model

        payload = {
            "model": effective_model,
            "prompt": prompt,
            "stream": stream,  # Use the stream parameter from the method arguments
            "options": {},
        }

        if temperature is not None:
            payload["options"]["temperature"] = temperature
        if max_tokens is not None:
            # Ollama calls this num_predict. Max tokens is a common name, so we adapt.
            payload["options"]["num_predict"] = max_tokens

        # Allow overriding other Ollama options if passed via kwargs
        # These should be Ollama-specific option names like "stop", "top_k", "top_p" etc.
        for key, value in kwargs.items():
            if key not in ["model", "prompt", "stream", "temperature", "max_tokens"]:
                payload["options"][key] = value

        # If no options were added, remove the empty dict to avoid sending "options": {}
        if not payload["options"]:
            del payload["options"]

        full_response_text = ""
        raw_response_parts = []

        try:
            # Ollama's /api/generate endpoint for non-streaming
            # If stream=True, the response format is a stream of JSON objects, one per line.
            # If stream=False, it's a single JSON object.
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
                            # The final part contains context and other summary details
                            # We capture all parts for the raw_response
                            pass
                # For streaming, the final collected text is the text_response
                # and raw_response is the list of all JSON objects received.
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
            else:  # stream=False
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
