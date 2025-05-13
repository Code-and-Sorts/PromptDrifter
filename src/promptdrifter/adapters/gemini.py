import os
from typing import Any, Dict, Optional

import httpx

from ..config.adapter_settings import (
    API_KEY_ENV_VAR_GEMINI,
    DEFAULT_GEMINI_MODEL,
    GEMINI_API_BASE_URL,
)
from .base import Adapter


class GeminiAdapter(Adapter):
    """Adapter for interacting with Google Gemini API via REST using httpx."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or os.getenv(API_KEY_ENV_VAR_GEMINI)
        if not self.api_key:
            raise ValueError(
                f"Gemini API key not provided. Set the {API_KEY_ENV_VAR_GEMINI} environment variable "
                f"or pass it to the adapter constructor."
            )
        self.base_url = base_url or GEMINI_API_BASE_URL
        # Initialize httpx.AsyncClient. Headers are not needed as API key is in query params.
        self.client = httpx.AsyncClient(base_url=self.base_url)

    async def execute(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Makes a REST request to the Google Gemini API."""
        effective_model = model or DEFAULT_GEMINI_MODEL
        endpoint = f"/models/{effective_model}:generateContent"
        params = {"key": self.api_key}

        # Construct the payload based on Gemini REST API structure
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }

        generation_config: Dict[str, Any] = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        if max_tokens is not None:
            # Gemini REST API uses maxOutputTokens
            generation_config["maxOutputTokens"] = max_tokens

        # Allow overriding other generationConfig parameters via kwargs
        # Example: kwargs = {"generation_config": {"topP": 0.9}}
        generation_config.update(kwargs.get("generation_config", {}))

        if generation_config:
             # Add generationConfig to the payload only if it's not empty
            payload["generationConfig"] = generation_config

        # Allow overriding safetySettings or other top-level parameters via kwargs
        # Example: kwargs = {"safetySettings": [...]}
        if "safetySettings" in kwargs:
             payload["safetySettings"] = kwargs["safetySettings"]
        # Add other potential top-level parameters if needed based on API docs


        try:
            response = await self.client.post(
                endpoint, params=params, json=payload, timeout=60.0
            )
            response.raise_for_status()  # Raise HTTPStatusError for bad responses (4xx or 5xx)
            response_data = response.json()

            # Extract text response based on the API documentation structure
            text_response = None
            finish_reason = None
            usage_metadata = response_data.get("usageMetadata")
            safety_ratings = None # Initialize safety_ratings

            parsing_error_message = None # To store a specific parsing error message

            try:
                # Navigate the response structure carefully
                candidates = response_data.get("candidates")
                if candidates and isinstance(candidates, list) and len(candidates) > 0:
                    first_candidate = candidates[0]
                    finish_reason = first_candidate.get("finishReason")
                    safety_ratings = first_candidate.get("safetyRatings") # Get safety_ratings here

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
                        parsing_error_message = "Content not found in the first candidate."
                else:
                    # This case is critical for the failing test
                    parsing_error_message = "Candidates not found, empty, or not a list in the response."

            except (KeyError, IndexError, TypeError) as e:
                # This handles errors during the above navigation if something is unexpectedly not a dict/list or an index is out of bounds
                parsing_error_message = f"Error parsing Gemini response structure: {e}"


            # If text_response is still None after attempting to parse, and no broader HTTP/Request error occurred,
            # it implies a structural issue with the response payload not caught by the specific exceptions above.
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
                "safety_ratings": safety_ratings
            }
        except httpx.HTTPStatusError as e:
            error_detail = "Unknown error"
            try:
                 # Try to parse error details from response if available
                error_data = e.response.json()
                error_detail = error_data.get("error", {}).get("message", e.response.text)
            except Exception:
                error_detail = e.response.text # Fallback to raw text
            return {
                "error": f"HTTP error {e.response.status_code} from Gemini API: {error_detail}",
                "raw_response": error_detail, # Provide error detail as raw response
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
            # Catch-all for other unexpected errors during the request or processing
            return {
                "error": f"An unexpected error occurred: {e}",
                "raw_response": None,
                "text_response": None,
                "model_used": effective_model,
            }

    async def close(self):
        """Close the underlying httpx client."""
        await self.client.aclose()
