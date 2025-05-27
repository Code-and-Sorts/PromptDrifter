from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class BaseAdapterConfig(BaseModel, ABC):
    """Base configuration for all LLM adapters."""
    base_url: str = Field(...)
    default_model: str = Field(...)
    api_key: Optional[str] = None
    max_tokens: Optional[int] = None

    @abstractmethod
    def get_headers(self) -> dict:
        pass

    @abstractmethod
    def get_payload(self, prompt: str, config_override: Optional["BaseAdapterConfig"] = None) -> dict:
        pass


class Adapter(ABC):
    """Abstract Base Class for all LLM adapters."""

    @abstractmethod
    async def execute(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        # Allow for other adapter-specific parameters
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Execute the prompt against the specified model and return the response.

        Args:
            prompt: The prompt to send to the model.
            model: The specific model to use (if applicable to the adapter).
            temperature: The sampling temperature.
            max_tokens: The maximum number of tokens to generate.
            **kwargs: Additional adapter-specific parameters.

        Returns:
            A dictionary containing the model's response and any other relevant information.
            Typically, this would include a key like 'text_response' or 'choices'.
        """
        pass

    async def close(self):
        """Close any resources used by the adapter."""
        pass
