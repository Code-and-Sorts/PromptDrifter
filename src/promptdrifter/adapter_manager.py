"""
Optimized adapter management system for PromptDrifter.
Eliminates expensive reflection and enables adapter reuse.
"""

import asyncio
import hashlib
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type

from .adapters.base import Adapter
from .adapters.claude import ClaudeAdapter
from .adapters.deepseek import DeepSeekAdapter
from .adapters.gemini import GeminiAdapter
from .adapters.grok import GrokAdapter
from .adapters.mistral import MistralAdapter
from .adapters.ollama import OllamaAdapter
from .adapters.openai import OpenAIAdapter
from .adapters.qwen import QwenAdapter
from .http_client_manager import get_http_client_manager


@dataclass(frozen=True)
class AdapterKey:
    """Immutable key for adapter caching."""
    adapter_type: str
    api_key_hash: Optional[str]
    base_url: Optional[str]

    @classmethod
    def from_config(cls, adapter_type: str, api_key: Optional[str], base_url: Optional[str]) -> "AdapterKey":
        """Create adapter key from configuration."""
        api_key_hash = None
        if api_key:
            # Hash API key for security - don't store raw keys
            api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]

        return cls(
            adapter_type=adapter_type.lower(),
            api_key_hash=api_key_hash,
            base_url=base_url
        )


class AdapterMetadata:
    """Pre-computed metadata for adapter classes to avoid reflection."""
    def __init__(self, adapter_class: Type[Adapter]):
        self.adapter_class = adapter_class
        self.config_class = getattr(adapter_class, 'config_class', None)

        if self.config_class:
            try:
                config_fields = self.config_class.model_fields
                self.supports_base_url = 'base_url' in config_fields
                self.supports_api_key = 'api_key' in config_fields
            except AttributeError:
                self.supports_base_url = hasattr(self.config_class, 'base_url')
                self.supports_api_key = hasattr(self.config_class, 'api_key')
        else:
            self.supports_base_url = False
            self.supports_api_key = False


class AdapterManager:
    """
    High-performance adapter manager with caching and connection pooling.
    Eliminates reflection overhead and enables adapter reuse.
    """

    def __init__(self):
        self._adapter_registry: Dict[str, AdapterMetadata] = {}

        self._adapter_cache: Dict[AdapterKey, Adapter] = {}

        self._usage_count: Dict[AdapterKey, int] = {}

        self._cache_lock = asyncio.Lock()

        self._initialize_registry()

    def _initialize_registry(self):
        """Initialize adapter registry with metadata."""
        adapter_classes = {
            "claude": ClaudeAdapter,
            "deepseek": DeepSeekAdapter,
            "gemini": GeminiAdapter,
            "grok": GrokAdapter,
            "mistral": MistralAdapter,
            "openai": OpenAIAdapter,
            "ollama": OllamaAdapter,
            "qwen": QwenAdapter,
        }

        for name, adapter_class in adapter_classes.items():
            self._adapter_registry[name] = AdapterMetadata(adapter_class)

    async def get_adapter(
        self,
        adapter_type: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> Optional[Adapter]:
        """
        Get an adapter instance, reusing existing ones when possible.

        Args:
            adapter_type: Type of adapter (e.g., 'openai', 'claude')
            api_key: API key for the adapter
            base_url: Base URL for the adapter

        Returns:
            Adapter instance or None if not found
        """
        adapter_type = adapter_type.lower()

        if adapter_type not in self._adapter_registry:
            return None

        metadata = self._adapter_registry[adapter_type]
        cache_key = AdapterKey.from_config(adapter_type, api_key, base_url)

        async with self._cache_lock:
            if cache_key in self._adapter_cache:
                self._usage_count[cache_key] = self._usage_count.get(cache_key, 0) + 1
                return self._adapter_cache[cache_key]

            try:
                adapter_instance = await self._create_adapter_instance(
                    metadata, api_key, base_url
                )

                if adapter_instance:
                    self._adapter_cache[cache_key] = adapter_instance
                    self._usage_count[cache_key] = 1

                return adapter_instance

            except Exception as e:
                print(f"Error creating adapter '{adapter_type}': {e}")
                return None

    async def _create_adapter_instance(
        self,
        metadata: AdapterMetadata,
        api_key: Optional[str],
        base_url: Optional[str]
    ) -> Optional[Adapter]:
        """Create a new adapter instance using pre-computed metadata."""
        if metadata.config_class:
            config_params = {}

            if metadata.supports_api_key and api_key:
                config_params["api_key"] = api_key

            if metadata.supports_base_url and base_url:
                config_params["base_url"] = base_url

            config = metadata.config_class(**config_params)
            return metadata.adapter_class(config=config)
        else:
            return metadata.adapter_class()

    @asynccontextmanager
    async def adapter_context(
        self,
        adapter_type: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        Context manager for adapter usage with automatic cleanup.

        Usage:
            async with adapter_manager.adapter_context("openai", api_key) as adapter:
                result = await adapter.execute("prompt")
        """
        adapter = await self.get_adapter(adapter_type, api_key, base_url)
        if adapter is None:
            raise ValueError(f"Could not create adapter: {adapter_type}")

        try:
            yield adapter
        finally:
            pass

    async def cleanup_unused_adapters(self, min_usage: int = 1):
        """Clean up adapters that haven't been used much."""
        async with self._cache_lock:
            keys_to_remove = []

            for cache_key, usage in self._usage_count.items():
                if usage < min_usage:
                    keys_to_remove.append(cache_key)

            for key in keys_to_remove:
                if key in self._adapter_cache:
                    adapter = self._adapter_cache[key]
                    if hasattr(adapter, "close"):
                        try:
                            await adapter.close()
                        except Exception:
                            pass

                    del self._adapter_cache[key]
                    del self._usage_count[key]

    async def close_all_adapters(self):
        """Close all cached adapters and HTTP connections."""
        async with self._cache_lock:
            close_tasks = []

            for adapter in self._adapter_cache.values():
                if hasattr(adapter, "close"):
                    close_tasks.append(adapter.close())

            if close_tasks:
                await asyncio.gather(*close_tasks, return_exceptions=True)

            self._adapter_cache.clear()
            self._usage_count.clear()

        http_manager = get_http_client_manager()
        await http_manager.close_all()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about adapter cache usage."""
        return {
            "cached_adapters": len(self._adapter_cache),
            "total_usage": sum(self._usage_count.values()),
            "adapter_types": list(set(key.adapter_type for key in self._adapter_cache.keys())),
            "usage_by_type": {
                adapter_type: sum(
                    usage for key, usage in self._usage_count.items()
                    if key.adapter_type == adapter_type
                )
                for adapter_type in set(key.adapter_type for key in self._adapter_cache.keys())
            }
        }


# Global adapter manager instance
_adapter_manager = None


def get_adapter_manager() -> AdapterManager:
    """Get the global adapter manager instance."""
    global _adapter_manager
    if _adapter_manager is None:
        _adapter_manager = AdapterManager()
    return _adapter_manager
