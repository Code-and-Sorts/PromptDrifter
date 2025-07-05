import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from promptdrifter.adapter_manager import (
    AdapterKey,
    AdapterManager,
    AdapterMetadata,
    get_adapter_manager,
)
from promptdrifter.adapters.claude import ClaudeAdapter
from promptdrifter.adapters.openai import OpenAIAdapter, OpenAIAdapterConfig

async_tests = pytest.mark.asyncio


class TestAdapterKey:

    def test_adapter_key_creation(self):
        key = AdapterKey(
            adapter_type="openai",
            api_key_hash="test_hash",
            base_url="https://api.openai.com"
        )
        assert key.adapter_type == "openai"
        assert key.api_key_hash == "test_hash"
        assert key.base_url == "https://api.openai.com"

    def test_adapter_key_from_config(self):
        api_key = "test-api-key"
        expected_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]

        key = AdapterKey.from_config("OpenAI", api_key, "https://api.openai.com")

        assert key.adapter_type == "openai"
        assert key.api_key_hash == expected_hash
        assert key.base_url == "https://api.openai.com"

    def test_adapter_key_from_config_no_api_key(self):
        key = AdapterKey.from_config("claude", None, "https://api.anthropic.com")

        assert key.adapter_type == "claude"
        assert key.api_key_hash is None
        assert key.base_url == "https://api.anthropic.com"

    def test_adapter_key_immutable(self):
        key = AdapterKey("openai", "hash", "url")

        with pytest.raises(AttributeError):
            key.adapter_type = "claude"


class TestAdapterMetadata:

    def test_adapter_metadata_with_config_class(self):
        mock_adapter = MagicMock()
        mock_adapter.config_class = OpenAIAdapterConfig

        metadata = AdapterMetadata(mock_adapter)

        assert metadata.adapter_class == mock_adapter
        assert metadata.config_class == OpenAIAdapterConfig
        assert metadata.supports_api_key is True
        assert metadata.supports_base_url is True

    def test_adapter_metadata_without_config_class(self):
        mock_adapter = MagicMock()
        del mock_adapter.config_class

        metadata = AdapterMetadata(mock_adapter)

        assert metadata.adapter_class == mock_adapter
        assert metadata.config_class is None
        assert metadata.supports_api_key is False
        assert metadata.supports_base_url is False

    def test_adapter_metadata_config_fallback(self):
        mock_adapter = MagicMock()
        mock_config = MagicMock()
        mock_config.model_fields = None
        mock_config.base_url = "test"
        mock_config.api_key = "test"
        mock_adapter.config_class = mock_config

        del mock_config.model_fields

        metadata = AdapterMetadata(mock_adapter)

        assert metadata.supports_api_key is True
        assert metadata.supports_base_url is True


class TestAdapterManager:

    @pytest.fixture
    def adapter_manager(self):
        return AdapterManager()

    def test_adapter_manager_initialization(self, adapter_manager):
        assert len(adapter_manager._adapter_registry) == 8
        assert "openai" in adapter_manager._adapter_registry
        assert "claude" in adapter_manager._adapter_registry
        assert "gemini" in adapter_manager._adapter_registry
        assert "grok" in adapter_manager._adapter_registry
        assert "mistral" in adapter_manager._adapter_registry
        assert "deepseek" in adapter_manager._adapter_registry
        assert "ollama" in adapter_manager._adapter_registry
        assert "qwen" in adapter_manager._adapter_registry

    @async_tests
    async def test_get_adapter_unknown_type(self, adapter_manager):
        adapter = await adapter_manager.get_adapter("unknown")
        assert adapter is None

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    @async_tests
    async def test_get_adapter_openai(self, adapter_manager):
        adapter = await adapter_manager.get_adapter("openai", "test-api-key")
        assert adapter is not None
        assert isinstance(adapter, OpenAIAdapter)

    @patch.dict('os.environ', {'CLAUDE_API_KEY': 'test-key'})
    @async_tests
    async def test_get_adapter_claude(self, adapter_manager):
        adapter = await adapter_manager.get_adapter("claude", "test-api-key")
        assert adapter is not None
        assert isinstance(adapter, ClaudeAdapter)

    @async_tests
    async def test_get_adapter_caching(self, adapter_manager):
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            adapter1 = await adapter_manager.get_adapter("openai", "test-api-key")
            assert adapter1 is not None

            adapter2 = await adapter_manager.get_adapter("openai", "test-api-key")
            assert adapter2 is adapter1

            stats = adapter_manager.get_cache_stats()
            assert stats["cached_adapters"] == 1
            assert stats["total_usage"] == 2

    @async_tests
    async def test_get_adapter_different_configs(self, adapter_manager):
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            adapter1 = await adapter_manager.get_adapter("openai", "api-key-1")
            adapter2 = await adapter_manager.get_adapter("openai", "api-key-2")

            assert adapter1 is not adapter2
            assert adapter_manager.get_cache_stats()["cached_adapters"] == 2

    @async_tests
    async def test_adapter_context_manager(self, adapter_manager):
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            async with adapter_manager.adapter_context("openai", "test-api-key") as adapter:
                assert adapter is not None
                assert isinstance(adapter, OpenAIAdapter)

    @async_tests
    async def test_adapter_context_manager_unknown_type(self, adapter_manager):
        with pytest.raises(ValueError, match="Could not create adapter: unknown"):
            async with adapter_manager.adapter_context("unknown"):
                pass

    @async_tests
    async def test_cleanup_unused_adapters(self, adapter_manager):
        mock_adapter = AsyncMock()
        mock_key = AdapterKey("test", "hash", "url")

        adapter_manager._adapter_cache[mock_key] = mock_adapter
        adapter_manager._usage_count[mock_key] = 1

        await adapter_manager.cleanup_unused_adapters(min_usage=2)

        assert mock_key not in adapter_manager._adapter_cache
        assert mock_key not in adapter_manager._usage_count
        mock_adapter.close.assert_called_once()

    @async_tests
    async def test_cleanup_unused_adapters_no_close_method(self, adapter_manager):
        mock_adapter = MagicMock()
        del mock_adapter.close
        mock_key = AdapterKey("test", "hash", "url")

        adapter_manager._adapter_cache[mock_key] = mock_adapter
        adapter_manager._usage_count[mock_key] = 1

        await adapter_manager.cleanup_unused_adapters(min_usage=2)

        assert mock_key not in adapter_manager._adapter_cache

    @async_tests
    async def test_close_all_adapters(self, adapter_manager):
        mock_adapter1 = AsyncMock()
        mock_adapter2 = AsyncMock()

        adapter_manager._adapter_cache[AdapterKey("test1", "hash1", "url1")] = mock_adapter1
        adapter_manager._adapter_cache[AdapterKey("test2", "hash2", "url2")] = mock_adapter2
        adapter_manager._usage_count[AdapterKey("test1", "hash1", "url1")] = 1
        adapter_manager._usage_count[AdapterKey("test2", "hash2", "url2")] = 1

        with patch('promptdrifter.adapter_manager.get_http_client_manager') as mock_http_manager:
            mock_http_manager.return_value.close_all = AsyncMock()

            await adapter_manager.close_all_adapters()

            assert len(adapter_manager._adapter_cache) == 0
            assert len(adapter_manager._usage_count) == 0
            mock_adapter1.close.assert_called_once()
            mock_adapter2.close.assert_called_once()
            mock_http_manager.return_value.close_all.assert_called_once()

    @async_tests
    async def test_close_all_adapters_with_exceptions(self, adapter_manager):
        mock_adapter = AsyncMock()
        mock_adapter.close.side_effect = Exception("Close failed")

        adapter_manager._adapter_cache[AdapterKey("test", "hash", "url")] = mock_adapter

        with patch('promptdrifter.adapter_manager.get_http_client_manager') as mock_http_manager:
            mock_http_manager.return_value.close_all = AsyncMock()

            await adapter_manager.close_all_adapters()

            assert len(adapter_manager._adapter_cache) == 0

    def test_get_cache_stats(self, adapter_manager):
        key1 = AdapterKey("openai", "hash1", "url1")
        key2 = AdapterKey("claude", "hash2", "url2")
        key3 = AdapterKey("openai", "hash3", "url3")

        adapter_manager._adapter_cache[key1] = MagicMock()
        adapter_manager._adapter_cache[key2] = MagicMock()
        adapter_manager._adapter_cache[key3] = MagicMock()

        adapter_manager._usage_count[key1] = 5
        adapter_manager._usage_count[key2] = 3
        adapter_manager._usage_count[key3] = 2

        stats = adapter_manager.get_cache_stats()

        assert stats["cached_adapters"] == 3
        assert stats["total_usage"] == 10
        assert set(stats["adapter_types"]) == {"openai", "claude"}
        assert stats["usage_by_type"]["openai"] == 7
        assert stats["usage_by_type"]["claude"] == 3

    @async_tests
    async def test_create_adapter_instance_with_config(self, adapter_manager):
        metadata = AdapterMetadata(OpenAIAdapter)
        metadata.config_class = OpenAIAdapterConfig
        metadata.supports_api_key = True
        metadata.supports_base_url = True

        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            adapter = await adapter_manager._create_adapter_instance(
                metadata, "test-api-key", "https://api.openai.com"
            )

            assert adapter is not None
            assert isinstance(adapter, OpenAIAdapter)

    @async_tests
    async def test_create_adapter_instance_without_config(self, adapter_manager):
        mock_adapter_class = MagicMock()
        metadata = AdapterMetadata(mock_adapter_class)
        metadata.config_class = None

        adapter = await adapter_manager._create_adapter_instance(metadata, None, None)

        assert adapter is not None
        mock_adapter_class.assert_called_once_with()

    @async_tests
    async def test_get_adapter_creation_error(self, adapter_manager):
        with patch.object(adapter_manager, '_create_adapter_instance') as mock_create:
            mock_create.side_effect = Exception("Creation failed")

            with patch('builtins.print') as mock_print:
                adapter = await adapter_manager.get_adapter("openai", "test-key")

                assert adapter is None
                mock_print.assert_called_once()
                assert "Error creating adapter 'openai'" in mock_print.call_args[0][0]


class TestGlobalAdapterManager:

    def test_get_adapter_manager_singleton(self):
        manager1 = get_adapter_manager()
        manager2 = get_adapter_manager()

        assert manager1 is manager2

    def test_get_adapter_manager_type(self):
        manager = get_adapter_manager()
        assert isinstance(manager, AdapterManager)
