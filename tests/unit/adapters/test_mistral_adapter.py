import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import ValidationError

from promptdrifter.adapters.mistral import MistralAdapter, MistralAdapterConfig
from promptdrifter.config.adapter_settings import (
    API_KEY_ENV_VAR_MISTRAL,
    DEFAULT_MISTRAL_MODEL,
    MISTRAL_API_BASE_URL,
)


@pytest.fixture(autouse=True)
def auto_patch_httpx_client():
    mock_client = MagicMock(spec=httpx.AsyncClient)
    mock_client.post = AsyncMock()
    mock_client.aclose = AsyncMock()
    with patch("promptdrifter.adapters.mistral.get_shared_client", return_value=mock_client) as patched_client:
        yield patched_client

@pytest.fixture
def adapter_config_data():
    return {
        "api_key": "test-mistral-api-key",
        "base_url": MISTRAL_API_BASE_URL,
        "default_model": DEFAULT_MISTRAL_MODEL,
        "max_tokens": 1024,
    }

@pytest.fixture
def adapter(adapter_config_data, monkeypatch, auto_patch_httpx_client):
    monkeypatch.setenv(API_KEY_ENV_VAR_MISTRAL, adapter_config_data["api_key"])
    adapter_instance = MistralAdapter(
        api_key=adapter_config_data["api_key"],
        base_url=adapter_config_data["base_url"],
        default_model=adapter_config_data["default_model"],
        max_tokens=adapter_config_data["max_tokens"]
    )
    auto_patch_httpx_client.assert_called_once_with(
        base_url=adapter_config_data["base_url"],
        headers={
            "Authorization": f"Bearer {adapter_config_data['api_key']}",
            "Content-Type": "application/json",
        }
    )
    return adapter_instance

@pytest.fixture
def mock_http_response():
    response = MagicMock(spec=httpx.Response)
    response.status_code = 200
    response.json.return_value = {
        "choices": [
            {
                "message": {"content": "Test response"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"total_tokens": 10},
    }
    return response

@pytest.fixture
def mock_http_client(mock_http_response):
    with patch("promptdrifter.adapters.mistral.get_shared_client") as mock_client:
        mock_client.return_value.post = AsyncMock(return_value=mock_http_response)
        yield mock_client

def test_mistral_adapter_init_with_direct_params():
    config = MistralAdapterConfig(api_key="test_key")
    adapter = MistralAdapter(config=config)
    assert adapter.config.api_key == "test_key"
    assert adapter.config.base_url == MISTRAL_API_BASE_URL
    assert adapter.config.default_model == DEFAULT_MISTRAL_MODEL

def test_mistral_adapter_init_with_env_key_and_defaults():
    with patch.dict(os.environ, {API_KEY_ENV_VAR_MISTRAL: "env_key"}):
        config = MistralAdapterConfig()
        adapter = MistralAdapter(config=config)
        assert adapter.config.api_key == "env_key"
        assert adapter.config.base_url == MISTRAL_API_BASE_URL
        assert adapter.config.default_model == DEFAULT_MISTRAL_MODEL

def test_mistral_adapter_init_no_key_raises_error():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValidationError):
            config = MistralAdapterConfig()
            MistralAdapter(config=config)

def test_mistral_adapter_init_with_config():
    config = MistralAdapterConfig(
        api_key="test_key",
        base_url="http://test.com",
        default_model="test-model",
    )
    adapter = MistralAdapter(config=config)
    assert adapter.config.api_key == "test_key"
    assert adapter.config.base_url == "http://test.com"
    assert adapter.config.default_model == "test-model"

@pytest.mark.asyncio
async def test_mistral_adapter_execute_success(mock_http_client):
    config = MistralAdapterConfig(api_key="test_key")
    adapter = MistralAdapter(config=config)
    result = await adapter.execute("Test prompt")
    assert result.text_response == "Test response"
    assert result.finish_reason == "stop"
    assert result.model_name == DEFAULT_MISTRAL_MODEL
    assert result.usage == {"total_tokens": 10}
    assert result.error is None

@pytest.mark.asyncio
async def test_mistral_adapter_execute_with_all_params(mock_http_client):
    config = MistralAdapterConfig(api_key="test_key")
    adapter = MistralAdapter(config=config)
    config_override = MistralAdapterConfig(
        api_key="test_key",
        default_model="test-model",
        max_tokens=100,
        temperature=0.7,
        system_prompt="Test system prompt",
    )
    result = await adapter.execute("Test prompt", config_override=config_override)
    assert result.text_response == "Test response"
    mock_http_client.return_value.post.assert_called_once()
    call_args = mock_http_client.return_value.post.call_args[1]
    assert call_args["json"]["model"] == "test-model"
    assert call_args["json"]["temperature"] == 0.7
    assert call_args["json"]["max_tokens"] == 100
    assert len(call_args["json"]["messages"]) == 2
    assert call_args["json"]["messages"][0]["role"] == "system"
    assert call_args["json"]["messages"][0]["content"] == "Test system prompt"

@pytest.mark.asyncio
async def test_mistral_adapter_execute_http_status_error(mock_http_client):
    config = MistralAdapterConfig(api_key="test_key")
    adapter = MistralAdapter(config=config)
    mock_http_client.return_value.post.return_value.raise_for_status.side_effect = (
        httpx.HTTPStatusError(
            "400 Bad Request",
            request=MagicMock(),
            response=MagicMock(
                status_code=400,
                json=MagicMock(return_value={"error": {"message": "Invalid request"}}),
            ),
        )
    )
    result = await adapter.execute("Test prompt")
    assert result.error == "API Error (HTTP 400): Invalid request"
    assert result.text_response is None
    assert result.finish_reason == "error"

@pytest.mark.asyncio
async def test_mistral_adapter_execute_request_error(mock_http_client):
    config = MistralAdapterConfig(api_key="test_key")
    adapter = MistralAdapter(config=config)
    mock_http_client.return_value.post.side_effect = httpx.RequestError("Connection error")
    result = await adapter.execute("Test prompt")
    assert result.error == "HTTP Client Error: RequestError - Connection error"
    assert result.text_response is None
    assert result.finish_reason == "error"

@pytest.mark.asyncio
async def test_mistral_adapter_execute_prompt_blocked(mock_http_client):
    config = MistralAdapterConfig(api_key="test_key")
    adapter = MistralAdapter(config=config)
    mock_http_client.return_value.post.return_value.raise_for_status.side_effect = (
        httpx.HTTPStatusError(
            "400 Bad Request",
            request=MagicMock(),
            response=MagicMock(
                status_code=400,
                json=MagicMock(
                    return_value={
                        "error": {
                            "message": "The prompt was blocked by the content filter."
                        }
                    }
                ),
            ),
        )
    )
    result = await adapter.execute("Test prompt")
    assert result.error == "API Error (HTTP 400): The prompt was blocked by the content filter."
    assert result.text_response is None
    assert result.finish_reason == "error"

@pytest.mark.asyncio
async def test_close_client():
    config = MistralAdapterConfig(api_key="test_key")
    adapter = MistralAdapter(config=config)
    await adapter.close()
