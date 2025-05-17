import os
from unittest.mock import (
    AsyncMock,
    MagicMock,
    patch,
)

import httpx
import pytest

from promptdrifter.adapters.openai import OpenAIAdapter, OpenAIAdapterConfig
from promptdrifter.config.adapter_settings import (
    API_KEY_ENV_VAR_OPENAI as config_API_KEY_ENV_VAR_OPENAI,
)
from promptdrifter.config.adapter_settings import (
    DEFAULT_OPENAI_MODEL as config_DEFAULT_OPENAI_MODEL,
)
from promptdrifter.config.adapter_settings import (
    OPENAI_API_BASE_URL as config_OPENAI_API_BASE_URL,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_response():
    """Creates a mock httpx.Response."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = 200
    response.raise_for_status = MagicMock()
    response.json = MagicMock()
    return response


@pytest.fixture
def mock_httpx_client():
    """Creates a mock httpx.AsyncClient."""
    client = MagicMock(spec=httpx.AsyncClient)
    client.post = AsyncMock()
    client.aclose = AsyncMock()
    return client


@pytest.fixture(autouse=True)
def patch_httpx_client(mock_httpx_client):
    """Patches httpx.AsyncClient to return our mock client."""
    with patch(
        "promptdrifter.adapters.openai.httpx.AsyncClient",
        return_value=mock_httpx_client,
    ) as patched_class_mock:
        yield patched_class_mock


@pytest.fixture
def adapter(patch_httpx_client):
    """Provides an OpenAIAdapter instance with a dummy API key handled by Pydantic."""
    with patch.dict(os.environ, {config_API_KEY_ENV_VAR_OPENAI: "test-api-key-env"}):
        adapter_instance = OpenAIAdapter()
        patch_httpx_client.assert_called_once_with(
            base_url=adapter_instance.config.base_url,
            headers={"Authorization": f"Bearer {adapter_instance.config.api_key}"},
        )
        return adapter_instance


async def test_openai_adapter_init_with_direct_key(monkeypatch, patch_httpx_client):
    monkeypatch.delenv(config_API_KEY_ENV_VAR_OPENAI, raising=False)
    adapter_instance = OpenAIAdapter(api_key="direct_key", base_url="custom_url")
    assert adapter_instance.config.api_key == "direct_key"
    assert adapter_instance.config.base_url == "custom_url"
    assert adapter_instance.config.default_model == config_DEFAULT_OPENAI_MODEL
    patch_httpx_client.assert_called_once_with(
        base_url="custom_url", headers={"Authorization": "Bearer direct_key"}
    )


async def test_openai_adapter_init_with_env_key(monkeypatch, patch_httpx_client):
    monkeypatch.setenv(config_API_KEY_ENV_VAR_OPENAI, "env_key")
    adapter_instance = OpenAIAdapter()
    assert adapter_instance.config.api_key == "env_key"
    assert adapter_instance.config.base_url == config_OPENAI_API_BASE_URL
    patch_httpx_client.assert_called_once_with(
        base_url=config_OPENAI_API_BASE_URL, headers={"Authorization": "Bearer env_key"}
    )


async def test_openai_adapter_init_no_key_raises_error(monkeypatch):
    monkeypatch.delenv(config_API_KEY_ENV_VAR_OPENAI, raising=False)
    with pytest.raises(ValueError) as excinfo:
        OpenAIAdapter()
    assert config_API_KEY_ENV_VAR_OPENAI in str(excinfo.value)


async def test_openai_adapter_init_with_config_object(monkeypatch, patch_httpx_client):
    monkeypatch.delenv(config_API_KEY_ENV_VAR_OPENAI, raising=False)
    config = OpenAIAdapterConfig(
        api_key="config_key", base_url="config_url", default_model="config_model"
    )
    adapter_instance = OpenAIAdapter(config=config)
    assert adapter_instance.config is config
    assert adapter_instance.config.api_key == "config_key"
    assert adapter_instance.config.base_url == "config_url"
    assert adapter_instance.config.default_model == "config_model"
    patch_httpx_client.assert_called_once_with(
        base_url="config_url", headers={"Authorization": "Bearer config_key"}
    )


async def test_execute_successful(patch_httpx_client):
    with patch.dict(
        os.environ, {config_API_KEY_ENV_VAR_OPENAI: "test_api_key_execute"}
    ):
        adapter_instance = OpenAIAdapter(base_url="execute_custom_url")

    mock_client_instance = patch_httpx_client.return_value

    mock_response_data = {
        "choices": [
            {"message": {"role": "assistant", "content": "Test response from OpenAI"}}
        ],
        "usage": {"total_tokens": 10},
    }
    mock_client_instance.post.return_value = httpx.Response(
        200, json=mock_response_data, request=httpx.Request("POST", "/chat/completions")
    )

    prompt = "Hello, OpenAI!"
    result = await adapter_instance.execute(
        prompt, model="gpt-4", temperature=0.5, max_tokens=50
    )

    mock_client_instance.post.assert_called_once()
    call_args = mock_client_instance.post.call_args
    assert call_args[0][0] == "/chat/completions"
    payload = call_args[1]["json"]
    assert payload["model"] == "gpt-4"
    assert payload["messages"] == [{"role": "user", "content": prompt}]
    assert payload["temperature"] == 0.5
    assert payload["max_tokens"] == 50

    assert result["text_response"] == "Test response from OpenAI"
    assert result["raw_response"] == mock_response_data
    assert result["model_used"] == "gpt-4"
    assert "error" not in result
    await adapter_instance.close()
    mock_client_instance.aclose.assert_called_once()


async def test_execute_uses_default_model(adapter, patch_httpx_client):
    mock_client_instance = patch_httpx_client.return_value
    mock_client_instance.post.return_value = httpx.Response(
        200,
        json={"choices": [{"message": {"content": "Default model response"}}]},
        request=httpx.Request("POST", "/"),
    )
    await adapter.execute("A prompt")
    payload = mock_client_instance.post.call_args[1]["json"]
    assert payload["model"] == adapter.config.default_model
    await adapter.close()
    mock_client_instance.aclose.assert_called_once()


async def test_execute_http_status_error(adapter, patch_httpx_client):
    mock_client_instance = patch_httpx_client.return_value
    error_response_content = (
        '{"error": {"message": "Invalid API key", "type": "auth_error"}}'
    )
    error_response = httpx.Response(
        401,
        content=error_response_content.encode("utf-8"),
        request=httpx.Request("POST", "/"),
    )
    mock_client_instance.post.return_value = error_response

    result = await adapter.execute("A prompt")

    assert "error" in result
    assert "HTTP error 401" in result["error"]
    assert "Invalid API key" in result["error"]
    assert result["raw_response"] is None
    assert result["text_response"] is None
    await adapter.close()
    mock_client_instance.aclose.assert_called_once()


async def test_execute_request_error(adapter, patch_httpx_client):
    mock_client_instance = patch_httpx_client.return_value
    mock_client_instance.post.side_effect = httpx.ConnectError("Connection failed")

    result = await adapter.execute("A prompt")

    assert "error" in result
    assert "Request error connecting to OpenAI" in result["error"]
    assert "Connection failed" in result["error"]
    await adapter.close()
    mock_client_instance.aclose.assert_called_once()


async def test_execute_unexpected_error(adapter, patch_httpx_client):
    mock_client_instance = patch_httpx_client.return_value
    mock_client_instance.post.side_effect = Exception("Something totally unexpected")

    result = await adapter.execute("A prompt")

    assert "error" in result
    assert "An unexpected error occurred" in result["error"]
    assert "Something totally unexpected" in result["error"]
    await adapter.close()
    mock_client_instance.aclose.assert_called_once()


async def test_adapter_close_called(adapter, patch_httpx_client):
    mock_client_instance = patch_httpx_client.return_value
    await adapter.close()
    mock_client_instance.aclose.assert_called_once()
