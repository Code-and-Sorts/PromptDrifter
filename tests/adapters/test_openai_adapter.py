from unittest.mock import (
    AsyncMock,
    MagicMock,
)

import httpx
import pytest

from promptdrifter.adapters.openai import (
    API_KEY_ENV_VAR,
    DEFAULT_OPENAI_MODEL,
    OPENAI_API_BASE_URL,
    OpenAIAdapter,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_httpx_client(mocker):
    mock_client = MagicMock(spec=httpx.AsyncClient)
    mock_client.post = AsyncMock()
    mock_client.aclose = AsyncMock()
    mocker.patch("httpx.AsyncClient", return_value=mock_client)
    return mock_client


@pytest.fixture
def openai_adapter_with_key(monkeypatch, mock_httpx_client):
    monkeypatch.setenv(API_KEY_ENV_VAR, "test_api_key_from_env")
    adapter = OpenAIAdapter(api_key="test_api_key_direct")
    adapter.client = mock_httpx_client
    assert adapter.api_key == "test_api_key_direct"
    return adapter


@pytest.fixture
def openai_adapter_no_key(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)
    return OpenAIAdapter


async def test_openai_adapter_init_with_direct_key(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)
    adapter = OpenAIAdapter(api_key="direct_key")
    assert adapter.api_key == "direct_key"
    assert adapter.base_url == OPENAI_API_BASE_URL
    await adapter.close()


async def test_openai_adapter_init_with_env_key(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV_VAR, "env_key")
    adapter = OpenAIAdapter()
    assert adapter.api_key == "env_key"
    await adapter.close()


async def test_openai_adapter_init_no_key(openai_adapter_no_key):
    with pytest.raises(ValueError) as excinfo:
        openai_adapter_no_key()
    assert API_KEY_ENV_VAR in str(excinfo.value)


async def test_execute_successful(openai_adapter_with_key, mock_httpx_client):
    adapter = openai_adapter_with_key
    mock_response_data = {
        "choices": [
            {"message": {"role": "assistant", "content": "Test response from OpenAI"}}
        ],
        "usage": {"total_tokens": 10},
    }
    mock_httpx_client.post.return_value = httpx.Response(
        200, json=mock_response_data, request=httpx.Request("POST", "/chat/completions")
    )

    prompt = "Hello, OpenAI!"
    result = await adapter.execute(
        prompt, model="gpt-4", temperature=0.5, max_tokens=50
    )

    mock_httpx_client.post.assert_called_once()
    call_args = mock_httpx_client.post.call_args
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
    await adapter.close()


async def test_execute_uses_default_model(openai_adapter_with_key, mock_httpx_client):
    adapter = openai_adapter_with_key
    mock_httpx_client.post.return_value = httpx.Response(
        200,
        json={"choices": [{"message": {"content": "Default model response"}}]},
        request=httpx.Request("POST", "/"),
    )
    await adapter.execute("A prompt")
    payload = mock_httpx_client.post.call_args[1]["json"]
    assert payload["model"] == DEFAULT_OPENAI_MODEL
    await adapter.close()


async def test_execute_http_status_error(openai_adapter_with_key, mock_httpx_client):
    adapter = openai_adapter_with_key
    error_response_content = (
        '{"error": {"message": "Invalid API key", "type": "auth_error"}}'
    )
    mock_httpx_client.post.return_value = httpx.Response(
        401,
        content=error_response_content.encode("utf-8"),
        request=httpx.Request("POST", "/"),
    )

    result = await adapter.execute("A prompt")

    assert "error" in result
    assert "HTTP error 401" in result["error"]
    assert "Invalid API key" in result["error"]
    assert result["raw_response"] is None
    assert result["text_response"] is None
    await adapter.close()


async def test_execute_request_error(openai_adapter_with_key, mock_httpx_client):
    adapter = openai_adapter_with_key
    mock_httpx_client.post.side_effect = httpx.ConnectError("Connection failed")

    result = await adapter.execute("A prompt")

    assert "error" in result
    assert "Request error connecting to OpenAI" in result["error"]
    assert "Connection failed" in result["error"]
    await adapter.close()


async def test_execute_unexpected_error(openai_adapter_with_key, mock_httpx_client):
    adapter = openai_adapter_with_key
    mock_httpx_client.post.side_effect = Exception("Something totally unexpected")

    result = await adapter.execute("A prompt")

    assert "error" in result
    assert "An unexpected error occurred" in result["error"]
    assert "Something totally unexpected" in result["error"]
    await adapter.close()


async def test_adapter_close_called(openai_adapter_with_key, mock_httpx_client):
    adapter = openai_adapter_with_key
    await adapter.close()
    mock_httpx_client.aclose.assert_called_once()
