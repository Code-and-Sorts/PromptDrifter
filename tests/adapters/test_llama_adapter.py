import os
from unittest.mock import (
    AsyncMock,
    MagicMock,
    patch,
)

import httpx
import pytest

from promptdrifter.adapters.llama import LlamaAdapter

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
        "promptdrifter.adapters.llama.httpx.AsyncClient",
        return_value=mock_httpx_client,
    ) as patched_class_mock:
        yield patched_class_mock


@pytest.fixture
def adapter(patch_httpx_client):
    """Provides a LlamaAdapter instance with a dummy API key."""
    with patch.dict(os.environ, {"META_LLAMA_API_KEY": "test-api-key-env"}):
        adapter_instance = LlamaAdapter()
        return adapter_instance


async def test_llama_adapter_init_with_direct_key(monkeypatch, patch_httpx_client):
    monkeypatch.delenv("META_LLAMA_API_KEY", raising=False)
    adapter_instance = LlamaAdapter(api_key="direct_key")
    assert adapter_instance.api_key == "direct_key"
    assert adapter_instance.API_ENDPOINT == "https://llama-api.meta.ai/v1/chat/completions"
    assert adapter_instance.DEFAULT_MODEL == "llama-3-70b-instruct"
    assert adapter_instance.headers == {
        "Authorization": "Bearer direct_key",
        "Content-Type": "application/json",
    }


async def test_llama_adapter_init_with_env_key(monkeypatch, patch_httpx_client):
    monkeypatch.setenv("META_LLAMA_API_KEY", "env_key")
    adapter_instance = LlamaAdapter()
    assert adapter_instance.api_key == "env_key"
    assert adapter_instance.headers == {
        "Authorization": "Bearer env_key",
        "Content-Type": "application/json",
    }


async def test_llama_adapter_init_no_key_raises_error(monkeypatch):
    monkeypatch.delenv("META_LLAMA_API_KEY", raising=False)
    with pytest.raises(ValueError) as excinfo:
        LlamaAdapter()
    assert "META_LLAMA_API_KEY" in str(excinfo.value)


async def test_execute_successful(adapter, patch_httpx_client):
    mock_client_instance = patch_httpx_client.return_value

    mock_response_data = {
        "choices": [
            {
                "message": {"content": "Test response from Llama"},
                "finish_reason": "stop"
            }
        ],
        "usage": {"total_tokens": 15},
    }
    mock_client_instance.post.return_value = httpx.Response(
        200, json=mock_response_data, request=httpx.Request("POST", adapter.API_ENDPOINT)
    )

    prompt = "Hello, Llama!"
    result = await adapter.execute(
        prompt, model="llama-3-8b-instruct", temperature=0.7, max_tokens=100, system_prompt="You are a helpful assistant."
    )

    mock_client_instance.post.assert_called_once()
    call_args = mock_client_instance.post.call_args
    assert call_args[0][0] == adapter.API_ENDPOINT
    payload = call_args[1]["json"]
    assert payload["model"] == "llama-3-8b-instruct"
    assert payload["messages"] == [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    assert payload["temperature"] == 0.7
    assert payload["max_tokens"] == 100

    assert result["text_response"] == "Test response from Llama"
    assert result["raw_response"] == mock_response_data
    assert result["model_name"] == "llama-3-8b-instruct"
    assert result["finish_reason"] == "stop"
    assert not result.get("error")
    assert result["usage"] == mock_response_data["usage"]

    mock_client_instance.aclose.assert_called_once()


async def test_execute_uses_default_model(adapter, patch_httpx_client):
    mock_client_instance = patch_httpx_client.return_value
    mock_client_instance.post.return_value = httpx.Response(
        200,
        json={"choices": [{"message": {"content": "Default model response"}, "finish_reason": "stop"}]},
        request=httpx.Request("POST", adapter.API_ENDPOINT),
    )
    await adapter.execute("A prompt")
    payload = mock_client_instance.post.call_args[1]["json"]
    assert payload["model"] == adapter.DEFAULT_MODEL
    mock_client_instance.aclose.assert_called_once()


async def test_execute_no_system_prompt(adapter, patch_httpx_client):
    mock_client_instance = patch_httpx_client.return_value
    mock_client_instance.post.return_value = httpx.Response(
        200,
        json={"choices": [{"message": {"content": "Response without system prompt"}, "finish_reason": "stop"}]},
        request=httpx.Request("POST", adapter.API_ENDPOINT),
    )
    await adapter.execute("A prompt without system prompt")
    payload = mock_client_instance.post.call_args[1]["json"]
    assert payload["messages"] == [{"role": "user", "content": "A prompt without system prompt"}]
    mock_client_instance.aclose.assert_called_once()


async def test_execute_http_status_error(adapter, patch_httpx_client):
    mock_client_instance = patch_httpx_client.return_value
    error_response_content = (
        '{"error": {"message": "Invalid API key", "type": "auth_error"}}'
    )
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 401
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "401 Unauthorized", request=MagicMock(), response=mock_response
    )
    mock_response.text = error_response_content
    mock_response.json.return_value = {"error": {"message": "Invalid API key", "type": "auth_error"}}

    mock_client_instance.post.return_value = mock_response

    result = await adapter.execute("A prompt")

    assert "error" in result
    assert "API Error (HTTP 401)" in result["error"]
    assert "Invalid API key" in result["error"]
    assert result["text_response"] is None
    assert "finish_reason" in result
    assert result["finish_reason"] == "error"
    mock_client_instance.aclose.assert_called_once()


async def test_execute_request_error(adapter, patch_httpx_client):
    mock_client_instance = patch_httpx_client.return_value
    mock_client_instance.post.side_effect = httpx.RequestError("Connection failed")

    result = await adapter.execute("A prompt")

    assert "error" in result
    assert "HTTP Client Error" in result["error"]
    assert "Connection failed" in result["error"]
    assert result["finish_reason"] == "error"
    mock_client_instance.aclose.assert_called_once()


async def test_execute_timeout_error(adapter, patch_httpx_client):
    mock_client_instance = patch_httpx_client.return_value
    mock_client_instance.post.side_effect = httpx.ReadTimeout("Request timed out")

    result = await adapter.execute("A prompt")

    assert "error" in result
    assert "Request timed out" in result["error"]
    assert result["finish_reason"] == "error"
    mock_client_instance.aclose.assert_called_once()


async def test_execute_unexpected_error(adapter, patch_httpx_client):
    mock_client_instance = patch_httpx_client.return_value
    mock_client_instance.post.side_effect = Exception("Something totally unexpected")

    result = await adapter.execute("A prompt")

    assert "error" in result
    assert "An unexpected error occurred" in result["error"]
    assert "Something totally unexpected" in result["error"]
    assert result["finish_reason"] == "error"
    mock_client_instance.aclose.assert_called_once()


async def test_execute_unexpected_response_structure(adapter, patch_httpx_client):
    mock_client_instance = patch_httpx_client.return_value

    async def modified_post(*args, **kwargs):
        response = httpx.Response(
            200,
            json={"unexpected_structure": True},
            request=httpx.Request("POST", adapter.API_ENDPOINT)
        )
        return response

    mock_client_instance.post = AsyncMock(side_effect=modified_post)

    with patch("promptdrifter.adapters.llama.console.print_exception"):
        result = await adapter.execute("A prompt")

    assert "error" in result
    assert "Unexpected response structure" in result["error"]
    assert result["text_response"] is None

    assert "Unexpected response structure" in result["error"]

    mock_client_instance.aclose.assert_called_once()


async def test_execute_empty_response_content(adapter, patch_httpx_client):
    mock_client_instance = patch_httpx_client.return_value
    mock_client_instance.post.return_value = httpx.Response(
        200,
        json={"choices": [{"message": {}}]},
        request=httpx.Request("POST", adapter.API_ENDPOINT),
    )

    result = await adapter.execute("A prompt")

    assert "error" in result
    assert "No text content found in successful response" in result["error"]
    assert result["text_response"] is None
    mock_client_instance.aclose.assert_called_once()


async def test_execute_no_api_key(monkeypatch):
    monkeypatch.setenv("META_LLAMA_API_KEY", "dummy_key")
    adapter_instance = LlamaAdapter()
    adapter_instance.api_key = None

    result = await adapter_instance.execute("A prompt")

    assert "error" in result
    assert "Meta Llama API key not configured" in result["error"]
    assert result["text_response"] is None
    assert result["finish_reason"] == "error"


async def test_execute_with_system_prompt(adapter, patch_httpx_client):
    mock_client_instance = patch_httpx_client.return_value
    mock_client_instance.post.return_value = httpx.Response(
        200,
        json={"choices": [{"message": {"content": "Response with system prompt"}, "finish_reason": "stop"}]},
        request=httpx.Request("POST", adapter.API_ENDPOINT),
    )

    system_prompt = "You are a helpful AI assistant that provides concise answers."
    await adapter.execute("A prompt with system prompt", system_prompt=system_prompt)

    payload = mock_client_instance.post.call_args[1]["json"]
    assert len(payload["messages"]) == 2
    assert payload["messages"][0] == {"role": "system", "content": system_prompt}
    assert payload["messages"][1] == {"role": "user", "content": "A prompt with system prompt"}
    mock_client_instance.aclose.assert_called_once()
