from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from promptdrifter.adapters.deepseek import DeepSeekAdapter

DEEPSEEK_API_KEY_ENV_VAR = "DEEPSEEK_API_KEY"


@pytest.fixture
def mock_response():
    """Creates a mock httpx.Response."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = 200
    response.json = MagicMock(return_value={})
    response.text = "Response text"
    response.raise_for_status = MagicMock()
    return response


@pytest.fixture
def patch_httpx_async_client(mock_response):
    """Patch httpx.AsyncClient to return a mock response directly from post."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = MagicMock()

        mock_client.post = AsyncMock(return_value=mock_response)

        mock_client.aclose = AsyncMock()

        mock_client_class.return_value = mock_client

        yield mock_client


@pytest.fixture
def mock_response_content():
    """Common response content fixture that can be configured by tests."""
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1699799906,
        "model": "deepseek-chat",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello from DeepSeek!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


def test_deepseek_adapter_init_with_direct_key(monkeypatch):
    monkeypatch.delenv(DEEPSEEK_API_KEY_ENV_VAR, raising=False)
    adapter = DeepSeekAdapter(api_key="direct_deepseek_key")
    assert adapter.api_key == "direct_deepseek_key"
    assert adapter.headers["Authorization"] == "Bearer direct_deepseek_key"


def test_deepseek_adapter_init_with_env_key(monkeypatch):
    monkeypatch.setenv(DEEPSEEK_API_KEY_ENV_VAR, "env_deepseek_key")
    adapter = DeepSeekAdapter()
    assert adapter.api_key == "env_deepseek_key"
    assert adapter.headers["Authorization"] == "Bearer env_deepseek_key"


def test_deepseek_adapter_init_no_key_raises_error(monkeypatch):
    monkeypatch.delenv(DEEPSEEK_API_KEY_ENV_VAR, raising=False)
    with pytest.raises(ValueError) as excinfo:
        DeepSeekAdapter()
    assert DEEPSEEK_API_KEY_ENV_VAR in str(excinfo.value)


@pytest.fixture
def deepseek_adapter_env_key(monkeypatch):
    """Provides a DeepSeekAdapter instance with API key from env var."""
    monkeypatch.setenv(DEEPSEEK_API_KEY_ENV_VAR, "test-deepseek-key")
    return DeepSeekAdapter()


@pytest.mark.asyncio
async def test_execute_successful(
    deepseek_adapter_env_key,
    patch_httpx_async_client,
    mock_response,
    mock_response_content,
):
    mock_response.json.return_value = mock_response_content

    prompt = "Hi DeepSeek"
    result = await deepseek_adapter_env_key.execute(
        prompt, model="deepseek-chat", temperature=0.7, max_tokens=100
    )

    patch_httpx_async_client.post.assert_called_once_with(
        DeepSeekAdapter.API_ENDPOINT,
        headers=deepseek_adapter_env_key.headers,
        json={
            "model": "deepseek-chat",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
        },
        timeout=60.0,
    )

    assert result["text_response"] == "Hello from DeepSeek!"
    assert result["raw_response"] == mock_response_content
    assert result["model_name"] == "deepseek-chat"
    assert result["finish_reason"] == "stop"
    assert result["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
    }
    assert result["error"] is None


@pytest.mark.asyncio
async def test_execute_uses_default_model(
    deepseek_adapter_env_key, patch_httpx_async_client, mock_response
):
    mock_response.json.return_value = {
        "choices": [
            {"message": {"content": "Default model response"}, "finish_reason": "max_tokens"}
        ]
    }

    await deepseek_adapter_env_key.execute("A prompt")

    payload = patch_httpx_async_client.post.call_args[1]["json"]
    assert payload["model"] == DeepSeekAdapter.DEFAULT_MODEL
    assert payload["max_tokens"] == DeepSeekAdapter.DEFAULT_MAX_TOKENS


@pytest.mark.asyncio
async def test_execute_with_system_prompt(
    deepseek_adapter_env_key, patch_httpx_async_client, mock_response
):
    mock_response.json.return_value = {
        "choices": [
            {"message": {"content": "Sys prompt response"}, "finish_reason": "stop"}
        ]
    }

    system_prompt_text = "You are a helpful assistant."
    await deepseek_adapter_env_key.execute("User prompt", system_prompt=system_prompt_text)

    payload = patch_httpx_async_client.post.call_args[1]["json"]
    messages = payload["messages"]
    assert len(messages) == 2
    assert messages[0] == {"role": "system", "content": system_prompt_text}
    assert messages[1] == {"role": "user", "content": "User prompt"}


@pytest.mark.asyncio
async def test_execute_no_text_in_response_content(
    deepseek_adapter_env_key, patch_httpx_async_client, mock_response
):
    mock_response.json.return_value = {
        "choices": [{"message": {}, "finish_reason": "stop"}]
    }

    result = await deepseek_adapter_env_key.execute("A prompt")
    assert result["text_response"] is None
    assert result["error"] == "No text content found in successful response."
    assert result["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_execute_empty_response_content(
    deepseek_adapter_env_key, patch_httpx_async_client, mock_response
):
    mock_response.json.return_value = {"choices": []}

    result = await deepseek_adapter_env_key.execute("A prompt")
    assert result["text_response"] is None
    assert result["error"] == "Unexpected response structure for 200 OK."


@pytest.mark.asyncio
async def test_execute_malformed_success_response(
    deepseek_adapter_env_key, patch_httpx_async_client, mock_response
):
    mock_response.json.return_value = {"unexpected_field": "no_choices_here"}

    result = await deepseek_adapter_env_key.execute("A prompt")
    assert result["text_response"] is None
    assert result["error"] == "Unexpected response structure for 200 OK."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status_code, error_msg, expected_error_detail",
    [
        (400, "Invalid request parameters", "Invalid request parameters"),
        (401, "Invalid API key", "Invalid API key"),
        (429, "Rate limit exceeded", "Rate limit exceeded"),
        (500, "Internal server error", "Internal server error"),
    ],
)
async def test_execute_http_status_error_json_body(
    deepseek_adapter_env_key,
    patch_httpx_async_client,
    mock_response,
    status_code,
    error_msg,
    expected_error_detail,
):
    error_json = {"error": {"message": error_msg}}
    mock_response.status_code = status_code
    mock_response.json.return_value = error_json

    http_error = httpx.HTTPStatusError(
        message=f"HTTP Error {status_code}",
        request=MagicMock(),
        response=mock_response
    )

    mock_response.raise_for_status.side_effect = http_error
    patch_httpx_async_client.post.return_value = mock_response

    result = await deepseek_adapter_env_key.execute("A prompt")

    assert result["error"] is not None
    assert f"API Error (HTTP {status_code})" in result["error"]
    assert expected_error_detail in result["error"]
    assert result["text_response"] is None


@pytest.mark.asyncio
async def test_execute_httpx_request_error(
    deepseek_adapter_env_key, patch_httpx_async_client
):
    error = httpx.RequestError("Connection refused", request=MagicMock())
    patch_httpx_async_client.post.side_effect = error

    result = await deepseek_adapter_env_key.execute("A prompt")

    assert result["error"] is not None
    assert "HTTP Client Error" in result["error"]
    assert "Connection refused" in result["error"]
    assert result["text_response"] is None
    assert result["finish_reason"] == "error"


@pytest.mark.asyncio
async def test_execute_timeout_error(
    deepseek_adapter_env_key, patch_httpx_async_client
):
    patch_httpx_async_client.post.side_effect = httpx.ReadTimeout("Request timed out")

    result = await deepseek_adapter_env_key.execute("A prompt")

    assert result["error"] is not None
    assert "Request timed out" in result["error"]
    assert result["text_response"] is None
    assert result["finish_reason"] == "error"


@pytest.mark.asyncio
async def test_execute_unexpected_exception(
    deepseek_adapter_env_key, patch_httpx_async_client
):
    error_msg = "Something unexpected happened"
    patch_httpx_async_client.post.side_effect = Exception(error_msg)

    result = await deepseek_adapter_env_key.execute("A prompt")

    assert result["error"] is not None
    assert "An unexpected error occurred" in result["error"]
    assert error_msg in result["error"]
    assert result["text_response"] is None
    assert result["finish_reason"] == "error"


@pytest.mark.asyncio
async def test_api_key_missing_during_execute():
    adapter = DeepSeekAdapter(api_key="temp_key")
    adapter.api_key = None

    result = await adapter.execute("A prompt")
    assert result["error"] == "DeepSeek API key not configured."
    assert result["text_response"] is None
    assert result["finish_reason"] == "error"
