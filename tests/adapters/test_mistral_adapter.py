from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from promptdrifter.adapters.mistral import MistralAdapter, MistralAdapterConfig
from promptdrifter.config.adapter_settings import (
    API_KEY_ENV_VAR_MISTRAL,
    DEFAULT_MISTRAL_MODEL,
)


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
def mock_httpx_client(mock_response):
    """Patch httpx.AsyncClient for the adapter."""
    client_mock = MagicMock()
    client_mock.post = AsyncMock(return_value=mock_response)
    client_mock.aclose = AsyncMock()
    return client_mock


@pytest.fixture
def mock_response_content():
    """Common response content fixture that can be configured by tests."""
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "mistral-large-latest",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello from Mistral!",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    }


def test_mistral_adapter_init_with_direct_key(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV_VAR_MISTRAL, raising=False)
    adapter = MistralAdapter(api_key="direct_mistral_key")
    assert adapter.config.api_key == "direct_mistral_key"
    assert adapter.headers["Authorization"] == "Bearer direct_mistral_key"


def test_mistral_adapter_init_with_env_key(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV_VAR_MISTRAL, "env_mistral_key")
    adapter = MistralAdapter()
    assert adapter.config.api_key == "env_mistral_key"
    assert adapter.headers["Authorization"] == "Bearer env_mistral_key"


def test_mistral_adapter_init_no_key_raises_error(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV_VAR_MISTRAL, raising=False)
    with pytest.raises(ValueError) as excinfo:
        MistralAdapter()
    assert API_KEY_ENV_VAR_MISTRAL in str(excinfo.value)


def test_mistral_adapter_init_with_config(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV_VAR_MISTRAL, raising=False)
    config = MistralAdapterConfig(api_key="config_mistral_key")
    adapter = MistralAdapter(config=config)
    assert adapter.config.api_key == "config_mistral_key"
    assert adapter.headers["Authorization"] == "Bearer config_mistral_key"


@pytest.fixture
def mistral_adapter_with_mocked_client(monkeypatch, mock_httpx_client):
    """Provides a MistralAdapter instance with mocked client."""
    monkeypatch.setenv(API_KEY_ENV_VAR_MISTRAL, "test-mistral-key")
    adapter = MistralAdapter()
    # Replace the real client with our mock
    adapter.client = mock_httpx_client
    return adapter


@pytest.mark.asyncio
async def test_execute_successful(
    mistral_adapter_with_mocked_client,
    mock_httpx_client,
    mock_response,
    mock_response_content,
):
    mock_response.json.return_value = mock_response_content

    prompt = "Hi Mistral"
    result = await mistral_adapter_with_mocked_client.execute(
        prompt, model="mistral-large-latest", temperature=0.7, max_tokens=100
    )

    mock_httpx_client.post.assert_called_once_with(
        "/chat/completions",
        json={
            "model": "mistral-large-latest",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 100,
        },
        timeout=60.0,
    )

    assert result["text_response"] == "Hello from Mistral!"
    assert result["raw_response"] == mock_response_content
    assert result["model_name"] == "mistral-large-latest"
    assert result["finish_reason"] == "stop"
    assert result["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30
    }
    assert result["error"] is None


@pytest.mark.asyncio
async def test_execute_uses_default_model(
    mistral_adapter_with_mocked_client, mock_httpx_client, mock_response, mock_response_content
):
    mock_response.json.return_value = mock_response_content
    await mistral_adapter_with_mocked_client.execute("A prompt")

    payload = mock_httpx_client.post.call_args[1]["json"]
    assert payload["model"] == DEFAULT_MISTRAL_MODEL


@pytest.mark.asyncio
async def test_execute_with_system_prompt(
    mistral_adapter_with_mocked_client, mock_httpx_client, mock_response, mock_response_content
):
    mock_response.json.return_value = mock_response_content

    system_prompt_text = "You are a helpful assistant."
    await mistral_adapter_with_mocked_client.execute(
        "User prompt", system_prompt=system_prompt_text
    )

    payload = mock_httpx_client.post.call_args[1]["json"]
    messages = payload["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == system_prompt_text
    assert messages[1]["role"] == "user"


@pytest.mark.asyncio
async def test_execute_with_custom_messages(
    mistral_adapter_with_mocked_client, mock_httpx_client, mock_response, mock_response_content
):
    mock_response.json.return_value = mock_response_content

    custom_messages = [
        {"role": "system", "content": "You are an AI assistant"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "user", "content": "How are you?"}
    ]

    await mistral_adapter_with_mocked_client.execute(
        "This prompt should be ignored", messages=custom_messages
    )

    payload = mock_httpx_client.post.call_args[1]["json"]
    assert payload["messages"] == custom_messages


@pytest.mark.asyncio
async def test_execute_no_choices_in_response(
    mistral_adapter_with_mocked_client, mock_httpx_client, mock_response
):
    mock_response.json.return_value = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "mistral-large-latest",
        "choices": []
    }

    result = await mistral_adapter_with_mocked_client.execute("A prompt")
    assert result["text_response"] is None
    assert result["error"] == "Unexpected response structure for 200 OK."


@pytest.mark.asyncio
async def test_execute_no_message_in_choice(
    mistral_adapter_with_mocked_client, mock_httpx_client, mock_response
):
    mock_response.json.return_value = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "mistral-large-latest",
        "choices": [{"index": 0, "finish_reason": "stop"}]
    }

    result = await mistral_adapter_with_mocked_client.execute("A prompt")
    assert result["text_response"] is None
    assert result["error"] == "No text content found in successful response."


@pytest.mark.asyncio
async def test_execute_malformed_success_response(
    mistral_adapter_with_mocked_client, mock_httpx_client, mock_response
):
    mock_response.json.return_value = {"unexpected_field": "no_content_here"}

    result = await mistral_adapter_with_mocked_client.execute("A prompt")
    assert result["text_response"] is None
    assert "Unexpected response structure" in result["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status_code, error_message, expected_error_detail",
    [
        (400, "Invalid request parameters", "Invalid request parameters"),
        (401, "Invalid API Key", "Invalid API Key"),
        (403, "You don't have access to this model", "You don't have access to this model"),
        (429, "Rate limit exceeded", "Rate limit exceeded"),
        (500, "Internal server error", "Internal server error"),
        (503, "Service unavailable", "Service unavailable"),
    ],
)
async def test_execute_http_status_error_json_body(
    mistral_adapter_with_mocked_client,
    mock_httpx_client,
    mock_response,
    status_code,
    error_message,
    expected_error_detail,
):
    error_json = {"error": {"message": error_message}}
    mock_response.status_code = status_code
    mock_response.json.return_value = error_json

    http_error = httpx.HTTPStatusError(
        message=f"HTTP Error {status_code}",
        request=MagicMock(),
        response=mock_response
    )

    mock_response.raise_for_status.side_effect = http_error
    mock_httpx_client.post.return_value = mock_response

    result = await mistral_adapter_with_mocked_client.execute("A prompt")

    assert result["error"] is not None
    assert f"API Error (HTTP {status_code})" in result["error"]
    assert expected_error_detail in result["error"]
    assert result["text_response"] is None


@pytest.mark.asyncio
async def test_execute_http_status_error_non_json_body(
    mistral_adapter_with_mocked_client, mock_httpx_client, mock_response
):
    status_code = 502
    error_text = "Bad Gateway"
    mock_response.status_code = status_code
    mock_response.json.side_effect = ValueError("Invalid JSON")
    mock_response.text = error_text

    http_error = httpx.HTTPStatusError(
        message=f"HTTP Error {status_code}",
        request=MagicMock(),
        response=mock_response
    )

    mock_response.raise_for_status.side_effect = http_error
    mock_httpx_client.post.return_value = mock_response

    result = await mistral_adapter_with_mocked_client.execute("A prompt")

    assert result["error"] is not None
    assert f"API Error (HTTP {status_code})" in result["error"]
    assert result["text_response"] is None


@pytest.mark.asyncio
async def test_execute_request_error(
    mistral_adapter_with_mocked_client, mock_httpx_client
):
    request_error = httpx.RequestError("Connection error", request=MagicMock())
    mock_httpx_client.post.side_effect = request_error

    result = await mistral_adapter_with_mocked_client.execute("A prompt")

    assert result["error"] is not None
    assert "HTTP Client Error" in result["error"]
    assert result["text_response"] is None
    assert result["finish_reason"] == "error"


@pytest.mark.asyncio
async def test_execute_timeout_error(
    mistral_adapter_with_mocked_client, mock_httpx_client
):
    timeout_error = httpx.ReadTimeout("Request timed out", request=MagicMock())
    mock_httpx_client.post.side_effect = timeout_error

    result = await mistral_adapter_with_mocked_client.execute("A prompt")

    assert result["error"] is not None
    assert "HTTP Client Error" in result["error"]
    assert result["text_response"] is None
    assert result["finish_reason"] == "error"


@pytest.mark.asyncio
async def test_execute_unexpected_exception(
    mistral_adapter_with_mocked_client, mock_httpx_client
):
    mock_httpx_client.post.side_effect = Exception("Unexpected error")

    result = await mistral_adapter_with_mocked_client.execute("A prompt")

    assert result["error"] is not None
    assert "An unexpected error occurred" in result["error"]
    assert result["text_response"] is None
    assert result["finish_reason"] == "error"


@pytest.mark.asyncio
async def test_api_key_missing_during_execute():
    adapter = MistralAdapter.__new__(MistralAdapter)
    adapter.config = MagicMock()
    adapter.config.api_key = None
    adapter.config.default_model = DEFAULT_MISTRAL_MODEL

    result = await adapter.execute("A prompt")

    assert result["error"] == "Mistral API key not configured."
    assert result["text_response"] is None
    assert result["finish_reason"] == "error"


@pytest.mark.asyncio
async def test_close_client(mistral_adapter_with_mocked_client, mock_httpx_client):
    await mistral_adapter_with_mocked_client.close()
    mock_httpx_client.aclose.assert_called_once()
