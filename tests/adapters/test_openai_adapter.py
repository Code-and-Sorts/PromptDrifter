import os
from unittest.mock import (
    AsyncMock,
    MagicMock,
    patch,
)

import httpx
import pytest

from promptdrifter.adapters.base import Adapter
from promptdrifter.adapters.openai import (
    API_KEY_ENV_VAR_OPENAI,
    DEFAULT_OPENAI_MODEL,
    OPENAI_API_BASE_URL,
    OpenAIAdapter,
)
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
    response.raise_for_status = MagicMock() # Does nothing for success
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
    # Need to patch httpx.AsyncClient within the adapters module context if imported there
    with patch("promptdrifter.adapters.openai.httpx.AsyncClient", return_value=mock_httpx_client) as patched_client:
        yield patched_client


@pytest.fixture
def adapter():
    """Provides an OpenAIAdapter instance with a dummy API key."""
    # Use patch.dict correctly for os.environ
    with patch.dict(os.environ, {API_KEY_ENV_VAR_OPENAI: "test-api-key"}):
        return OpenAIAdapter()


@pytest.fixture
def openai_adapter_with_key(monkeypatch, mock_httpx_client):
    monkeypatch.setenv(API_KEY_ENV_VAR_OPENAI, "test_api_key_from_env")
    adapter = OpenAIAdapter(api_key="test_api_key_direct")
    adapter.client = mock_httpx_client
    assert adapter.api_key == "test_api_key_direct"
    return adapter


@pytest.fixture
def openai_adapter_no_key_factory(monkeypatch):
    """Factory fixture to get OpenAIAdapter class when env var is unset."""
    monkeypatch.delenv(API_KEY_ENV_VAR_OPENAI, raising=False)
    return OpenAIAdapter


async def test_openai_adapter_init_with_direct_key(monkeypatch):
    """Test initialization with a directly passed API key."""
    monkeypatch.delenv(API_KEY_ENV_VAR_OPENAI, raising=False)
    adapter_instance = OpenAIAdapter(api_key="direct_key", base_url="custom")
    assert adapter_instance.api_key == "direct_key"
    assert adapter_instance.base_url == "custom"


async def test_openai_adapter_init_with_env_key(monkeypatch):
    """Test initialization uses API key from environment variable."""
    monkeypatch.setenv(API_KEY_ENV_VAR_OPENAI, "env_key")
    adapter_instance = OpenAIAdapter()
    assert adapter_instance.api_key == "env_key"
    assert adapter_instance.base_url == OPENAI_API_BASE_URL


async def test_openai_adapter_init_no_key_raises_error(openai_adapter_no_key_factory):
    """Test initialization raises ValueError if no API key is found."""
    with pytest.raises(ValueError) as excinfo:
        openai_adapter_no_key_factory()
    assert API_KEY_ENV_VAR_OPENAI in str(excinfo.value)


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


@pytest.mark.asyncio
async def test_openai_adapter_initialization_success_env(monkeypatch):
    """Test successful initialization with API key from env."""
    monkeypatch.setenv(API_KEY_ENV_VAR_OPENAI, "test-key-from-env")
    adapter_instance = OpenAIAdapter()
    assert adapter_instance.api_key == "test-key-from-env"
    assert adapter_instance.base_url == OPENAI_API_BASE_URL


@pytest.mark.asyncio
async def test_openai_adapter_initialization_with_args():
    """Test successful initialization with arguments."""
    # Ensure env var isn't interfering
    with patch.dict(os.environ, {}, clear=True):
        adapter_instance = OpenAIAdapter(api_key="test-key-arg", base_url="http://custom.url")
    assert adapter_instance.api_key == "test-key-arg"
    assert adapter_instance.base_url == "http://custom.url"


@pytest.mark.asyncio
async def test_openai_adapter_execute_success(adapter, mock_httpx_client, mock_response):
    """Test successful execution and response parsing."""
    prompt = "Test prompt"
    expected_response = "Test response content"
    mock_response.json.return_value = {
        "choices": [
            {"message": {"role": "assistant", "content": expected_response}}
        ],
        "model": DEFAULT_OPENAI_MODEL,
        "usage": {"total_tokens": 10}
    }
    mock_httpx_client.post.return_value = mock_response

    result = await adapter.execute(prompt=prompt)

    mock_httpx_client.post.assert_awaited_once()
    call_args, call_kwargs = mock_httpx_client.post.call_args
    endpoint_url = call_args[0]
    payload = call_kwargs['json']

    assert endpoint_url == "/chat/completions"
    assert payload["model"] == DEFAULT_OPENAI_MODEL
    assert payload["messages"][0]["content"] == prompt
    assert "error" not in result
    assert result["text_response"] == expected_response
    assert result["model_used"] == DEFAULT_OPENAI_MODEL


@pytest.mark.asyncio
async def test_openai_adapter_execute_with_params(adapter, mock_httpx_client, mock_response):
    """Test passing parameters like model, temperature, max_tokens."""
    prompt = "Test prompt"
    model = "gpt-4-test"
    temperature = 0.5
    max_tokens = 50
    custom_kwarg = {"top_p": 0.9}

    mock_response.json.return_value = {"choices": [{"message": {"content": "..."}}]}
    mock_httpx_client.post.return_value = mock_response

    await adapter.execute(
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        **custom_kwarg
    )

    mock_httpx_client.post.assert_awaited_once()
    _, call_kwargs = mock_httpx_client.post.call_args
    payload = call_kwargs['json']

    assert payload["model"] == model
    assert payload["temperature"] == temperature
    assert payload["max_tokens"] == max_tokens
    assert payload["top_p"] == custom_kwarg["top_p"]


@pytest.mark.asyncio
async def test_openai_adapter_execute_http_error(adapter, mock_httpx_client, mock_response):
    """Test handling of HTTPStatusError."""
    prompt = "Error prompt"
    status_code = 401
    error_text = "Unauthorized - Invalid API Key"

    mock_response.status_code = status_code
    mock_response.text = error_text
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        message=f"{status_code} Client Error", request=MagicMock(), response=mock_response
    )
    mock_httpx_client.post.return_value = mock_response

    result = await adapter.execute(prompt=prompt)

    assert "error" in result
    assert f"HTTP error {status_code}" in result["error"]
    assert error_text in result["error"]
    assert result["text_response"] is None


@pytest.mark.asyncio
async def test_openai_adapter_execute_request_error(adapter, mock_httpx_client):
    """Test handling of RequestError (e.g., connection error)."""
    prompt = "Request error prompt"
    error_msg = "Connection failed"
    mock_httpx_client.post.side_effect = httpx.RequestError(error_msg, request=MagicMock())

    result = await adapter.execute(prompt=prompt)

    assert "error" in result
    assert "Request error connecting to OpenAI" in result["error"]
    assert error_msg in result["error"]
    assert result["text_response"] is None


@pytest.mark.asyncio
async def test_openai_adapter_close(adapter, mock_httpx_client):
    """Test that the close method calls the client's aclose."""
    await adapter.close()
    mock_httpx_client.aclose.assert_awaited_once()
