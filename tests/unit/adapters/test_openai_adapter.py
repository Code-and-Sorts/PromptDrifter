import json
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

TEST_API_KEY = "test-openai-api-key"
TEST_PROMPT = "Hello, OpenAI from test!"
TEST_MODEL = "gpt-4-test"
CUSTOM_BASE_URL = "http://localhost:8000"

SUCCESS_RESPONSE_PAYLOAD = {
    "id": "chatcmpl-mockid",
    "object": "chat.completion",
    "created": 1677652288,
    "model": config_DEFAULT_OPENAI_MODEL,
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "This is a test response from OpenAI.",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
}

API_ERROR_RESPONSE_PAYLOAD = {
    "error": {
        "message": "The API key provided is invalid.",
        "type": "invalid_request_error",
        "param": None,
        "code": "invalid_api_key",
    }
}

@pytest.fixture
def mock_response():
    response = MagicMock(spec=httpx.Response)
    response.status_code = 200
    response.raise_for_status = MagicMock()
    response.json = MagicMock()
    return response

@pytest.fixture
def mock_httpx_client():
    client = MagicMock(spec=httpx.AsyncClient)
    client.post = AsyncMock()
    client.aclose = AsyncMock()
    return client

@pytest.fixture(autouse=True)
def patch_shared_client(mock_httpx_client):
    async_mock = AsyncMock(return_value=mock_httpx_client)
    with patch(
        "promptdrifter.adapters.openai.get_shared_client",
        async_mock,
    ) as patched_get_shared_client:
        yield patched_get_shared_client

@pytest.fixture
def adapter_config_data():
    return {
        "api_key": TEST_API_KEY,
        "base_url": config_OPENAI_API_BASE_URL,
        "default_model": config_DEFAULT_OPENAI_MODEL,
        "max_tokens": 1024,
    }

@pytest.fixture
def adapter(patch_shared_client, adapter_config_data, monkeypatch):
    monkeypatch.setenv(config_API_KEY_ENV_VAR_OPENAI, adapter_config_data["api_key"])
    config = OpenAIAdapterConfig(
        api_key=adapter_config_data["api_key"],
        base_url=adapter_config_data["base_url"],
        default_model=adapter_config_data["default_model"],
        max_tokens=adapter_config_data["max_tokens"]
    )
    adapter_instance = OpenAIAdapter(config=config)
    return adapter_instance

async def test_openai_adapter_init_with_direct_params(monkeypatch, patch_shared_client):
    monkeypatch.delenv(config_API_KEY_ENV_VAR_OPENAI, raising=False)
    config = OpenAIAdapterConfig(
        api_key="direct_key",
        base_url="custom_url",
        default_model="custom_model",
        max_tokens=512
    )
    adapter_instance = OpenAIAdapter(config=config)
    assert adapter_instance.config.api_key == "direct_key"
    assert adapter_instance.config.base_url == "custom_url"
    assert adapter_instance.config.default_model == "custom_model"
    assert adapter_instance.config.max_tokens == 512

async def test_openai_adapter_init_with_env_key_and_defaults(monkeypatch, patch_shared_client):
    monkeypatch.setenv(config_API_KEY_ENV_VAR_OPENAI, "env_key_defaults")
    config = OpenAIAdapterConfig()
    adapter_instance = OpenAIAdapter(config=config)
    assert adapter_instance.config.api_key == "env_key_defaults"
    assert adapter_instance.config.base_url == config_OPENAI_API_BASE_URL
    assert adapter_instance.config.default_model == config_DEFAULT_OPENAI_MODEL
    assert adapter_instance.config.max_tokens == OpenAIAdapterConfig().max_tokens

async def test_openai_adapter_init_no_key_raises_error(monkeypatch):
    monkeypatch.delenv(config_API_KEY_ENV_VAR_OPENAI, raising=False)
    with pytest.raises(ValueError) as excinfo:
        OpenAIAdapter()
    assert config_API_KEY_ENV_VAR_OPENAI in str(excinfo.value)

async def test_openai_adapter_init_with_config_object(monkeypatch, patch_shared_client):
    monkeypatch.delenv(config_API_KEY_ENV_VAR_OPENAI, raising=False)
    config = OpenAIAdapterConfig(
        api_key="config_key",
        base_url="config_url",
        default_model="config_model",
        max_tokens=256
    )
    adapter_instance = OpenAIAdapter(config=config)
    assert adapter_instance.config is config
    assert adapter_instance.config.api_key == "config_key"
    assert adapter_instance.config.base_url == "config_url"
    assert adapter_instance.config.default_model == "config_model"
    assert adapter_instance.config.max_tokens == 256

async def test_execute_successful(adapter, mock_httpx_client):
    mock_http_response = MagicMock(spec=httpx.Response)
    mock_http_response.status_code = 200
    mock_http_response.json = MagicMock(return_value=SUCCESS_RESPONSE_PAYLOAD)
    mock_http_response.raise_for_status = MagicMock()
    mock_httpx_client.post.return_value = mock_http_response

    prompt = "Hello, OpenAI!"
    config_override = OpenAIAdapterConfig(
        default_model="gpt-4-override",
        temperature=0.5,
        max_tokens=50
    )
    result = await adapter.execute(prompt, config_override=config_override)

    mock_httpx_client.post.assert_called_once()
    call_args = mock_httpx_client.post.call_args
    assert call_args[0][0] == "/chat/completions"
    payload = call_args[1]["json"]
    assert payload["model"] == "gpt-4-override"
    assert payload["messages"] == [{"role": "user", "content": prompt}]
    assert payload["temperature"] == 0.5
    assert payload["max_tokens"] == 50

    assert result.text_response == SUCCESS_RESPONSE_PAYLOAD["choices"][0]["message"]["content"]
    assert result.raw_response == SUCCESS_RESPONSE_PAYLOAD
    assert result.model_name == "gpt-4-override"
    assert result.finish_reason == "stop"
    assert result.usage == SUCCESS_RESPONSE_PAYLOAD["usage"]
    assert result.error is None

    await adapter.close()

async def test_execute_uses_config_default_max_tokens(adapter, mock_httpx_client):
    mock_http_response = MagicMock(spec=httpx.Response)
    mock_http_response.status_code = 200
    mock_http_response.json = MagicMock(return_value=SUCCESS_RESPONSE_PAYLOAD)
    mock_http_response.raise_for_status = MagicMock()
    mock_httpx_client.post.return_value = mock_http_response

    await adapter.execute("Test prompt")
    payload = mock_httpx_client.post.call_args[1]["json"]
    assert payload["max_tokens"] == adapter.config.max_tokens
    assert payload["model"] == adapter.config.default_model

    result = await adapter.execute("Test prompt")
    assert result.usage == SUCCESS_RESPONSE_PAYLOAD["usage"]
    assert result.finish_reason == "stop"

async def test_execute_http_status_error(adapter, mock_httpx_client):
    error_response_content_str = json.dumps(API_ERROR_RESPONSE_PAYLOAD)

    mock_error_http_response = MagicMock(spec=httpx.Response)
    mock_error_http_response.status_code = 401
    mock_error_http_response.text = error_response_content_str
    mock_error_http_response.json = MagicMock(return_value=API_ERROR_RESPONSE_PAYLOAD)
    mock_error_http_response.request = httpx.Request("POST", "/chat/completions")
    mock_error_http_response.raise_for_status = MagicMock(side_effect=httpx.HTTPStatusError(
        message="Client error '401 Unauthorized' for url 'https://api.openai.com/v1/chat/completions'",
        request=mock_error_http_response.request,
        response=mock_error_http_response
    ))
    mock_httpx_client.post.return_value = mock_error_http_response

    result = await adapter.execute("A prompt")

    assert result.error.startswith("API Error (HTTP 401):")
    assert error_response_content_str in result.error
    assert result.raw_response == {"error_detail": error_response_content_str}
    assert result.text_response is None
    assert result.model_name == adapter.config.default_model
    assert result.finish_reason == "error"
    assert result.usage is None

async def test_execute_request_error(adapter, mock_httpx_client):
    mock_httpx_client.post.side_effect = httpx.ConnectError("Connection failed")

    result = await adapter.execute("A prompt")

    assert result.error.startswith("HTTP Client Error: RequestError")
    assert "Connection failed" in result.error
    assert result.raw_response == {"error_detail": "Connection failed"}
    assert result.text_response is None
    assert result.model_name == adapter.config.default_model
    assert result.finish_reason == "error"

async def test_execute_timeout_error(adapter, mock_httpx_client):
    timeout_message = "Read operation timed out"
    mock_httpx_client.post.side_effect = httpx.ReadTimeout(timeout_message)

    result = await adapter.execute("A prompt for timeout")

    assert result.error.startswith("HTTP Client Error: RequestError")
    assert timeout_message in result.error
    assert result.raw_response == {"error_detail": timeout_message}
    assert result.text_response is None
    assert result.finish_reason == "error"

async def test_execute_unexpected_error(adapter, mock_httpx_client):
    mock_httpx_client.post.side_effect = Exception("Something totally unexpected")

    result = await adapter.execute("A prompt for unexpected")

    assert result.error is not None
    assert "An unexpected error occurred" in result.error
    assert "Something totally unexpected" in result.error
    assert result.raw_response == {"error_detail": "Something totally unexpected"}
    assert result.text_response is None
    assert result.finish_reason == "error"

async def test_adapter_close_called(adapter, mock_httpx_client):
    await adapter.close()

async def test_execute_with_system_prompt_in_kwargs(adapter, mock_httpx_client):
    mock_http_response = MagicMock(spec=httpx.Response)
    mock_http_response.status_code = 200
    mock_http_response.json = MagicMock(return_value=SUCCESS_RESPONSE_PAYLOAD)
    mock_http_response.raise_for_status = MagicMock()
    mock_httpx_client.post.return_value = mock_http_response

    config_override = OpenAIAdapterConfig(
        default_model=adapter.config.default_model,
        system_prompt="You are a test bot."
    )
    await adapter.execute("Hello bot", config_override=config_override)
    payload = mock_httpx_client.post.call_args[1]["json"]
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][0]["content"] == "You are a test bot."
    assert payload["messages"][1]["role"] == "user"
    assert payload["messages"][1]["content"] == "Hello bot"
    result = await adapter.execute("Hello bot", config_override=config_override)
    assert result.text_response == SUCCESS_RESPONSE_PAYLOAD["choices"][0]["message"]["content"]
