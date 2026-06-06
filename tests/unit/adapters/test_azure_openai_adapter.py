import json
from unittest.mock import (
    AsyncMock,
    MagicMock,
    patch,
)

import httpx
import pytest

from promptdrifter.adapters.azure_openai import (
    AzureOpenAIAdapter,
    AzureOpenAIAdapterConfig,
)
from promptdrifter.config.adapter_settings import (
    API_KEY_ENV_VAR_AZURE_OPENAI as config_API_KEY_ENV_VAR_AZURE_OPENAI,
)
from promptdrifter.config.adapter_settings import (
    DEFAULT_AZURE_OPENAI_API_VERSION as config_DEFAULT_AZURE_OPENAI_API_VERSION,
)
from promptdrifter.config.adapter_settings import (
    DEFAULT_AZURE_OPENAI_MODEL as config_DEFAULT_AZURE_OPENAI_MODEL,
)
from promptdrifter.config.adapter_settings import (
    ENDPOINT_ENV_VAR_AZURE_OPENAI as config_ENDPOINT_ENV_VAR_AZURE_OPENAI,
)

pytestmark = pytest.mark.asyncio

TEST_API_KEY = "test-azure-openai-api-key"
TEST_ENDPOINT = "https://test-resource.openai.azure.com"
TEST_DEPLOYMENT = "gpt-4o-test"

SUCCESS_RESPONSE_PAYLOAD = {
    "id": "chatcmpl-mockid",
    "object": "chat.completion",
    "created": 1677652288,
    "model": config_DEFAULT_AZURE_OPENAI_MODEL,
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "This is a test response from Azure OpenAI.",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
}

API_ERROR_RESPONSE_PAYLOAD = {
    "error": {
        "message": "Access denied due to invalid subscription key.",
        "type": "invalid_request_error",
        "param": None,
        "code": "401",
    }
}


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
        "promptdrifter.adapters.azure_openai.get_shared_client",
        async_mock,
    ) as patched_get_shared_client:
        yield patched_get_shared_client


@pytest.fixture
def adapter(patch_shared_client, monkeypatch):
    monkeypatch.setenv(config_API_KEY_ENV_VAR_AZURE_OPENAI, TEST_API_KEY)
    monkeypatch.setenv(config_ENDPOINT_ENV_VAR_AZURE_OPENAI, TEST_ENDPOINT)
    config = AzureOpenAIAdapterConfig(
        api_key=TEST_API_KEY,
        base_url=TEST_ENDPOINT,
        default_model=TEST_DEPLOYMENT,
        max_tokens=1024,
    )
    return AzureOpenAIAdapter(config=config)


async def test_init_with_direct_params(monkeypatch, patch_shared_client):
    monkeypatch.delenv(config_API_KEY_ENV_VAR_AZURE_OPENAI, raising=False)
    config = AzureOpenAIAdapterConfig(
        api_key="direct_key",
        base_url="https://my-resource.openai.azure.com",
        default_model="custom-deployment",
        api_version="2025-01-01-preview",
        max_tokens=512,
    )
    adapter_instance = AzureOpenAIAdapter(config=config)
    assert adapter_instance.config.api_key == "direct_key"
    assert adapter_instance.config.base_url == "https://my-resource.openai.azure.com"
    assert adapter_instance.config.default_model == "custom-deployment"
    assert adapter_instance.config.api_version == "2025-01-01-preview"
    assert adapter_instance.config.max_tokens == 512


async def test_init_with_env_key_and_endpoint_and_defaults(monkeypatch, patch_shared_client):
    monkeypatch.setenv(config_API_KEY_ENV_VAR_AZURE_OPENAI, "env_key_defaults")
    monkeypatch.setenv(config_ENDPOINT_ENV_VAR_AZURE_OPENAI, TEST_ENDPOINT)
    config = AzureOpenAIAdapterConfig()
    adapter_instance = AzureOpenAIAdapter(config=config)
    assert adapter_instance.config.api_key == "env_key_defaults"
    assert adapter_instance.config.base_url == TEST_ENDPOINT
    assert adapter_instance.config.default_model == config_DEFAULT_AZURE_OPENAI_MODEL
    assert adapter_instance.config.api_version == config_DEFAULT_AZURE_OPENAI_API_VERSION


async def test_init_no_key_raises_error(monkeypatch):
    monkeypatch.delenv(config_API_KEY_ENV_VAR_AZURE_OPENAI, raising=False)
    monkeypatch.setenv(config_ENDPOINT_ENV_VAR_AZURE_OPENAI, TEST_ENDPOINT)
    with pytest.raises(ValueError) as excinfo:
        AzureOpenAIAdapter()
    assert config_API_KEY_ENV_VAR_AZURE_OPENAI in str(excinfo.value)


async def test_init_no_endpoint_raises_error(monkeypatch):
    monkeypatch.setenv(config_API_KEY_ENV_VAR_AZURE_OPENAI, TEST_API_KEY)
    monkeypatch.delenv(config_ENDPOINT_ENV_VAR_AZURE_OPENAI, raising=False)
    with pytest.raises(ValueError) as excinfo:
        AzureOpenAIAdapter()
    assert config_ENDPOINT_ENV_VAR_AZURE_OPENAI in str(excinfo.value)


async def test_headers_use_api_key(adapter):
    headers = adapter.config.get_headers()
    assert headers["api-key"] == TEST_API_KEY
    assert headers["Content-Type"] == "application/json"
    assert "Authorization" not in headers


async def test_execute_successful(adapter, mock_httpx_client):
    mock_http_response = MagicMock(spec=httpx.Response)
    mock_http_response.status_code = 200
    mock_http_response.json = MagicMock(return_value=SUCCESS_RESPONSE_PAYLOAD)
    mock_http_response.raise_for_status = MagicMock()
    mock_httpx_client.post.return_value = mock_http_response

    prompt = "Hello, Azure OpenAI!"
    config_override = AzureOpenAIAdapterConfig(
        default_model="gpt-4o-override",
        api_version="2025-01-01-preview",
        temperature=0.5,
        max_tokens=50,
    )
    result = await adapter.execute(prompt, config_override=config_override)

    mock_httpx_client.post.assert_called_once()
    call_args = mock_httpx_client.post.call_args
    assert call_args[0][0] == "/openai/deployments/gpt-4o-override/chat/completions"
    assert call_args[1]["params"] == {"api-version": "2025-01-01-preview"}
    payload = call_args[1]["json"]
    assert "model" not in payload
    assert payload["messages"] == [{"role": "user", "content": prompt}]
    assert payload["temperature"] == 0.5
    assert payload["max_tokens"] == 50

    assert result.text_response == SUCCESS_RESPONSE_PAYLOAD["choices"][0]["message"]["content"]
    assert result.raw_response == SUCCESS_RESPONSE_PAYLOAD
    assert result.model_name == "gpt-4o-override"
    assert result.finish_reason == "stop"
    assert result.usage == SUCCESS_RESPONSE_PAYLOAD["usage"]
    assert result.error is None

    await adapter.close()


async def test_execute_uses_instance_defaults(adapter, mock_httpx_client):
    mock_http_response = MagicMock(spec=httpx.Response)
    mock_http_response.status_code = 200
    mock_http_response.json = MagicMock(return_value=SUCCESS_RESPONSE_PAYLOAD)
    mock_http_response.raise_for_status = MagicMock()
    mock_httpx_client.post.return_value = mock_http_response

    await adapter.execute("Test prompt")
    call_args = mock_httpx_client.post.call_args
    assert call_args[0][0] == f"/openai/deployments/{TEST_DEPLOYMENT}/chat/completions"
    assert call_args[1]["params"] == {"api-version": config_DEFAULT_AZURE_OPENAI_API_VERSION}
    payload = call_args[1]["json"]
    assert payload["max_tokens"] == adapter.config.max_tokens


async def test_execute_with_system_prompt(adapter, mock_httpx_client):
    mock_http_response = MagicMock(spec=httpx.Response)
    mock_http_response.status_code = 200
    mock_http_response.json = MagicMock(return_value=SUCCESS_RESPONSE_PAYLOAD)
    mock_http_response.raise_for_status = MagicMock()
    mock_httpx_client.post.return_value = mock_http_response

    config_override = AzureOpenAIAdapterConfig(
        default_model=adapter.config.default_model,
        system_prompt="You are a test bot.",
    )
    await adapter.execute("Hello bot", config_override=config_override)
    payload = mock_httpx_client.post.call_args[1]["json"]
    assert payload["messages"][0] == {"role": "system", "content": "You are a test bot."}
    assert payload["messages"][1] == {"role": "user", "content": "Hello bot"}


async def test_execute_http_status_error(adapter, mock_httpx_client):
    error_response_content_str = json.dumps(API_ERROR_RESPONSE_PAYLOAD)

    mock_error_http_response = MagicMock(spec=httpx.Response)
    mock_error_http_response.status_code = 401
    mock_error_http_response.text = error_response_content_str
    mock_error_http_response.request = httpx.Request("POST", "/chat/completions")
    mock_error_http_response.raise_for_status = MagicMock(side_effect=httpx.HTTPStatusError(
        message="Client error '401 Unauthorized'",
        request=mock_error_http_response.request,
        response=mock_error_http_response,
    ))
    mock_httpx_client.post.return_value = mock_error_http_response

    result = await adapter.execute("A prompt")

    assert result.error.startswith("API Error (HTTP 401):")
    assert error_response_content_str in result.error
    assert result.raw_response == {"error_detail": error_response_content_str}
    assert result.text_response is None
    assert result.model_name == adapter.config.default_model
    assert result.finish_reason == "error"


async def test_execute_request_error(adapter, mock_httpx_client):
    mock_httpx_client.post.side_effect = httpx.ConnectError("Connection failed")

    result = await adapter.execute("A prompt")

    assert result.error.startswith("HTTP Client Error: RequestError")
    assert "Connection failed" in result.error
    assert result.raw_response == {"error_detail": "Connection failed"}
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


async def test_adapter_close(adapter):
    await adapter.close()
    assert adapter._client is None
