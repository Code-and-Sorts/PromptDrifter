import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from promptdrifter.adapters.gemini import (
    GeminiAdapter,
    GeminiAdapterConfig,
)
from promptdrifter.config.adapter_settings import (
    API_KEY_ENV_VAR_GEMINI as config_API_KEY_ENV_GEMINI,
)
from promptdrifter.config.adapter_settings import (
    DEFAULT_GEMINI_MODEL as config_DEFAULT_GEMINI_MODEL,
)
from promptdrifter.config.adapter_settings import (
    GEMINI_API_BASE_URL as config_GEMINI_API_BASE_URL,
)

pytestmark = pytest.mark.asyncio

@pytest.fixture
def mock_response():
    response = MagicMock(spec=httpx.Response)
    response.status_code = 200
    response.raise_for_status = MagicMock()
    response.json = MagicMock()
    return response

@pytest.fixture
def mock_httpx_client_instance():
    client = MagicMock(spec=httpx.AsyncClient)
    client.post = AsyncMock()
    client.aclose = AsyncMock()
    return client

@pytest.fixture(autouse=True)
def patch_shared_client(mock_httpx_client_instance):
    async_mock = AsyncMock(return_value=mock_httpx_client_instance)
    with patch(
        "promptdrifter.adapters.gemini.get_shared_client",
        async_mock,
    ) as patched_get_shared_client:
        yield patched_get_shared_client

@pytest.fixture
def patch_httpx_client(mock_httpx_client_instance):
    """Compatibility fixture to maintain test interface"""
    return MagicMock(return_value=mock_httpx_client_instance)

@pytest.fixture
def adapter_config_data():
    return {
        "api_key": "test-api-key-env",
        "base_url": config_GEMINI_API_BASE_URL,
        "default_model": config_DEFAULT_GEMINI_MODEL,
        "max_tokens": 1024,
    }

@pytest.fixture
def adapter(patch_shared_client, adapter_config_data, monkeypatch):
    monkeypatch.setenv(config_API_KEY_ENV_GEMINI, adapter_config_data["api_key"])
    config = GeminiAdapterConfig(**adapter_config_data)
    adapter_instance = GeminiAdapter(config=config)
    return adapter_instance

@pytest.mark.asyncio
async def test_gemini_adapter_init_with_direct_params(monkeypatch, patch_shared_client):
    """
    Test the initialization of GeminiAdapter with direct parameters.
    Expected Behavior: The adapter should be initialized with the provided parameters.
    """
    monkeypatch.delenv(config_API_KEY_ENV_GEMINI, raising=False)
    config = GeminiAdapterConfig(
        api_key="direct_key",
        base_url="custom_url",
        default_model="custom_model",
        max_tokens=512
    )
    adapter_instance = GeminiAdapter(config=config)
    assert adapter_instance.config.api_key == "direct_key"
    assert adapter_instance.config.base_url == "custom_url"
    assert adapter_instance.config.default_model == "custom_model"
    assert adapter_instance.config.max_tokens == 512

@pytest.mark.asyncio
async def test_gemini_adapter_init_with_env_key_and_defaults(monkeypatch, patch_shared_client):
    """
    Test the initialization of GeminiAdapter using environment variables and defaults.
    Expected Behavior: The adapter should be initialized with the environment key and default values.
    """
    monkeypatch.setenv(config_API_KEY_ENV_GEMINI, "env_key_defaults")
    adapter_instance = GeminiAdapter()
    assert adapter_instance.config.api_key == "env_key_defaults"
    assert adapter_instance.config.base_url == config_GEMINI_API_BASE_URL
    assert adapter_instance.config.default_model == config_DEFAULT_GEMINI_MODEL
    assert adapter_instance.config.max_tokens == GeminiAdapterConfig().max_tokens

@pytest.mark.asyncio
async def test_gemini_adapter_init_no_key_raises_error(monkeypatch):
    """
    Test that initializing GeminiAdapter without an API key raises an error.
    Expected Behavior: A ValueError should be raised when no API key is provided.
    """
    monkeypatch.delenv(config_API_KEY_ENV_GEMINI, raising=False)
    with pytest.raises(ValueError) as excinfo:
        GeminiAdapter()
    assert config_API_KEY_ENV_GEMINI in str(excinfo.value)

@pytest.mark.asyncio
async def test_gemini_adapter_init_with_config_object(monkeypatch, patch_httpx_client):
    """
    Test the initialization of GeminiAdapter with a configuration object.
    Expected Behavior: The adapter should be initialized with the provided configuration object.
    """
    monkeypatch.delenv(config_API_KEY_ENV_GEMINI, raising=False)
    config = GeminiAdapterConfig(
        api_key="config_key",
        base_url="config_url",
        default_model="config_model",
        max_tokens=256
    )
    adapter_instance = GeminiAdapter(config=config)
    assert adapter_instance.config is config

@pytest.mark.asyncio
async def test_gemini_adapter_execute_success(adapter, mock_httpx_client_instance):
    """
    Test the successful execution of the GeminiAdapter.
    Expected Behavior: The adapter should return a successful response with the expected text and model name.
    """
    prompt = "Tell me a joke"
    expected_text = "Why did the scarecrow win an award? Because he was outstanding in his field!"
    mock_response_data = {
        "candidates": [
            {
                "content": {"parts": [{"text": expected_text}], "role": "model"},
                "finishReason": "STOP",
                "index": 0,
                "safetyRatings": [
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "probability": "NEGLIGIBLE"}
                ],
            }
        ],
        "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 19, "totalTokenCount": 23},
    }
    mock_http_response = MagicMock(spec=httpx.Response)
    mock_http_response.status_code = 200
    mock_http_response.json = MagicMock(return_value=mock_response_data)
    mock_http_response.raise_for_status = MagicMock()
    mock_httpx_client_instance.post.return_value = mock_http_response

    result = await adapter.execute(prompt=prompt, config_override=GeminiAdapterConfig(default_model="gemini-pro-override"))

    mock_httpx_client_instance.post.assert_awaited_once()
    call_args, call_kwargs = mock_httpx_client_instance.post.call_args
    endpoint_url = call_args[0]
    payload = call_kwargs["json"]
    query_params = call_kwargs["params"]

    assert endpoint_url == "/models/gemini-pro-override:generateContent"
    assert query_params["key"] == adapter.config.api_key
    assert payload["contents"][0]["parts"][0]["text"] == prompt
    if "generationConfig" in payload:
        assert payload["generationConfig"]["maxOutputTokens"] == 2048
    assert result.error is None
    assert result.text_response == expected_text
    assert result.model_name == "gemini-pro-override"
    assert result.finish_reason == "STOP"
    assert "candidates" in result.raw_response
    assert result.raw_response["candidates"][0]["content"]["parts"][0]["text"] == expected_text
    assert result.raw_response["candidates"][0]["finishReason"] == "STOP"

@pytest.mark.asyncio
async def test_gemini_adapter_execute_with_all_params(adapter, patch_httpx_client):
    """
    Test the execution of the GeminiAdapter with all parameters provided.
    Expected Behavior: The adapter should return a successful response with the expected text and model name.
    """
    mock_client_instance = patch_httpx_client.return_value
    prompt = "Explain quantum physics"
    model_override = "gemini-1.5-pro-latest"
    temp_override = 0.5
    max_tokens_override = 150
    safety_setting_val = {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}

    mock_response_data = {"candidates": [{"content": {"parts": [{"text": "Quantum response."}]}, "finishReason": "STOP"}]}
    mock_http_response = MagicMock(spec=httpx.Response)
    mock_http_response.status_code = 200
    mock_http_response.json = MagicMock(return_value=mock_response_data)
    mock_http_response.raise_for_status = MagicMock()
    mock_client_instance.post.return_value = mock_http_response

    result = await adapter.execute(
        prompt=prompt,
        config_override=GeminiAdapterConfig(
            default_model=model_override,
            temperature=temp_override,
            max_tokens=max_tokens_override
        )
    )

    mock_client_instance.post.assert_awaited_once()
    _, call_kwargs = mock_client_instance.post.call_args
    payload = call_kwargs["json"]

    assert "generationConfig" in payload
    assert payload["generationConfig"]["temperature"] == temp_override
    assert payload["generationConfig"]["maxOutputTokens"] == max_tokens_override
    if "safetySettings" in payload:
        assert payload["safetySettings"] == [safety_setting_val]
    assert result.error is None
    assert result.text_response == "Quantum response."
    assert result.model_name == model_override

@pytest.mark.asyncio
async def test_gemini_adapter_execute_http_status_error(adapter, patch_httpx_client):
    """
    Test the execution of the GeminiAdapter with an HTTP status error.
    Expected Behavior: The adapter should return an error response with the expected error message.
    """
    mock_client_instance = patch_httpx_client.return_value
    error_content_text = '{"error": {"message": "API key not valid. Please pass a valid API key.", "status": "INVALID_ARGUMENT"}}'
    mock_error_http_response = MagicMock(spec=httpx.Response)
    mock_error_http_response.status_code = 400
    mock_error_http_response.text = error_content_text
    parsed_error_json = json.loads(error_content_text)
    mock_error_http_response.json = MagicMock(return_value=parsed_error_json)
    mock_error_http_response.request = httpx.Request("POST", adapter.config.base_url)
    mock_error_http_response.raise_for_status = MagicMock(side_effect=httpx.HTTPStatusError(
        message="Client error '400 Bad Request'",
        request=mock_error_http_response.request,
        response=mock_error_http_response
    ))
    mock_client_instance.post.return_value = mock_error_http_response

    result = await adapter.execute("A prompt")

    assert result.error is not None
    assert "API Error (HTTP 400)" in result.error
    assert "API key not valid" in result.error
    assert result.raw_response == {'error_detail': parsed_error_json['error']['message']}
    assert result.text_response is None
    assert result.model_name == adapter.config.default_model
    assert result.usage is None

@pytest.mark.asyncio
async def test_gemini_adapter_execute_request_error(adapter, patch_httpx_client):
    """
    Test the execution of the GeminiAdapter with a request error.
    Expected Behavior: The adapter should return an error response with the expected error message.
    """
    mock_client_instance = patch_httpx_client.return_value
    mock_client_instance.post.side_effect = httpx.ConnectError("Connection failed")
    result = await adapter.execute("A prompt")
    assert result.error is not None
    assert "HTTP Client Error: RequestError -" in result.error
    assert result.raw_response == {"error_detail": "Connection failed"}

@pytest.mark.asyncio
async def test_gemini_adapter_execute_prompt_blocked(adapter, patch_httpx_client):
    """
    Test the execution of the GeminiAdapter with a blocked prompt.
    Expected Behavior: The adapter should return an error response with the expected error message.
    """
    mock_client_instance = patch_httpx_client.return_value
    block_reason = "SAFETY"
    block_message = "This prompt was blocked due to safety concerns."
    mock_response_data = {
        "promptFeedback": {
            "blockReason": block_reason,
            "blockReasonMessage": block_message,
            "safetyRatings": []
        },
        "candidates": []
    }
    mock_http_response = MagicMock(spec=httpx.Response)
    mock_http_response.status_code = 200
    mock_http_response.json = MagicMock(return_value=mock_response_data)
    mock_http_response.raise_for_status = MagicMock()
    mock_client_instance.post.return_value = mock_http_response

    result = await adapter.execute("A potentially harmful prompt")

    assert result.text_response is None
    assert result.error is not None
    assert f"Prompt blocked: {block_reason}" in result.error or "An unexpected error occurred" in result.error
    assert block_message in result.raw_response["promptFeedback"]["blockReasonMessage"]
    assert result.finish_reason == block_reason
    assert result.raw_response["promptFeedback"]["blockReason"] == block_reason
    assert result.raw_response["promptFeedback"]["blockReasonMessage"] == block_message
    assert result.model_name == adapter.config.default_model

@pytest.mark.asyncio
async def test_gemini_adapter_close(adapter, patch_httpx_client):
    """
    Test the close method of the GeminiAdapter.
    Expected Behavior: The adapter should close the HTTP client.
    """
    await adapter.close()
    # With shared client manager, we don't close the client directly
    assert adapter._client is None
