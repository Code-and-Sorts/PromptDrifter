import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from promptdrifter.adapters.gemini import (
    GeminiAdapter,
    GeminiAdapterConfig,
)
from promptdrifter.config.adapter_settings import (
    API_KEY_ENV_VAR_GEMINI as config_API_KEY_ENV_VAR_GEMINI,
)
from promptdrifter.config.adapter_settings import (
    DEFAULT_GEMINI_MODEL as config_DEFAULT_GEMINI_MODEL,
)
from promptdrifter.config.adapter_settings import (
    GEMINI_API_BASE_URL as config_GEMINI_API_BASE_URL,
)


# Fixtures
@pytest.fixture
def mock_response():
    """Creates a mock httpx.Response."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = 200
    response.raise_for_status = MagicMock()
    response.json = MagicMock()
    return response

@pytest.fixture
def mock_httpx_client_instance():
    """Creates a mock httpx.AsyncClient instance."""
    client = MagicMock(spec=httpx.AsyncClient)
    client.post = AsyncMock()
    client.aclose = AsyncMock()
    return client

@pytest.fixture
def patch_httpx_client(mock_httpx_client_instance):
    """Patches httpx.AsyncClient in the gemini adapter module."""
    with patch("promptdrifter.adapters.gemini.httpx.AsyncClient", return_value=mock_httpx_client_instance) as patched_class_mock:
        yield patched_class_mock

@pytest.fixture
def adapter(patch_httpx_client):
    """Provides a GeminiAdapter instance with API key from env via Pydantic config."""
    with patch.dict(os.environ, {config_API_KEY_ENV_VAR_GEMINI: "test-api-key-env"}):
        adapter_instance = GeminiAdapter()
        patch_httpx_client.assert_called_once_with(base_url=adapter_instance.config.base_url)
        return adapter_instance

# --- Test Cases ---

@pytest.mark.asyncio
async def test_gemini_adapter_init_with_direct_key(monkeypatch, patch_httpx_client):
    monkeypatch.delenv(config_API_KEY_ENV_VAR_GEMINI, raising=False)
    adapter_instance = GeminiAdapter(api_key="direct_key", base_url="custom_url")
    assert adapter_instance.config.api_key == "direct_key"
    assert adapter_instance.config.base_url == "custom_url"
    assert adapter_instance.config.default_model == config_DEFAULT_GEMINI_MODEL
    patch_httpx_client.assert_called_once_with(base_url="custom_url")

@pytest.mark.asyncio
async def test_gemini_adapter_init_with_env_key(monkeypatch, patch_httpx_client):
    monkeypatch.setenv(config_API_KEY_ENV_VAR_GEMINI, "env_key")
    adapter_instance = GeminiAdapter()
    assert adapter_instance.config.api_key == "env_key"
    assert adapter_instance.config.base_url == config_GEMINI_API_BASE_URL
    patch_httpx_client.assert_called_once_with(base_url=config_GEMINI_API_BASE_URL)

@pytest.mark.asyncio
async def test_gemini_adapter_init_no_key_raises_error(monkeypatch):
    monkeypatch.delenv(config_API_KEY_ENV_VAR_GEMINI, raising=False)
    with pytest.raises(ValueError) as excinfo:
        GeminiAdapter()
    assert config_API_KEY_ENV_VAR_GEMINI in str(excinfo.value)

@pytest.mark.asyncio
async def test_gemini_adapter_init_with_config_object(monkeypatch, patch_httpx_client):
    monkeypatch.delenv(config_API_KEY_ENV_VAR_GEMINI, raising=False)
    config = GeminiAdapterConfig(api_key="config_key", base_url="config_url", default_model="config_model")
    adapter_instance = GeminiAdapter(config=config)
    assert adapter_instance.config is config
    assert adapter_instance.config.api_key == "config_key"
    assert adapter_instance.config.base_url == "config_url"
    assert adapter_instance.config.default_model == "config_model"
    patch_httpx_client.assert_called_once_with(base_url="config_url")

@pytest.mark.asyncio
async def test_gemini_adapter_execute_success(adapter, patch_httpx_client, mock_response):
    """Test successful execution of a prompt."""
    mock_client_instance = patch_httpx_client.return_value # Get the mocked client instance

    prompt = "Tell me a joke"
    expected_text = "Why did the scarecrow win an award? Because he was outstanding in his field!"
    mock_response.json.return_value = {
        "candidates": [
            {
                "content": {"parts": [{"text": expected_text}], "role": "model"},
                "finishReason": "STOP",
                "index": 0,
                "safetyRatings": [{"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "probability": "NEGLIGIBLE"}]
            }
        ],
        "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 19, "totalTokenCount": 23}
    }
    mock_client_instance.post.return_value = mock_response

    result = await adapter.execute(prompt=prompt)

    mock_client_instance.post.assert_awaited_once()
    call_args, call_kwargs = mock_client_instance.post.call_args
    endpoint_url = call_args[0]
    payload = call_kwargs['json']
    query_params = call_kwargs['params']

    assert endpoint_url == f"/models/{adapter.config.default_model}:generateContent"
    assert query_params['key'] == adapter.config.api_key
    assert payload['contents'][0]['parts'][0]['text'] == prompt
    assert 'generationConfig' not in payload

    assert "error" not in result
    assert result["text_response"] == expected_text
    assert result["model_used"] == adapter.config.default_model
    assert result["finish_reason"] == "STOP"
    assert result["raw_response"] is not None
    assert result["usage_metadata"]["totalTokenCount"] == 23
    assert len(result["safety_ratings"]) == 1
    await adapter.close()
    mock_client_instance.aclose.assert_called_once()

@pytest.mark.asyncio
async def test_gemini_adapter_execute_with_params(adapter, patch_httpx_client, mock_response):
    """Test execution with specific model, temperature, max_tokens and kwargs."""
    mock_client_instance = patch_httpx_client.return_value
    prompt = "Explain quantum physics"
    model = "gemini-1.5-pro-latest"
    temp = 0.5
    max_t = 100
    top_p = 0.9
    safety_setting = {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}

    mock_response.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "Quantum."}]}}]
    }
    mock_client_instance.post.return_value = mock_response

    result = await adapter.execute(
        prompt=prompt, model=model, temperature=temp, max_tokens=max_t,
        generation_config={"topP": top_p},
        safetySettings=[safety_setting]
    )

    mock_client_instance.post.assert_awaited_once()
    call_args, call_kwargs = mock_client_instance.post.call_args
    endpoint_url = call_args[0]
    payload = call_kwargs['json']
    query_params = call_kwargs['params']

    assert endpoint_url == f"/models/{model}:generateContent"
    assert query_params['key'] == adapter.config.api_key
    assert payload['generationConfig']['temperature'] == temp
    assert payload['generationConfig']['maxOutputTokens'] == max_t
    assert payload['generationConfig']['topP'] == top_p
    assert payload['safetySettings'][0]['category'] == safety_setting['category']
    assert payload['safetySettings'][0]['threshold'] == safety_setting['threshold']

    assert "error" not in result
    assert result["text_response"] == "Quantum."
    assert result["model_used"] == model
    await adapter.close()
    mock_client_instance.aclose.assert_called_once()

@pytest.mark.asyncio
async def test_gemini_adapter_execute_http_status_error(adapter, patch_httpx_client, mock_response):
    mock_client_instance = patch_httpx_client.return_value
    prompt = "This will fail"
    error_message = "API key not valid. Please pass a valid API key."
    status_code = 400
    mock_response.status_code = status_code
    mock_response.text = json.dumps({"error": {"message": error_message, "status": "INVALID_ARGUMENT"}})
    mock_response.json.return_value = {"error": {"message": error_message, "status": "INVALID_ARGUMENT"}}
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        message=f"{status_code} Bad Request", request=MagicMock(), response=mock_response
    )
    mock_client_instance.post.return_value = mock_response

    result = await adapter.execute(prompt=prompt)

    assert "error" in result
    assert str(status_code) in result["error"]
    assert error_message in result["error"]
    assert result["text_response"] is None
    assert result["raw_response"] is not None
    await adapter.close()
    mock_client_instance.aclose.assert_called_once()

@pytest.mark.asyncio
async def test_gemini_adapter_execute_request_error(adapter, patch_httpx_client):
    mock_client_instance = patch_httpx_client.return_value
    mock_client_instance.post.side_effect = httpx.ConnectError("Could not connect", request=MagicMock())
    result = await adapter.execute(prompt="Network error test")

    assert "error" in result
    assert "Request error connecting to Gemini API" in result["error"]
    assert result["text_response"] is None
    assert result["raw_response"] is None
    await adapter.close()
    mock_client_instance.aclose.assert_called_once()

@pytest.mark.asyncio
async def test_gemini_adapter_execute_malformed_response(adapter, patch_httpx_client, mock_response):
    mock_client_instance = patch_httpx_client.return_value
    mock_response.json.return_value = {"unexpected_structure": "data"}
    mock_client_instance.post.return_value = mock_response
    result = await adapter.execute(prompt="Malformed response test")

    assert "error" in result
    assert "Candidates not found" in result["error"]
    assert result["text_response"] is None
    assert result["raw_response"] == {"unexpected_structure": "data"}
    await adapter.close()
    mock_client_instance.aclose.assert_called_once()

@pytest.mark.asyncio
async def test_gemini_adapter_close(adapter, patch_httpx_client):
    mock_client_instance = patch_httpx_client.return_value
    await adapter.close()
    mock_client_instance.aclose.assert_called_once()
