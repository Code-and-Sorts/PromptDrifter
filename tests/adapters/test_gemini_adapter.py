import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from promptdrifter.adapters.gemini import GeminiAdapter
from promptdrifter.config.adapter_settings import (
    API_KEY_ENV_VAR_GEMINI,
    DEFAULT_GEMINI_MODEL,
    GEMINI_API_BASE_URL,
)


# Fixtures
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
    with patch("httpx.AsyncClient", return_value=mock_httpx_client) as patched_client:
        yield patched_client

@pytest.fixture
def adapter():
    """Provides a GeminiAdapter instance with a dummy API key."""
    # Ensure the env var is not set for consistent testing, or set a dummy one
    with patch.dict("os.environ", {API_KEY_ENV_VAR_GEMINI: "test-api-key"}):
        return GeminiAdapter()

# --- Test Cases ---

@pytest.mark.asyncio
async def test_gemini_adapter_initialization_success(monkeypatch):
    """Test successful initialization with API key from env."""
    monkeypatch.setenv(API_KEY_ENV_VAR_GEMINI, "test-key-from-env")
    adapter = GeminiAdapter()
    assert adapter.api_key == "test-key-from-env"
    assert adapter.base_url == GEMINI_API_BASE_URL

@pytest.mark.asyncio
async def test_gemini_adapter_initialization_with_args():
    """Test successful initialization with arguments."""
    adapter = GeminiAdapter(api_key="test-key-arg", base_url="http://custom.url")
    assert adapter.api_key == "test-key-arg"
    assert adapter.base_url == "http://custom.url"

@pytest.mark.asyncio
async def test_gemini_adapter_initialization_no_key_raises_error(monkeypatch):
    """Test initialization raises ValueError if no API key is provided."""
    monkeypatch.delenv(API_KEY_ENV_VAR_GEMINI, raising=False)
    with pytest.raises(ValueError, match=f"{API_KEY_ENV_VAR_GEMINI} environment variable"):
        GeminiAdapter()

@pytest.mark.asyncio
async def test_gemini_adapter_execute_success(adapter, mock_httpx_client, mock_response):
    """Test successful execution of a prompt."""
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
    mock_httpx_client.post.return_value = mock_response

    result = await adapter.execute(prompt=prompt)

    mock_httpx_client.post.assert_awaited_once()
    call_args, call_kwargs = mock_httpx_client.post.call_args
    endpoint_url = call_args[0]
    payload = call_kwargs['json']
    query_params = call_kwargs['params']

    assert endpoint_url == f"/models/{DEFAULT_GEMINI_MODEL}:generateContent"
    assert query_params['key'] == "test-api-key"
    assert payload['contents'][0]['parts'][0]['text'] == prompt
    assert 'generationConfig' not in payload # No extra params passed

    assert "error" not in result
    assert result["text_response"] == expected_text
    assert result["model_used"] == DEFAULT_GEMINI_MODEL
    assert result["finish_reason"] == "STOP"
    assert result["raw_response"] is not None
    assert result["usage_metadata"]["totalTokenCount"] == 23
    assert len(result["safety_ratings"]) == 1

@pytest.mark.asyncio
async def test_gemini_adapter_execute_with_params(adapter, mock_httpx_client, mock_response):
    """Test execution with specific model, temperature, max_tokens and kwargs."""
    prompt = "Explain quantum physics"
    model = "gemini-1.5-pro-latest"
    temp = 0.5
    max_t = 100
    top_p = 0.9
    safety_setting = {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}

    mock_response.json.return_value = { # Simple response for this test
        "candidates": [{"content": {"parts": [{"text": "Quantum."}]}}]
    }
    mock_httpx_client.post.return_value = mock_response

    result = await adapter.execute(
        prompt=prompt,
        model=model,
        temperature=temp,
        max_tokens=max_t,
        generation_config={"topP": top_p}, # Passed via kwargs
        safetySettings=[safety_setting] # Passed via kwargs
    )

    mock_httpx_client.post.assert_awaited_once()
    call_args, call_kwargs = mock_httpx_client.post.call_args
    endpoint_url = call_args[0]
    payload = call_kwargs['json']

    assert endpoint_url == f"/models/{model}:generateContent"
    assert payload['generationConfig']['temperature'] == temp
    assert payload['generationConfig']['maxOutputTokens'] == max_t
    assert payload['generationConfig']['topP'] == top_p
    assert payload['safetySettings'][0]['category'] == safety_setting['category']
    assert payload['safetySettings'][0]['threshold'] == safety_setting['threshold']

    assert "error" not in result
    assert result["text_response"] == "Quantum."
    assert result["model_used"] == model

@pytest.mark.asyncio
async def test_gemini_adapter_execute_http_status_error(adapter, mock_httpx_client, mock_response):
    """Test handling of HTTPStatusError."""
    prompt = "This will fail"
    error_message = "API key not valid. Please pass a valid API key."
    status_code = 400
    # Configure mock response for error
    mock_response.status_code = status_code
    mock_response.text = json.dumps({"error": {"message": error_message, "status": "INVALID_ARGUMENT"}})
    mock_response.json.return_value = {"error": {"message": error_message, "status": "INVALID_ARGUMENT"}}
    # Configure raise_for_status to actually raise the error
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        message=f"{status_code} Bad Request", request=MagicMock(), response=mock_response
    )
    mock_httpx_client.post.return_value = mock_response

    result = await adapter.execute(prompt=prompt)

    assert "error" in result
    assert str(status_code) in result["error"]
    assert error_message in result["error"]
    assert result["text_response"] is None
    assert result["raw_response"] is not None # Should contain error detail

@pytest.mark.asyncio
async def test_gemini_adapter_execute_request_error(adapter, mock_httpx_client):
    """Test handling of RequestError (e.g., connection issue)."""
    prompt = "Network error test"
    error_message = "Could not connect"
    mock_httpx_client.post.side_effect = httpx.RequestError(error_message, request=MagicMock())

    result = await adapter.execute(prompt=prompt)

    assert "error" in result
    assert "Request error connecting to Gemini API" in result["error"]
    assert error_message in result["error"]
    assert result["text_response"] is None
    assert result["raw_response"] is None

@pytest.mark.asyncio
async def test_gemini_adapter_execute_malformed_response(adapter, mock_httpx_client, mock_response):
    """Test handling of unexpected/malformed JSON response."""
    prompt = "Malformed response test"
    mock_response.json.return_value = {"unexpected_structure": "data"} # Missing 'candidates'
    mock_httpx_client.post.return_value = mock_response

    result = await adapter.execute(prompt=prompt)

    assert "error" in result
    assert "Candidates not found, empty, or not a list in the response." in result["error"]
    assert result["text_response"] is None
    # raw_response should contain the malformed data for debugging
    assert result["raw_response"] == {"unexpected_structure": "data"}

@pytest.mark.asyncio
async def test_gemini_adapter_close(adapter, mock_httpx_client):
    """Test that the close method calls the client's aclose."""
    await adapter.close()
    mock_httpx_client.aclose.assert_awaited_once()
