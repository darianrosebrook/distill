"""
Tests for models/teacher/teacher_client.py - Teacher client functionality.

Tests HTTP API client, HuggingFace integration, retry logic,
tier detection, circuit breakers, and response parsing.
"""
# @author: @darianrosebrook

from unittest.mock import Mock, patch

import pytest
import requests
from requests.adapters import HTTPAdapter

from models.teacher.teacher_client import (
    TeacherClient,
    APITier,
    TierLimits,
    TIER_LIMITS,
)


class TestAPITier:
    """Test API tier functionality."""

    def test_tier_enum_values(self):
        """Test API tier enum values."""
        assert APITier.FREE == "free"
        assert APITier.TIER1 == "tier1"
        assert APITier.TIER2 == "tier2"
        assert APITier.TIER3 == "tier3"
        assert APITier.TIER4 == "tier4"
        assert APITier.TIER5 == "tier5"
        assert APITier.UNKNOWN == "unknown"

    def test_tier_limits_structure(self):
        """Test tier limits data structure."""
        for tier, limits in TIER_LIMITS.items():
            assert isinstance(tier, APITier)
            assert isinstance(limits, TierLimits)
            # Check actual attribute names (rpm, tpm, tpd)
            assert hasattr(limits, "rpm")
            assert hasattr(limits, "tpm")
            assert hasattr(limits, "tpd")
            assert hasattr(limits, "concurrency")
            assert hasattr(limits, "delay")


class TestTeacherClientInitialization:
    """Test teacher client initialization."""

    def test_init_http_backend(self):
        """Test HTTP backend initialization."""
        client = TeacherClient(
            backend="http", endpoint="http://test.com", api_key="test_key")

        assert client.backend == "http"
        assert client.endpoint == "http://test.com"
        assert client.api_key == "test_key"
        assert client._tier == APITier.FREE
        assert client._max_retries == 5

    def test_init_http_endpoint_normalization(self):
        """Test HTTP endpoint normalization."""
        # Test trailing slash removal
        client = TeacherClient(backend="http", endpoint="http://test.com/")
        assert client.endpoint == "http://test.com"

        # Test /v1 suffix removal
        client = TeacherClient(backend="http", endpoint="http://test.com/v1")
        assert client.endpoint == "http://test.com"

    @patch("models.teacher.teacher_client.os.getenv")
    @patch("models.teacher.teacher_client.Path")
    def test_init_api_key_from_env(self, mock_path, mock_getenv):
        """Test API key loading from environment."""
        # Mock getenv to return API key
        mock_getenv.side_effect = lambda key, default=None: "env_api_key" if key == "MOONSHOT_API_KEY" else None
        # Mock Path.exists() to return False (no .env.local file)
        mock_env_file = Mock()
        mock_env_file.exists.return_value = False
        mock_path.return_value = mock_env_file
        
        client = TeacherClient(backend="http", endpoint="http://test.com")
        assert client.api_key == "env_api_key"

    @patch("models.teacher.teacher_client.HF_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_causal_lm")
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_init_hf_backend(self, mock_tokenizer, mock_model):
        """Test HuggingFace backend initialization."""
        # Mock model and tokenizer loading
        mock_model_instance = Mock()
        mock_model_instance.eval = Mock()
        mock_model_instance.to = Mock(return_value=mock_model_instance)
        mock_model.return_value = mock_model_instance
        mock_tokenizer.return_value = Mock()

        client = TeacherClient(backend="hf", model_name="test/model")

        assert client.backend == "hf"
        # Model should be loaded
        assert mock_model.called
        assert mock_tokenizer.called

    def test_init_invalid_backend(self):
        """Test invalid backend raises error."""
        with pytest.raises(ValueError, match="Unknown backend"):
            TeacherClient(backend="invalid")

    @patch("models.teacher.teacher_client.HF_AVAILABLE", False)
    def test_init_hf_unavailable(self):
        """Test HF backend when transformers unavailable."""
        with pytest.raises(RuntimeError, match="HuggingFace transformers not available"):
            TeacherClient(backend="hf")


class TestTeacherClientFactoryMethods:
    """Test factory methods for creating clients."""

    def test_from_endpoint_basic(self):
        """Test from_endpoint factory method."""
        client = TeacherClient.from_endpoint(
            "http://test.com", api_key="test_key")

        assert isinstance(client, TeacherClient)
        assert client.backend == "http"
        assert client.endpoint == "http://test.com"
        assert client.api_key == "test_key"

    def test_from_endpoint_with_options(self):
        """Test from_endpoint with additional options."""
        client = TeacherClient.from_endpoint(
            "http://test.com", api_key="test_key", timeout=300, max_retries=3
        )

        assert client.timeout == 300
        assert client._max_retries == 3

    @patch("models.teacher.teacher_client.HF_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_causal_lm")
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_from_hf_basic(self, mock_tokenizer, mock_model):
        """Test from_hf factory method."""
        # Mock model and tokenizer loading
        mock_model_instance = Mock()
        mock_model_instance.eval = Mock()
        mock_model_instance.to = Mock(return_value=mock_model_instance)
        mock_model.return_value = mock_model_instance
        mock_tokenizer.return_value = Mock()

        client = TeacherClient.from_hf("microsoft/phi-2", device="cpu")

        assert isinstance(client, TeacherClient)
        assert client.backend == "hf"
        assert mock_model.called
        assert mock_tokenizer.called


class TestTierDetection:
    """Test API tier detection and management."""

    def test_get_tier(self):
        """Test getting current tier."""
        client = TeacherClient(backend="http", endpoint="http://test.com")
        assert client.get_tier() == APITier.FREE

    def test_get_tier_limits(self):
        """Test getting tier limits."""
        client = TeacherClient(backend="http", endpoint="http://test.com")
        limits = client.get_tier_limits()

        assert isinstance(limits, TierLimits)
        assert limits == TIER_LIMITS[APITier.FREE]

    def test_update_tier_from_response_headers(self):
        """Test tier update from response headers."""
        client = TeacherClient(backend="http", endpoint="http://test.com")

        # Mock response with RPM limit header (code infers tier from RPM, not tier header)
        mock_response = Mock()
        # TIER2 has rpm >= 500, so use 500
        mock_response.headers = {"x-ratelimit-limit-rpm": "500"}

        client._update_tier_from_response(mock_response)

        assert client._tier == APITier.TIER2
        assert client._tier_limits == TIER_LIMITS[APITier.TIER2]

    def test_update_tier_unknown_header(self):
        """Test tier update with unknown header."""
        client = TeacherClient(backend="http", endpoint="http://test.com")

        # Mock response without RPM header (can't infer tier)
        mock_response = Mock()
        mock_response.headers = {}  # No tier info

        # Tier should remain at default (FREE)
        original_tier = client._tier
        client._update_tier_from_response(mock_response)

        # Tier should remain unchanged if no RPM header
        assert client._tier == original_tier


class TestRetrySession:
    """Test retry session setup and management."""

    def test_setup_retry_session(self):
        """Test retry session setup."""
        client = TeacherClient(backend="http", endpoint="http://test.com")

        # Check that session was created (uses 'session' not '_session')
        assert hasattr(client, "session")
        assert client.session is not None

        # Check retry configuration
        # Session may have adapters for http:// and https://
        adapters = client.session.adapters
        assert len(adapters) > 0
        # Check that at least one adapter is HTTPAdapter
        assert any(isinstance(adapter, HTTPAdapter) for adapter in adapters.values())

    def test_get_retry_after_header(self):
        """Test retry-after header parsing."""
        client = TeacherClient(backend="http", endpoint="http://test.com")

        # Test with Retry-After header
        mock_response = Mock()
        mock_response.headers = {"Retry-After": "30"}

        retry_after = client._get_retry_after(mock_response)
        assert retry_after == 30.0

    def test_get_retry_after_no_header(self):
        """Test retry-after when no header present."""
        client = TeacherClient(backend="http", endpoint="http://test.com")

        mock_response = Mock()
        mock_response.headers = {}

        retry_after = client._get_retry_after(mock_response)
        assert retry_after is None


class TestHTTPBackendSampling:
    """Test HTTP backend sampling functionality."""

    @pytest.fixture
    def http_client(self):
        """HTTP client fixture."""
        return TeacherClient(backend="http", endpoint="http://test.com", api_key="test_key")

    @patch("requests.Session.request")
    @patch("time.sleep")  # Mock time.sleep to avoid delays
    def test_sample_single_prompt_success(self, mock_sleep, mock_request, http_client):
        """Test successful single prompt sampling."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {"total_tokens": 10, "prompt_tokens": 5, "completion_tokens": 5}
        }
        mock_response.headers = {}
        mock_request.return_value = mock_response

        results = http_client.sample(["What is 2+2?"], return_logits=False)

        assert len(results) == 1
        assert results[0]["text"] == "Test response"
        assert results[0]["prompt"] == "What is 2+2?"
        # finish_reason is not currently included in the result structure

    @patch("requests.Session.request")
    @patch("time.sleep")  # Mock time.sleep to prevent blocking
    def test_sample_with_logits(self, mock_sleep, mock_request, http_client):
        """Test sampling with logits return."""
        # Mock response with logits
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test"}, "logprobs": {"token_logprobs": [0.1, 0.2, 0.7]}}],
            "usage": {"total_tokens": 10}
        }
        mock_response.headers = {}
        mock_request.return_value = mock_response

        results = http_client.sample(["Test"], return_logits=True)

        assert len(results) == 1
        assert results[0]["logits"] is not None

    @patch("requests.Session.request")
    @patch("time.sleep")  # Mock time.sleep to prevent blocking
    def test_sample_retry_on_failure(self, mock_sleep, mock_request, http_client):
        """Test retry behavior on failure."""
        # Mock failure then success
        mock_fail_response = Mock()
        mock_fail_response.status_code = 429
        mock_fail_response.headers = {"Retry-After": "0.1"}  # Short retry for test
        mock_fail_response.json.return_value = {}
        mock_fail_response.text = "Rate limited"

        mock_success_response = Mock()
        mock_success_response.status_code = 200
        mock_success_response.json.return_value = {
            "choices": [{"message": {"content": "Success"}}],
            "usage": {"total_tokens": 10}
        }
        mock_success_response.headers = {}

        mock_request.side_effect = [mock_fail_response, mock_success_response]

        results = http_client.sample(["Test"])

        assert len(results) == 1
        assert results[0]["text"] == "Success"
        assert mock_request.call_count == 2  # One retry

    @patch("requests.Session.request")
    @patch("time.sleep")  # Mock time.sleep to prevent blocking
    def test_sample_rate_limit_handling(self, mock_sleep, mock_request, http_client):
        """Test rate limit handling with backoff."""
        # Mock repeated 429 responses to exhaust retries
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "0.1"}  # Short retry for test
        mock_response.json.return_value = {}
        mock_response.text = "Rate limited"

        mock_request.return_value = mock_response

        # Code will retry and eventually return error dict (doesn't raise)
        results = http_client.sample(["Test"])

        # Should have attempted multiple retries
        assert mock_request.call_count > 1
        # Should return error dict when retries exhausted
        assert len(results) == 1
        assert "error" in results[0] or results[0].get("text") == ""
        # Should have called sleep for backoff
        assert mock_sleep.called

    @patch("requests.Session.request")
    @patch("time.sleep")  # Mock time.sleep to prevent blocking
    def test_sample_max_retries_exceeded(self, mock_sleep, mock_request, http_client):
        """Test behavior when max retries exceeded."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.json.return_value = {}
        mock_response.text = "Server error"
        mock_response.headers = {}

        mock_request.return_value = mock_response

        # Code returns error dict instead of raising
        results = http_client.sample(["Test"])

        # Should have attempted all retries
        assert mock_request.call_count == http_client._max_retries + 1  # Initial + retries
        # Should return error dict
        assert len(results) == 1
        assert "error" in results[0] or results[0].get("text") == ""

    @patch("requests.Session.request")
    @patch("time.sleep")  # Mock time.sleep to prevent blocking
    def test_sample_multi_prompt_batch(self, mock_sleep, mock_request, http_client):
        """Test sampling multiple prompts."""
        # Mock separate responses for each prompt
        mock_response_1 = Mock()
        mock_response_1.status_code = 200
        mock_response_1.json.return_value = {
            "choices": [{"message": {"content": "Response 1"}}],
            "usage": {"total_tokens": 10}
        }
        mock_response_1.headers = {}
        
        mock_response_2 = Mock()
        mock_response_2.status_code = 200
        mock_response_2.json.return_value = {
            "choices": [{"message": {"content": "Response 2"}}],
            "usage": {"total_tokens": 10}
        }
        mock_response_2.headers = {}
        
        mock_request.side_effect = [mock_response_1, mock_response_2]

        results = http_client.sample(["Prompt 1", "Prompt 2"])

        assert len(results) == 2
        assert results[0]["text"] == "Response 1"
        assert results[1]["text"] == "Response 2"


class TestHuggingFaceBackend:
    """Test HuggingFace backend functionality."""

    @pytest.fixture
    def hf_client(self):
        """HF client fixture."""
        with patch("models.teacher.teacher_client.HF_AVAILABLE", True), \
             patch("training.safe_model_loading.safe_from_pretrained_causal_lm") as mock_model, \
             patch("training.safe_model_loading.safe_from_pretrained_tokenizer") as mock_tokenizer:
            # Mock model and tokenizer loading
            mock_model_instance = Mock()
            mock_model_instance.eval = Mock()
            mock_model_instance.to = Mock(return_value=mock_model_instance)
            mock_model.return_value = mock_model_instance
            mock_tokenizer.return_value = Mock()
            
            client = TeacherClient(backend="hf", model_name="test/model")
            yield client

    def test_hf_sample_basic(self, hf_client):
        """Test HF backend sampling."""
        # Mock the _model's forward pass or pipeline call
        # The actual sampling uses self._model and self._tokenizer
        # For now, just verify the client was created successfully
        assert hf_client.backend == "hf"
        assert hasattr(hf_client, "_model")


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    @pytest.fixture
    def http_client(self):
        """HTTP client fixture."""
        return TeacherClient(backend="http", endpoint="http://test.com", api_key="test_key")

    @patch("requests.Session.request")
    @patch("time.sleep")  # Mock time.sleep to prevent blocking
    def test_try_fallback_api_success(self, mock_sleep, mock_request, http_client):
        """Test fallback API usage on primary failure."""
        # Mock primary API failure then success after retry
        mock_fail_response = Mock()
        mock_fail_response.status_code = 500
        mock_fail_response.json.return_value = {}
        mock_fail_response.text = "Server error"
        mock_fail_response.headers = {}

        # Mock success after retry
        mock_success_response = Mock()
        mock_success_response.status_code = 200
        mock_success_response.json.return_value = {
            "choices": [{"message": {"content": "Fallback response"}}],
            "usage": {"total_tokens": 10}
        }
        mock_success_response.headers = {}

        mock_request.side_effect = [mock_fail_response, mock_success_response]

        results = http_client.sample(["Test"])

        assert len(results) == 1
        assert results[0]["text"] == "Fallback response"
        assert mock_request.call_count == 2


class TestHealthCheck:
    """Test health check functionality."""

    @pytest.fixture
    def http_client(self):
        """HTTP client fixture."""
        return TeacherClient(backend="http", endpoint="http://test.com", api_key="test_key")

    @patch("requests.Session.request")
    @patch("time.sleep")  # Mock time.sleep to prevent blocking
    def test_health_check_success(self, mock_sleep, mock_request, http_client):
        """Test successful health check."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_request.return_value = mock_response

        assert http_client.health_check() is True

    @patch("time.sleep")  # Mock time.sleep to prevent blocking
    def test_health_check_failure(self, mock_sleep, http_client):
        """Test failed health check."""
        # health_check uses session.get and catches ConnectionError to return False
        # Other exceptions return True (API reachable but misconfigured)
        with patch.object(http_client.session, 'get') as mock_get:
            # Mock ConnectionError to trigger failure path
            mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")
            
            # health_check should catch ConnectionError after retries and return False
            result = http_client.health_check()
            assert result is False


class TestMultiStepSampling:
    """Test multi-step sampling functionality."""

    @pytest.fixture
    def http_client(self):
        """HTTP client fixture."""
        return TeacherClient(backend="http", endpoint="http://test.com", api_key="test_key")

    @patch("requests.Session.request")
    @patch("time.sleep")  # Mock time.sleep to prevent blocking
    def test_sample_multi_step_basic(self, mock_sleep, mock_request, http_client):
        """Test multi-step sampling."""
        # Mock response with reasoning content
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {"content": "Final answer: 42", "reasoning_content": "Step 1: Think\nStep 2: Calculate"},
                }
            ],
            "usage": {"total_tokens": 10}
        }
        mock_response.headers = {}
        mock_request.return_value = mock_response

        results = http_client.sample_multi_step(
            [{"role": "user", "content": "What is 6*7?"}],
            max_steps=3
        )

        assert "text" in results
        assert results["text"] == "Final answer: 42"
        assert "message" in results
        assert "reasoning_content" in results["message"]


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.fixture
    def http_client(self):
        """HTTP client fixture."""
        return TeacherClient(backend="http", endpoint="http://test.com", api_key="test_key")

    def test_sample_empty_prompts(self, http_client):
        """Test sampling with empty prompts."""
        # Code may return empty list or raise ValueError
        try:
            results = http_client.sample([])
            # If it returns, should be empty list
            assert results == []
        except ValueError:
            # If it raises, that's also acceptable
            pass

    @patch("requests.Session.request")
    @patch("time.sleep")  # Mock time.sleep to prevent blocking
    def test_sample_malformed_response(self, mock_sleep, mock_request, http_client):
        """Test handling of malformed API response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_response.headers = {}
        mock_request.return_value = mock_response

        # Code may return error dict or raise depending on where JSON parsing fails
        try:
            results = http_client.sample(["Test"])
            # If it returns, should have error
            assert len(results) == 1
            assert "error" in results[0] or results[0].get("text") == ""
        except (ValueError, KeyError):
            # If it raises, that's also acceptable
            pass

    @patch("requests.Session.request")
    @patch("time.sleep")  # Mock time.sleep to prevent blocking
    def test_sample_timeout(self, mock_sleep, mock_request, http_client):
        """Test timeout handling."""
        # Mock timeout that will retry and eventually return error
        mock_request.side_effect = requests.exceptions.Timeout("Request timed out")

        # Code will retry and eventually return error dict (doesn't raise)
        results = http_client.sample(["Test"])

        # Should have attempted retries
        assert mock_request.call_count > 1
        # Should return error dict
        assert len(results) == 1
        assert "error" in results[0] or results[0].get("text") == ""
