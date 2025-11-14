"""
Tests for models/teacher/teacher_client.py - Teacher client functionality.

Tests HTTP API client, HuggingFace integration, retry logic,
tier detection, circuit breakers, and response parsing.
"""
# @author: @darianrosebrook

from unittest.mock import Mock, patch

import pytest
import requests

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
            assert hasattr(limits, "rpm_limit")
            assert hasattr(limits, "tpm_limit")
            assert hasattr(limits, "daily_limit")


class TestTeacherClientInitialization:
    """Test teacher client initialization."""

    def test_init_http_backend(self):
        """Test HTTP backend initialization."""
        client = TeacherClient(backend="http", endpoint="http://test.com", api_key="test_key")

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

    @patch.dict("os.environ", {"OPENAI_API_KEY": "env_api_key"})
    def test_init_api_key_from_env(self):
        """Test API key loading from environment."""
        client = TeacherClient(backend="http", endpoint="http://test.com")

        assert client.api_key == "env_api_key"

    @patch("models.teacher.teacher_client.HF_AVAILABLE", True)
    @patch("models.teacher.teacher_client.AutoTokenizer")
    @patch("models.teacher.teacher_client.AutoModelForCausalLM")
    @patch("models.teacher.teacher_client.pipeline")
    def test_init_hf_backend(self, mock_pipeline, mock_model, mock_tokenizer):
        """Test HuggingFace backend initialization."""
        mock_pipeline.return_value = Mock()

        client = TeacherClient(backend="hf", model_name="test/model")

        assert client.backend == "hf"
        mock_pipeline.assert_called_once()

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
        client = TeacherClient.from_endpoint("http://test.com", api_key="test_key")

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
    @patch("models.teacher.teacher_client.AutoTokenizer")
    @patch("models.teacher.teacher_client.AutoModelForCausalLM")
    @patch("models.teacher.teacher_client.pipeline")
    def test_from_hf_basic(self, mock_pipeline, mock_model, mock_tokenizer):
        """Test from_hf factory method."""
        mock_pipeline.return_value = Mock()

        client = TeacherClient.from_hf("microsoft/phi-2", device="cpu")

        assert isinstance(client, TeacherClient)
        assert client.backend == "hf"


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

        # Mock response with tier header
        mock_response = Mock()
        mock_response.headers = {"x-ratelimit-tier": "tier2"}

        client._update_tier_from_response(mock_response)

        assert client._tier == APITier.TIER2
        assert client._tier_limits == TIER_LIMITS[APITier.TIER2]

    def test_update_tier_unknown_header(self):
        """Test tier update with unknown header."""
        client = TeacherClient(backend="http", endpoint="http://test.com")

        mock_response = Mock()
        mock_response.headers = {"x-ratelimit-tier": "unknown_tier"}

        client._update_tier_from_response(mock_response)

        assert client._tier == APITier.UNKNOWN


class TestRetrySession:
    """Test retry session setup and management."""

    def test_setup_retry_session(self):
        """Test retry session setup."""
        client = TeacherClient(backend="http", endpoint="http://test.com")

        # Check that session was created
        assert hasattr(client, "_session")
        assert client._session is not None

        # Check retry configuration
        adapter = client._session.adapters["https://"]
        assert isinstance(adapter, HTTPAdapter)

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
    def test_sample_single_prompt_success(self, mock_request, http_client):
        """Test successful single prompt sampling."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "Test response"}}]}
        mock_response.headers = {}
        mock_request.return_value = mock_response

        results = http_client.sample(["What is 2+2?"], return_logits=False)

        assert len(results) == 1
        assert results[0]["text"] == "Test response"
        assert "finish_reason" in results[0]

    @patch("requests.Session.request")
    def test_sample_with_logits(self, mock_request, http_client):
        """Test sampling with logits return."""
        # Mock response with logits
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test"}}],
            "logits": [[0.1, 0.2, 0.7]],
        }
        mock_response.headers = {}
        mock_request.return_value = mock_response

        results = http_client.sample(["Test"], return_logits=True)

        assert len(results) == 1
        assert results[0]["logits"] is not None

    @patch("requests.Session.request")
    def test_sample_retry_on_failure(self, mock_request, http_client):
        """Test retry behavior on failure."""
        # Mock failure then success
        mock_fail_response = Mock()
        mock_fail_response.status_code = 429
        mock_fail_response.raise_for_status.side_effect = requests.exceptions.HTTPError()

        mock_success_response = Mock()
        mock_success_response.status_code = 200
        mock_success_response.json.return_value = {"choices": [{"message": {"content": "Success"}}]}
        mock_success_response.headers = {}

        mock_request.side_effect = [mock_fail_response, mock_success_response]

        results = http_client.sample(["Test"])

        assert len(results) == 1
        assert results[0]["text"] == "Success"
        assert mock_request.call_count == 2  # One retry

    @patch("requests.Session.request")
    def test_sample_rate_limit_handling(self, mock_request, http_client):
        """Test rate limit handling with backoff."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "2"}
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError()

        mock_request.return_value = mock_response

        with patch("time.sleep") as mock_sleep:
            with pytest.raises(requests.exceptions.HTTPError):
                http_client.sample(["Test"])

            # Should sleep for retry-after duration
            mock_sleep.assert_called_with(2)

    @patch("requests.Session.request")
    def test_sample_max_retries_exceeded(self, mock_request, http_client):
        """Test behavior when max retries exceeded."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError()

        mock_request.return_value = mock_response

        with pytest.raises(requests.exceptions.HTTPError):
            http_client.sample(["Test"])

        assert mock_request.call_count == http_client._max_retries

    @patch("requests.Session.request")
    def test_sample_multi_prompt_batch(self, mock_request, http_client):
        """Test sampling multiple prompts."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": "Response 1"}},
                {"message": {"content": "Response 2"}},
            ]
        }
        mock_response.headers = {}
        mock_request.return_value = mock_response

        results = http_client.sample(["Prompt 1", "Prompt 2"])

        assert len(results) == 2
        assert results[0]["text"] == "Response 1"
        assert results[1]["text"] == "Response 2"


class TestHuggingFaceBackend:
    """Test HuggingFace backend functionality."""

    @pytest.fixture
    @patch("models.teacher.teacher_client.HF_AVAILABLE", True)
    @patch("models.teacher.teacher_client.pipeline")
    def hf_client(self, mock_pipeline):
        """HF client fixture."""
        mock_pipeline.return_value = Mock()
        return TeacherClient(backend="hf", model_name="test/model")

    @patch("models.teacher.teacher_client.pipeline")
    def test_hf_sample_basic(self, mock_pipeline, hf_client):
        """Test HF backend sampling."""
        # Mock pipeline response
        mock_pipe = Mock()
        mock_pipe.return_value = [{"generated_text": "Test response"}]
        mock_pipeline.return_value = mock_pipe

        results = hf_client.sample(["Test prompt"])

        assert len(results) == 1
        assert results[0]["text"] == "Test response"


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    @pytest.fixture
    def http_client(self):
        """HTTP client fixture."""
        return TeacherClient(backend="http", endpoint="http://test.com", api_key="test_key")

    @patch("requests.Session.request")
    def test_try_fallback_api_success(self, mock_request, http_client):
        """Test fallback API usage on primary failure."""
        # Mock primary API failure
        mock_fail_response = Mock()
        mock_fail_response.status_code = 500
        mock_fail_response.raise_for_status.side_effect = requests.exceptions.HTTPError()

        # Mock fallback API success
        mock_success_response = Mock()
        mock_success_response.status_code = 200
        mock_success_response.json.return_value = {
            "choices": [{"message": {"content": "Fallback response"}}]
        }
        mock_success_response.headers = {}

        mock_request.side_effect = [mock_fail_response, mock_success_response]

        # Set fallback endpoint
        http_client.fallback_endpoints = ["http://fallback.com"]

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
    def test_health_check_success(self, mock_request, http_client):
        """Test successful health check."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_request.return_value = mock_response

        assert http_client.health_check() is True

    @patch("requests.Session.request")
    def test_health_check_failure(self, mock_request, http_client):
        """Test failed health check."""
        mock_request.side_effect = requests.exceptions.RequestException()

        assert http_client.health_check() is False


class TestMultiStepSampling:
    """Test multi-step sampling functionality."""

    @pytest.fixture
    def http_client(self):
        """HTTP client fixture."""
        return TeacherClient(backend="http", endpoint="http://test.com", api_key="test_key")

    @patch("requests.Session.request")
    def test_sample_multi_step_basic(self, mock_request, http_client):
        """Test multi-step sampling."""
        # Mock response with reasoning content
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {"content": "Final answer: 42"},
                    "reasoning_content": ["Step 1: Think", "Step 2: Calculate"],
                }
            ]
        }
        mock_response.headers = {}
        mock_request.return_value = mock_response

        results = http_client.sample_multi_step(["What is 6*7?"], max_steps=3)

        assert len(results) == 1
        assert results[0]["text"] == "Final answer: 42"
        assert "reasoning_content" in results[0]
        assert len(results[0]["reasoning_content"]) == 2


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.fixture
    def http_client(self):
        """HTTP client fixture."""
        return TeacherClient(backend="http", endpoint="http://test.com", api_key="test_key")

    def test_sample_empty_prompts(self, http_client):
        """Test sampling with empty prompts."""
        with pytest.raises(ValueError):
            http_client.sample([])

    @patch("requests.Session.request")
    def test_sample_malformed_response(self, mock_request, http_client):
        """Test handling of malformed API response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_request.return_value = mock_response

        with pytest.raises(ValueError):
            http_client.sample(["Test"])

    @patch("requests.Session.request")
    def test_sample_timeout(self, mock_request, http_client):
        """Test timeout handling."""
        mock_request.side_effect = requests.exceptions.Timeout()

        with pytest.raises(requests.exceptions.Timeout):
            http_client.sample(["Test"])
