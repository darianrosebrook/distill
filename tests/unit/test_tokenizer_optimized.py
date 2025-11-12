"""
Unit tests for optimized tokenizer I/O.

Tests pre-allocated buffers and ring buffers for M-series Apple Silicon optimization.
"""
import pytest
import numpy as np
from unittest.mock import Mock

try:
    from coreml.runtime.tokenizer_optimized import OptimizedTokenizer, RingBuffer, wrap_tokenizer
    TOKENIZER_OPTIMIZED_AVAILABLE = True
except ImportError:
    TOKENIZER_OPTIMIZED_AVAILABLE = False


@pytest.mark.skipif(not TOKENIZER_OPTIMIZED_AVAILABLE, reason="Optimized tokenizer not available")
class TestRingBuffer:
    """Tests for RingBuffer class."""
    
    def test_initialization(self):
        """Test ring buffer initialization."""
        buffer = RingBuffer(size=100)
        
        assert buffer.size == 100
        assert len(buffer.buffer) == 100
        assert buffer.write_pos == 0
        assert buffer.count == 0
    
    def test_write_and_read(self):
        """Test writing and reading tokens."""
        buffer = RingBuffer(size=10)
        
        tokens = [1, 2, 3, 4, 5]
        written = buffer.write(tokens)
        
        assert written == 5
        assert buffer.count == 5
        
        read_tokens = buffer.read(5)
        assert len(read_tokens) == 5
        assert np.array_equal(read_tokens, np.array([1, 2, 3, 4, 5], dtype=np.int32))
    
    def test_wrap_around(self):
        """Test ring buffer wrap-around behavior."""
        buffer = RingBuffer(size=5)
        
        # Write more than buffer size
        tokens = [1, 2, 3, 4, 5, 6, 7]
        written = buffer.write(tokens)
        
        assert written == 7
        assert buffer.count == 5  # Buffer size limits count
        
        # Read should return last 5 tokens (wrapped)
        read_tokens = buffer.read(5)
        assert len(read_tokens) == 5
    
    def test_clear(self):
        """Test clearing buffer."""
        buffer = RingBuffer(size=10)
        
        buffer.write([1, 2, 3])
        assert buffer.count == 3
        
        buffer.clear()
        assert buffer.count == 0
        assert buffer.write_pos == 0


@pytest.mark.skipif(not TOKENIZER_OPTIMIZED_AVAILABLE, reason="Optimized tokenizer not available")
class TestOptimizedTokenizer:
    """Tests for OptimizedTokenizer class."""
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])
        tokenizer.decode = Mock(return_value="decoded text")
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 2
        return tokenizer
    
    def test_initialization(self, mock_tokenizer):
        """Test optimized tokenizer initialization."""
        opt_tokenizer = OptimizedTokenizer(mock_tokenizer, max_seq_length=4096)
        
        assert opt_tokenizer.tokenizer == mock_tokenizer
        assert opt_tokenizer.max_seq_length == 4096
        assert len(opt_tokenizer.input_buffer_np) == 4096
        assert len(opt_tokenizer.ring_buffer.buffer) == 4096
    
    def test_encode_optimized(self, mock_tokenizer):
        """Test optimized encoding."""
        opt_tokenizer = OptimizedTokenizer(mock_tokenizer, max_seq_length=100)
        
        tokens = opt_tokenizer.encode_optimized("test text", return_numpy=True)
        
        assert isinstance(tokens, np.ndarray)
        assert len(tokens) == 5
        mock_tokenizer.encode.assert_called_once()
    
    def test_encode_batch_optimized(self, mock_tokenizer):
        """Test batch encoding."""
        opt_tokenizer = OptimizedTokenizer(mock_tokenizer, max_seq_length=100)
        
        texts = ["text 1", "text 2"]
        batch = opt_tokenizer.encode_batch_optimized(texts, padding=True)
        
        assert batch.shape == (2, 100)
        assert mock_tokenizer.encode.call_count == 2
    
    def test_encode_streaming(self, mock_tokenizer):
        """Test streaming encoding."""
        opt_tokenizer = OptimizedTokenizer(mock_tokenizer, max_seq_length=100)
        
        tokens = list(opt_tokenizer.encode_streaming("test text"))
        
        assert len(tokens) == 5
        assert tokens == [1, 2, 3, 4, 5]
    
    def test_decode_optimized(self, mock_tokenizer):
        """Test optimized decoding."""
        opt_tokenizer = OptimizedTokenizer(mock_tokenizer, max_seq_length=100)
        
        token_ids = np.array([1, 2, 3], dtype=np.int32)
        text = opt_tokenizer.decode_optimized(token_ids)
        
        assert text == "decoded text"
        mock_tokenizer.decode.assert_called_once()
    
    def test_decode_batch_optimized(self, mock_tokenizer):
        """Test batch decoding."""
        opt_tokenizer = OptimizedTokenizer(mock_tokenizer, max_seq_length=100)
        
        batch_token_ids = np.array([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]], dtype=np.int32)
        texts = opt_tokenizer.decode_batch_optimized(batch_token_ids)
        
        assert len(texts) == 2
        assert mock_tokenizer.decode.call_count == 2
    
    def test_stats(self, mock_tokenizer):
        """Test statistics tracking."""
        opt_tokenizer = OptimizedTokenizer(mock_tokenizer, max_seq_length=100)
        
        opt_tokenizer.encode_optimized("test")
        opt_tokenizer.decode_optimized(np.array([1, 2, 3]))
        
        stats = opt_tokenizer.stats()
        
        assert stats["total_encodes"] == 1
        assert stats["total_decodes"] == 1
        assert stats["max_seq_length"] == 100


@pytest.mark.skipif(not TOKENIZER_OPTIMIZED_AVAILABLE, reason="Optimized tokenizer not available")
class TestWrapTokenizer:
    """Tests for wrap_tokenizer convenience function."""
    
    def test_wrap_tokenizer(self):
        """Test wrapping a tokenizer."""
        mock_tokenizer = Mock()
        
        opt_tokenizer = wrap_tokenizer(mock_tokenizer, max_seq_length=2048)
        
        assert isinstance(opt_tokenizer, OptimizedTokenizer)
        assert opt_tokenizer.max_seq_length == 2048

