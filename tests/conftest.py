"""
Pytest configuration and fixtures for ClaudeSqueeze tests.
"""

import pytest
from unittest.mock import MagicMock, Mock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from path_compression import PathCompressionEngine, CompressionMapping
from llm_proxy import CompressionEngine, MetricsCollector, LLMProxyHandler


@pytest.fixture
def compression_mapping():
    """Create a fresh CompressionMapping instance."""
    return CompressionMapping()


@pytest.fixture
def path_compression_engine():
    """Create a fresh PathCompressionEngine instance."""
    return PathCompressionEngine()


@pytest.fixture
def compression_engine_low():
    """Create a CompressionEngine with low compression level."""
    return CompressionEngine(level="low")


@pytest.fixture
def compression_engine_medium():
    """Create a CompressionEngine with medium compression level."""
    return CompressionEngine(level="medium")


@pytest.fixture
def compression_engine_high():
    """Create a CompressionEngine with high compression level."""
    return CompressionEngine(level="high")


@pytest.fixture
def compression_engine_with_path():
    """Create a CompressionEngine with path compression enabled."""
    return CompressionEngine(level="high", enable_path_compression=True)


@pytest.fixture
def metrics_collector():
    """Create a fresh MetricsCollector instance."""
    return MetricsCollector()


@pytest.fixture
def sample_messages():
    """Return sample messages in OpenAI/Anthropic format."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Please could you help me with this task?"},
        {"role": "assistant", "content": "I would be happy to help you."},
        {"role": "user", "content": "What is the weather today?"},
    ]


@pytest.fixture
def sample_messages_with_content_blocks():
    """Return sample messages with Anthropic content block format."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please analyze this image"},
                {"type": "image", "source": {"type": "base64", "data": "..."}}
            ]
        }
    ]


@pytest.fixture
def mock_request_handler():
    """Create a mock HTTP request handler."""
    handler = MagicMock(spec=LLMProxyHandler)
    handler.headers = {}
    handler.path = "/v1/messages"
    handler.rfile = MagicMock()
    handler.wfile = MagicMock()
    return handler


@pytest.fixture
def sample_stats():
    """Return sample compression stats."""
    return {
        "original_tokens": 1000,
        "compressed_tokens": 750,
        "tokens_saved": 250,
        "reduction_pct": 25.0,
    }


@pytest.fixture
def unix_path_samples():
    """Return sample Unix paths for testing."""
    return {
        "simple": "/home/user/file.txt",
        "long": "/home/user/projects/myapp/src/components/Button.tsx",
        "with_dots": "/home/user.name/projects/test.file.py",
        "multiple_dirs": "/a/b/c/d/e/f/g/h/i/j/k/l/m/n/o/p/file.txt",
    }


@pytest.fixture
def windows_path_samples():
    """Return sample Windows paths for testing."""
    return {
        "simple": r"C:\Users\user\file.txt",
        "long": r"C:\Users\user\Projects\MyApp\src\components\Button.tsx",
        "with_spaces": r"C:\Program Files\My App\file.exe",
    }


@pytest.fixture
def url_samples():
    """Return sample URLs for testing."""
    return {
        "simple": "https://example.com",
        "with_path": "https://api.example.com/v1/users/123",
        "long": "https://raw.githubusercontent.com/user/repo/main/src/file.py",
    }


@pytest.fixture
def python_traceback():
    """Return a sample Python traceback for testing."""
    return '''Traceback (most recent call last):
  File "/home/user/projects/myapp/src/main.py", line 42, in <module>
    result = process_data(data)
  File "/home/user/projects/myapp/src/utils.py", line 15, in process_data
    return transform(item)
  File "/home/user/projects/myapp/src/transform.py", line 88, in transform
    raise ValueError("Invalid data")
ValueError: Invalid data
'''


@pytest.fixture
def js_traceback():
    """Return a sample JavaScript traceback for testing."""
    return '''Error: Something went wrong
    at processData (/home/user/projects/myapp/src/main.js:10:15)
    at transform (/home/user/projects/myapp/src/utils.js:25:10)
    at Object.<anonymous> (/home/user/projects/myapp/src/app.js:5:1)
    at Module._compile (internal/modules/cjs/loader.js:999:30)
'''


@pytest.fixture
def mock_config():
    """Return a sample configuration dictionary."""
    return {
        "target_api": {"url": "https://api.anthropic.com"},
        "compression": {"level": "high", "passthrough_mode": False},
        "server": {"port": 8080, "host": "127.0.0.1"},
        "logging": {"level": "info", "log_compression": True},
        "cache_control": {"enabled": True, "cache_roles": ["system"], "cache_first_n_messages": 2},
    }


@pytest.fixture
def mock_response():
    """Create a mock requests Response object."""
    response = MagicMock()
    response.status_code = 200
    response.headers = {"Content-Type": "application/json"}
    response.content = b'{"id": "test-123", "choices": [{"message": {"content": "Hello"}}]}'
    response.iter_content.return_value = [b'data: {"chunk": 1}\n\n', b'data: {"chunk": 2}\n\n']
    return response
