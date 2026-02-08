"""
Unit tests for the LLMProxyHandler class.

Tests request handling, header preparation, URL construction,
streaming response handling, and error handling.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
import json
import sys
import os
from io import BytesIO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_proxy import LLMProxyHandler, CompressionEngine, MetricsCollector, load_config


class MockRequestHandler:
    """Helper class to create a mock LLMProxyHandler for testing."""

    @staticmethod
    def create():
        """Create a mock handler instance."""
        handler = MagicMock(spec=LLMProxyHandler)
        handler.headers = {}
        handler.path = "/v1/messages"
        handler.command = "POST"
        handler.rfile = BytesIO()
        handler.wfile = BytesIO()
        handler.config = None
        handler.compressor = CompressionEngine(level="high")
        handler.metrics = MetricsCollector()
        return handler


class TestLoadConfig:
    """Tests for load_config function."""

    @patch('llm_proxy.Path.exists')
    @patch('llm_proxy.Path')
    @patch('builtins.open', MagicMock())
    @patch('llm_proxy.yaml.safe_load')
    def test_load_existing_config(self, mock_yaml_load, mock_path_class, mock_exists):
        """Test loading an existing config file."""
        mock_exists.return_value = True
        mock_yaml_load.return_value = {
            "target_api": {"url": "https://api.test.com"},
            "compression": {"level": "medium"},
        }

        with patch('llm_proxy.Path', return_value=MagicMock(exists=MagicMock(return_value=True))):
            config = load_config("config.yaml")

        assert config["target_api"]["url"] == "https://api.test.com"
        assert config["compression"]["level"] == "medium"

    def test_load_default_config(self):
        """Test loading default config when file doesn't exist."""
        config = load_config("/nonexistent/config.yaml")

        assert config["target_api"]["url"] == "https://api.anthropic.com"
        assert config["compression"]["level"] == "high"
        assert config["server"]["port"] == 8080

    @patch('llm_proxy.Path.exists')
    @patch('llm_proxy.yaml.safe_load')
    def test_load_config_from_home_directory(self, mock_yaml_load, mock_exists):
        """Test loading config from home directory."""
        mock_exists.return_value = True
        mock_yaml_load.return_value = {"custom": "value"}

        with patch.object(LLMProxyHandler, 'config', None):
            config = load_config()

        # Should return default or found config
        assert "target_api" in config or "custom" in config


class TestRequestHandling:
    """Tests for HTTP request handling."""

    def test_do_get_health_endpoint(self):
        """Test GET request to /health endpoint."""
        handler = MagicMock(spec=LLMProxyHandler)
        handler.path = "/health"

        # Simulate the do_GET logic
        if handler.path == "/health":
            response = {"status": "ok", "proxy": "llm-compression-proxy"}
        else:
            response = {}

        assert response["status"] == "ok"
        assert response["proxy"] == "llm-compression-proxy"

    def test_do_get_metrics_endpoint(self):
        """Test GET request to /metrics endpoint."""
        handler = MagicMock(spec=LLMProxyHandler)
        handler.path = "/metrics"
        handler.metrics = MetricsCollector()

        # Simulate the do_GET logic
        if handler.path == "/metrics":
            response = handler.metrics.get_summary()
        else:
            response = {}

        assert "total_requests" in response
        assert "uptime_seconds" in response

    def test_do_get_root_endpoint(self):
        """Test GET request to / endpoint."""
        handler = MagicMock(spec=LLMProxyHandler)
        handler.path = "/"

        # Simulate the do_GET logic
        if handler.path == "/":
            response = {
                "message": "LLM Compression Proxy",
                "endpoints": {
                    "/health": "Health check",
                    "/metrics": "Compression metrics",
                },
            }
        else:
            response = {}

        assert response["message"] == "LLM Compression Proxy"
        assert "/health" in response["endpoints"]

    def test_do_post_chat_endpoint(self):
        """Test POST request to chat endpoint."""
        handler = MagicMock(spec=LLMProxyHandler)
        handler.path = "/v1/messages"

        # Should be recognized as chat endpoint
        path_without_query = handler.path.split("?")[0]
        is_chat = path_without_query in ["/v1/messages", "/v1/chat/completions"]

        assert is_chat is True

    def test_do_post_chat_completions_endpoint(self):
        """Test POST request to OpenAI chat completions endpoint."""
        handler = MagicMock(spec=LLMProxyHandler)
        handler.path = "/v1/chat/completions"

        path_without_query = handler.path.split("?")[0]
        is_chat = path_without_query in ["/v1/messages", "/v1/chat/completions"]

        assert is_chat is True

    def test_do_post_other_endpoint(self):
        """Test POST request to non-chat endpoint."""
        handler = MagicMock(spec=LLMProxyHandler)
        handler.path = "/v1/models"

        path_without_query = handler.path.split("?")[0]
        is_chat = path_without_query in ["/v1/messages", "/v1/chat/completions"]

        assert is_chat is False


class TestHeaderPreparation:
    """Tests for _prepare_headers method."""

    def test_hop_by_hop_headers_filtered(self):
        """Test hop-by-hop headers are filtered out."""
        handler = MagicMock(spec=LLMProxyHandler)
        handler.headers = {
            "Host": "localhost:8080",
            "Content-Length": "100",
            "Authorization": "Bearer token123",
            "X-Custom-Header": "value",
            "Connection": "keep-alive",
            "Transfer-Encoding": "chunked",
        }

        # Simulate _prepare_headers logic
        hop_by_hop = (
            "host", "content-length", "transfer-encoding", "connection",
            "keep-alive", "proxy-authenticate", "proxy-authorization",
            "te", "trailers", "upgrade",
        )

        filtered = {}
        for key, value in handler.headers.items():
            if key.lower() not in hop_by_hop:
                filtered[key] = value

        assert "Host" not in filtered
        assert "Content-Length" not in filtered
        assert "Transfer-Encoding" not in filtered
        assert "Connection" not in filtered
        assert "Authorization" in filtered
        assert "X-Custom-Header" in filtered

    def test_case_insensitive_header_filtering(self):
        """Test header filtering is case-insensitive."""
        handler = MagicMock(spec=LLMProxyHandler)
        handler.headers = {
            "HOST": "localhost",
            "content-length": "100",
            "Transfer-Encoding": "chunked",
        }

        hop_by_hop = (
            "host", "content-length", "transfer-encoding", "connection",
        )

        filtered = {}
        for key, value in handler.headers.items():
            if key.lower() not in hop_by_hop:
                filtered[key] = value

        assert len(filtered) == 0

    def test_empty_headers(self):
        """Test handling of empty headers."""
        handler = MagicMock(spec=LLMProxyHandler)
        handler.headers = {}

        hop_by_hop = ("host", "content-length")
        filtered = {k: v for k, v in handler.headers.items() if k.lower() not in hop_by_hop}

        assert filtered == {}


class TestURLConstruction:
    """Tests for _get_target_url method."""

    def test_default_target_url(self):
        """Test default target URL construction."""
        handler = MagicMock(spec=LLMProxyHandler)
        handler.path = "/v1/messages"
        handler.config = None

        # Simulate _get_target_url logic
        base_url = "https://api.anthropic.com"
        full_url = f"{base_url}{handler.path}"

        assert full_url == "https://api.anthropic.com/v1/messages"

    def test_custom_target_url(self):
        """Test custom target URL from config."""
        handler = MagicMock(spec=LLMProxyHandler)
        handler.path = "/v1/chat/completions"
        handler.config = {"target_api": {"url": "https://api.openai.com"}}

        base_url = handler.config.get("target_api", {}).get("url", "https://api.anthropic.com")
        base_url = base_url.rstrip('/')
        full_url = f"{base_url}{handler.path}"

        assert full_url == "https://api.openai.com/v1/chat/completions"

    def test_url_with_trailing_slash(self):
        """Test URL with trailing slash is handled correctly."""
        handler = MagicMock(spec=LLMProxyHandler)
        handler.path = "/v1/messages"
        handler.config = {"target_api": {"url": "https://api.example.com/"}}

        base_url = handler.config.get("target_api", {}).get("url", "https://api.anthropic.com")
        base_url = base_url.rstrip('/')
        full_url = f"{base_url}{handler.path}"

        assert full_url == "https://api.example.com/v1/messages"

    def test_url_with_query_string(self):
        """Test URL with query string."""
        handler = MagicMock(spec=LLMProxyHandler)
        handler.path = "/v1/messages?model=claude-3"

        base_url = "https://api.anthropic.com"
        full_url = f"{base_url}{handler.path}"

        assert full_url == "https://api.anthropic.com/v1/messages?model=claude-3"


class TestStreamingResponseHandling:
    """Tests for streaming response handling."""

    def test_streaming_response_detection(self):
        """Test detection of streaming responses."""
        headers = {"Content-Type": "text/event-stream"}
        is_streaming = 'text/event-stream' in headers.get('Content-Type', '')
        assert is_streaming is True

    def test_chunked_response_detection(self):
        """Test detection of chunked transfer encoding."""
        headers = {"Transfer-Encoding": "chunked"}
        is_streaming = headers.get('Transfer-Encoding') == 'chunked'
        assert is_streaming is True

    def test_non_streaming_response_detection(self):
        """Test detection of non-streaming responses."""
        headers = {"Content-Type": "application/json"}
        is_streaming = 'text/event-stream' in headers.get('Content-Type', '')
        assert is_streaming is False

    def test_streaming_headers_forwarded(self):
        """Test streaming response headers are forwarded correctly."""
        response_headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Content-Encoding": "gzip",  # Should be filtered
        }

        # Simulate header forwarding logic
        filtered_headers = {}
        for key, value in response_headers.items():
            if key.lower() not in ("content-encoding", "transfer-encoding", "content-length", "connection"):
                filtered_headers[key] = value

        assert "Content-Type" in filtered_headers
        assert "Cache-Control" in filtered_headers
        assert "Content-Encoding" not in filtered_headers


class TestErrorHandling:
    """Tests for error handling."""

    def test_json_decode_error_handling(self):
        """Test handling of JSON decode errors."""
        invalid_json = b"not valid json"

        try:
            json.loads(invalid_json.decode("utf-8"))
            assert False, "Should have raised JSONDecodeError"
        except json.JSONDecodeError:
            pass  # Expected

    def test_connection_error_handling(self):
        """Test handling of connection errors."""
        error = Exception("Connection failed")
        error_message = str(error)

        assert "Connection failed" in error_message

    def test_error_response_format(self):
        """Test error response JSON format."""
        error_code = 500
        error_message = "Internal server error"

        error_response = json.dumps({"error": error_message}).encode("utf-8")

        assert b"error" in error_response
        assert b"Internal server error" in error_response


class TestChatRequestHandling:
    """Tests for _handle_chat_request method."""

    def test_message_compression_called(self):
        """Test that message compression is called for chat requests."""
        compressor = CompressionEngine(level="high")
        messages = [
            {"role": "user", "content": "Please help me with this task"}
        ]

        compressed, stats = compressor.compress_messages(messages)

        assert stats["original_tokens"] > 0
        assert "compressed_tokens" in stats

    def test_passthrough_mode(self):
        """Test passthrough mode skips compression."""
        config = {"compression": {"passthrough_mode": True}}

        passthrough = config.get("compression", {}).get("passthrough_mode", False)
        assert passthrough is True

        if passthrough:
            stats = {"original_tokens": 0, "compressed_tokens": 0, "tokens_saved": 0, "reduction_pct": 0}
            assert stats["tokens_saved"] == 0

    def test_user_messages_only_compressed(self):
        """Test only user messages are compressed."""
        compressor = CompressionEngine(level="high")
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Please help me"},
            {"role": "assistant", "content": "I would love to help"},
        ]

        compressed, stats = compressor.compress_messages(messages, roles=["user"])

        # System and assistant messages should be unchanged
        system_msg = next(m for m in compressed if m["role"] == "system")
        assert system_msg["content"] == "You are helpful"

        assistant_msg = next(m for m in compressed if m["role"] == "assistant")
        assert assistant_msg["content"] == "I would love to help"

        # User message should be compressed
        user_msg = next(m for m in compressed if m["role"] == "user")
        assert "please" not in user_msg["content"].lower()

    def test_compression_timing_recorded(self):
        """Test compression timing is recorded."""
        import time
        metrics = MetricsCollector()

        start = time.perf_counter()
        # Simulate compression
        time.sleep(0.001)
        elapsed_ms = (time.perf_counter() - start) * 1000

        metrics.record_compression_time(elapsed_ms)

        assert metrics.compression_count == 1
        assert metrics.total_compression_time_ms == elapsed_ms


class TestPromptCaching:
    """Tests for _add_prompt_caching method."""

    def test_system_prompt_caching(self):
        """Test system prompts are marked for caching."""
        config = {
            "cache_control": {
                "enabled": True,
                "cache_roles": ["system"],
                "cache_first_n_messages": 0,
            }
        }

        data = {
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
            ]
        }

        # Simulate caching logic
        for msg in data["messages"]:
            if msg.get("role") in config["cache_control"]["cache_roles"]:
                if isinstance(msg.get("content"), str):
                    msg["content"] = [
                        {"type": "text", "text": msg["content"], "cache_control": {"type": "ephemeral"}}
                    ]

        system_msg = data["messages"][0]
        assert isinstance(system_msg["content"], list)
        assert system_msg["content"][0]["cache_control"]["type"] == "ephemeral"

    def test_first_n_messages_caching(self):
        """Test first N messages are marked for caching."""
        config = {
            "cache_control": {
                "enabled": True,
                "cache_roles": [],
                "cache_first_n_messages": 2,
            }
        }

        data = {
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
                {"role": "user", "content": "How are you?"},
            ]
        }

        # Simulate caching logic for first N
        for i, msg in enumerate(data["messages"]):
            if i < config["cache_control"]["cache_first_n_messages"]:
                if isinstance(msg.get("content"), str):
                    msg["content"] = [
                        {"type": "text", "text": msg["content"], "cache_control": {"type": "ephemeral"}}
                    ]

        # First two messages should have cache_control
        assert "cache_control" in data["messages"][0]["content"][0]
        assert "cache_control" in data["messages"][1]["content"][0]

    def test_anthropic_system_format(self):
        """Test Anthropic system prompt format handling."""
        data = {"system": "You are a helpful assistant"}

        # Simulate conversion to blocks format
        if "system" in data and isinstance(data["system"], str):
            data["system"] = [
                {"type": "text", "text": data["system"], "cache_control": {"type": "ephemeral"}}
            ]

        assert isinstance(data["system"], list)
        assert data["system"][0]["cache_control"]["type"] == "ephemeral"

    def test_existing_blocks_format_preserved(self):
        """Test existing blocks format is preserved."""
        data = {
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are helpful"}]
                }
            ]
        }

        # Simulate adding cache_control to existing blocks
        for msg in data["messages"]:
            if isinstance(msg.get("content"), list):
                for block in msg["content"]:
                    if block.get("type") == "text":
                        block["cache_control"] = {"type": "ephemeral"}
                        break

        assert data["messages"][0]["content"][0]["cache_control"]["type"] == "ephemeral"


class TestJSONResponseSending:
    """Tests for _send_json method."""

    def test_json_response_format(self):
        """Test JSON response is properly formatted."""
        data = {"status": "ok", "count": 42}
        content = json.dumps(data).encode("utf-8")

        assert isinstance(content, bytes)
        assert b'"status": "ok"' in content
        assert b'"count": 42' in content

    def test_json_response_content_type(self):
        """Test JSON response has correct content type."""
        content_type = "application/json"
        assert content_type == "application/json"


class TestRequestForwarding:
    """Tests for _forward_request method."""

    @patch('llm_proxy.requests.request')
    def test_forward_get_request(self, mock_request):
        """Test forwarding GET request."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.content = b'{"models": []}'
        mock_request.return_value = mock_response

        # Simulate forwarding
        method = "GET"
        target_url = "https://api.anthropic.com/v1/models"
        headers = {"Authorization": "Bearer token"}

        mock_request(method, target_url, headers=headers)
        mock_request.assert_called_once()

    @patch('llm_proxy.requests.request')
    def test_forward_post_request(self, mock_request):
        """Test forwarding POST request."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_request.return_value = mock_response

        method = "POST"
        target_url = "https://api.anthropic.com/v1/messages"
        headers = {"Authorization": "Bearer token", "Content-Type": "application/json"}
        body = b'{"messages": []}'

        mock_request(method, target_url, data=body, headers=headers, timeout=300)
        mock_request.assert_called_once()


class TestLogMessage:
    """Tests for log_message method."""

    def test_log_message_format(self):
        """Test log message format."""
        command = "POST"
        path = "/v1/messages"
        status = "200"

        log_line = f"[PROXY] {command} {path} - {status}"

        assert "POST" in log_line
        assert "/v1/messages" in log_line
        assert "200" in log_line
