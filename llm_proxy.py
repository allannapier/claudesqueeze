#!/usr/bin/env python3
"""
LLM API Proxy with Prompt Compression
======================================

Intercepts LLM API calls from Claude Code and other tools,
compresses prompts using token efficiency techniques, then
forwards to the real API.

Usage:
    python llm_proxy.py --port 8080

Then configure Claude Code:
    export ANTHROPIC_BASE_URL=http://localhost:8080
    export ANTHROPIC_API_KEY=your-api-key
    claude
"""

import argparse
import json
import os
import re
import sys
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

try:
    import requests
except ImportError:
    print("Error: requests is required. Install with: pip install requests")
    sys.exit(1)

try:
    import yaml
except ImportError:
    print("Error: pyyaml is required. Install with: pip install pyyaml")
    sys.exit(1)


def load_config(config_path: str = None) -> dict:
    """Load proxy configuration from YAML file."""
    if config_path is None:
        # Look for config in standard locations
        possible_paths = [
            Path("proxy_config.yaml"),
            Path(__file__).parent / "proxy_config.yaml",
            Path.home() / ".llm-proxy" / "config.yaml",
        ]
        for path in possible_paths:
            if path.exists():
                config_path = path
                break

    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            return yaml.safe_load(f)

    # Return default config
    return {
        "target_api": {"url": "https://api.anthropic.com"},
        "compression": {"level": "high"},
        "server": {"port": 8080, "host": "127.0.0.1"},
        "logging": {"level": "info", "log_compression": True},
    }


class CompressionEngine:
    """
    Token compression engine based on research findings.
    Implements the compression_high strategy.
    """

    def __init__(self, level: str = "high", enable_path_compression: bool = False):
        """
        Initialize compression engine.

        Args:
            level: "low", "medium", or "high"
            enable_path_compression: EXPERIMENTAL - compress file paths/URLs
        """
        self.level = level
        self.enable_path_compression = enable_path_compression
        if enable_path_compression:
            try:
                from path_compression import PathCompressionEngine
                self.path_compressor = PathCompressionEngine()
            except ImportError:
                print("[WARN] Path compression module not found, disabling")
                self.enable_path_compression = False
                self.path_compressor = None

    # Filler words and phrases to remove
    FILLER_PATTERNS = [
        r'\bplease\b',
        r'\bcould you\b',
        r'\bwould you\b',
        r'\bkindly\b',
        r'\bi would like to\b',
        r'\bi want to\b',
        r'\bi need to\b',
        r'\bjust\b',
        r'\bactually\b',
        r'\bbasically\b',
        r'\bliterally\b',
        r'\bquite\b',
        r'\brather\b',
        r'\breally\b',
        r'\bvery\b',
        r'\bsomewhat\b',
        r'\bperhaps\b',
        r'\bmaybe\b',
        r'\bin order to\b',
        r'\bdue to the fact that\b',
        r'\bin spite of the fact that\b',
        r'\bwith regard to\b',
        r'\bin the event that\b',
        r'\bat this point in time\b',
        r'\bin the near future\b',
        r'\bfor all intents and purposes\b',
        r'\bi was wondering if\b',
        r'\bi think that\b',
        r'\bit seems that\b',
        r'\bthe thing is\b',
    ]

    # Common phrase abbreviations
    ABBREVIATIONS = {
        'for your information': 'FYI',
        'as soon as possible': 'ASAP',
        'by the way': 'BTW',
        'in my opinion': 'IMO',
        'in my humble opinion': 'IMHO',
        'for example': 'e.g.',
        'that is': 'i.e.',
        'et cetera': 'etc.',
        'and so on': 'etc.',
        'application programming interface': 'API',
        'user interface': 'UI',
        'user experience': 'UX',
        'artificial intelligence': 'AI',
        'machine learning': 'ML',
        'large language model': 'LLM',
        'natural language processing': 'NLP',
        'return on investment': 'ROI',
        'key performance indicator': 'KPI',
        'chief executive officer': 'CEO',
        'chief technology officer': 'CTO',
        'human resources': 'HR',
        'public relations': 'PR',
        'customer service': 'CS',
        'frequently asked questions': 'FAQ',
        'terms of service': 'TOS',
        'end of day': 'EOD',
        'end of week': 'EOW',
        'work in progress': 'WIP',
        'as a matter of fact': 'fact is',
        'in the meantime': 'meanwhile',
        'at the present time': 'now',
        'in the process of': 'working on',
    }

    # Word to symbol mappings
    SYMBOLS = {
        'greater than or equal to': '≥',
        'less than or equal to': '≤',
        'greater than': '>',
        'less than': '<',
        'equals': '=',
        'equal to': '=',
        'approximately': '~',
        'roughly': '~',
        'about': '~',
        'percent': '%',
        'percentage': '%',
        'degrees': '°',
        'multiplied by': '×',
        'times': '×',
        'divided by': '÷',
        'over': '÷',
        'plus': '+',
        'minus': '-',
        'arrow': '→',
        'leads to': '→',
        'implies': '⇒',
        'therefore': '∴',
        'because': '∵',
        'sum': '∑',
        'square root': '√',
        'infinity': '∞',
        'alpha': 'α',
        'beta': 'β',
        'gamma': 'γ',
        'delta': 'Δ',
        'theta': 'θ',
        'lambda': 'λ',
        'sigma': 'Σ',
        'pi': 'π',
        'not equal': '≠',
    }

    # Number words to digits (small numbers only to avoid false positives)
    NUMBER_WORDS = {
        'zero': '0',
        'one': '1',
        'two': '2',
        'three': '3',
        'four': '4',
        'five': '5',
        'six': '6',
        'seven': '7',
        'eight': '8',
        'nine': '9',
        'ten': '10',
    }

    # Code and log-specific patterns for compression
    CODE_PATTERNS = {
        # Common code terms
        'function': 'fn',
        'return': 'ret',
        'console.log': 'log',
        'document.getElementById': 'byId',
        'addEventListener': 'ael',
        'querySelector': 'qs',
        'querySelectorAll': 'qsa',
        'getElementsByClassName': 'byCls',
        'getElementsByTagName': 'byTag',
        'JSON.parse': 'J.p',
        'JSON.stringify': 'J.s',
        'Object.keys': 'O.k',
        'Object.values': 'O.v',
        'Object.entries': 'O.e',
        'Array.isArray': 'A.is',
        'Promise.all': 'P.all',
        'async function': 'async fn',
        '=>': '→',
        # Common error patterns
        'Error:': 'Err:',
        'Warning:': 'Warn:',
        'Exception:': 'Exc:',
        'Traceback': 'TB',
        'File "': 'F"',
        'line ': 'L',
        'in <module>': '<mod>',
        'at ': '@',
        # File paths
        '/home/': '~/',
        '/Users/': '~/',
        'node_modules': 'nm',
        '__pycache__': 'pyc',
        '.gitignore': '.gi',
        'package.json': 'pkg.json',
        'requirements.txt': 'req.txt',
        # Log patterns
        '[INFO]': '[I]',
        '[DEBUG]': '[D]',
        '[ERROR]': '[E]',
        '[WARN]': '[W]',
        '[WARNING]': '[W]',
        'timestamp': 'ts',
        'undefined': 'undef',
        'null': '∅',
        'true': 'T',
        'false': 'F',
    }

    # Whitespace compression patterns
    WHITESPACE_PATTERNS = [
        (r'\n{3,}', '\n\n'),      # 3+ newlines → 2
        (r'[ \t]+', ' '),          # multiple spaces/tabs → 1 space
        (r' ?\n ?', '\n'),         # trim spaces around newlines
        (r'^ +', ''),              # leading spaces per line
        (r' +$', ''),              # trailing spaces per line
    ]

    def compress(self, text: str) -> str:
        """
        Apply compression techniques to text.

        Args:
            text: Input text to compress

        Returns:
            Compressed text
        """
        if not text or not isinstance(text, str):
            return text

        # Apply path compression first if enabled (EXPERIMENTAL)
        path_stats = None
        if self.enable_path_compression and self.path_compressor:
            text, path_stats = self.path_compressor.compress(text)

        # Apply level-based compression
        if self.level == "low":
            result = self._light_compression(text)
        elif self.level == "medium":
            result = self._medium_compression(text)
        else:  # high
            result = self._heavy_compression(text)

        return result

    def _light_compression(self, text: str) -> str:
        """Minimal compression - just abbreviations (case-insensitive)."""
        result = text

        # Apply abbreviations (case-insensitive replacement)
        for phrase, abbreviation in self.ABBREVIATIONS.items():
            result = re.sub(r'\b' + re.escape(phrase) + r'\b', abbreviation, result, flags=re.IGNORECASE)

        # Clean up extra whitespace
        result = re.sub(r'\s+', ' ', result).strip()

        return result

    def _medium_compression(self, text: str) -> str:
        """Standard compression - abbreviations + filler words + symbols (preserves case)."""
        result = text

        # Apply abbreviations (case-insensitive)
        for phrase, abbreviation in self.ABBREVIATIONS.items():
            result = re.sub(r'\b' + re.escape(phrase) + r'\b', abbreviation, result, flags=re.IGNORECASE)

        # Remove filler words
        for pattern in self.FILLER_PATTERNS:
            result = re.sub(pattern, '', result, flags=re.IGNORECASE)

        # Apply symbols (case-insensitive)
        for phrase, symbol in self.SYMBOLS.items():
            result = re.sub(r'\b' + re.escape(phrase) + r'\b', symbol, result, flags=re.IGNORECASE)

        # Convert small number words (case-insensitive)
        for word, digit in self.NUMBER_WORDS.items():
            result = re.sub(rf'\b{word}\b', digit, result, flags=re.IGNORECASE)

        # Clean up
        result = re.sub(r'\s+', ' ', result).strip()
        result = re.sub(r'\s+,', ',', result)
        result = re.sub(r'\s+\.', '.', result)

        return result

    def _heavy_compression(self, text: str) -> str:
        """Aggressive compression - maximum token reduction."""
        result = self._medium_compression(text)

        # Remove articles (case-insensitive)
        result = re.sub(r'\b(a|an|the)\b', '', result, flags=re.IGNORECASE)

        # Remove relative pronouns where possible (case-insensitive)
        result = re.sub(r'\b(that|which|who|whom)\s+', '', result, flags=re.IGNORECASE)

        # Remove auxiliary verbs in some contexts (case-insensitive)
        result = re.sub(r'\b(is|are|was|were)\s+', ' ', result, flags=re.IGNORECASE)

        # Replace "and" with comma in lists (careful with this)
        # Only do this for simple lists (case-insensitive)
        result = re.sub(r'(\w+)\s+and\s+(\w+)([,.])', r'\1, \2\3', result, flags=re.IGNORECASE)

        # Apply code/log patterns
        for pattern, replacement in self.CODE_PATTERNS.items():
            result = result.replace(pattern, replacement)

        # Aggressive whitespace compression
        for pattern, replacement in self.WHITESPACE_PATTERNS:
            result = re.sub(pattern, replacement, result, flags=re.MULTILINE)

        # Clean up aggressively
        result = re.sub(r'\s+', ' ', result).strip()
        result = re.sub(r',+', ',', result)
        result = re.sub(r'\s+,', ',', result)
        result = re.sub(r',\s*\.', '.', result)
        result = re.sub(r'\s+([.,;:!?])', r'\1', result)

        return result

    def compress_messages(self, messages: list[dict], roles: list[str] = None) -> tuple[list[dict], dict]:
        """
        Compress message content in OpenAI/Anthropic format.

        Args:
            messages: List of message dicts with "content" keys
            roles: List of roles to compress (default: ["user"])

        Returns:
            Tuple of (compressed_messages, stats)
        """
        if roles is None:
            roles = ["user"]  # Only compress user messages by default

        original_tokens = self._estimate_tokens(messages)

        compressed = []
        for msg in messages:
            new_msg = msg.copy()
            # Only compress specified roles
            if msg.get("role") in roles:
                if isinstance(msg.get("content"), str):
                    new_msg["content"] = self.compress(msg["content"])
                elif isinstance(msg.get("content"), list):
                    # Handle content blocks (Anthropic format)
                    new_content = []
                    for block in msg["content"]:
                        if isinstance(block, dict) and block.get("type") == "text":
                            new_block = block.copy()
                            new_block["text"] = self.compress(block["text"])
                            new_content.append(new_block)
                        else:
                            new_content.append(block)
                    new_msg["content"] = new_content
            compressed.append(new_msg)

        compressed_tokens = self._estimate_tokens(compressed)

        stats = {
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "tokens_saved": original_tokens - compressed_tokens,
            "reduction_pct": round(
                (original_tokens - compressed_tokens) / original_tokens * 100, 1
            ) if original_tokens > 0 else 0,
        }

        return compressed, stats

    def _estimate_tokens(self, messages: list[dict]) -> int:
        """
        Estimate token count for messages.
        Simple heuristic: 1 token ≈ 0.75 words
        """
        total_chars = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        total_chars += len(block.get("text", ""))

        # Rough estimate: 1 token per 4 characters
        return total_chars // 4


class MetricsCollector:
    """Collect and report compression metrics."""

    def __init__(self):
        self.requests = 0
        self.total_original_tokens = 0
        self.total_compressed_tokens = 0
        self.start_time = time.time()
        # Timing metrics
        self.total_compression_time_ms = 0
        self.compression_count = 0
        self.min_compression_time_ms = float('inf')
        self.max_compression_time_ms = 0

    def record_request(self, stats: dict):
        """Record metrics from a request."""
        self.requests += 1
        self.total_original_tokens += stats.get("original_tokens", 0)
        self.total_compressed_tokens += stats.get("compressed_tokens", 0)

    def record_compression_time(self, time_ms: float):
        """Record compression latency."""
        self.total_compression_time_ms += time_ms
        self.compression_count += 1
        self.min_compression_time_ms = min(self.min_compression_time_ms, time_ms)
        self.max_compression_time_ms = max(self.max_compression_time_ms, time_ms)

    def get_summary(self) -> dict:
        """Get summary statistics."""
        elapsed = time.time() - self.start_time
        tokens_saved = self.total_original_tokens - self.total_compressed_tokens

        result = {
            "total_requests": self.requests,
            "total_original_tokens": self.total_original_tokens,
            "total_compressed_tokens": self.total_compressed_tokens,
            "total_tokens_saved": tokens_saved,
            "average_reduction_pct": round(
                tokens_saved / self.total_original_tokens * 100, 1
            ) if self.total_original_tokens > 0 else 0,
            "uptime_seconds": round(elapsed, 1),
            "tokens_saved_per_minute": round(tokens_saved / (elapsed / 60), 0) if elapsed > 0 else 0,
        }

        # Add compression timing metrics
        if self.compression_count > 0:
            result["compression_stats"] = {
                "total_compressions": self.compression_count,
                "avg_compression_time_ms": round(self.total_compression_time_ms / self.compression_count, 3),
                "min_compression_time_ms": round(self.min_compression_time_ms, 3),
                "max_compression_time_ms": round(self.max_compression_time_ms, 3),
                "total_compression_time_ms": round(self.total_compression_time_ms, 3),
            }

        return result


class LLMProxyHandler(BaseHTTPRequestHandler):
    """HTTP request handler for LLM API proxy."""

    compressor = CompressionEngine(level="high")
    metrics = MetricsCollector()
    config = None  # Set at startup from proxy_config.yaml

    def do_GET(self):
        """Handle GET requests (for health checks, etc.)."""
        if self.path == "/health":
            self._send_json({"status": "ok", "proxy": "llm-compression-proxy"})
        elif self.path == "/metrics":
            self._send_json(self.metrics.get_summary())
        elif self.path == "/":
            self._send_json({
                "message": "LLM Compression Proxy",
                "endpoints": {
                    "/health": "Health check",
                    "/metrics": "Compression metrics",
                    "/v1/messages": "Anthropic Messages API",
                    "/v1/chat/completions": "OpenAI Chat API",
                },
            })
        else:
            # Forward to target
            self._forward_request("GET")

    def do_POST(self):
        """Handle POST requests (API calls)."""
        # Strip query string from path for matching
        path_without_query = self.path.split("?")[0]
        if path_without_query in ["/v1/messages", "/v1/chat/completions"]:
            self._handle_chat_request()
        else:
            self._forward_request("POST")

    def _handle_chat_request(self):
        """Handle chat completion requests with compression."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        # Log step 1: Request received
        print(f"\n[STEP 1] 📥 Request received from Claude Code")
        print(f"         Path: {self.path}")
        print(f"         Content-Length: {content_length} bytes")

        try:
            data = json.loads(body.decode("utf-8"))
            print(f"[STEP 2] ✅ JSON parsed successfully")

            # Check if passthrough mode is enabled
            passthrough = self.config.get("compression", {}).get("passthrough_mode", False) if self.config else False

            if passthrough:
                print(f"[STEP 3] ⏩ PASSTHROUGH MODE - No compression applied")
                stats = {"original_tokens": 0, "compressed_tokens": 0, "tokens_saved": 0, "reduction_pct": 0}
            else:
                # Compress prompts if messages are present
                # Only compress USER messages, not assistant responses
                stats = {}
                if "messages" in data:
                    original_messages = data["messages"]
                    # Count user messages for logging
                    user_count = len([m for m in original_messages if m.get("role") == "user"])
                    print(f"[STEP 3] 🗜️  Compressing {user_count} user messages...")

                    # Only compress user messages, keep assistant/system messages unchanged
                    compress_start = time.perf_counter()
                    data["messages"], stats = self.compressor.compress_messages(
                        original_messages, roles=["user"]
                    )
                    compress_time_ms = (time.perf_counter() - compress_start) * 1000
                    self.metrics.record_request(stats)
                    self.metrics.record_compression_time(compress_time_ms)

                    # Log compression
                    print(
                        f"[STEP 4] ✅ Compression complete: {stats['original_tokens']} → "
                        f"{stats['compressed_tokens']} tokens "
                        f"({stats['reduction_pct']}% saved, {compress_time_ms:.3f}ms)"
                    )

            # Determine target API
            target_url = self._get_target_url()
            print(f"[STEP 5] 🌐 Target API: {target_url}")

            headers = self._prepare_headers()

            # Add provider prompt caching headers for Anthropic
            if self.config and self.config.get("cache_control", {}).get("enabled", False):
                data = self._add_prompt_caching(data)
                print(f"[STEP 5b] 💾 Prompt caching enabled")

            print(f"[STEP 6] 📤 Forwarding request to target API...")

            # Forward to target API
            print(f"[STEP 6b] Target URL: {target_url}")
            print(f"[STEP 6c] Headers: {list(headers.keys())}")

            # Use requests library instead of httpx (better DNS handling)
            try:
                print(f"[STEP 6d] Sending request via requests library...")
                response = requests.post(
                    target_url,
                    json=data,
                    headers=headers,
                    timeout=300,
                    allow_redirects=True,
                    stream=True
                )
                print(f"[STEP 7] 📥 Response received: {response.status_code}")

                # Return response to client
                self._send_response_requests(response)
                print(f"[STEP 8] ✅ Response sent back to Claude Code\n")

            except requests.exceptions.ConnectionError as e:
                print(f"[ERROR] ❌ Connection failed: {e}")
                print(f"        Target URL: {target_url}")
                print(f"        Headers: {list(headers.keys())}")
                self._send_error(502, f"Cannot connect to target API: {e}")

        except json.JSONDecodeError as e:
            print(f"[ERROR] ❌ Invalid JSON: {e}")
            self._send_error(400, f"Invalid JSON: {e}")
        except Exception as e:
            print(f"[ERROR] ❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            self._send_error(500, str(e))

    def _add_prompt_caching(self, data: dict) -> dict:
        """
        Add Anthropic prompt caching headers to messages.

        Marks static content (system prompt, early messages) as cacheable.
        First request writes to cache, subsequent reads get 90% discount.
        """
        config = self.config.get("cache_control", {})
        cache_roles = config.get("cache_roles", ["system"])
        cache_first_n = config.get("cache_first_n_messages", 0)

        # Add cache_control to specified roles
        if "messages" in data:
            for i, msg in enumerate(data["messages"]):
                # Cache by role (e.g., system prompts)
                if msg.get("role") in cache_roles:
                    if isinstance(msg.get("content"), list):
                        # Anthropic format - add to first text block
                        for block in msg["content"]:
                            if block.get("type") == "text":
                                block["cache_control"] = {"type": "ephemeral"}
                                break
                    elif isinstance(msg.get("content"), str):
                        # Convert to blocks format with caching
                        msg["content"] = [
                            {
                                "type": "text",
                                "text": msg["content"],
                                "cache_control": {"type": "ephemeral"}
                            }
                        ]

                # Cache first N messages of conversation (static context)
                if cache_first_n > 0 and i < cache_first_n:
                    if isinstance(msg.get("content"), list):
                        for block in msg["content"]:
                            if block.get("type") == "text" and "cache_control" not in block:
                                block["cache_control"] = {"type": "ephemeral"}
                                break
                    elif isinstance(msg.get("content"), str):
                        msg["content"] = [
                            {
                                "type": "text",
                                "text": msg["content"],
                                "cache_control": {"type": "ephemeral"}
                            }
                        ]

        # Handle Anthropic system prompt (outside messages array)
        if "system" in data and isinstance(data["system"], str):
            data["system"] = [
                {
                    "type": "text",
                    "text": data["system"],
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        elif "system" in data and isinstance(data["system"], list):
            for block in data["system"]:
                if block.get("type") == "text":
                    block["cache_control"] = {"type": "ephemeral"}

        return data

    def _forward_request(self, method: str):
        """Forward request without modification."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""

        target_url = self._get_target_url()
        headers = self._prepare_headers()

        try:
            response = requests.request(
                method,
                target_url,
                data=body if body else None,
                headers=headers,
                timeout=300,
                allow_redirects=True
            )
            self._send_response_requests(response)
        except Exception as e:
            self._send_error(500, str(e))

    def _get_target_url(self) -> str:
        """Determine target API URL from config."""
        # Use target API from config file
        if self.config and "target_api" in self.config:
            base_url = self.config["target_api"].get("url", "https://api.anthropic.com")
        else:
            base_url = "https://api.anthropic.com"

        # Ensure base_url doesn't end with trailing slash for proper joining
        base_url = base_url.rstrip('/')

        # Construct full URL - path already includes leading slash
        full_url = f"{base_url}{self.path}"

        print(f"[DEBUG] URL Construction:")
        print(f"        Base URL: {base_url}")
        print(f"        Path: {self.path}")
        print(f"        Full URL: {full_url}")

        return full_url

    def _prepare_headers(self) -> dict:
        """Prepare headers for forwarding to target API."""
        filtered = {}
        for key, value in self.headers.items():
            key_lower = key.lower()
            # Skip hop-by-hop headers
            if key_lower not in (
                "host",
                "content-length",
                "transfer-encoding",
                "connection",
                "keep-alive",
                "proxy-authenticate",
                "proxy-authorization",
                "te",
                "trailers",
                "upgrade",
            ):
                filtered[key] = value
        return filtered

    def _send_response(self, response):
        """Send HTTP response back to client."""
        self.send_response(response.status_code)

        # Forward headers
        for key, value in response.headers.items():
            if key.lower() not in ("content-encoding", "transfer-encoding", "content-length"):
                self.send_header(key, value)

        content = response.content if hasattr(response, 'content') else response.body
        self.send_header("Content-Length", len(content))
        self.end_headers()
        self.wfile.write(content)

    def _send_response_requests(self, response):
        """Send HTTP response back to client (requests library version)."""
        self.send_response(response.status_code)

        # Check if this is a streaming response
        content_type = response.headers.get('Content-Type', '')
        is_streaming = 'text/event-stream' in content_type or response.headers.get('Transfer-Encoding') == 'chunked'

        if is_streaming:
            # For streaming responses, forward chunks as they arrive
            for key, value in response.headers.items():
                if key.lower() not in ("content-encoding", "transfer-encoding", "content-length", "connection"):
                    self.send_header(key, value)
            # Force connection close after streaming
            self.send_header("Connection", "close")
            self.end_headers()

            # Stream chunks to client
            try:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        self.wfile.write(chunk)
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                # Client closed connection - this is normal for streaming
                pass
            finally:
                response.close()
        else:
            # For regular responses, send complete content
            for key, value in response.headers.items():
                if key.lower() not in ("content-encoding", "transfer-encoding", "content-length"):
                    self.send_header(key, value)

            self.send_header("Content-Length", len(response.content))
            self.end_headers()
            self.wfile.write(response.content)

    def _send_json(self, data: dict):
        """Send JSON response."""
        content = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(content))
        self.end_headers()
        self.wfile.write(content)

    def _send_error(self, code: int, message: str):
        """Send error response."""
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        error_content = json.dumps({"error": message}).encode("utf-8")
        self.send_header("Content-Length", len(error_content))
        self.end_headers()
        self.wfile.write(error_content)

    def log_message(self, format: str, *args):
        """Custom logging."""
        print(f"[PROXY] {self.command} {self.path} - {args[0] if args else ''}")


def run_proxy(port: int = 8080, compression_level: str = "high", config_path: str = None):
    """Start the proxy server."""
    # Load configuration
    config = load_config(config_path)
    LLMProxyHandler.config = config

    # Override with command line args if provided
    enable_path_compression = False
    if config and "compression" in config:
        compression_level = config["compression"].get("level", compression_level)
        enable_path_compression = config["compression"].get("enable_path_compression", False)
    if config and "server" in config:
        port = config["server"].get("port", port)

    # Configure compression
    LLMProxyHandler.compressor = CompressionEngine(
        level=compression_level,
        enable_path_compression=enable_path_compression
    )

    path_comp_status = "ENABLED ⚡" if enable_path_compression else "disabled"

    # Get target API from config
    target_api = config.get("target_api", {}).get("url", "https://api.anthropic.com") if config else "https://api.anthropic.com"

    # Check if passthrough mode is enabled
    passthrough = config.get("compression", {}).get("passthrough_mode", False) if config else False
    mode_str = "PASSTHROUGH (no compression)" if passthrough else f"COMPRESSION ({compression_level})"

    server = ThreadingHTTPServer(("localhost", port), LLMProxyHandler)

    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  LLM Compression Proxy                                           ║
╠══════════════════════════════════════════════════════════════════╣
║  Listening on:    http://localhost:{port:<5}                        ║
║  Mode:            {mode_str:<45}║
║  Target API:      {target_api:<45}║
║  Path Compress:   {path_comp_status:<45}║
╠══════════════════════════════════════════════════════════════════╣
║  Configure Claude Code:                                          ║
║    export ANTHROPIC_BASE_URL=http://localhost:{port}              ║
║    claude                                                        ║
╠══════════════════════════════════════════════════════════════════╣
║  Endpoints:                                                      ║
║    http://localhost:{port}/health     - Health check              ║
║    http://localhost:{port}/metrics    - Compression statistics    ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\n[PROXY] Shutting down...")

        # Print final stats
        stats = LLMProxyHandler.metrics.get_summary()
        print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  Final Statistics                                                ║
╠══════════════════════════════════════════════════════════════════╣
║  Total Requests:        {stats['total_requests']:<6}                                   ║
║  Original Tokens:       {stats['total_original_tokens']:<6}                                   ║
║  Compressed Tokens:     {stats['total_compressed_tokens']:<6}                                   ║
║  Tokens Saved:          {stats['total_tokens_saved']:<6}                                   ║
║  Average Reduction:     {stats['average_reduction_pct']:<5}%                                  ║
╚══════════════════════════════════════════════════════════════════╝
        """)


def main():
    parser = argparse.ArgumentParser(
        description="LLM API Proxy with Prompt Compression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default settings (high compression)
    python llm_proxy.py

    # Run on different port
    python llm_proxy.py --port 9000

    # Use medium compression (less aggressive)
    python llm_proxy.py --level medium

    # Use custom config file
    python llm_proxy.py --config proxy_config.yaml

    # View metrics while running
    curl http://localhost:8080/metrics
        """,
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to listen on (default: 8080)",
    )
    parser.add_argument(
        "--level",
        choices=["low", "medium", "high"],
        default="high",
        help="Compression level (default: high)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: proxy_config.yaml)",
    )
    args = parser.parse_args()

    run_proxy(port=args.port, compression_level=args.level, config_path=args.config)


if __name__ == "__main__":
    main()
