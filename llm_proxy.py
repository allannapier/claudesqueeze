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
from urllib.parse import urljoin, urlparse

# Security constants
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max request size
MAX_HEADER_LENGTH = 8192  # 8KB max header size
ALLOWED_TARGET_HOSTS = [
    "api.anthropic.com",
    "api.openai.com",
    "api.groq.com",
    "api.cohere.com",
    "api.mistral.ai",
    "api.z.ai",
    "api.moonshot.cn",
    "api.kimi.com",          # Kimi
    "api.minimax.chat",         # MiniMax
    "api.portkey.ai",           # Portkey
    "generativelanguage.googleapis.com",  # Gemini
]
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Content-Security-Policy": "default-src 'none'",
}

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

# Import failover modules - handle both direct and module execution
try:
    from account_pool import AccountPool
    from failover_handler import FailoverHandler, AllAccountsExhaustedError
    from rate_limiter import RateLimiter
except ImportError:
    # Try importing as local modules (when run as module)
    try:
        from .account_pool import AccountPool
        from .failover_handler import FailoverHandler, AllAccountsExhaustedError
        from .rate_limiter import RateLimiter
    except ImportError:
        # Modules not available - define dummy exception and classes
        AccountPool = None
        FailoverHandler = None
        RateLimiter = None

        # Define dummy exception for compatibility
        class AllAccountsExhaustedError(Exception):
            def __init__(self, status):
                self.status = status
                super().__init__(f"All accounts exhausted. Status: {status}")


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
        # Validate config path is not a symlink to prevent path traversal
        config_path_obj = Path(config_path).resolve()
        if config_path_obj.is_symlink():
            raise ValueError(f"Config path cannot be a symlink: {config_path}")
        # Validate config file is readable and within expected directories
        allowed_roots = [
            Path.cwd().resolve(),
            Path(__file__).parent.resolve(),
            Path.home().resolve() / ".llm-proxy",
        ]
        if not any(str(config_path_obj).startswith(str(root)) for root in allowed_roots):
            raise ValueError(f"Config path outside allowed directories: {config_path}")
        with open(config_path_obj) as f:
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
    account_pool = None  # Set at startup for account failover
    failover_handler = None  # Set at startup for automatic failover

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

        # Validate content length to prevent memory exhaustion
        if content_length > MAX_CONTENT_LENGTH:
            self._send_error(413, f"Request body too large. Max: {MAX_CONTENT_LENGTH} bytes")
            return

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
            try:
                target_url = self._get_target_url()
            except ValueError as e:
                self._send_error(400, f"Invalid target URL: {e}")
                return
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

                # Define the request function for failover handler
                def make_request(account, data, headers):
                    """Make request with the given account's API key and endpoint."""
                    # Use account's endpoint if specified, otherwise fall back to global target_url
                    if account.endpoint:
                        # Account endpoint is the FULL URL (use it directly without appending path)
                        url = account.endpoint.rstrip('/')  # Remove trailing slash to avoid double slashes
                        # Add query string from original request if not present
                        if '?' in target_url:
                            query_part = target_url.split('?', 1)[1]
                            separator = '&' if '?' in url else '?'
                            url = f"{url}{separator}{query_part}"
                        print(f"[STEP 6e] Using account endpoint: {account.name} ({account.id})")
                        print(f"[STEP 6f] Full URL: {url}")
                    else:
                        # No account endpoint, use global target_url
                        url = target_url
                        print(f"[STEP 6e] Using account: {account.name} ({account.id})")
                        print(f"[STEP 6f] Endpoint: {url}")
                    return requests.post(
                        url,
                        json=data,
                        headers=headers,
                        timeout=300,
                        allow_redirects=False,  # Disable redirects to prevent SSRF
                        stream=True
                    )

                # Use failover handler if configured, otherwise direct request
                if self.failover_handler:
                    print(f"[STEP 6e] Using failover handler for automatic account switching")
                    response = self.failover_handler.execute_stream_with_failover(
                        make_request,
                        data,
                        headers,
                        target_url
                    )
                else:
                    # Direct request without failover (single account mode)
                    response = requests.post(
                        target_url,
                        json=data,
                        headers=headers,
                        timeout=300,
                        allow_redirects=False,  # Disable redirects to prevent SSRF
                        stream=True
                    )

                print(f"[STEP 7] 📥 Response received: {response.status_code}")

                # Debug: log response details
                content_type = response.headers.get('Content-Type', 'unknown')
                content_length = response.headers.get('Content-Length', 'unknown')
                transfer_encoding = response.headers.get('Transfer-Encoding', 'none')
                location = response.headers.get('location', 'none')
                print(f"[DEBUG] Content-Type: {content_type}")
                print(f"[DEBUG] Content-Length: {content_length}")
                print(f"[DEBUG] Transfer-Encoding: {transfer_encoding}")
                print(f"[DEBUG] Location: {location}")
                print(f"[DEBUG] Headers: {list(response.headers.keys())}")

                # For redirects, log where it's trying to redirect to
                if response.status_code in [301, 302, 303, 307, 308]:
                    print(f"[WARN] Got {response.status_code} redirect to: {location}")
                    print(f"[WARN] Proxy does not follow redirects for security reasons")
                    print(f"[WARN] Update your account endpoint to the correct URL")

                # For non-streaming responses (small JSON), log the body
                if 'application/json' in content_type:
                    try:
                        body = response.text
                        if body:
                            print(f"[DEBUG] Response body: {body}")
                    except:
                        pass

                # Return response to client
                self._send_response_requests(response)
                print(f"[STEP 8] ✅ Response sent back to Claude Code\n")

            except AllAccountsExhaustedError as e:
                print(f"[ERROR] ❌ All accounts exhausted or rate-limited")
                print(f"        Status: {e.status}")
                self._send_error(503, "All API accounts are currently rate-limited. Please try again later.")
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

        # Validate content length to prevent memory exhaustion
        if content_length > MAX_CONTENT_LENGTH:
            self._send_error(413, f"Request body too large. Max: {MAX_CONTENT_LENGTH} bytes")
            return

        body = self.rfile.read(content_length) if content_length > 0 else b""

        try:
            target_url = self._get_target_url()
        except ValueError as e:
            self._send_error(400, f"Invalid target URL: {e}")
            return

        headers = self._prepare_headers()

        try:
            response = requests.request(
                method,
                target_url,
                data=body if body else None,
                headers=headers,
                timeout=300,
                allow_redirects=False  # Disable redirects to prevent SSRF
            )
            self._send_response_requests(response)
        except Exception as e:
            self._send_error(500, str(e))

    def _get_target_url(self) -> str:
        """Determine target API URL from config with SSRF protection."""
        # Use target API from config file
        if self.config and "target_api" in self.config:
            base_url = self.config["target_api"].get("url", "https://api.anthropic.com")
        else:
            base_url = "https://api.anthropic.com"

        # Validate base URL to prevent SSRF
        parsed_base = urlparse(base_url)
        if parsed_base.scheme not in ("https", "http"):
            raise ValueError(f"Invalid URL scheme: {parsed_base.scheme}")

        # Check if host is in allowlist (if configured)
        allowed_hosts = self.config.get("security", {}).get("allowed_hosts", ALLOWED_TARGET_HOSTS)
        if allowed_hosts and parsed_base.hostname not in allowed_hosts:
            raise ValueError(f"Host not in allowlist: {parsed_base.hostname}")

        # Validate path to prevent path traversal and SSRF
        path = self.path.split("?")[0]  # Remove query string
        # Block paths that attempt traversal or have suspicious patterns
        dangerous_patterns = [
            "..", "//", "\\", "@", " ", "\t", "\n", "\r",
            "#", "%", "\x00", "\x01", "\x02", "\x03",
        ]
        for pattern in dangerous_patterns:
            if pattern in path:
                raise ValueError(f"Invalid path characters detected: {repr(pattern)}")

        # Ensure base_url doesn't end with trailing slash for proper joining
        base_url = base_url.rstrip('/')

        # Construct full URL - path already includes leading slash
        full_url = f"{base_url}{self.path}"

        # Final validation: ensure constructed URL doesn't redirect to internal addresses
        parsed_full = urlparse(full_url)
        if parsed_full.hostname in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
            raise ValueError("Cannot proxy to localhost/loopback addresses")

        return full_url

    def _prepare_headers(self) -> dict:
        """Prepare headers for forwarding to target API with security filtering."""
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
                # Validate header value to prevent injection attacks
                if not self._is_safe_header_value(value):
                    continue
                # Limit header length to prevent DoS
                if len(value) > MAX_HEADER_LENGTH:
                    value = value[:MAX_HEADER_LENGTH]
                filtered[key] = value
        return filtered

    def _is_safe_header_value(self, value: str) -> bool:
        """Check if header value is safe (no injection attempts)."""
        if not isinstance(value, str):
            return False
        # Reject values with newlines to prevent header injection
        if '\n' in value or '\r' in value:
            return False
        # Reject null bytes
        if '\x00' in value:
            return False
        return True

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
        """Send HTTP response back to client (requests library version) with security headers."""
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
            # Add security headers
            for header, sec_value in SECURITY_HEADERS.items():
                self.send_header(header, sec_value)
            self.end_headers()

            # Stream chunks to client
            chunk_count = 0
            total_bytes = 0
            try:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        self.wfile.write(chunk)
                        self.wfile.flush()
                        chunk_count += 1
                        total_bytes += len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                # Client closed connection - this is normal for streaming
                pass
            finally:
                print(f"[DEBUG] Streaming complete: {chunk_count} chunks, {total_bytes} bytes")
                response.close()
        else:
            # For regular responses, send complete content
            for key, value in response.headers.items():
                if key.lower() not in ("content-encoding", "transfer-encoding", "content-length"):
                    self.send_header(key, value)

            content = response.content
            self.send_header("Content-Length", len(content))
            # Add security headers
            for header, sec_value in SECURITY_HEADERS.items():
                self.send_header(header, sec_value)
            self.end_headers()

            # Debug log small responses (likely errors)
            if len(content) < 200:
                print(f"[DEBUG] Small response body ({len(content)} bytes): {content.decode('utf-8', errors='replace')}")

            self.wfile.write(content)

    def _send_json(self, data: dict):
        """Send JSON response with security headers."""
        content = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(content))
        # Add security headers
        for header, value in SECURITY_HEADERS.items():
            self.send_header(header, value)
        self.end_headers()
        self.wfile.write(content)

    def _send_error(self, code: int, message: str):
        """Send error response with security headers."""
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        error_content = json.dumps({"error": message}).encode("utf-8")
        self.send_header("Content-Length", len(error_content))
        # Add security headers
        for header, value in SECURITY_HEADERS.items():
            self.send_header(header, value)
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

    # Initialize account pool and failover handler if configured
    failover_enabled = False
    accounts_config = config.get("accounts", []) if config else []
    failover_config = config.get("failover", {}) if config else {}

    if accounts_config and len(accounts_config) > 1 and failover_config.get("enabled", True):
        try:
            strategy = failover_config.get("strategy", "priority")

            # Check if failover modules are available
            if AccountPool is None or FailoverHandler is None:
                print(f"[WARN] Failover modules not found. Running in single-account mode.")
                print(f"[WARN] Ensure account_pool.py and failover_handler.py are in the same directory.")
            else:
                LLMProxyHandler.account_pool = AccountPool(accounts_config, strategy=strategy)

                LLMProxyHandler.failover_handler = FailoverHandler(
                    account_pool=LLMProxyHandler.account_pool,
                    max_retries=failover_config.get("max_retries", len(accounts_config)),
                    backoff_multiplier=failover_config.get("backoff_multiplier", 2.0),
                    max_backoff_seconds=failover_config.get("max_backoff_seconds", 300),
                    default_backoff_seconds=failover_config.get("cooldown_seconds", 60),
                )
                failover_enabled = True
                print(f"[FAILOVER] Enabled with {len(accounts_config)} accounts (strategy: {strategy})")
        except Exception as e:
            print(f"[WARN] Failed to initialize failover: {e}")
            print(f"[WARN] Running in single-account mode")
    elif accounts_config:
        print(f"[INFO] Single account configured - failover disabled")
        print(f"[INFO] Add multiple accounts to enable automatic failover")

    # Check if passthrough mode is enabled
    passthrough = config.get("compression", {}).get("passthrough_mode", False) if config else False
    mode_str = "PASSTHROUGH (no compression)" if passthrough else f"COMPRESSION ({compression_level})"

    server = ThreadingHTTPServer(("localhost", port), LLMProxyHandler)

    # Build failover status string
    if failover_enabled and LLMProxyHandler.account_pool:
        num_accounts = len(LLMProxyHandler.account_pool.get_all_accounts())
        failover_status = f"ENABLED ({num_accounts} accounts)"
    else:
        failover_status = "disabled"

    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  LLM Compression Proxy                                           ║
╠══════════════════════════════════════════════════════════════════╣
║  Listening on:    http://localhost:{port:<5}                        ║
║  Mode:            {mode_str:<45}║
║  Target API:      {target_api:<45}║
║  Path Compress:   {path_comp_status:<45}║
║  Auto Failover:   {failover_status:<45}║
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


def run_setup_wizard(config_path: str = "config.yaml"):
    """Interactive configuration wizard to set up or update config file."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  ClaudeSqueeze Configuration Wizard                             ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    # Check if config already exists
    config_path_obj = Path(config_path)
    existing_config = None
    if config_path_obj.exists():
        try:
            with open(config_path_obj) as f:
                existing_config = yaml.safe_load(f)
            print(f"[INFO] Found existing config at: {config_path}")
            print("[INFO] The wizard will update your existing settings.")
            print()
        except Exception:
            print("[WARN] Could not read existing config, creating new one.")

    # Gather configuration from user
    print("Let's configure your LLM API proxy settings.")
    print("Press Enter to use the [default] value.\n")

    # API Endpoint
    default_url = "https://api.anthropic.com"
    if existing_config and "target_api" in existing_config:
        default_url = existing_config["target_api"].get("url", default_url)

    api_url = input(f"API endpoint URL [{default_url}]: ").strip()
    if not api_url:
        api_url = default_url

    # Compression level
    default_level = "high"
    if existing_config and "compression" in existing_config:
        default_level = existing_config["compression"].get("level", default_level)

    print("\nCompression level controls how aggressively prompts are compressed:")
    print("  - low:    Minimal compression, preserves most detail")
    print("  - medium: Balanced compression and quality")
    print("  - high:   Maximum compression, lowest token usage")
    compression_level = input(f"\nCompression level [low/medium/high] [{default_level}]: ").strip().lower()
    if not compression_level or compression_level not in ["low", "medium", "high"]:
        compression_level = default_level

    # Port
    default_port = 8080
    if existing_config and "server" in existing_config:
        default_port = existing_config["server"].get("port", default_port)

    port_input = input(f"\nProxy server port [{default_port}]: ").strip()
    try:
        port = int(port_input) if port_input else default_port
    except ValueError:
        print(f"[WARN] Invalid port, using default: {default_port}")
        port = default_port

    # Ask about optional features
    print("\nOptional features (y/n):")

    enable_path_compression = True
    if existing_config and "compression" in existing_config:
        enable_path_compression = existing_config["compression"].get("enable_path_compression", True)

    path_choice = input(f"  Enable path compression (reduces file paths/URLs) [{'y' if enable_path_compression else 'n'}]: ").strip().lower()
    if path_choice:
        enable_path_compression = path_choice.startswith('y')

    enable_cache = True
    if existing_config and "cache_control" in existing_config:
        enable_cache = existing_config["cache_control"].get("enabled", True)

    cache_choice = input(f"  Enable prompt caching (90% cost reduction on repeated content) [{'y' if enable_cache else 'n'}]: ").strip().lower()
    if cache_choice:
        enable_cache = cache_choice.startswith('y')

    # Account configuration for failover
    print("\n" + "="*66)
    print("API ACCOUNT CONFIGURATION (for automatic failover)")
    print("="*66)
    print("\nYou can configure multiple API accounts for automatic failover.")
    print("When one account hits rate limits (429 error), the proxy will")
    print("automatically switch to the next available account.\n")

    # Get existing accounts if available
    existing_accounts = existing_config.get("accounts", []) if existing_config else []

    if existing_accounts:
        print(f"[INFO] Found {len(existing_accounts)} existing account(s)")
        keep_existing = input(f"Keep existing accounts and add more? [{'y'}/n]: ").strip().lower()
        if keep_existing.startswith('n'):
            existing_accounts = []
            print("[INFO] Existing accounts will be replaced")

    accounts = list(existing_accounts)

    # Add new accounts
    while True:
        print(f"\n--- Configure Account {len(accounts) + 1} ---")
        print("(Leave API key blank to finish adding accounts)")

        api_key = input(f"API key for account {len(accounts) + 1}: ").strip()

        if not api_key:
            if len(accounts) == 0:
                print("[WARN] At least one account is required. Please enter an API key.")
                continue
            break

        account_name = input(f"Account name (e.g., 'Primary', 'Backup') [Account {len(accounts) + 1}]: ").strip()
        if not account_name:
            account_name = f"Account {len(accounts) + 1}"

        account_id = input(f"Account ID (unique identifier) [account_{len(accounts) + 1}]: ").strip().lower()
        if not account_id:
            account_id = f"account_{len(accounts) + 1}"

        # Ask for account-specific endpoint (optional - uses global target_api.url if not specified)
        print("\n  You can specify a different API endpoint for this account.")
        print(f"  This should be the FULL URL including the path (e.g., https://api.z.ai/api/anthropic/v1/messages)")
        print(f"  Leave blank to use the global endpoint: {api_url}")
        account_endpoint = input(f"  Endpoint for this account (full URL, or press Enter to use global): ").strip()
        if not account_endpoint:
            account_endpoint = None  # Will use global target_api.url

        accounts.append({
            "id": account_id,
            "name": account_name,
            "api_key": api_key,
            "endpoint": account_endpoint,  # Per-account endpoint
            "priority": len(accounts) + 1,
        })

        print(f"\n[INFO] Added account: {account_name} ({account_id})")

        # Ask if they want to add more
        if len(accounts) >= 2:
            add_more = input(f"\nAdd another account? [{'y'}/n]: ").strip().lower()
            if add_more.startswith('n'):
                break
        else:
            print("\n[INFO] Add at least 2 accounts to enable automatic failover")
            add_more = input(f"Add another account? [{'y'}/n]: ").strip().lower()
            if add_more.startswith('n'):
                break

    # Configure failover settings if multiple accounts
    failover_config = {"enabled": len(accounts) > 1}
    if len(accounts) > 1:
        print("\n--- Failover Settings ---")
        print("Strategy: How accounts are selected when one fails")
        print("  - priority: Use accounts in order (1, 2, 3...)")
        print("  - round_robin: Rotate through accounts evenly")

        strategy = input(f"Selection strategy [priority/round_robin] [priority]: ").strip().lower()
        if strategy not in ["priority", "round_robin"]:
            strategy = "priority"

        failover_config["strategy"] = strategy
        failover_config["max_retries"] = len(accounts)
        failover_config["backoff_multiplier"] = 2.0
        failover_config["max_backoff_seconds"] = 300
        failover_config["cooldown_seconds"] = 3600  # 1 hour default - don't retry failed accounts quickly

    # Build config dictionary
    config = {
        "target_api": {
            "url": api_url
        },
        "accounts": accounts,
        "failover": failover_config,
        "compression": {
            "level": compression_level,
            "compress_system": True,
            "compress_user": True,
            "compress_assistant": False,
            "passthrough_mode": False,
            "enable_path_compression": enable_path_compression
        },
        "server": {
            "port": port,
            "host": "127.0.0.1",
            "timeout": 300
        },
        "logging": {
            "level": "info",
            "log_compression": True,
            "log_bodies": False,
            "log_steps": False
        },
        "metrics": {
            "enabled": True,
            "path": "/metrics"
        },
        "cache_control": {
            "enabled": enable_cache,
            "cache_roles": ["system"],
            "cache_first_n_messages": 2
        },
        "providers": {
            "anthropic": {
                "api_version": "2023-06-01"
            }
        }
    }

    # Write config file
    try:
        config_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path_obj, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        # Build account summary
        num_accounts = len(accounts)
        failover_status = "ENABLED" if num_accounts > 1 else "disabled (need 2+ accounts)"

        print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  Configuration Saved!                                            ║
╠══════════════════════════════════════════════════════════════════╣
║  Config file:    {config_path:<50}║
║                                                                   ║
║  Settings:                                                        ║
║    API Endpoint:      {api_url:<41}║
║    Compression:       {compression_level.upper():<41}║
║    Port:              {port:<41}║
║    Path Compression:  {'ENABLED' if enable_path_compression else 'DISABLED':<41}║
║    Prompt Caching:    {'ENABLED' if enable_cache else 'DISABLED':<41}║
║    Accounts:          {num_accounts} configured{' (' + ', '.join([a.get('name', a['id']) for a in accounts[:3]]) + ')' if num_accounts > 0 else '':<21}║
║    Auto Failover:     {failover_status:<41}║
╠══════════════════════════════════════════════════════════════════╣
║  To start the proxy:                                              ║
║    python claudesqueeze.py --config {config_path:<30}║
║                                                                   ║
║  Then configure your client:                                      ║
║    export ANTHROPIC_BASE_URL=http://localhost:{port}              ║
╚══════════════════════════════════════════════════════════════════╝
        """)
    except Exception as e:
        print(f"\n[ERROR] Failed to save config: {e}")
        sys.exit(1)


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
