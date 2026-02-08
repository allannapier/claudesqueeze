#!/usr/bin/env python3
"""
Rate Limiter Detector
=====================

Detects rate limit errors (429) from different LLM providers and extracts
rate limit information from response headers.
"""

import re
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class RateLimitInfo:
    """Information about a rate limit."""
    is_rate_limited: bool
    retry_after_seconds: Optional[int] = None
    requests_remaining: Optional[int] = None
    tokens_remaining: Optional[int] = None
    reset_time: Optional[str] = None
    provider: Optional[str] = None
    error_message: Optional[str] = None


class RateLimiter:
    """
    Detects rate limits from different LLM providers.

    Supports:
    - Anthropic: 429 with rate limit headers
    - OpenAI: 429 with x-ratelimit headers
    - Groq: 429 with x-ratelimit headers
    - Cohere: 429 with body retry_after
    - Kimi: 429 (no headers)
    - MiniMax: 429 (no headers)
    - Portkey: 429 with unified headers
    - Gemini: 429 with x-ratelimit headers
    """

    # Status codes that indicate rate limiting
    RATE_LIMIT_STATUS_CODES = [429, 500, 502, 503]

    # Error patterns that indicate rate limiting
    RATE_LIMIT_PATTERNS = [
        r"rate limit",
        r"quota exceeded",
        r"too many requests",
        r"rate_limited",
        r"throttled",
        r"overloaded",
    ]

    # Provider-specific header mappings
    ANTHROPIC_HEADERS = {
        "requests_remaining": "anthropic-ratelimit-requests-remaining",
        "tokens_remaining": "anthropic-ratelimit-tokens-remaining",
        "retry_after": "retry-after",
    }

    OPENAI_HEADERS = {
        "requests_remaining": "x-ratelimit-remaining-requests",
        "tokens_remaining": "x-ratelimit-remaining-tokens",
        "reset_time": "x-ratelimit-reset-timestamp",
    }

    GROQ_HEADERS = {
        "requests_remaining": "x-ratelimit-requests-remaining",
        "tokens_remaining": "x-ratelimit-tokens-remaining",
        "retry_after": "retry-after",
    }

    GEMINI_HEADERS = {
        "remaining": "x-ratelimit-remaining",
        "retry_after": "retry-after",
    }

    @classmethod
    def detect_from_response(cls, response: Any, url: str = "") -> RateLimitInfo:
        """
        Detect if response indicates rate limiting.

        Args:
            response: requests.Response object
            url: The URL that was requested (for provider detection)

        Returns:
            RateLimitInfo with detected rate limit information
        """
        if not hasattr(response, "status_code"):
            return RateLimitInfo(is_rate_limited=False)

        # Check status code
        if response.status_code not in cls.RATE_LIMIT_STATUS_CODES:
            return RateLimitInfo(is_rate_limited=False)

        # Check for rate limit patterns in response body
        error_message = cls._extract_error_message(response)

        info = RateLimitInfo(
            is_rate_limited=True,
            provider=cls._detect_provider(url),
            error_message=error_message,
        )

        # Extract headers
        headers = getattr(response, "headers", {})

        # Try to extract rate limit info from headers
        if response.status_code == 429:
            info = cls._extract_from_headers(headers, info, url)
            info = cls._extract_from_body(response, info)

        # If no retry_after found, use default
        if info.is_rate_limited and info.retry_after_seconds is None:
            info.retry_after_seconds = cls._calculate_default_backoff(response.status_code)

        return info

    @classmethod
    def _extract_from_headers(cls, headers: Dict[str, str], info: RateLimitInfo, url: str) -> RateLimitInfo:
        """Extract rate limit info from response headers."""
        provider = cls._detect_provider(url)

        if provider == "anthropic":
            return cls._extract_anthropic_headers(headers, info)
        elif provider == "openai":
            return cls._extract_openai_headers(headers, info)
        elif provider == "groq":
            return cls._extract_groq_headers(headers, info)
        elif provider == "gemini":
            return cls._extract_gemini_headers(headers, info)

        # Generic retry-after header
        if "retry-after" in headers:
            try:
                info.retry_after_seconds = int(headers["retry-after"])
            except (ValueError, TypeError):
                pass

        return info

    @classmethod
    def _extract_anthropic_headers(cls, headers: Dict[str, str], info: RateLimitInfo) -> RateLimitInfo:
        """Extract Anthropic-specific rate limit headers."""
        if "anthropic-ratelimit-requests-remaining" in headers:
            try:
                info.requests_remaining = int(headers["anthropic-ratelimit-requests-remaining"])
            except (ValueError, TypeError):
                pass

        if "anthropic-ratelimit-tokens-remaining" in headers:
            try:
                info.tokens_remaining = int(headers["anthropic-ratelimit-tokens-remaining"])
            except (ValueError, TypeError):
                pass

        if "retry-after" in headers:
            try:
                info.retry_after_seconds = int(headers["retry-after"])
            except (ValueError, TypeError):
                pass

        return info

    @classmethod
    def _extract_openai_headers(cls, headers: Dict[str, str], info: RateLimitInfo) -> RateLimitInfo:
        """Extract OpenAI-specific rate limit headers."""
        if "x-ratelimit-remaining-requests" in headers:
            try:
                info.requests_remaining = int(headers["x-ratelimit-remaining-requests"])
            except (ValueError, TypeError):
                pass

        if "x-ratelimit-remaining-tokens" in headers:
            try:
                info.tokens_remaining = int(headers["x-ratelimit-remaining-tokens"])
            except (ValueError, TypeError):
                pass

        if "x-ratelimit-reset-timestamp" in headers:
            info.reset_time = headers["x-ratelimit-reset-timestamp"]

        return info

    @classmethod
    def _extract_groq_headers(cls, headers: Dict[str, str], info: RateLimitInfo) -> RateLimitInfo:
        """Extract Groq-specific rate limit headers."""
        if "x-ratelimit-requests-remaining" in headers:
            try:
                info.requests_remaining = int(headers["x-ratelimit-requests-remaining"])
            except (ValueError, TypeError):
                pass

        if "x-ratelimit-tokens-remaining" in headers:
            try:
                info.tokens_remaining = int(headers["x-ratelimit-tokens-remaining"])
            except (ValueError, TypeError):
                pass

        if "retry-after" in headers:
            try:
                info.retry_after_seconds = int(headers["retry-after"])
            except (ValueError, TypeError):
                pass

        return info

    @classmethod
    def _extract_gemini_headers(cls, headers: Dict[str, str], info: RateLimitInfo) -> RateLimitInfo:
        """Extract Gemini-specific rate limit headers."""
        if "x-ratelimit-remaining" in headers:
            try:
                info.requests_remaining = int(headers["x-ratelimit-remaining"])
            except (ValueError, TypeError):
                pass

        if "retry-after" in headers:
            try:
                info.retry_after_seconds = int(headers["retry-after"])
            except (ValueError, TypeError):
                pass

        return info

    @classmethod
    def _extract_from_body(cls, response: Any, info: RateLimitInfo) -> RateLimitInfo:
        """Extract rate limit info from response body (for providers like Cohere)."""
        try:
            if hasattr(response, "json"):
                data = response.json()
                if isinstance(data, dict):
                    # Cohere style
                    if "retry_after" in data:
                        try:
                            info.retry_after_seconds = int(data["retry_after"])
                        except (ValueError, TypeError):
                            pass
        except Exception:
            pass

        return info

    @classmethod
    def _extract_error_message(cls, response: Any) -> Optional[str]:
        """Extract error message from response."""
        try:
            if hasattr(response, "text"):
                return response.text[:200]  # Truncate for safety
            elif hasattr(response, "json"):
                data = response.json()
                if isinstance(data, dict):
                    return str(data.get("error", {}))[:200]
        except Exception:
            pass
        return None

    @classmethod
    def _detect_provider(cls, url: str) -> Optional[str]:
        """Detect provider from URL."""
        url_lower = url.lower()

        if "api.anthropic.com" in url_lower:
            return "anthropic"
        elif "api.openai.com" in url_lower:
            return "openai"
        elif "api.groq.com" in url_lower:
            return "groq"
        elif "api.cohere.com" in url_lower:
            return "cohere"
        elif "api.moonshot.cn" in url_lower:
            return "kimi"
        elif "api.minimax.chat" in url_lower:
            return "minimax"
        elif "api.portkey.ai" in url_lower:
            return "portkey"
        elif "generativelanguage.googleapis.com" in url_lower:
            return "gemini"
        elif "api.z.ai" in url_lower:
            return "z_ai"

        return None

    @classmethod
    def _calculate_default_backoff(cls, status_code: int) -> int:
        """Calculate default backoff seconds based on status code."""
        if status_code == 429:
            return 60  # 1 minute for rate limit
        elif status_code in [500, 502, 503]:
            return 30  # 30 seconds for server errors
        return 60

    @classmethod
    def should_retry(cls, response: Any) -> bool:
        """
        Check if request should be retried based on response.

        Returns True if response indicates a retryable error.
        """
        if not hasattr(response, "status_code"):
            return False

        # Rate limit errors
        if response.status_code == 429:
            return True

        # Server errors (retryable)
        if response.status_code in [500, 502, 503]:
            return True

        return False

    @classmethod
    def is_rate_limit_pattern(cls, text: str) -> bool:
        """Check if text matches rate limit error patterns."""
        text_lower = text.lower()
        for pattern in cls.RATE_LIMIT_PATTERNS:
            if re.search(pattern, text_lower):
                return True
        return False
