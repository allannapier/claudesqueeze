#!/usr/bin/env python3
"""
Failover Handler
================

Orchestrates automatic account failover on rate limit errors with
exponential backoff and retry logic.
"""

import time
import logging
from typing import Callable, Any, Optional
from account_pool import AccountPool, AccountConfig
from rate_limiter import RateLimiter, RateLimitInfo


logger = logging.getLogger(__name__)


class AllAccountsExhaustedError(Exception):
    """Raised when all accounts are rate-limited or unavailable."""

    def __init__(self, status: dict):
        self.status = status
        super().__init__(f"All accounts exhausted. Status: {status}")


class FailoverHandler:
    """
    Handles automatic failover between API accounts.

    Features:
    - Automatic retry on 429 errors
    - Exponential backoff
    - Account state tracking
    - Configurable retry limits
    """

    def __init__(
        self,
        account_pool: AccountPool,
        max_retries: int = 3,
        backoff_multiplier: float = 2.0,
        max_backoff_seconds: int = 300,
        default_backoff_seconds: int = 60,
    ):
        """
        Initialize failover handler.

        Args:
            account_pool: AccountPool instance
            max_retries: Maximum number of retries per request
            backoff_multiplier: Multiplier for exponential backoff
            max_backoff_seconds: Maximum backoff time
            default_backoff_seconds: Default backoff when not provided by API
        """
        self.account_pool = account_pool
        self.max_retries = max_retries
        self.backoff_multiplier = backoff_multiplier
        self.max_backoff_seconds = max_backoff_seconds
        self.default_backoff_seconds = default_backoff_seconds

    def execute_with_failover(
        self,
        request_fn: Callable[[AccountConfig, Any, dict], Any],
        data: Any,
        headers: dict,
        request_url: str = "",
    ) -> Any:
        """
        Execute request with automatic failover on 429 errors.

        Args:
            request_fn: Function to execute request (receives account, data, headers)
            data: Request body/data
            headers: Request headers
            request_url: URL being requested (for provider detection)

        Returns:
            Response from successful request

        Raises:
            AllAccountsExhaustedError: When all accounts are unavailable
        """
        current_account = self.account_pool.select_account()
        excluded_accounts = set()
        attempt = 0
        last_error = None

        while attempt < self.max_retries and current_account is not None:
            attempt += 1

            logger.info(f"[FAILOVER] Attempt {attempt}/{self.max_retries} using account: {current_account.id}")

            try:
                # Execute request with current account's API key
                request_headers = self._inject_api_key(headers, current_account.api_key)
                response = request_fn(current_account, data, request_headers)

                # Check if response indicates rate limiting
                rate_info = RateLimiter.detect_from_response(response, request_url)

                if rate_info.is_rate_limited:
                    # Mark account as rate-limited
                    backoff = rate_info.retry_after_seconds or self.default_backoff_seconds
                    backoff = min(backoff * (self.backoff_multiplier ** (attempt - 1)), self.max_backoff_seconds)

                    logger.warning(
                        f"[FAILOVER] Account {current_account.id} rate-limited. "
                        f"Backing off for {backoff}s"
                    )

                    self.account_pool.mark_rate_limited(current_account.id, backoff)
                    excluded_accounts.add(current_account.id)

                    # Get next available account
                    current_account = self.account_pool.get_next_available_account(current_account.id)

                    if current_account is None:
                        # All accounts exhausted
                        status = self.account_pool.get_status()
                        logger.error(f"[FAILOVER] All accounts exhausted after {attempt} attempts")
                        raise AllAccountsExhaustedError(status)

                    # Continue to next attempt with new account
                    last_error = f"Rate limited: {rate_info.error_message}"
                    continue

                # Success - mark account as healthy and return response
                self.account_pool.mark_success(current_account.id)
                logger.info(f"[FAILOVER] Success using account: {current_account.id}")

                return response

            except AllAccountsExhaustedError:
                # Re-raise immediately
                raise
            except Exception as e:
                logger.error(f"[FAILOVER] Request failed with account {current_account.id}: {e}")
                last_error = str(e)

                # Don't retry on certain errors
                if self._is_non_retryable_error(e):
                    raise

                # Try next account
                excluded_accounts.add(current_account.id)
                current_account = self.account_pool.get_next_available_account(current_account.id)

                if current_account is None:
                    status = self.account_pool.get_status()
                    logger.error(f"[FAILOVER] All accounts exhausted after {attempt} attempts")
                    raise AllAccountsExhaustedError(status) from e

        # If we get here, all retries exhausted
        status = self.account_pool.get_status()
        logger.error(f"[FAILOVER] Max retries ({self.max_retries}) exhausted")
        raise AllAccountsExhaustedError(status)

    def _inject_api_key(self, headers: dict, api_key: str) -> dict:
        """
        Inject API key into request headers.

        Handles different provider formats:
        - Anthropic: x-api-key
        - OpenAI: Authorization: Bearer <key>
        - etc.
        """
        # Create a copy to avoid modifying original
        new_headers = headers.copy()

        # Remove existing API key headers
        new_headers.pop("x-api-key", None)
        new_headers.pop("authorization", None)
        new_headers.pop("Authorization", None)

        # Inject new API key
        # Most providers use x-api-key for Anthropic-compatible APIs
        new_headers["x-api-key"] = api_key

        return new_headers

    def _is_non_retryable_error(self, error: Exception) -> bool:
        """Check if error is non-retryable (e.g., auth, invalid request)."""
        error_str = str(error).lower()

        # Authentication errors
        if any(pattern in error_str for pattern in ["unauthorized", "authentication", "invalid api key", "401"]):
            return True

        # Invalid request errors
        if any(pattern in error_str for pattern in ["invalid request", "bad request", "400"]):
            return True

        # Network errors that might be retryable
        if isinstance(error, (ConnectionError, TimeoutError)):
            return False

        return False

    def get_status(self) -> dict:
        """Get current status of all accounts."""
        return self.account_pool.get_status()

    def execute_stream_with_failover(
        self,
        request_fn: Callable[[AccountConfig, Any, dict], Any],
        data: Any,
        headers: dict,
        request_url: str = "",
    ) -> Any:
        """
        Execute streaming request with automatic failover.

        For streaming, we attempt the request and if it fails immediately
        (before streaming starts), we retry with a different account.

        Once streaming starts, failures are not retried as the context
        would be lost.
        """
        current_account = self.account_pool.select_account()
        excluded_accounts = set()
        attempt = 0

        while attempt < self.max_retries and current_account is not None:
            attempt += 1

            logger.info(f"[STREAM_FAILOVER] Attempt {attempt}/{self.max_retries} using account: {current_account.id}")

            try:
                request_headers = self._inject_api_key(headers, current_account.api_key)

                # For streaming, we return the response object directly
                # The caller handles the streaming
                response = request_fn(current_account, data, request_headers)

                # Check if initial response indicates rate limiting
                rate_info = RateLimiter.detect_from_response(response, request_url)

                if rate_info.is_rate_limited:
                    backoff = rate_info.retry_after_seconds or self.default_backoff_seconds
                    logger.warning(f"[STREAM_FAILOVER] Account {current_account.id} rate-limited")

                    self.account_pool.mark_rate_limited(current_account.id, backoff)
                    excluded_accounts.add(current_account.id)

                    current_account = self.account_pool.get_next_available_account(current_account.id)

                    if current_account is None:
                        status = self.account_pool.get_status()
                        raise AllAccountsExhaustedError(status)

                    continue

                # Success - return response for streaming
                logger.info(f"[STREAM_FAILOVER] Success using account: {current_account.id}")
                return response

            except AllAccountsExhaustedError:
                raise
            except Exception as e:
                logger.error(f"[STREAM_FAILOVER] Request failed: {e}")

                if self._is_non_retryable_error(e):
                    raise

                excluded_accounts.add(current_account.id)
                current_account = self.account_pool.get_next_available_account(current_account.id)

                if current_account is None:
                    status = self.account_pool.get_status()
                    raise AllAccountsExhaustedError(status) from e

        status = self.account_pool.get_status()
        raise AllAccountsExhaustedError(status)
