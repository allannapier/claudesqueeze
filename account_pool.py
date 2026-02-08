#!/usr/bin/env python3
"""
Account Pool Manager
====================

Manages multiple API accounts with thread-safe selection and state tracking
for rate limit backoff and failover.
"""

import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List
from queue import Queue, Empty


@dataclass
class AccountConfig:
    """Configuration for a single API account."""
    id: str
    name: str
    api_key: str
    endpoint: Optional[str] = None  # Per-account endpoint (overrides global target_api.url)
    priority: int = 1
    max_retries: int = 3
    retry_delay_ms: int = 1000

    # Runtime state (not persisted)
    backoff_until: Optional[datetime] = None
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None

    @property
    def is_available(self) -> bool:
        """Check if account is available (not in backoff)."""
        if self.backoff_until is None:
            return True
        return datetime.now() >= self.backoff_until

    def mark_rate_limited(self, backoff_seconds: int = 60):
        """Mark account as rate-limited with backoff."""
        self.backoff_until = datetime.now() + timedelta(seconds=backoff_seconds)
        self.failure_count += 1
        self.last_failure_time = datetime.now()

    def mark_success(self):
        """Mark account as successful (reset failure state)."""
        self.backoff_until = None
        self.failure_count = 0
        self.last_failure_time = None

    def get_backoff_remaining_seconds(self) -> float:
        """Get remaining backoff time in seconds."""
        if self.backoff_until is None:
            return 0
        remaining = (self.backoff_until - datetime.now()).total_seconds()
        return max(0, remaining)


class AccountPool:
    """
    Thread-safe pool of API accounts with selection strategies.

    Strategies:
    - priority: Select lowest priority number (highest priority first)
    - round_robin: Rotate through accounts in order
    """

    def __init__(self, accounts_config: List[dict], strategy: str = "priority"):
        """
        Initialize account pool.

        Args:
            accounts_config: List of account dicts from config file
            strategy: Selection strategy - "priority" or "round_robin"
        """
        self.strategy = strategy
        self.accounts: List[AccountConfig] = []
        self.lock = threading.RLock()
        self._round_robin_index = 0

        # Load accounts from config
        self._load_accounts(accounts_config)

    def _load_accounts(self, accounts_config: List[dict]):
        """Load accounts from config, expanding environment variables."""
        for acc_config in accounts_config:
            api_key = acc_config.get("api_key", "")

            # Expand environment variables
            if isinstance(api_key, str) and api_key.startswith("${") and api_key.endswith("}"):
                env_var = api_key[2:-1]
                api_key = os.environ.get(env_var, "")

            if not api_key:
                print(f"[WARN] Skipping account '{acc_config.get('id')}' - no API key")
                continue

            account = AccountConfig(
                id=acc_config.get("id", f"account_{len(self.accounts) + 1}"),
                name=acc_config.get("name", "Unnamed Account"),
                api_key=api_key,
                endpoint=acc_config.get("endpoint"),  # Per-account endpoint
                priority=acc_config.get("priority", 1),
                max_retries=acc_config.get("max_retries", 3),
                retry_delay_ms=acc_config.get("retry_delay_ms", 1000),
            )
            self.accounts.append(account)

        # Sort by priority if using priority strategy
        if self.strategy == "priority":
            self.accounts.sort(key=lambda a: a.priority)

        if not self.accounts:
            raise ValueError("No valid accounts configured")

    def select_account(self, exclude_ids: Optional[set] = None) -> Optional[AccountConfig]:
        """
        Select the next available account based on strategy.

        Args:
            exclude_ids: Set of account IDs to exclude from selection

        Returns:
            AccountConfig or None if no accounts available
        """
        with self.lock:
            exclude_ids = exclude_ids or set()

            # Clear expired backoffs
            self._clear_expired_backoffs()

            available = [a for a in self.accounts if a.is_available and a.id not in exclude_ids]

            if not available:
                return None

            if self.strategy == "priority":
                # Return first available (already sorted by priority)
                return available[0]
            elif self.strategy == "round_robin":
                # Find next available starting from current index
                for _ in range(len(self.accounts)):
                    account = self.accounts[self._round_robin_index]
                    self._round_robin_index = (self._round_robin_index + 1) % len(self.accounts)

                    if account.is_available and account.id not in exclude_ids:
                        return account
                return available[0]
            else:
                # Default to first available
                return available[0]

    def get_next_available_account(self, current_account_id: str) -> Optional[AccountConfig]:
        """
        Get the next available account excluding the current one.

        Args:
            current_account_id: ID of the account to exclude

        Returns:
            AccountConfig or None if no other accounts available
        """
        return self.select_account(exclude_ids={current_account_id})

    def mark_rate_limited(self, account_id: str, backoff_seconds: int = 60):
        """Mark an account as rate-limited."""
        with self.lock:
            account = self._get_account_by_id(account_id)
            if account:
                account.mark_rate_limited(backoff_seconds)

    def mark_success(self, account_id: str):
        """Mark an account as successful (reset failure state)."""
        with self.lock:
            account = self._get_account_by_id(account_id)
            if account:
                account.mark_success()

    def get_account_by_id(self, account_id: str) -> Optional[AccountConfig]:
        """Get account by ID (without lock)."""
        return self._get_account_by_id(account_id)

    def _get_account_by_id(self, account_id: str) -> Optional[AccountConfig]:
        """Get account by ID (internal, assumes lock held)."""
        for account in self.accounts:
            if account.id == account_id:
                return account
        return None

    def _clear_expired_backoffs(self):
        """Clear expired backoff periods for all accounts."""
        now = datetime.now()
        for account in self.accounts:
            if account.backoff_until and now >= account.backoff_until:
                account.backoff_until = None

    def get_all_accounts(self) -> List[AccountConfig]:
        """Get all accounts (for status/logging)."""
        with self.lock:
            return list(self.accounts)

    def get_status(self) -> dict:
        """Get status of all accounts."""
        with self.lock:
            status = {
                "total_accounts": len(self.accounts),
                "available_accounts": sum(1 for a in self.accounts if a.is_available),
                "accounts": []
            }
            for account in self.accounts:
                status["accounts"].append({
                    "id": account.id,
                    "name": account.name,
                    "priority": account.priority,
                    "available": account.is_available,
                    "backoff_remaining_seconds": account.get_backoff_remaining_seconds(),
                    "failure_count": account.failure_count,
                })
            return status


def expand_env_vars(value: str) -> str:
    """
    Expand environment variables in a string.

    Supports ${VAR_NAME} syntax.
    """
    if not isinstance(value, str):
        return value

    pattern = r'\$\{([^}]+)\}'

    def replacer(match):
        env_var = match.group(1)
        return os.environ.get(env_var, match.group(0))

    return re.sub(pattern, replacer, value)
