"""
Unit tests for the MetricsCollector class.

Tests metrics collection, timing calculations, and summary generation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_proxy import MetricsCollector


class TestMetricsCollectorInitialization:
    """Tests for MetricsCollector initialization."""

    def test_default_initialization(self):
        """Test MetricsCollector initializes with default values."""
        metrics = MetricsCollector()

        assert metrics.requests == 0
        assert metrics.total_original_tokens == 0
        assert metrics.total_compressed_tokens == 0
        assert metrics.total_compression_time_ms == 0
        assert metrics.compression_count == 0
        assert metrics.min_compression_time_ms == float('inf')
        assert metrics.max_compression_time_ms == 0

    def test_start_time_set(self):
        """Test start time is set on initialization."""
        before = time.time()
        metrics = MetricsCollector()
        after = time.time()

        assert before <= metrics.start_time <= after


class TestRecordRequest:
    """Tests for record_request method."""

    def test_record_single_request(self):
        """Test recording a single request."""
        metrics = MetricsCollector()
        stats = {
            "original_tokens": 1000,
            "compressed_tokens": 750,
        }

        metrics.record_request(stats)

        assert metrics.requests == 1
        assert metrics.total_original_tokens == 1000
        assert metrics.total_compressed_tokens == 750

    def test_record_multiple_requests(self):
        """Test recording multiple requests."""
        metrics = MetricsCollector()

        metrics.record_request({"original_tokens": 100, "compressed_tokens": 80})
        metrics.record_request({"original_tokens": 200, "compressed_tokens": 150})
        metrics.record_request({"original_tokens": 300, "compressed_tokens": 250})

        assert metrics.requests == 3
        assert metrics.total_original_tokens == 600
        assert metrics.total_compressed_tokens == 480

    def test_record_request_with_missing_stats(self):
        """Test recording request with missing stats."""
        metrics = MetricsCollector()

        metrics.record_request({})  # Empty stats
        metrics.record_request({"original_tokens": 100})  # Missing compressed_tokens
        metrics.record_request({"compressed_tokens": 50})  # Missing original_tokens

        assert metrics.requests == 3
        # Should handle missing keys gracefully (default to 0)

    def test_record_request_with_zero_tokens(self):
        """Test recording request with zero tokens."""
        metrics = MetricsCollector()

        metrics.record_request({"original_tokens": 0, "compressed_tokens": 0})

        assert metrics.requests == 1
        assert metrics.total_original_tokens == 0
        assert metrics.total_compressed_tokens == 0


class TestRecordCompressionTime:
    """Tests for record_compression_time method."""

    def test_record_single_compression_time(self):
        """Test recording a single compression time."""
        metrics = MetricsCollector()

        metrics.record_compression_time(5.5)

        assert metrics.compression_count == 1
        assert metrics.total_compression_time_ms == 5.5
        assert metrics.min_compression_time_ms == 5.5
        assert metrics.max_compression_time_ms == 5.5

    def test_record_multiple_compression_times(self):
        """Test recording multiple compression times."""
        metrics = MetricsCollector()

        metrics.record_compression_time(10.0)
        metrics.record_compression_time(5.0)
        metrics.record_compression_time(15.0)

        assert metrics.compression_count == 3
        assert metrics.total_compression_time_ms == 30.0
        assert metrics.min_compression_time_ms == 5.0
        assert metrics.max_compression_time_ms == 15.0

    def test_min_compression_time_updated(self):
        """Test min compression time is correctly updated."""
        metrics = MetricsCollector()

        metrics.record_compression_time(10.0)
        assert metrics.min_compression_time_ms == 10.0

        metrics.record_compression_time(5.0)
        assert metrics.min_compression_time_ms == 5.0

        metrics.record_compression_time(8.0)
        assert metrics.min_compression_time_ms == 5.0  # Should stay at minimum

    def test_max_compression_time_updated(self):
        """Test max compression time is correctly updated."""
        metrics = MetricsCollector()

        metrics.record_compression_time(5.0)
        assert metrics.max_compression_time_ms == 5.0

        metrics.record_compression_time(10.0)
        assert metrics.max_compression_time_ms == 10.0

        metrics.record_compression_time(8.0)
        assert metrics.max_compression_time_ms == 10.0  # Should stay at maximum

    def test_record_zero_compression_time(self):
        """Test recording zero compression time."""
        metrics = MetricsCollector()

        metrics.record_compression_time(0.0)

        assert metrics.compression_count == 1
        assert metrics.total_compression_time_ms == 0.0
        assert metrics.min_compression_time_ms == 0.0
        assert metrics.max_compression_time_ms == 0.0

    def test_record_very_small_compression_time(self):
        """Test recording very small compression time."""
        metrics = MetricsCollector()

        metrics.record_compression_time(0.001)

        assert metrics.compression_count == 1
        assert metrics.total_compression_time_ms == 0.001


class TestGetSummary:
    """Tests for get_summary method."""

    def test_empty_summary(self):
        """Test summary with no recorded data."""
        metrics = MetricsCollector()

        summary = metrics.get_summary()

        assert summary["total_requests"] == 0
        assert summary["total_original_tokens"] == 0
        assert summary["total_compressed_tokens"] == 0
        assert summary["total_tokens_saved"] == 0
        assert summary["average_reduction_pct"] == 0
        assert summary["uptime_seconds"] >= 0
        assert summary["tokens_saved_per_minute"] == 0

    def test_summary_with_data(self):
        """Test summary with recorded data."""
        metrics = MetricsCollector()

        metrics.record_request({"original_tokens": 1000, "compressed_tokens": 750})
        metrics.record_compression_time(5.0)

        summary = metrics.get_summary()

        assert summary["total_requests"] == 1
        assert summary["total_original_tokens"] == 1000
        assert summary["total_compressed_tokens"] == 750
        assert summary["total_tokens_saved"] == 250
        assert summary["average_reduction_pct"] == 25.0

    def test_tokens_saved_calculation(self):
        """Test tokens saved calculation."""
        metrics = MetricsCollector()

        metrics.record_request({"original_tokens": 500, "compressed_tokens": 400})
        metrics.record_request({"original_tokens": 300, "compressed_tokens": 200})

        summary = metrics.get_summary()

        assert summary["total_tokens_saved"] == 200  # (500-400) + (300-200)

    def test_average_reduction_calculation(self):
        """Test average reduction percentage calculation."""
        metrics = MetricsCollector()

        # First request: 50% reduction
        metrics.record_request({"original_tokens": 100, "compressed_tokens": 50})
        # Second request: 25% reduction
        metrics.record_request({"original_tokens": 100, "compressed_tokens": 75})

        summary = metrics.get_summary()

        # Average: (50 + 25) / 200 * 100 = 37.5%
        assert summary["average_reduction_pct"] == 37.5

    def test_tokens_saved_per_minute(self):
        """Test tokens saved per minute calculation."""
        metrics = MetricsCollector()
        # Mock start time to be 1 minute ago
        metrics.start_time = time.time() - 60

        metrics.record_request({"original_tokens": 600, "compressed_tokens": 300})

        summary = metrics.get_summary()

        # 300 tokens saved in 1 minute = 300 per minute
        assert summary["tokens_saved_per_minute"] == 300.0

    def test_uptime_calculation(self):
        """Test uptime calculation."""
        metrics = MetricsCollector()
        # Mock start time to be 5 seconds ago
        metrics.start_time = time.time() - 5

        summary = metrics.get_summary()

        assert 4 <= summary["uptime_seconds"] <= 6

    def test_summary_with_compression_stats(self):
        """Test summary includes compression timing stats."""
        metrics = MetricsCollector()

        metrics.record_compression_time(5.0)
        metrics.record_compression_time(10.0)
        metrics.record_compression_time(15.0)

        summary = metrics.get_summary()

        assert "compression_stats" in summary
        assert summary["compression_stats"]["total_compressions"] == 3
        assert summary["compression_stats"]["avg_compression_time_ms"] == 10.0
        assert summary["compression_stats"]["min_compression_time_ms"] == 5.0
        assert summary["compression_stats"]["max_compression_time_ms"] == 15.0
        assert summary["compression_stats"]["total_compression_time_ms"] == 30.0

    def test_summary_without_compression_stats(self):
        """Test summary when no compression times recorded."""
        metrics = MetricsCollector()

        metrics.record_request({"original_tokens": 100, "compressed_tokens": 80})

        summary = metrics.get_summary()

        # Should not include compression_stats if no compressions
        assert "compression_stats" not in summary

    def test_average_compression_time_calculation(self):
        """Test average compression time calculation."""
        metrics = MetricsCollector()

        metrics.record_compression_time(10.0)
        metrics.record_compression_time(20.0)
        metrics.record_compression_time(30.0)

        summary = metrics.get_summary()

        assert summary["compression_stats"]["avg_compression_time_ms"] == 20.0

    def test_rounding_in_summary(self):
        """Test values are properly rounded in summary."""
        metrics = MetricsCollector()
        metrics.start_time = time.time() - 3.33333

        metrics.record_request({"original_tokens": 1000, "compressed_tokens": 333})

        summary = metrics.get_summary()

        # Check rounding
        assert isinstance(summary["average_reduction_pct"], float)
        assert len(str(summary["average_reduction_pct"]).split('.')[-1]) <= 1

        assert isinstance(summary["uptime_seconds"], float)
        assert len(str(summary["uptime_seconds"]).split('.')[-1]) <= 1


class TestEdgeCases:
    """Tests for edge cases."""

    def test_division_by_zero_protection(self):
        """Test protection against division by zero."""
        metrics = MetricsCollector()

        # No requests made, original_tokens is 0
        summary = metrics.get_summary()

        assert summary["average_reduction_pct"] == 0

    def test_very_large_numbers(self):
        """Test handling of very large numbers."""
        metrics = MetricsCollector()

        large_number = 10**9  # 1 billion
        metrics.record_request({
            "original_tokens": large_number,
            "compressed_tokens": large_number // 2
        })

        summary = metrics.get_summary()

        assert summary["total_tokens_saved"] == large_number // 2

    def test_negative_tokens_not_expected(self):
        """Test behavior with negative token counts (shouldn't happen but test anyway)."""
        metrics = MetricsCollector()

        metrics.record_request({"original_tokens": 100, "compressed_tokens": 150})  # More compressed than original

        summary = metrics.get_summary()

        # Tokens saved would be negative
        assert summary["total_tokens_saved"] == -50
        # Average reduction would be negative
        assert summary["average_reduction_pct"] == -50.0

    def test_zero_elapsed_time(self):
        """Test handling when elapsed time is zero."""
        metrics = MetricsCollector()
        metrics.start_time = time.time()  # Current time, so elapsed is ~0

        metrics.record_request({"original_tokens": 100, "compressed_tokens": 50})

        summary = metrics.get_summary()

        # Should handle near-zero elapsed time
        assert summary["tokens_saved_per_minute"] >= 0


class TestIntegration:
    """Integration tests for MetricsCollector."""

    def test_full_workflow(self):
        """Test full workflow with multiple requests and compression times."""
        metrics = MetricsCollector()

        # Simulate multiple requests
        for i in range(5):
            metrics.record_request({
                "original_tokens": 1000 + i * 100,
                "compressed_tokens": 800 + i * 80
            })
            metrics.record_compression_time(5.0 + i)

        summary = metrics.get_summary()

        assert summary["total_requests"] == 5
        assert summary["total_original_tokens"] == 1000 + 1100 + 1200 + 1300 + 1400
        assert summary["total_compressed_tokens"] == 800 + 880 + 960 + 1040 + 1120
        assert summary["compression_stats"]["total_compressions"] == 5
        assert summary["compression_stats"]["avg_compression_time_ms"] == 7.0  # (5+6+7+8+9)/5

    def test_metrics_consistency(self):
        """Test that metrics remain consistent after multiple operations."""
        metrics = MetricsCollector()

        # Record some data
        metrics.record_request({"original_tokens": 1000, "compressed_tokens": 800})
        metrics.record_compression_time(10.0)

        # Get summary
        summary1 = metrics.get_summary()

        # Get summary again
        summary2 = metrics.get_summary()

        # Should be identical
        assert summary1 == summary2

    def test_thread_safety_concern(self):
        """Note: MetricsCollector is not thread-safe by design."""
        # This test documents that MetricsCollector should be used in a single-threaded context
        # or external synchronization should be provided
        metrics = MetricsCollector()

        # In real usage, the handler is per-request but metrics is shared
        # The ThreadingHTTPServer handles one request per thread
        # This could lead to race conditions in production

        # For now, we just verify basic functionality
        metrics.record_request({"original_tokens": 100, "compressed_tokens": 80})
        assert metrics.requests == 1
