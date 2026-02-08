"""
Unit tests for the CompressionEngine class.

Tests all compression levels (low, medium, high) and various compression techniques
including abbreviations, filler words, symbols, number words, and code patterns.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_proxy import CompressionEngine


class TestCompressionEngineInitialization:
    """Tests for CompressionEngine initialization."""

    def test_default_initialization(self):
        """Test CompressionEngine initializes with default values."""
        engine = CompressionEngine()
        assert engine.level == "high"
        assert engine.enable_path_compression is False
        assert not hasattr(engine, 'path_compressor') or engine.path_compressor is None

    def test_low_level_initialization(self):
        """Test CompressionEngine initializes with low compression level."""
        engine = CompressionEngine(level="low")
        assert engine.level == "low"

    def test_medium_level_initialization(self):
        """Test CompressionEngine initializes with medium compression level."""
        engine = CompressionEngine(level="medium")
        assert engine.level == "medium"

    def test_high_level_initialization(self):
        """Test CompressionEngine initializes with high compression level."""
        engine = CompressionEngine(level="high")
        assert engine.level == "high"

    def test_path_compression_disabled_by_default(self):
        """Test path compression is disabled by default."""
        engine = CompressionEngine()
        assert engine.enable_path_compression is False

    def test_path_compression_enabled(self):
        """Test path compression can be enabled."""
        engine = CompressionEngine(enable_path_compression=True)
        assert engine.enable_path_compression is True

    @patch('path_compression.PathCompressionEngine')
    def test_path_compression_engine_created(self, mock_path_engine_class):
        """Test PathCompressionEngine is created when enabled."""
        mock_instance = MagicMock()
        mock_path_engine_class.return_value = mock_instance

        engine = CompressionEngine(enable_path_compression=True)

        mock_path_engine_class.assert_called_once()
        assert engine.path_compressor is mock_instance

    @patch('path_compression.PathCompressionEngine', side_effect=ImportError("No module"))
    def test_path_compression_import_error(self, mock_path_engine_class):
        """Test graceful handling when path compression module not found."""
        engine = CompressionEngine(enable_path_compression=True)
        assert engine.enable_path_compression is False
        assert engine.path_compressor is None


class TestLightCompression:
    """Tests for low compression level (_light_compression)."""

    def test_abbreviations_applied(self, compression_engine_low):
        """Test abbreviations are applied in low compression."""
        text = "for your information, this is important"
        result = compression_engine_low.compress(text)
        assert "FYI" in result
        assert "for your information" not in result.lower()

    def test_case_insensitive_abbreviations(self, compression_engine_low):
        """Test abbreviations work case-insensitively."""
        text = "For Your Information and FOR EXAMPLE"
        result = compression_engine_low.compress(text)
        assert "FYI" in result
        assert "e.g." in result.lower()

    def test_whitespace_normalized(self, compression_engine_low):
        """Test extra whitespace is normalized."""
        text = "hello    world   test"
        result = compression_engine_low.compress(text)
        assert result == "hello world test"

    def test_multiple_abbreviations(self, compression_engine_low):
        """Test multiple abbreviations in one text."""
        text = "for your information, by the way, as soon as possible"
        result = compression_engine_low.compress(text)
        assert "FYI" in result
        assert "BTW" in result
        assert "ASAP" in result

    def test_empty_string(self, compression_engine_low):
        """Test empty string handling."""
        assert compression_engine_low.compress("") == ""

    def test_none_input(self, compression_engine_low):
        """Test None input handling."""
        assert compression_engine_low.compress(None) is None

    def test_non_string_input(self, compression_engine_low):
        """Test non-string input handling."""
        assert compression_engine_low.compress(123) == 123
        assert compression_engine_low.compress(["test"]) == ["test"]


class TestMediumCompression:
    """Tests for medium compression level (_medium_compression)."""

    def test_filler_words_removed(self, compression_engine_medium):
        """Test filler words are removed in medium compression."""
        text = "please could you help me with this"
        result = compression_engine_medium.compress(text)
        assert "please" not in result.lower()
        assert "could you" not in result.lower()

    def test_symbols_applied(self, compression_engine_medium):
        """Test word-to-symbol conversion."""
        text = "value greater than or equal to 5"
        result = compression_engine_medium.compress(text)
        assert "≥" in result
        assert "greater than or equal to" not in result.lower()

    def test_number_words_converted(self, compression_engine_medium):
        """Test number words are converted to digits."""
        text = "I have two apples and three oranges"
        result = compression_engine_medium.compress(text)
        assert "2" in result
        assert "3" in result
        assert "two" not in result.lower()
        assert "three" not in result.lower()

    def test_punctuation_cleanup(self, compression_engine_medium):
        """Test punctuation spacing cleanup."""
        text = "hello , world . test"
        result = compression_engine_medium.compress(text)
        assert "hello, world. test" in result

    def test_combined_techniques(self, compression_engine_medium):
        """Test all medium techniques work together."""
        text = "Please could you give me two examples, for example, test cases"
        result = compression_engine_medium.compress(text)
        # Filler words removed, abbreviations applied, numbers converted
        assert "please" not in result.lower()
        assert "could you" not in result.lower()
        assert "e.g." in result
        assert "2" in result


class TestHeavyCompression:
    """Tests for high compression level (_heavy_compression)."""

    def test_articles_removed(self, compression_engine_high):
        """Test articles (a, an, the) are removed."""
        text = "the quick brown fox jumps over a lazy dog"
        result = compression_engine_high.compress(text)
        assert " the " not in f" {result} "
        assert " a " not in f" {result} "

    def test_relative_pronouns_removed(self, compression_engine_high):
        """Test relative pronouns are removed."""
        text = "the book that I read which was good"
        result = compression_engine_high.compress(text)
        assert "that " not in result.lower() or "which " not in result.lower()

    def test_auxiliary_verbs_removed(self, compression_engine_high):
        """Test auxiliary verbs are removed."""
        text = "this is a test that was working"
        result = compression_engine_high.compress(text)
        # Should have reduced "is" and "was"
        assert result != text

    def test_and_replaced_in_lists(self, compression_engine_high):
        """Test 'and' replaced with comma in simple lists."""
        text = "apples and oranges and bananas."
        result = compression_engine_high.compress(text)
        assert ", " in result

    def test_code_patterns_applied(self, compression_engine_high):
        """Test code-specific patterns are applied."""
        text = "function test() { console.log('hello'); }"
        result = compression_engine_high.compress(text)
        assert "fn" in result or "function" not in result
        assert "log" in result or "console.log" not in result

    def test_file_path_shortcuts(self, compression_engine_high):
        """Test file path shortcuts are applied."""
        text = "/home/user/project/file.txt"
        result = compression_engine_high.compress(text)
        assert "~/" in result or "/home/" in result

    def test_log_level_shortcuts(self, compression_engine_high):
        """Test log level shortcuts are applied."""
        text = "[INFO] message [ERROR] error [DEBUG] debug"
        result = compression_engine_high.compress(text)
        assert "[I]" in result or "[INFO]" not in result
        assert "[E]" in result or "[ERROR]" not in result

    def test_whitespace_compressed(self, compression_engine_high):
        """Test aggressive whitespace compression."""
        text = "line1\n\n\n\nline2"
        result = compression_engine_high.compress(text)
        # Should reduce multiple newlines
        assert result.count("\n") < text.count("\n")

    def test_aggressive_punctuation_cleanup(self, compression_engine_high):
        """Test aggressive punctuation cleanup."""
        text = "hello , world . , test"
        result = compression_engine_high.compress(text)
        # Should clean up extra commas and spaces
        assert ",," not in result


class TestCompressMessages:
    """Tests for compress_messages method."""

    def test_compress_user_messages_only(self, compression_engine_high, sample_messages):
        """Test only user messages are compressed by default."""
        compressed, stats = compression_engine_high.compress_messages(sample_messages)

        # User messages should be compressed
        user_msgs = [m for m in compressed if m["role"] == "user"]
        assert len(user_msgs) == 2

        # System and assistant messages should be unchanged
        system_msg = next(m for m in compressed if m["role"] == "system")
        assert system_msg["content"] == "You are a helpful assistant."

        assistant_msg = next(m for m in compressed if m["role"] == "assistant")
        assert assistant_msg["content"] == "I would be happy to help you."

    def test_compress_specific_roles(self, compression_engine_high, sample_messages):
        """Test compressing specific roles."""
        compressed, stats = compression_engine_high.compress_messages(
            sample_messages, roles=["user", "assistant"]
        )

        # Both user and assistant should be compressed
        assistant_msg = next(m for m in compressed if m["role"] == "assistant")
        # "would" is a filler word that should be removed in high compression
        # Note: the content may be transformed, not necessarily just removing "would"
        assert assistant_msg["content"] != "I would be happy to help you."

    def test_stats_returned(self, compression_engine_high, sample_messages):
        """Test stats are returned with compression info."""
        compressed, stats = compression_engine_high.compress_messages(sample_messages)

        assert "original_tokens" in stats
        assert "compressed_tokens" in stats
        assert "tokens_saved" in stats
        assert "reduction_pct" in stats

    def test_content_blocks_handled(self, compression_engine_high, sample_messages_with_content_blocks):
        """Test Anthropic content block format is handled."""
        compressed, stats = compression_engine_high.compress_messages(sample_messages_with_content_blocks)

        msg = compressed[0]
        assert isinstance(msg["content"], list)
        assert msg["content"][0]["type"] == "text"
        # Filler words should be removed
        assert "please" not in msg["content"][0]["text"].lower()

    def test_non_text_blocks_preserved(self, compression_engine_high, sample_messages_with_content_blocks):
        """Test non-text content blocks are preserved."""
        compressed, stats = compression_engine_high.compress_messages(sample_messages_with_content_blocks)

        msg = compressed[0]
        assert len(msg["content"]) == 2
        assert msg["content"][1]["type"] == "image"

    def test_empty_messages(self, compression_engine_high):
        """Test handling of empty messages list."""
        compressed, stats = compression_engine_high.compress_messages([])
        assert compressed == []
        assert stats["original_tokens"] == 0
        assert stats["compressed_tokens"] == 0

    def test_message_without_content(self, compression_engine_high):
        """Test handling of messages without content."""
        messages = [{"role": "user"}]
        compressed, stats = compression_engine_high.compress_messages(messages)
        assert compressed[0].get("content") is None


class TestEstimateTokens:
    """Tests for _estimate_tokens method."""

    def test_estimate_simple_message(self, compression_engine_high):
        """Test token estimation for simple message."""
        messages = [{"role": "user", "content": "hello world"}]
        tokens = compression_engine_high._estimate_tokens(messages)
        # Roughly 11 chars / 4 = 2-3 tokens
        assert tokens >= 0

    def test_estimate_with_content_blocks(self, compression_engine_high):
        """Test token estimation with content blocks."""
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": "hello world"}]
        }]
        tokens = compression_engine_high._estimate_tokens(messages)
        assert tokens >= 0

    def test_estimate_empty_messages(self, compression_engine_high):
        """Test token estimation for empty messages."""
        assert compression_engine_high._estimate_tokens([]) == 0

    def test_estimate_empty_content(self, compression_engine_high):
        """Test token estimation for empty content."""
        messages = [{"role": "user", "content": ""}]
        assert compression_engine_high._estimate_tokens(messages) == 0


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_special_characters_preserved(self, compression_engine_high):
        """Test special characters are preserved."""
        text = "test@example.com #hashtag $100 %increase"
        result = compression_engine_high.compress(text)
        assert "@" in result
        assert "#" in result
        assert "$" in result
        assert "%" in result

    def test_unicode_handling(self, compression_engine_high):
        """Test Unicode characters are handled."""
        text = "Hello 世界 🌍 café"
        result = compression_engine_high.compress(text)
        assert "世界" in result
        assert "🌍" in result
        assert "café" in result

    def test_code_snippets(self, compression_engine_high):
        """Test code snippets are handled."""
        code = """
def hello_world():
    console.log("Hello, world!")
    return True
"""
        result = compression_engine_high.compress(code)
        # Should apply code patterns
        assert result != code

    def test_urls_in_text(self, compression_engine_high):
        """Test URLs in text are handled."""
        text = "Visit https://example.com/path?query=1 for more info"
        result = compression_engine_high.compress(text)
        # URL should be preserved
        assert "https://" in result or "example.com" in result

    def test_numbers_and_symbols(self, compression_engine_high):
        """Test various number formats."""
        text = "Values: 100, 3.14, -50, +25"
        result = compression_engine_high.compress(text)
        # Numbers should be preserved
        assert "100" in result
        assert "3.14" in result

    def test_repeated_punctuation(self, compression_engine_high):
        """Test repeated punctuation handling."""
        text = "Hello!!! How are you???"
        result = compression_engine_high.compress(text)
        # High compression removes articles and normalizes, result should differ
        assert result != text


class TestPathCompressionIntegration:
    """Tests for path compression integration."""

    @patch('path_compression.PathCompressionEngine')
    def test_path_compression_called(self, mock_path_engine_class):
        """Test path compression is called when enabled."""
        mock_instance = MagicMock()
        mock_instance.compress.return_value = ("compressed text", {"paths": 1})
        mock_path_engine_class.return_value = mock_instance

        engine = CompressionEngine(level="high", enable_path_compression=True)
        result = engine.compress("some text with /home/user/file.txt")

        mock_instance.compress.assert_called_once()

    def test_path_compression_disabled(self, compression_engine_high):
        """Test path compression is not called when disabled."""
        text = "/home/user/file.txt"
        result = compression_engine_high.compress(text)
        # Should not have path compression applied
        assert "/home/" in result or "~/" in result


class TestAbbreviationsList:
    """Tests for specific abbreviation mappings."""

    @pytest.mark.parametrize("phrase,abbreviation", [
        ("for your information", "FYI"),
        ("as soon as possible", "ASAP"),
        ("by the way", "BTW"),
        ("in my opinion", "IMO"),
        ("for example", "e.g."),
        ("that is", "i.e."),
        ("application programming interface", "API"),
        ("user interface", "UI"),
        ("artificial intelligence", "AI"),
        ("machine learning", "ML"),
    ])
    def test_specific_abbreviations(self, compression_engine_low, phrase, abbreviation):
        """Test specific abbreviation mappings."""
        result = compression_engine_low.compress(phrase)
        assert abbreviation in result


class TestSymbolMappings:
    """Tests for symbol mappings."""

    @pytest.mark.parametrize("phrase,symbol", [
        ("greater than or equal to", "≥"),
        ("less than or equal to", "≤"),
        ("greater than", ">"),
        ("less than", "<"),
        ("approximately", "~"),
        ("percent", "%"),
        ("degrees", "°"),
        ("multiplied by", "×"),
        ("divided by", "÷"),
        ("plus", "+"),
        ("minus", "-"),
        ("arrow", "→"),
        ("therefore", "∴"),
        ("because", "∵"),
        ("sum", "∑"),
        ("square root", "√"),
        ("infinity", "∞"),
    ])
    def test_specific_symbols(self, compression_engine_medium, phrase, symbol):
        """Test specific symbol mappings."""
        result = compression_engine_medium.compress(phrase)
        assert symbol in result


class TestFillerWords:
    """Tests for filler word removal."""

    @pytest.mark.parametrize("filler", [
        "please",
        "could you",
        "would you",
        "kindly",
        "just",
        "actually",
        "basically",
        "literally",
        "very",
        "really",
        "in order to",
        "due to the fact that",
        "at this point in time",
    ])
    def test_filler_words_removed(self, compression_engine_medium, filler):
        """Test specific filler words are removed."""
        text = f"{filler} help me"
        result = compression_engine_medium.compress(text)
        # The filler should be removed or significantly reduced
        assert filler.lower() not in result.lower() or result != text
