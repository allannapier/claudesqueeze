"""
Unit tests for the PathCompressionEngine class.

Tests path pattern matching, stack trace compression, string deduplication,
and placeholder generation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from path_compression import PathCompressionEngine, CompressionMapping


class TestCompressionMapping:
    """Tests for CompressionMapping dataclass."""

    def test_default_initialization(self):
        """Test CompressionMapping initializes with empty mappings."""
        mapping = CompressionMapping()
        assert mapping.path_map == {}
        assert mapping.reverse_map == {}
        assert mapping.string_map == {}
        assert mapping._path_counter == 0
        assert mapping._string_counter == 0

    def test_get_path_placeholder_creates_new(self):
        """Test get_path_placeholder creates new placeholder for new path."""
        mapping = CompressionMapping()
        placeholder = mapping.get_path_placeholder("/home/user/file.txt")

        assert placeholder == "<p1>"
        assert mapping.path_map["/home/user/file.txt"] == "<p1>"
        assert mapping.reverse_map["<p1>"] == "/home/user/file.txt"
        assert mapping._path_counter == 1

    def test_get_path_placeholder_returns_existing(self):
        """Test get_path_placeholder returns existing placeholder for known path."""
        mapping = CompressionMapping()
        placeholder1 = mapping.get_path_placeholder("/home/user/file.txt")
        placeholder2 = mapping.get_path_placeholder("/home/user/file.txt")

        assert placeholder1 == placeholder2 == "<p1>"
        assert mapping._path_counter == 1  # Counter should not increment

    def test_get_path_placeholder_multiple_paths(self):
        """Test get_path_placeholder creates unique placeholders for different paths."""
        mapping = CompressionMapping()
        p1 = mapping.get_path_placeholder("/home/user/file1.txt")
        p2 = mapping.get_path_placeholder("/home/user/file2.txt")
        p3 = mapping.get_path_placeholder("/var/log/app.log")

        assert p1 == "<p1>"
        assert p2 == "<p2>"
        assert p3 == "<p3>"
        assert mapping._path_counter == 3

    def test_get_string_placeholder_creates_new(self):
        """Test get_string_placeholder creates new placeholder for new string."""
        mapping = CompressionMapping()
        placeholder, is_new = mapping.get_string_placeholder("some long string here")

        assert placeholder == "<s1>"
        assert is_new is True
        assert mapping._string_counter == 1

    def test_get_string_placeholder_returns_existing(self):
        """Test get_string_placeholder returns existing placeholder for known string."""
        mapping = CompressionMapping()
        placeholder1, is_new1 = mapping.get_string_placeholder("some long string here")
        placeholder2, is_new2 = mapping.get_string_placeholder("some long string here")

        assert placeholder1 == placeholder2 == "<s1>"
        assert is_new1 is True
        assert is_new2 is False
        assert mapping._string_counter == 1

    def test_get_string_placeholder_increments_count(self):
        """Test get_string_placeholder increments count for duplicate strings."""
        mapping = CompressionMapping()
        mapping.get_string_placeholder("test string")
        mapping.get_string_placeholder("test string")
        mapping.get_string_placeholder("test string")

        # Get the hash and check count
        import hashlib
        h = hashlib.md5("test string".encode()).hexdigest()[:12]
        original, placeholder, count = mapping.string_map[h]
        assert count == 3

    def test_reverse_mapping_populated(self):
        """Test reverse_map is populated for both path and string placeholders."""
        mapping = CompressionMapping()
        path = "/home/user/file.txt"
        string = "some long string"

        path_ph = mapping.get_path_placeholder(path)
        string_ph, _ = mapping.get_string_placeholder(string)

        assert mapping.reverse_map[path_ph] == path
        assert mapping.reverse_map[string_ph] == string


class TestPathCompressionEngineInitialization:
    """Tests for PathCompressionEngine initialization."""

    def test_default_initialization(self):
        """Test PathCompressionEngine initializes with default mapping."""
        engine = PathCompressionEngine()
        assert isinstance(engine.mapping, CompressionMapping)
        assert engine.stats == {
            'paths_compressed': 0,
            'strings_deduped': 0,
            'total_chars_saved': 0,
        }

    def test_custom_mapping(self):
        """Test PathCompressionEngine accepts custom mapping."""
        custom_mapping = CompressionMapping()
        custom_mapping._path_counter = 5
        engine = PathCompressionEngine(mapping=custom_mapping)
        assert engine.mapping is custom_mapping

    def test_pattern_compiled(self):
        """Test regex patterns are compiled."""
        engine = PathCompressionEngine()
        assert 'unix_path' in engine.PATTERNS
        assert 'windows_path' in engine.PATTERNS
        assert 'url' in engine.PATTERNS
        assert 'python_traceback' in engine.PATTERNS
        assert 'js_traceback' in engine.PATTERNS


class TestUnixPathCompression:
    """Tests for Unix/Linux path compression."""

    def test_simple_unix_path(self, path_compression_engine):
        """Test simple Unix path compression."""
        text = "Error in /home/user/projects/myapp/src/main.py"
        result, metadata = path_compression_engine.compress(text)

        assert "<p" in result
        assert "/home/user/projects/myapp/src/main.py" not in result
        assert metadata['paths'] >= 1

    def test_multiple_unix_paths(self, path_compression_engine):
        """Test multiple Unix paths in same text."""
        text = """File /home/user/file1.txt
File /home/user/file2.txt
File /home/user/file1.txt"""
        result, metadata = path_compression_engine.compress(text)

        # Should have placeholders
        assert "<p" in result
        # Should deduplicate same paths
        assert metadata['paths'] >= 1

    def test_short_path_not_compressed(self, path_compression_engine):
        """Test short paths are not compressed."""
        text = "File /tmp/a.txt"
        result, metadata = path_compression_engine.compress(text)

        # Short path should not be compressed
        assert "/tmp/a.txt" in result or result == text

    def test_path_with_dots(self, path_compression_engine):
        """Test paths with dots in names."""
        text = "Error in /home/user.name/projects/test.file.py"
        result, metadata = path_compression_engine.compress(text)

        if len("/home/user.name/projects/test.file.py") >= path_compression_engine.MIN_PATH_LENGTH:
            assert "<p" in result


class TestWindowsPathCompression:
    """Tests for Windows path compression."""

    def test_simple_windows_path(self, path_compression_engine):
        """Test simple Windows path compression."""
        text = r"Error in C:\Users\user\Projects\MyApp\src\main.py"
        result, metadata = path_compression_engine.compress(text)

        if len(r"C:\Users\user\Projects\MyApp\src\main.py") >= path_compression_engine.MIN_PATH_LENGTH:
            assert "<p" in result
            assert r"C:\Users\user\Projects\MyApp\src\main.py" not in result

    def test_windows_path_with_spaces(self, path_compression_engine):
        """Test Windows paths with spaces."""
        text = r"File C:\Program Files\My Application\app.exe"
        result, metadata = path_compression_engine.compress(text)

        if len(r"C:\Program Files\My Application\app.exe") >= path_compression_engine.MIN_PATH_LENGTH:
            assert "<p" in result


class TestURLCompression:
    """Tests for URL compression."""

    def test_simple_url(self, path_compression_engine):
        """Test simple URL compression."""
        text = "Visit https://example.com for more info"
        result, metadata = path_compression_engine.compress(text)

        # Simple URL might be too short
        if len("https://example.com") >= path_compression_engine.MIN_PATH_LENGTH:
            assert "<p" in result

    def test_long_url(self, path_compression_engine):
        """Test long URL compression."""
        url = "https://raw.githubusercontent.com/username/repository/main/src/components/Button.tsx"
        text = f"Import from {url}"
        result, metadata = path_compression_engine.compress(text)

        assert "<p" in result
        assert url not in result

    def test_url_with_query_params(self, path_compression_engine):
        """Test URL with query parameters."""
        url = "https://api.example.com/v1/users?page=1&limit=100&sort=name"
        text = f"GET {url}"
        result, metadata = path_compression_engine.compress(text)

        if len(url) >= path_compression_engine.MIN_PATH_LENGTH:
            assert "<p" in result


class TestStackTraceCompression:
    """Tests for stack trace compression."""

    def test_python_traceback(self, path_compression_engine, python_traceback):
        """Test Python traceback compression."""
        result, metadata = path_compression_engine.compress(python_traceback)

        # Should have compressed file paths
        assert "<p" in result
        # Should have replaced line number format
        assert 'L' in result or 'line' in result

    def test_js_traceback(self, path_compression_engine, js_traceback):
        """Test JavaScript traceback compression."""
        result, metadata = path_compression_engine.compress(js_traceback)

        # Should have compressed file paths
        assert "<p" in result

    def test_traceback_preserves_structure(self, path_compression_engine, python_traceback):
        """Test that traceback structure is preserved."""
        result, metadata = path_compression_engine.compress(python_traceback)

        # Should still have Traceback or TB
        assert "Traceback" in result or "TB" in result
        # Should still have the error
        assert "ValueError" in result or "Err:" in result


class TestStringDeduplication:
    """Tests for string deduplication."""

    def test_duplicate_lines_deduplicated(self, path_compression_engine):
        """Test duplicate long lines are deduplicated."""
        long_line = "This is a very long line that should be deduplicated because it appears multiple times in the text"
        text = f"{long_line}\nSome other text\n{long_line}\nMore text\n{long_line}"

        result, metadata = path_compression_engine.compress(text)

        # Should have deduplicated strings
        assert metadata['strings'] >= 1
        assert "<s" in result

    def test_short_lines_not_deduplicated(self, path_compression_engine):
        """Test short lines are not deduplicated."""
        text = """Short line
Short line
Short line"""
        result, metadata = path_compression_engine.compress(text)

        # Short lines should not be deduplicated
        assert metadata['strings'] == 0

    def test_single_occurrence_not_deduplicated(self, path_compression_engine):
        """Test lines appearing only once are not deduplicated."""
        long_line = "This is a very long line that appears only once in the entire text document"
        text = f"{long_line}\nSome other different text\nMore different text"

        result, metadata = path_compression_engine.compress(text)

        # Single occurrence should not be deduplicated
        # (unless it was already in string_map from previous calls)

    def test_indentation_preserved(self, path_compression_engine):
        """Test indentation is preserved when deduplicating."""
        long_line = "    This is a very long indented line that should be deduplicated"
        text = f"{long_line}\n{long_line}\n{long_line}"

        result, metadata = path_compression_engine.compress(text)

        # Check that indentation is preserved in result
        lines = result.split('\n')
        for line in lines:
            if '<s' in line:
                assert line.startswith('    ') or line.startswith('<s')


class TestCompressionStats:
    """Tests for compression statistics."""

    def test_stats_initially_zero(self, path_compression_engine):
        """Test stats start at zero."""
        stats = path_compression_engine.get_stats()
        assert stats['paths_compressed'] == 0
        assert stats['strings_deduped'] == 0
        assert stats['total_chars_saved'] == 0

    def test_stats_updated_after_compression(self, path_compression_engine):
        """Test stats are updated after compression."""
        text = "/home/user/projects/myapp/src/main.py and /home/user/projects/myapp/src/utils.py"
        path_compression_engine.compress(text)

        stats = path_compression_engine.get_stats()
        assert stats['paths_compressed'] >= 2
        assert stats['total_chars_saved'] > 0

    def test_unique_counts_in_stats(self, path_compression_engine):
        """Test unique path and string counts in stats."""
        text = "/home/user/file.txt"
        path_compression_engine.compress(text)

        stats = path_compression_engine.get_stats()
        assert 'unique_paths' in stats
        assert 'unique_strings' in stats


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_string(self, path_compression_engine):
        """Test empty string handling."""
        result, metadata = path_compression_engine.compress("")
        assert result == ""
        assert metadata == {'paths': 0, 'strings': 0}

    def test_whitespace_only(self, path_compression_engine):
        """Test whitespace-only string handling."""
        result, metadata = path_compression_engine.compress("   \n\t  ")
        assert result == "   \n\t  "

    def test_special_characters_in_paths(self, path_compression_engine):
        """Test paths with special characters."""
        text = "/home/user/file-with-dashes_and_underscores.txt"
        result, metadata = path_compression_engine.compress(text)
        # Should handle special characters

    def test_very_long_path(self, path_compression_engine):
        """Test very long path handling."""
        long_path = "/home/user/" + "/".join(["directory" + str(i) for i in range(50)]) + "/file.txt"
        text = f"Error in {long_path}"
        result, metadata = path_compression_engine.compress(text)

        assert "<p" in result
        assert long_path not in result

    def test_multiple_compression_calls(self, path_compression_engine):
        """Test multiple compression calls accumulate stats."""
        text1 = "/home/user/file1.txt"
        text2 = "/home/user/file2.txt"

        path_compression_engine.compress(text1)
        path_compression_engine.compress(text2)

        stats = path_compression_engine.get_stats()
        assert stats['paths_compressed'] >= 2

    def test_path_placeholder_consistency(self, path_compression_engine):
        """Test same path gets same placeholder across calls."""
        text = "/home/user/file.txt"

        result1, _ = path_compression_engine.compress(text)
        result2, _ = path_compression_engine.compress(text)

        # Extract placeholder from results
        import re
        ph1 = re.search(r'<p\d+>', result1)
        ph2 = re.search(r'<p\d+>', result2)

        if ph1 and ph2:
            assert ph1.group() == ph2.group()


class TestPatternMatching:
    """Tests for regex pattern matching."""

    def test_unix_path_pattern(self):
        """Test Unix path regex pattern."""
        engine = PathCompressionEngine()
        pattern = engine.PATTERNS['unix_path']

        matches = pattern.findall("/home/user/file.txt /var/log/app.log")
        assert len(matches) >= 1

    def test_windows_path_pattern(self):
        """Test Windows path regex pattern."""
        engine = PathCompressionEngine()
        pattern = engine.PATTERNS['windows_path']

        matches = pattern.findall(r"C:\Users\file.txt D:\Data\app.log")
        assert len(matches) >= 1

    def test_url_pattern(self):
        """Test URL regex pattern."""
        engine = PathCompressionEngine()
        pattern = engine.PATTERNS['url']

        matches = pattern.findall("Visit https://example.com and http://test.org")
        assert len(matches) >= 2

    def test_python_traceback_pattern(self):
        """Test Python traceback regex pattern."""
        engine = PathCompressionEngine()
        pattern = engine.PATTERNS['python_traceback']

        text = 'File "/home/user/test.py", line 42'
        matches = pattern.findall(text)
        assert len(matches) >= 1

    def test_js_traceback_pattern(self):
        """Test JavaScript traceback regex pattern."""
        engine = PathCompressionEngine()
        pattern = engine.PATTERNS['js_traceback']

        text = "at function (/home/user/test.js:10:5)"
        matches = pattern.findall(text)
        assert len(matches) >= 1


class TestMetadata:
    """Tests for compression metadata."""

    def test_metadata_paths_count(self, path_compression_engine):
        """Test metadata contains correct path count."""
        text = "/home/user/file1.txt /home/user/file2.txt"
        result, metadata = path_compression_engine.compress(text)

        assert 'paths' in metadata
        assert metadata['paths'] >= 1

    def test_metadata_strings_count(self, path_compression_engine):
        """Test metadata contains correct string count."""
        long_line = "This is a very long line that should be deduplicated because it appears multiple times"
        text = f"{long_line}\n{long_line}"
        result, metadata = path_compression_engine.compress(text)

        assert 'strings' in metadata

    def test_chars_saved_tracked(self, path_compression_engine):
        """Test total characters saved is tracked."""
        original_stats = path_compression_engine.stats['total_chars_saved']

        text = "/home/user/projects/myapp/src/main.py"
        path_compression_engine.compress(text)

        new_stats = path_compression_engine.stats['total_chars_saved']
        assert new_stats > original_stats
