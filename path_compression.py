#!/usr/bin/env python3
"""
Path and String Deduplication Compression Engine
================================================

Compresses repetitive file paths, URLs, and error strings in LLM prompts.
Uses placeholder substitution with session-scoped mapping tables.

This is EXPERIMENTAL - may cause issues with tool calls or path references.
Disable in config if problems occur.
"""

import re
import hashlib
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

# Security constants
MAX_INPUT_SIZE = 10 * 1024 * 1024  # 10MB max input
MAX_STRING_CACHE_SIZE = 10000  # Max unique strings to cache
MAX_PLACEHOLDERS = 5000  # Max placeholders to prevent memory exhaustion


@dataclass
class CompressionMapping:
    """Holds compression mappings for a session."""
    # Path/URL mappings: long string -> short placeholder
    path_map: Dict[str, str] = field(default_factory=dict)
    # Reverse mapping for decompression (if needed)
    reverse_map: Dict[str, str] = field(default_factory=dict)
    # String deduplication: hash -> (original, placeholder, count)
    string_map: Dict[str, Tuple[str, str, int]] = field(default_factory=dict)
    # Counter for generating unique placeholders
    _path_counter: int = field(default=0)
    _string_counter: int = field(default=0)

    def get_path_placeholder(self, path: str) -> str:
        """Get or create placeholder for a path."""
        if path not in self.path_map:
            self._path_counter += 1
            placeholder = f"<p{self._path_counter}>"
            self.path_map[path] = placeholder
            self.reverse_map[placeholder] = path
        return self.path_map[path]

    def get_string_placeholder(self, s: str) -> Tuple[str, bool]:
        """
        Get or create placeholder for a repeated string.
        Returns (placeholder, is_new) tuple.
        """
        # Use SHA-256 instead of MD5 for better security
        h = hashlib.sha256(s.encode()).hexdigest()[:16]

        if h in self.string_map:
            original, placeholder, count = self.string_map[h]
            self.string_map[h] = (original, placeholder, count + 1)
            return placeholder, False

        # Limit number of placeholders to prevent memory exhaustion
        if self._string_counter >= MAX_PLACEHOLDERS:
            # Return original string if limit reached
            return s, False

        self._string_counter += 1
        placeholder = f"<s{self._string_counter}>"
        self.string_map[h] = (s, placeholder, 1)
        self.reverse_map[placeholder] = s
        return placeholder, True


class PathCompressionEngine:
    """
    Compresses file paths, URLs, and repetitive strings in prompts.
    """

    # Regex patterns for different path types
    PATTERNS = {
        # File paths - Unix/Linux/Mac (min 20 chars to avoid short paths)
        'unix_path': re.compile(
            r'(?:/[\\\\\w\-\.]+){2,}(?:/[\w\-\.]+)*(?:\.[\w]+)?',
            re.MULTILINE
        ),
        # Windows paths
        'windows_path': re.compile(
            r'[a-zA-Z]:\\(?:[^\\:*?"<>|\r\n]+\\?)+',
            re.MULTILINE
        ),
        # URLs
        'url': re.compile(
            r'https?://(?:[\w\-\.]+\.)+[\w\-\.]+(?:/[^\s<>"{}|\\^`\[\]]*)?',
            re.MULTILINE
        ),
        # Stack trace patterns
        'python_traceback': re.compile(
            r'File "([^"]+)", line (\d+)',
            re.MULTILINE
        ),
        'js_traceback': re.compile(
            r'at (?:[^\s(]+ )?\(?(?:file://)?([^\s:)]+):(\d+):(\d+)\)?',
            re.MULTILINE
        ),
    }

    MIN_PATH_LENGTH = 20  # Don't compress short paths
    MIN_DEDUP_OCCURRENCES = 2
    MIN_STRING_LENGTH = 60

    def __init__(self, mapping: Optional[CompressionMapping] = None):
        self.mapping = mapping or CompressionMapping()
        self.stats = {
            'paths_compressed': 0,
            'strings_deduped': 0,
            'total_chars_saved': 0,
        }

    def compress(self, text: str) -> Tuple[str, Dict]:
        """
        Compress paths and deduplicate strings in text.

        Returns:
            Tuple of (compressed_text, metadata)
        """
        # Validate input size to prevent memory exhaustion
        if len(text) > MAX_INPUT_SIZE:
            # Return original if too large
            return text, {'paths': 0, 'strings': 0, 'error': 'Input too large'}

        original_len = len(text)
        metadata = {'paths': 0, 'strings': 0}

        # Phase 1: Extract and replace paths/URLs
        text = self._compress_paths(text)

        # Phase 2: Deduplicate repeated strings
        text = self._deduplicate_strings(text)

        # Calculate savings
        compressed_len = len(text)
        self.stats['total_chars_saved'] += (original_len - compressed_len)
        metadata['paths'] = len(self.mapping.path_map)
        metadata['strings'] = len([s for s in self.mapping.string_map.values() if s[2] >= self.MIN_DEDUP_OCCURRENCES])

        return text, metadata

    def _compress_paths(self, text: str) -> str:
        """Replace long paths with placeholders."""
        paths_found = []

        # Find all paths
        for path_type, pattern in self.PATTERNS.items():
            if path_type in ['python_traceback', 'js_traceback']:
                continue

            for match in pattern.finditer(text):
                path = match.group(0)
                if len(path) >= self.MIN_PATH_LENGTH:
                    paths_found.append((match.start(), match.end(), path))

        # Sort by position and deduplicate
        paths_found.sort(key=lambda x: x[0], reverse=True)
        unique_paths = {}
        for start, end, path in paths_found:
            if path not in unique_paths:
                placeholder = self.mapping.get_path_placeholder(path)
                unique_paths[path] = placeholder

        # Replace in text (longest first to avoid conflicts)
        for path, placeholder in sorted(unique_paths.items(),
                                         key=lambda x: len(x[0]),
                                         reverse=True):
            text = text.replace(path, placeholder)
            self.stats['paths_compressed'] += text.count(placeholder)

        # Handle stack traces specially
        text = self._compress_stack_traces(text)

        return text

    def _compress_stack_traces(self, text: str) -> str:
        """Special handling for stack traces - compress file paths."""
        def replace_python_trace(match):
            path = match.group(1)
            line_num = match.group(2)
            if len(path) >= self.MIN_PATH_LENGTH:
                placeholder = self.mapping.get_path_placeholder(path)
                return f'File "{placeholder}", L{line_num}'
            return match.group(0)

        text = self.PATTERNS['python_traceback'].sub(replace_python_trace, text)

        def replace_js_trace(match):
            full_match = match.group(0)
            path = match.group(1)
            line_num = match.group(2)
            col_num = match.group(3)
            if len(path) >= self.MIN_PATH_LENGTH:
                placeholder = self.mapping.get_path_placeholder(path)
                return full_match.replace(path, placeholder)
            return full_match

        text = self.PATTERNS['js_traceback'].sub(replace_js_trace, text)

        return text

    def _deduplicate_strings(self, text: str) -> str:
        """Find and deduplicate repeated long strings."""
        lines = text.split('\n')

        # Limit number of lines to prevent memory exhaustion
        if len(lines) > 100000:
            # Skip deduplication for very large inputs
            return text

        # Count occurrences of substantial lines
        line_counts = defaultdict(list)
        for i, line in enumerate(lines):
            stripped = line.strip()
            if len(stripped) >= self.MIN_STRING_LENGTH:
                # Limit line length to prevent hash collision attacks
                if len(stripped) > 10000:
                    stripped = stripped[:10000]
                # Normalize whitespace for matching
                normalized = ' '.join(stripped.split())
                line_counts[normalized].append(i)
                # Limit cache size
                if len(line_counts) > MAX_STRING_CACHE_SIZE:
                    # Stop collecting if cache is full
                    break

        # Find lines that appear multiple times
        duplicates = {
            line: indices
            for line, indices in line_counts.items()
            if len(indices) >= self.MIN_DEDUP_OCCURRENCES
        }

        # Replace duplicates with placeholders
        for line, indices in sorted(duplicates.items(),
                                     key=lambda x: len(x[0]),
                                     reverse=True):
            placeholder, is_new = self.mapping.get_string_placeholder(line)

            if not is_new or len(indices) >= self.MIN_DEDUP_OCCURRENCES:
                for idx in indices:
                    original = lines[idx]
                    indent = original[:len(original) - len(original.lstrip())]
                    lines[idx] = indent + placeholder
                self.stats['strings_deduped'] += len(indices)

        return '\n'.join(lines)

    def get_stats(self) -> Dict:
        """Get compression statistics."""
        return {
            **self.stats,
            'unique_paths': len(self.mapping.path_map),
            'unique_strings': len(self.mapping.string_map),
        }
