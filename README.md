# ClaudeSqueeze

**⚠️ EXPERIMENTAL - NOT FOR PRODUCTION USE ⚠️**

A real-time LLM API proxy that compresses prompts to reduce token usage and costs. Sits between Claude Code (or similar tools) and your LLM provider, transparently applying compression techniques to minimize token count while preserving semantic meaning.

**Sister project to [ClaudeSavvy](https://github.com/yourusername/claudesavvy)** - while ClaudeSavvy reports on token usage, ClaudeSqueeze actively reduces it.

---

## ⚠️ Important Warnings

**THIS IS EXPERIMENTAL SOFTWARE**

- **Not recommended for enterprise/production environments**
- **May cause unexpected behavior with tool calls, file paths, or code execution**
- **Compression may alter prompt semantics in subtle ways**
- **Test thoroughly in non-critical workflows before relying on it**
- **No warranty or guarantee of correctness**

Use at your own risk. The authors are not responsible for any issues arising from use of this software.

---

## What It Does

ClaudeSqueeze intercepts API calls to LLM providers (Anthropic, OpenAI, Kimi, etc.), applies compression techniques to reduce token count, then forwards the compressed request. Responses are returned unchanged.

### Compression Techniques

| Technique | Description | Savings |
|-----------|-------------|---------|
| **Filler Word Removal** | Removes "please", "could you", "basically", etc. | ~2-3% |
| **Abbreviation** | "for example" → "e.g.", "as soon as possible" → "ASAP" | ~2-4% |
| **Symbol Substitution** | "greater than" → ">", "arrow" → "→" | ~1-2% |
| **Path Compression** | `/home/user/project/src/file.js` → `<p1>` | ~5-15% |
| **Whitespace Normalization** | Collapses multiple newlines/spaces | ~2-5% |
| **Aggressive Grammar** | Removes articles, auxiliary verbs | ~3-5% |

**Combined savings: 10-25% typical, up to 45% on code-heavy prompts**

---

## Quick Start

### Prerequisites

- Python 3.9+
- `requests` library
- `pyyaml` library

```bash
pip install requests pyyaml
```

### Installation

```bash
git clone https://github.com/yourusername/claudesqueeze.git
cd claudesqueeze
```

### Configuration

Copy the example config and edit:

```bash
cp config.yaml.example config.yaml
```

Edit `config.yaml`:

```yaml
target_api:
  url: "https://api.anthropic.com"  # Your LLM provider

compression:
  level: "high"                     # low, medium, high
  passthrough_mode: false           # Set true to disable compression
  enable_path_compression: false    # EXPERIMENTAL - may break tool calls

server:
  port: 8080
  host: "127.0.0.1"
```

### Running

```bash
python llm_proxy.py --config config.yaml
```

You should see:

```
╔══════════════════════════════════════════════════════════════════╗
║  LLM Compression Proxy                                           ║
╠══════════════════════════════════════════════════════════════════╣
║  Listening on:    http://localhost:8080                          ║
║  Mode:            COMPRESSION (high)                             ║
║  Target API:      https://api.anthropic.com                      ║
║  Path Compress:   disabled                                       ║
╚══════════════════════════════════════════════════════════════════╝
```

### Configure Claude Code

Claude Code uses a settings.json file for configuration. Update your user-level settings:

**Location:** `~/.claude/settings.json`

Add or update the following:

```json
{
  "anthropicBaseUrl": "http://localhost:8080",
  "anthropicApiKey": "your-api-key"
}
```

**For Kimi or other providers:**

```json
{
  "anthropicBaseUrl": "http://localhost:8080",
  "anthropicApiKey": "your-kimi-api-key"
}
```

**Note:** Claude Code uses the Anthropic API format internally. The proxy handles translation to other providers.

After updating settings, restart Claude Code for changes to take effect.

---

## Viewing Metrics

While the proxy is running, check compression stats:

```bash
curl http://localhost:8080/metrics
```

Example output:

```json
{
  "total_requests": 72,
  "total_original_tokens": 230906,
  "total_compressed_tokens": 189782,
  "total_tokens_saved": 41124,
  "average_reduction_pct": 17.8,
  "compression_stats": {
    "total_compressions": 72,
    "avg_compression_time_ms": 48.389,
    "min_compression_time_ms": 0.679,
    "max_compression_time_ms": 742.003
  }
}
```

---

## Configuration Options

### Compression Levels

- **`low`**: Minimal compression - abbreviations only, safe
- **`medium`**: Standard - abbreviations + filler words + symbols
- **`high`**: Aggressive - all techniques including grammar reduction (default)

### Path Compression (Experimental)

When `enable_path_compression: true`:

- File paths are replaced with placeholders (`<p1>`, `<p2>`, etc.)
- Stack traces are specially handled
- Repeated strings are deduplicated

**⚠️ May cause issues with:**
- Tool calls that reference file paths
- Code execution requiring actual paths
- Debugging workflows

Enable only if you primarily send logs/code for analysis, not for tool execution.

### Provider Prompt Caching

The proxy can inject Anthropic's `cache_control` headers:

```yaml
cache_control:
  enabled: true
  cache_roles:
    - "system"
  cache_first_n_messages: 2
```

This enables provider-level caching (90% cost reduction on cached tokens), separate from ClaudeSqueeze's compression.

---

## Architecture

```
┌─────────────┐     ┌─────────────────┐     ┌──────────────┐
│ Claude Code │────▶│  ClaudeSqueeze  │────▶│  LLM API     │
│  (Client)   │     │   (Proxy)       │     │ (Kimi/etc)   │
└─────────────┘     └─────────────────┘     └──────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  Compression │
                    │  Engine      │
                    └──────────────┘
```

**Request Flow:**
1. Client sends request to proxy (localhost:8080)
2. Proxy compresses user messages only (preserves assistant history)
3. Forward compressed request to LLM API
4. Stream response back to client unchanged

---

## Known Limitations

1. **Streaming Only**: Optimized for SSE streaming responses
2. **Text Content**: Best for natural language and code; may not help with binary/base64
3. **Single User**: Per-session path mappings (not shared across instances)
4. **No Decompression**: Responses remain compressed if they contain placeholders

---

## Troubleshooting

### Proxy not responding

```bash
# Check if proxy is running
curl http://localhost:8080/health

# Check for errors in proxy console
```

### Claude Code not using proxy

Verify environment variable:
```bash
echo $ANTHROPIC_BASE_URL  # Should show http://localhost:8080
```

### Compression causing issues

Disable in config:
```yaml
compression:
  passthrough_mode: true
```

Or disable path compression specifically:
```yaml
compression:
  enable_path_compression: false
```

---

## Development

### Project Structure

```
claudesqueeze/
├── llm_proxy.py          # Main proxy server
├── path_compression.py   # Path/URL compression engine
├── config.yaml.example   # Example configuration
└── README.md            # This file
```

### Running Tests

```bash
# TODO: Add tests
python -m pytest tests/
```

---

## Contributing

This is an experimental research project. Contributions welcome but:

- Keep changes focused and well-tested
- Document any new compression techniques
- Maintain backwards compatibility
- Add warnings for experimental features

---

## License

MIT License - See LICENSE file

---

## Related Projects

- **[ClaudeSavvy](https://github.com/yourusername/claudesavvy)** - Token usage reporting for Claude Code
- **[LLM Communication Research](../README.md)** - Research on token efficiency strategies

---

**⚠️ REMINDER: This is experimental software. Not for production use. ⚠️**
