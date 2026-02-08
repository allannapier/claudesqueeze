# LLM Provider 429 Error Handling Research

## Executive Summary
Research findings on how different LLM providers handle rate limiting and 429 errors for token_saver project.

---

## 1. Anthropic (Claude API)

### Status Code & Signaling
- **HTTP Status**: 429
- **Error Response**: JSON with error description indicating which rate limit was exceeded

### Rate Limit Headers
```http
retry-after: <seconds>
anthropic-ratelimit-requests-limit: <number>
anthropic-ratelimit-requests-remaining: <number>
anthropic-ratelimit-requests-reset: <RFC3339_timestamp>
anthropic-ratelimit-tokens-limit: <number>
anthropic-ratelimit-tokens-remaining: <number>
anthropic-ratelimit-tokens-reset: <RFC3339_timestamp>
anthropic-ratelimit-input-tokens-limit: <number>
anthropic-ratelimit-input-tokens-remaining: <number>
anthropic-ratelimit-input-tokens-reset: <RFC3339_timestamp>
anthropic-ratelimit-output-tokens-limit: <number>
anthropic-ratelimit-output-tokens-remaining: <number>
anthropic-ratelimit-output-tokens-reset: <RFC3339_timestamp>
```

### Reset Info
- **Retry-After Header**: YES - Number of seconds to wait
- **Reset Timestamp**: YES - RFC 3339 format in headers
- Uses **token bucket algorithm** (continuous replenishment, not fixed interval resets)

### Rate Limit Types
- RPM (Requests Per Minute)
- ITPM (Input Tokens Per Minute)
- OTPM (Output Tokens Per Minute)

### Example 429 Response
```json
{
  "error": {
    "type": "rate_limit_error",
    "message": "Rate limit exceeded: input tokens per minute"
  }
}
```

---

## 2. OpenAI API

### Status Code & Signaling
- **HTTP Status**: 429
- **Error Types**: "rate_limit_reached", "insufficient_quota"

### Rate Limit Headers
```http
x-ratelimit-limit-requests: <number>
x-ratelimit-remaining-requests: <number>
x-ratelimit-reset-requests: <duration>
x-ratelimit-limit-tokens: <number>
x-ratelimit-remaining-tokens: <number>
x-ratelimit-reset-tokens: <duration>
```

### Reset Info
- **Reset Headers**: YES - Duration format (e.g., "1s", "6m0s")
- Clear indication of time until limit resets

### Rate Limit Types
- RPM (Requests Per Minute)
- RPD (Requests Per Day)
- TPM (Tokens Per Minute)
- TPD (Tokens Per Day)
- IPM (Images Per Minute)

### Example 429 Response
```json
{
  "error": {
    "message": "Rate limit reached for requests",
    "type": "rate_limit_error",
    "code": "rate_limit_reached"
  }
}
```

---

## 3. Moonshot Kimi API

### Status Code & Signaling
- **HTTP Status**: 429
- **Error Response**: JSON with type field

### Rate Limit Headers
- **No standardized headers documented**
- Information varies by endpoint

### Reset Info
- **Retry-After Header**: NO standard header
- Reset info included in error message body

### Example 429 Response
```json
{
  "error": {
    "message": "Organization Rate limit exceeded, please try again after 1 seconds",
    "type": "rate_limit_reached_error"
  }
}
```

**Alternative (Chinese)**:
```
MCP error 429: 您的账户已达到速率限制，请您控制请求频率
```

---

## 4. MiniMax API

### Status Code & Signaling
- **HTTP Status**: 429
- **Error Response**: Simple string message format

### Rate Limit Headers
- **No standardized headers documented**

### Reset Info
- **Retry-After Header**: NO
- Reset info NOT provided in response

### Example 429 Response
```
MCP error 429: 您的账户已达到速率限制，请您控制请求频率
```

**Translation**: "Your account has reached the rate limit, please control your request frequency"

---

## 5. Portkey (AI Gateway)

### Status Code & Signaling
- **HTTP Status**: 429 (from upstream providers)
- Portkey acts as a gateway that can handle retries automatically

### Retry Behavior
- **Default retry codes**: [429, 500, 502, 503, 504]
- **Max attempts**: 5 (configurable)
- **Backoff strategy**: Exponential (1s, 2s, 4s, 8s, 16s)

### Configuration
```json
{
  "retry": {
    "attempts": 3,
    "on_status_codes": [408, 429, 401]
  }
}
```

### Enterprise Headers
- Enterprise customers can use specific headers for retry wait times instead of default exponential backoff

---

## 6. Google Gemini API

### Status Code & Signaling
- **HTTP Status**: 429
- **Error Type**: "RESOURCE_EXHAUSTED"

### Rate Limit Headers
- **Limited public documentation** on specific headers
- Rate limits viewable in Google AI Studio

### Reset Info
- **Retry-After Header**: Possible but not well documented
- Quotas reset at midnight Pacific Time for RPD

### Rate Limit Types
- RPM (Requests Per Minute)
- TPM (Tokens Per Minute)
- RPD (Requests Per Day)

### Example 429 Response
```json
{
  "error": {
    "code": 429,
    "message": "Resource has been exhausted (e.g. check quota)",
    "status": "RESOURCE_EXHAUSTED"
  }
}
```

---

## Key Differences Summary

| Provider | Retry-After Header | Reset Timestamp | Detailed Headers | Error Format |
|----------|-------------------|-----------------|------------------|--------------|
| Anthropic | YES (seconds) | YES (RFC3339) | Extensive | JSON |
| OpenAI | NO | YES (duration) | Detailed | JSON |
| Kimi | NO | NO (body only) | Minimal | JSON |
| MiniMax | NO | NO | None | String |
| Portkey | Gateway handled | Gateway handled | Pass-through | Configurable |
| Gemini | Unclear | Unclear | Limited | JSON |

## Recommendations for Token Saver

1. **Parse provider-specific headers** when available (Anthropic, OpenAI)
2. **Implement exponential backoff** as universal fallback
3. **Extract retry info from error body** when headers unavailable (Kimi, MiniMax)
4. **Consider Portkey integration** for automatic retry handling
5. **Track rate limits proactively** using remaining headers to avoid 429s

## Sources
- [Anthropic Rate Limits Documentation](https://docs.anthropic.com/en/api/rate-limits)
- [OpenAI Rate Limits Documentation](https://platform.openai.com/docs/guides/rate-limits)
- [Gemini API Rate Limits Documentation](https://ai.google.dev/gemini-api/docs/rate-limits)
- [Portkey Automatic Retries Documentation](https://docs.portkey.ai/docs/product/ai-gateway/automatic-retries)
- [Kimi API FAQ](https://platform.moonshot.cn/docs/guide/faq)
