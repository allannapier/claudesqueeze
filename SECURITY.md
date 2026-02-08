# Security Policy for ClaudeSqueeze

## Overview

ClaudeSqueeze is an experimental LLM API proxy that intercepts and compresses prompts. This document outlines security considerations, known limitations, and best practices for safe deployment.

**⚠️ IMPORTANT: This is experimental software. Not for production use.**

## Security Considerations

### 1. SSRF (Server-Side Request Forgery) Protection

The proxy forwards requests to configured LLM API endpoints. To prevent SSRF attacks:

- **Host Allowlisting**: By default, only requests to known LLM API hosts are allowed:
  - `api.anthropic.com`
  - `api.openai.com`
  - `api.groq.com`
  - `api.cohere.com`
  - `api.mistral.ai`

- **Path Validation**: Paths are validated to prevent traversal attacks (`..`, `//`, etc.)

- **Loopback Protection**: Requests to `localhost`, `127.0.0.1`, `0.0.0.0`, or `::1` are blocked

- **Redirect Prevention**: Automatic redirects are disabled to prevent redirect-based SSRF

#### Customizing Allowed Hosts

Add to your `proxy_config.yaml`:

```yaml
security:
  allowed_hosts:
    - api.anthropic.com
    - api.openai.com
    - your-custom-api.example.com
```

### 2. Request Size Limits

To prevent memory exhaustion (DoS):

- **Maximum Request Body**: 50MB (`MAX_CONTENT_LENGTH`)
- **Maximum Header Length**: 8KB per header
- **Compression Input Limit**: 10MB (`MAX_INPUT_SIZE` in path_compression.py)

Requests exceeding these limits will receive a `413 Payload Too Large` response.

### 3. Header Injection Prevention

The proxy implements several protections against HTTP header injection:

- Headers containing newlines (`\n`, `\r`) are rejected
- Headers containing null bytes (`\x00`) are rejected
- Hop-by-hop headers are stripped before forwarding
- Header values are truncated to 8KB maximum

### 4. Configuration File Security

When loading configuration:

- Symlinked config files are rejected
- Config files outside allowed directories are rejected
- Allowed directories:
  - Current working directory
  - Script directory
  - `~/.llm-proxy/`

### 5. Security Headers

All responses include security headers:

```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Referrer-Policy: strict-origin-when-cross-origin
Content-Security-Policy: default-src 'none'
```

## Known Limitations and Risks

### 1. Local Network Exposure

By default, the proxy binds to `localhost` only. **Never** bind to `0.0.0.0` on untrusted networks without additional authentication.

### 2. API Key Exposure

The proxy requires your LLM API key to function. Ensure:
- API keys are stored in environment variables, not config files
- File permissions on config files are restrictive (e.g., `chmod 600`)
- API keys are rotated regularly

### 3. Prompt Content Visibility

The proxy processes and logs prompt content for compression statistics. In multi-user environments:
- Prompts may be visible in process memory
- Compression statistics may reveal usage patterns

### 4. Compression Side Effects

Aggressive compression may:
- Alter the semantic meaning of prompts
- Break code examples or file paths
- Cause unexpected LLM behavior

Always test thoroughly before use.

### 5. No Encryption in Transit (Local)

The proxy listens on HTTP (not HTTPS) on localhost. This is acceptable for local use but:
- Do not expose the proxy port over the network
- Use a reverse proxy with TLS if remote access is needed

### 6. Rate Limiting

The proxy does not implement rate limiting. For production use:
- Implement rate limiting at the reverse proxy level (nginx, etc.)
- Monitor for abuse patterns
- Set up alerting for unusual traffic

## Reporting Security Issues

If you discover a security vulnerability in ClaudeSqueeze:

1. **Do not** open a public issue
2. Email security concerns to the project maintainer
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

Response time: We aim to acknowledge within 48 hours and provide updates every 72 hours until resolved.

## Best Practices for Safe Deployment

### Development/Local Use

```bash
# Run on localhost only (default)
python claudesqueeze.py --port 8080

# Set restrictive permissions on config
chmod 600 proxy_config.yaml
```

### With Reverse Proxy (nginx)

```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;

        # Rate limiting
        limit_req zone=llm_proxy burst=10 nodelay;
    }
}
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

# Run as non-root user
RUN useradd -m -u 1000 proxyuser
USER proxyuser

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose only to localhost within container
EXPOSE 8080

CMD ["python", "claudesqueeze.py", "--port", "8080"]
```

### Environment Variables

Store sensitive configuration in environment variables:

```bash
export ANTHROPIC_API_KEY="your-key-here"
export ANTHROPIC_BASE_URL="http://localhost:8080"
```

## Security Checklist

Before deploying:

- [ ] Proxy binds to localhost only (not `0.0.0.0`)
- [ ] API keys stored in environment variables
- [ ] Config file has restrictive permissions (600)
- [ ] Allowed hosts list is configured appropriately
- [ ] Rate limiting is implemented (if needed)
- [ ] Logging does not capture sensitive data
- [ ] Regular dependency updates (`pip install --upgrade`)
- [ ] TLS/SSL in place for any network exposure

## Vulnerability History

| Date | Issue | Severity | Status |
|------|-------|----------|--------|
| 2026-02-08 | Initial security review | - | Completed |

## Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CWE/SANS Top 25](https://cwe.mitre.org/top25/)
- [Security Headers Best Practices](https://securityheaders.com/)

---

**Disclaimer**: This software is provided "as is" without warranty of any kind. Use at your own risk. The authors are not responsible for any damages or losses resulting from the use of this software.
