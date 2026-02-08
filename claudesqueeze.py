#!/usr/bin/env python3
"""
ClaudeSqueeze - LLM API Proxy with Prompt Compression
======================================================

Real-time token compression proxy for Claude Code and other LLM tools.
Reduces API costs by 10-25% through intelligent prompt compression.

⚠️  EXPERIMENTAL - NOT FOR PRODUCTION USE ⚠️

Usage:
    python claudesqueeze.py --config config.yaml

Then configure Claude Code:
    export ANTHROPIC_BASE_URL=http://localhost:8080
    claude
"""

import argparse
import sys
from llm_proxy import run_proxy


def main():
    parser = argparse.ArgumentParser(
        description="ClaudeSqueeze - LLM API Proxy with Prompt Compression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default config
    python claudesqueeze.py

    # Run on different port
    python claudesqueeze.py --port 9000

    # Use medium compression (less aggressive)
    python claudesqueeze.py --level medium

    # Use custom config file
    python claudesqueeze.py --config myconfig.yaml

    # View metrics while running
    curl http://localhost:8080/metrics

⚠️  IMPORTANT: This is experimental software. Not for production use.
        """,
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to listen on (default: 8080)",
    )
    parser.add_argument(
        "--level",
        choices=["low", "medium", "high"],
        default="high",
        help="Compression level (default: high)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════════╗
║  ClaudeSqueeze - LLM Token Compression Proxy                    ║
║  ⚠️  EXPERIMENTAL - NOT FOR PRODUCTION USE ⚠️                    ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    try:
        run_proxy(port=args.port, compression_level=args.level, config_path=args.config)
    except KeyboardInterrupt:
        print("\n\n[ClaudeSqueeze] Shutting down...")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
