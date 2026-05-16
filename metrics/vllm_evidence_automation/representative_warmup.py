#!/usr/bin/env python3
import json
import os
import sys
import urllib.request


def main() -> int:
    host = os.environ.get("HOST", "127.0.0.1")
    port = os.environ.get("PORT", "8000")
    model = os.environ["MODEL"]
    prompt_words = int(os.environ.get("REP_WARMUP_PROMPT_WORDS", "1000"))
    max_tokens = int(os.environ.get("REP_WARMUP_MAX_TOKENS", "256"))

    payload = {
        "model": model,
        "prompt": "hello " * prompt_words,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"http://{host}:{port}/v1/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        body = resp.read(300).decode("utf-8", errors="replace")
        print(resp.status)
        print(body)
    return 0


if __name__ == "__main__":
    sys.exit(main())

