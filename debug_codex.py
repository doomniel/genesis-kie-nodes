"""Debug script for GPT Codex endpoint failures.

Bypasses KieClient + Node abstraction. Posts manually with 3 variants
to identify which body shape works.

Costs ~$0.06 USD if all 3 succeed; $0 if they all fail with 500.

Run from repo root: C:\\dev\\genesis-kie-nodes\\
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx


def load_dotenv():
    env_file = Path.cwd() / ".env"
    if not env_file.exists():
        print(f"WARN: .env not found at {env_file}")
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and v:
            os.environ.setdefault(k, v)


def post_raw(body: dict, label: str) -> bool:
    print()
    print("=" * 70)
    print(f"  VARIANT: {label}")
    print("=" * 70)
    print(f"Body sent:")
    print(json.dumps(body, indent=2, ensure_ascii=False))

    api_key = os.environ.get("KIE_API_KEY")
    if not api_key:
        print("FAIL: no KIE_API_KEY")
        return False

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                "https://api.kie.ai/api/v1/responses",
                headers=headers,
                json=body,
            )
    except Exception as e:
        print(f"FAIL (transport error): {type(e).__name__}: {e}")
        return False

    print()
    print(f"Response status: {response.status_code}")
    print(f"Response headers (relevant): "
          f"content-type={response.headers.get('content-type')}, "
          f"content-length={response.headers.get('content-length')}")
    print(f"Response body (raw, full):")
    text = response.text
    if len(text) > 2000:
        print(text[:1000])
        print(f"... [truncated, total {len(text)} chars] ...")
        print(text[-1000:])
    else:
        print(text)
    print()

    if response.status_code == 200:
        try:
            data = response.json()
            # Detect success
            if data.get("status") == "completed" or "output" in data:
                outputs = data.get("output") or []
                for block in outputs:
                    if block.get("type") == "message":
                        content = block.get("content") or []
                        for part in content:
                            if part.get("type") in ("output_text", "text"):
                                print(f"EXTRACTED TEXT: {part.get('text','')[:200]}")
                                return True
            print("PASS (200) but no extractable text — see body above")
            return True
        except Exception as e:
            print(f"PARTIAL (200 but bad json): {e}")
            return False
    else:
        print(f"FAIL ({response.status_code})")
        return False


def main():
    load_dotenv()
    print(f"KIE_API_KEY: {'set' if os.environ.get('KIE_API_KEY') else 'MISSING'}")

    # ─── Variant 1: exactly the docs.kie.ai cURL (with tools + high reasoning) ───
    variant1 = {
        "model": "gpt-5.1-codex",
        "stream": False,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Write a Python one-liner that prints 'hello world'."}
                ],
            }
        ],
        "tools": [{"type": "web_search"}],
        "reasoning": {"effort": "high"},
    }
    ok1 = post_raw(variant1, "Variant 1 — docs.kie.ai cURL verbatim (gpt-5.1-codex, tools=[web_search], reasoning=high)")

    # ─── Variant 2: my current node body, but with reasoning=high ───
    variant2 = {
        "model": "gpt-5.4-codex",
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Write a Python one-liner that prints 'hello world'."}
                ],
            }
        ],
        "stream": False,
        "reasoning": {"effort": "high"},
        "max_output_tokens": 128,
    }
    ok2 = post_raw(variant2, "Variant 2 — my node body (gpt-5.4-codex, reasoning=high, max_output_tokens=128, NO tools)")

    # ─── Variant 3: stripped down (no tools, no max_output_tokens, reasoning=medium) ───
    variant3 = {
        "model": "gpt-5.4-codex",
        "stream": False,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Write a Python one-liner that prints 'hello world'."}
                ],
            }
        ],
        "reasoning": {"effort": "medium"},
    }
    ok3 = post_raw(variant3, "Variant 3 — minimal body (gpt-5.4-codex, reasoning=medium, NO tools, NO max_output_tokens)")

    print()
    print("=" * 70)
    print("  DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print(f"  Variant 1 (docs cURL, 5.1, tools+high):    {'PASS' if ok1 else 'FAIL'}")
    print(f"  Variant 2 (my node, 5.4, high+max_tokens): {'PASS' if ok2 else 'FAIL'}")
    print(f"  Variant 3 (minimal, 5.4, medium, no extra): {'PASS' if ok3 else 'FAIL'}")
    print("=" * 70)
    print()
    print("Interpretation:")
    if ok1 and not ok2:
        print("  → Likely culprit: tools array required (or version 5.4 down)")
    elif ok3 and not ok2:
        print("  → Likely culprit: max_output_tokens crashes /api/v1/responses")
    elif ok1 and ok3 and not ok2:
        print("  → Confirmed: max_output_tokens causes the 500")
    elif not ok1 and not ok2 and not ok3:
        print("  → Codex API kie.ai is broken right now (server-side issue)")
        print("  → Recommendation: tag v0.6.0 anyway, document as known issue")
    elif ok2:
        print("  → My node body actually works now — transient 500 resolved")
    else:
        print("  → Mixed results; see Variants details above")


if __name__ == "__main__":
    main()
