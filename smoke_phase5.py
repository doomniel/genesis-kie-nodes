"""Smoke test for Genesis-Kie Phase 5 LLM nodes.

Runs nodes DIRECTLY (no ComfyUI workflow) via .run() method.

Usage:
    python smoke_phase5.py gpt           # GPT 5.2 (cheapest)        ~$0.005
    python smoke_phase5.py claude_haiku  # Claude Haiku 4.5          ~$0.005
    python smoke_phase5.py gemini_flash  # Gemini 3.5 Flash          ~$0.003
    python smoke_phase5.py gpt55         # GPT 5.5 (reasoning)       ~$0.04
    python smoke_phase5.py claude_opus48 # Claude Opus 4.8 frontier  ~$0.05
    python smoke_phase5.py gemini_pro    # Gemini 3.1 Pro frontier   ~$0.02
    python smoke_phase5.py codex         # GPT Codex                 ~$0.02
    python smoke_phase5.py all_cheap     # 3 cheapest nodes          ~$0.015

Run from repo root: C:\\dev\\genesis-kie-nodes\\
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path


def load_package():
    repo_root = Path.cwd()
    if not (repo_root / "__init__.py").exists():
        raise SystemExit(f"FATAL: not in repo root. cwd={repo_root}")

    sys.path.insert(0, str(repo_root.parent))
    pkg_name = repo_root.name.replace("-", "_")
    spec = importlib.util.spec_from_file_location(
        pkg_name,
        str(repo_root / "__init__.py"),
        submodule_search_locations=[str(repo_root)],
    )
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = pkg
    spec.loader.exec_module(pkg)
    return pkg


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
    print(f"OK: loaded .env (KIE_API_KEY={'set' if os.environ.get('KIE_API_KEY') else 'MISSING'})")


def smoke_one(pkg, node_key, kwargs, label, est_cost):
    print()
    print("=" * 60)
    print(f"  SMOKE: {label}")
    print(f"  Node:  {node_key}")
    print(f"  Cost:  ~{est_cost} USD")
    print("=" * 60)

    cls = pkg.NODE_CLASS_MAPPINGS.get(node_key)
    if cls is None:
        print(f"FAIL: node {node_key} not registered.")
        return False

    print(f"OK: class loaded: {cls.__name__}")
    print(f"   CATEGORY: {cls.CATEGORY}")
    print(f"   ENDPOINT: {cls.ENDPOINT}")
    print(f"   MODEL_ID: {cls.MODEL_ID}")
    print(f"   RETURN_TYPES: {cls.RETURN_TYPES}")

    instance = cls()
    print(f"   inputs:")
    for k, v in kwargs.items():
        v_str = str(v)
        if len(v_str) > 80: v_str = v_str[:77] + "..."
        print(f"     {k} = {v_str}")

    print(f"   calling .run() ...")
    t0 = time.monotonic()
    try:
        result = instance.run(**kwargs)
    except Exception as e:
        elapsed = time.monotonic() - t0
        print(f"FAIL after {elapsed:.1f}s: {type(e).__name__}: {e}")
        return False

    elapsed = time.monotonic() - t0
    print(f"OK: .run() returned in {elapsed:.1f}s")

    if not isinstance(result, tuple) or len(result) != 2:
        print(f"FAIL: expected 2-tuple, got {type(result).__name__} len={len(result) if hasattr(result,'__len__') else '?'}")
        return False

    text, tokens = result
    text_preview = text[:200] + "..." if len(text) > 200 else text
    print(f"   output[0] (text, {len(text)} chars):")
    for line in text_preview.split("\n")[:5]:
        print(f"     {line}")
    print(f"   output[1] (tokens_used): {tokens}")

    if not text or not text.strip():
        print(f"FAIL: empty text output")
        return False

    print(f"PASS: {label}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "which",
        choices=[
            "gpt", "claude_haiku", "gemini_flash",
            "gpt55", "claude_opus48", "gemini_pro",
            "codex", "all_cheap",
        ],
    )
    args = parser.parse_args()

    print("Loading .env and package...")
    load_dotenv()
    pkg = load_package()
    print(f"OK: package loaded, {len(pkg.NODE_CLASS_MAPPINGS)} nodes registered")

    user_prompt = "Reply with exactly: 'GENESIS-KIE-OK ' followed by today's day of the week in English."

    tests = {
        # Cheap tier (~$0.003-0.005 each):
        "gpt": (
            "GenesisKieGPT5_2",
            {
                "user_prompt": user_prompt,
                "system_prompt": "You follow instructions precisely.",
                "image_url": "",
                "max_tokens": 64,
                "temperature": 0.0,
            },
            "GPT 5.2 (Chat Completions)",
            "0.005",
        ),
        "claude_haiku": (
            "GenesisKieClaudeHaiku4_5",
            {
                "user_prompt": user_prompt,
                "system_prompt": "You follow instructions precisely.",
                "image_url": "",
                "max_tokens": 64,
                "temperature": 0.0,
                "thinking": False,
                "thinking_budget": 4096,
            },
            "Claude Haiku 4.5 (Messages)",
            "0.005",
        ),
        "gemini_flash": (
            "GenesisKieGemini3_5Flash",
            {
                "user_prompt": user_prompt,
                "system_prompt": "You follow instructions precisely.",
                "image_url": "",
                "max_tokens": 64,
                "temperature": 0.0,
            },
            "Gemini 3.5 Flash (Chat Completions)",
            "0.003",
        ),
        # Premium tier (~$0.02-0.05 each):
        "gpt55": (
            "GenesisKieGPT5_5",
            {
                "user_prompt": user_prompt,
                "system_prompt": "You follow instructions precisely.",
                "image_url": "",
                "max_tokens": 128,
                "temperature": 0.0,
                "reasoning_effort": "low",
            },
            "GPT 5.5 (Responses API + reasoning=low)",
            "0.04",
        ),
        "claude_opus48": (
            "GenesisKieClaudeOpus4_8",
            {
                "user_prompt": user_prompt,
                "system_prompt": "You follow instructions precisely.",
                "image_url": "",
                "max_tokens": 128,
                "temperature": 0.0,
                "thinking": False,
                "thinking_budget": 4096,
            },
            "Claude Opus 4.8 (frontier)",
            "0.05",
        ),
        "gemini_pro": (
            "GenesisKieGemini3_1Pro",
            {
                "user_prompt": user_prompt,
                "system_prompt": "You follow instructions precisely.",
                "image_url": "",
                "max_tokens": 128,
                "temperature": 0.0,
            },
            "Gemini 3.1 Pro (frontier)",
            "0.02",
        ),
        "codex": (
            "GenesisKieGPTCodex",
            {
                "user_prompt": "Write a Python one-liner that prints 'hello world'.",
                "system_prompt": "",
                "image_url": "",
                "max_tokens": 2048,
                "temperature": 0.0,
                "reasoning_effort": "medium",
                "codex_version": "gpt-5.4-codex",
            },
            "GPT Codex 5.4 (Responses API)",
            "0.37",
        ),
    }

    if args.which == "all_cheap":
        to_run = ["gemini_flash", "claude_haiku", "gpt"]
    else:
        to_run = [args.which]

    results: dict[str, bool] = {}
    total_cost = 0.0

    for key in to_run:
        node_key, kwargs, label, cost = tests[key]
        ok = smoke_one(pkg, node_key, kwargs, label, cost)
        results[key] = ok
        try:
            total_cost += float(cost)
        except ValueError:
            pass

    print()
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for key, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {status:5s}  {key}")
    print(f"  Approx spend: ~${total_cost:.3f}")
    print("=" * 60)

    failed = sum(1 for ok in results.values() if not ok)
    sys.exit(failed)


if __name__ == "__main__":
    main()
