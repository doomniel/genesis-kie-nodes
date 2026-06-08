"""Smoke test for Genesis-Kie Phase 3 image nodes.

Runs nodes DIRECTLY (no ComfyUI workflow) via their .run() method.
Tests real Kie API calls and verifies IMAGE tensor output.

Usage:
    python smoke_phase3.py gpt4o          # ~$0.03
    python smoke_phase3.py kontext        # ~$0.04
    python smoke_phase3.py seedream45     # ~$0.03
    python smoke_phase3.py all            # runs all 3 (~$0.10)

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
    """Load the genesis-kie-nodes package by file path (skip Python path issues)."""
    repo_root = Path.cwd()
    if not (repo_root / "__init__.py").exists():
        raise SystemExit(f"FATAL: not in repo root. cwd={repo_root}")

    # parent of repo as sys.path so the import works under repo name
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
    """Manually load .env into os.environ (no python-dotenv required)."""
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


def describe_tensor(t) -> str:
    """Describe a torch tensor without requiring torch at import time."""
    try:
        shape = tuple(t.shape)
        dtype = str(t.dtype)
        return f"tensor shape={shape} dtype={dtype}"
    except Exception as e:
        return f"<not-a-tensor: {type(t).__name__}: {e}>"


def smoke_one(pkg, node_key: str, kwargs: dict, label: str, est_cost: str) -> bool:
    """Instantiate and run one node. Return True if success."""
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
    print(f"   MODEL:    {getattr(cls, 'MODEL', '(none)')}")

    instance = cls()
    print(f"OK: instance created")
    print(f"   inputs sent:")
    for k, v in kwargs.items():
        v_str = str(v)
        if len(v_str) > 80:
            v_str = v_str[:77] + "..."
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

    # Validate output shape
    if not isinstance(result, tuple):
        print(f"FAIL: expected tuple, got {type(result).__name__}")
        return False
    if len(result) != 1:
        print(f"FAIL: expected 1-tuple, got {len(result)}-tuple")
        return False

    tensor = result[0]
    print(f"   output[0]: {describe_tensor(tensor)}")

    # Should be (B, H, W, C) with B>=1, C=3
    try:
        shape = tuple(tensor.shape)
        if len(shape) != 4 or shape[3] != 3 or shape[0] < 1:
            print(f"FAIL: unexpected tensor shape {shape}, want (B>=1, H, W, 3)")
            return False
    except Exception as e:
        print(f"FAIL: cannot inspect tensor: {e}")
        return False

    print(f"PASS: {label} (~${est_cost})")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("which", choices=["gpt4o", "kontext", "seedream45", "all"])
    args = parser.parse_args()

    print("Loading .env and package...")
    load_dotenv()
    pkg = load_package()
    print(f"OK: package loaded, {len(pkg.NODE_CLASS_MAPPINGS)} nodes registered")

    # --- Test definitions ---------------------------------------------------

    tests = {
        "gpt4o": (
            "GenesisKieGPT4oImage",
            {
                "prompt": "A photorealistic sunrise over snowy mountain peaks, golden light, cinematic.",
                "size": "1:1",
                "n_variants": 1,
                "is_enhance": False,
                "upload_cn": False,
                "enable_fallback": False,
                "fallback_model": "NONE",
                "files_url": "",
            },
            "GPT 4o Image (dedicated)",
            "0.03",
        ),
        "kontext": (
            "GenesisKieFluxKontextPro",
            {
                "prompt": "A serene mountain landscape at sunset with a lake reflecting the orange sky.",
                "model": "flux-kontext-pro",
                "aspect_ratio": "16:9",
                "output_format": "jpeg",
                "input_image": "",
                "prompt_upsampling": False,
                "enable_translation": True,
                "safety_tolerance": 2,
            },
            "Flux Kontext Pro (dedicated)",
            "0.04",
        ),
        "seedream45": (
            "GenesisKieSeedream45T2I",
            {
                "prompt": "A minimalist poster for a coffee shop, modern typography, warm tones.",
                "aspect_ratio": "1:1",
                "quality": "basic",
                "nsfw_checker": False,
            },
            "Seedream 4.5 T2I (Market sanity check)",
            "0.03",
        ),
    }

    # --- Execute ------------------------------------------------------------

    to_run = [args.which] if args.which != "all" else ["gpt4o", "kontext", "seedream45"]
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
        if not ok and args.which == "all":
            print()
            print("Stopping 'all' run after first failure to save budget.")
            break

    # --- Summary ------------------------------------------------------------

    print()
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for key, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {status:5s}  {key}")
    print(f"  Approx spend: ~${total_cost:.2f}")
    print("=" * 60)

    failed = sum(1 for ok in results.values() if not ok)
    sys.exit(failed)


if __name__ == "__main__":
    main()
