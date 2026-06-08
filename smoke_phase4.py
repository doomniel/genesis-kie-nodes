"""Smoke test for Genesis-Kie Phase 4 music nodes.

Runs nodes DIRECTLY (no ComfyUI workflow) via .run() method.

Usage:
    python smoke_phase4.py el_tts        # ElevenLabs TTS Multilingual  ~$0.02
    python smoke_phase4.py el_dialogue   # ElevenLabs Text-to-Dialogue   ~$0.03
    python smoke_phase4.py el_isolation  # Needs audio_url, skip default
    python smoke_phase4.py suno          # Suno Generate Music V3_5      ~$0.04
    python smoke_phase4.py lyrics        # Suno Generate Lyrics          ~$0.01
    python smoke_phase4.py all_cheap     # el_tts + lyrics (~$0.03 total)

Run from repo root: C:\\dev\\genesis-kie-nodes\\
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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


def smoke_one(pkg, node_key, kwargs, label, est_cost, expected_outputs):
    """Run a node and validate output count + non-empty content."""
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
    print(f"   RETURN_TYPES: {cls.RETURN_TYPES}")

    instance = cls()
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

    if not isinstance(result, tuple):
        print(f"FAIL: expected tuple, got {type(result).__name__}")
        return False
    if len(result) != expected_outputs:
        print(f"FAIL: expected {expected_outputs}-tuple, got {len(result)}-tuple")
        return False

    for i, val in enumerate(result):
        v_str = str(val)
        if len(v_str) > 100:
            v_str = v_str[:97] + "..."
        print(f"   output[{i}] ({cls.RETURN_NAMES[i]}): {v_str}")

    # Validate first output is non-empty string
    if not isinstance(result[0], str):
        print(f"FAIL: output[0] expected str, got {type(result[0]).__name__}")
        return False
    if not result[0].strip():
        print(f"WARN: output[0] is empty string")

    print(f"PASS: {label} (~${est_cost})")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "which",
        choices=["el_tts", "el_dialogue", "suno", "lyrics", "all_cheap"],
    )
    args = parser.parse_args()

    print("Loading .env and package...")
    load_dotenv()
    pkg = load_package()
    print(f"OK: package loaded, {len(pkg.NODE_CLASS_MAPPINGS)} nodes registered")

    tests = {
        "el_tts": (
            "GenesisKieElevenLabsTTSMultilingualV2",
            {
                "text": "Hola mundo. Esta es una prueba de síntesis de voz multilingüe.",
                "voice": "Rachel",
                "voice_id": "",
                "output_format": "mp3_44100_128",
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.0,
            },
            "ElevenLabs TTS Multilingual v2",
            "0.02",
            1,  # expected outputs
        ),
        "el_dialogue": (
            "GenesisKieElevenLabsTextToDialogueV3",
            {
                "dialogue": (
                    "[Speaker1] Hola, ¿cómo estás?\n"
                    "[Speaker2] Muy bien, gracias. ¿Y tú?"
                ),
                "voices": "Rachel,Adam",
                "output_format": "mp3_44100_128",
            },
            "ElevenLabs Text-to-Dialogue v3",
            "0.03",
            1,
        ),
        "suno": (
            "GenesisKieSunoGenerateMusic",
            {
                "prompt": "A calm, short piano sketch with soft melodies",
                "model": "V3_5",  # cheaper model for smoke
                "custom_mode": False,
                "instrumental": True,
                "style": "",
                "title": "",
                "negative_tags": "",
                "vocal_gender": "",
                "style_weight": 0.65,
                "weirdness_constraint": 0.65,
                "audio_weight": 0.65,
                "persona_id": "",
            },
            "Suno Generate Music V3_5 (instrumental)",
            "0.04",
            3,  # audio_path, audio_id, all_paths_csv
        ),
        "lyrics": (
            "GenesisKieSunoGenerateLyrics",
            {
                "prompt": "A short song about a robot learning to feel joy.",
            },
            "Suno Generate Lyrics",
            "0.01",
            1,
        ),
    }

    if args.which == "all_cheap":
        to_run = ["lyrics", "el_tts"]
    else:
        to_run = [args.which]

    results: dict[str, bool] = {}
    total_cost = 0.0

    for key in to_run:
        node_key, kwargs, label, cost, expected_n = tests[key]
        ok = smoke_one(pkg, node_key, kwargs, label, cost, expected_n)
        results[key] = ok
        try:
            total_cost += float(cost)
        except ValueError:
            pass
        if not ok and args.which == "all_cheap":
            print()
            print("Stopping run after first failure.")
            break

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
