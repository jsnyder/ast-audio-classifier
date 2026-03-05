#!/usr/bin/env python3
"""E2E test: Upload WAV samples to /classify endpoint and verify classifications.

Usage: python3 test_classify_api.py [HOST]
Default HOST: http://localhost:8080
"""

import json
import subprocess
import sys
from pathlib import Path

HOST = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"
SAMPLES_DIR = Path(__file__).parent / "samples"

# Expected group for each sample (pipe-separated alternatives)
EXPECTED = {
    "dog_bark": "dog_bark",
    "cat_meow": "cat_meow",
    "glass_break": "glass_break",
    "smoke_alarm": "smoke_alarm|alarm_beep",
    "doorbell": "doorbell|alarm_beep",
    "baby_cry": "crying",
    "speech": "speech",
    "bird_song": "",  # no specific group required
}

passed = 0
failed = 0
results = []


def curl_json(url: str) -> dict:
    r = subprocess.run(
        ["curl", "-sf", url], capture_output=True, text=True, timeout=10
    )
    if r.returncode != 0:
        raise RuntimeError(f"curl failed: {r.stderr}")
    return json.loads(r.stdout)


def curl_upload(url: str, filepath: Path) -> dict:
    r = subprocess.run(
        ["curl", "-sf", "-X", "POST", url, "-F", f"file=@{filepath}"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if r.returncode != 0:
        raise RuntimeError(f"curl failed: {r.stderr}")
    return json.loads(r.stdout)


print("=== AST Audio Classifier E2E API Test ===")
print(f"Host: {HOST}")
print(f"Samples: {SAMPLES_DIR}")
print()

# Health check
print("--- Health Check ---")
try:
    health = curl_json(f"{HOST}/health")
    print(json.dumps(health, indent=2))
except Exception as e:
    print(f"FAIL: Cannot reach {HOST}/health: {e}")
    sys.exit(1)
print()

# Classification tests
print("--- Classification Tests ---")
for wav in sorted(SAMPLES_DIR.glob("*.wav")):
    name = wav.stem
    expected = EXPECTED.get(name, "")
    expected_groups = set(expected.split("|")) if expected else set()

    print(f"  {name}: ", end="", flush=True)

    try:
        resp = curl_upload(f"{HOST}/classify", wav)
    except Exception as e:
        print(f"FAIL (request error: {e})")
        failed += 1
        results.append(f"FAIL {name}: request error")
        continue

    classifications = resp.get("results", [])
    db_level = resp.get("db_level", "?")
    num = len(classifications)

    if num == 0:
        if not expected_groups:
            print(f"OK (no group expected, 0 results, {db_level} dB)")
            passed += 1
            results.append(f"PASS {name}: no results (acceptable)")
        else:
            print(f"FAIL (0 results, expected {expected})")
            failed += 1
            results.append(f"FAIL {name}: 0 results, expected {expected}")
        continue

    top = classifications[0]
    top_group = top["group"]
    top_conf = top["confidence"]
    all_groups = ", ".join(
        f"{r['group']}({r['confidence']:.2f})" for r in classifications
    )

    if not expected_groups:
        print(f"OK ({all_groups}, {db_level} dB)")
        passed += 1
        results.append(f"PASS {name}: {all_groups}")
    elif top_group in expected_groups:
        print(f"OK {top_group} ({top_conf:.3f}) [{db_level} dB] [{all_groups}]")
        passed += 1
        results.append(f"PASS {name}: {top_group} ({top_conf:.3f})")
    else:
        # Check if ANY result matches expected (not just top)
        any_match = [r for r in classifications if r["group"] in expected_groups]
        if any_match:
            m = any_match[0]
            print(
                f"WARN top={top_group}({top_conf:.2f}), "
                f"but {m['group']}({m['confidence']:.2f}) also found "
                f"[{db_level} dB] [{all_groups}]"
            )
            passed += 1
            results.append(
                f"PASS {name}: {m['group']}({m['confidence']:.2f}) "
                f"(top was {top_group})"
            )
        else:
            print(
                f"UNEXPECTED {top_group} ({top_conf:.3f}), "
                f"expected {expected} [{db_level} dB] [{all_groups}]"
            )
            failed += 1
            results.append(f"FAIL {name}: got {top_group}, expected {expected}")

print()
print("=== Summary ===")
print(f"Passed: {passed}")
print(f"Failed: {failed}")
print()
for r in results:
    print(f"  {r}")

sys.exit(min(failed, 125))
