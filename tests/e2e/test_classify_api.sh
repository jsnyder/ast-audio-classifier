#!/usr/bin/env bash
# E2E test: Upload WAV samples to /classify endpoint and verify classifications.
# Usage: ./test_classify_api.sh [HOST]
# Default HOST: http://localhost:8080

set -euo pipefail

HOST="${1:-http://localhost:8080}"
SAMPLES_DIR="$(cd "$(dirname "$0")/samples" && pwd)"
PASS=0
FAIL=0
RESULTS=()

# Expected group for each sample (primary expected classification)
declare -A EXPECTED=(
  [dog_bark]="dog_bark"
  [cat_meow]="cat_meow"
  [glass_break]="glass_break"
  [smoke_alarm]="smoke_alarm|alarm_beep"
  [doorbell]="doorbell|alarm_beep"
  [baby_cry]="crying"
  [speech]="speech"
  [bird_song]=""  # no specific group expected, just verify it classifies
)

echo "=== AST Audio Classifier E2E API Test ==="
echo "Host: $HOST"
echo "Samples: $SAMPLES_DIR"
echo ""

# Check health first
echo "--- Health Check ---"
HEALTH=$(curl -sf "$HOST/health" 2>&1) || { echo "FAIL: Cannot reach $HOST/health"; exit 1; }
echo "$HEALTH" | python3 -m json.tool 2>/dev/null || echo "$HEALTH"
echo ""

echo "--- Classification Tests ---"
for wav in "$SAMPLES_DIR"/*.wav; do
  name=$(basename "$wav" .wav)
  expected="${EXPECTED[$name]:-}"

  echo -n "  $name: "
  RESPONSE=$(curl -sf -X POST "$HOST/classify" -F "file=@$wav" 2>&1) || {
    echo "FAIL (request error)"
    FAIL=$((FAIL + 1))
    RESULTS+=("FAIL $name: request error")
    continue
  }

  # Extract results
  NUM_RESULTS=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('results',[])))" 2>/dev/null || echo 0)
  DB_LEVEL=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('db_level','?'))" 2>/dev/null || echo "?")

  if [ "$NUM_RESULTS" = "0" ]; then
    if [ -z "$expected" ]; then
      echo "OK (no group expected, $NUM_RESULTS results, ${DB_LEVEL} dB)"
      PASS=$((PASS + 1))
      RESULTS+=("PASS $name: no results (acceptable)")
    else
      echo "FAIL (0 results, expected $expected)"
      FAIL=$((FAIL + 1))
      RESULTS+=("FAIL $name: 0 results, expected $expected")
    fi
    continue
  fi

  # Get top result group and confidence
  TOP_GROUP=$(echo "$RESPONSE" | python3 -c "
import sys, json
d = json.load(sys.stdin)
r = d['results']
print(r[0]['group'] if r else 'none')
" 2>/dev/null || echo "parse_error")

  TOP_CONF=$(echo "$RESPONSE" | python3 -c "
import sys, json
d = json.load(sys.stdin)
r = d['results']
print(f\"{r[0]['confidence']:.3f}\" if r else '0')
" 2>/dev/null || echo "?")

  ALL_GROUPS=$(echo "$RESPONSE" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(', '.join(f\"{r['group']}({r['confidence']:.2f})\" for r in d['results']))
" 2>/dev/null || echo "?")

  if [ -z "$expected" ]; then
    # No specific group expected, just report
    echo "OK ($ALL_GROUPS, ${DB_LEVEL} dB)"
    PASS=$((PASS + 1))
    RESULTS+=("PASS $name: $ALL_GROUPS")
  elif echo "$expected" | grep -qE "(^|\\|)$TOP_GROUP($|\\|)"; then
    echo "OK $TOP_GROUP ($TOP_CONF) [${DB_LEVEL} dB] [$ALL_GROUPS]"
    PASS=$((PASS + 1))
    RESULTS+=("PASS $name: $TOP_GROUP ($TOP_CONF)")
  else
    echo "UNEXPECTED $TOP_GROUP ($TOP_CONF), expected $expected [${DB_LEVEL} dB] [$ALL_GROUPS]"
    FAIL=$((FAIL + 1))
    RESULTS+=("FAIL $name: got $TOP_GROUP, expected $expected")
  fi
done

echo ""
echo "=== Summary ==="
echo "Passed: $PASS"
echo "Failed: $FAIL"
echo ""
for r in "${RESULTS[@]}"; do
  echo "  $r"
done

exit "$FAIL"
