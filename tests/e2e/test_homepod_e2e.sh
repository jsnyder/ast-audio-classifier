#!/usr/bin/env bash
# E2E test: Play audio samples through HomePod and verify AST classifier
# detects them via MQTT state changes in Home Assistant.
#
# Requirements:
#   - HA accessible at homeassistant.local
#   - ./scripts/homeassist CLI configured
#   - HomePod media_player entity configured
#   - AST Audio Classifier addon running
#
# Usage: ./test_homepod_e2e.sh [MEDIA_PLAYER] [SAMPLE_DIR]

set -euo pipefail

HOMEASSIST="./scripts/homeassist"
MEDIA_PLAYER="${1:-media_player.living_room_homepod}"
SAMPLES_DIR="${2:-$(cd "$(dirname "$0")/samples" && pwd)}"
# The camera that should hear the HomePod playback
CAMERA="living_room"
WAIT_SECONDS=15
PASS=0
FAIL=0
SKIP=0
RESULTS=()

# Samples to test and their expected entity groups
declare -A TESTS=(
  [dog_bark]="binary_sensor.ast_audio_classifier_living_room_ast_living_room_dog_bark"
  [cat_meow]="binary_sensor.ast_audio_classifier_living_room_ast_living_room_cat_meow"
  [smoke_alarm]="binary_sensor.ast_audio_classifier_living_room_ast_living_room_smoke_alarm"
  [speech]="binary_sensor.ast_audio_classifier_living_room_ast_living_room_speech"
  [doorbell]="binary_sensor.ast_audio_classifier_living_room_ast_living_room_doorbell"
  [glass_break]="binary_sensor.ast_audio_classifier_living_room_ast_living_room_glass_break"
)

echo "=== AST Audio Classifier HomePod E2E Test ==="
echo "Media player: $MEDIA_PLAYER"
echo "Camera: $CAMERA"
echo "Samples: $SAMPLES_DIR"
echo "Wait time: ${WAIT_SECONDS}s per sample"
echo ""

# Verify HA connection
echo "--- Checking HA Connection ---"
$HOMEASSIST health 2>&1 | head -3
echo ""

# Verify AST classifier is running
echo "--- Checking AST Classifier ---"
STATUS=$(curl -sf "http://homeassistant.local:8080/health" 2>&1) || {
  echo "FAIL: AST classifier not reachable"
  exit 1
}
echo "$STATUS" | python3 -m json.tool 2>/dev/null || echo "$STATUS"
echo ""

# First, copy WAV files to HA's www directory for media playback
echo "--- Uploading test samples to HA ---"
for wav in "$SAMPLES_DIR"/*.wav; do
  name=$(basename "$wav")
  scp -q -i ~/.ssh/id_ed25519 "$wav" "root@homeassistant.local:/config/www/ast_test_$name" 2>/dev/null && \
    echo "  Uploaded: ast_test_$name" || \
    echo "  WARN: Failed to upload $name"
done
echo ""

echo "--- Running E2E Tests ---"
for sample_name in "${!TESTS[@]}"; do
  entity="${TESTS[$sample_name]}"
  wav_file="$SAMPLES_DIR/${sample_name}.wav"

  if [ ! -f "$wav_file" ]; then
    echo "  $sample_name: SKIP (no WAV file)"
    SKIP=$((SKIP + 1))
    continue
  fi

  echo -n "  $sample_name: "

  # Record current state of the entity
  BEFORE=$($HOMEASSIST entities get "$entity" 2>/dev/null | grep -oE '"state": *"[^"]*"' | head -1 || echo "unknown")

  # Play the audio through HomePod
  $HOMEASSIST services call media_player play_media "{
    \"entity_id\": \"$MEDIA_PLAYER\",
    \"media_content_id\": \"/local/ast_test_${sample_name}.wav\",
    \"media_content_type\": \"music\"
  }" >/dev/null 2>&1 || {
    echo "FAIL (could not play media)"
    FAIL=$((FAIL + 1))
    RESULTS+=("FAIL $sample_name: media play failed")
    continue
  }

  # Wait for detection
  DETECTED=false
  for i in $(seq 1 "$WAIT_SECONDS"); do
    sleep 1
    STATE=$($HOMEASSIST entities get "$entity" 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('state', 'unknown'))
except:
    print('unknown')
" 2>/dev/null || echo "unknown")

    if [ "$STATE" = "on" ]; then
      DETECTED=true
      # Get attributes for confidence
      ATTRS=$($HOMEASSIST entities get "$entity" 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    a = d.get('attributes', {})
    print(f\"conf={a.get('confidence','?')}, db={a.get('db_level','?')}\")
except:
    print('?')
" 2>/dev/null || echo "?")
      echo "DETECTED in ${i}s ($ATTRS)"
      PASS=$((PASS + 1))
      RESULTS+=("PASS $sample_name: detected in ${i}s ($ATTRS)")
      break
    fi
  done

  if [ "$DETECTED" = "false" ]; then
    # Check last_event sensor for what WAS detected
    LAST_EVENT=$($HOMEASSIST entities get "sensor.ast_audio_classifier_${CAMERA}_ast_${CAMERA}_last_event" 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    a = d.get('attributes', {})
    print(f\"state={d.get('state','?')}, group={a.get('group','?')}, conf={a.get('confidence','?')}\")
except:
    print('?')
" 2>/dev/null || echo "?")
    echo "NOT DETECTED after ${WAIT_SECONDS}s (last_event: $LAST_EVENT)"
    FAIL=$((FAIL + 1))
    RESULTS+=("FAIL $sample_name: not detected after ${WAIT_SECONDS}s (last: $LAST_EVENT)")
  fi

  # Small pause between tests to avoid cooldown overlap
  sleep 3
done

echo ""
echo "=== Summary ==="
echo "Passed: $PASS / Failed: $FAIL / Skipped: $SKIP"
echo ""
for r in "${RESULTS[@]}"; do
  echo "  $r"
done

# Cleanup test files from HA
echo ""
echo "--- Cleanup ---"
ssh -i ~/.ssh/id_ed25519 root@homeassistant.local "rm -f /config/www/ast_test_*.wav" 2>/dev/null && \
  echo "  Removed test samples from HA" || \
  echo "  WARN: Could not clean up test samples"

exit "$FAIL"
