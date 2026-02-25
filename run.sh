#!/usr/bin/env bash

set -e

# =============================================================================
# AST Audio Classifier — Startup Script
# Works in both standalone Docker and HA addon contexts.
# =============================================================================

OPTIONS_FILE="/data/options.json"
CONFIG_PATH="${CONFIG_PATH:-/data/config.yaml}"
PORT="${PORT:-8080}"

# -----------------------------------------------------------------------
# HA Addon mode: generate config.yaml from /data/options.json
# -----------------------------------------------------------------------
if [ -f "$OPTIONS_FILE" ]; then
    echo "HA addon mode — reading options from $OPTIONS_FILE"

    LOG_LEVEL=$(jq -r '.log_level // "info"' "$OPTIONS_FILE")

    # Try supervisor MQTT service discovery first, fall back to addon options
    SVC_HOST="" ; SVC_PORT="" ; SVC_USER="" ; SVC_PASS=""
    if [ -n "${SUPERVISOR_TOKEN:-}" ]; then
        echo "DEBUG: SUPERVISOR_TOKEN is set, querying MQTT service..."
        MQTT_SVC=$(curl -sS -H "Authorization: Bearer ${SUPERVISOR_TOKEN}" \
            http://supervisor/services/mqtt 2>&1 || echo "CURL_FAILED")
        echo "DEBUG: MQTT service response: ${MQTT_SVC:0:500}"
        if [ -n "$MQTT_SVC" ] && [ "$MQTT_SVC" != "CURL_FAILED" ]; then
            SVC_HOST=$(echo "$MQTT_SVC" | jq -r '.data.host // empty' 2>/dev/null)
            SVC_PORT=$(echo "$MQTT_SVC" | jq -r '.data.port // empty' 2>/dev/null)
            SVC_USER=$(echo "$MQTT_SVC" | jq -r '.data.username // empty' 2>/dev/null)
            SVC_PASS=$(echo "$MQTT_SVC" | jq -r '.data.password // empty' 2>/dev/null)
            echo "DEBUG: Discovered MQTT: host=${SVC_HOST} port=${SVC_PORT} user=${SVC_USER} pass_len=${#SVC_PASS}"
            if [ -n "$SVC_HOST" ]; then
                echo "Using MQTT credentials from supervisor service discovery"
            fi
        fi
    else
        echo "DEBUG: SUPERVISOR_TOKEN not set, skipping service discovery"
    fi

    MQTT_HOST="${SVC_HOST:-$(jq -r '.mqtt_host // "core-mosquitto"' "$OPTIONS_FILE")}"
    MQTT_PORT="${SVC_PORT:-$(jq -r '.mqtt_port // 1883' "$OPTIONS_FILE")}"
    MQTT_USER="${SVC_USER:-$(jq -r '.mqtt_username // ""' "$OPTIONS_FILE")}"
    MQTT_PASS="${SVC_PASS:-$(jq -r '.mqtt_password // ""' "$OPTIONS_FILE")}"

    CONFIDENCE=$(jq -r '.confidence_threshold // 0.15' "$OPTIONS_FILE")
    AUTO_OFF=$(jq -r '.auto_off_seconds // 30' "$OPTIONS_FILE")
    CLIP_DUR=$(jq -r '.clip_duration_seconds // 3' "$OPTIONS_FILE")

    OO_HOST=$(jq -r '.openobserve_host // ""' "$OPTIONS_FILE")
    OO_PORT=$(jq -r '.openobserve_port // 5080' "$OPTIONS_FILE")
    OO_ORG=$(jq -r '.openobserve_org // "default"' "$OPTIONS_FILE")
    OO_STREAM=$(jq -r '.openobserve_stream // "ast_audio"' "$OPTIONS_FILE")
    OO_USER=$(jq -r '.openobserve_username // ""' "$OPTIONS_FILE")
    OO_PASS=$(jq -r '.openobserve_password // ""' "$OPTIONS_FILE")

    # Build config.yaml
    cat > "$CONFIG_PATH" <<YAML
mqtt:
  host: "${MQTT_HOST}"
  port: ${MQTT_PORT}
YAML

    if [ -n "$MQTT_USER" ] && [ "$MQTT_USER" != "null" ]; then
        cat >> "$CONFIG_PATH" <<YAML
  username: "${MQTT_USER}"
  password: "${MQTT_PASS}"
YAML
    fi

    # Cameras from JSON array
    echo "" >> "$CONFIG_PATH"
    echo "cameras:" >> "$CONFIG_PATH"
    CAMERA_COUNT=$(jq '.cameras | length' "$OPTIONS_FILE")
    for i in $(seq 0 $(( CAMERA_COUNT - 1 ))); do
        CAM_NAME=$(jq -r ".cameras[$i].name" "$OPTIONS_FILE")
        CAM_URL=$(jq -r ".cameras[$i].rtsp_url" "$OPTIONS_FILE")
        CAM_DB=$(jq -r ".cameras[$i].db_threshold // -35" "$OPTIONS_FILE")
        CAM_CD=$(jq -r ".cameras[$i].cooldown_seconds // 10" "$OPTIONS_FILE")
        CAM_BATT=$(jq -r ".cameras[$i].battery // false" "$OPTIONS_FILE")
        CAM_RECON=$(jq -r ".cameras[$i].reconnect_interval // 5" "$OPTIONS_FILE")

        cat >> "$CONFIG_PATH" <<YAML
  - name: "${CAM_NAME}"
    rtsp_url: "${CAM_URL}"
    db_threshold: ${CAM_DB}
    cooldown_seconds: ${CAM_CD}
    battery: ${CAM_BATT}
    reconnect_interval: ${CAM_RECON}
YAML
    done

    # OpenObserve (optional)
    if [ -n "$OO_HOST" ] && [ "$OO_HOST" != "null" ] && [ "$OO_HOST" != "" ]; then
        cat >> "$CONFIG_PATH" <<YAML

openobserve:
  host: "${OO_HOST}"
  port: ${OO_PORT}
  org: "${OO_ORG}"
  stream: "${OO_STREAM}"
YAML
        if [ -n "$OO_USER" ] && [ "$OO_USER" != "null" ]; then
            cat >> "$CONFIG_PATH" <<YAML
  username: "${OO_USER}"
  password: "${OO_PASS}"
YAML
        fi
    fi

    # CLAP verification (optional)
    CLAP_ENABLED=$(jq -r '.clap_enabled // false' "$OPTIONS_FILE")
    if [ "$CLAP_ENABLED" = "true" ]; then
        CLAP_MODEL=$(jq -r '.clap_model // "laion/clap-htsat-fused"' "$OPTIONS_FILE")
        CLAP_CONFIRM=$(jq -r '.clap_confirm_threshold // 0.25' "$OPTIONS_FILE")
        CLAP_SUPPRESS=$(jq -r '.clap_suppress_threshold // 0.15' "$OPTIONS_FILE")
        CLAP_OVERRIDE=$(jq -r '.clap_override_threshold // 0.40' "$OPTIONS_FILE")
        CLAP_DISCOVERY=$(jq -r '.clap_discovery_threshold // 0.50' "$OPTIONS_FILE")

        cat >> "$CONFIG_PATH" <<YAML

clap:
  enabled: true
  model: "${CLAP_MODEL}"
  confirm_threshold: ${CLAP_CONFIRM}
  suppress_threshold: ${CLAP_SUPPRESS}
  override_threshold: ${CLAP_OVERRIDE}
  discovery_threshold: ${CLAP_DISCOVERY}
YAML

        # Custom prompts (optional JSON object)
        CLAP_PROMPTS=$(jq -r '.clap_custom_prompts // empty' "$OPTIONS_FILE" 2>/dev/null)
        if [ -n "$CLAP_PROMPTS" ] && [ "$CLAP_PROMPTS" != "null" ]; then
            echo "  custom_prompts:" >> "$CONFIG_PATH"
            echo "$CLAP_PROMPTS" | jq -r 'to_entries[] | "    \(.key):\n" + (.value | map("      - \"\(.)\"") | join("\n"))' >> "$CONFIG_PATH"
        fi
    fi

    # Defaults
    cat >> "$CONFIG_PATH" <<YAML

defaults:
  confidence_threshold: ${CONFIDENCE}
  auto_off_seconds: ${AUTO_OFF}
  clip_duration_seconds: ${CLIP_DUR}
YAML

    echo "Generated config at $CONFIG_PATH with ${CAMERA_COUNT} cameras"

# -----------------------------------------------------------------------
# Standalone Docker mode: use provided config file
# -----------------------------------------------------------------------
else
    echo "Standalone mode — using config at $CONFIG_PATH"
    LOG_LEVEL="${LOG_LEVEL:-INFO}"
fi

echo "=========================================="
echo " AST Audio Classifier"
echo "=========================================="
echo "Version: 0.1.0"
echo "Config:  $CONFIG_PATH"
echo "Log:     $LOG_LEVEL"
echo "Port:    $PORT"
echo "=========================================="

export CONFIG_PATH
export LOG_LEVEL

cd /app
exec python -m uvicorn src.main:create_app \
    --factory \
    --host "0.0.0.0" \
    --port "$PORT" \
    --log-level "$(echo "$LOG_LEVEL" | tr '[:upper:]' '[:lower:]')"
