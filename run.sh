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
        echo "Querying supervisor MQTT service discovery..."
        MQTT_SVC=$(curl -sS -H "Authorization: Bearer ${SUPERVISOR_TOKEN}" \
            http://supervisor/services/mqtt 2>&1 || echo "CURL_FAILED")
        if [ -n "$MQTT_SVC" ] && [ "$MQTT_SVC" != "CURL_FAILED" ]; then
            SVC_HOST=$(echo "$MQTT_SVC" | jq -r '.data.host // empty' 2>/dev/null)
            SVC_PORT=$(echo "$MQTT_SVC" | jq -r '.data.port // empty' 2>/dev/null)
            SVC_USER=$(echo "$MQTT_SVC" | jq -r '.data.username // empty' 2>/dev/null)
            SVC_PASS=$(echo "$MQTT_SVC" | jq -r '.data.password // empty' 2>/dev/null)
            if [ -n "$SVC_HOST" ]; then
                echo "MQTT discovered via supervisor: host=${SVC_HOST} port=${SVC_PORT} user=${SVC_USER}"
            fi
        fi
    fi

    MQTT_HOST="${SVC_HOST:-$(jq -r '.mqtt_host // "core-mosquitto"' "$OPTIONS_FILE")}"
    MQTT_PORT="${SVC_PORT:-$(jq -r '.mqtt_port // 1883' "$OPTIONS_FILE")}"
    MQTT_USER="${SVC_USER:-$(jq -r '.mqtt_username // ""' "$OPTIONS_FILE")}"
    MQTT_PASS="${SVC_PASS:-$(jq -r '.mqtt_password // ""' "$OPTIONS_FILE")}"

    SCRYPTED_API_URL=$(jq -r '.scrypted_api_url // ""' "$OPTIONS_FILE")

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
        CAM_HPF=$(jq -r ".cameras[$i].highpass_freq // 0" "$OPTIONS_FILE")
        CAM_ADAPT=$(jq -r ".cameras[$i].adaptive_threshold // false" "$OPTIONS_FILE")
        CAM_MARGIN=$(jq -r ".cameras[$i].adaptive_margin_db // 8.0" "$OPTIONS_FILE")
        CAM_SCRYPTED_ID=$(jq -r ".cameras[$i].scrypted_device_id // \"\"" "$OPTIONS_FILE")

        cat >> "$CONFIG_PATH" <<YAML
  - name: "${CAM_NAME}"
    rtsp_url: "${CAM_URL}"
    db_threshold: ${CAM_DB}
    cooldown_seconds: ${CAM_CD}
    battery: ${CAM_BATT}
    reconnect_interval: ${CAM_RECON}
    highpass_freq: ${CAM_HPF}
    adaptive_threshold: ${CAM_ADAPT}
    adaptive_margin_db: ${CAM_MARGIN}
YAML
        if [ -n "$CAM_SCRYPTED_ID" ] && [ "$CAM_SCRYPTED_ID" != "null" ]; then
            echo "    scrypted_device_id: \"${CAM_SCRYPTED_ID}\"" >> "$CONFIG_PATH"
        fi

        # Confounders (optional per-camera)
        CONF_COUNT=$(jq ".cameras[$i].confounders | length" "$OPTIONS_FILE" 2>/dev/null || echo 0)
        if [ "$CONF_COUNT" -gt 0 ] 2>/dev/null; then
            echo "    confounders:" >> "$CONFIG_PATH"
            for j in $(seq 0 $(( CONF_COUNT - 1 ))); do
                CONF_ENTITY=$(jq -r ".cameras[$i].confounders[$j].entity_id" "$OPTIONS_FILE")
                CONF_WHEN=$(jq -r ".cameras[$i].confounders[$j].active_when" "$OPTIONS_FILE")
                CG_COUNT=$(jq ".cameras[$i].confounders[$j].confused_groups | length" "$OPTIONS_FILE" 2>/dev/null || echo 0)
                if [ "$CG_COUNT" -eq 0 ] 2>/dev/null; then
                    echo "WARNING: confounder $j on camera $CAM_NAME has no confused_groups, skipping" >&2
                    continue
                fi
                cat >> "$CONFIG_PATH" <<YAML
      - entity_id: "${CONF_ENTITY}"
        active_when: "${CONF_WHEN}"
        confused_groups:
YAML
                for k in $(seq 0 $(( CG_COUNT - 1 ))); do
                    CG=$(jq -r ".cameras[$i].confounders[$j].confused_groups[$k]" "$OPTIONS_FILE")
                    echo "          - \"${CG}\"" >> "$CONFIG_PATH"
                done
            done
        fi
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
        CLAP_CONFIRM=$(jq -r '.clap_confirm_threshold // 0.30' "$OPTIONS_FILE")
        CLAP_SUPPRESS=$(jq -r '.clap_suppress_threshold // 0.15' "$OPTIONS_FILE")
        CLAP_OVERRIDE=$(jq -r '.clap_override_threshold // 0.40' "$OPTIONS_FILE")
        CLAP_DISCOVERY=$(jq -r '.clap_discovery_threshold // 0.50' "$OPTIONS_FILE")
        CLAP_MARGIN=$(jq -r '.clap_confirm_margin // 0.20' "$OPTIONS_FILE")

        cat >> "$CONFIG_PATH" <<YAML

clap:
  enabled: true
  model: "${CLAP_MODEL}"
  confirm_threshold: ${CLAP_CONFIRM}
  suppress_threshold: ${CLAP_SUPPRESS}
  override_threshold: ${CLAP_OVERRIDE}
  discovery_threshold: ${CLAP_DISCOVERY}
  confirm_margin: ${CLAP_MARGIN}
YAML

        # Custom prompts (optional JSON object)
        CLAP_PROMPTS=$(jq -r '.clap_custom_prompts // empty' "$OPTIONS_FILE" 2>/dev/null)
        if [ -n "$CLAP_PROMPTS" ] && [ "$CLAP_PROMPTS" != "null" ]; then
            echo "  custom_prompts:" >> "$CONFIG_PATH"
            echo "$CLAP_PROMPTS" | jq -r 'to_entries[] | "    \(.key):\n" + (.value | map("      - \"\(.)\"") | join("\n"))' >> "$CONFIG_PATH"
        fi
    fi

    # LLM Judge (optional)
    LLM_JUDGE_ENABLED=$(jq -r '.llm_judge_enabled // false' "$OPTIONS_FILE")
    if [ "$LLM_JUDGE_ENABLED" = "true" ]; then
        LLM_JUDGE_API_BASE=$(jq -r '.llm_judge_api_base // ""' "$OPTIONS_FILE")
        # Export API key as env var — config.py substitutes ${LLM_JUDGE_API_KEY}
        export LLM_JUDGE_API_KEY
        LLM_JUDGE_API_KEY=$(jq -r '.llm_judge_api_key // ""' "$OPTIONS_FILE")
        LLM_JUDGE_MODEL=$(jq -r '.llm_judge_model // "gemini-3.1-pro-preview"' "$OPTIONS_FILE")
        LLM_JUDGE_SAMPLE_RATE=$(jq -r '.llm_judge_sample_rate // 0.10' "$OPTIONS_FILE")
        LLM_JUDGE_MAX_CLIPS=$(jq -r '.llm_judge_max_clips // 5000' "$OPTIONS_FILE")

        cat >> "$CONFIG_PATH" <<'YAML'

llm_judge:
  enabled: true
YAML
        cat >> "$CONFIG_PATH" <<YAML
  api_base: "${LLM_JUDGE_API_BASE}"
  api_key: "\${LLM_JUDGE_API_KEY}"
  model: "${LLM_JUDGE_MODEL}"
  sample_rate: ${LLM_JUDGE_SAMPLE_RATE}
  clip_dir: "/media/ast-audio-classifier/clips"
  max_clips: ${LLM_JUDGE_MAX_CLIPS}
YAML
    fi

    # Noise stress scorer (optional)
    NS_ENABLED=$(jq -r '.noise_stress_enabled // false' "$OPTIONS_FILE")
    if [ "$NS_ENABLED" = "true" ]; then
        NS_INTERVAL=$(jq -r '.noise_stress_update_interval // 30' "$OPTIONS_FILE")
        NS_HALF_LIFE=$(jq -r '.noise_stress_decay_half_life // 180.0' "$OPTIONS_FILE")
        NS_SATURATION=$(jq -r '.noise_stress_saturation // 25.0' "$OPTIONS_FILE")

        cat >> "$CONFIG_PATH" <<YAML

noise_stress:
  enabled: true
  update_interval_seconds: ${NS_INTERVAL}
  decay_half_life_seconds: ${NS_HALF_LIFE}
  saturation_constant: ${NS_SATURATION}
YAML

        # Indoor cameras list
        NS_INDOOR_COUNT=$(jq '.noise_stress_indoor_cameras | length' "$OPTIONS_FILE" 2>/dev/null || echo 0)
        if [ "$NS_INDOOR_COUNT" -gt 0 ] 2>/dev/null; then
            echo "  indoor_cameras:" >> "$CONFIG_PATH"
            for i in $(seq 0 $(( NS_INDOOR_COUNT - 1 ))); do
                NS_CAM=$(jq -r ".noise_stress_indoor_cameras[$i]" "$OPTIONS_FILE")
                echo "    - \"${NS_CAM}\"" >> "$CONFIG_PATH"
            done
        fi
    fi

    # Scrypted API URL (optional — enables direct RTSP URL resolution)
    if [ -n "$SCRYPTED_API_URL" ] && [ "$SCRYPTED_API_URL" != "null" ] && [ "$SCRYPTED_API_URL" != "" ]; then
        cat >> "$CONFIG_PATH" <<YAML

scrypted_api_url: "${SCRYPTED_API_URL}"
YAML
    fi

    # Per-group thresholds (hardcoded from LLM judge accuracy data)
    # car_horn=6%, gunshot_explosion=17%, music=29%, rain_storm=31%,
    # cough_sneeze=31%, vehicle=33%, hvac_mechanical=44%, cat_meow=50%, aircraft=51%
    cat >> "$CONFIG_PATH" <<'YAML'

groups:
  car_horn:
    enabled: false
  gunshot_explosion:
    enabled: false
  music:
    confidence_threshold: 0.60
  rain_storm:
    confidence_threshold: 0.55
  cough_sneeze:
    confidence_threshold: 0.55
  vehicle:
    confidence_threshold: 0.65
  hvac_mechanical:
    confidence_threshold: 0.45
  cat_meow:
    confidence_threshold: 0.40
  aircraft:
    confidence_threshold: 0.40
YAML

    # Weather entity for dynamic threshold adjustment (optional)
    WEATHER_ENTITY=$(jq -r '.weather_entity // ""' "$OPTIONS_FILE")
    if [ -n "$WEATHER_ENTITY" ] && [ "$WEATHER_ENTITY" != "null" ] && [ "$WEATHER_ENTITY" != "" ]; then
        echo "" >> "$CONFIG_PATH"
        echo "weather_entity: \"${WEATHER_ENTITY}\"" >> "$CONFIG_PATH"
    fi

    # Consolidated events (optional)
    CONSOLIDATED_ENABLED=$(jq -r '.consolidated_enabled // false' "$OPTIONS_FILE")
    CONSOLIDATED_WINDOW=$(jq -r '.consolidated_window_seconds // 5.0' "$OPTIONS_FILE")

    # Defaults
    cat >> "$CONFIG_PATH" <<YAML

defaults:
  confidence_threshold: ${CONFIDENCE}
  auto_off_seconds: ${AUTO_OFF}
  clip_duration_seconds: ${CLIP_DUR}
  consolidated_enabled: ${CONSOLIDATED_ENABLED}
  consolidated_window_seconds: ${CONSOLIDATED_WINDOW}
YAML

    echo "Generated config at $CONFIG_PATH with ${CAMERA_COUNT} cameras"

# -----------------------------------------------------------------------
# Standalone Docker mode: use provided config file
# -----------------------------------------------------------------------
else
    echo "Standalone mode — using config at $CONFIG_PATH"
    LOG_LEVEL="${LOG_LEVEL:-INFO}"
fi

VERSION=$(python -c "from src import __version__; print(__version__)" 2>/dev/null || echo "unknown")

echo "=========================================="
echo " AST Audio Classifier"
echo "=========================================="
echo "Version: $VERSION"
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
