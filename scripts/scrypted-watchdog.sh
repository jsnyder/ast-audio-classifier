#!/bin/bash
# Scrypted watchdog — restarts Scrypted LXC on sierra when cameras stop responding.
# Installed on echo.5745.house:/usr/local/bin/scrypted-watchdog.sh
# Cron: */10 * * * * root /usr/local/bin/scrypted-watchdog.sh
#
# Two trigger paths:
#   1. ALL cameras failing right now (fast path, confirmed with 30s retry)
#   2. 2+ cameras failing for N consecutive polls (~N*10 min). Single-camera
#      stuckness is usually a device/WiFi problem that a Scrypted reboot won't
#      fix — so we require at least two before restarting.
# A cooldown prevents restart loops.

set -u

SCRYPTED_HOST="192.168.0.107"
SCRYPTED_PORT="11080"
SCRYPTED_NODE="sierra"
SCRYPTED_VMID="10443"

# Cameras to probe. 37=basement_litter, 865=living_room, 932=back_porch, 99=front_door
CAMERAS=(37 865 932 99)

STATE_DIR="/var/lib/scrypted-watchdog"
COOLDOWN_SECONDS=1200         # 20 min between restarts
STUCK_POLL_THRESHOLD=3        # N consecutive failures per camera → stuck (~30 min at 10-min cron)

mkdir -p "${STATE_DIR}"
LAST_RESTART_FILE="${STATE_DIR}/last-restart"

probe_camera() {
    local device_id="$1"
    curl -sf --max-time 5 \
        "http://${SCRYPTED_HOST}:${SCRYPTED_PORT}/endpoint/scrypted-camera-api/public/snapshot/${device_id}" \
        -o /dev/null -w '%{http_code}' 2>/dev/null
}

in_cooldown() {
    [ -f "${LAST_RESTART_FILE}" ] || return 1
    local last now delta
    last=$(cat "${LAST_RESTART_FILE}" 2>/dev/null || echo 0)
    now=$(date +%s)
    delta=$((now - last))
    [ "${delta}" -lt "${COOLDOWN_SECONDS}" ]
}

record_restart() {
    date +%s > "${LAST_RESTART_FILE}"
}

restart_scrypted() {
    local reason="$1"
    if in_cooldown; then
        logger -t scrypted-watchdog "Restart requested (${reason}) but in cooldown; skipping"
        return
    fi
    logger -t scrypted-watchdog "${reason} — rebooting Scrypted LXC ${SCRYPTED_VMID}"
    if pvesh create "/nodes/${SCRYPTED_NODE}/lxc/${SCRYPTED_VMID}/status/reboot" >/dev/null 2>&1; then
        record_restart
    else
        logger -t scrypted-watchdog "pvesh reboot call failed"
    fi
}

# --- Path 1: all cameras failing right now, confirmed after 30s ----------------

all_failing_now() {
    local fails=0
    for id in "${CAMERAS[@]}"; do
        code=$(probe_camera "${id}")
        [ "${code}" != "200" ] && fails=$((fails + 1))
    done
    [ "${fails}" -eq "${#CAMERAS[@]}" ]
}

if all_failing_now; then
    sleep 30
    if all_failing_now; then
        restart_scrypted "All ${#CAMERAS[@]} cameras failing (confirmed)"
        exit 0
    fi
fi

# --- Path 2: 2+ cameras stuck across N consecutive polls ----------------------

# First pass: update each camera's consecutive-failure counter.
for id in "${CAMERAS[@]}"; do
    count_file="${STATE_DIR}/cam-${id}.count"
    code=$(probe_camera "${id}")
    if [ "${code}" = "200" ]; then
        : > "${count_file}"
    else
        prev=$(cat "${count_file}" 2>/dev/null)
        prev=${prev:-0}
        echo "$((prev + 1))" > "${count_file}"
    fi
done

# Second pass: collect cameras currently over threshold and fire only if 2+.
stuck_summary=""
stuck_count=0
for id in "${CAMERAS[@]}"; do
    count=$(cat "${STATE_DIR}/cam-${id}.count" 2>/dev/null)
    count=${count:-0}
    if [ "${count}" -ge "${STUCK_POLL_THRESHOLD}" ]; then
        stuck_count=$((stuck_count + 1))
        stuck_summary="${stuck_summary}${id}(${count}) "
    fi
done

if [ "${stuck_count}" -ge 2 ]; then
    restart_scrypted "${stuck_count} cameras stuck: ${stuck_summary}"
    for c in "${CAMERAS[@]}"; do
        : > "${STATE_DIR}/cam-${c}.count"
    done
fi
