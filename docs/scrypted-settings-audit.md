# Scrypted Settings Audit

**Date**: 2026-04-05
**Version**: Scrypted v0.143.0
**Host**: LXC 10443 on Proxmox `sierra` (192.168.0.107)

## Installed Plugins

| Plugin | Package | Version | Status | Notes |
|--------|---------|---------|--------|-------|
| Advanced Notifier | @apocaliss92/scrypted-advanced-notifier | v5.0.20 | KEEP | LLMVision triggers. Log level set to DEBUG (should lower). |
| ~~Active Streams Info~~ | @apocaliss92/scrypted-active-streams-info | - | **REMOVED** | Broken MQTT URL (`undefined`), spamming unhandled rejections every 5s. Potential cause of daily crashes. |
| Amcrest Plugin | @scrypted/amcrest | v0.0.168 | KEEP | Back Porch Litter (IP4M-1041W) |
| Arlo Camera Plugin | @scrypted/arlo | v0.11.54 | KEEP | Cloud Arlo cameras |
| Arlo Local Device Plugin | @scrypted/arlo-local | v0.6.0 | KEEP | Local Arlo access |
| Camera API | scrypted-camera-api | v0.2.0 | KEEP | **Critical** — AST classifier uses this for dynamic stream URL discovery |
| cosmotop | @bjia56/scrypted-cosmotop | v0.1.34 | REVIEW | System monitoring — low priority, could remove |
| FFmpeg Camera Plugin | @scrypted/ffmpeg-camera | v0.0.23 | KEEP | |
| GStreamer Camera Plugin | @scrypted/gstreamer-camera | v0.0.5 | REVIEW | Unused? |
| Home Assistant | @scrypted/homeassistant | v0.1.16 | KEEP | HA integration |
| HomeKit | @scrypted/homekit | v1.2.65 | KEEP | Used — 154 RPC objects |
| Large Language Model Plugin | @blueharford/scrypted-llm | v0.0.89 | KEEP | LLM vision integration |
| MQTT | @scrypted/mqtt | v0.0.87 | KEEP | Used by AN |
| ONVIF Camera Plugin | @scrypted/onvif | v0.1.31 | KEEP | Living Room, Back Porch Tapo |
| Rebroadcast Plugin | @scrypted/prebuffer-mixin | v0.10.66 | KEEP | **Critical** — prebuffer/rebroadcast. 237 RPC objects. |
| Reolink Camera Plugin | @scrypted/reolink | v0.0.111 | REVIEW | Any Reolink cameras? |
| RTSP Camera Plugin | @scrypted/rtsp | v0.0.55 | KEEP | |
| Scrypted NVR | @scrypted/nvr | v0.12.61 | KEEP | NVR recording — 283 RPC objects |
| Snapshot Plugin | @scrypted/snapshot | v0.2.68 | KEEP | For AN/LLMVision |
| Tapo Camera Plugin | @scrypted/tapo | v0.0.22 | KEEP | Tapo cameras |
| Video Analysis Plugin | @scrypted/objectdetector | v0.1.77 | KEEP | Object detection for AN |
| WebRTC Plugin | @scrypted/webrtc | v0.2.88 | KEEP | UI live view |
| X11 Virtual Camera Plugin | @scrypted/x11-camera | v0.0.4 | REVIEW | 143 RPC objects — needed? |

## Per-Camera Extension Settings (after cleanup)

Extensions disabled on all applicable cameras on 2026-04-05:

| Extension | Status | Reason |
|-----------|--------|--------|
| Rebroadcast Plugin | **ENABLED** | Core — provides RTSP rebroadcast for AST classifier (via Camera API) |
| WebRTC Plugin | **ENABLED** | UI live view |
| Snapshot Plugin | **ENABLED** | AN/LLMVision needs snapshots |
| Adaptive Streaming | **ENABLED** | Core streaming infrastructure |
| HomeKit | **ENABLED** | User uses HomeKit |
| Scrypted NVR | **ENABLED** | Recording |
| Scrypted NVR Object Detection | **ENABLED** | Feeds AN/LLMVision |
| Advanced Notifier | **ENABLED** | LLMVision triggers |
| MQTT | **ENABLED** | AN uses MQTT |
| Tapo Two Way Audio | **DISABLED** | Unnecessary, may cause stream instability |
| FFmpeg Audio Detection | **DISABLED** | AST Audio Classifier replaces this entirely |
| Accelerated Motion Detection | DISABLED | Was already off |
| Add to Launcher | DISABLED | Was already off |
| ONVIF PTZ | DISABLED | Was already off |
| OpenCV Motion Detection | DISABLED | Was already off |

## Cameras

| ID | Name | Model | Manufacturer | IP | Plugin | Scrypted Device ID |
|----|------|-------|-------------|-----|--------|-------------------|
| 865 | Living Room | ONVIF | tp-link | 10.57.3.94 | ONVIF Camera | 865 |
| 932 | Back Porch | Tapo C110 | tp-link | 10.57.3.97 | ONVIF Camera | 932 |
| 92 | Backyard (Local) | VMC4040P | Arlo | - | Arlo Local Device | 92 |
| 99 | Front Door | ONVIF | - | - | ONVIF Camera | 99 |
| 91 | Alley (Local) | VMC4040P | Arlo | - | Arlo Local Device | 91 |
| 37 | Basement Litter | IP4M-1041W | Amcrest | 192.168.6.101 | Amcrest | 37 |
| 131 | Back Porch Litter | IP4M-1041W | Amcrest | 192.168.6.101 | Amcrest | 131 |
| 148 | ? | VMC4040P | Arlo | - | Arlo Camera | 148 |
| 117 | ? | - | - | - | - | 117 |
| 93 | ? | - | Arlo | - | Arlo Camera | 93 |

## NVR Storage

| Mount | Backend | Size | Used | Status |
|-------|---------|------|------|--------|
| `/mnt/nvr` | nvr_5t (local ZFS on sierra) | 4.4TB | 88% | Working — old recordings (stale since Oct 2024) |
| `/mnt/nvr_fast` | local-zfs | 300GB | 86% | Working |
| `/mnt/nvr_fast_sata` | vidpool (local ZFS) | 900GB | 85% | Working |
| `/mnt/nvr_pool` | NFS from pool.5745.house | 10TB | 88% | **FIXED** — was unmounted, remounted + added `_netdev` to fstab |

## Issues Found and Fixed

### Fixed
1. **`/mnt/nvr_pool` NFS not mounted** — NFS from `pool.5745.house:/mnt/hpool/nvr/scrypted` was not mounted after reboot. Remounted and added `_netdev` to `/etc/fstab` to survive reboots.
2. **Active Streams Info plugin removed** — Was spamming `TypeError: Invalid URL` with `undefined` MQTT URL every 5 seconds. Unhandled rejections may have caused daily Scrypted crashes.
3. **FFmpeg Audio Detection disabled on all cameras** — Redundant with AST Audio Classifier.
4. **Tapo Two Way Audio disabled on all cameras** — Unnecessary, potential stability impact.
5. **AST classifier updated to use Camera API directly** — Eliminates go2rtc intermediary. Scrypted assigns new random ports on restart; Camera API provides fresh URLs dynamically.

### Known / Outstanding
1. **Scrypted `ERR_SOCKET_DGRAM_NOT_RUNNING`** — Rebroadcast UDP sockets die and don't recover. Root cause of EOF streams. Full Scrypted restart required to fix.
2. **Scrypted daily crashes** — 1-2x/day since March 28. `StandardOutput=null` in systemd hides crash details. Should change to `StandardOutput=journal`.
3. **Ephemeral rebroadcast ports** — Scrypted assigns new random ports/tokens on every restart. Previously caused go2rtc config staleness. Resolved by switching to Camera API direct resolution; go2rtc code fully removed in v0.8.6.
4. **Living Room prebuffer 60000s restart delay** — Seen in logs but may be a Scrypted log formatting issue (could be 60000ms = 60s displayed oddly).
5. **OpenObserve alerts needed** — Scrypted logs go to syslog at `192.168.1.63:515` (tag: `scrypted-app`). Should set up alerts for: stream EOF patterns, NVR recording errors, crash/restart events.

## Architecture Decision: Camera API vs go2rtc

**Previous**: AST Classifier → go2rtc (hardcoded RTSP URLs) → Scrypted Rebroadcast → Camera
**Current (v0.8.6)**: AST Classifier → Scrypted Rebroadcast (URLs from Camera API) → Camera

go2rtc was added as an intermediary to provide stable URLs, but Scrypted's ephemeral ports made go2rtc's config go stale on every restart. The Camera API (`scrypted-camera-api` plugin) provides fresh rebroadcast URLs that point to the existing prebuffer sessions — no new sessions created, no competing connections.

All go2rtc resolution code (`ScryptedUrlResolver`, `RtspUrl`, `_attempt_discovery()`, `DISCOVERING` state, `auto_discovery` flag, `go2rtc_stream` config) was fully removed in v0.8.6. The Camera API is the sole resolution path.
