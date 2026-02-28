"""MQTT publishing with Home Assistant auto-discovery.

Publishes binary_sensor and sensor entities via MQTT discovery,
then updates state/attributes topics on classification events.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime

import paho.mqtt.client as mqtt

from . import __version__
from .classifier import ClassificationResult
from .config import AppConfig, CameraConfig
from .labels import LABEL_GROUPS

logger = logging.getLogger(__name__)

DISCOVERY_PREFIX = "homeassistant"
TOPIC_PREFIX = "ast_audio"


class MqttPublisher:
    """Manages MQTT connection, discovery, and state publishing."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._client = mqtt.Client(
            client_id="ast-audio-classifier",
            protocol=mqtt.MQTTv311,
        )
        if config.mqtt.username:
            self._client.username_pw_set(config.mqtt.username, config.mqtt.password)

        # LWT: mark all cameras offline on unexpected disconnect
        self._client.will_set(
            f"{TOPIC_PREFIX}/status",
            payload="offline",
            qos=1,
            retain=True,
        )
        self._connected = False
        self._connected_event = asyncio.Event()

    def connect(self) -> None:
        """Connect to MQTT broker."""
        self._loop = asyncio.get_running_loop()
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.connect(self._config.mqtt.host, self._config.mqtt.port)
        self._client.loop_start()

    def disconnect(self) -> None:
        """Gracefully disconnect."""
        # Publish offline status before disconnecting
        self._client.publish(f"{TOPIC_PREFIX}/status", "offline", qos=1, retain=True)
        for cam in self._config.cameras:
            self._client.publish(
                f"{TOPIC_PREFIX}/{cam.name}/availability",
                "offline",
                qos=1,
                retain=True,
            )
        self._client.loop_stop()
        self._client.disconnect()

    def _on_connect(
        self, client: mqtt.Client, userdata: object, flags: dict, rc: int
    ) -> None:
        if rc == 0:
            logger.info(
                "Connected to MQTT broker at %s:%s",
                self._config.mqtt.host,
                self._config.mqtt.port,
            )
            self._connected = True
            self._loop.call_soon_threadsafe(self._connected_event.set)
            self._publish_discovery()
            self._client.publish(f"{TOPIC_PREFIX}/status", "online", qos=1, retain=True)
        else:
            logger.error("MQTT connection failed with code %s", rc)

    def _on_disconnect(self, client: mqtt.Client, userdata: object, rc: int) -> None:
        self._connected = False
        self._loop.call_soon_threadsafe(self._connected_event.clear)
        if rc != 0:
            logger.warning("Unexpected MQTT disconnect (rc=%s), will reconnect", rc)

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def connected_event(self) -> asyncio.Event:
        return self._connected_event

    def _publish_discovery(self) -> None:
        """Publish HA MQTT discovery configs for all cameras and groups."""
        for cam in self._config.cameras:
            self._publish_camera_discovery(cam)
        logger.info(
            "Published MQTT discovery for %d cameras", len(self._config.cameras)
        )

    def _publish_camera_discovery(self, cam: CameraConfig) -> None:
        """Publish discovery payloads for one camera's entities."""
        avail_topic = f"{TOPIC_PREFIX}/{cam.name}/availability"

        # Binary sensors for each label group
        for group in LABEL_GROUPS:
            object_id = f"ast_{cam.name}_{group}"
            discovery_topic = f"{DISCOVERY_PREFIX}/binary_sensor/{object_id}/config"

            payload = {
                "name": f"AST {cam.name.replace('_', ' ').title()} {group.replace('_', ' ').title()}",
                "unique_id": object_id,
                "state_topic": f"{TOPIC_PREFIX}/{cam.name}/{group}/state",
                "json_attributes_topic": f"{TOPIC_PREFIX}/{cam.name}/{group}/attributes",
                "payload_on": "ON",
                "payload_off": "OFF",
                "off_delay": self._config.auto_off_seconds,
                "device_class": self._device_class_for_group(group),
                "availability_topic": avail_topic,
                "device": self._device_info(cam),
                "icon": self._icon_for_group(group),
            }
            self._client.publish(
                discovery_topic, json.dumps(payload), qos=1, retain=True
            )

        # Sensor: last audio event
        last_event_id = f"ast_{cam.name}_last_event"
        self._client.publish(
            f"{DISCOVERY_PREFIX}/sensor/{last_event_id}/config",
            json.dumps(
                {
                    "name": f"AST {cam.name.replace('_', ' ').title()} Last Audio Event",
                    "unique_id": last_event_id,
                    "state_topic": f"{TOPIC_PREFIX}/{cam.name}/last_event/state",
                    "json_attributes_topic": f"{TOPIC_PREFIX}/{cam.name}/last_event/attributes",
                    "availability_topic": avail_topic,
                    "device": self._device_info(cam),
                    "icon": "mdi:waveform",
                }
            ),
            qos=1,
            retain=True,
        )

        # Binary sensor: stream status
        status_id = f"ast_{cam.name}_status"
        self._client.publish(
            f"{DISCOVERY_PREFIX}/binary_sensor/{status_id}/config",
            json.dumps(
                {
                    "name": f"AST {cam.name.replace('_', ' ').title()} Status",
                    "unique_id": status_id,
                    "state_topic": f"{TOPIC_PREFIX}/{cam.name}/availability",
                    "payload_on": "online",
                    "payload_off": "offline",
                    "device_class": "connectivity",
                    "device": self._device_info(cam),
                    "icon": "mdi:connection",
                }
            ),
            qos=1,
            retain=True,
        )

    def _device_info(self, cam: CameraConfig) -> dict:
        return {
            "identifiers": [f"ast_audio_{cam.name}"],
            "name": f"AST Audio Classifier - {cam.name.replace('_', ' ').title()}",
            "manufacturer": "MIT",
            "model": "Audio Spectrogram Transformer",
            "sw_version": __version__,
        }

    @staticmethod
    def _device_class_for_group(group: str) -> str | None:
        mapping = {
            "smoke_alarm": "smoke",
            "glass_break": "safety",
            "water_leak": "moisture",
        }
        return mapping.get(group, "sound")

    @staticmethod
    def _icon_for_group(group: str) -> str:
        icons = {
            # Safety & security
            "smoke_alarm": "mdi:smoke-detector-variant",
            "glass_break": "mdi:glass-fragile",
            "siren": "mdi:alarm-light",
            "gunshot_explosion": "mdi:pistol",
            "screaming": "mdi:account-alert",
            # People & pets
            "dog_bark": "mdi:dog",
            "cat_meow": "mdi:cat",
            "crying": "mdi:emoticon-cry-outline",
            "speech": "mdi:account-voice",
            "cough_sneeze": "mdi:account-alert-outline",
            "footsteps": "mdi:shoe-print",
            # Doors & entry
            "doorbell": "mdi:doorbell",
            "knock": "mdi:door-closed",
            "door": "mdi:door",
            "cabinet": "mdi:cupboard-outline",
            # Environment
            "rain_storm": "mdi:weather-pouring",
            "music": "mdi:music",
            "vehicle": "mdi:car",
            "car_horn": "mdi:bugle",
            "aircraft": "mdi:airplane",
            # Household
            "vacuum_cleaner": "mdi:robot-vacuum",
            "water_running": "mdi:water",
            "kitchen_appliance": "mdi:stove",
            "power_tools": "mdi:saw-blade",
            "alarm_beep": "mdi:alarm-note",
            # Equipment monitoring
            "hvac_mechanical": "mdi:hvac",
            "mechanical_anomaly": "mdi:alert-circle-outline",
            "water_leak": "mdi:water-alert",
            "electrical_anomaly": "mdi:flash-alert",
        }
        return icons.get(group, "mdi:waveform")

    def publish_detection(self, camera_name: str, result: ClassificationResult) -> None:
        """Publish a detection event for a camera."""
        group = result.group
        now = datetime.now(UTC).isoformat()

        # Binary sensor ON
        self._client.publish(
            f"{TOPIC_PREFIX}/{camera_name}/{group}/state",
            "ON",
            qos=1,
        )

        # Attributes
        attrs = {
            "confidence": result.confidence,
            "raw_label": result.label,
            "db_level": result.db_level,
            "timestamp": now,
        }
        # Include CLAP attributes when available
        if result.clap_verified is not None:
            attrs["clap_verified"] = result.clap_verified
        if result.clap_score is not None:
            attrs["clap_score"] = result.clap_score
        if result.clap_label is not None:
            attrs["clap_label"] = result.clap_label
        if result.source != "ast":
            attrs["source"] = result.source
        self._client.publish(
            f"{TOPIC_PREFIX}/{camera_name}/{group}/attributes",
            json.dumps(attrs),
            qos=1,
        )

        # Last event sensor
        self._client.publish(
            f"{TOPIC_PREFIX}/{camera_name}/last_event/state",
            result.label,
            qos=1,
        )
        last_event_attrs = {
            "group": result.group,
            "confidence": result.confidence,
            "db_level": result.db_level,
            "top_5": result.top_5,
            "timestamp": now,
        }
        if result.clap_verified is not None:
            last_event_attrs["clap_verified"] = result.clap_verified
        if result.clap_score is not None:
            last_event_attrs["clap_score"] = result.clap_score
        if result.clap_label is not None:
            last_event_attrs["clap_label"] = result.clap_label
        if result.source != "ast":
            last_event_attrs["source"] = result.source
        self._client.publish(
            f"{TOPIC_PREFIX}/{camera_name}/last_event/attributes",
            json.dumps(last_event_attrs),
            qos=1,
        )

    def publish_camera_online(self, camera_name: str) -> None:
        """Mark a camera stream as online."""
        self._client.publish(
            f"{TOPIC_PREFIX}/{camera_name}/availability",
            "online",
            qos=1,
            retain=True,
        )

    def publish_camera_offline(self, camera_name: str) -> None:
        """Mark a camera stream as offline."""
        self._client.publish(
            f"{TOPIC_PREFIX}/{camera_name}/availability",
            "offline",
            qos=1,
            retain=True,
        )

    def publish_consolidated_discovery(self, auto_off_seconds: int) -> None:
        """Publish HA MQTT discovery configs for consolidated binary sensors."""
        for group in LABEL_GROUPS:
            object_id = f"ast_consolidated_{group}"
            discovery_topic = f"{DISCOVERY_PREFIX}/binary_sensor/{object_id}/config"

            payload = {
                "name": f"AST Consolidated {group.replace('_', ' ').title()}",
                "unique_id": object_id,
                "state_topic": f"{TOPIC_PREFIX}/consolidated/{group}/state",
                "json_attributes_topic": f"{TOPIC_PREFIX}/consolidated/{group}/attributes",
                "payload_on": "ON",
                "payload_off": "OFF",
                "off_delay": auto_off_seconds,
                "device_class": self._device_class_for_group(group),
                "device": {
                    "identifiers": ["ast_audio_consolidated"],
                    "name": "AST Audio Classifier - Consolidated",
                    "manufacturer": "MIT",
                    "model": "Audio Spectrogram Transformer",
                    "sw_version": __version__,
                },
                "icon": self._icon_for_group(group),
            }
            self._client.publish(
                discovery_topic, json.dumps(payload), qos=1, retain=True
            )
        logger.info("Published consolidated MQTT discovery for %d groups", len(LABEL_GROUPS))

    def publish_noise_stress_discovery(self) -> None:
        """Publish HA MQTT discovery config for the noise stress sensor."""
        object_id = "ast_noise_stress_score"
        discovery_topic = f"{DISCOVERY_PREFIX}/sensor/{object_id}/config"

        payload = {
            "name": "AST Noise Stress Score",
            "unique_id": object_id,
            "state_topic": f"{TOPIC_PREFIX}/noise_stress/state",
            "json_attributes_topic": f"{TOPIC_PREFIX}/noise_stress/attributes",
            "unit_of_measurement": "",
            "state_class": "measurement",
            "availability_topic": f"{TOPIC_PREFIX}/status",
            "payload_available": "online",
            "payload_not_available": "offline",
            "device": {
                "identifiers": ["ast_audio_noise_stress"],
                "name": "AST Audio Classifier - Noise Stress",
                "manufacturer": "MIT",
                "model": "Audio Spectrogram Transformer",
                "sw_version": __version__,
            },
            "icon": "mdi:head-alert",
        }
        self._client.publish(
            discovery_topic, json.dumps(payload), qos=1, retain=True
        )
        logger.info("Published noise stress MQTT discovery")

    def publish_noise_stress_score(self, score_data: dict) -> None:
        """Publish a noise stress score update."""
        self._client.publish(
            f"{TOPIC_PREFIX}/noise_stress/state",
            str(score_data["score"]),
            qos=1,
        )
        attrs = {
            "ambient_component": score_data["ambient_component"],
            "event_component": score_data["event_component"],
            "sustained_component": score_data["sustained_component"],
            "recent_event_count": score_data["recent_event_count"],
            "top_stressor": score_data.get("top_stressor"),
            "dominant_camera": score_data.get("dominant_camera"),
            "active_high_stress": score_data.get("active_high_stress", False),
            "daily_avg": score_data.get("daily_avg", 0.0),
            "daily_min": score_data.get("daily_min", 0.0),
            "daily_max": score_data.get("daily_max", 0.0),
            "daily_samples": score_data.get("daily_samples", 0),
        }
        self._client.publish(
            f"{TOPIC_PREFIX}/noise_stress/attributes",
            json.dumps(attrs),
            qos=1,
        )

    def publish_consolidated_event(
        self,
        group: str,
        cameras: list[str],
        max_confidence: float,
        detection_count: int,
        duration_seconds: float,
        first_detected: str,
        last_detected: str,
    ) -> None:
        """Publish a consolidated detection event."""
        self._client.publish(
            f"{TOPIC_PREFIX}/consolidated/{group}/state",
            "ON",
            qos=1,
        )
        attrs = {
            "cameras": cameras,
            "max_confidence": max_confidence,
            "detection_count": detection_count,
            "duration_seconds": round(duration_seconds, 2),
            "first_detected": first_detected,
            "last_detected": last_detected,
        }
        self._client.publish(
            f"{TOPIC_PREFIX}/consolidated/{group}/attributes",
            json.dumps(attrs),
            qos=1,
        )
