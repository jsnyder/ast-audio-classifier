"""Tests for the MQTT publisher module.

Mocks paho.mqtt.client entirely since it is not available in the test venv.
"""

import json
import sys
from types import ModuleType
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Stub out paho.mqtt.client before importing the module under test.
# The real paho-mqtt is only installed in the runtime container.
# ---------------------------------------------------------------------------
_mqtt_mod = ModuleType("paho")
_mqtt_client_mod = ModuleType("paho.mqtt")
_mqtt_client_client_mod = ModuleType("paho.mqtt.client")
_mqtt_client_client_mod.MQTTv311 = 4  # type: ignore[attr-defined]
_mqtt_client_client_mod.Client = MagicMock  # type: ignore[attr-defined]

sys.modules.setdefault("paho", _mqtt_mod)
sys.modules.setdefault("paho.mqtt", _mqtt_client_mod)
sys.modules.setdefault("paho.mqtt.client", _mqtt_client_client_mod)

from src import __version__  # noqa: E402
from src.classifier import ClassificationResult  # noqa: E402
from src.config import AppConfig, CameraConfig, MqttConfig  # noqa: E402
from src.labels import LABEL_GROUPS  # noqa: E402
from src.mqtt_publisher import (  # noqa: E402
    DISCOVERY_PREFIX,
    TOPIC_PREFIX,
    MqttPublisher,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    cameras: list[CameraConfig] | None = None,
    auto_off_seconds: int = 30,
    mqtt_username: str | None = None,
    mqtt_password: str | None = None,
) -> AppConfig:
    """Build a minimal AppConfig for testing."""
    return AppConfig(
        mqtt=MqttConfig(
            host="localhost",
            port=1883,
            username=mqtt_username,
            password=mqtt_password,
        ),
        cameras=cameras
        or [
            CameraConfig(name="front_door", rtsp_url="rtsp://192.168.1.10/stream"),
        ],
        auto_off_seconds=auto_off_seconds,
    )


def _make_result(
    label: str = "Dog",
    group: str = "dog_bark",
    confidence: float = 0.85,
    db_level: float = -25.3,
    top_5: list | None = None,
    clap_verified: bool | None = None,
    clap_score: float | None = None,
    clap_label: str | None = None,
    source: str = "ast",
) -> ClassificationResult:
    return ClassificationResult(
        label=label,
        group=group,
        confidence=confidence,
        top_5=top_5 or [("Dog", 0.85), ("Bark", 0.72)],
        db_level=db_level,
        clap_verified=clap_verified,
        clap_score=clap_score,
        clap_label=clap_label,
        source=source,
    )


def _published_payloads(mock_client: MagicMock) -> list[tuple[str, str, dict]]:
    """Extract (topic, payload_str, kwargs) from all publish calls."""
    results = []
    for c in mock_client.publish.call_args_list:
        args, kwargs = c
        topic = args[0]
        payload = args[1] if len(args) > 1 else kwargs.get("payload", "")
        results.append((topic, payload, kwargs))
    return results


def _find_publish(mock_client: MagicMock, topic_substring: str) -> list[tuple[str, str]]:
    """Return (topic, payload) pairs where topic contains the given substring."""
    matches = []
    for c in mock_client.publish.call_args_list:
        args, kwargs = c
        topic = args[0]
        payload = args[1] if len(args) > 1 else kwargs.get("payload", "")
        if topic_substring in topic:
            matches.append((topic, payload))
    return matches


# ---------------------------------------------------------------------------
# Tests: Constructor & LWT
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_lwt_is_set_on_client(self):
        config = _make_config()
        pub = MqttPublisher(config)
        pub._client.will_set.assert_called_once_with(
            f"{TOPIC_PREFIX}/status",
            payload="offline",
            qos=1,
            retain=True,
        )

    def test_username_password_set_when_provided(self):
        config = _make_config(mqtt_username="user", mqtt_password="secret")
        pub = MqttPublisher(config)
        pub._client.username_pw_set.assert_called_once_with("user", "secret")

    def test_username_password_not_set_when_absent(self):
        config = _make_config()
        pub = MqttPublisher(config)
        pub._client.username_pw_set.assert_not_called()

    def test_initial_state_not_connected(self):
        config = _make_config()
        pub = MqttPublisher(config)
        assert pub.connected is False


# ---------------------------------------------------------------------------
# Tests: Connect / Disconnect lifecycle
# ---------------------------------------------------------------------------


class TestConnectDisconnect:
    def test_connect_sets_callbacks_and_starts_loop(self):
        import asyncio

        async def _run():
            config = _make_config()
            pub = MqttPublisher(config)
            pub.connect()

            pub._client.connect.assert_called_once_with("localhost", 1883)
            pub._client.loop_start.assert_called_once()
            assert pub._client.on_connect is not None
            assert pub._client.on_disconnect is not None

        asyncio.run(_run())

    def test_disconnect_publishes_offline_and_stops_loop(self):
        config = _make_config(
            cameras=[
                CameraConfig(name="cam_a", rtsp_url="rtsp://h/a"),
                CameraConfig(name="cam_b", rtsp_url="rtsp://h/b"),
            ]
        )
        pub = MqttPublisher(config)
        pub._client.reset_mock()
        pub.disconnect()

        # Should publish global offline
        calls = pub._client.publish.call_args_list
        topics = [c.args[0] for c in calls]
        assert f"{TOPIC_PREFIX}/status" in topics
        # Should publish per-camera offline
        assert f"{TOPIC_PREFIX}/cam_a/availability" in topics
        assert f"{TOPIC_PREFIX}/cam_b/availability" in topics

        pub._client.loop_stop.assert_called_once()
        pub._client.disconnect.assert_called_once()

    def test_disconnect_camera_availability_payloads_are_offline(self):
        config = _make_config()
        pub = MqttPublisher(config)
        pub._client.reset_mock()
        pub.disconnect()

        cam_avail_calls = [
            c for c in pub._client.publish.call_args_list if "availability" in c.args[0]
        ]
        for c in cam_avail_calls:
            assert c.args[1] == "offline"
            assert c.kwargs.get("qos", c[1].get("qos", None)) == 1 or c[1]["qos"] == 1


# ---------------------------------------------------------------------------
# Tests: Discovery payloads
# ---------------------------------------------------------------------------


class TestCameraDiscovery:
    def _publish_discovery_for_cam(self, cam_name: str = "front_door"):
        cam = CameraConfig(name=cam_name, rtsp_url="rtsp://h/s")
        config = _make_config(cameras=[cam], auto_off_seconds=45)
        pub = MqttPublisher(config)
        pub._client.reset_mock()
        pub._publish_camera_discovery(cam)
        return pub

    def test_binary_sensor_topics_for_all_groups(self):
        pub = self._publish_discovery_for_cam()
        published_topics = [c.args[0] for c in pub._client.publish.call_args_list]
        for group in LABEL_GROUPS:
            expected = f"{DISCOVERY_PREFIX}/binary_sensor/ast_front_door_{group}/config"
            assert expected in published_topics, f"Missing discovery topic for group {group}"

    def test_last_event_sensor_discovery(self):
        pub = self._publish_discovery_for_cam()
        published = _find_publish(pub._client, "sensor/ast_front_door_last_event/config")
        assert len(published) == 1
        _topic, raw_payload = published[0]
        payload = json.loads(raw_payload)
        assert payload["unique_id"] == "ast_front_door_last_event"
        assert payload["name"] == "AST Front Door Last Audio Event"
        assert payload["icon"] == "mdi:waveform"
        assert payload["state_topic"] == f"{TOPIC_PREFIX}/front_door/last_event/state"
        assert (
            payload["json_attributes_topic"] == f"{TOPIC_PREFIX}/front_door/last_event/attributes"
        )
        assert payload["availability_topic"] == f"{TOPIC_PREFIX}/front_door/availability"

    def test_status_binary_sensor_discovery(self):
        pub = self._publish_discovery_for_cam()
        published = _find_publish(pub._client, "binary_sensor/ast_front_door_status/config")
        assert len(published) == 1
        payload = json.loads(published[0][1])
        assert payload["unique_id"] == "ast_front_door_status"
        assert payload["device_class"] == "connectivity"
        assert payload["payload_on"] == "online"
        assert payload["payload_off"] == "offline"
        assert payload["icon"] == "mdi:connection"

    def test_binary_sensor_payload_fields(self):
        """Verify detailed fields of a binary_sensor discovery payload."""
        pub = self._publish_discovery_for_cam()
        # Pick a specific group to check
        published = _find_publish(pub._client, "binary_sensor/ast_front_door_dog_bark/config")
        assert len(published) == 1
        payload = json.loads(published[0][1])
        assert payload["unique_id"] == "ast_front_door_dog_bark"
        assert payload["name"] == "AST Front Door Dog Bark"
        assert payload["state_topic"] == f"{TOPIC_PREFIX}/front_door/dog_bark/state"
        assert payload["json_attributes_topic"] == f"{TOPIC_PREFIX}/front_door/dog_bark/attributes"
        assert payload["payload_on"] == "ON"
        assert payload["payload_off"] == "OFF"
        assert payload["off_delay"] == 45
        assert payload["device_class"] == "sound"
        assert payload["availability_topic"] == f"{TOPIC_PREFIX}/front_door/availability"
        assert payload["icon"] == "mdi:dog"

    def test_device_info_in_discovery(self):
        pub = self._publish_discovery_for_cam()
        published = _find_publish(pub._client, "binary_sensor/ast_front_door_dog_bark/config")
        payload = json.loads(published[0][1])
        device = payload["device"]
        assert device["identifiers"] == ["ast_audio_front_door"]
        assert device["name"] == "AST Audio Classifier - Front Door"
        assert device["manufacturer"] == "MIT"
        assert device["model"] == "Audio Spectrogram Transformer"
        assert device["sw_version"] == __version__

    def test_smoke_alarm_device_class(self):
        pub = self._publish_discovery_for_cam()
        published = _find_publish(pub._client, "binary_sensor/ast_front_door_smoke_alarm/config")
        payload = json.loads(published[0][1])
        assert payload["device_class"] == "smoke"

    def test_glass_break_device_class(self):
        pub = self._publish_discovery_for_cam()
        published = _find_publish(pub._client, "binary_sensor/ast_front_door_glass_break/config")
        payload = json.loads(published[0][1])
        assert payload["device_class"] == "safety"

    def test_water_leak_device_class(self):
        pub = self._publish_discovery_for_cam()
        published = _find_publish(pub._client, "binary_sensor/ast_front_door_water_leak/config")
        payload = json.loads(published[0][1])
        assert payload["device_class"] == "moisture"

    def test_all_discovery_publishes_use_qos1_retained(self):
        pub = self._publish_discovery_for_cam()
        for c in pub._client.publish.call_args_list:
            _, kwargs = c
            assert kwargs.get("qos") == 1, f"Expected qos=1 for {c.args[0]}"
            assert kwargs.get("retain") is True, f"Expected retain=True for {c.args[0]}"

    def test_total_discovery_messages_per_camera(self):
        """Each camera should produce len(LABEL_GROUPS) binary sensors + 1 sensor + 1 status."""
        pub = self._publish_discovery_for_cam()
        expected_count = len(LABEL_GROUPS) + 2  # +1 last_event sensor, +1 status binary_sensor
        assert pub._client.publish.call_count == expected_count


# ---------------------------------------------------------------------------
# Tests: Device class & icon mapping
# ---------------------------------------------------------------------------


class TestDeviceClassMapping:
    def test_smoke_alarm(self):
        assert MqttPublisher._device_class_for_group("smoke_alarm") == "smoke"

    def test_glass_break(self):
        assert MqttPublisher._device_class_for_group("glass_break") == "safety"

    def test_water_leak(self):
        assert MqttPublisher._device_class_for_group("water_leak") == "moisture"

    def test_default_is_sound(self):
        assert MqttPublisher._device_class_for_group("dog_bark") == "sound"

    def test_unknown_group_default(self):
        assert MqttPublisher._device_class_for_group("nonexistent_group") == "sound"


class TestIconMapping:
    def test_known_icons(self):
        known = {
            "smoke_alarm": "mdi:smoke-detector-variant",
            "glass_break": "mdi:glass-fragile",
            "dog_bark": "mdi:dog",
            "cat_meow": "mdi:cat",
            "doorbell": "mdi:doorbell",
            "rain_storm": "mdi:weather-pouring",
            "vacuum_cleaner": "mdi:robot-vacuum",
            "hvac_mechanical": "mdi:hvac",
            "water_leak": "mdi:water-alert",
            "electrical_anomaly": "mdi:flash-alert",
            "siren": "mdi:alarm-light",
            "speech": "mdi:account-voice",
            "crying": "mdi:emoticon-cry-outline",
            "knock": "mdi:door-closed",
            "music": "mdi:music",
            "vehicle": "mdi:car",
        }
        for group, expected_icon in known.items():
            assert MqttPublisher._icon_for_group(group) == expected_icon

    def test_unknown_group_returns_default_icon(self):
        assert MqttPublisher._icon_for_group("nonexistent") == "mdi:waveform"


# ---------------------------------------------------------------------------
# Tests: publish_detection
# ---------------------------------------------------------------------------


class TestPublishDetection:
    def test_publishes_state_on(self):
        config = _make_config()
        pub = MqttPublisher(config)
        pub._client.reset_mock()

        result = _make_result()
        pub.publish_detection("front_door", result)

        state_calls = _find_publish(pub._client, "front_door/dog_bark/state")
        assert len(state_calls) == 1
        assert state_calls[0][1] == "ON"

    def test_publishes_attributes_json(self):
        config = _make_config()
        pub = MqttPublisher(config)
        pub._client.reset_mock()

        result = _make_result(confidence=0.92, db_level=-28.5)
        pub.publish_detection("front_door", result)

        attr_calls = _find_publish(pub._client, "front_door/dog_bark/attributes")
        assert len(attr_calls) == 1
        attrs = json.loads(attr_calls[0][1])
        assert attrs["confidence"] == 0.92
        assert attrs["raw_label"] == "Dog"
        assert attrs["db_level"] == -28.5
        assert "timestamp" in attrs

    def test_publishes_last_event_state(self):
        config = _make_config()
        pub = MqttPublisher(config)
        pub._client.reset_mock()

        result = _make_result(label="Bark", group="dog_bark")
        pub.publish_detection("front_door", result)

        last_event_calls = _find_publish(pub._client, "front_door/last_event/state")
        assert len(last_event_calls) == 1
        assert last_event_calls[0][1] == "Bark"

    def test_publishes_last_event_attributes(self):
        config = _make_config()
        pub = MqttPublisher(config)
        pub._client.reset_mock()

        result = _make_result(
            label="Bark",
            group="dog_bark",
            confidence=0.88,
            db_level=-30.0,
            top_5=[("Bark", 0.88), ("Dog", 0.75), ("Howl", 0.10)],
        )
        pub.publish_detection("front_door", result)

        attr_calls = _find_publish(pub._client, "front_door/last_event/attributes")
        assert len(attr_calls) == 1
        attrs = json.loads(attr_calls[0][1])
        assert attrs["group"] == "dog_bark"
        assert attrs["confidence"] == 0.88
        assert attrs["db_level"] == -30.0
        assert attrs["top_5"] == [["Bark", 0.88], ["Dog", 0.75], ["Howl", 0.10]]
        assert "timestamp" in attrs

    def test_four_publish_calls_per_detection(self):
        """Each detection should produce 4 publishes: state, attrs, last_event_state, last_event_attrs."""
        config = _make_config()
        pub = MqttPublisher(config)
        pub._client.reset_mock()

        pub.publish_detection("front_door", _make_result())
        assert pub._client.publish.call_count == 4

    def test_detection_uses_qos1(self):
        config = _make_config()
        pub = MqttPublisher(config)
        pub._client.reset_mock()

        pub.publish_detection("front_door", _make_result())
        for c in pub._client.publish.call_args_list:
            assert c.kwargs.get("qos") == 1


class TestPublishDetectionClapFields:
    def test_clap_fields_included_when_present(self):
        config = _make_config()
        pub = MqttPublisher(config)
        pub._client.reset_mock()

        result = _make_result(
            clap_verified=True,
            clap_score=0.72,
            clap_label="a dog barking loudly",
        )
        pub.publish_detection("front_door", result)

        # Check group attributes
        attr_calls = _find_publish(pub._client, "front_door/dog_bark/attributes")
        attrs = json.loads(attr_calls[0][1])
        assert attrs["clap_verified"] is True
        assert attrs["clap_score"] == 0.72
        assert attrs["clap_label"] == "a dog barking loudly"

        # Check last_event attributes
        last_attr_calls = _find_publish(pub._client, "front_door/last_event/attributes")
        last_attrs = json.loads(last_attr_calls[0][1])
        assert last_attrs["clap_verified"] is True
        assert last_attrs["clap_score"] == 0.72
        assert last_attrs["clap_label"] == "a dog barking loudly"

    def test_clap_fields_absent_when_none(self):
        config = _make_config()
        pub = MqttPublisher(config)
        pub._client.reset_mock()

        result = _make_result()  # No CLAP fields
        pub.publish_detection("front_door", result)

        attr_calls = _find_publish(pub._client, "front_door/dog_bark/attributes")
        attrs = json.loads(attr_calls[0][1])
        assert "clap_verified" not in attrs
        assert "clap_score" not in attrs
        assert "clap_label" not in attrs
        assert "source" not in attrs

    def test_source_included_when_not_ast(self):
        config = _make_config()
        pub = MqttPublisher(config)
        pub._client.reset_mock()

        result = _make_result(
            source="clap",
            clap_verified=True,
            clap_score=0.65,
            clap_label="a vacuum cleaner running",
        )
        pub.publish_detection("front_door", result)

        attr_calls = _find_publish(pub._client, "front_door/dog_bark/attributes")
        attrs = json.loads(attr_calls[0][1])
        assert attrs["source"] == "clap"

        last_attr_calls = _find_publish(pub._client, "front_door/last_event/attributes")
        last_attrs = json.loads(last_attr_calls[0][1])
        assert last_attrs["source"] == "clap"

    def test_source_omitted_when_ast(self):
        config = _make_config()
        pub = MqttPublisher(config)
        pub._client.reset_mock()

        result = _make_result(source="ast")
        pub.publish_detection("front_door", result)

        attr_calls = _find_publish(pub._client, "front_door/dog_bark/attributes")
        attrs = json.loads(attr_calls[0][1])
        assert "source" not in attrs

    def test_clap_verified_false(self):
        """clap_verified=False should still be included (it is not None)."""
        config = _make_config()
        pub = MqttPublisher(config)
        pub._client.reset_mock()

        result = _make_result(clap_verified=False, clap_score=0.08, clap_label="silence")
        pub.publish_detection("front_door", result)

        attr_calls = _find_publish(pub._client, "front_door/dog_bark/attributes")
        attrs = json.loads(attr_calls[0][1])
        assert attrs["clap_verified"] is False
        assert attrs["clap_score"] == 0.08


# ---------------------------------------------------------------------------
# Tests: Camera online / offline
# ---------------------------------------------------------------------------


class TestCameraOnlineOffline:
    def test_publish_camera_online(self):
        config = _make_config()
        pub = MqttPublisher(config)
        pub._client.reset_mock()

        pub.publish_camera_online("front_door")

        pub._client.publish.assert_called_once_with(
            f"{TOPIC_PREFIX}/front_door/availability",
            "online",
            qos=1,
            retain=True,
        )

    def test_publish_camera_offline(self):
        config = _make_config()
        pub = MqttPublisher(config)
        pub._client.reset_mock()

        pub.publish_camera_offline("front_door")

        pub._client.publish.assert_called_once_with(
            f"{TOPIC_PREFIX}/front_door/availability",
            "offline",
            qos=1,
            retain=True,
        )


# ---------------------------------------------------------------------------
# Tests: Consolidated discovery & event publishing
# ---------------------------------------------------------------------------


class TestConsolidatedDiscovery:
    def test_publishes_discovery_for_all_groups(self):
        config = _make_config()
        pub = MqttPublisher(config)
        pub._client.reset_mock()

        pub.publish_consolidated_discovery(auto_off_seconds=30)

        published_topics = [c.args[0] for c in pub._client.publish.call_args_list]
        for group in LABEL_GROUPS:
            expected = f"{DISCOVERY_PREFIX}/binary_sensor/ast_consolidated_{group}/config"
            assert expected in published_topics

    def test_consolidated_payload_fields(self):
        config = _make_config()
        pub = MqttPublisher(config)
        pub._client.reset_mock()

        pub.publish_consolidated_discovery(auto_off_seconds=20)

        published = _find_publish(pub._client, "ast_consolidated_dog_bark/config")
        assert len(published) == 1
        payload = json.loads(published[0][1])
        assert payload["unique_id"] == "ast_consolidated_dog_bark"
        assert payload["name"] == "AST Consolidated Dog Bark"
        assert payload["state_topic"] == f"{TOPIC_PREFIX}/consolidated/dog_bark/state"
        assert (
            payload["json_attributes_topic"] == f"{TOPIC_PREFIX}/consolidated/dog_bark/attributes"
        )
        assert payload["off_delay"] == 20
        assert payload["device_class"] == "sound"
        assert payload["icon"] == "mdi:dog"

    def test_consolidated_device_info(self):
        config = _make_config()
        pub = MqttPublisher(config)
        pub._client.reset_mock()

        pub.publish_consolidated_discovery(auto_off_seconds=30)

        published = _find_publish(pub._client, "ast_consolidated_smoke_alarm/config")
        payload = json.loads(published[0][1])
        device = payload["device"]
        assert device["identifiers"] == ["ast_audio_consolidated"]
        assert device["name"] == "AST Audio Classifier - Consolidated"
        assert device["sw_version"] == __version__

    def test_consolidated_discovery_count(self):
        config = _make_config()
        pub = MqttPublisher(config)
        pub._client.reset_mock()

        pub.publish_consolidated_discovery(auto_off_seconds=30)
        assert pub._client.publish.call_count == len(LABEL_GROUPS)


class TestPublishConsolidatedEvent:
    def test_publishes_state_on(self):
        config = _make_config()
        pub = MqttPublisher(config)
        pub._client.reset_mock()

        pub.publish_consolidated_event(
            group="dog_bark",
            cameras=["front_door", "backyard"],
            max_confidence=0.92,
            detection_count=3,
            duration_seconds=4.567,
            first_detected="2025-01-01T00:00:00",
            last_detected="2025-01-01T00:00:04",
        )

        state_calls = _find_publish(pub._client, "consolidated/dog_bark/state")
        assert len(state_calls) == 1
        assert state_calls[0][1] == "ON"

    def test_publishes_attributes(self):
        config = _make_config()
        pub = MqttPublisher(config)
        pub._client.reset_mock()

        pub.publish_consolidated_event(
            group="speech",
            cameras=["living_room"],
            max_confidence=0.75,
            detection_count=1,
            duration_seconds=2.1234,
            first_detected="2025-06-01T12:00:00",
            last_detected="2025-06-01T12:00:02",
        )

        attr_calls = _find_publish(pub._client, "consolidated/speech/attributes")
        assert len(attr_calls) == 1
        attrs = json.loads(attr_calls[0][1])
        assert attrs["cameras"] == ["living_room"]
        assert attrs["max_confidence"] == 0.75
        assert attrs["detection_count"] == 1
        assert attrs["duration_seconds"] == 2.12  # rounded to 2 decimals
        assert attrs["first_detected"] == "2025-06-01T12:00:00"
        assert attrs["last_detected"] == "2025-06-01T12:00:02"

    def test_consolidated_event_publish_count(self):
        config = _make_config()
        pub = MqttPublisher(config)
        pub._client.reset_mock()

        pub.publish_consolidated_event(
            group="dog_bark",
            cameras=["cam"],
            max_confidence=0.5,
            detection_count=1,
            duration_seconds=0.0,
            first_detected="t1",
            last_detected="t2",
        )
        assert pub._client.publish.call_count == 2  # state + attributes


# ---------------------------------------------------------------------------
# Tests: Noise stress discovery & score
# ---------------------------------------------------------------------------


class TestNoiseStressDiscovery:
    def test_publishes_sensor_discovery(self):
        config = _make_config()
        pub = MqttPublisher(config)
        pub._client.reset_mock()

        pub.publish_noise_stress_discovery()

        published = _find_publish(pub._client, "sensor/ast_noise_stress_score/config")
        assert len(published) == 1
        payload = json.loads(published[0][1])
        assert payload["unique_id"] == "ast_noise_stress_score"
        assert payload["name"] == "AST Noise Stress Score"
        assert payload["state_topic"] == f"{TOPIC_PREFIX}/noise_stress/state"
        assert payload["json_attributes_topic"] == f"{TOPIC_PREFIX}/noise_stress/attributes"
        assert payload["unit_of_measurement"] == ""
        assert payload["state_class"] == "measurement"
        assert payload["availability_topic"] == f"{TOPIC_PREFIX}/status"
        assert payload["payload_available"] == "online"
        assert payload["payload_not_available"] == "offline"
        assert payload["icon"] == "mdi:head-alert"

    def test_noise_stress_device_info(self):
        config = _make_config()
        pub = MqttPublisher(config)
        pub._client.reset_mock()

        pub.publish_noise_stress_discovery()

        published = _find_publish(pub._client, "sensor/ast_noise_stress_score/config")
        payload = json.loads(published[0][1])
        device = payload["device"]
        assert device["identifiers"] == ["ast_audio_noise_stress"]
        assert device["name"] == "AST Audio Classifier - Noise Stress"
        assert device["sw_version"] == __version__


class TestPublishNoiseStressScore:
    def test_publishes_state_and_attributes(self):
        config = _make_config()
        pub = MqttPublisher(config)
        pub._client.reset_mock()

        score_data = {
            "score": 42,
            "ambient_component": 10.5,
            "event_component": 20.0,
            "sustained_component": 11.5,
            "recent_event_count": 7,
            "top_stressor": "dog_bark",
            "dominant_camera": "backyard",
            "active_high_stress": True,
            "daily_avg": 35.2,
            "daily_min": 10.0,
            "daily_max": 78.0,
            "daily_samples": 120,
        }
        pub.publish_noise_stress_score(score_data)

        state_calls = _find_publish(pub._client, "noise_stress/state")
        assert len(state_calls) == 1
        assert state_calls[0][1] == "42"

        attr_calls = _find_publish(pub._client, "noise_stress/attributes")
        assert len(attr_calls) == 1
        attrs = json.loads(attr_calls[0][1])
        assert attrs["ambient_component"] == 10.5
        assert attrs["event_component"] == 20.0
        assert attrs["sustained_component"] == 11.5
        assert attrs["recent_event_count"] == 7
        assert attrs["top_stressor"] == "dog_bark"
        assert attrs["dominant_camera"] == "backyard"
        assert attrs["active_high_stress"] is True
        assert attrs["daily_avg"] == 35.2
        assert attrs["daily_min"] == 10.0
        assert attrs["daily_max"] == 78.0
        assert attrs["daily_samples"] == 120

    def test_defaults_for_optional_fields(self):
        config = _make_config()
        pub = MqttPublisher(config)
        pub._client.reset_mock()

        score_data = {
            "score": 5,
            "ambient_component": 3.0,
            "event_component": 1.0,
            "sustained_component": 1.0,
            "recent_event_count": 0,
        }
        pub.publish_noise_stress_score(score_data)

        attr_calls = _find_publish(pub._client, "noise_stress/attributes")
        attrs = json.loads(attr_calls[0][1])
        assert attrs["top_stressor"] is None
        assert attrs["dominant_camera"] is None
        assert attrs["active_high_stress"] is False
        assert attrs["daily_avg"] == 0.0
        assert attrs["daily_min"] == 0.0
        assert attrs["daily_max"] == 0.0
        assert attrs["daily_samples"] == 0

    def test_noise_stress_publish_count(self):
        config = _make_config()
        pub = MqttPublisher(config)
        pub._client.reset_mock()

        pub.publish_noise_stress_score(
            {
                "score": 0,
                "ambient_component": 0,
                "event_component": 0,
                "sustained_component": 0,
                "recent_event_count": 0,
            }
        )
        assert pub._client.publish.call_count == 2  # state + attributes


# ---------------------------------------------------------------------------
# Tests: _on_connect callback
# ---------------------------------------------------------------------------


class TestOnConnect:
    def test_on_connect_success_publishes_discovery_and_online(self):
        import asyncio

        async def _run():
            config = _make_config()
            pub = MqttPublisher(config)
            pub._loop = asyncio.get_running_loop()
            pub._client.reset_mock()

            pub._on_connect(pub._client, None, {}, 0)

            assert pub.connected is True
            # Should publish online status
            online_calls = [
                c
                for c in pub._client.publish.call_args_list
                if c.args[0] == f"{TOPIC_PREFIX}/status" and c.args[1] == "online"
            ]
            assert len(online_calls) == 1

        asyncio.run(_run())

    def test_on_connect_failure_does_not_set_connected(self):
        import asyncio

        async def _run():
            config = _make_config()
            pub = MqttPublisher(config)
            pub._loop = asyncio.get_running_loop()

            pub._on_connect(pub._client, None, {}, 5)  # rc=5 = not authorized

            assert pub.connected is False

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Tests: _on_disconnect callback
# ---------------------------------------------------------------------------


class TestOnDisconnect:
    def test_on_disconnect_clears_connected(self):
        import asyncio

        async def _run():
            config = _make_config()
            pub = MqttPublisher(config)
            pub._loop = asyncio.get_running_loop()
            pub._connected = True
            pub._connected_event.set()

            pub._on_disconnect(pub._client, None, 0)

            assert pub.connected is False

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Tests: Multi-camera scenarios
# ---------------------------------------------------------------------------


class TestMultiCamera:
    def test_publish_discovery_creates_entities_for_each_camera(self):
        cameras = [
            CameraConfig(name="cam_a", rtsp_url="rtsp://h/a"),
            CameraConfig(name="cam_b", rtsp_url="rtsp://h/b"),
        ]
        config = _make_config(cameras=cameras)
        pub = MqttPublisher(config)
        pub._client.reset_mock()

        pub._publish_discovery()

        published_topics = [c.args[0] for c in pub._client.publish.call_args_list]
        # Spot-check that both cameras got discovery
        assert any("ast_cam_a_dog_bark" in t for t in published_topics)
        assert any("ast_cam_b_dog_bark" in t for t in published_topics)
        assert any("ast_cam_a_last_event" in t for t in published_topics)
        assert any("ast_cam_b_last_event" in t for t in published_topics)

    def test_detection_routes_to_correct_camera(self):
        cameras = [
            CameraConfig(name="cam_a", rtsp_url="rtsp://h/a"),
            CameraConfig(name="cam_b", rtsp_url="rtsp://h/b"),
        ]
        config = _make_config(cameras=cameras)
        pub = MqttPublisher(config)
        pub._client.reset_mock()

        pub.publish_detection("cam_b", _make_result(group="cat_meow"))

        state_calls = _find_publish(pub._client, "cam_b/cat_meow/state")
        assert len(state_calls) == 1
        assert state_calls[0][1] == "ON"

        # cam_a should not have any publishes
        cam_a_calls = _find_publish(pub._client, "cam_a/")
        assert len(cam_a_calls) == 0
