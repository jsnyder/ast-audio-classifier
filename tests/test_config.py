"""Tests for configuration loading."""

import pytest
import yaml

from src.config import (
    AppConfig,
    CameraConfig,
    CLAPConfig,
    MqttConfig,
    OpenObserveConfig,
    load_config,
)


MINIMAL_CONFIG = {
    "mqtt": {"host": "localhost"},
    "cameras": [
        {"name": "test_cam", "rtsp_url": "rtsp://192.168.1.1:8554/test"},
    ],
}


FULL_CONFIG = {
    "mqtt": {
        "host": "core-mosquitto",
        "port": 1883,
        "username": "ast_classifier",
        "password": "secret",
    },
    "cameras": [
        {
            "name": "living_room",
            "rtsp_url": "rtsp://192.168.0.107:8554/living_room",
            "db_threshold": -40,
            "cooldown_seconds": 15,
        },
        {
            "name": "backyard",
            "rtsp_url": "rtsp://192.168.0.107:8554/backyard_local",
            "db_threshold": -30,
            "battery": True,
            "reconnect_interval": 60,
        },
    ],
    "defaults": {
        "confidence_threshold": 0.2,
        "auto_off_seconds": 45,
        "clip_duration_seconds": 4,
    },
}


class TestMqttConfig:
    def test_defaults(self):
        cfg = MqttConfig(host="localhost")
        assert cfg.host == "localhost"
        assert cfg.port == 1883
        assert cfg.username is None
        assert cfg.password is None

    def test_full(self):
        cfg = MqttConfig(host="broker", port=8883, username="user", password="pass")
        assert cfg.port == 8883
        assert cfg.username == "user"


class TestCameraConfig:
    def test_defaults(self):
        cfg = CameraConfig(name="test", rtsp_url="rtsp://host/stream")
        assert cfg.db_threshold == -35
        assert cfg.cooldown_seconds == 10
        assert cfg.battery is False
        assert cfg.reconnect_interval == 5

    def test_battery_camera(self):
        cfg = CameraConfig(
            name="outdoor",
            rtsp_url="rtsp://host/stream",
            battery=True,
            reconnect_interval=60,
        )
        assert cfg.battery is True
        assert cfg.reconnect_interval == 60


class TestAppConfig:
    def test_minimal(self):
        cfg = AppConfig(
            mqtt=MqttConfig(host="localhost"),
            cameras=[CameraConfig(name="cam", rtsp_url="rtsp://host/s")],
        )
        assert cfg.confidence_threshold == 0.15
        assert cfg.auto_off_seconds == 30
        assert cfg.clip_duration_seconds == 3
        assert cfg.health_port == 8080

    def test_custom_defaults(self):
        cfg = AppConfig(
            mqtt=MqttConfig(host="localhost"),
            cameras=[CameraConfig(name="cam", rtsp_url="rtsp://host/s")],
            confidence_threshold=0.25,
            auto_off_seconds=60,
            clip_duration_seconds=5,
        )
        assert cfg.confidence_threshold == 0.25
        assert cfg.auto_off_seconds == 60
        assert cfg.clip_duration_seconds == 5


class TestOpenObserveConfig:
    def test_defaults(self):
        cfg = OpenObserveConfig(host="192.168.1.63")
        assert cfg.port == 5080
        assert cfg.org == "default"
        assert cfg.stream == "ast_audio"
        assert cfg.username is None
        assert cfg.password is None

    def test_full(self):
        cfg = OpenObserveConfig(
            host="oo.example.com",
            port=443,
            org="prod",
            stream="audio_events",
            username="admin",
            password="secret",
        )
        assert cfg.host == "oo.example.com"
        assert cfg.port == 443
        assert cfg.org == "prod"
        assert cfg.stream == "audio_events"


class TestAppConfigWithOpenObserve:
    def test_openobserve_none_by_default(self):
        cfg = AppConfig(
            mqtt=MqttConfig(host="localhost"),
            cameras=[CameraConfig(name="cam", rtsp_url="rtsp://host/s")],
        )
        assert cfg.openobserve is None

    def test_openobserve_set(self):
        cfg = AppConfig(
            mqtt=MqttConfig(host="localhost"),
            cameras=[CameraConfig(name="cam", rtsp_url="rtsp://host/s")],
            openobserve=OpenObserveConfig(host="192.168.1.63"),
        )
        assert cfg.openobserve is not None
        assert cfg.openobserve.host == "192.168.1.63"


class TestLoadConfig:
    def test_load_minimal(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump(MINIMAL_CONFIG))

        cfg = load_config(str(cfg_file))
        assert isinstance(cfg, AppConfig)
        assert cfg.mqtt.host == "localhost"
        assert len(cfg.cameras) == 1
        assert cfg.cameras[0].name == "test_cam"

    def test_load_full(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump(FULL_CONFIG))

        cfg = load_config(str(cfg_file))
        assert cfg.mqtt.username == "ast_classifier"
        assert len(cfg.cameras) == 2
        assert cfg.cameras[1].battery is True
        assert cfg.confidence_threshold == 0.2
        assert cfg.auto_off_seconds == 45

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_missing_cameras_raises(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump({"mqtt": {"host": "localhost"}}))
        with pytest.raises((KeyError, TypeError, ValueError)):
            load_config(str(cfg_file))

    def test_missing_mqtt_raises(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            yaml.dump({"cameras": [{"name": "x", "rtsp_url": "rtsp://h/s"}]})
        )
        with pytest.raises((KeyError, TypeError, ValueError)):
            load_config(str(cfg_file))

    def test_env_var_substitution(self, tmp_path, monkeypatch):
        """Config values with ${VAR} should be substituted from env."""
        monkeypatch.setenv("MQTT_PASSWORD", "from_env")
        cfg_data = {
            "mqtt": {"host": "localhost", "password": "${MQTT_PASSWORD}"},
            "cameras": [{"name": "cam", "rtsp_url": "rtsp://h/s"}],
        }
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump(cfg_data))

        cfg = load_config(str(cfg_file))
        assert cfg.mqtt.password == "from_env"

    def test_load_with_openobserve(self, tmp_path):
        cfg_data = {
            **MINIMAL_CONFIG,
            "openobserve": {
                "host": "192.168.1.63",
                "port": 5080,
                "org": "default",
                "stream": "ast_audio",
                "username": "admin",
                "password": "secret",
            },
        }
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump(cfg_data))

        cfg = load_config(str(cfg_file))
        assert cfg.openobserve is not None
        assert cfg.openobserve.host == "192.168.1.63"
        assert cfg.openobserve.username == "admin"

    def test_load_without_openobserve(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump(MINIMAL_CONFIG))

        cfg = load_config(str(cfg_file))
        assert cfg.openobserve is None


class TestCLAPConfig:
    def test_defaults(self):
        cfg = CLAPConfig()
        assert cfg.enabled is True
        assert cfg.model == "laion/clap-htsat-fused"
        assert cfg.confirm_threshold == 0.25
        assert cfg.suppress_threshold == 0.15
        assert cfg.override_threshold == 0.40
        assert cfg.discovery_threshold == 0.50
        assert cfg.never_suppress is None
        assert cfg.custom_prompts is None

    def test_custom(self):
        cfg = CLAPConfig(
            enabled=True,
            confirm_threshold=0.30,
            never_suppress=["smoke_alarm", "glass_break"],
            custom_prompts={"vacuum_cleaner": ["a roomba running"]},
        )
        assert cfg.confirm_threshold == 0.30
        assert cfg.never_suppress == ["smoke_alarm", "glass_break"]
        assert cfg.custom_prompts["vacuum_cleaner"] == ["a roomba running"]


class TestLoadConfigCLAP:
    def test_load_with_clap(self, tmp_path):
        cfg_data = {
            **MINIMAL_CONFIG,
            "clap": {
                "enabled": True,
                "model": "laion/clap-htsat-fused",
                "confirm_threshold": 0.30,
                "suppress_threshold": 0.10,
                "override_threshold": 0.45,
                "discovery_threshold": 0.55,
                "never_suppress": ["smoke_alarm", "glass_break", "siren"],
                "custom_prompts": {
                    "vacuum_cleaner": [
                        "a robot vacuum cleaner running",
                        "a vacuum cleaner motor",
                    ],
                },
            },
        }
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump(cfg_data))

        cfg = load_config(str(cfg_file))
        assert cfg.clap is not None
        assert cfg.clap.enabled is True
        assert cfg.clap.confirm_threshold == 0.30
        assert cfg.clap.suppress_threshold == 0.10
        assert cfg.clap.override_threshold == 0.45
        assert cfg.clap.discovery_threshold == 0.55
        assert cfg.clap.never_suppress == ["smoke_alarm", "glass_break", "siren"]
        assert len(cfg.clap.custom_prompts["vacuum_cleaner"]) == 2

    def test_load_clap_disabled(self, tmp_path):
        cfg_data = {
            **MINIMAL_CONFIG,
            "clap": {"enabled": False},
        }
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump(cfg_data))

        cfg = load_config(str(cfg_file))
        assert cfg.clap is not None
        assert cfg.clap.enabled is False

    def test_load_without_clap(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump(MINIMAL_CONFIG))

        cfg = load_config(str(cfg_file))
        assert cfg.clap is None

    def test_load_clap_minimal(self, tmp_path):
        """CLAP section with only enabled=true should use all defaults."""
        cfg_data = {
            **MINIMAL_CONFIG,
            "clap": {"enabled": True},
        }
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump(cfg_data))

        cfg = load_config(str(cfg_file))
        assert cfg.clap is not None
        assert cfg.clap.enabled is True
        assert cfg.clap.model == "laion/clap-htsat-fused"
        assert cfg.clap.confirm_threshold == 0.25
