"""Tests for configuration loading."""

import pytest
import yaml

from src.config import (
    AppConfig,
    CameraConfig,
    CLAPOptions,
    LLMJudgeConfig,
    MqttConfig,
    NoiseStressConfig,
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
        assert cfg.highpass_freq == 0
        assert cfg.adaptive_threshold is False
        assert cfg.adaptive_margin_db == 8.0

    def test_battery_camera(self):
        cfg = CameraConfig(
            name="outdoor",
            rtsp_url="rtsp://host/stream",
            battery=True,
            reconnect_interval=60,
        )
        assert cfg.battery is True
        assert cfg.reconnect_interval == 60

    def test_highpass_freq(self):
        cfg = CameraConfig(
            name="outdoor",
            rtsp_url="rtsp://host/stream",
            highpass_freq=120,
        )
        assert cfg.highpass_freq == 120

    def test_adaptive_threshold(self):
        cfg = CameraConfig(
            name="outdoor",
            rtsp_url="rtsp://host/stream",
            adaptive_threshold=True,
            adaptive_margin_db=6.0,
        )
        assert cfg.adaptive_threshold is True
        assert cfg.adaptive_margin_db == 6.0

    def test_negative_highpass_freq_raises(self):
        with pytest.raises(ValueError, match="highpass_freq must be >= 0"):
            CameraConfig(name="test", rtsp_url="rtsp://host/s", highpass_freq=-1)

    def test_zero_adaptive_margin_raises(self):
        with pytest.raises(ValueError, match="adaptive_margin_db must be > 0"):
            CameraConfig(
                name="test",
                rtsp_url="rtsp://host/s",
                adaptive_threshold=True,
                adaptive_margin_db=0.0,
            )

    def test_negative_adaptive_margin_raises(self):
        with pytest.raises(ValueError, match="adaptive_margin_db must be > 0"):
            CameraConfig(
                name="test",
                rtsp_url="rtsp://host/s",
                adaptive_threshold=True,
                adaptive_margin_db=-2.0,
            )

    def test_adaptive_margin_ignored_when_disabled(self):
        """Zero margin is fine when adaptive_threshold is disabled."""
        cfg = CameraConfig(
            name="test",
            rtsp_url="rtsp://host/s",
            adaptive_threshold=False,
            adaptive_margin_db=0.0,
        )
        assert cfg.adaptive_margin_db == 0.0


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
        assert cfg.consolidated_enabled is False
        assert cfg.consolidated_window_seconds == 5.0

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

    def test_consolidated_config(self):
        cfg = AppConfig(
            mqtt=MqttConfig(host="localhost"),
            cameras=[CameraConfig(name="cam", rtsp_url="rtsp://host/s")],
            consolidated_enabled=True,
            consolidated_window_seconds=3.0,
        )
        assert cfg.consolidated_enabled is True
        assert cfg.consolidated_window_seconds == 3.0


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

    def test_load_with_new_camera_fields(self, tmp_path):
        cfg_data = {
            "mqtt": {"host": "localhost"},
            "cameras": [
                {
                    "name": "outdoor",
                    "rtsp_url": "rtsp://host/stream",
                    "highpass_freq": 120,
                    "adaptive_threshold": True,
                    "adaptive_margin_db": 6.0,
                },
            ],
        }
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump(cfg_data))

        cfg = load_config(str(cfg_file))
        assert cfg.cameras[0].highpass_freq == 120
        assert cfg.cameras[0].adaptive_threshold is True
        assert cfg.cameras[0].adaptive_margin_db == 6.0

    def test_load_consolidated_config(self, tmp_path):
        cfg_data = {
            **MINIMAL_CONFIG,
            "defaults": {
                "consolidated_enabled": True,
                "consolidated_window_seconds": 3.0,
            },
        }
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump(cfg_data))

        cfg = load_config(str(cfg_file))
        assert cfg.consolidated_enabled is True
        assert cfg.consolidated_window_seconds == 3.0

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


class TestCLAPOptions:
    def test_defaults(self):
        cfg = CLAPOptions()
        assert cfg.enabled is True
        assert cfg.model == "laion/clap-htsat-fused"
        assert cfg.confirm_threshold == 0.30
        assert cfg.suppress_threshold == 0.15
        assert cfg.override_threshold == 0.40
        assert cfg.discovery_threshold == 0.50
        assert cfg.confirm_margin == 0.20
        assert cfg.never_suppress is None
        assert cfg.custom_prompts is None

    def test_custom(self):
        cfg = CLAPOptions(
            enabled=True,
            confirm_threshold=0.30,
            never_suppress=["smoke_alarm", "glass_break"],
            custom_prompts={"vacuum_cleaner": ["a roomba running"]},
        )
        assert cfg.confirm_threshold == 0.30
        assert cfg.never_suppress == ["smoke_alarm", "glass_break"]
        assert cfg.custom_prompts["vacuum_cleaner"] == ["a roomba running"]


class TestCLAPOptionsValidation:
    def test_suppress_gte_confirm_raises(self):
        with pytest.raises(ValueError, match="suppress_threshold must be < confirm_threshold"):
            CLAPOptions(suppress_threshold=0.30, confirm_threshold=0.15)

    def test_suppress_equals_confirm_raises(self):
        with pytest.raises(ValueError, match="suppress_threshold must be < confirm_threshold"):
            CLAPOptions(suppress_threshold=0.25, confirm_threshold=0.25)

    def test_override_lte_suppress_raises(self):
        with pytest.raises(ValueError, match="override_threshold must be > suppress_threshold"):
            CLAPOptions(override_threshold=0.10, suppress_threshold=0.15)

    def test_negative_confirm_margin_raises(self):
        with pytest.raises(ValueError, match="confirm_margin must be >= 0"):
            CLAPOptions(confirm_margin=-0.05)

    def test_valid_thresholds_pass(self):
        """Non-default but valid thresholds should not raise."""
        cfg = CLAPOptions(
            suppress_threshold=0.10,
            confirm_threshold=0.20,
            override_threshold=0.35,
            confirm_margin=0.0,
        )
        assert cfg.suppress_threshold == 0.10


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
        assert cfg.clap.confirm_threshold == 0.30
        assert cfg.clap.confirm_margin == 0.20


class TestLLMJudgeConfig:
    def test_defaults(self):
        cfg = LLMJudgeConfig()
        assert cfg.enabled is False
        assert cfg.api_base == ""
        assert cfg.api_key == ""
        assert cfg.model == "gemini-2.5-flash"
        assert cfg.sample_rate == 0.10
        assert cfg.clip_dir == "/media/ast-audio-classifier/clips"
        assert cfg.max_clips == 5000
        assert cfg.timeout_seconds == 30

    def test_custom(self):
        cfg = LLMJudgeConfig(
            enabled=True,
            api_base="https://example.com/v1",
            api_key="sk-test",
            model="gpt-4o-audio",
            sample_rate=0.25,
            clip_dir="/tmp/clips",
            max_clips=1000,
            timeout_seconds=60,
        )
        assert cfg.enabled is True
        assert cfg.api_key == "sk-test"
        assert cfg.sample_rate == 0.25
        assert cfg.max_clips == 1000


class TestLLMJudgeConfigValidation:
    def test_enabled_without_api_base_raises(self):
        """Enabling LLM judge without api_base must raise immediately."""
        with pytest.raises(ValueError, match="llm_judge.api_base must be set"):
            LLMJudgeConfig(enabled=True, api_base="")

    def test_enabled_with_api_base_passes(self):
        cfg = LLMJudgeConfig(enabled=True, api_base="https://example.com/v1")
        assert cfg.enabled is True

    def test_sample_rate_out_of_range_raises(self):
        with pytest.raises(ValueError, match="sample_rate"):
            LLMJudgeConfig(enabled=True, api_base="https://example.com/v1", sample_rate=1.5)

    def test_max_clips_zero_raises(self):
        with pytest.raises(ValueError, match="max_clips"):
            LLMJudgeConfig(enabled=True, api_base="https://example.com/v1", max_clips=0)

    def test_timeout_zero_raises(self):
        with pytest.raises(ValueError, match="timeout_seconds"):
            LLMJudgeConfig(enabled=True, api_base="https://example.com/v1", timeout_seconds=0)


class TestLoadConfigLLMJudge:
    def test_llm_judge_none_when_absent(self, tmp_path):
        """Config without llm_judge section → config.llm_judge is None."""
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump(MINIMAL_CONFIG))

        cfg = load_config(str(cfg_file))
        assert cfg.llm_judge is None

    def test_load_with_llm_judge(self, tmp_path):
        cfg_data = {
            **MINIMAL_CONFIG,
            "llm_judge": {
                "enabled": True,
                "api_base": "https://llm-proxy.example.com/v1",
                "api_key": "sk-test-key",
                "model": "gemini-2.5-flash",
                "sample_rate": 0.20,
                "clip_dir": "/tmp/test_clips",
                "max_clips": 2000,
                "timeout_seconds": 45,
            },
        }
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump(cfg_data))

        cfg = load_config(str(cfg_file))
        assert cfg.llm_judge is not None
        assert cfg.llm_judge.enabled is True
        assert cfg.llm_judge.api_key == "sk-test-key"
        assert cfg.llm_judge.sample_rate == 0.20
        assert cfg.llm_judge.max_clips == 2000

    def test_load_llm_judge_disabled(self, tmp_path):
        cfg_data = {
            **MINIMAL_CONFIG,
            "llm_judge": {"enabled": False},
        }
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump(cfg_data))

        cfg = load_config(str(cfg_file))
        assert cfg.llm_judge is not None
        assert cfg.llm_judge.enabled is False

    def test_load_llm_judge_env_var_substitution(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LLM_JUDGE_API_KEY", "from_env_key")
        cfg_data = {
            **MINIMAL_CONFIG,
            "llm_judge": {
                "enabled": True,
                "api_base": "https://litellm.example.com/v1",
                "api_key": "${LLM_JUDGE_API_KEY}",
            },
        }
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump(cfg_data))

        cfg = load_config(str(cfg_file))
        assert cfg.llm_judge is not None
        assert cfg.llm_judge.api_key == "from_env_key"

    def test_load_llm_judge_minimal(self, tmp_path):
        """LLM judge section with enabled=true and api_base should use other defaults."""
        cfg_data = {
            **MINIMAL_CONFIG,
            "llm_judge": {"enabled": True, "api_base": "https://litellm.example.com/v1"},
        }
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump(cfg_data))

        cfg = load_config(str(cfg_file))
        assert cfg.llm_judge is not None
        assert cfg.llm_judge.enabled is True
        assert cfg.llm_judge.model == "gemini-2.5-flash"
        assert cfg.llm_judge.sample_rate == 0.10


class TestNoiseStressConfig:
    def test_defaults(self):
        cfg = NoiseStressConfig()
        assert cfg.enabled is False
        assert cfg.update_interval_seconds == 30.0
        assert cfg.decay_half_life_seconds == 180.0
        assert cfg.saturation_constant == 25.0
        assert cfg.indoor_cameras is None

    def test_custom(self):
        cfg = NoiseStressConfig(
            enabled=True,
            update_interval_seconds=15.0,
            decay_half_life_seconds=60.0,
            saturation_constant=8.0,
            indoor_cameras=["living_room", "basement"],
        )
        assert cfg.enabled is True
        assert cfg.update_interval_seconds == 15.0
        assert cfg.decay_half_life_seconds == 60.0
        assert cfg.saturation_constant == 8.0
        assert cfg.indoor_cameras == ["living_room", "basement"]


class TestAppConfigNoiseStress:
    def test_noise_stress_none_by_default(self):
        cfg = AppConfig(
            mqtt=MqttConfig(host="localhost"),
            cameras=[CameraConfig(name="cam", rtsp_url="rtsp://host/s")],
        )
        assert cfg.noise_stress is None

    def test_noise_stress_set(self):
        cfg = AppConfig(
            mqtt=MqttConfig(host="localhost"),
            cameras=[CameraConfig(name="cam", rtsp_url="rtsp://host/s")],
            noise_stress=NoiseStressConfig(enabled=True),
        )
        assert cfg.noise_stress is not None
        assert cfg.noise_stress.enabled is True


class TestLoadConfigNoiseStress:
    def test_load_with_noise_stress(self, tmp_path):
        cfg_data = {
            **MINIMAL_CONFIG,
            "noise_stress": {
                "enabled": True,
                "update_interval_seconds": 15,
                "decay_half_life_seconds": 90.0,
                "saturation_constant": 8.0,
                "indoor_cameras": ["living_room", "basement_litter"],
            },
        }
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump(cfg_data))

        cfg = load_config(str(cfg_file))
        assert cfg.noise_stress is not None
        assert cfg.noise_stress.enabled is True
        assert cfg.noise_stress.update_interval_seconds == 15
        assert cfg.noise_stress.decay_half_life_seconds == 90.0
        assert cfg.noise_stress.saturation_constant == 8.0
        assert cfg.noise_stress.indoor_cameras == ["living_room", "basement_litter"]

    def test_load_without_noise_stress(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump(MINIMAL_CONFIG))

        cfg = load_config(str(cfg_file))
        assert cfg.noise_stress is None

    def test_load_noise_stress_minimal(self, tmp_path):
        """Noise stress with only enabled=true should use defaults."""
        cfg_data = {
            **MINIMAL_CONFIG,
            "noise_stress": {"enabled": True},
        }
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump(cfg_data))

        cfg = load_config(str(cfg_file))
        assert cfg.noise_stress is not None
        assert cfg.noise_stress.enabled is True
        assert cfg.noise_stress.update_interval_seconds == 30.0
        assert cfg.noise_stress.decay_half_life_seconds == 180.0
        assert cfg.noise_stress.saturation_constant == 25.0
