"""Configuration loading with dataclass validation and env var substitution."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class MqttConfig:
    host: str
    port: int = 1883
    username: str | None = None
    password: str | None = None


@dataclass
class ConfounderConfig:
    """A confounder is an HA entity whose state affects which sound groups are trustworthy."""

    entity_id: str
    active_when: str  # "!off", ">200", "playing", "on", etc.
    confused_groups: list[str]


@dataclass
class CameraConfig:
    name: str
    rtsp_url: str
    db_threshold: float = -35.0
    cooldown_seconds: int = 10
    battery: bool = False
    reconnect_interval: int = 5
    highpass_freq: int = 0
    adaptive_threshold: bool = False
    adaptive_margin_db: float = 8.0
    go2rtc_stream: str | None = None
    confounders: list[ConfounderConfig] | None = None

    def __post_init__(self) -> None:
        if self.highpass_freq < 0:
            msg = f"highpass_freq must be >= 0 (0 = disabled), got {self.highpass_freq}"
            raise ValueError(msg)
        if self.adaptive_threshold and self.adaptive_margin_db <= 0:
            msg = f"adaptive_margin_db must be > 0 when adaptive_threshold=true, got {self.adaptive_margin_db}"
            raise ValueError(msg)


@dataclass
class OpenObserveConfig:
    host: str
    port: int = 5080
    org: str = "default"
    stream: str = "ast_audio"
    username: str | None = None
    password: str | None = None


@dataclass
class CLAPOptions:
    """Raw CLAP settings as loaded from YAML / HA addon options."""

    enabled: bool = True
    model: str = "laion/clap-htsat-fused"
    confirm_threshold: float = 0.30
    suppress_threshold: float = 0.15
    override_threshold: float = 0.40
    discovery_threshold: float = 0.50
    confirm_margin: float = 0.20
    ast_bypass_threshold: float = 0.80
    never_suppress: list[str] | None = None
    custom_prompts: dict[str, list[str]] | None = None

    def __post_init__(self) -> None:
        if self.suppress_threshold >= self.confirm_threshold:
            msg = (
                f"suppress_threshold must be < confirm_threshold, "
                f"got suppress={self.suppress_threshold}, confirm={self.confirm_threshold}"
            )
            raise ValueError(msg)
        if self.override_threshold <= self.suppress_threshold:
            msg = (
                f"override_threshold must be > suppress_threshold, "
                f"got override={self.override_threshold}, suppress={self.suppress_threshold}"
            )
            raise ValueError(msg)
        if self.confirm_margin < 0:
            msg = f"confirm_margin must be >= 0, got {self.confirm_margin}"
            raise ValueError(msg)


@dataclass
class LLMJudgeConfig:
    enabled: bool = False
    api_base: str = ""
    api_key: str = ""
    model: str = "gemini-2.5-flash"
    sample_rate: float = 0.10
    clip_dir: str = "/media/ast-audio-classifier/clips"
    max_clips: int = 5000
    timeout_seconds: int = 30

    def __post_init__(self) -> None:
        if not (0.0 <= self.sample_rate <= 1.0):
            msg = f"sample_rate must be in [0, 1], got {self.sample_rate}"
            raise ValueError(msg)
        if self.max_clips < 1:
            msg = f"max_clips must be >= 1, got {self.max_clips}"
            raise ValueError(msg)
        if self.timeout_seconds < 1:
            msg = f"timeout_seconds must be >= 1, got {self.timeout_seconds}"
            raise ValueError(msg)
        if self.enabled and not self.api_base:
            msg = "llm_judge.api_base must be set when enabled=true"
            raise ValueError(msg)


@dataclass
class NoiseStressConfig:
    enabled: bool = False
    update_interval_seconds: float = 30.0
    decay_half_life_seconds: float = 180.0
    saturation_constant: float = 25.0
    indoor_cameras: list[str] | None = None


@dataclass
class AppConfig:
    mqtt: MqttConfig
    cameras: list[CameraConfig]
    openobserve: OpenObserveConfig | None = None
    clap: CLAPOptions | None = None
    llm_judge: LLMJudgeConfig | None = None
    noise_stress: NoiseStressConfig | None = None
    confidence_threshold: float = 0.15
    auto_off_seconds: int = 30
    clip_duration_seconds: int = 3
    health_port: int = 8080
    consolidated_enabled: bool = False
    consolidated_window_seconds: float = 5.0


_ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)\}")


def _substitute_env_vars(value: str) -> str:
    """Replace ${VAR} patterns with environment variable values."""

    def _replace(match: re.Match) -> str:
        var_name = match.group(1)
        env_val = os.environ.get(var_name)
        if env_val is None:
            msg = f"Environment variable {var_name!r} not set"
            raise ValueError(msg)
        return env_val

    if isinstance(value, str):
        return _ENV_VAR_PATTERN.sub(_replace, value)
    return value


def _walk_and_substitute(obj: dict | list | str) -> dict | list | str:
    """Recursively substitute env vars in a config dict."""
    if isinstance(obj, dict):
        return {k: _walk_and_substitute(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_and_substitute(item) for item in obj]
    if isinstance(obj, str):
        return _substitute_env_vars(obj)
    return obj


def load_config(path: str) -> AppConfig:
    """Load and validate config from a YAML file.

    Supports ${ENV_VAR} substitution in string values.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        ValueError: If required fields are missing.
    """
    config_path = Path(path)
    if not config_path.exists():
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)

    with config_path.open() as f:
        raw = yaml.safe_load(f)

    raw = _walk_and_substitute(raw)

    if "mqtt" not in raw:
        msg = "Config missing required 'mqtt' section"
        raise ValueError(msg)
    if "cameras" not in raw or not raw["cameras"]:
        msg = "Config missing required 'cameras' section"
        raise ValueError(msg)

    mqtt = MqttConfig(**raw["mqtt"])
    cameras = []
    for cam_raw in raw["cameras"]:
        cam_dict = dict(cam_raw)
        confounders_raw = cam_dict.pop("confounders", None)
        if confounders_raw:
            cam_dict["confounders"] = [
                ConfounderConfig(**c) for c in confounders_raw
            ]
        cameras.append(CameraConfig(**cam_dict))

    openobserve = None
    if "openobserve" in raw:
        openobserve = OpenObserveConfig(**raw["openobserve"])

    clap = None
    if "clap" in raw:
        clap_raw = dict(raw["clap"])
        clap = CLAPOptions(**clap_raw)

    llm_judge = None
    if "llm_judge" in raw:
        llm_judge = LLMJudgeConfig(**raw["llm_judge"])

    noise_stress = None
    if "noise_stress" in raw:
        noise_stress = NoiseStressConfig(**raw["noise_stress"])

    defaults = raw.get("defaults", {})
    return AppConfig(
        mqtt=mqtt,
        cameras=cameras,
        openobserve=openobserve,
        clap=clap,
        llm_judge=llm_judge,
        noise_stress=noise_stress,
        confidence_threshold=defaults.get("confidence_threshold", 0.15),
        auto_off_seconds=defaults.get("auto_off_seconds", 30),
        clip_duration_seconds=defaults.get("clip_duration_seconds", 3),
        health_port=defaults.get("health_port", 8080),
        consolidated_enabled=defaults.get("consolidated_enabled", False),
        consolidated_window_seconds=defaults.get("consolidated_window_seconds", 5.0),
    )
