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
class CameraConfig:
    name: str
    rtsp_url: str
    db_threshold: float = -35.0
    cooldown_seconds: int = 10
    battery: bool = False
    reconnect_interval: int = 5


@dataclass
class OpenObserveConfig:
    host: str
    port: int = 5080
    org: str = "default"
    stream: str = "ast_audio"
    username: str | None = None
    password: str | None = None


@dataclass
class AppConfig:
    mqtt: MqttConfig
    cameras: list[CameraConfig]
    openobserve: OpenObserveConfig | None = None
    confidence_threshold: float = 0.15
    auto_off_seconds: int = 30
    clip_duration_seconds: int = 3
    health_port: int = 8080


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
    cameras = [CameraConfig(**cam) for cam in raw["cameras"]]

    openobserve = None
    if "openobserve" in raw:
        openobserve = OpenObserveConfig(**raw["openobserve"])

    defaults = raw.get("defaults", {})
    return AppConfig(
        mqtt=mqtt,
        cameras=cameras,
        openobserve=openobserve,
        confidence_threshold=defaults.get("confidence_threshold", 0.15),
        auto_off_seconds=defaults.get("auto_off_seconds", 30),
        clip_duration_seconds=defaults.get("clip_duration_seconds", 3),
        health_port=defaults.get("health_port", 8080),
    )
