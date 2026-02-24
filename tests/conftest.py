"""Shared test fixtures for ast-audio-classifier."""

import pytest


@pytest.fixture()
def sample_config_dict():
    """Minimal valid configuration dictionary."""
    return {
        "mqtt": {"host": "localhost", "port": 1883},
        "cameras": [
            {
                "name": "test_camera",
                "rtsp_url": "rtsp://192.168.1.100:8554/test",
                "db_threshold": -35,
            }
        ],
    }
