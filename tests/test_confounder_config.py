"""Tests for confounder config parsing."""

import tempfile

from src.classifier import ClassificationResult
from src.config import ConfounderConfig, load_config


class TestConfounderConfigParsing:
    """Test that confounders are correctly parsed from YAML config."""

    def _write_config(self, yaml_content: str) -> str:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            return f.name

    def test_camera_without_confounders(self):
        path = self._write_config("""
mqtt:
  host: localhost
cameras:
  - name: test_cam
    rtsp_url: rtsp://fake
""")
        config = load_config(path)
        assert config.cameras[0].confounders is None

    def test_camera_with_confounders(self):
        path = self._write_config("""
mqtt:
  host: localhost
cameras:
  - name: living_room
    rtsp_url: rtsp://fake
    confounders:
      - entity_id: media_player.tv
        active_when: "!off"
        confused_groups:
          - car_horn
          - siren
      - entity_id: sensor.furnace_power
        active_when: ">200"
        confused_groups:
          - kitchen_appliance
""")
        config = load_config(path)
        cam = config.cameras[0]
        assert cam.confounders is not None
        assert len(cam.confounders) == 2

        c1 = cam.confounders[0]
        assert isinstance(c1, ConfounderConfig)
        assert c1.entity_id == "media_player.tv"
        assert c1.active_when == "!off"
        assert c1.confused_groups == ["car_horn", "siren"]

        c2 = cam.confounders[1]
        assert c2.entity_id == "sensor.furnace_power"
        assert c2.active_when == ">200"
        assert c2.confused_groups == ["kitchen_appliance"]

    def test_mixed_cameras_with_and_without_confounders(self):
        path = self._write_config("""
mqtt:
  host: localhost
cameras:
  - name: living_room
    rtsp_url: rtsp://fake
    confounders:
      - entity_id: media_player.tv
        active_when: "!off"
        confused_groups:
          - car_horn
  - name: front_door
    rtsp_url: rtsp://fake2
""")
        config = load_config(path)
        assert config.cameras[0].confounders is not None
        assert len(config.cameras[0].confounders) == 1
        assert config.cameras[1].confounders is None


class TestClassificationResultConfounderFields:
    """Test confounder fields on ClassificationResult."""

    def test_default_not_confounded(self):
        r = ClassificationResult(
            label="Dog",
            group="dog_bark",
            confidence=0.85,
            top_5=[],
            db_level=-25.0,
        )
        assert r.confounded is False
        assert r.confounder_entity is None

    def test_confounded_to_dict(self):
        r = ClassificationResult(
            label="Car horn",
            group="car_horn",
            confidence=0.70,
            top_5=[],
            db_level=-20.0,
            confounded=True,
            confounder_entity="media_player.tv",
        )
        d = r.to_dict()
        assert d["confounded"] is True
        assert d["confounder"] == "media_player.tv"

    def test_not_confounded_to_dict_excludes_fields(self):
        r = ClassificationResult(
            label="Dog",
            group="dog_bark",
            confidence=0.85,
            top_5=[],
            db_level=-25.0,
        )
        d = r.to_dict()
        assert "confounded" not in d
        assert "confounder" not in d
