"""Tests for confounder monitor — condition evaluation and state tracking."""

from src.config import CameraConfig, ConfounderConfig
from src.confounder_monitor import ConfounderMonitor, evaluate_condition


class TestEvaluateCondition:
    """Test the active_when condition evaluator."""

    def test_negate_off(self):
        """'!off' is True when state is not 'off'."""
        assert evaluate_condition("!off", "playing") is True
        assert evaluate_condition("!off", "idle") is True
        assert evaluate_condition("!off", "on") is True
        assert evaluate_condition("!off", "off") is False

    def test_negate_other(self):
        assert evaluate_condition("!standby", "on") is True
        assert evaluate_condition("!standby", "standby") is False

    def test_greater_than(self):
        assert evaluate_condition(">200", "250") is True
        assert evaluate_condition(">200", "200") is False
        assert evaluate_condition(">200", "150") is False

    def test_greater_than_equal(self):
        assert evaluate_condition(">=200", "200") is True
        assert evaluate_condition(">=200", "201") is True
        assert evaluate_condition(">=200", "199") is False

    def test_less_than(self):
        assert evaluate_condition("<50", "30") is True
        assert evaluate_condition("<50", "50") is False
        assert evaluate_condition("<50", "70") is False

    def test_less_than_equal(self):
        assert evaluate_condition("<=50", "50") is True
        assert evaluate_condition("<=50", "30") is True
        assert evaluate_condition("<=50", "51") is False

    def test_exact_match(self):
        assert evaluate_condition("playing", "playing") is True
        assert evaluate_condition("playing", "paused") is False
        assert evaluate_condition("on", "on") is True
        assert evaluate_condition("on", "off") is False

    def test_unavailable_always_false(self):
        """Unavailable/unknown entities should never match any condition."""
        assert evaluate_condition("!off", "unavailable") is False
        assert evaluate_condition(">200", "unavailable") is False
        assert evaluate_condition("playing", "unavailable") is False
        assert evaluate_condition("!off", "unknown") is False
        assert evaluate_condition(">200", "unknown") is False

    def test_numeric_with_non_numeric_state(self):
        """Non-numeric state for numeric conditions should return False."""
        assert evaluate_condition(">200", "playing") is False
        assert evaluate_condition(">=100", "on") is False
        assert evaluate_condition("<50", "off") is False

    def test_float_comparison(self):
        assert evaluate_condition(">200.5", "201") is True
        assert evaluate_condition(">200.5", "200") is False
        assert evaluate_condition(">200", "200.1") is True


class TestConfounderMonitor:
    """Test ConfounderMonitor state tracking (without real HA API)."""

    def _make_camera(
        self,
        name: str = "living_room",
        confounders: list[ConfounderConfig] | None = None,
    ) -> CameraConfig:
        return CameraConfig(
            name=name,
            rtsp_url="rtsp://fake",
            confounders=confounders,
        )

    def _make_monitor(self, cameras: list[CameraConfig]) -> ConfounderMonitor:
        """Create a monitor with no SUPERVISOR_TOKEN (standalone mode)."""
        monitor = ConfounderMonitor(cameras, poll_interval=10.0)
        return monitor

    def test_no_confounders_returns_empty(self):
        cam = self._make_camera()
        monitor = self._make_monitor([cam])
        assert monitor.get_confused_groups("living_room") == frozenset()
        assert monitor.get_active_confounders("living_room") == []

    def test_unknown_camera_returns_empty(self):
        cam = self._make_camera()
        monitor = self._make_monitor([cam])
        assert monitor.get_confused_groups("nonexistent") == frozenset()

    def test_inactive_confounder_returns_empty(self):
        """When entity state doesn't match condition, no groups are confused."""
        conf = ConfounderConfig(
            entity_id="media_player.tv",
            active_when="!off",
            confused_groups=["car_horn", "siren"],
        )
        cam = self._make_camera(confounders=[conf])
        monitor = self._make_monitor([cam])
        # Default state is "unavailable" (not polled yet) — should not match "!off"
        assert monitor.get_confused_groups("living_room") == frozenset()

    def test_active_confounder_returns_groups(self):
        """When entity state matches condition, confused groups are returned."""
        conf = ConfounderConfig(
            entity_id="media_player.tv",
            active_when="!off",
            confused_groups=["car_horn", "siren"],
        )
        cam = self._make_camera(confounders=[conf])
        monitor = self._make_monitor([cam])
        # Simulate state being "playing"
        monitor._entity_states["media_player.tv"] = "playing"
        assert monitor.get_confused_groups("living_room") == frozenset(
            {"car_horn", "siren"}
        )

    def test_multiple_confounders_union(self):
        """Multiple active confounders should union their confused groups."""
        conf1 = ConfounderConfig(
            entity_id="media_player.tv",
            active_when="!off",
            confused_groups=["car_horn", "siren"],
        )
        conf2 = ConfounderConfig(
            entity_id="sensor.furnace_power",
            active_when=">200",
            confused_groups=["kitchen_appliance", "car_horn"],
        )
        cam = self._make_camera(confounders=[conf1, conf2])
        monitor = self._make_monitor([cam])
        monitor._entity_states["media_player.tv"] = "playing"
        monitor._entity_states["sensor.furnace_power"] = "350"
        assert monitor.get_confused_groups("living_room") == frozenset(
            {"car_horn", "siren", "kitchen_appliance"}
        )

    def test_partial_active_confounders(self):
        """Only active confounders contribute to confused groups."""
        conf1 = ConfounderConfig(
            entity_id="media_player.tv",
            active_when="!off",
            confused_groups=["car_horn", "siren"],
        )
        conf2 = ConfounderConfig(
            entity_id="sensor.furnace_power",
            active_when=">200",
            confused_groups=["kitchen_appliance"],
        )
        cam = self._make_camera(confounders=[conf1, conf2])
        monitor = self._make_monitor([cam])
        monitor._entity_states["media_player.tv"] = "playing"
        monitor._entity_states["sensor.furnace_power"] = "50"  # Not active
        assert monitor.get_confused_groups("living_room") == frozenset(
            {"car_horn", "siren"}
        )

    def test_get_confounder_context_returns_match(self):
        conf = ConfounderConfig(
            entity_id="media_player.tv",
            active_when="!off",
            confused_groups=["car_horn", "siren"],
        )
        cam = self._make_camera(confounders=[conf])
        monitor = self._make_monitor([cam])
        monitor._entity_states["media_player.tv"] = "playing"
        ctx = monitor.get_confounder_context("living_room", "car_horn")
        assert ctx is not None
        assert ctx["entity_id"] == "media_player.tv"
        assert ctx["state"] == "playing"
        assert ctx["active_when"] == "!off"

    def test_get_confounder_context_returns_none_when_not_confused(self):
        conf = ConfounderConfig(
            entity_id="media_player.tv",
            active_when="!off",
            confused_groups=["car_horn", "siren"],
        )
        cam = self._make_camera(confounders=[conf])
        monitor = self._make_monitor([cam])
        monitor._entity_states["media_player.tv"] = "playing"
        # dog_bark is not in confused_groups
        assert monitor.get_confounder_context("living_room", "dog_bark") is None

    def test_get_confounder_context_returns_none_when_inactive(self):
        conf = ConfounderConfig(
            entity_id="media_player.tv",
            active_when="!off",
            confused_groups=["car_horn"],
        )
        cam = self._make_camera(confounders=[conf])
        monitor = self._make_monitor([cam])
        monitor._entity_states["media_player.tv"] = "off"
        assert monitor.get_confounder_context("living_room", "car_horn") is None

    def test_standalone_mode_not_available(self):
        """Without SUPERVISOR_TOKEN, monitor should report not available."""
        cam = self._make_camera(confounders=[
            ConfounderConfig(
                entity_id="media_player.tv",
                active_when="!off",
                confused_groups=["car_horn"],
            )
        ])
        monitor = self._make_monitor([cam])
        assert monitor.available is False

    def test_entity_ids_collected(self):
        """Monitor should collect unique entity IDs from all cameras."""
        conf1 = ConfounderConfig(
            entity_id="media_player.tv",
            active_when="!off",
            confused_groups=["car_horn"],
        )
        conf2 = ConfounderConfig(
            entity_id="sensor.furnace_power",
            active_when=">200",
            confused_groups=["kitchen_appliance"],
        )
        cam1 = self._make_camera(name="cam1", confounders=[conf1])
        cam2 = self._make_camera(name="cam2", confounders=[conf2])
        cam3 = self._make_camera(name="cam3")  # No confounders
        monitor = self._make_monitor([cam1, cam2, cam3])
        assert monitor._entity_ids == {"media_player.tv", "sensor.furnace_power"}


class TestConfounderMonitorGetActiveConfounders:
    """Test get_active_confounders returns correct ConfounderConfig objects."""

    def test_returns_active_confounder_objects(self):
        conf = ConfounderConfig(
            entity_id="media_player.tv",
            active_when="!off",
            confused_groups=["car_horn", "siren"],
        )
        cam = CameraConfig(name="lr", rtsp_url="rtsp://fake", confounders=[conf])
        monitor = ConfounderMonitor([cam])
        monitor._entity_states["media_player.tv"] = "playing"
        active = monitor.get_active_confounders("lr")
        assert len(active) == 1
        assert active[0].entity_id == "media_player.tv"

    def test_returns_empty_when_none_active(self):
        conf = ConfounderConfig(
            entity_id="media_player.tv",
            active_when="!off",
            confused_groups=["car_horn"],
        )
        cam = CameraConfig(name="lr", rtsp_url="rtsp://fake", confounders=[conf])
        monitor = ConfounderMonitor([cam])
        monitor._entity_states["media_player.tv"] = "off"
        assert monitor.get_active_confounders("lr") == []
