"""Tests for cross-camera event consolidation."""

import time

from src.event_consolidator import (
    SAFETY_GROUPS,
    ConsolidatedEpisode,
    EventConsolidator,
)


class TestConsolidatedEpisode:
    def test_defaults(self):
        ep = ConsolidatedEpisode(group="dog_bark")
        assert ep.group == "dog_bark"
        assert ep.cameras == []
        assert ep.max_confidence == 0.0
        assert ep.detection_count == 0
        assert ep.published is False


class TestEventConsolidatorBasic:
    def test_single_detection_publishes(self):
        """Single detection on one camera should publish."""
        published = []
        consolidator = EventConsolidator(
            window_seconds=5.0,
            on_consolidated=lambda g, ep: published.append((g, ep)),
        )

        now = time.monotonic()
        consolidator.report_detection("cam1", "dog_bark", 0.85, now)

        assert len(published) == 1
        group, ep = published[0]
        assert group == "dog_bark"
        assert ep.cameras == ["cam1"]
        assert ep.max_confidence == 0.85
        assert ep.detection_count == 1

    def test_no_callback_does_not_crash(self):
        """Consolidator with no callback should not raise."""
        consolidator = EventConsolidator(window_seconds=5.0)
        consolidator.report_detection("cam1", "dog_bark", 0.85, time.monotonic())


class TestMultiCameraMerge:
    def test_two_cameras_within_window(self):
        """Detections from two cameras within window should merge."""
        published = []
        consolidator = EventConsolidator(
            window_seconds=5.0,
            on_consolidated=lambda g, ep: published.append((g, ep)),
        )

        now = time.monotonic()
        consolidator.report_detection("cam1", "dog_bark", 0.85, now)
        consolidator.report_detection("cam2", "dog_bark", 0.90, now + 1.0)

        # Should have published twice (initial + update with new camera)
        assert len(published) == 2
        _, last_ep = published[-1]
        assert set(last_ep.cameras) == {"cam1", "cam2"}
        assert last_ep.max_confidence == 0.90
        assert last_ep.detection_count == 2

    def test_two_cameras_outside_window(self):
        """Detections outside window should create separate episodes."""
        published = []
        consolidator = EventConsolidator(
            window_seconds=5.0,
            on_consolidated=lambda g, ep: published.append((g, ep)),
        )

        now = time.monotonic()
        consolidator.report_detection("cam1", "dog_bark", 0.85, now)
        consolidator.report_detection("cam2", "dog_bark", 0.90, now + 10.0)

        assert len(published) == 2
        # Each should have only one camera
        assert published[0][1].cameras == ["cam1"]
        assert published[1][1].cameras == ["cam2"]

    def test_three_cameras_within_window(self):
        """Three cameras within window should all merge."""
        published = []
        consolidator = EventConsolidator(
            window_seconds=5.0,
            on_consolidated=lambda g, ep: published.append((g, ep)),
        )

        now = time.monotonic()
        consolidator.report_detection("cam1", "dog_bark", 0.80, now)
        consolidator.report_detection("cam2", "dog_bark", 0.85, now + 1.0)
        consolidator.report_detection("cam3", "dog_bark", 0.90, now + 2.0)

        _, last_ep = published[-1]
        assert set(last_ep.cameras) == {"cam1", "cam2", "cam3"}
        assert last_ep.max_confidence == 0.90
        assert last_ep.detection_count == 3


class TestSafetyBypass:
    def test_safety_groups_publish_immediately(self):
        """Safety-critical groups should fire immediately."""
        for group in SAFETY_GROUPS:
            published = []
            consolidator = EventConsolidator(
                window_seconds=5.0,
                on_consolidated=lambda g, ep, _p=published: _p.append((g, ep)),
            )

            consolidator.report_detection("cam1", group, 0.95, time.monotonic())

            assert len(published) == 1, f"Safety group {group} was not published immediately"
            assert published[0][1].published is True


class TestDurationTracking:
    def test_duration_from_trigger_times(self):
        """Duration should be computed from trigger timestamps."""
        published = []
        consolidator = EventConsolidator(
            window_seconds=5.0,
            on_consolidated=lambda g, ep: published.append((g, ep)),
        )

        now = time.monotonic()
        consolidator.report_detection("cam1", "dog_bark", 0.85, now)
        consolidator.report_detection("cam2", "dog_bark", 0.90, now + 2.5)

        _, ep = published[-1]
        duration = ep.last_detected - ep.first_detected
        assert abs(duration - 2.5) < 0.01


class TestMaxConfidence:
    def test_max_confidence_tracks_highest(self):
        """max_confidence should track the highest across all cameras."""
        published = []
        consolidator = EventConsolidator(
            window_seconds=5.0,
            on_consolidated=lambda g, ep: published.append((g, ep)),
        )

        now = time.monotonic()
        consolidator.report_detection("cam1", "dog_bark", 0.70, now)
        consolidator.report_detection("cam2", "dog_bark", 0.95, now + 1.0)
        consolidator.report_detection("cam3", "dog_bark", 0.80, now + 2.0)

        _, ep = published[-1]
        assert ep.max_confidence == 0.95


class TestDifferentGroups:
    def test_different_groups_separate_episodes(self):
        """Different label groups should create independent episodes."""
        published = []
        consolidator = EventConsolidator(
            window_seconds=5.0,
            on_consolidated=lambda g, ep: published.append((g, ep)),
        )

        now = time.monotonic()
        consolidator.report_detection("cam1", "dog_bark", 0.85, now)
        consolidator.report_detection("cam1", "speech", 0.70, now + 0.5)

        assert len(published) == 2
        groups = {p[0] for p in published}
        assert groups == {"dog_bark", "speech"}


class TestStaleCleanup:
    def test_cleanup_removes_old_episodes(self):
        """Stale episodes beyond auto_off + window should be cleaned up."""
        consolidator = EventConsolidator(
            window_seconds=5.0,
            auto_off_seconds=30,
        )

        # Use a trigger_time far in the past
        old_time = time.monotonic() - 100
        consolidator.report_detection("cam1", "dog_bark", 0.85, old_time)

        consolidator.cleanup_stale()

        # _episodes should be empty
        assert len(consolidator._episodes) == 0

    def test_cleanup_keeps_recent_episodes(self):
        """Recent episodes should survive cleanup."""
        consolidator = EventConsolidator(
            window_seconds=5.0,
            auto_off_seconds=30,
        )

        consolidator.report_detection("cam1", "dog_bark", 0.85, time.monotonic())
        consolidator.cleanup_stale()

        assert "dog_bark" in consolidator._episodes
        assert len(consolidator._episodes["dog_bark"]) == 1


class TestAutoOffSeconds:
    def test_consolidated_auto_off_is_base_plus_10(self):
        """Consolidated auto_off should be base + 10."""
        consolidator = EventConsolidator(auto_off_seconds=30)
        assert consolidator.auto_off_seconds == 40

    def test_consolidated_auto_off_custom(self):
        consolidator = EventConsolidator(auto_off_seconds=60)
        assert consolidator.auto_off_seconds == 70


class TestSameCameraMultipleDetections:
    def test_same_camera_within_window_increments_count(self):
        """Same camera detecting twice within window should increment count."""
        published = []
        consolidator = EventConsolidator(
            window_seconds=5.0,
            on_consolidated=lambda g, ep: published.append((g, ep)),
        )

        now = time.monotonic()
        consolidator.report_detection("cam1", "dog_bark", 0.85, now)
        consolidator.report_detection("cam1", "dog_bark", 0.90, now + 1.0)

        # Should only publish once (initial), not re-publish for same camera
        assert len(published) == 1, "Same-camera repeat should not re-publish"
        _, ep = published[0]
        assert ep.cameras == ["cam1"]  # Not duplicated
        assert ep.detection_count == 2
        assert ep.max_confidence == 0.90


class TestOutOfOrderTriggerTimes:
    def test_last_detected_uses_max_not_latest_reported(self):
        """Out-of-order trigger_times should not regress last_detected."""
        published = []
        consolidator = EventConsolidator(
            window_seconds=5.0,
            on_consolidated=lambda g, ep: published.append((g, ep)),
        )

        now = time.monotonic()
        # cam1 triggered later but classified first
        consolidator.report_detection("cam1", "dog_bark", 0.85, now + 1.0)
        # cam2 triggered earlier but classified second
        consolidator.report_detection("cam2", "dog_bark", 0.90, now)

        _, ep = published[-1]
        assert ep.last_detected == now + 1.0, (
            "last_detected should be max(trigger_times), not latest reported"
        )
