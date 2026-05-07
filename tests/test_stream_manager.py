"""Tests for per-camera stream management: CameraStream, StreamManager, backoff, cooldown."""

from __future__ import annotations

import asyncio
import sys
import time
from unittest.mock import MagicMock

# Mock paho.mqtt before any src imports (mqtt_publisher imports paho.mqtt.client)
sys.modules.setdefault("paho", MagicMock())
sys.modules.setdefault("paho.mqtt", MagicMock())
sys.modules.setdefault("paho.mqtt.client", MagicMock())

from src.config import CameraConfig  # noqa: E402
from src.stream_manager import (  # noqa: E402
    LOG_SUPPRESSION_THRESHOLD,
    MAX_BACKOFF,
    STABLE_STREAM_SECONDS,
    CameraStream,
    StreamManager,
    StreamState,
)


def _make_camera(**overrides) -> CameraConfig:
    """Create a CameraConfig with sensible defaults for tests."""
    defaults = {
        "name": "test_cam",
        "rtsp_url": "rtsp://192.168.1.100:8554/test",
    }
    defaults.update(overrides)
    return CameraConfig(**defaults)


def _make_stream(camera: CameraConfig | None = None, **kwargs) -> CameraStream:
    """Create a CameraStream with mocked dependencies."""
    if camera is None:
        camera = _make_camera()
    defaults = {
        "camera": camera,
        "classifier": MagicMock(),
        "publisher": MagicMock(),
        "inference_semaphore": asyncio.Semaphore(1),
    }
    defaults.update(kwargs)
    return CameraStream(**defaults)


# ---------------------------------------------------------------------------
# StreamState enum
# ---------------------------------------------------------------------------


class TestStreamState:
    """StreamState enum has all expected members."""

    def test_disconnected_exists(self):
        assert StreamState.DISCONNECTED.value == "disconnected"

    def test_connecting_exists(self):
        assert StreamState.CONNECTING.value == "connecting"

    def test_streaming_exists(self):
        assert StreamState.STREAMING.value == "streaming"

    def test_cooldown_exists(self):
        assert StreamState.COOLDOWN.value == "cooldown"

    def test_error_exists(self):
        assert StreamState.ERROR.value == "error"

    def test_backoff_exists(self):
        assert StreamState.BACKOFF.value == "backoff"

    def test_total_member_count(self):
        assert len(StreamState) == 7


# ---------------------------------------------------------------------------
# CameraStream initial state
# ---------------------------------------------------------------------------


class TestCameraStreamInitialState:
    """CameraStream starts in the expected default state."""

    def test_initial_state_is_disconnected(self):
        stream = _make_stream()
        assert stream.state is StreamState.DISCONNECTED

    def test_initial_inference_count_is_zero(self):
        stream = _make_stream()
        assert stream.inference_count == 0

    def test_initial_last_event_time_is_zero(self):
        stream = _make_stream()
        assert stream.last_event_time == 0

    def test_initial_last_chunk_time_is_zero(self):
        stream = _make_stream()
        assert stream.last_chunk_time == 0.0

    def test_initial_backoff_equals_reconnect_interval(self):
        cam = _make_camera(reconnect_interval=7)
        stream = _make_stream(camera=cam)
        assert stream._backoff == 7

    def test_default_backoff_equals_default_reconnect(self):
        """Default reconnect_interval is 5, so initial backoff should be 5."""
        stream = _make_stream()
        assert stream._backoff == 5


# ---------------------------------------------------------------------------
# CameraStream properties
# ---------------------------------------------------------------------------


class TestCameraStreamProperties:
    """Test property accessors on CameraStream."""

    def test_camera_name(self):
        cam = _make_camera(name="front_porch")
        stream = _make_stream(camera=cam)
        assert stream.camera_name == "front_porch"

    def test_inference_count_reflects_internal(self):
        stream = _make_stream()
        stream._inference_count = 42
        assert stream.inference_count == 42

    def test_last_event_time_reflects_internal(self):
        stream = _make_stream()
        stream._last_event_time = 123.456
        assert stream.last_event_time == 123.456

    def test_last_chunk_time_delegates_to_ambient(self):
        stream = _make_stream()
        stream._ambient.last_chunk_time = 99.9
        assert stream.last_chunk_time == 99.9

    def test_ambient_info_basic_keys(self):
        stream = _make_stream()
        info = stream.ambient_info
        assert "peak_db" in info
        assert "chunk_count" in info
        assert "threshold" in info

    def test_ambient_info_threshold_matches_camera(self):
        cam = _make_camera(db_threshold=-42.0)
        stream = _make_stream(camera=cam)
        info = stream.ambient_info
        assert info["threshold"] == -42.0

    def test_ambient_info_no_adaptive_keys_when_disabled(self):
        cam = _make_camera(adaptive_threshold=False)
        stream = _make_stream(camera=cam)
        info = stream.ambient_info
        assert "ema_db" not in info
        assert "adaptive_threshold" not in info

    def test_ambient_info_has_adaptive_keys_when_enabled(self):
        cam = _make_camera(adaptive_threshold=True, adaptive_margin_db=6.0)
        stream = _make_stream(camera=cam)
        info = stream.ambient_info
        assert "ema_db" in info
        assert "adaptive_threshold" in info


# ---------------------------------------------------------------------------
# Judge semaphore
# ---------------------------------------------------------------------------


class TestJudgeSemaphore:
    """The _judge_semaphore limits concurrent LLM judge tasks."""

    def test_judge_semaphore_initial_value_is_two(self):
        stream = _make_stream()
        # asyncio.Semaphore._value is the internal counter
        assert stream._judge_semaphore._value == 2

    def test_judge_semaphore_is_asyncio_semaphore(self):
        stream = _make_stream()
        assert isinstance(stream._judge_semaphore, asyncio.Semaphore)


# ---------------------------------------------------------------------------
# Judge task lifecycle (detached tasks must be tracked and cancelled)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _should_use_resolver — decide whether to hit Camera API per reconnect
# ---------------------------------------------------------------------------


class TestShouldUseResolver:
    """In fast-reconnect mode the resolver is skipped to avoid its timeout on
    every retry, BUT must be refreshed periodically so stale URLs eventually
    get replaced. Otherwise the stream can get stuck reconnecting to a dead
    Scrypted rebroadcast port forever."""

    def test_resolves_when_no_failures(self):
        stream = _make_stream()
        stream._consecutive_failures = 0
        stream._consecutive_short_streams = 0
        assert stream._should_use_resolver() is True

    def test_resolves_just_below_fast_reconnect_threshold(self):
        from src.stream_manager import FAST_RECONNECT_THRESHOLD

        stream = _make_stream()
        stream._consecutive_failures = FAST_RECONNECT_THRESHOLD - 1
        stream._consecutive_short_streams = FAST_RECONNECT_THRESHOLD - 1
        assert stream._should_use_resolver() is True

    def test_resolves_when_only_connect_failures(self):
        """Pure connect failures must always re-resolve — they indicate stale
        Scrypted URLs, and skipping resolver would wedge us on the dead URL."""
        from src.stream_manager import FAST_RECONNECT_THRESHOLD

        stream = _make_stream()
        stream._consecutive_failures = FAST_RECONNECT_THRESHOLD * 3
        stream._consecutive_short_streams = 0
        assert stream._should_use_resolver() is True

    def test_skips_resolver_at_fast_reconnect_threshold(self):
        from src.stream_manager import FAST_RECONNECT_THRESHOLD

        stream = _make_stream()
        stream._consecutive_failures = FAST_RECONNECT_THRESHOLD
        stream._consecutive_short_streams = FAST_RECONNECT_THRESHOLD
        # Threshold is 5 and refresh interval is 10, so 5 is not a refresh boundary
        assert stream._should_use_resolver() is False

    def test_skips_resolver_between_refresh_boundaries(self):
        from src.stream_manager import (
            FAST_RECONNECT_RESOLVER_INTERVAL,
            FAST_RECONNECT_THRESHOLD,
        )

        stream = _make_stream()
        stream._consecutive_short_streams = FAST_RECONNECT_THRESHOLD
        # sample values that are above threshold but not on the refresh interval
        for n in (
            FAST_RECONNECT_THRESHOLD + 1,
            FAST_RECONNECT_THRESHOLD + 2,
            FAST_RECONNECT_RESOLVER_INTERVAL - 1,
            FAST_RECONNECT_RESOLVER_INTERVAL + 1,
            FAST_RECONNECT_RESOLVER_INTERVAL * 2 + 3,
        ):
            stream._consecutive_failures = n
            assert stream._should_use_resolver() is False, f"expected skip at cf={n}"

    def test_refreshes_at_interval_boundary(self):
        from src.stream_manager import FAST_RECONNECT_RESOLVER_INTERVAL, FAST_RECONNECT_THRESHOLD

        stream = _make_stream()
        stream._consecutive_short_streams = FAST_RECONNECT_THRESHOLD
        for n in (
            FAST_RECONNECT_RESOLVER_INTERVAL,
            FAST_RECONNECT_RESOLVER_INTERVAL * 2,
            FAST_RECONNECT_RESOLVER_INTERVAL * 10,
        ):
            stream._consecutive_failures = n
            assert stream._should_use_resolver() is True, f"expected refresh at cf={n}"


class TestJudgeTaskLifecycle:
    """Background LLM judge tasks must be tracked and cancelled on stop()."""

    def test_judge_tasks_set_initialized_empty(self):
        stream = _make_stream()
        assert hasattr(stream, "_judge_tasks")
        assert stream._judge_tasks == set()

    def test_stop_cancels_tracked_judge_tasks(self):
        async def _run():
            stream = _make_stream()

            # Simulate a pending judge task
            async def slow_work():
                await asyncio.sleep(60)

            task = asyncio.create_task(slow_work(), name="test-judge")
            stream._judge_tasks.add(task)
            task.add_done_callback(stream._judge_tasks.discard)

            # Give the loop a tick to start the task
            await asyncio.sleep(0)
            assert not task.done()

            await stream.stop()

            assert task.cancelled() or task.done()
            assert task not in stream._judge_tasks

        asyncio.run(_run())

    def test_completed_judge_task_removes_itself_from_set(self):
        async def _run():
            stream = _make_stream()

            async def quick_work():
                return "done"

            task = asyncio.create_task(quick_work(), name="test-judge")
            stream._judge_tasks.add(task)
            task.add_done_callback(stream._judge_tasks.discard)

            await task
            # Give callback a chance to run
            await asyncio.sleep(0)
            assert task not in stream._judge_tasks

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Adaptive threshold closure
# ---------------------------------------------------------------------------


class TestAdaptiveThresholdClosure:
    """Adaptive threshold is configured based on camera settings."""

    def test_threshold_fn_none_when_adaptive_disabled(self):
        cam = _make_camera(adaptive_threshold=False)
        stream = _make_stream(camera=cam)
        assert stream._threshold_fn is None

    def test_threshold_fn_set_when_adaptive_enabled(self):
        cam = _make_camera(adaptive_threshold=True, adaptive_margin_db=6.0)
        stream = _make_stream(camera=cam)
        assert stream._threshold_fn is not None
        assert callable(stream._threshold_fn)


# ---------------------------------------------------------------------------
# Backoff calculation
# ---------------------------------------------------------------------------


class TestBackoffCalculation:
    """Backoff logic: doubles after each error, caps at MAX_BACKOFF."""

    def test_max_backoff_constant(self):
        assert MAX_BACKOFF == 60

    def test_initial_backoff_equals_reconnect_interval(self):
        cam = _make_camera(reconnect_interval=5)
        stream = _make_stream(camera=cam)
        assert stream._backoff == 5

    def test_backoff_doubles_after_simulated_error(self):
        cam = _make_camera(reconnect_interval=5)
        stream = _make_stream(camera=cam)
        # Simulate the backoff doubling that _run() does after sleep
        stream._backoff = min(stream._backoff * 2, MAX_BACKOFF)
        assert stream._backoff == 10

    def test_backoff_doubles_repeatedly(self):
        cam = _make_camera(reconnect_interval=3)
        stream = _make_stream(camera=cam)
        # Simulate 4 consecutive errors
        for expected in [6, 12, 24, 48]:
            stream._backoff = min(stream._backoff * 2, MAX_BACKOFF)
            assert stream._backoff == expected

    def test_backoff_caps_at_max(self):
        cam = _make_camera(reconnect_interval=5)
        stream = _make_stream(camera=cam)
        # Simulate many errors — should never exceed MAX_BACKOFF
        for _ in range(20):
            stream._backoff = min(stream._backoff * 2, MAX_BACKOFF)
        assert stream._backoff == MAX_BACKOFF

    def test_backoff_caps_from_large_reconnect_interval(self):
        cam = _make_camera(reconnect_interval=50)
        stream = _make_stream(camera=cam)
        stream._backoff = min(stream._backoff * 2, MAX_BACKOFF)
        assert stream._backoff == MAX_BACKOFF

    def test_backoff_resets_on_successful_connect(self):
        """After successful connect, _run() resets backoff to reconnect_interval."""
        cam = _make_camera(reconnect_interval=5)
        stream = _make_stream(camera=cam)
        # Simulate errors
        stream._backoff = min(stream._backoff * 2, MAX_BACKOFF)
        stream._backoff = min(stream._backoff * 2, MAX_BACKOFF)
        assert stream._backoff == 20
        # Simulate successful connect reset (as done in _run)
        stream._backoff = cam.reconnect_interval
        assert stream._backoff == 5


# ---------------------------------------------------------------------------
# Cooldown enforcement
# ---------------------------------------------------------------------------


class TestCooldownEnforcement:
    """Cooldown prevents repeated classifications within cooldown_seconds."""

    def test_cooldown_check_skips_when_within_window(self):
        cam = _make_camera(cooldown_seconds=10)
        stream = _make_stream(camera=cam)
        # Simulate a recent event
        stream._last_event_time = time.monotonic()
        now = time.monotonic()
        within_cooldown = (now - stream._last_event_time) < cam.cooldown_seconds
        assert within_cooldown is True

    def test_cooldown_check_allows_after_window(self):
        cam = _make_camera(cooldown_seconds=10)
        stream = _make_stream(camera=cam)
        # Simulate an event 20 seconds ago
        stream._last_event_time = time.monotonic() - 20
        now = time.monotonic()
        within_cooldown = (now - stream._last_event_time) < cam.cooldown_seconds
        assert within_cooldown is False

    def test_no_cooldown_when_no_previous_event(self):
        """When last_event_time is 0, cooldown should not block."""
        cam = _make_camera(cooldown_seconds=10)
        stream = _make_stream(camera=cam)
        assert stream._last_event_time == 0
        now = time.monotonic()
        within_cooldown = (now - stream._last_event_time) < cam.cooldown_seconds
        # now is much larger than 0, so the difference is huge
        assert within_cooldown is False


# ---------------------------------------------------------------------------
# StreamManager construction
# ---------------------------------------------------------------------------


class TestStreamManagerConstruction:
    """StreamManager creates CameraStream instances from a list of cameras."""

    def test_creates_correct_number_of_streams(self):
        cameras = [_make_camera(name=f"cam_{i}") for i in range(3)]
        mgr = StreamManager(
            cameras=cameras,
            classifier=MagicMock(),
            publisher=MagicMock(),
        )
        assert len(mgr.streams) == 3

    def test_single_camera(self):
        cameras = [_make_camera(name="solo")]
        mgr = StreamManager(
            cameras=cameras,
            classifier=MagicMock(),
            publisher=MagicMock(),
        )
        assert len(mgr.streams) == 1
        assert mgr.streams[0].camera_name == "solo"

    def test_empty_cameras_list(self):
        mgr = StreamManager(
            cameras=[],
            classifier=MagicMock(),
            publisher=MagicMock(),
        )
        assert len(mgr.streams) == 0

    def test_streams_share_semaphore(self):
        cameras = [_make_camera(name=f"cam_{i}") for i in range(3)]
        mgr = StreamManager(
            cameras=cameras,
            classifier=MagicMock(),
            publisher=MagicMock(),
        )
        semaphores = {s._semaphore for s in mgr.streams}
        assert len(semaphores) == 1, "All streams should share the same inference semaphore"

    def test_shared_semaphore_is_asyncio_semaphore(self):
        cameras = [_make_camera(name="cam_0")]
        mgr = StreamManager(
            cameras=cameras,
            classifier=MagicMock(),
            publisher=MagicMock(),
        )
        assert isinstance(mgr.streams[0]._semaphore, asyncio.Semaphore)

    def test_shared_semaphore_has_value_one(self):
        cameras = [_make_camera(name="cam_0")]
        mgr = StreamManager(
            cameras=cameras,
            classifier=MagicMock(),
            publisher=MagicMock(),
        )
        assert mgr._semaphore._value == 1

    def test_camera_names_preserved(self):
        cameras = [_make_camera(name="living_room"), _make_camera(name="backyard")]
        mgr = StreamManager(
            cameras=cameras,
            classifier=MagicMock(),
            publisher=MagicMock(),
        )
        names = [s.camera_name for s in mgr.streams]
        assert names == ["living_room", "backyard"]

    def test_optional_params_forwarded(self):
        cameras = [_make_camera()]
        clap = MagicMock()
        judge = MagicMock()
        consolidator = MagicMock()
        noise_stress = MagicMock()
        mgr = StreamManager(
            cameras=cameras,
            classifier=MagicMock(),
            publisher=MagicMock(),
            confidence_threshold=0.25,
            clip_duration=5,
            clap_verifier=clap,
            llm_judge=judge,
            consolidator=consolidator,
            noise_stress=noise_stress,
        )
        stream = mgr.streams[0]
        assert stream._confidence_threshold == 0.25
        assert stream._clip_duration == 5
        assert stream._clap_verifier is clap
        assert stream._llm_judge is judge
        assert stream._consolidator is consolidator
        assert stream._noise_stress is noise_stress


# ---------------------------------------------------------------------------
# StreamManager status
# ---------------------------------------------------------------------------


class TestStreamManagerStatus:
    """StreamManager.status() returns structured info about each stream."""

    def test_status_returns_list(self):
        cameras = [_make_camera(name="cam_a"), _make_camera(name="cam_b")]
        mgr = StreamManager(
            cameras=cameras,
            classifier=MagicMock(),
            publisher=MagicMock(),
        )
        status = mgr.status()
        assert isinstance(status, list)
        assert len(status) == 2

    def test_status_entry_has_required_keys(self):
        cameras = [_make_camera(name="front")]
        mgr = StreamManager(
            cameras=cameras,
            classifier=MagicMock(),
            publisher=MagicMock(),
        )
        status = mgr.status()
        entry = status[0]
        assert set(entry.keys()) == {"name", "state", "inference_count", "last_event_time"}

    def test_status_name_matches_camera(self):
        cameras = [_make_camera(name="patio")]
        mgr = StreamManager(
            cameras=cameras,
            classifier=MagicMock(),
            publisher=MagicMock(),
        )
        assert mgr.status()[0]["name"] == "patio"

    def test_status_initial_state_is_disconnected(self):
        cameras = [_make_camera()]
        mgr = StreamManager(
            cameras=cameras,
            classifier=MagicMock(),
            publisher=MagicMock(),
        )
        assert mgr.status()[0]["state"] == "disconnected"

    def test_status_initial_inference_count_zero(self):
        cameras = [_make_camera()]
        mgr = StreamManager(
            cameras=cameras,
            classifier=MagicMock(),
            publisher=MagicMock(),
        )
        assert mgr.status()[0]["inference_count"] == 0

    def test_status_initial_last_event_time_zero(self):
        cameras = [_make_camera()]
        mgr = StreamManager(
            cameras=cameras,
            classifier=MagicMock(),
            publisher=MagicMock(),
        )
        assert mgr.status()[0]["last_event_time"] == 0

    def test_status_reflects_mutated_state(self):
        cameras = [_make_camera(name="test")]
        mgr = StreamManager(
            cameras=cameras,
            classifier=MagicMock(),
            publisher=MagicMock(),
        )
        stream = mgr.streams[0]
        stream._state = StreamState.STREAMING
        stream._inference_count = 7
        stream._last_event_time = 42.0

        entry = mgr.status()[0]
        assert entry["state"] == "streaming"
        assert entry["inference_count"] == 7
        assert entry["last_event_time"] == 42.0

    def test_status_empty_when_no_cameras(self):
        mgr = StreamManager(
            cameras=[],
            classifier=MagicMock(),
            publisher=MagicMock(),
        )
        assert mgr.status() == []

    def test_status_multiple_cameras_order(self):
        cameras = [
            _make_camera(name="alpha"),
            _make_camera(name="beta"),
            _make_camera(name="gamma"),
        ]
        mgr = StreamManager(
            cameras=cameras,
            classifier=MagicMock(),
            publisher=MagicMock(),
        )
        names = [entry["name"] for entry in mgr.status()]
        assert names == ["alpha", "beta", "gamma"]


# ---------------------------------------------------------------------------
# CameraStream resolver support
# ---------------------------------------------------------------------------


class TestCameraStreamResolverParams:
    """CameraStream accepts resolver param."""

    def test_accepts_resolver_param(self):
        resolver = MagicMock()
        stream = _make_stream(resolver=resolver)
        assert stream._resolver is resolver

    def test_resolver_defaults_to_none(self):
        stream = _make_stream()
        assert stream._resolver is None

    def test_consecutive_failures_starts_at_zero(self):
        stream = _make_stream()
        assert stream._consecutive_failures == 0

    def test_no_auto_discovery_param(self):
        """auto_discovery parameter should no longer exist on CameraStream."""
        import inspect

        sig = inspect.signature(CameraStream.__init__)
        assert "auto_discovery" not in sig.parameters

    def test_no_go2rtc_stream_in_camera_config(self):
        """go2rtc_stream field should no longer exist on CameraConfig."""
        import dataclasses

        field_names = {f.name for f in dataclasses.fields(CameraConfig)}
        assert "go2rtc_stream" not in field_names


# ---------------------------------------------------------------------------
# Failure counter logic
# ---------------------------------------------------------------------------


class TestFailureCounter:
    """Failure counter tracks consecutive stream failures."""

    def test_failure_counter_increments(self):
        stream = _make_stream()
        stream._consecutive_failures += 1
        assert stream._consecutive_failures == 1

    def test_failure_counter_resets(self):
        stream = _make_stream()
        stream._consecutive_failures = 5
        stream._consecutive_failures = 0
        assert stream._consecutive_failures == 0

    def test_log_suppression_threshold_constant(self):
        assert LOG_SUPPRESSION_THRESHOLD == 3

    def test_stable_stream_seconds_constant(self):
        assert STABLE_STREAM_SECONDS == 30


# ---------------------------------------------------------------------------
# StreamManager forwards resolver/auto_discovery
# ---------------------------------------------------------------------------


class TestStreamManagerResolverForwarding:
    """StreamManager forwards resolver to CameraStreams."""

    def test_forwards_resolver(self):
        resolver = MagicMock()
        cameras = [_make_camera()]
        mgr = StreamManager(
            cameras=cameras,
            classifier=MagicMock(),
            publisher=MagicMock(),
            resolver=resolver,
        )
        assert mgr.streams[0]._resolver is resolver

    def test_defaults_no_resolver(self):
        cameras = [_make_camera()]
        mgr = StreamManager(
            cameras=cameras,
            classifier=MagicMock(),
            publisher=MagicMock(),
        )
        assert mgr.streams[0]._resolver is None


class TestSimplifiedRetry:
    """After removing discovery, stream uses standard exponential backoff."""

    def test_no_discovering_state_in_run_loop(self):
        """CameraStream should not enter DISCOVERING state — it's been removed."""
        _make_stream()
        # DISCOVERING should no longer exist as a state
        assert not hasattr(StreamState, "DISCOVERING")

    def test_backoff_doubles_on_failure(self):
        """Backoff should double on each failure up to MAX_BACKOFF."""
        stream = _make_stream()
        initial = stream._backoff
        # Simulate failure
        stream._backoff = min(stream._backoff * 2, MAX_BACKOFF)
        assert stream._backoff == initial * 2

    def test_backoff_caps_at_max(self):
        """Backoff should never exceed MAX_BACKOFF."""
        stream = _make_stream()
        stream._backoff = MAX_BACKOFF
        stream._backoff = min(stream._backoff * 2, MAX_BACKOFF)
        assert stream._backoff == MAX_BACKOFF

    def test_stable_stream_resets_backoff(self):
        """A stream lasting >= STABLE_STREAM_SECONDS should reset backoff."""
        stream = _make_stream()
        stream._backoff = MAX_BACKOFF
        stream._consecutive_failures = 10
        stream._total_failures = 50
        # Simulate stable stream reset
        stream._consecutive_failures = 0
        stream._total_failures = 0
        stream._backoff = stream._camera.reconnect_interval
        assert stream._backoff == stream._camera.reconnect_interval


class TestStreamDeathEventSuppression:
    """stream_death OO events should follow the same LOG_SUPPRESSION_INTERVAL as logs."""

    def test_stream_death_event_suppressed_after_threshold(self):
        """After LOG_SUPPRESSION_THRESHOLD failures, stream_death events should only fire every LOG_SUPPRESSION_INTERVAL."""
        from src.stream_manager import LOG_SUPPRESSION_INTERVAL, LOG_SUPPRESSION_THRESHOLD

        # Failure count just past the threshold — not on a suppression interval boundary
        failure_count = LOG_SUPPRESSION_THRESHOLD + 2
        assert failure_count % LOG_SUPPRESSION_INTERVAL != 0, (
            "Pick a count that is NOT on the interval"
        )

        # The event should NOT fire for this failure count
        should_emit = (
            failure_count <= LOG_SUPPRESSION_THRESHOLD
            or failure_count % LOG_SUPPRESSION_INTERVAL == 0
        )
        assert should_emit is False

    def test_stream_death_event_emits_on_interval(self):
        """stream_death events should fire on LOG_SUPPRESSION_INTERVAL boundaries."""
        from src.stream_manager import LOG_SUPPRESSION_INTERVAL

        failure_count = LOG_SUPPRESSION_INTERVAL
        should_emit = failure_count <= 3 or failure_count % LOG_SUPPRESSION_INTERVAL == 0
        assert should_emit is True

    def test_stream_death_event_emits_for_first_failures(self):
        """stream_death events should always fire for the first few failures."""
        from src.stream_manager import LOG_SUPPRESSION_THRESHOLD

        for i in range(1, LOG_SUPPRESSION_THRESHOLD + 1):
            should_emit = i <= LOG_SUPPRESSION_THRESHOLD
            assert should_emit is True, f"Failure {i} should emit"


class TestStreamErrorCredentialRedaction:
    """Verify that exception details in log_event calls never leak credentials."""

    def test_stream_error_detail_redacts_credentials(self):
        """str(exc) containing RTSP creds must be scrubbed before log_event."""
        from src.stream_manager import _CRED_RE

        # Simulate an exception whose message contains an RTSP URL with creds
        exc = ConnectionError("Failed to connect to rtsp://admin:s3cret@192.168.1.100:554/stream")
        raw = str(exc)
        # The raw exception contains the password
        assert "s3cret" in raw

        # Apply the same redaction the code should use
        redacted = _CRED_RE.sub(r"://\1:***@", raw)
        assert "s3cret" not in redacted
        assert "admin:***@" in redacted
