"""Tests for audio pipeline: dB computation, AmbientMonitor, HPF, adaptive thresholds, and pre-trigger buffer."""

import time
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from src.audio_pipeline import (
    CHUNK_SAMPLES,
    DB_FLOOR,
    EMA_ATTACK_ALPHA,
    EMA_RELEASE_ALPHA,
    EMA_STARTUP_CHUNKS,
    PRE_TRIGGER_CHUNKS,
    AmbientMonitor,
    compute_rms_db,
    read_audio_clip,
    start_ffmpeg,
)


class TestComputeRmsDb:
    """Test RMS dB calculation from PCM samples."""

    def test_silence_returns_floor(self):
        silence = np.zeros(1600, dtype=np.int16)
        assert compute_rms_db(silence) == DB_FLOOR

    def test_empty_returns_floor(self):
        empty = np.array([], dtype=np.int16)
        assert compute_rms_db(empty) == DB_FLOOR

    def test_full_scale_near_zero_db(self):
        # Full-scale square wave: all samples at max amplitude
        full_scale = np.full(1600, 32767, dtype=np.int16)
        db = compute_rms_db(full_scale)
        assert -0.1 < db < 0.0  # Should be ~0 dBFS

    def test_half_scale_near_minus_6(self):
        # Half amplitude = ~-6 dBFS
        half = np.full(1600, 16384, dtype=np.int16)
        db = compute_rms_db(half)
        assert -7.0 < db < -5.0

    def test_low_level_below_threshold(self):
        # Very quiet signal
        quiet = np.full(1600, 10, dtype=np.int16)
        db = compute_rms_db(quiet)
        assert db < -60.0

    def test_returns_float(self):
        samples = np.full(1600, 1000, dtype=np.int16)
        assert isinstance(compute_rms_db(samples), float)


class TestAmbientMonitor:
    """Test ambient dB level tracking and periodic reporting."""

    def test_initial_state(self):
        mon = AmbientMonitor(camera_name="test_cam")
        assert mon.camera_name == "test_cam"
        assert mon.peak_db == DB_FLOOR
        assert mon.chunk_count == 0

    def test_update_tracks_peak(self):
        mon = AmbientMonitor(camera_name="test")
        mon.update(-50.0, -40.0)
        mon.update(-30.0, -40.0)
        mon.update(-45.0, -40.0)
        assert mon.peak_db == -30.0
        assert mon.chunk_count == 3

    def test_update_increments_chunk_count(self):
        mon = AmbientMonitor()
        for _ in range(10):
            mon.update(-60.0, -40.0)
        assert mon.chunk_count == 10

    def test_report_resets_state(self):
        mon = AmbientMonitor(report_interval=0.0)  # Report immediately
        # Force last_report into the past
        mon.last_report = time.monotonic() - 1.0
        mon.update(-30.0, -40.0)
        # After report, state should be reset
        assert mon.peak_db == DB_FLOOR
        assert mon.chunk_count == 0

    def test_report_logs_with_camera_name(self):
        mon = AmbientMonitor(camera_name="back_porch", report_interval=0.0)
        mon.last_report = time.monotonic() - 1.0
        with patch("src.audio_pipeline.logger") as mock_logger:
            mon.update(-35.0, -40.0)
            mock_logger.info.assert_called_once()
            log_msg = mock_logger.info.call_args[0][1]
            assert "[back_porch]" in log_msg

    def test_report_logs_without_camera_name(self):
        mon = AmbientMonitor(camera_name="", report_interval=0.0)
        mon.last_report = time.monotonic() - 1.0
        with patch("src.audio_pipeline.logger") as mock_logger:
            mon.update(-35.0, -40.0)
            mock_logger.info.assert_called_once()
            log_msg = mock_logger.info.call_args[0][1]
            assert "[" not in log_msg

    def test_no_report_before_interval(self):
        mon = AmbientMonitor(report_interval=60.0)
        with patch("src.audio_pipeline.logger") as mock_logger:
            mon.update(-35.0, -40.0)
            mock_logger.info.assert_not_called()
        assert mon.chunk_count == 1  # Not reset

    def test_state_persists_across_updates(self):
        """Verify state isn't lost between calls (the core bug from review)."""
        mon = AmbientMonitor(camera_name="test", report_interval=120.0)

        # Simulate many updates spread across multiple "clip captures"
        for i in range(50):
            mon.update(-60.0 + i * 0.5, -40.0)

        # State should accumulate, not reset
        assert mon.chunk_count == 50
        assert mon.peak_db == -60.0 + 49 * 0.5  # -35.5


class TestAmbientMonitorEMA:
    """Test asymmetric EMA behavior in AmbientMonitor."""

    def test_ema_initializes_to_first_value(self):
        mon = AmbientMonitor()
        mon.update(-50.0, -40.0)
        assert mon.ema_db == -50.0

    def test_ema_release_faster_than_attack(self):
        """When ambient gets quieter, EMA should drop faster than it rises."""
        mon = AmbientMonitor()
        # Seed with loud ambient
        mon.update(-20.0, -40.0)
        initial = mon.ema_db

        # Feed quieter values — should drop quickly (release)
        for _ in range(50):
            mon.update(-60.0, -40.0)
        after_quiet = mon.ema_db

        # Reset and seed with quiet ambient
        mon2 = AmbientMonitor()
        mon2.update(-60.0, -40.0)

        # Feed louder values — should rise slowly (attack)
        for _ in range(50):
            mon2.update(-20.0, -40.0)
        after_loud = mon2.ema_db

        # Release should have moved more from -20 toward -60
        release_delta = abs(after_quiet - initial)
        # Attack should have moved less from -60 toward -20
        attack_delta = abs(after_loud - (-60.0))

        assert release_delta > attack_delta, (
            f"Release moved {release_delta:.1f} dB but attack only moved {attack_delta:.1f} dB"
        )

    def test_ema_release_alpha_applied_for_quieter_signal(self):
        """Verify release alpha is used when signal drops below EMA."""
        mon = AmbientMonitor()
        mon.update(-30.0, -40.0)  # Initialize EMA to -30

        # Next update is quieter: should use release alpha
        mon.update(-50.0, -40.0)
        expected = EMA_RELEASE_ALPHA * (-50.0) + (1 - EMA_RELEASE_ALPHA) * (-30.0)
        assert abs(mon.ema_db - expected) < 0.01

    def test_ema_attack_alpha_applied_for_louder_signal(self):
        """Verify attack alpha is used when signal rises above EMA."""
        mon = AmbientMonitor()
        mon.update(-50.0, -40.0)  # Initialize EMA to -50

        # Next update is louder: should use attack alpha
        mon.update(-20.0, -40.0)
        expected = EMA_ATTACK_ALPHA * (-20.0) + (1 - EMA_ATTACK_ALPHA) * (-50.0)
        assert abs(mon.ema_db - expected) < 0.01


class TestAdaptiveThreshold:
    """Test get_adaptive_threshold behavior."""

    def test_returns_fixed_during_startup(self):
        mon = AmbientMonitor()
        # Feed fewer than EMA_STARTUP_CHUNKS
        for _ in range(EMA_STARTUP_CHUNKS - 1):
            mon.update(-50.0, -40.0)
        threshold = mon.get_adaptive_threshold(-35.0, 8.0)
        assert threshold == -35.0

    def test_returns_adaptive_after_startup(self):
        mon = AmbientMonitor()
        # Feed exactly EMA_STARTUP_CHUNKS quiet values
        for _ in range(EMA_STARTUP_CHUNKS):
            mon.update(-60.0, -40.0)
        # EMA should be close to -60, so adaptive = -60 + 8 = -52
        # But fixed is -35, so max(-35, -52) = -35
        threshold = mon.get_adaptive_threshold(-35.0, 8.0)
        assert threshold == -35.0  # Fixed wins when adaptive is lower

    def test_adaptive_raises_threshold_in_noisy_environment(self):
        mon = AmbientMonitor()
        # Seed with initial value
        mon.update(-20.0, -40.0)
        # Mark as past startup
        mon._ema_count = EMA_STARTUP_CHUNKS + 1
        mon._ema_db = -20.0  # Very noisy environment

        # Adaptive = -20 + 8 = -12, fixed = -35
        # max(-35, -12) = -12 → adaptive wins
        threshold = mon.get_adaptive_threshold(-35.0, 8.0)
        assert threshold == -12.0

    def test_fixed_threshold_is_floor(self):
        mon = AmbientMonitor()
        mon._ema_count = EMA_STARTUP_CHUNKS + 1
        mon._ema_db = -70.0  # Very quiet

        # Adaptive = -70 + 8 = -62, fixed = -35
        # max(-35, -62) = -35 → fixed is the floor
        threshold = mon.get_adaptive_threshold(-35.0, 8.0)
        assert threshold == -35.0

    def test_startup_guard_suppresses_noisy_ema(self):
        """During startup, even a noisy EMA should not raise the threshold."""
        mon = AmbientMonitor()
        for _ in range(EMA_STARTUP_CHUNKS - 1):
            mon.update(-10.0, -35.0)
        # EMA ~-10, adaptive would be -10 + 8 = -2, but startup guard active
        threshold = mon.get_adaptive_threshold(-35.0, 8.0)
        assert threshold == -35.0

    def test_adaptive_activates_at_startup_boundary(self):
        """Adaptive threshold activates after exactly EMA_STARTUP_CHUNKS updates."""
        mon = AmbientMonitor()
        for _ in range(EMA_STARTUP_CHUNKS):
            mon.update(-10.0, -35.0)
        # Now past startup — EMA ~-10, adaptive = -10 + 8 = -2 > -35
        threshold = mon.get_adaptive_threshold(-35.0, 8.0)
        assert threshold > -35.0


class TestStartFfmpegHPF:
    """Test high-pass filter insertion in ffmpeg command."""

    @pytest.mark.asyncio
    async def test_no_hpf_when_zero(self):
        """highpass_freq=0 should NOT add -af filter to ffmpeg command."""
        with patch("src.audio_pipeline.asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.pid = 12345
            mock_proc.stderr = None
            mock_exec.return_value = mock_proc

            await start_ffmpeg("rtsp://host/stream", highpass_freq=0)

            cmd = mock_exec.call_args[0]
            assert "-af" not in cmd

    @pytest.mark.asyncio
    async def test_hpf_inserted_when_nonzero(self):
        """highpass_freq>0 should insert -af highpass=f=N between -vn and -acodec."""
        with patch("src.audio_pipeline.asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.pid = 12345
            mock_proc.stderr = None
            mock_exec.return_value = mock_proc

            await start_ffmpeg("rtsp://host/stream", highpass_freq=120)

            cmd = mock_exec.call_args[0]
            assert "-af" in cmd
            af_idx = cmd.index("-af")
            assert cmd[af_idx + 1] == "highpass=f=120"
            # Should be after -vn and before -acodec
            vn_idx = cmd.index("-vn")
            acodec_idx = cmd.index("-acodec")
            assert vn_idx < af_idx < acodec_idx

    @pytest.mark.asyncio
    async def test_hpf_various_frequencies(self):
        """Different HPF frequencies should produce correct filter strings."""
        for freq in [80, 100, 200]:
            with patch("src.audio_pipeline.asyncio.create_subprocess_exec") as mock_exec:
                mock_proc = AsyncMock()
                mock_proc.pid = 12345
                mock_proc.stderr = None
                mock_exec.return_value = mock_proc

                await start_ffmpeg("rtsp://host/stream", highpass_freq=freq)

                cmd = mock_exec.call_args[0]
                af_idx = cmd.index("-af")
                assert cmd[af_idx + 1] == f"highpass=f={freq}"


def _make_pcm_chunk(amplitude: int = 10) -> bytes:
    """Create a single PCM chunk with the given amplitude."""
    return np.full(CHUNK_SAMPLES, amplitude, dtype=np.int16).tobytes()


def _make_loud_chunk(amplitude: int = 20000) -> bytes:
    """Create a loud PCM chunk that will exceed typical dB thresholds."""
    return np.full(CHUNK_SAMPLES, amplitude, dtype=np.int16).tobytes()


class _FakeStdout:
    """Fake async stdout that yields a sequence of chunks then EOF."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)
        self._index = 0

    async def read(self, n: int) -> bytes:
        if self._index >= len(self._chunks):
            return b""
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk


class _FakeProcess:
    """Minimal fake asyncio.subprocess.Process with a fake stdout."""

    def __init__(self, chunks: list[bytes]) -> None:
        self.stdout = _FakeStdout(chunks)


class TestPreTriggerBuffer:
    """Test the pre-trigger ring buffer in read_audio_clip."""

    def test_pre_trigger_chunks_constant(self):
        assert PRE_TRIGGER_CHUNKS == 5

    @pytest.mark.asyncio
    async def test_clip_includes_pre_buffer(self):
        """Pre-buffer chunks should be prepended to the captured clip."""
        quiet_chunks = [_make_pcm_chunk(10) for _ in range(5)]
        loud_chunks = [_make_loud_chunk() for _ in range(10)]
        process = _FakeProcess(quiet_chunks + loud_chunks)

        result = await read_audio_clip(process, db_threshold=-60.0, clip_duration_seconds=1)
        assert result is not None
        audio, _trigger_db, _trigger_time = result
        assert len(audio) == 16000
        pre_section = audio[:8000]
        expected_quiet = np.full(CHUNK_SAMPLES, 10, dtype=np.int16).astype(np.float32) / 32768.0
        np.testing.assert_allclose(pre_section[:CHUNK_SAMPLES], expected_quiet, atol=1e-6)

    @pytest.mark.asyncio
    async def test_pre_buffer_bounded(self):
        """Pre-buffer should not exceed PRE_TRIGGER_CHUNKS entries."""
        quiet_chunks = [_make_pcm_chunk(10 + i) for i in range(20)]
        loud_chunks = [_make_loud_chunk() for _ in range(10)]
        process = _FakeProcess(quiet_chunks + loud_chunks)

        result = await read_audio_clip(process, db_threshold=-60.0, clip_duration_seconds=1)
        assert result is not None
        audio, _, _ = result
        assert len(audio) == 16000
        first_samples = audio[:CHUNK_SAMPLES]
        expected = np.full(CHUNK_SAMPLES, 25, dtype=np.int16).astype(np.float32) / 32768.0
        np.testing.assert_allclose(first_samples, expected, atol=1e-6)

    @pytest.mark.asyncio
    async def test_trigger_on_first_chunk(self):
        """If the first chunk is loud, clip works with no pre-buffer."""
        loud_chunks = [_make_loud_chunk() for _ in range(10)]
        process = _FakeProcess(loud_chunks)

        result = await read_audio_clip(process, db_threshold=-60.0, clip_duration_seconds=1)
        assert result is not None
        audio, _, _ = result
        assert len(audio) == 16000

    @pytest.mark.asyncio
    async def test_partial_pre_buffer(self):
        """If fewer than PRE_TRIGGER_CHUNKS quiet chunks before trigger, use what's available."""
        quiet_chunks = [_make_pcm_chunk(10) for _ in range(2)]
        loud_chunks = [_make_loud_chunk() for _ in range(10)]
        process = _FakeProcess(quiet_chunks + loud_chunks)

        result = await read_audio_clip(process, db_threshold=-60.0, clip_duration_seconds=1)
        assert result is not None
        audio, _, _ = result
        assert len(audio) == 16000

    @pytest.mark.asyncio
    async def test_samples_recorded_accounts_for_pre_buffer(self):
        """Total clip length should account for pre-buffer samples correctly."""
        quiet_chunks = [_make_pcm_chunk(10) for _ in range(5)]
        loud_chunks = [_make_loud_chunk() for _ in range(30)]
        process = _FakeProcess(quiet_chunks + loud_chunks)

        result = await read_audio_clip(process, db_threshold=-60.0, clip_duration_seconds=3)
        assert result is not None
        audio, _, _ = result
        assert len(audio) == 48000

    @pytest.mark.asyncio
    async def test_stream_end_returns_none(self):
        """If stream ends before trigger, return None."""
        quiet_chunks = [_make_pcm_chunk(10) for _ in range(3)]
        process = _FakeProcess(quiet_chunks)

        result = await read_audio_clip(process, db_threshold=-20.0, clip_duration_seconds=1)
        assert result is None


class TestReadAudioClipTriggerTime:
    """Test trigger_time in read_audio_clip return value."""

    @pytest.mark.asyncio
    async def test_returns_3_tuple(self):
        """read_audio_clip should return (audio, trigger_db, trigger_time)."""
        loud_chunks = [_make_loud_chunk() for _ in range(10)]
        process = _FakeProcess(loud_chunks)

        result = await read_audio_clip(process, db_threshold=-60.0, clip_duration_seconds=1)
        assert result is not None
        assert len(result) == 3
        audio, trigger_db, trigger_time = result
        assert isinstance(audio, np.ndarray)
        assert isinstance(trigger_db, float)
        assert isinstance(trigger_time, float)

    @pytest.mark.asyncio
    async def test_trigger_time_is_monotonic(self):
        """trigger_time should be a monotonic timestamp near now."""
        loud_chunks = [_make_loud_chunk() for _ in range(10)]
        process = _FakeProcess(loud_chunks)

        before = time.monotonic()
        result = await read_audio_clip(process, db_threshold=-60.0, clip_duration_seconds=1)
        after = time.monotonic()

        assert result is not None
        _, _, trigger_time = result
        assert before <= trigger_time <= after


class TestReadAudioClipThresholdFn:
    """Test threshold_fn parameter in read_audio_clip."""

    @pytest.mark.asyncio
    async def test_threshold_fn_overrides_fixed(self):
        """When threshold_fn returns a high value, quiet chunks won't trigger."""
        quiet_chunks = [_make_pcm_chunk(100) for _ in range(5)]  # ~-50 dB
        process = _FakeProcess(quiet_chunks)

        # Fixed threshold would trigger, but fn returns very high threshold
        result = await read_audio_clip(
            process,
            db_threshold=-60.0,  # Fixed — would trigger
            clip_duration_seconds=1,
            threshold_fn=lambda: 0.0,  # 0 dBFS — nothing triggers
        )
        assert result is None  # Stream ends without triggering

    @pytest.mark.asyncio
    async def test_threshold_fn_allows_trigger(self):
        """threshold_fn returning low value should allow trigger."""
        loud_chunks = [_make_loud_chunk() for _ in range(10)]
        process = _FakeProcess(loud_chunks)

        result = await read_audio_clip(
            process,
            db_threshold=-20.0,  # Fixed — would not trigger for moderate
            clip_duration_seconds=1,
            threshold_fn=lambda: -60.0,  # Very low — everything triggers
        )
        assert result is not None
        audio, _, _ = result
        assert len(audio) == 16000

    @pytest.mark.asyncio
    async def test_none_threshold_fn_uses_fixed(self):
        """threshold_fn=None should fall back to fixed db_threshold."""
        loud_chunks = [_make_loud_chunk() for _ in range(10)]
        process = _FakeProcess(loud_chunks)

        result = await read_audio_clip(
            process,
            db_threshold=-60.0,
            clip_duration_seconds=1,
            threshold_fn=None,
        )
        assert result is not None
