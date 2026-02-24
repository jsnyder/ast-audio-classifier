"""Tests for audio pipeline: dB computation and AmbientMonitor."""

import time
from unittest.mock import patch

import numpy as np

from src.audio_pipeline import (
    DB_FLOOR,
    AmbientMonitor,
    compute_rms_db,
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
