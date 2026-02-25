"""FFmpeg audio extraction and dB-gated clip capture.

Each camera gets an ffmpeg subprocess that pipes raw PCM audio.
When volume exceeds the dB threshold, we buffer a clip for classification.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time

import numpy as np

logger = logging.getLogger(__name__)

CHUNK_SAMPLES = 1600  # 100ms at 16kHz
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2  # 16-bit PCM
CHUNK_BYTES = CHUNK_SAMPLES * BYTES_PER_SAMPLE
DB_FLOOR = -96.0


def compute_rms_db(pcm_int16: np.ndarray) -> float:
    """Compute RMS dB level from 16-bit PCM samples.

    Returns dB relative to full scale (0 dBFS).
    Silence returns DB_FLOOR (-96 dB).
    """
    if len(pcm_int16) == 0:
        return DB_FLOOR

    samples = pcm_int16.astype(np.float64)
    rms = np.sqrt(np.mean(samples**2))
    if rms < 1.0:
        return DB_FLOOR

    db = 20.0 * np.log10(rms / 32768.0)
    return max(db, DB_FLOOR)


async def start_ffmpeg(rtsp_url: str) -> asyncio.subprocess.Process:
    """Start ffmpeg to extract 16kHz mono PCM from an RTSP stream.

    Returns the subprocess. Audio is read from stdout as raw s16le PCM.
    """
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-nostats",
        "-rtsp_transport",
        "tcp",
        "-allowed_media_types",
        "audio",
        "-timeout",
        "10000000",  # 10s RTSP timeout (microseconds)
        "-i",
        rtsp_url,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        "1",
        "-f",
        "s16le",
        "pipe:1",
    ]

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    safe_url = re.sub(r"://([^:]+):([^@]+)@", r"://\1:***@", rtsp_url)
    logger.info("ffmpeg started for %s (pid=%s)", safe_url, process.pid)

    # Drain stderr in a background task to prevent pipe deadlock
    async def _drain_stderr() -> None:
        if process.stderr is None:
            return
        try:
            while True:
                try:
                    line = await process.stderr.readline()
                except ValueError:
                    # LimitOverrunError: ffmpeg wrote a very long line (e.g. progress stats)
                    # Read and discard up to 64KB to recover
                    await process.stderr.read(65536)
                    continue
                if not line:
                    break
                logger.warning(
                    "ffmpeg[%s]: %s",
                    process.pid,
                    line.decode(errors="replace").rstrip(),
                )
        except Exception:
            logger.debug("ffmpeg stderr drain ended for pid=%s", process.pid)

    asyncio.create_task(_drain_stderr(), name=f"ffmpeg-stderr-{process.pid}")
    return process


class AmbientMonitor:
    """Tracks peak dB levels across read_audio_clip calls for periodic reporting."""

    def __init__(self, camera_name: str = "", report_interval: float = 60.0) -> None:
        self.camera_name = camera_name
        self.report_interval = report_interval
        self.peak_db = DB_FLOOR
        self.chunk_count = 0
        self.last_report = time.monotonic()
        self.last_chunk_time: float = 0.0  # monotonic timestamp of last received chunk

    def update(self, db: float, db_threshold: float) -> None:
        """Update tracking and log if report interval has elapsed."""
        self.chunk_count += 1
        self.last_chunk_time = time.monotonic()
        if db > self.peak_db:
            self.peak_db = db
        now = time.monotonic()
        if now - self.last_report >= self.report_interval:
            cam_label = f"[{self.camera_name}] " if self.camera_name else ""
            logger.info(
                "%sAmbient: peak=%.1f dB, threshold=%.1f dB (%d chunks/%.0fs)",
                cam_label,
                self.peak_db,
                db_threshold,
                self.chunk_count,
                now - self.last_report,
            )
            self.peak_db = DB_FLOOR
            self.chunk_count = 0
            self.last_report = now


async def read_audio_clip(
    process: asyncio.subprocess.Process,
    db_threshold: float,
    clip_duration_seconds: int = 3,
    *,
    ambient_monitor: AmbientMonitor | None = None,
) -> tuple[np.ndarray, float] | None:
    """Read PCM chunks from ffmpeg, gate on dB level, capture a clip.

    This is a coroutine that blocks until either:
    - A clip is captured (dB exceeded threshold, then clip_duration_seconds recorded)
    - The ffmpeg process exits (returns None)

    Returns:
        (audio_float32, trigger_db) tuple, or None if stream ended.
    """
    clip_samples = clip_duration_seconds * SAMPLE_RATE
    buffer: list[bytes] = []
    recording = False
    trigger_db = 0.0
    samples_recorded = 0

    if process.stdout is None:
        return None

    while True:
        try:
            chunk = await asyncio.wait_for(
                process.stdout.read(CHUNK_BYTES), timeout=30.0
            )
        except asyncio.TimeoutError:
            logger.warning("ffmpeg stdout read timeout")
            return None
        if not chunk:
            return None  # Stream ended

        # Decode PCM chunk
        pcm = np.frombuffer(chunk, dtype=np.int16)
        db = compute_rms_db(pcm)

        # Track ambient dB levels (persists across calls via AmbientMonitor)
        if ambient_monitor is not None:
            ambient_monitor.update(db, db_threshold)

        if recording:
            buffer.append(chunk)
            samples_recorded += len(pcm)
            if samples_recorded >= clip_samples:
                # Clip complete
                all_bytes = b"".join(buffer)
                audio_int16 = np.frombuffer(all_bytes, dtype=np.int16)
                audio_float32 = audio_int16.astype(np.float32) / 32768.0
                return (audio_float32, trigger_db)
        elif db > db_threshold:
            # Start recording
            recording = True
            trigger_db = db
            buffer = [chunk]
            samples_recorded = len(pcm)
            logger.debug("Trigger at %.1f dB (threshold %.1f)", db, db_threshold)
