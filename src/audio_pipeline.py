"""FFmpeg audio extraction and dB-gated clip capture.

Each camera gets an ffmpeg subprocess that pipes raw PCM audio.
When volume exceeds the dB threshold, we buffer a clip for classification.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import deque
from collections.abc import Callable

import numpy as np

from .openobserve import log_event

logger = logging.getLogger(__name__)

# Patterns for classifying ffmpeg stderr errors into structured event types
_FFMPEG_ERROR_PATTERNS: list[tuple[str, str]] = [
    ("404 Not Found", "rtsp_404"),
    ("Connection timed out", "connection_timeout"),
    ("Connection refused", "connection_refused"),
    ("Server returned 4", "http_4xx"),
    ("Server returned 5", "http_5xx"),
    ("No route to host", "no_route"),
    ("Network is unreachable", "network_unreachable"),
]


def _emit_ffmpeg_error_event(text: str, safe_url: str) -> None:
    """Parse ffmpeg stderr line and emit a structured failure event to OO."""
    for pattern, error_type in _FFMPEG_ERROR_PATTERNS:
        if pattern in text:
            log_event(
                "stream_failure",
                error_type=error_type,
                url=safe_url,
                detail=text[:200],
            )
            return

CHUNK_SAMPLES = 1600  # 100ms at 16kHz
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2  # 16-bit PCM
CHUNK_BYTES = CHUNK_SAMPLES * BYTES_PER_SAMPLE
DB_FLOOR = -96.0
PRE_TRIGGER_CHUNKS = 5  # 500ms at 100ms/chunk

# Strong references to fire-and-forget background tasks to prevent GC
_background_tasks: set[asyncio.Task[None]] = set()


def compute_spectral_flatness(audio_float32: np.ndarray) -> float:
    """Compute spectral flatness (Wiener entropy) of an audio signal.

    Returns a value in [0, 1] where:
      - 0.0 = pure tone (single frequency)
      - 1.0 = white noise (flat spectrum)

    Low flatness (< 0.1) with narrow bandwidth suggests codec/AGC artifacts.
    """
    if len(audio_float32) < 256:
        return 1.0  # Too short to analyze, assume broadband

    # Compute magnitude spectrum
    spectrum = np.abs(np.fft.rfft(audio_float32))
    # Avoid log(0)
    spectrum = np.maximum(spectrum, 1e-10)

    # Spectral flatness = geometric mean / arithmetic mean
    log_mean = np.mean(np.log(spectrum))
    geo_mean = np.exp(log_mean)
    arith_mean = np.mean(spectrum)

    if arith_mean < 1e-10:
        return 1.0

    return float(geo_mean / arith_mean)


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


async def start_ffmpeg(
    rtsp_url: str, *, highpass_freq: int = 0
) -> asyncio.subprocess.Process:
    """Start ffmpeg to extract 16kHz mono PCM from an RTSP stream.

    Args:
        rtsp_url: RTSP stream URL.
        highpass_freq: High-pass filter cutoff in Hz. 0 = disabled.

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
        "30000000",  # 30s RTSP read timeout (microseconds) — tolerate transient Scrypted rebroadcast pauses
        "-i",
        rtsp_url,
        "-vn",
    ]

    if highpass_freq > 0:
        cmd.extend(["-af", f"highpass=f={highpass_freq},alimiter=limit=0.95:attack=5:release=50"])

    cmd.extend([
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        "1",
        "-f",
        "s16le",
        "pipe:1",
    ])

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
                text = line.decode(errors="replace").rstrip()
                # Redact RTSP credentials from ffmpeg error output
                text = re.sub(r"://([^:]+):([^@]+)@", r"://\1:***@", text)
                logger.warning("ffmpeg[%s]: %s", process.pid, text)
                # Ship structured failure events for queryable error patterns
                _emit_ffmpeg_error_event(text, safe_url)
        except Exception:
            logger.debug("ffmpeg stderr drain ended for pid=%s", process.pid)

    task = asyncio.create_task(_drain_stderr(), name=f"ffmpeg-stderr-{process.pid}")
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return process


EMA_STARTUP_CHUNKS = 100  # ~10s at 100ms/chunk — use fixed threshold during startup
EMA_RELEASE_ALPHA = 0.02  # fast release (~5s time constant) — adapts to quiet
EMA_ATTACK_ALPHA = 0.0005  # slow attack (~2-5 min) — resists sustained noise


class AmbientMonitor:
    """Tracks peak dB levels and adaptive EMA across read_audio_clip calls."""

    def __init__(self, camera_name: str = "", report_interval: float = 60.0) -> None:
        self.camera_name = camera_name
        self.report_interval = report_interval
        self.peak_db = DB_FLOOR
        self.chunk_count = 0
        self.last_report = time.monotonic()
        self.last_chunk_time: float = 0.0  # monotonic timestamp of last received chunk
        # Adaptive EMA state
        self._ema_db: float = DB_FLOOR
        self._ema_count: int = 0

    @property
    def ema_db(self) -> float:
        return self._ema_db

    def update(self, db: float, db_threshold: float) -> None:
        """Update tracking, EMA, and log if report interval has elapsed."""
        self.chunk_count += 1
        self.last_chunk_time = time.monotonic()
        if db > self.peak_db:
            self.peak_db = db

        # Update asymmetric EMA
        self._ema_count += 1
        if self._ema_count == 1:
            self._ema_db = db
        else:
            # Release (adapting to quieter): fast alpha
            # Attack (adapting to louder): slow alpha
            alpha = EMA_RELEASE_ALPHA if db < self._ema_db else EMA_ATTACK_ALPHA
            self._ema_db = alpha * db + (1.0 - alpha) * self._ema_db

        now = time.monotonic()
        if now - self.last_report >= self.report_interval:
            cam_label = f"[{self.camera_name}] " if self.camera_name else ""
            logger.info(
                "%sAmbient: peak=%.1f dB, threshold=%.1f dB, ema=%.1f dB (%d chunks/%.0fs)",
                cam_label,
                self.peak_db,
                db_threshold,
                self._ema_db,
                self.chunk_count,
                now - self.last_report,
            )
            self.peak_db = DB_FLOOR
            self.chunk_count = 0
            self.last_report = now

    def get_adaptive_threshold(self, fixed: float, margin: float) -> float:
        """Compute adaptive threshold from EMA + margin, floored by fixed threshold.

        During startup (first EMA_STARTUP_CHUNKS), returns fixed threshold.
        """
        if self._ema_count < EMA_STARTUP_CHUNKS:
            return fixed
        return max(fixed, self._ema_db + margin)


async def read_audio_clip(
    process: asyncio.subprocess.Process,
    db_threshold: float,
    clip_duration_seconds: int = 3,
    *,
    ambient_monitor: AmbientMonitor | None = None,
    threshold_fn: Callable[[], float] | None = None,
) -> tuple[np.ndarray, float, float] | None:
    """Read PCM chunks from ffmpeg, gate on dB level, capture a clip.

    This is a coroutine that blocks until either:
    - A clip is captured (dB exceeded threshold, then clip_duration_seconds recorded)
    - The ffmpeg process exits (returns None)

    Args:
        threshold_fn: Optional callable returning dynamic dB threshold each iteration.
            When provided, overrides db_threshold on each chunk.

    Returns:
        (audio_float32, trigger_db, trigger_time) tuple, or None if stream ended.
        trigger_time is a monotonic timestamp captured at the trigger point.
    """
    clip_samples = clip_duration_seconds * SAMPLE_RATE
    buffer: list[bytes] = []
    pre_buffer: deque[bytes] = deque(maxlen=PRE_TRIGGER_CHUNKS)
    recording = False
    trigger_db = 0.0
    trigger_time = 0.0
    samples_recorded = 0

    if process.stdout is None:
        return None

    while True:
        try:
            chunk = await asyncio.wait_for(
                process.stdout.readexactly(CHUNK_BYTES), timeout=30.0
            )
        except TimeoutError:
            logger.warning("ffmpeg stdout read timeout (pid=%s), killing process", process.pid)
            try:
                process.kill()
            except ProcessLookupError:
                pass
            return None
        except asyncio.IncompleteReadError:
            return None  # Stream ended (EOF with partial chunk)

        # Decode PCM chunk
        pcm = np.frombuffer(chunk, dtype=np.int16)
        db = compute_rms_db(pcm)

        # Resolve effective threshold (adaptive or fixed)
        effective_threshold = threshold_fn() if threshold_fn is not None else db_threshold

        # Track ambient dB levels (persists across calls via AmbientMonitor)
        if ambient_monitor is not None:
            ambient_monitor.update(db, effective_threshold)

        if recording:
            buffer.append(chunk)
            samples_recorded += len(pcm)
            if samples_recorded >= clip_samples:
                # Clip complete
                all_bytes = b"".join(buffer)
                audio_int16 = np.frombuffer(all_bytes, dtype=np.int16)
                audio_float32 = audio_int16.astype(np.float32) / 32768.0
                return (audio_float32, trigger_db, trigger_time)
        elif db > effective_threshold:
            # Start recording — prepend pre-buffer to capture attack transient
            recording = True
            trigger_db = db
            trigger_time = time.monotonic()
            buffer = [*pre_buffer, chunk]
            samples_recorded = sum(len(np.frombuffer(b, dtype=np.int16)) for b in buffer)
            logger.debug(
                "Trigger at %.1f dB (threshold %.1f), pre-buffer=%d chunks",
                db, effective_threshold, len(pre_buffer),
            )
        else:
            pre_buffer.append(chunk)
