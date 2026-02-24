"""Per-camera stream management with reconnect and backoff.

Each camera runs as an independent asyncio task with states:
DISCONNECTED -> CONNECTING -> STREAMING -> COOLDOWN -> STREAMING
                    |                        |
                 ERROR -> BACKOFF -> CONNECTING
"""

from __future__ import annotations

import asyncio
import enum
import logging
import time

from .audio_pipeline import AmbientMonitor, read_audio_clip, start_ffmpeg
from .classifier import ASTClassifier
from .config import CameraConfig
from .mqtt_publisher import MqttPublisher
from .openobserve import log_event

logger = logging.getLogger(__name__)

MAX_BACKOFF = 60  # seconds


class StreamState(enum.Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    STREAMING = "streaming"
    COOLDOWN = "cooldown"
    ERROR = "error"
    BACKOFF = "backoff"


class CameraStream:
    """Manages a single camera's audio stream lifecycle."""

    def __init__(
        self,
        camera: CameraConfig,
        classifier: ASTClassifier,
        publisher: MqttPublisher,
        inference_semaphore: asyncio.Semaphore,
        confidence_threshold: float = 0.15,
        clip_duration: int = 3,
    ) -> None:
        self._camera = camera
        self._classifier = classifier
        self._publisher = publisher
        self._semaphore = inference_semaphore
        self._confidence_threshold = confidence_threshold
        self._clip_duration = clip_duration

        self._state = StreamState.DISCONNECTED
        self._backoff = camera.reconnect_interval
        self._last_event_time: float = 0
        self._inference_count = 0
        self._process: asyncio.subprocess.Process | None = None
        self._task: asyncio.Task | None = None
        self._ambient = AmbientMonitor(camera_name=camera.name)

    @property
    def state(self) -> StreamState:
        return self._state

    @property
    def camera_name(self) -> str:
        return self._camera.name

    @property
    def inference_count(self) -> int:
        return self._inference_count

    @property
    def last_event_time(self) -> float:
        return self._last_event_time

    def start(self) -> asyncio.Task:
        """Start the stream processing task."""
        self._task = asyncio.create_task(
            self._run(), name=f"stream-{self._camera.name}"
        )
        return self._task

    async def stop(self) -> None:
        """Stop the stream and clean up."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._process:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()
        self._publisher.publish_camera_offline(self._camera.name)

    async def _run(self) -> None:
        """Main loop: connect, stream, classify, reconnect on failure."""
        while True:
            try:
                self._state = StreamState.CONNECTING
                logger.info(
                    "[%s] Connecting to %s", self._camera.name, self._camera.rtsp_url
                )

                self._process = await start_ffmpeg(self._camera.rtsp_url)
                self._state = StreamState.STREAMING
                self._backoff = self._camera.reconnect_interval
                self._publisher.publish_camera_online(self._camera.name)
                log_event("stream_online", camera=self._camera.name)
                logger.info("[%s] Streaming", self._camera.name)

                await self._stream_loop()

            except asyncio.CancelledError:
                raise
            except Exception:
                self._state = StreamState.ERROR
                log_event("stream_error", camera=self._camera.name)
                logger.exception("[%s] Stream error", self._camera.name)
            finally:
                if self._process:
                    self._process.terminate()
                    self._process = None
                self._publisher.publish_camera_offline(self._camera.name)

            # Backoff before reconnect
            self._state = StreamState.BACKOFF
            logger.info("[%s] Reconnecting in %ds", self._camera.name, self._backoff)
            await asyncio.sleep(self._backoff)
            self._backoff = min(self._backoff * 2, MAX_BACKOFF)

    async def _stream_loop(self) -> None:
        """Read clips from ffmpeg and classify when triggered."""
        while True:
            result = await read_audio_clip(
                self._process,
                self._camera.db_threshold,
                self._clip_duration,
                ambient_monitor=self._ambient,
            )
            if result is None:
                logger.warning("[%s] Stream ended", self._camera.name)
                return

            audio, trigger_db = result

            # Cooldown check
            now = time.monotonic()
            if now - self._last_event_time < self._camera.cooldown_seconds:
                self._state = StreamState.COOLDOWN
                logger.debug("[%s] Cooldown, skipping", self._camera.name)
                self._state = StreamState.STREAMING
                continue

            # Classify with semaphore (one inference at a time)
            async with self._semaphore:
                classifications = await asyncio.to_thread(
                    self._classifier.classify,
                    audio,
                    trigger_db,
                    self._confidence_threshold,
                )

            if classifications:
                self._last_event_time = now
                self._inference_count += 1
                for cls_result in classifications:
                    logger.info(
                        "[%s] Detected %s (%.2f, %.1f dB)",
                        self._camera.name,
                        cls_result.group,
                        cls_result.confidence,
                        cls_result.db_level,
                    )
                    self._publisher.publish_detection(self._camera.name, cls_result)
                    log_event(
                        "detection",
                        camera=self._camera.name,
                        group=cls_result.group,
                        confidence=cls_result.confidence,
                        db_level=cls_result.db_level,
                        raw_label=cls_result.label,
                    )


class StreamManager:
    """Manages all camera streams."""

    def __init__(
        self,
        cameras: list[CameraConfig],
        classifier: ASTClassifier,
        publisher: MqttPublisher,
        confidence_threshold: float = 0.15,
        clip_duration: int = 3,
    ) -> None:
        self._semaphore = asyncio.Semaphore(1)
        self._streams = [
            CameraStream(
                camera=cam,
                classifier=classifier,
                publisher=publisher,
                inference_semaphore=self._semaphore,
                confidence_threshold=confidence_threshold,
                clip_duration=clip_duration,
            )
            for cam in cameras
        ]

    @property
    def streams(self) -> list[CameraStream]:
        return self._streams

    def start_all(self) -> list[asyncio.Task]:
        """Start all camera streams."""
        return [stream.start() for stream in self._streams]

    async def stop_all(self) -> None:
        """Stop all camera streams."""
        for stream in self._streams:
            await stream.stop()

    def status(self) -> list[dict]:
        """Return status of all streams."""
        return [
            {
                "name": s.camera_name,
                "state": s.state.value,
                "inference_count": s.inference_count,
                "last_event_time": s.last_event_time,
            }
            for s in self._streams
        ]
