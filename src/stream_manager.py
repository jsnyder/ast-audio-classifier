"""Per-camera stream management with reconnect, backoff, and URL discovery.

Each camera runs as an independent asyncio task with states:
DISCONNECTED -> CONNECTING -> STREAMING -> COOLDOWN -> STREAMING
                    |                        |
                 ERROR -> BACKOFF -> CONNECTING
                    |
              DISCOVERING -> CONNECTING  (when stream repeatedly dies quickly)

When a camera has go2rtc_stream configured, the stream manager connects
exclusively through the go2rtc proxy and never falls back to direct RTSP
URLs.  This prevents opening competing connections to cameras that only
support a single RTSP consumer (e.g. battery-powered Arlo cameras managed
by Scrypted).
"""

from __future__ import annotations

import asyncio
import contextlib
import enum
import logging
import re
import time
from typing import TYPE_CHECKING

from .audio_pipeline import AmbientMonitor, read_audio_clip, start_ffmpeg
from .clap_verifier import CLAPVerifier
from .classifier import ASTClassifier
from .config import CameraConfig
from .mqtt_publisher import MqttPublisher
from .openobserve import log_event

if TYPE_CHECKING:
    from .event_consolidator import EventConsolidator
    from .llm_judge import LLMJudge
    from .noise_stress import NoiseStressScorer
    from .url_resolver import ScryptedUrlResolver

logger = logging.getLogger(__name__)

MAX_BACKOFF = 60  # seconds
DISCOVERY_THRESHOLD = 3  # consecutive short-lived failures before URL discovery
STABLE_STREAM_SECONDS = 30  # stream must last this long to be considered stable


class StreamState(enum.Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    STREAMING = "streaming"
    COOLDOWN = "cooldown"
    ERROR = "error"
    BACKOFF = "backoff"
    DISCOVERING = "discovering"


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
        clap_verifier: CLAPVerifier | None = None,
        llm_judge: LLMJudge | None = None,
        consolidator: EventConsolidator | None = None,
        noise_stress: NoiseStressScorer | None = None,
        resolver: ScryptedUrlResolver | None = None,
        auto_discovery: bool = False,
    ) -> None:
        self._camera = camera
        self._classifier = classifier
        self._publisher = publisher
        self._semaphore = inference_semaphore
        self._confidence_threshold = confidence_threshold
        self._clip_duration = clip_duration
        self._clap_verifier = clap_verifier
        self._llm_judge = llm_judge
        self._consolidator = consolidator
        self._noise_stress = noise_stress
        self._resolver = resolver
        self._auto_discovery = auto_discovery

        self._state = StreamState.DISCONNECTED
        self._backoff = camera.reconnect_interval
        self._last_event_time: float = 0
        self._inference_count = 0
        self._consecutive_failures = 0
        self._process: asyncio.subprocess.Process | None = None
        self._task: asyncio.Task | None = None
        self._ambient = AmbientMonitor(camera_name=camera.name)
        # Limit concurrent LLM judge tasks to prevent unbounded accumulation
        self._judge_semaphore = asyncio.Semaphore(2)
        # Current effective URL (may be updated by discovery)
        self._effective_url: str = camera.rtsp_url

        # Build adaptive threshold closure if enabled
        self._threshold_fn = None
        if camera.adaptive_threshold:
            def _adaptive_fn() -> float:
                return self._ambient.get_adaptive_threshold(
                    camera.db_threshold, camera.adaptive_margin_db
                )
            self._threshold_fn = _adaptive_fn

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

    @property
    def last_chunk_time(self) -> float:
        """Monotonic timestamp of last audio chunk received (0.0 if none)."""
        return self._ambient.last_chunk_time

    @property
    def ambient_info(self) -> dict:
        """Return ambient monitoring stats for diagnostics."""
        info: dict = {
            "peak_db": round(self._ambient.peak_db, 1),
            "chunk_count": self._ambient.chunk_count,
            "threshold": self._camera.db_threshold,
        }
        if self._camera.adaptive_threshold:
            info["ema_db"] = round(self._ambient.ema_db, 1)
            info["adaptive_threshold"] = round(
                self._ambient.get_adaptive_threshold(
                    self._camera.db_threshold, self._camera.adaptive_margin_db
                ),
                1,
            )
        return info

    def _get_connect_url(self) -> str:
        """Return the URL to connect to.

        For cameras with go2rtc_stream configured, always use the go2rtc
        proxy URL.  This ensures we never open a direct connection to the
        camera, which would compete with Scrypted's single-stream slot.
        """
        if self._camera.go2rtc_stream:
            return self._effective_url
        return self._effective_url

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
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        if self._process:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except TimeoutError:
                self._process.kill()
        self._publisher.publish_camera_offline(self._camera.name)

    async def _attempt_discovery(self) -> None:
        """Try to discover a fresh RTSP URL via the resolver.

        For cameras with go2rtc_stream, discovery only updates the go2rtc
        proxy URL (never reverts to a direct camera URL).
        """
        if not self._auto_discovery or self._resolver is None:
            return

        self._state = StreamState.DISCOVERING
        logger.info(
            "[%s] Attempting URL discovery (current: %s)",
            self._camera.name,
            re.sub(r"://([^:]+):([^@]+)@", r"://\1:***@", self._effective_url),
        )

        try:
            resolved = await self._resolver.resolve(
                self._camera.rtsp_url,
                go2rtc_stream=self._camera.go2rtc_stream,
            )
        except Exception:
            logger.debug("[%s] URL discovery failed", self._camera.name, exc_info=True)
            resolved = None

        if resolved is not None:
            safe = re.sub(r"://([^:]+):([^@]+)@", r"://\1:***@", resolved)
            logger.info("[%s] Discovered URL: %s", self._camera.name, safe)
            self._effective_url = resolved
        elif self._camera.go2rtc_stream:
            # For go2rtc-only cameras, stay on go2rtc proxy — do NOT revert
            # to the original direct URL which would open a competing connection.
            logger.info(
                "[%s] Discovery returned no result; staying on go2rtc proxy "
                "(go2rtc_stream=%s)",
                self._camera.name,
                self._camera.go2rtc_stream,
            )
        else:
            safe = re.sub(
                r"://([^:]+):([^@]+)@", r"://\1:***@", self._camera.rtsp_url
            )
            logger.info(
                "[%s] Reverting to original URL: %s", self._camera.name, safe
            )
            self._effective_url = self._camera.rtsp_url

    async def _run(self) -> None:
        """Main loop: connect, stream, classify, reconnect on failure."""
        while True:
            stream_start = time.monotonic()
            try:
                self._state = StreamState.CONNECTING
                url = self._get_connect_url()
                safe_url = re.sub(
                    r"://([^:]+):([^@]+)@", r"://\1:***@", url
                )
                logger.info("[%s] Connecting to %s", self._camera.name, safe_url)

                self._process = await start_ffmpeg(
                    url,
                    highpass_freq=self._camera.highpass_freq,
                )
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
                    try:
                        self._process.terminate()
                        await asyncio.wait_for(self._process.wait(), timeout=5.0)
                    except (TimeoutError, ProcessLookupError):
                        with contextlib.suppress(ProcessLookupError):
                            self._process.kill()
                    self._process = None
                self._publisher.publish_camera_offline(self._camera.name)

            # Track stream stability for discovery decisions
            stream_duration = time.monotonic() - stream_start
            if stream_duration >= STABLE_STREAM_SECONDS:
                # Stream was stable — reset failure counter
                self._consecutive_failures = 0
            else:
                self._consecutive_failures += 1
                logger.warning(
                    "[%s] Stream died after %.1fs (< %ds), failure %d/%d",
                    self._camera.name,
                    stream_duration,
                    STABLE_STREAM_SECONDS,
                    self._consecutive_failures,
                    DISCOVERY_THRESHOLD,
                )

            # Trigger URL discovery after repeated short-lived streams
            if self._consecutive_failures >= DISCOVERY_THRESHOLD:
                await self._attempt_discovery()
                self._consecutive_failures = 0

            # Backoff before reconnect
            self._state = StreamState.BACKOFF
            logger.info(
                "[%s] Reconnecting in %ds", self._camera.name, self._backoff
            )
            await asyncio.sleep(self._backoff)
            self._backoff = min(self._backoff * 2, MAX_BACKOFF)

    async def _run_judge(
        self, audio: object, classifications: list
    ) -> None:
        """Run LLM judge with back-pressure semaphore."""
        async with self._judge_semaphore:
            await self._llm_judge.evaluate(audio, classifications, self._camera.name)

    async def _stream_loop(self) -> None:
        """Read clips from ffmpeg and classify when triggered."""
        while True:
            result = await read_audio_clip(
                self._process,
                self._camera.db_threshold,
                self._clip_duration,
                ambient_monitor=self._ambient,
                threshold_fn=self._threshold_fn,
            )
            if result is None:
                logger.warning("[%s] Stream ended", self._camera.name)
                return

            audio, trigger_db, trigger_time = result

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
                # CLAP verification (inside semaphore — sequential with AST)
                if classifications and self._clap_verifier is not None:
                    classifications = await asyncio.to_thread(
                        self._clap_verifier.verify,
                        audio,
                        classifications,
                        self._camera.name,
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
                    oo_fields: dict = {
                        "group": cls_result.group,
                        "confidence": cls_result.confidence,
                        "db_level": cls_result.db_level,
                        "raw_label": cls_result.label,
                    }
                    if cls_result.clap_verified is not None:
                        oo_fields["clap_verified"] = cls_result.clap_verified
                    if cls_result.clap_score is not None:
                        oo_fields["clap_score"] = cls_result.clap_score
                    if cls_result.clap_label is not None:
                        oo_fields["clap_label"] = cls_result.clap_label
                    if cls_result.source != "ast":
                        oo_fields["source"] = cls_result.source
                    log_event("detection", camera=self._camera.name, **oo_fields)
                    # Report to consolidator for cross-camera dedup
                    if self._consolidator is not None:
                        self._consolidator.report_detection(
                            camera_name=self._camera.name,
                            group=cls_result.group,
                            confidence=cls_result.confidence,
                            trigger_time=trigger_time,
                        )
                    # Report to noise stress scorer
                    if self._noise_stress is not None:
                        self._noise_stress.record_event(
                            group=cls_result.group,
                            trigger_db=cls_result.db_level,
                            camera=self._camera.name,
                            confidence=cls_result.confidence,
                        )

                # LLM judge: fire-and-forget (outside semaphore — I/O-bound)
                if self._llm_judge is not None and self._llm_judge.should_sample():
                    if not self._judge_semaphore.locked():
                        asyncio.create_task(  # noqa: RUF006
                            self._run_judge(audio, classifications),
                            name=f"llm-judge-{self._camera.name}",
                        )
                    else:
                        logger.debug(
                            "[%s] LLM judge backlog full, skipping",
                            self._camera.name,
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
        clap_verifier: CLAPVerifier | None = None,
        llm_judge: LLMJudge | None = None,
        consolidator: EventConsolidator | None = None,
        noise_stress: NoiseStressScorer | None = None,
        resolver: ScryptedUrlResolver | None = None,
        auto_discovery: bool = False,
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
                clap_verifier=clap_verifier,
                llm_judge=llm_judge,
                consolidator=consolidator,
                noise_stress=noise_stress,
                resolver=resolver,
                auto_discovery=auto_discovery,
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
