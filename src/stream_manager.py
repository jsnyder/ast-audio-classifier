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

from dataclasses import replace

from .audio_pipeline import AmbientMonitor, compute_spectral_flatness, read_audio_clip, start_ffmpeg
from .clap_verifier import CLAPVerifier
from .classifier import ASTClassifier
from .config import CameraConfig
from .mqtt_publisher import MqttPublisher
from .openobserve import log_event

from .url_resolver import ScryptedApiResolver

if TYPE_CHECKING:
    from .confounder_monitor import ConfounderMonitor
    from .event_consolidator import EventConsolidator
    from .llm_judge import LLMJudge
    from .noise_stress import NoiseStressScorer
    from .url_resolver import ScryptedUrlResolver

logger = logging.getLogger(__name__)

_CRED_RE = re.compile(r"://([^:]+):([^@]+)@")

MAX_BACKOFF = 60  # seconds
DISCOVERY_THRESHOLD = 3  # consecutive short-lived failures before URL discovery
STABLE_STREAM_SECONDS = 30  # stream must last this long to be considered stable


STUCK_THRESHOLD_SECONDS = 300  # 5 minutes of continuous failure → stuck


class StreamState(enum.Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    STREAMING = "streaming"
    COOLDOWN = "cooldown"
    ERROR = "error"
    BACKOFF = "backoff"
    DISCOVERING = "discovering"
    STUCK = "stuck"


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
        resolver: ScryptedApiResolver | ScryptedUrlResolver | None = None,
        auto_discovery: bool = False,
        confounder_monitor: ConfounderMonitor | None = None,
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
        self._confounder_monitor = confounder_monitor

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
        # Stuck-state tracking
        self._failure_start: float = 0.0  # monotonic time of first failure in current run
        self._is_stuck = False
        self._fresh_discovery = False  # set when discovery just found a new URL

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
        """Return the URL to connect to."""
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
                await asyncio.wait_for(self._process.wait(), timeout=2.0)
        self._publisher.publish_camera_offline(self._camera.name)

    async def _attempt_discovery(self) -> None:
        """Try to discover a fresh RTSP URL via the resolver.

        Resolution chain:
        1. ScryptedApiResolver (if resolver has .resolve(device_id) and
           scrypted_device_id is configured) — authoritative, live URL
        2. go2rtc stable proxy fallback (if go2rtc_stream is configured)
        3. Revert to original URL

        For cameras with go2rtc_stream, discovery never reverts to a direct
        camera URL (prevents competing connections).
        """
        if not self._auto_discovery or self._resolver is None:
            return

        self._state = StreamState.DISCOVERING
        logger.info(
            "[%s] Attempting URL discovery (current: %s)",
            self._camera.name,
            _CRED_RE.sub(r"://\1:***@", self._effective_url),
        )

        resolved = None

        # Step 1: Try ScryptedApiResolver with device ID
        if self._camera.scrypted_device_id is not None and isinstance(
            self._resolver, ScryptedApiResolver
        ):
            try:
                resolved = await self._resolver.resolve(
                    self._camera.scrypted_device_id,
                )
            except Exception:
                logger.debug(
                    "[%s] Scrypted API discovery failed",
                    self._camera.name,
                    exc_info=True,
                )

        # Step 2: Fall back to go2rtc stable proxy if camera has a stream name
        if resolved is None and self._camera.go2rtc_stream:
            fallback_url = (
                f"rtsp://a889bffc-go2rtc:8554/{self._camera.go2rtc_stream}"
            )
            logger.info(
                "[%s] Falling back to go2rtc proxy: %s",
                self._camera.name,
                fallback_url,
            )
            resolved = fallback_url

        if resolved is not None:
            safe = _CRED_RE.sub(r"://\1:***@", resolved)
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

    async def _send_stuck_notification(self) -> None:
        """Send a HA persistent notification when stream enters stuck state."""
        try:
            import json
            import os
            from urllib.request import Request, urlopen

            token = os.environ.get("SUPERVISOR_TOKEN")
            if not token:
                logger.debug("No SUPERVISOR_TOKEN, skipping stuck notification")
                return

            data = json.dumps({
                "title": f"AST Audio: {self._camera.name} stream stuck",
                "message": (
                    f"Camera **{self._camera.name}** audio stream has been failing "
                    f"for over {STUCK_THRESHOLD_SECONDS // 60} minutes. "
                    f"Check Scrypted/RTSP source."
                ),
                "notification_id": f"ast_stream_stuck_{self._camera.name}",
            }).encode()

            req = Request(
                "http://supervisor/core/api/services/persistent_notification/create",
                data=data,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            await asyncio.to_thread(urlopen, req, timeout=5)
            logger.info("[%s] Sent stuck notification to HA", self._camera.name)
        except Exception:
            logger.debug("[%s] Failed to send stuck notification", self._camera.name, exc_info=True)

    async def _clear_stuck_notification(self) -> None:
        """Clear the HA persistent notification when stream recovers."""
        try:
            import json
            import os
            from urllib.request import Request, urlopen

            token = os.environ.get("SUPERVISOR_TOKEN")
            if not token:
                return

            data = json.dumps({
                "notification_id": f"ast_stream_stuck_{self._camera.name}",
            }).encode()

            req = Request(
                "http://supervisor/core/api/services/persistent_notification/dismiss",
                data=data,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            await asyncio.to_thread(urlopen, req, timeout=5)
            logger.info("[%s] Cleared stuck notification", self._camera.name)
        except Exception:
            logger.debug("[%s] Failed to clear stuck notification", self._camera.name, exc_info=True)

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
                        with contextlib.suppress(TimeoutError, ProcessLookupError):
                            await asyncio.wait_for(self._process.wait(), timeout=2.0)
                    self._process = None
                self._publisher.publish_camera_offline(self._camera.name)

            # Track stream stability for discovery decisions
            stream_duration = time.monotonic() - stream_start
            if stream_duration >= STABLE_STREAM_SECONDS:
                # Stream was stable — reset failure counter, backoff, and stuck state
                self._consecutive_failures = 0
                self._failure_start = 0.0
                self._backoff = self._camera.reconnect_interval
                if self._is_stuck:
                    self._is_stuck = False
                    self._backoff = self._camera.reconnect_interval
                    logger.info("[%s] Stream recovered from stuck state", self._camera.name)
                    log_event("stream_recovered", camera=self._camera.name)
                    self._publisher.publish_camera_online(self._camera.name)
                    await self._clear_stuck_notification()
            else:
                self._consecutive_failures += 1
                if self._failure_start == 0.0:
                    self._failure_start = time.monotonic()
                logger.warning(
                    "[%s] Stream died after %.1fs (< %ds), failure %d/%d",
                    self._camera.name,
                    stream_duration,
                    STABLE_STREAM_SECONDS,
                    self._consecutive_failures,
                    DISCOVERY_THRESHOLD,
                )

            # Trigger URL discovery after repeated short-lived streams,
            # or on every attempt when stuck (Scrypted ephemeral sessions
            # require fresh URLs right before each connection).
            self._fresh_discovery = False
            if self._consecutive_failures >= DISCOVERY_THRESHOLD or self._is_stuck:
                old_url = self._effective_url
                await self._attempt_discovery()
                if not self._is_stuck:
                    self._consecutive_failures = 0
                # If discovery found a new URL, try it immediately —
                # Scrypted creates on-demand rebroadcast sessions that
                # expire within seconds if nobody connects.
                if self._effective_url != old_url:
                    self._fresh_discovery = True
                    self._backoff = 1

            # Check for stuck state (continuous failure for too long)
            if (
                self._failure_start > 0
                and time.monotonic() - self._failure_start >= STUCK_THRESHOLD_SECONDS
                and not self._is_stuck
            ):
                self._is_stuck = True
                self._state = StreamState.STUCK
                if not self._fresh_discovery:
                    self._backoff = MAX_BACKOFF
                logger.error(
                    "[%s] Stream STUCK — failing for %.0fs",
                    self._camera.name,
                    time.monotonic() - self._failure_start,
                )
                log_event(
                    "stream_stuck",
                    camera=self._camera.name,
                    failure_duration=round(time.monotonic() - self._failure_start, 1),
                )
                await self._send_stuck_notification()

            # Backoff before reconnect
            if not self._is_stuck:
                self._state = StreamState.BACKOFF
            logger.info(
                "[%s] Reconnecting in %ds%s",
                self._camera.name,
                self._backoff,
                " (STUCK)" if self._is_stuck else "",
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

            # Spectral flatness check: skip pure tonal artifacts (codec/AGC)
            FLATNESS_ARTIFACT_THRESHOLD = 0.05
            flatness = compute_spectral_flatness(audio)
            if flatness < FLATNESS_ARTIFACT_THRESHOLD:
                logger.debug(
                    "[%s] Skipping tonal artifact (flatness=%.4f)",
                    self._camera.name, flatness,
                )
                log_event(
                    "artifact_skipped",
                    camera=self._camera.name,
                    flatness=round(flatness, 4),
                    db_level=round(trigger_db, 1),
                )
                continue

            # Classify with semaphore (one inference at a time)
            async with self._semaphore:
                classifications = await asyncio.to_thread(
                    self._classifier.classify,
                    audio,
                    trigger_db,
                    self._confidence_threshold,
                )
                # Filter out disabled groups for this camera
                if classifications and self._camera.disabled_groups:
                    disabled = set(self._camera.disabled_groups)
                    classifications = [
                        c for c in classifications if c.group not in disabled
                    ]
                # CLAP verification (inside semaphore — sequential with AST)
                suppressed: list = []
                if classifications and self._clap_verifier is not None:
                    classifications = await asyncio.to_thread(
                        self._clap_verifier.verify,
                        audio,
                        classifications,
                        self._camera.name,
                    )
                    suppressed = list(self._clap_verifier.last_suppressed)

            # Tag classifications with confounder context
            if classifications and self._confounder_monitor:
                confused = self._confounder_monitor.get_confused_groups(
                    self._camera.name
                )
                if confused:
                    tagged = []
                    for cls_result in classifications:
                        if cls_result.group in confused:
                            ctx = self._confounder_monitor.get_confounder_context(
                                self._camera.name, cls_result.group
                            )
                            cls_result = replace(
                                cls_result,
                                confounded=True,
                                confounder_entity=ctx["entity_id"] if ctx else None,
                            )
                        tagged.append(cls_result)
                    classifications = tagged

            if classifications or suppressed:
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
                    if cls_result.confounded:
                        oo_fields["confounded"] = True
                        if cls_result.confounder_entity:
                            oo_fields["confounder"] = cls_result.confounder_entity
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

                # Log suppressed events to OpenObserve for analysis
                for sup_result in suppressed:
                    log_event(
                        "suppressed",
                        camera=self._camera.name,
                        group=sup_result.group,
                        confidence=sup_result.confidence,
                        db_level=sup_result.db_level,
                        raw_label=sup_result.label,
                        clap_score=sup_result.clap_score,
                        clap_label=sup_result.clap_label,
                    )

                # LLM judge: evaluate both verified and suppressed events
                if self._llm_judge is not None and self._llm_judge.should_sample():
                    all_for_judge = classifications + suppressed
                    if not self._judge_semaphore.locked():
                        asyncio.create_task(  # noqa: RUF006
                            self._run_judge(audio, all_for_judge),
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
        resolver: ScryptedApiResolver | ScryptedUrlResolver | None = None,
        auto_discovery: bool = False,
        confounder_monitor: ConfounderMonitor | None = None,
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
                confounder_monitor=confounder_monitor,
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
