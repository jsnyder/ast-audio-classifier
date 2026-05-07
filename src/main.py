"""FastAPI application with health endpoints and startup sequence.

Loads config, initializes the AST model, connects MQTT,
starts per-camera stream tasks, and serves the health API.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import os
import time
from datetime import UTC

import numpy as np
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import JSONResponse

from . import __version__
from .classifier import ASTClassifier
from .config import load_config
from .labels import LABEL_GROUPS
from .mqtt_publisher import MqttPublisher
from .stream_manager import StreamManager

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


def create_app(config_path: str | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="AST Audio Classifier", version=__version__)

    if config_path is None:
        config_path = os.environ.get("CONFIG_PATH", "/config/config.yaml")

    @app.on_event("startup")
    async def startup() -> None:
        log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
        logging.basicConfig(
            level=getattr(logging, log_level, logging.INFO),
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        )

        logger.info("AST Audio Classifier v%s starting", __version__)

        config = load_config(config_path)
        logger.info(
            "Config loaded: %d cameras, MQTT=%s:%s",
            len(config.cameras),
            config.mqtt.host,
            config.mqtt.port,
        )

        # Optional: OpenObserve structured logging
        if config.openobserve:
            from .openobserve import setup_openobserve_logging

            app.state.oo_handler = setup_openobserve_logging(
                host=config.openobserve.host,
                port=config.openobserve.port,
                org=config.openobserve.org,
                stream=config.openobserve.stream,
                username=config.openobserve.username,
                password=config.openobserve.password,
            )

        # Load AST model (slow — ~30-120s on modest hardware)
        app.state.classifier = ASTClassifier()

        # Load CLAP verifier (optional, ~150MB model)
        app.state.clap_verifier = None
        if config.clap and config.clap.enabled:
            from .clap_verifier import (
                DEFAULT_NEVER_SUPPRESS,
                CLAPVerifier,
            )
            from .clap_verifier import (
                CLAPConfig as CLAPVerifierConfig,
            )

            never_suppress = (
                frozenset(config.clap.never_suppress)
                if config.clap.never_suppress
                else DEFAULT_NEVER_SUPPRESS
            )
            clap_cfg = CLAPVerifierConfig(
                enabled=True,
                model=config.clap.model,
                confirm_threshold=config.clap.confirm_threshold,
                suppress_threshold=config.clap.suppress_threshold,
                override_threshold=config.clap.override_threshold,
                discovery_threshold=config.clap.discovery_threshold,
                confirm_margin=config.clap.confirm_margin,
                ast_bypass_threshold=config.clap.ast_bypass_threshold,
                never_suppress=never_suppress,
                custom_prompts=config.clap.custom_prompts,
            )
            app.state.clap_verifier = CLAPVerifier(clap_cfg)
            logger.info("CLAP verifier loaded")

        # Optional: LLM Judge for ground-truth evaluation
        app.state.llm_judge = None
        if config.llm_judge and config.llm_judge.enabled:
            from .llm_judge import LLMJudge

            app.state.llm_judge = LLMJudge(config.llm_judge)
            logger.info(
                "LLM Judge enabled (model=%s, sample_rate=%.0f%%)",
                config.llm_judge.model,
                config.llm_judge.sample_rate * 100,
            )

        # Connect MQTT and publish discovery
        app.state.publisher = MqttPublisher(config)
        app.state.publisher.connect()

        # Wait for MQTT connection via event (timeout 5s)
        try:
            await asyncio.wait_for(app.state.publisher.connected_event.wait(), timeout=5.0)
        except TimeoutError:
            logger.error("Failed to connect to MQTT broker within 5s")

        # Optional: Event consolidator for cross-camera dedup
        app.state.consolidator = None
        if config.consolidated_enabled:
            from .event_consolidator import EventConsolidator

            def _on_consolidated(group, episode):
                from datetime import datetime, timedelta

                mono_now = time.monotonic()
                utc_now = datetime.now(UTC)
                first_detected_iso = (
                    utc_now - timedelta(seconds=(mono_now - episode.first_detected))
                ).isoformat()
                last_detected_iso = (
                    utc_now - timedelta(seconds=(mono_now - episode.last_detected))
                ).isoformat()
                duration = episode.last_detected - episode.first_detected

                app.state.publisher.publish_consolidated_event(
                    group=group,
                    cameras=list(episode.cameras),
                    max_confidence=round(episode.max_confidence, 4),
                    detection_count=episode.detection_count,
                    duration_seconds=duration,
                    first_detected=first_detected_iso,
                    last_detected=last_detected_iso,
                )

            app.state.consolidator = EventConsolidator(
                window_seconds=config.consolidated_window_seconds,
                auto_off_seconds=config.auto_off_seconds,
                on_consolidated=_on_consolidated,
            )
            # Publish consolidated discovery
            app.state.publisher.publish_consolidated_discovery(
                app.state.consolidator.auto_off_seconds
            )

            # Periodic cleanup task
            async def _cleanup_loop():
                while True:
                    await asyncio.sleep(10)
                    app.state.consolidator.cleanup_stale()

            app.state.cleanup_task = asyncio.create_task(
                _cleanup_loop(), name="consolidator-cleanup"
            )
            logger.info(
                "Event consolidator enabled (window=%.1fs)",
                config.consolidated_window_seconds,
            )

        # Optional: Noise stress scorer
        app.state.noise_stress = None
        app.state.noise_stress_task = None
        if config.noise_stress and config.noise_stress.enabled:
            from .noise_stress import NoiseStressScorer

            indoor = frozenset(config.noise_stress.indoor_cameras or [])
            app.state.noise_stress = NoiseStressScorer(
                half_life=config.noise_stress.decay_half_life_seconds,
                saturation=config.noise_stress.saturation_constant,
                indoor_cameras=indoor,
                update_interval=config.noise_stress.update_interval_seconds,
            )
            app.state.publisher.publish_noise_stress_discovery()

            interval = config.noise_stress.update_interval_seconds

            async def _noise_stress_loop():
                while True:
                    await asyncio.sleep(interval)
                    try:
                        sm = getattr(app.state, "stream_manager", None)
                        ambient_data = {}
                        if sm:
                            for stream in sm.streams:
                                ambient_data[stream.camera_name] = stream.ambient_info
                        score_data = app.state.noise_stress.compute(ambient_data)
                        app.state.publisher.publish_noise_stress_score(score_data)
                    except Exception:
                        logger.exception("Error in noise stress update loop")

            app.state.noise_stress_task = asyncio.create_task(
                _noise_stress_loop(), name="noise-stress-update"
            )
            logger.info(
                "Noise stress scorer enabled (interval=%.0fs, half_life=%.0fs)",
                interval,
                config.noise_stress.decay_half_life_seconds,
            )

        # Optional: Confounder monitor for per-camera context-aware tagging
        app.state.confounder_monitor = None
        cameras_with_confounders = [c for c in config.cameras if c.confounders]
        if cameras_with_confounders:
            from .confounder_monitor import ConfounderMonitor

            app.state.confounder_monitor = ConfounderMonitor(config.cameras)
            await app.state.confounder_monitor.start()
            logger.info(
                "Confounder monitor enabled for %d cameras",
                len(cameras_with_confounders),
            )

        # Optional: Weather prior for dynamic outdoor threshold adjustment
        app.state.weather_prior = None
        if config.weather_entity:
            from .weather_prior import WeatherPrior

            app.state.weather_prior = WeatherPrior(
                entity_id=config.weather_entity,
                poll_interval=300.0,
            )
            await app.state.weather_prior.start()
            logger.info("Weather prior enabled: %s", config.weather_entity)

        # Optional: Scrypted API resolver for direct RTSP URL discovery
        app.state.resolver = None
        if config.scrypted_api_url:
            from .url_resolver import ScryptedApiResolver

            app.state.resolver = ScryptedApiResolver(config.scrypted_api_url)
            logger.info("Scrypted API resolver enabled: %s", config.scrypted_api_url)

        # Start camera streams
        app.state.stream_manager = StreamManager(
            cameras=config.cameras,
            classifier=app.state.classifier,
            publisher=app.state.publisher,
            confidence_threshold=config.confidence_threshold,
            clip_duration=config.clip_duration_seconds,
            clap_verifier=app.state.clap_verifier,
            llm_judge=app.state.llm_judge,
            consolidator=app.state.consolidator,
            noise_stress=app.state.noise_stress,
            confounder_monitor=app.state.confounder_monitor,
            resolver=app.state.resolver,
            groups_config=config.groups,
            weather_prior=app.state.weather_prior,
        )
        app.state.stream_manager.start_all()
        app.state.start_time = time.monotonic()
        logger.info("All %d camera streams started", len(config.cameras))

    @app.on_event("shutdown")
    async def shutdown() -> None:
        cleanup_task = getattr(app.state, "cleanup_task", None)
        if cleanup_task and not cleanup_task.done():
            cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await cleanup_task
        noise_stress_task = getattr(app.state, "noise_stress_task", None)
        if noise_stress_task and not noise_stress_task.done():
            noise_stress_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await noise_stress_task
        wp = getattr(app.state, "weather_prior", None)
        if wp:
            await wp.stop()
        cm = getattr(app.state, "confounder_monitor", None)
        sm = getattr(app.state, "stream_manager", None)
        pub = getattr(app.state, "publisher", None)
        oo = getattr(app.state, "oo_handler", None)
        if cm:
            await cm.stop()
        if sm:
            await sm.stop_all()
        if pub:
            pub.disconnect()
        if oo:
            oo.close()
        logger.info("Shutdown complete")

    # Grace period after startup before requiring active streams
    stream_grace_period = 180  # seconds
    # Max time without any stream producing audio before unhealthy
    stream_stale_threshold = 120  # seconds

    @app.get("/health")
    async def health(request: Request) -> JSONResponse:
        clf = getattr(request.app.state, "classifier", None)
        pub = getattr(request.app.state, "publisher", None)
        sm = getattr(request.app.state, "stream_manager", None)
        clap = getattr(request.app.state, "clap_verifier", None)
        llm_judge = getattr(request.app.state, "llm_judge", None)
        start = getattr(request.app.state, "start_time", 0)

        model_loaded = clf is not None and clf.loaded
        mqtt_connected = pub is not None and pub.connected
        clap_loaded = clap is not None and clap.loaded

        # Check stream health: at least one camera producing audio recently
        now = time.monotonic()
        uptime = now - start if start else 0
        streams_active = 0
        streams_total = 0
        if sm:
            streams_total = len(sm.streams)
            for stream in sm.streams:
                last = stream.last_chunk_time
                if last > 0 and (now - last) < stream_stale_threshold:
                    streams_active += 1

        # After grace period, require at least one active stream
        past_grace = uptime > stream_grace_period
        streams_healthy = streams_active > 0 if past_grace else True

        healthy = model_loaded and mqtt_connected and streams_healthy

        return JSONResponse(
            status_code=200 if healthy else 503,
            content={
                "status": "healthy" if healthy else "unhealthy",
                "model_loaded": model_loaded,
                "clap_loaded": clap_loaded,
                "llm_judge_enabled": llm_judge is not None,
                "mqtt_connected": mqtt_connected,
                "streams_active": streams_active,
                "streams_total": streams_total,
                "streams_healthy": streams_healthy,
                "uptime_seconds": round(uptime, 1),
                "version": __version__,
            },
        )

    @app.get("/status")
    async def status(request: Request) -> JSONResponse:
        start = getattr(request.app.state, "start_time", 0)
        clf = getattr(request.app.state, "classifier", None)
        clap = getattr(request.app.state, "clap_verifier", None)
        llm_judge_status = getattr(request.app.state, "llm_judge", None)
        pub = getattr(request.app.state, "publisher", None)
        sm = getattr(request.app.state, "stream_manager", None)

        uptime = time.monotonic() - start if start else 0
        cameras = sm.status() if sm else []
        online = sum(1 for c in cameras if c["state"] == "streaming")

        # Include ambient dB info per camera for diagnostics
        ambient_info = {}
        if sm:
            for stream in sm.streams:
                ambient_info[stream.camera_name] = stream.ambient_info

        # Noise stress status
        ns = getattr(request.app.state, "noise_stress", None)

        content = {
            "version": __version__,
            "uptime_seconds": round(uptime, 1),
            "model_loaded": clf is not None and clf.loaded,
            "clap_loaded": clap is not None and clap.loaded,
            "llm_judge_enabled": llm_judge_status is not None,
            "mqtt_connected": pub is not None and pub.connected,
            "cameras": cameras,
            "cameras_online": f"{online}/{len(cameras)}",
            "total_inferences": sum(c["inference_count"] for c in cameras),
            "label_groups_count": len(LABEL_GROUPS),
            "ambient": ambient_info,
        }
        if ns is not None:
            content["noise_stress"] = ns.status()

        return JSONResponse(content=content)

    @app.get("/status/cameras")
    async def status_cameras(request: Request) -> JSONResponse:
        sm = getattr(request.app.state, "stream_manager", None)
        cameras = sm.status() if sm else []
        return JSONResponse(content={"cameras": cameras})

    @app.post("/classify")
    async def classify_upload(file: UploadFile, request: Request) -> JSONResponse:
        """Manual classification endpoint for testing.

        Upload a WAV or raw PCM file and get classification results.
        """
        clf = getattr(request.app.state, "classifier", None)
        if clf is None:
            return JSONResponse(status_code=503, content={"error": "Model not loaded"})

        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            return JSONResponse(status_code=413, content={"error": "File too large (max 10MB)"})

        # Try to detect WAV header
        if content[:4] == b"RIFF":
            import wave

            try:
                with wave.open(io.BytesIO(content)) as wav:
                    frames = wav.readframes(wav.getnframes())
                    framerate = wav.getframerate()
            except (wave.Error, EOFError) as e:
                return JSONResponse(status_code=400, content={"error": f"Invalid WAV file: {e}"})
            audio_int16 = np.frombuffer(frames, dtype=np.int16)
            audio = audio_int16.astype(np.float32) / 32768.0
            if framerate != 16000:
                import librosa

                audio = librosa.resample(
                    y=audio,
                    orig_sr=framerate,
                    target_sr=16000,
                )
        else:
            # Assume raw 16-bit PCM at 16kHz
            audio_int16 = np.frombuffer(content, dtype=np.int16)
            audio = audio_int16.astype(np.float32) / 32768.0

        from .audio_pipeline import compute_rms_db

        db = compute_rms_db(
            np.frombuffer(content[:3200], dtype=np.int16)
            if len(content) >= 3200
            else np.frombuffer(content, dtype=np.int16)
        )

        results = await asyncio.to_thread(
            clf.classify,
            audio,
            db,
            0.05,  # Low threshold for testing
        )

        # CLAP verification if available
        clap_v = getattr(request.app.state, "clap_verifier", None)
        if results and clap_v is not None:
            results = await asyncio.to_thread(clap_v.verify, audio, results, "upload")

        return JSONResponse(
            content={
                "results": [r.to_dict() for r in results],
                "audio_duration_seconds": round(len(audio) / 16000, 2),
                "db_level": round(db, 1),
            }
        )

    return app
