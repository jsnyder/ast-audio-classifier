"""FastAPI application with health endpoints and startup sequence.

Loads config, initializes the AST model, connects MQTT,
starts per-camera stream tasks, and serves the health API.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import time

import numpy as np
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import JSONResponse

from . import __version__
from .classifier import ASTClassifier
from .config import load_config
from .mqtt_publisher import MqttPublisher
from .labels import LABEL_GROUPS
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

        # Connect MQTT and publish discovery
        app.state.publisher = MqttPublisher(config)
        app.state.publisher.connect()

        # Wait for MQTT connection via event (timeout 5s)
        try:
            await asyncio.wait_for(
                app.state.publisher.connected_event.wait(), timeout=5.0
            )
        except asyncio.TimeoutError:
            logger.error("Failed to connect to MQTT broker within 5s")

        # Start camera streams
        app.state.stream_manager = StreamManager(
            cameras=config.cameras,
            classifier=app.state.classifier,
            publisher=app.state.publisher,
            confidence_threshold=config.confidence_threshold,
            clip_duration=config.clip_duration_seconds,
        )
        app.state.stream_manager.start_all()
        app.state.start_time = time.monotonic()
        logger.info("All %d camera streams started", len(config.cameras))

    @app.on_event("shutdown")
    async def shutdown() -> None:
        sm = getattr(app.state, "stream_manager", None)
        pub = getattr(app.state, "publisher", None)
        oo = getattr(app.state, "oo_handler", None)
        if sm:
            await sm.stop_all()
        if pub:
            pub.disconnect()
        if oo:
            oo.close()
        logger.info("Shutdown complete")

    @app.get("/health")
    async def health(request: Request) -> JSONResponse:
        clf = getattr(request.app.state, "classifier", None)
        pub = getattr(request.app.state, "publisher", None)
        model_loaded = clf is not None and clf.loaded
        mqtt_connected = pub is not None and pub.connected
        healthy = model_loaded and mqtt_connected

        return JSONResponse(
            status_code=200 if healthy else 503,
            content={
                "status": "healthy" if healthy else "unhealthy",
                "model_loaded": model_loaded,
                "mqtt_connected": mqtt_connected,
                "version": __version__,
            },
        )

    @app.get("/status")
    async def status(request: Request) -> JSONResponse:
        start = getattr(request.app.state, "start_time", 0)
        clf = getattr(request.app.state, "classifier", None)
        pub = getattr(request.app.state, "publisher", None)
        sm = getattr(request.app.state, "stream_manager", None)

        uptime = time.monotonic() - start if start else 0
        cameras = sm.status() if sm else []
        online = sum(1 for c in cameras if c["state"] == "streaming")

        # Include ambient dB info per camera for diagnostics
        ambient_info = {}
        if sm:
            for stream in sm.streams:
                amb = stream._ambient
                ambient_info[stream.camera_name] = {
                    "peak_db": round(amb.peak_db, 1),
                    "chunk_count": amb.chunk_count,
                    "threshold": stream._camera.db_threshold,
                }

        return JSONResponse(
            content={
                "version": __version__,
                "uptime_seconds": round(uptime, 1),
                "model_loaded": clf is not None and clf.loaded,
                "mqtt_connected": pub is not None and pub.connected,
                "cameras": cameras,
                "cameras_online": f"{online}/{len(cameras)}",
                "total_inferences": sum(c["inference_count"] for c in cameras),
                "label_groups_count": len(LABEL_GROUPS),
                "ambient": ambient_info,
            }
        )

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
            return JSONResponse(
                status_code=413, content={"error": "File too large (max 10MB)"}
            )

        # Try to detect WAV header
        if content[:4] == b"RIFF":
            import wave

            wav = wave.open(io.BytesIO(content))
            frames = wav.readframes(wav.getnframes())
            audio_int16 = np.frombuffer(frames, dtype=np.int16)
            audio = audio_int16.astype(np.float32) / 32768.0
            if wav.getframerate() != 16000:
                import librosa

                audio = librosa.resample(
                    y=audio,
                    orig_sr=wav.getframerate(),
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

        return JSONResponse(
            content={
                "results": [r.to_dict() for r in results],
                "audio_duration_seconds": round(len(audio) / 16000, 2),
                "db_level": round(db, 1),
            }
        )

    return app
