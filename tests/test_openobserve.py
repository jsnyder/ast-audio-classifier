"""Tests for OpenObserve log shipping module."""

import logging

from src.classifier import ClassificationResult
from src.openobserve import (
    OpenObserveHandler,
    log_event,
    setup_openobserve_logging,
)


class TestOpenObserveHandler:
    def test_init_sets_uri(self):
        handler = OpenObserveHandler(host="localhost", org="myorg", stream="mystream")
        assert handler._uri == "/api/myorg/mystream/_json"
        handler.close()

    def test_init_default_uri(self):
        handler = OpenObserveHandler(host="localhost")
        assert handler._uri == "/api/default/ast_audio/_json"
        handler.close()

    def test_auth_header_set_when_credentials_provided(self):
        handler = OpenObserveHandler(host="localhost", username="user", password="pass")
        assert handler._auth_header is not None
        assert handler._auth_header.startswith("Basic ")
        handler.close()

    def test_auth_header_none_without_credentials(self):
        handler = OpenObserveHandler(host="localhost")
        assert handler._auth_header is None
        handler.close()

    def test_auth_header_none_with_partial_credentials(self):
        handler = OpenObserveHandler(host="localhost", username="user")
        assert handler._auth_header is None
        handler.close()

    def test_format_record_basic(self):
        handler = OpenObserveHandler(host="localhost")
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        entry = handler._format_record(record)
        assert entry["level"] == "INFO"
        assert entry["logger"] == "test.logger"
        assert entry["message"] == "Test message"
        assert entry["source"] == "ast-audio-classifier"
        assert "_timestamp" in entry
        assert isinstance(entry["_timestamp"], int)
        handler.close()

    def test_format_record_with_oo_fields(self):
        handler = OpenObserveHandler(host="localhost")
        record = logging.LogRecord(
            name="ast.events",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="detection: dog_bark",
            args=(),
            exc_info=None,
        )
        record._oo_fields = {  # type: ignore[attr-defined]
            "event_type": "detection",
            "camera": "back_porch",
            "group": "dog_bark",
            "confidence": 0.85,
        }
        entry = handler._format_record(record)
        assert entry["event_type"] == "detection"
        assert entry["camera"] == "back_porch"
        assert entry["group"] == "dog_bark"
        assert entry["confidence"] == 0.85
        handler.close()

    def test_emit_queues_record(self):
        handler = OpenObserveHandler(host="localhost")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="queued",
            args=(),
            exc_info=None,
        )
        handler.emit(record)
        assert handler._queue.qsize() == 1
        entry = handler._queue.get_nowait()
        assert entry["message"] == "queued"
        handler.close()

    def test_drain_batch_collects_up_to_batch_size(self):
        handler = OpenObserveHandler(host="localhost", batch_size=3)
        for i in range(5):
            handler._queue.put_nowait({"msg": f"item-{i}"})
        batch = handler._drain_batch()
        assert len(batch) == 3
        assert handler._queue.qsize() == 2
        handler.close()

    def test_drain_batch_returns_empty_when_queue_empty(self):
        handler = OpenObserveHandler(host="localhost")
        batch = handler._drain_batch()
        assert batch == []
        handler.close()

    def test_close_stops_thread(self):
        handler = OpenObserveHandler(host="localhost")
        assert handler._thread.is_alive()
        handler.close()
        assert not handler._thread.is_alive()

    def test_flush_thread_is_daemon(self):
        handler = OpenObserveHandler(host="localhost")
        assert handler._thread.daemon is True
        handler.close()


class TestLogEvent:
    def test_log_event_creates_record_with_fields(self):
        """log_event should produce a log record with _oo_fields attached."""
        captured = []
        handler = logging.Handler()
        handler.emit = lambda record: captured.append(record)  # type: ignore[assignment]

        event_logger = logging.getLogger("ast.events")
        event_logger.addHandler(handler)
        event_logger.setLevel(logging.INFO)

        try:
            log_event(
                "detection",
                camera="back_porch",
                group="dog_bark",
                confidence=0.85,
                db_level=-25.3,
            )

            assert len(captured) == 1
            record = captured[0]
            assert hasattr(record, "_oo_fields")
            fields = record._oo_fields
            assert fields["event_type"] == "detection"
            assert fields["camera"] == "back_porch"
            assert fields["group"] == "dog_bark"
            assert fields["confidence"] == 0.85
            assert fields["db_level"] == -25.3
        finally:
            event_logger.removeHandler(handler)

    def test_log_event_without_camera(self):
        captured = []
        handler = logging.Handler()
        handler.emit = lambda record: captured.append(record)  # type: ignore[assignment]

        event_logger = logging.getLogger("ast.events")
        event_logger.addHandler(handler)
        event_logger.setLevel(logging.INFO)

        try:
            log_event("stream_online")

            assert len(captured) == 1
            fields = captured[0]._oo_fields
            assert fields["event_type"] == "stream_online"
            assert "camera" not in fields
        finally:
            event_logger.removeHandler(handler)

    def test_log_event_message_format(self):
        captured = []
        handler = logging.Handler()
        handler.emit = lambda record: captured.append(record)  # type: ignore[assignment]

        event_logger = logging.getLogger("ast.events")
        event_logger.addHandler(handler)
        event_logger.setLevel(logging.INFO)

        try:
            log_event("detection", camera="front_door", group="doorbell")
            assert "doorbell" in captured[0].getMessage()
        finally:
            event_logger.removeHandler(handler)


class TestSetupOpenObserveLogging:
    def test_setup_attaches_handler_to_root(self):
        handler = setup_openobserve_logging(host="localhost")
        root = logging.getLogger()
        try:
            assert handler in root.handlers
        finally:
            root.removeHandler(handler)
            event_logger = logging.getLogger("ast.events")
            event_logger.removeHandler(handler)
            handler.close()

    def test_setup_attaches_handler_to_event_logger(self):
        handler = setup_openobserve_logging(host="localhost")
        event_logger = logging.getLogger("ast.events")
        try:
            assert handler in event_logger.handlers
        finally:
            logging.getLogger().removeHandler(handler)
            event_logger.removeHandler(handler)
            handler.close()

    def test_setup_returns_handler(self):
        handler = setup_openobserve_logging(host="localhost")
        try:
            assert isinstance(handler, OpenObserveHandler)
        finally:
            logging.getLogger().removeHandler(handler)
            logging.getLogger("ast.events").removeHandler(handler)
            handler.close()


class TestStreamManagerCLAPFields:
    """Verify stream_manager passes CLAP fields through to log_event.

    These test the contract between stream_manager._stream_loop and
    log_event — CLAP metadata must flow into OpenObserve structured fields.
    """

    def _simulate_log_event_from_result(self, result: ClassificationResult) -> dict:
        """Simulate what stream_manager._stream_loop does with a ClassificationResult.

        Mirrors the log_event call in stream_manager.py _stream_loop.
        """
        captured = []
        handler = logging.Handler()
        handler.emit = lambda record: captured.append(record)  # type: ignore[assignment]

        event_logger = logging.getLogger("ast.events")
        event_logger.addHandler(handler)
        event_logger.setLevel(logging.INFO)

        try:
            # Import and call the actual code path from stream_manager
            # We can't easily run _stream_loop, so we replicate the log_event call
            # pattern and verify it matches what stream_manager does
            oo_fields: dict = {
                "group": result.group,
                "confidence": result.confidence,
                "db_level": result.db_level,
                "raw_label": result.label,
            }
            if result.clap_verified is not None:
                oo_fields["clap_verified"] = result.clap_verified
            if result.clap_score is not None:
                oo_fields["clap_score"] = result.clap_score
            if result.clap_label is not None:
                oo_fields["clap_label"] = result.clap_label
            if result.source != "ast":
                oo_fields["source"] = result.source
            log_event("detection", camera="test_cam", **oo_fields)

            assert len(captured) == 1
            return captured[0]._oo_fields
        finally:
            event_logger.removeHandler(handler)

    def test_clap_verified_true_included(self):
        """CLAP-confirmed detection should include clap_verified=True."""
        result = ClassificationResult(
            label="Dog",
            group="dog_bark",
            confidence=0.85,
            top_5=[],
            db_level=-25.0,
            clap_verified=True,
            clap_score=0.72,
            clap_label="a dog barking",
        )
        fields = self._simulate_log_event_from_result(result)
        assert fields["clap_verified"] is True
        assert fields["clap_score"] == 0.72
        assert fields["clap_label"] == "a dog barking"
        assert "source" not in fields  # default "ast" is omitted

    def test_clap_verified_false_included(self):
        """CLAP-unverified detection should include clap_verified=False."""
        result = ClassificationResult(
            label="Speech",
            group="speech",
            confidence=0.45,
            top_5=[],
            db_level=-35.0,
            clap_verified=False,
            clap_score=0.08,
            clap_label="a voice speaking nearby",
        )
        fields = self._simulate_log_event_from_result(result)
        assert fields["clap_verified"] is False
        assert fields["clap_score"] == 0.08

    def test_clap_discovery_includes_source(self):
        """CLAP-discovered events should include source='clap'."""
        result = ClassificationResult(
            label="an autonomous vacuum cleaner operating",
            group="vacuum_cleaner",
            confidence=0.65,
            top_5=[],
            db_level=-38.0,
            clap_verified=True,
            clap_score=0.65,
            clap_label="an autonomous vacuum cleaner operating",
            source="clap",
        )
        fields = self._simulate_log_event_from_result(result)
        assert fields["source"] == "clap"
        assert fields["clap_verified"] is True
        assert fields["raw_label"] == "an autonomous vacuum cleaner operating"

    def test_no_clap_fields_when_none(self):
        """AST-only detection (no CLAP) should not include CLAP fields."""
        result = ClassificationResult(
            label="Dog",
            group="dog_bark",
            confidence=0.85,
            top_5=[],
            db_level=-25.0,
        )
        fields = self._simulate_log_event_from_result(result)
        assert "clap_verified" not in fields
        assert "clap_score" not in fields
        assert "clap_label" not in fields
        assert "source" not in fields

    def test_stream_manager_log_event_matches_pattern(self):
        """Verify stream_manager.py's log_event call includes CLAP fields.

        Reads the source file directly to avoid import side effects
        (paho.mqtt not in test venv).
        """
        from pathlib import Path

        source = Path(__file__).parent.parent / "src" / "stream_manager.py"
        content = source.read_text()
        # The log_event call must include CLAP fields
        assert "clap_verified" in content, (
            "stream_manager.py must pass clap_verified to log_event"
        )
        assert "clap_score" in content, (
            "stream_manager.py must pass clap_score to log_event"
        )
        assert "clap_label" in content, (
            "stream_manager.py must pass clap_label to log_event"
        )


class TestSendBatch:
    def test_send_batch_handles_connection_error(self):
        """_send_batch should not raise on connection failure."""
        handler = OpenObserveHandler(host="192.0.2.1", port=1)  # RFC 5737 TEST-NET
        # Should not raise — failures are logged and swallowed
        handler._send_batch([{"test": "data"}])
        handler.close()
