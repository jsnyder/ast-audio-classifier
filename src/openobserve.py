"""OpenObserve direct log shipping via HTTP JSON API.

Sends structured log events to OpenObserve in batches.
Classification events get first-class fields (camera, group, confidence, db_level)
that are directly queryable in OpenObserve without VRL parsing.

Fluent Bit already captures stdout for basic logs — this module adds
structured event telemetry on top.
"""

from __future__ import annotations

import atexit
import json
import logging
import threading
from base64 import b64encode
from http.client import HTTPConnection
from queue import Empty, Queue

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 50
DEFAULT_FLUSH_INTERVAL = 5.0  # seconds


class OpenObserveHandler(logging.Handler):
    """Python logging handler that ships structured JSON to OpenObserve.

    Batches log records and flushes them periodically or when the batch is full.
    Uses a background thread to avoid blocking the event loop.
    """

    def __init__(
        self,
        host: str,
        port: int = 5080,
        org: str = "default",
        stream: str = "ast_audio",
        username: str | None = None,
        password: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        flush_interval: float = DEFAULT_FLUSH_INTERVAL,
    ) -> None:
        super().__init__()
        self._host = host
        self._port = port
        self._uri = f"/api/{org}/{stream}/_json"
        self._batch_size = batch_size
        self._flush_interval = flush_interval

        self._auth_header: str | None = None
        if username and password:
            creds = b64encode(f"{username}:{password}".encode()).decode()
            self._auth_header = f"Basic {creds}"

        self._queue: Queue[dict] = Queue(maxsize=1000)
        self._shutdown = threading.Event()
        self._thread = threading.Thread(
            target=self._flush_loop, daemon=True, name="oo-flush"
        )
        self._thread.start()
        atexit.register(self.close)

    def emit(self, record: logging.LogRecord) -> None:
        """Queue a log record for batch shipping."""
        try:
            entry = self._format_record(record)
            self._queue.put_nowait(entry)
        except Exception:
            self.handleError(record)

    def _format_record(self, record: logging.LogRecord) -> dict:
        """Convert a LogRecord to an OpenObserve-friendly dict."""
        entry = {
            "_timestamp": int(record.created * 1_000_000),  # microseconds
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "source": "ast-audio-classifier",
        }

        # Merge any structured fields attached to the record
        # (e.g., from log_event() calls)
        extra = getattr(record, "_oo_fields", None)
        if extra:
            entry.update(extra)

        return entry

    def _flush_loop(self) -> None:
        """Background thread: drain queue and send batches."""
        while not self._shutdown.is_set():
            batch = self._drain_batch()
            if batch:
                self._send_batch(batch)
            self._shutdown.wait(timeout=self._flush_interval)

        # Final flush on shutdown
        batch = self._drain_batch()
        if batch:
            self._send_batch(batch)

    def _drain_batch(self) -> list[dict]:
        """Collect up to batch_size items from the queue."""
        batch: list[dict] = []
        while len(batch) < self._batch_size:
            try:
                batch.append(self._queue.get_nowait())
            except Empty:
                break
        return batch

    def _send_batch(self, batch: list[dict]) -> None:
        """Send a batch of records to OpenObserve via HTTP."""
        body = json.dumps(batch).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
        }
        if self._auth_header:
            headers["Authorization"] = self._auth_header

        try:
            conn = HTTPConnection(self._host, self._port, timeout=10)
            conn.request("POST", self._uri, body=body, headers=headers)
            resp = conn.getresponse()
            resp.read()  # drain response
            if resp.status >= 400:
                logger.warning(
                    "OpenObserve batch rejected: %s %s",
                    resp.status,
                    resp.reason,
                )
            conn.close()
        except Exception:
            logger.debug("Failed to send batch to OpenObserve", exc_info=True)

    def close(self) -> None:
        """Shut down the flush thread and send remaining records."""
        self._shutdown.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)
        super().close()


def log_event(
    event_type: str,
    camera: str | None = None,
    **fields: object,
) -> None:
    """Log a structured classification event to OpenObserve.

    These records get first-class queryable fields in OpenObserve:
      event_type, camera, group, confidence, db_level, raw_label, etc.

    Example:
        log_event("detection", camera="back_porch", group="dog_bark",
                  confidence=0.85, db_level=-25.3, raw_label="Dog")
    """
    oo_logger = logging.getLogger("ast.events")
    record = oo_logger.makeRecord(
        name="ast.events",
        level=logging.INFO,
        fn="",
        lno=0,
        msg=f"{event_type}: {fields.get('group', camera or 'system')}",
        args=(),
        exc_info=None,
    )
    oo_fields: dict = {"event_type": event_type}
    if camera:
        oo_fields["camera"] = camera
    oo_fields.update(fields)
    record._oo_fields = oo_fields  # type: ignore[attr-defined]
    oo_logger.handle(record)


def setup_openobserve_logging(
    host: str,
    port: int = 5080,
    org: str = "default",
    stream: str = "ast_audio",
    username: str | None = None,
    password: str | None = None,
) -> OpenObserveHandler:
    """Attach the OpenObserve handler to the root and event loggers."""
    handler = OpenObserveHandler(
        host=host,
        port=port,
        org=org,
        stream=stream,
        username=username,
        password=password,
    )

    # Attach to root logger (all logs go to OpenObserve)
    root = logging.getLogger()
    root.addHandler(handler)

    # Also attach to event logger specifically
    event_logger = logging.getLogger("ast.events")
    event_logger.addHandler(handler)
    event_logger.setLevel(logging.INFO)

    logger.info(
        "OpenObserve logging enabled → %s:%s%s",
        host,
        port,
        f"/api/{org}/{stream}/_json",
    )
    return handler
