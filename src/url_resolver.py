"""RTSP URL parsing and Scrypted URL auto-discovery.

When Scrypted NVR restarts, it assigns new random high ports to RTSP
rebroadcast streams. The stream paths remain stable but ports change,
leaving cameras offline until manual config update.

ScryptedUrlResolver queries the go2rtc API for fresh RTSP URLs, or
falls back to the go2rtc stable rebroadcast proxy.
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
from dataclasses import dataclass
from http.client import HTTPConnection
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class RtspUrl:
    """Parse and reconstruct RTSP URLs."""

    host: str
    port: int
    path: str
    username: str | None = None
    password: str | None = None

    @classmethod
    def parse(cls, url: str) -> RtspUrl:
        """Parse an RTSP URL into components."""
        parsed = urlparse(url)
        return cls(
            host=parsed.hostname or "",
            port=parsed.port or 554,
            path=parsed.path,
            username=parsed.username,
            password=parsed.password,
        )

    def with_port(self, port: int) -> str:
        """Return a new URL string with a different port."""
        return str(RtspUrl(
            host=self.host,
            port=port,
            path=self.path,
            username=self.username,
            password=self.password,
        ))

    def __str__(self) -> str:
        if self.username and self.password:
            return f"rtsp://{self.username}:{self.password}@{self.host}:{self.port}{self.path}"
        return f"rtsp://{self.host}:{self.port}{self.path}"


class ScryptedUrlResolver:
    """Resolve fresh RTSP URLs via the go2rtc API or stable proxy fallback.

    Resolution chain:
    1. Query go2rtc API for streams with a matching RTSP path
    2. Fall back to go2rtc stable proxy URL (if go2rtc_stream is configured)
    3. Return None if both methods fail
    """

    def __init__(
        self,
        go2rtc_host: str = "a889bffc-go2rtc",
        go2rtc_api_port: int = 1984,
        go2rtc_rtsp_port: int = 8554,
    ) -> None:
        self._go2rtc_host = go2rtc_host
        self._go2rtc_api_port = go2rtc_api_port
        self._go2rtc_rtsp_port = go2rtc_rtsp_port

    async def resolve(
        self, original_url: str, go2rtc_stream: str | None = None
    ) -> str | None:
        """Attempt to resolve a fresh RTSP URL for the given camera.

        Args:
            original_url: The currently-configured RTSP URL (possibly stale).
            go2rtc_stream: Optional go2rtc stream name for stable proxy fallback.

        Returns:
            A fresh RTSP URL string, or None if resolution failed.
        """
        original = RtspUrl.parse(original_url)

        # Step 1: Try go2rtc API for a fresh port
        result, discovered_stream = await self._try_go2rtc_api(original)
        if result is not None:
            return result

        # Step 2: Try go2rtc stable proxy fallback
        # Use explicit go2rtc_stream, or auto-discovered stream name from API
        fallback_stream = go2rtc_stream or discovered_stream
        if fallback_stream is not None:
            result = await self._try_go2rtc_fallback(fallback_stream)
            if result is not None:
                return result

        return None

    async def _try_go2rtc_api(self, original: RtspUrl) -> tuple[str | None, str | None]:
        """Query go2rtc /api/streams for a producer URL matching our path.

        Returns:
            Tuple of (fresh_url, discovered_stream_name).
            fresh_url is set only if a different port was found.
            discovered_stream_name is the go2rtc stream name that matched the path.
        """
        try:
            streams = await asyncio.to_thread(self._fetch_go2rtc_streams)
        except Exception:
            logger.debug("go2rtc API query failed", exc_info=True)
            return None, None

        if not streams:
            return None, None

        # Search all stream producers for a matching RTSP path
        for stream_name, stream_info in streams.items():
            producers = stream_info.get("producers", [])
            for producer in producers:
                producer_url = producer.get("url", "")
                if not producer_url.startswith("rtsp://"):
                    continue
                candidate = RtspUrl.parse(producer_url)
                if candidate.path == original.path:
                    if candidate.port != original.port:
                        # Found a fresh URL with a new port
                        fresh = RtspUrl(
                            host=candidate.host,
                            port=candidate.port,
                            path=candidate.path,
                            username=original.username,
                            password=original.password,
                        )
                        logger.info(
                            "go2rtc API: resolved port %d -> %d for path %s",
                            original.port,
                            candidate.port,
                            original.path,
                        )
                        return str(fresh), stream_name
                    # Same port — can't get a fresh URL, but we know the stream name
                    logger.debug(
                        "go2rtc API: path %s matches stream %r (same port %d)",
                        original.path,
                        stream_name,
                        original.port,
                    )
                    return None, stream_name

        return None, None

    def _fetch_go2rtc_streams(self) -> dict:
        """Fetch streams from go2rtc API (runs in thread)."""
        conn = HTTPConnection(self._go2rtc_host, self._go2rtc_api_port, timeout=5)
        conn.request("GET", "/api/streams")
        resp = conn.getresponse()
        body = resp.read()
        conn.close()
        if resp.status >= 400:
            logger.warning("go2rtc API returned %d", resp.status)
            return {}
        return json.loads(body)

    async def _try_go2rtc_fallback(self, go2rtc_stream: str) -> str | None:
        """Construct and probe the go2rtc stable RTSP proxy URL."""
        fallback_url = (
            f"rtsp://{self._go2rtc_host}:{self._go2rtc_rtsp_port}/{go2rtc_stream}"
        )

        # Probe with RTSP OPTIONS to verify the stream is reachable
        ok = await asyncio.to_thread(
            self._probe_rtsp, self._go2rtc_host, self._go2rtc_rtsp_port
        )
        if ok:
            logger.info("go2rtc fallback: using stable proxy %s", fallback_url)
            return fallback_url

        logger.debug("go2rtc fallback probe failed for %s", fallback_url)
        return None

    def _probe_rtsp(self, host: str, port: int) -> bool:
        """Send an RTSP OPTIONS request to verify connectivity."""
        try:
            sock = socket.create_connection((host, port), timeout=5)
            sock.sendall(b"OPTIONS rtsp://%b:%d/ RTSP/1.0\r\nCSeq: 1\r\n\r\n"
                         % (host.encode(), port))
            data = sock.recv(1024)
            sock.close()
            return data.startswith(b"RTSP/")
        except Exception:
            return False
