"""RTSP URL parsing and Scrypted URL auto-discovery.

When Scrypted NVR restarts, it assigns new random high ports to RTSP
rebroadcast streams. The stream paths remain stable but ports change,
leaving cameras offline until manual config update.

ScryptedApiResolver queries the scrypted-camera-api plugin directly
for authoritative, live RTSP URLs.

ScryptedUrlResolver queries the go2rtc API for fresh RTSP URLs, or
falls back to the go2rtc stable rebroadcast proxy.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import socket
import ssl
from dataclasses import dataclass
from http.client import HTTPConnection
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

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


class ScryptedApiResolver:
    """Resolve RTSP URLs by querying the scrypted-camera-api plugin directly.

    This is the preferred resolver — it returns authoritative, live RTSP URLs
    straight from Scrypted without depending on go2rtc having correct upstream
    configuration.

    Usage:
        resolver = ScryptedApiResolver("https://192.168.0.107:10443/endpoint/scrypted-camera-api/public")
        url = await resolver.resolve("99")  # Scrypted device ID
    """

    def __init__(self, base_url: str, timeout: int = 10) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._ssl_ctx = ssl.create_default_context()
        self._ssl_ctx.check_hostname = False
        self._ssl_ctx.verify_mode = ssl.CERT_NONE

    async def resolve(self, device_id: str) -> str | None:
        """Query scrypted-camera-api for the current RTSP URL of a device.

        Args:
            device_id: Scrypted numeric device ID (e.g. "99").

        Returns:
            RTSP URL string, or None if resolution failed.
        """
        url = f"{self._base_url}/stream/{device_id}"
        try:
            data = await asyncio.to_thread(self._fetch, url)
        except Exception:
            logger.debug("Scrypted API query failed for device %s", device_id, exc_info=True)
            return None

        if data is None:
            return None

        streams = data.get("streams", [])
        for stream in streams:
            urls = stream.get("urls", [])
            if urls:
                safe = re.sub(r"://([^:]+):([^@]+)@", r"://\1:***@", urls[0])
                logger.info(
                    "Scrypted API: resolved device %s -> %s",
                    device_id,
                    safe,
                )
                return urls[0]
            single_url = stream.get("url")
            if single_url:
                safe = re.sub(r"://([^:]+):([^@]+)@", r"://\1:***@", single_url)
                logger.info(
                    "Scrypted API: resolved device %s -> %s",
                    device_id,
                    safe,
                )
                return single_url

        logger.debug("Scrypted API: device %s has no stream URLs", device_id)
        return None

    def _fetch(self, url: str) -> dict | None:
        """Fetch JSON from the Scrypted API (runs in thread)."""
        req = Request(url)
        try:
            resp = urlopen(req, timeout=self._timeout, context=self._ssl_ctx)
        except HTTPError as e:
            if e.code == 404:
                logger.debug("Scrypted API: device not found (404) for %s", url)
            else:
                logger.warning("Scrypted API returned HTTP %d for %s", e.code, url)
            return None
        except (URLError, OSError, TimeoutError):
            logger.debug("Scrypted API unreachable: %s", url, exc_info=True)
            return None
        body = resp.read()
        return json.loads(body)


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
