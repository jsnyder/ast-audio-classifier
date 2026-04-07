"""Scrypted URL auto-discovery via the Camera API.

When Scrypted NVR restarts, it assigns new random high ports to RTSP
rebroadcast streams. The stream paths remain stable but ports change,
leaving cameras offline until manual config update.

ScryptedApiResolver queries the scrypted-camera-api plugin directly
for authoritative, live RTSP URLs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import ssl
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .openobserve import log_event

logger = logging.getLogger(__name__)


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
            with urlopen(req, timeout=self._timeout, context=self._ssl_ctx) as resp:
                body = resp.read()
        except HTTPError as e:
            if e.code == 404:
                logger.debug("Scrypted API: device not found (404) for %s", url)
                log_event("resolver_failure", error_type="scrypted_404", url=url, http_status=404)
            else:
                logger.warning("Scrypted API returned HTTP %d for %s", e.code, url)
                log_event("resolver_failure", error_type=f"scrypted_{e.code}", url=url, http_status=e.code)
            return None
        except (URLError, OSError, TimeoutError) as e:
            logger.debug("Scrypted API unreachable: %s", url, exc_info=True)
            log_event("resolver_failure", error_type="scrypted_unreachable", url=url, detail=str(e)[:200])
            return None
        return json.loads(body)
