"""Tests for Scrypted URL auto-discovery via Camera API."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.url_resolver import ScryptedApiResolver

# ---------------------------------------------------------------------------
# ScryptedApiResolver — direct Scrypted camera API resolution
# ---------------------------------------------------------------------------


def _make_scrypted_stream_response(device_id: str, name: str, urls: list[str]) -> bytes:
    """Build a JSON response mimicking scrypted-camera-api /stream/:deviceId."""
    streams = []
    if urls:
        streams.append(
            {
                "id": "default",
                "name": "Default",
                "url": urls[0],
                "urls": urls,
            }
        )
    return json.dumps(
        {
            "id": device_id,
            "name": name,
            "streams": streams,
        }
    ).encode("utf-8")


class TestScryptedApiResolver:
    """ScryptedApiResolver resolves URLs via the scrypted-camera-api plugin."""

    @pytest.mark.asyncio
    async def test_returns_rtsp_url_for_device(self):
        """API returns stream info -> extracts first RTSP URL."""
        body = _make_scrypted_stream_response(
            "99",
            "Front Door Camera",
            ["rtsp://admin:pass@192.168.0.107:40004/abc123"],
        )
        resp = MagicMock()
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        resp.status = 200
        resp.read.return_value = body

        with patch("src.url_resolver.urlopen", return_value=resp):
            resolver = ScryptedApiResolver(
                "https://192.168.0.107:10443/endpoint/scrypted-camera-api/public"
            )
            result = await resolver.resolve("99")

        assert result == "rtsp://admin:pass@192.168.0.107:40004/abc123"

    @pytest.mark.asyncio
    async def test_returns_first_url_from_urls_array(self):
        """When multiple URLs exist, returns urls[0] (external address)."""
        body = _make_scrypted_stream_response(
            "865",
            "Living Room",
            ["rtsp://admin:pass@192.168.0.107:40001/xyz", "rtsp://127.0.0.1:40001/xyz"],
        )
        resp = MagicMock()
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        resp.status = 200
        resp.read.return_value = body

        with patch("src.url_resolver.urlopen", return_value=resp):
            resolver = ScryptedApiResolver(
                "https://scrypted.local:10443/endpoint/scrypted-camera-api/public"
            )
            result = await resolver.resolve("865")

        assert result == "rtsp://admin:pass@192.168.0.107:40001/xyz"

    @pytest.mark.asyncio
    async def test_device_not_found_returns_none(self):
        """API returns 404 for unknown device -> returns None."""
        from urllib.error import HTTPError

        err = HTTPError(
            url="https://host/stream/999",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=None,
        )

        with patch("src.url_resolver.urlopen", side_effect=err):
            resolver = ScryptedApiResolver("https://host")
            result = await resolver.resolve("999")

        assert result is None

    @pytest.mark.asyncio
    async def test_api_unreachable_returns_none(self):
        """API is unreachable -> returns None without raising."""
        with patch("src.url_resolver.urlopen", side_effect=OSError("Connection refused")):
            resolver = ScryptedApiResolver("https://host")
            result = await resolver.resolve("99")

        assert result is None

    @pytest.mark.asyncio
    async def test_api_timeout_returns_none(self):
        """API times out -> returns None without raising."""
        with patch("src.url_resolver.urlopen", side_effect=TimeoutError("timed out")):
            resolver = ScryptedApiResolver("https://host")
            result = await resolver.resolve("99")

        assert result is None

    @pytest.mark.asyncio
    async def test_empty_streams_returns_none(self):
        """API returns device with no streams -> returns None."""
        body = json.dumps(
            {
                "id": "69",
                "name": "ESP32 Camera",
                "streams": [],
            }
        ).encode()
        resp = MagicMock()
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        resp.status = 200
        resp.read.return_value = body

        with patch("src.url_resolver.urlopen", return_value=resp):
            resolver = ScryptedApiResolver("https://host")
            result = await resolver.resolve("69")

        assert result is None

    @pytest.mark.asyncio
    async def test_stream_with_error_no_url_returns_none(self):
        """API returns stream with error field and no URL -> returns None."""
        body = json.dumps(
            {
                "id": "117",
                "name": "Side Door",
                "streams": [{"id": "default", "name": "Default", "error": "Camera offline"}],
            }
        ).encode()
        resp = MagicMock()
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        resp.status = 200
        resp.read.return_value = body

        with patch("src.url_resolver.urlopen", return_value=resp):
            resolver = ScryptedApiResolver("https://host")
            result = await resolver.resolve("117")

        assert result is None

    @pytest.mark.asyncio
    async def test_constructs_correct_url(self):
        """Resolver constructs the correct API URL for the device."""
        body = _make_scrypted_stream_response("99", "Front Door", ["rtsp://x:1234/y"])
        resp = MagicMock()
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        resp.status = 200
        resp.read.return_value = body

        with patch("src.url_resolver.urlopen", return_value=resp) as mock_urlopen:
            resolver = ScryptedApiResolver(
                "https://192.168.0.107:10443/endpoint/scrypted-camera-api/public"
            )
            await resolver.resolve("99")

        call_args = mock_urlopen.call_args
        request_obj = call_args[0][0]
        assert request_obj.full_url == (
            "https://192.168.0.107:10443/endpoint/scrypted-camera-api/public/stream/99"
        )

    @pytest.mark.asyncio
    async def test_strips_trailing_slash_from_base_url(self):
        """Trailing slash in base URL doesn't cause double-slash."""
        body = _make_scrypted_stream_response("99", "Front Door", ["rtsp://x:1234/y"])
        resp = MagicMock()
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        resp.status = 200
        resp.read.return_value = body

        with patch("src.url_resolver.urlopen", return_value=resp) as mock_urlopen:
            resolver = ScryptedApiResolver(
                "https://host:10443/endpoint/scrypted-camera-api/public/"
            )
            await resolver.resolve("99")

        request_obj = mock_urlopen.call_args[0][0]
        assert "//" not in request_obj.full_url.split("://")[1]

    @pytest.mark.asyncio
    async def test_ssl_context_disables_verification(self):
        """Resolver uses an SSL context with verification disabled."""
        body = _make_scrypted_stream_response("99", "Cam", ["rtsp://x:1/y"])
        resp = MagicMock()
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        resp.status = 200
        resp.read.return_value = body

        with patch("src.url_resolver.urlopen", return_value=resp) as mock_urlopen:
            resolver = ScryptedApiResolver("https://host")
            await resolver.resolve("99")

        call_args = mock_urlopen.call_args
        # urlopen called with positional or keyword context arg
        ctx = call_args.kwargs.get("context")
        if ctx is None and len(call_args.args) > 1:
            ctx = call_args.args[1]
        assert ctx is not None
        assert ctx.check_hostname is False
