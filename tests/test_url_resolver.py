"""Tests for RTSP URL parsing and Scrypted URL auto-discovery."""

from __future__ import annotations

import json
from http.client import HTTPConnection
from unittest.mock import MagicMock, patch

import pytest

from src.url_resolver import RtspUrl, ScryptedUrlResolver

# ---------------------------------------------------------------------------
# RtspUrl parsing and reconstruction
# ---------------------------------------------------------------------------


class TestRtspUrl:
    """RtspUrl parses, reconstructs, and manipulates RTSP URLs."""

    def test_parse_url_with_credentials(self):
        url = "rtsp://admin:secret@192.168.1.100:7447/837b18e440b226c1"
        parsed = RtspUrl.parse(url)
        assert parsed.host == "192.168.1.100"
        assert parsed.port == 7447
        assert parsed.path == "/837b18e440b226c1"
        assert parsed.username == "admin"
        assert parsed.password == "secret"

    def test_parse_url_without_credentials(self):
        url = "rtsp://192.168.1.100:8554/test_stream"
        parsed = RtspUrl.parse(url)
        assert parsed.host == "192.168.1.100"
        assert parsed.port == 8554
        assert parsed.path == "/test_stream"
        assert parsed.username is None
        assert parsed.password is None

    def test_parse_url_non_standard_port(self):
        url = "rtsp://10.0.0.1:12345/cam1"
        parsed = RtspUrl.parse(url)
        assert parsed.port == 12345

    def test_parse_url_default_rtsp_port(self):
        """RTSP default port is 554 when not specified."""
        url = "rtsp://192.168.1.100/stream"
        parsed = RtspUrl.parse(url)
        assert parsed.port == 554

    def test_with_port_reconstructs_correctly(self):
        url = "rtsp://admin:secret@192.168.1.100:7447/837b18e440b226c1"
        parsed = RtspUrl.parse(url)
        new_url = parsed.with_port(9999)
        assert new_url == "rtsp://admin:secret@192.168.1.100:9999/837b18e440b226c1"

    def test_with_port_no_credentials(self):
        url = "rtsp://192.168.1.100:8554/test"
        parsed = RtspUrl.parse(url)
        new_url = parsed.with_port(1234)
        assert new_url == "rtsp://192.168.1.100:1234/test"

    def test_round_trip_with_credentials(self):
        url = "rtsp://admin:secret@192.168.1.100:7447/837b18e440b226c1"
        assert str(RtspUrl.parse(url)) == url

    def test_round_trip_without_credentials(self):
        url = "rtsp://192.168.1.100:8554/test_stream"
        assert str(RtspUrl.parse(url)) == url

    def test_path_extraction(self):
        url = "rtsp://host:8554/837b18e440b226c1"
        parsed = RtspUrl.parse(url)
        assert parsed.path == "/837b18e440b226c1"

    def test_parse_hostname_dns(self):
        url = "rtsp://scrypted.local:7447/stream"
        parsed = RtspUrl.parse(url)
        assert parsed.host == "scrypted.local"


# ---------------------------------------------------------------------------
# ScryptedUrlResolver — go2rtc API resolution
# ---------------------------------------------------------------------------


def _make_go2rtc_api_response(streams: dict) -> bytes:
    """Build a JSON response mimicking the go2rtc /api/streams endpoint."""
    return json.dumps(streams).encode("utf-8")


def _mock_http_response(body: bytes, status: int = 200):
    """Create a mock HTTP response object."""
    resp = MagicMock()
    resp.status = status
    resp.read.return_value = body
    return resp


class TestScryptedUrlResolverGo2rtcApi:
    """ScryptedUrlResolver resolves URLs via the go2rtc API."""

    @pytest.mark.asyncio
    async def test_api_returns_matching_stream(self):
        """go2rtc API has a stream whose producer URL matches our path -> fresh URL."""
        streams = {
            "living_room": {
                "producers": [
                    {"url": "rtsp://admin:pass@192.168.1.100:39123/837b18e440b226c1"}
                ]
            }
        }
        body = _make_go2rtc_api_response(streams)

        conn = MagicMock(spec=HTTPConnection)
        conn.getresponse.return_value = _mock_http_response(body)

        with patch("src.url_resolver.HTTPConnection", return_value=conn):
            resolver = ScryptedUrlResolver()
            result = await resolver.resolve(
                "rtsp://admin:pass@192.168.1.100:7447/837b18e440b226c1"
            )

        # Should return a URL with the new port from go2rtc, preserving credentials
        assert result is not None
        parsed = RtspUrl.parse(result)
        assert parsed.port == 39123
        assert parsed.username == "admin"
        assert parsed.password == "pass"
        assert parsed.path == "/837b18e440b226c1"

    @pytest.mark.asyncio
    async def test_api_no_matching_stream(self):
        """go2rtc API has streams but none match our path -> falls through."""
        streams = {
            "other_cam": {
                "producers": [
                    {"url": "rtsp://192.168.1.100:39123/different_path"}
                ]
            }
        }
        body = _make_go2rtc_api_response(streams)

        conn = MagicMock(spec=HTTPConnection)
        conn.getresponse.return_value = _mock_http_response(body)

        with patch("src.url_resolver.HTTPConnection", return_value=conn):
            resolver = ScryptedUrlResolver()
            result = await resolver.resolve(
                "rtsp://192.168.1.100:7447/837b18e440b226c1"
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_api_unreachable_no_exception(self):
        """go2rtc API is unreachable -> returns None without raising."""
        with patch(
            "src.url_resolver.HTTPConnection",
            side_effect=OSError("Connection refused"),
        ):
            resolver = ScryptedUrlResolver()
            result = await resolver.resolve(
                "rtsp://192.168.1.100:7447/837b18e440b226c1"
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_api_timeout_no_exception(self):
        """go2rtc API times out -> returns None without raising."""
        with patch(
            "src.url_resolver.HTTPConnection",
            side_effect=TimeoutError("timed out"),
        ):
            resolver = ScryptedUrlResolver()
            result = await resolver.resolve(
                "rtsp://192.168.1.100:7447/837b18e440b226c1"
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_credentials_preserved_from_original(self):
        """When go2rtc returns a URL without credentials, use originals."""
        streams = {
            "cam": {
                "producers": [
                    {"url": "rtsp://192.168.1.100:39123/837b18e440b226c1"}
                ]
            }
        }
        body = _make_go2rtc_api_response(streams)

        conn = MagicMock(spec=HTTPConnection)
        conn.getresponse.return_value = _mock_http_response(body)

        with patch("src.url_resolver.HTTPConnection", return_value=conn):
            resolver = ScryptedUrlResolver()
            result = await resolver.resolve(
                "rtsp://myuser:mypass@192.168.1.100:7447/837b18e440b226c1"
            )

        assert result is not None
        parsed = RtspUrl.parse(result)
        assert parsed.username == "myuser"
        assert parsed.password == "mypass"
        assert parsed.port == 39123

    @pytest.mark.asyncio
    async def test_api_http_error_status(self):
        """go2rtc API returns 500 -> falls through."""
        conn = MagicMock(spec=HTTPConnection)
        conn.getresponse.return_value = _mock_http_response(b"error", status=500)

        with patch("src.url_resolver.HTTPConnection", return_value=conn):
            resolver = ScryptedUrlResolver()
            result = await resolver.resolve(
                "rtsp://192.168.1.100:7447/837b18e440b226c1"
            )

        assert result is None


# ---------------------------------------------------------------------------
# ScryptedUrlResolver — go2rtc fallback
# ---------------------------------------------------------------------------


class TestScryptedUrlResolverFallback:
    """ScryptedUrlResolver falls back to go2rtc stable proxy URL."""

    @pytest.mark.asyncio
    async def test_fallback_with_go2rtc_stream_and_probe_success(self):
        """go2rtc_stream set + RTSP probe succeeds -> returns stable URL."""
        # API fails
        api_conn = MagicMock(spec=HTTPConnection)
        api_conn.getresponse.return_value = _mock_http_response(b"{}", status=200)

        # Probe succeeds
        probe_sock = MagicMock()
        probe_sock.recv.return_value = b"RTSP/1.0 200 OK"

        with (
            patch("src.url_resolver.HTTPConnection", return_value=api_conn),
            patch("src.url_resolver.socket.create_connection", return_value=probe_sock),
        ):
            resolver = ScryptedUrlResolver()
            result = await resolver.resolve(
                "rtsp://192.168.1.100:7447/837b18e440b226c1",
                go2rtc_stream="living_room",
            )

        assert result is not None
        assert result == "rtsp://a889bffc-go2rtc:8554/living_room"

    @pytest.mark.asyncio
    async def test_fallback_skipped_when_go2rtc_stream_none(self):
        """go2rtc_stream is None -> fallback skipped, returns None."""
        api_conn = MagicMock(spec=HTTPConnection)
        api_conn.getresponse.return_value = _mock_http_response(b"{}", status=200)

        with patch("src.url_resolver.HTTPConnection", return_value=api_conn):
            resolver = ScryptedUrlResolver()
            result = await resolver.resolve(
                "rtsp://192.168.1.100:7447/837b18e440b226c1",
                go2rtc_stream=None,
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_fallback_probe_fails(self):
        """go2rtc_stream set but probe fails -> returns None."""
        api_conn = MagicMock(spec=HTTPConnection)
        api_conn.getresponse.return_value = _mock_http_response(b"{}", status=200)

        with (
            patch("src.url_resolver.HTTPConnection", return_value=api_conn),
            patch(
                "src.url_resolver.socket.create_connection",
                side_effect=OSError("Connection refused"),
            ),
        ):
            resolver = ScryptedUrlResolver()
            result = await resolver.resolve(
                "rtsp://192.168.1.100:7447/837b18e440b226c1",
                go2rtc_stream="living_room",
            )

        assert result is None


# ---------------------------------------------------------------------------
# ScryptedUrlResolver — full resolution chain
# ---------------------------------------------------------------------------


class TestScryptedUrlResolverChain:
    """Full resolution chain: API -> fallback -> None."""

    @pytest.mark.asyncio
    async def test_api_fails_fallback_works(self):
        """API has no match, fallback with go2rtc_stream succeeds."""
        # API returns empty streams
        api_conn = MagicMock(spec=HTTPConnection)
        api_conn.getresponse.return_value = _mock_http_response(b"{}", status=200)

        probe_sock = MagicMock()
        probe_sock.recv.return_value = b"RTSP/1.0 200 OK"

        with (
            patch("src.url_resolver.HTTPConnection", return_value=api_conn),
            patch("src.url_resolver.socket.create_connection", return_value=probe_sock),
        ):
            resolver = ScryptedUrlResolver()
            result = await resolver.resolve(
                "rtsp://192.168.1.100:7447/test",
                go2rtc_stream="test_cam",
            )

        assert result == "rtsp://a889bffc-go2rtc:8554/test_cam"

    @pytest.mark.asyncio
    async def test_everything_fails_returns_none(self):
        """Both API and fallback fail -> returns None."""
        with (
            patch(
                "src.url_resolver.HTTPConnection",
                side_effect=OSError("Connection refused"),
            ),
            patch(
                "src.url_resolver.socket.create_connection",
                side_effect=OSError("Connection refused"),
            ),
        ):
            resolver = ScryptedUrlResolver()
            result = await resolver.resolve(
                "rtsp://192.168.1.100:7447/test",
                go2rtc_stream="test_cam",
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_api_match_preferred_over_fallback(self):
        """When API matches, fallback is not used (even if go2rtc_stream set)."""
        streams = {
            "cam": {
                "producers": [
                    {"url": "rtsp://192.168.1.100:55555/837b18e440b226c1"}
                ]
            }
        }
        body = _make_go2rtc_api_response(streams)

        conn = MagicMock(spec=HTTPConnection)
        conn.getresponse.return_value = _mock_http_response(body)

        with patch("src.url_resolver.HTTPConnection", return_value=conn):
            resolver = ScryptedUrlResolver()
            result = await resolver.resolve(
                "rtsp://admin:pass@192.168.1.100:7447/837b18e440b226c1",
                go2rtc_stream="living_room",
            )

        # Should return API result, not fallback
        assert result is not None
        parsed = RtspUrl.parse(result)
        assert parsed.port == 55555
        assert "go2rtc" not in result

    @pytest.mark.asyncio
    async def test_same_port_auto_discovers_stream_for_fallback(self):
        """API path matches but same port -> auto-discover stream name for fallback."""
        streams = {
            "backyard_local": {
                "producers": [
                    {"url": "rtsp://192.168.1.100:41621/283bbf527b3ed2a5"}
                ]
            }
        }
        body = _make_go2rtc_api_response(streams)

        api_conn = MagicMock(spec=HTTPConnection)
        api_conn.getresponse.return_value = _mock_http_response(body)

        probe_sock = MagicMock()
        probe_sock.recv.return_value = b"RTSP/1.0 200 OK"

        with (
            patch("src.url_resolver.HTTPConnection", return_value=api_conn),
            patch("src.url_resolver.socket.create_connection", return_value=probe_sock),
        ):
            resolver = ScryptedUrlResolver()
            result = await resolver.resolve(
                "rtsp://192.168.1.100:41621/283bbf527b3ed2a5",
                go2rtc_stream=None,  # no explicit stream
            )

        # Should use auto-discovered stream name for fallback
        assert result == "rtsp://a889bffc-go2rtc:8554/backyard_local"

    @pytest.mark.asyncio
    async def test_custom_go2rtc_host_and_ports(self):
        """Resolver uses custom go2rtc host/ports when configured."""
        streams = {
            "cam": {
                "producers": [
                    {"url": "rtsp://192.168.1.100:55555/stream"}
                ]
            }
        }
        body = _make_go2rtc_api_response(streams)

        conn = MagicMock(spec=HTTPConnection)
        conn.getresponse.return_value = _mock_http_response(body)

        with patch("src.url_resolver.HTTPConnection", return_value=conn) as mock_cls:
            resolver = ScryptedUrlResolver(
                go2rtc_host="custom-host",
                go2rtc_api_port=9999,
                go2rtc_rtsp_port=7777,
            )
            await resolver.resolve("rtsp://192.168.1.100:7447/stream")

        # Verify it connected to custom host:port
        mock_cls.assert_called_with("custom-host", 9999, timeout=5)
