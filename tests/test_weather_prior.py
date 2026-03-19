"""Tests for weather prior — HA entity polling for outdoor threshold modulation."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.weather_prior import WeatherCondition, WeatherPrior


class TestWeatherCondition:
    def test_rainy_conditions(self):
        for state in ("rainy", "pouring", "lightning-rainy", "hail", "lightning"):
            assert WeatherCondition.from_ha_state(state) == WeatherCondition.RAINY

    def test_clear_conditions(self):
        for state in ("sunny", "clear-night", "partlycloudy", "windy"):
            assert WeatherCondition.from_ha_state(state) == WeatherCondition.CLEAR

    def test_cloudy_conditions(self):
        for state in ("cloudy", "fog", "snowy"):
            assert WeatherCondition.from_ha_state(state) == WeatherCondition.CLOUDY

    def test_unknown_returns_unknown(self):
        assert WeatherCondition.from_ha_state("unavailable") == WeatherCondition.UNKNOWN
        assert WeatherCondition.from_ha_state("exceptional") == WeatherCondition.UNKNOWN


class TestWeatherPrior:
    def test_default_condition_is_unknown(self):
        wp = WeatherPrior(entity_id="weather.home")
        assert wp.condition == WeatherCondition.UNKNOWN

    def test_get_threshold_modifier_rainy(self):
        wp = WeatherPrior(entity_id="weather.home")
        wp._condition = WeatherCondition.RAINY
        mod = wp.get_threshold_modifier("rain_storm")
        assert mod < 0

    def test_get_threshold_modifier_clear(self):
        wp = WeatherPrior(entity_id="weather.home")
        wp._condition = WeatherCondition.CLEAR
        mod = wp.get_threshold_modifier("rain_storm")
        assert mod > 0

    def test_get_threshold_modifier_unaffected_group(self):
        wp = WeatherPrior(entity_id="weather.home")
        wp._condition = WeatherCondition.RAINY
        mod = wp.get_threshold_modifier("dog_bark")
        assert mod == 0.0

    def test_get_threshold_modifier_unknown_weather(self):
        wp = WeatherPrior(entity_id="weather.home")
        wp._condition = WeatherCondition.UNKNOWN
        mod = wp.get_threshold_modifier("rain_storm")
        assert mod == 0.0

    def test_poll_updates_condition(self):
        wp = WeatherPrior(entity_id="weather.home", supervisor_token="test-token")
        resp = MagicMock()
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        resp.read.return_value = json.dumps({"state": "rainy"}).encode()
        with patch("src.weather_prior.urllib.request.urlopen", return_value=resp):
            wp._poll_sync()
        assert wp.condition == WeatherCondition.RAINY

    def test_poll_error_sets_unknown(self):
        wp = WeatherPrior(entity_id="weather.home", supervisor_token="test-token")
        wp._condition = WeatherCondition.RAINY
        with patch("src.weather_prior.urllib.request.urlopen", side_effect=Exception("timeout")):
            wp._poll_sync()
        assert wp.condition == WeatherCondition.UNKNOWN
