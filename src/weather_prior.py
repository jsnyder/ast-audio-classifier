"""Weather prior — polls HA weather entity to modulate outdoor group thresholds.

When it's actually raining, rain_storm detections are more trustworthy (lower threshold).
When it's clear/sunny, rain_storm detections are likely false positives (raise threshold).
"""

from __future__ import annotations

import asyncio
import enum
import json
import logging
import os
import urllib.request

logger = logging.getLogger(__name__)

_WEATHER_MODIFIERS: dict[str, dict[str, float]] = {
    "rain_storm": {
        "rainy": -0.20,
        "clear": 0.15,
        "cloudy": 0.0,
    },
}


class WeatherCondition(enum.Enum):
    RAINY = "rainy"
    CLEAR = "clear"
    CLOUDY = "cloudy"
    UNKNOWN = "unknown"

    @classmethod
    def from_ha_state(cls, state: str) -> WeatherCondition:
        rainy = {"rainy", "pouring", "lightning-rainy", "hail", "lightning"}
        clear = {"sunny", "clear-night", "partlycloudy", "windy"}
        cloudy = {"cloudy", "fog", "snowy"}
        if state in rainy:
            return cls.RAINY
        if state in clear:
            return cls.CLEAR
        if state in cloudy:
            return cls.CLOUDY
        return cls.UNKNOWN


class WeatherPrior:
    """Polls an HA weather entity and provides threshold modifiers."""

    def __init__(
        self,
        entity_id: str = "weather.home",
        poll_interval: float = 300.0,
        supervisor_token: str | None = None,
    ) -> None:
        self._entity_id = entity_id
        self._poll_interval = poll_interval
        self._supervisor_token = supervisor_token or os.environ.get("SUPERVISOR_TOKEN", "")
        self._condition = WeatherCondition.UNKNOWN
        self._task: asyncio.Task | None = None

    @property
    def condition(self) -> WeatherCondition:
        return self._condition

    def get_threshold_modifier(self, group: str) -> float:
        """Return threshold adjustment for a group based on current weather.

        Returns 0.0 for unaffected groups or unknown weather.
        Negative = lower threshold (more sensitive).
        Positive = raise threshold (less sensitive).
        """
        if self._condition == WeatherCondition.UNKNOWN:
            return 0.0
        group_mods = _WEATHER_MODIFIERS.get(group)
        if group_mods is None:
            return 0.0
        return group_mods.get(self._condition.value, 0.0)

    def _poll_sync(self) -> None:
        """Synchronous poll — runs in a thread."""
        url = f"http://supervisor/core/api/states/{self._entity_id}"
        headers = {"Authorization": f"Bearer {self._supervisor_token}"}
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            state = data.get("state", "unavailable")
            new_condition = WeatherCondition.from_ha_state(state)
            if new_condition != self._condition:
                logger.info(
                    "Weather prior: %s -> %s (HA state: %s)",
                    self._condition.value, new_condition.value, state,
                )
            self._condition = new_condition
        except Exception:
            logger.debug("Weather prior poll failed", exc_info=True)
            self._condition = WeatherCondition.UNKNOWN

    async def start(self) -> None:
        """Start periodic polling (idempotent)."""
        if self._task and not self._task.done():
            return
        await asyncio.to_thread(self._poll_sync)
        self._task = asyncio.create_task(self._poll_loop(), name="weather-prior")
        logger.info(
            "Weather prior started: entity=%s, interval=%.0fs, condition=%s",
            self._entity_id, self._poll_interval, self._condition.value,
        )

    async def _poll_loop(self) -> None:
        while True:
            await asyncio.sleep(self._poll_interval)
            await asyncio.to_thread(self._poll_sync)

    async def stop(self) -> None:
        task = self._task
        self._task = None
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
