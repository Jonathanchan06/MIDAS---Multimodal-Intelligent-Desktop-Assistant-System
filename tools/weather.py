"""Current-conditions weather via Open-Meteo. No API key, no account.

Deterministic fetch only — no LLM calls, mirroring tools/briefing.py. Not
registered as an arch-router tool: nothing here is a conversational action
on its own, it's an input to tools/wardrobe.py's suggest_fit — same
relationship as tools/notifications.py has to tools/briefing.py.
"""

from __future__ import annotations

import requests

import config

_WEATHER_CODES = {
    0: "Clear",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Fog",
    51: "Light drizzle",
    53: "Drizzle",
    55: "Heavy drizzle",
    61: "Light rain",
    63: "Rain",
    65: "Heavy rain",
    71: "Light snow",
    73: "Snow",
    75: "Heavy snow",
    80: "Rain showers",
    81: "Rain showers",
    82: "Heavy rain showers",
    95: "Thunderstorm",
    96: "Thunderstorm",
    99: "Thunderstorm",
}


def get_current_weather(lat: float | None = None, lon: float | None = None) -> dict:
    """Fetch current temperature/conditions plus today's high/low and
    rain chance. Never raises — returns {"error": ...} on any failure so
    a dead network degrades a caller's result instead of killing it,
    same convention as tools/briefing.py's get_market_summary."""
    lat = lat if lat is not None else config.WEATHER_LAT
    lon = lon if lon is not None else config.WEATHER_LON

    try:
        response = requests.get(
            config.WEATHER_API_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,apparent_temperature,precipitation,"
                "weather_code,wind_speed_10m,is_day",
                "daily": "temperature_2m_max,temperature_2m_min,"
                "precipitation_probability_max",
                "temperature_unit": config.WEATHER_TEMPERATURE_UNIT,
                "wind_speed_unit": config.WEATHER_WIND_SPEED_UNIT,
                "timezone": "auto",
                "forecast_days": 1,
            },
            timeout=config.WEATHER_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        current = payload["current"]
        daily = payload["daily"]

        return {
            "temperature": current["temperature_2m"],
            "feels_like": current["apparent_temperature"],
            "high": daily["temperature_2m_max"][0],
            "low": daily["temperature_2m_min"][0],
            "precipitation_probability": daily["precipitation_probability_max"][0],
            "wind_speed": current["wind_speed_10m"],
            "description": _WEATHER_CODES.get(current["weather_code"], "Unknown"),
            "unit": config.WEATHER_TEMPERATURE_UNIT,
        }
    except Exception as exc:
        return {"error": str(exc)}
