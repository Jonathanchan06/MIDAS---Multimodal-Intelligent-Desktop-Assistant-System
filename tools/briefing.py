"""Morning briefing tool. Deterministic data fetching only — no LLM calls
live here. The orchestrator turns the raw dict this returns into prose.
"""

from __future__ import annotations

import feedparser
import yfinance as yf

import config

BRIEFING_SCHEMA = {
    "type": "function",
    "function": {
        "name": "run_morning_briefing",
        "description": (
            "Fetch current watchlist stock prices and top news headlines. "
            "Call this when the user asks for a briefing, market update, "
            "or news summary."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}


def get_market_summary(tickers: list[str] | None = None) -> dict:
    tickers = tickers or config.WATCHLIST_TICKERS
    summary = {}
    for ticker in tickers:
        try:
            history = yf.Ticker(ticker).history(period="2d")
            if history.empty or len(history) < 2:
                summary[ticker] = {"error": "no data available"}
                continue
            previous_close = float(history["Close"].iloc[-2])
            last_price = float(history["Close"].iloc[-1])
            change_pct = ((last_price - previous_close) / previous_close) * 100
            summary[ticker] = {
                "price": round(last_price, 2),
                "change_pct": round(change_pct, 2),
            }
        except Exception as exc:
            summary[ticker] = {"error": str(exc)}
    return summary


def get_top_headlines(
    feeds: list[str] | None = None, limit: int = config.HEADLINE_LIMIT
) -> list[dict]:
    feeds = feeds or config.NEWS_FEEDS
    headlines = []
    for feed_url in feeds:
        try:
            parsed = feedparser.parse(feed_url)
            source = parsed.feed.get("title", feed_url)
            for entry in parsed.entries[:limit]:
                headlines.append(
                    {
                        "title": entry.get("title", "").strip(),
                        "source": source,
                        "link": entry.get("link", ""),
                    }
                )
        except Exception as exc:
            headlines.append({"title": f"[failed to fetch {feed_url}: {exc}]"})
    return headlines[:limit]


def run_morning_briefing() -> dict:
    return {
        "markets": get_market_summary(),
        "headlines": get_top_headlines(),
    }
