from __future__ import annotations

import json
import threading
import time
from urllib.error import HTTPError

import pytest

from tradingagents.dataflows import reddit


class _Response:
    def __init__(self, payload: dict):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


def _http_error(status: int) -> HTTPError:
    return HTTPError("https://reddit.example", status, "failed", None, None)


@pytest.mark.parametrize("status", [403, 429, 500])
def test_reddit_http_unavailable_returns_missing_signal_placeholder(monkeypatch, status):
    def fail(request, timeout):
        raise _http_error(status)

    monkeypatch.setattr(reddit, "urlopen", fail)

    result = reddit.fetch_reddit_posts(
        "AAPL",
        subreddits=("stocks",),
        limit_per_sub=1,
        inter_request_delay=0,
    )

    assert "unavailable" in result
    assert str(status) in result
    assert "missing, not neutral" in result


def test_reddit_empty_200_response_stays_empty_list(monkeypatch):
    monkeypatch.setattr(
        reddit,
        "urlopen",
        lambda request, timeout: _Response({"data": {"children": []}}),
    )

    assert reddit._fetch_subreddit("AAPL", "stocks", 1, 1.0) == []


def test_reddit_fetches_default_subreddits_concurrently(monkeypatch):
    active = 0
    max_active = 0
    calls = []
    lock = threading.Lock()

    def fake_fetch(ticker, sub, limit, timeout):
        nonlocal active, max_active
        with lock:
            calls.append(sub)
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return [{"title": f"{sub} post", "score": 1, "num_comments": 2}]

    monkeypatch.setattr(reddit, "_fetch_subreddit", fake_fetch)

    result = reddit.fetch_reddit_posts(
        "AAPL",
        subreddits=("wallstreetbets", "stocks", "investing"),
        limit_per_sub=1,
        inter_request_delay=0.4,
    )

    assert set(calls) == {"wallstreetbets", "stocks", "investing"}
    assert max_active > 1
    assert "r/wallstreetbets" in result
    assert "r/stocks" in result
    assert "r/investing" in result
