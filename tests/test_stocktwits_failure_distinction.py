from __future__ import annotations

import json
from urllib.error import HTTPError

import pytest

from tradingagents.dataflows import stocktwits


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
    return HTTPError("https://stocktwits.example", status, "failed", None, None)


@pytest.mark.parametrize("status", [403, 429, 500])
def test_stocktwits_http_unavailable_returns_missing_signal_placeholder(
    monkeypatch, status
):
    def fail(request, timeout):
        raise _http_error(status)

    monkeypatch.setattr(stocktwits, "urlopen", fail)

    result = stocktwits.fetch_stocktwits_messages("AAPL")

    assert "unavailable" in result
    assert str(status) in result
    assert "missing, not neutral" in result


def test_stocktwits_empty_200_response_keeps_no_messages_placeholder(monkeypatch):
    monkeypatch.setattr(
        stocktwits,
        "urlopen",
        lambda request, timeout: _Response({"messages": []}),
    )

    assert stocktwits.fetch_stocktwits_messages("AAPL") == (
        "<no StockTwits messages found for $AAPL>"
    )
