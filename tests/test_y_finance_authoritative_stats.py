"""Regression test for the authoritative-stats header prepended by
get_YFin_data_online — addresses the SOXX 2026-05-17 hallucination where
the market analyst LLM cited $122.54 as the period low when the actual
low was $309.79. The fix injects computed stats into the tool output so
the LLM has authoritative numbers it MUST cite rather than deriving its
own (which it does badly).
"""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from tradingagents.dataflows.symbol_utils import NoMarketDataError

# Skip in environments where the real yfinance/stockstats stack is not
# installed; in CI/dev with deps installed these tests will execute and
# protect against the SOXX hallucination regression.
pytest.importorskip("yfinance")
pytest.importorskip("stockstats")


def _import_target():
    from tradingagents.dataflows.y_finance import get_YFin_data_online
    return get_YFin_data_online


def _make_fixture_frame() -> pd.DataFrame:
    """3 trading days with known min/max/return so we can assert exactly."""
    idx = pd.to_datetime(["2026-03-02", "2026-03-03", "2026-03-04"])
    return pd.DataFrame(
        {
            "Open":  [350.00, 330.00, 340.00],
            "High":  [355.00, 335.00, 345.00],
            "Low":   [345.00, 325.00, 335.00],
            "Close": [352.05, 309.79, 341.54],   # min=309.79 on day 2, last=341.54
            "Volume": [7_284_000, 17_946_100, 10_649_900],
        },
        index=idx,
    )


class _StubTicker:
    def __init__(self, _symbol: str):
        pass

    def history(self, **_kw):
        return _make_fixture_frame()


def test_authoritative_stats_block_is_present_and_correct():
    get_YFin_data_online = _import_target()
    with patch("tradingagents.dataflows.y_finance.yf.Ticker", _StubTicker), \
         patch("tradingagents.dataflows.y_finance.yf_retry",
               lambda fn: fn()):
        out = get_YFin_data_online("SOXX", "2026-03-02", "2026-03-04")

    assert "AUTHORITATIVE STATS" in out, "stats block must be present"
    # Exact values from the fixture so any future arithmetic drift is caught.
    assert "period_first_close: 352.05 (2026-03-02)" in out
    assert "period_last_close: 341.54 (2026-03-04)" in out
    assert "period_min_close: 309.79 (2026-03-03)" in out
    assert "period_max_close: 352.05 (2026-03-02)" in out
    # period_return = (341.54 - 352.05) / 352.05 * 100 ≈ -2.99%
    assert "period_return_pct: -2.99" in out
    assert "trading_days: 3" in out
    assert "total_volume: 35,880,000" in out


def test_csv_payload_unchanged_after_header():
    """The actual CSV that follows the stats block must still be valid CSV
    so downstream tools (get_indicators, stockstats) keep working.
    """
    get_YFin_data_online = _import_target()
    with patch("tradingagents.dataflows.y_finance.yf.Ticker", _StubTicker), \
         patch("tradingagents.dataflows.y_finance.yf_retry",
               lambda fn: fn()):
        out = get_YFin_data_online("SOXX", "2026-03-02", "2026-03-04")

    # CSV begins at the first non-comment line.
    csv_part = "\n".join(
        line for line in out.splitlines() if not line.startswith("#")
    ).strip()
    header_line, *data_lines = csv_part.splitlines()
    assert "Date" in header_line and "Close" in header_line
    assert len(data_lines) == 3
    assert "2026-03-03" in data_lines[1] and "309.79" in data_lines[1]


def test_volatility_is_finite_and_nonzero_for_movement():
    """Daily volatility must be computed (not NaN / not 0) for non-flat input."""
    get_YFin_data_online = _import_target()
    with patch("tradingagents.dataflows.y_finance.yf.Ticker", _StubTicker), \
         patch("tradingagents.dataflows.y_finance.yf_retry",
               lambda fn: fn()):
        out = get_YFin_data_online("SOXX", "2026-03-02", "2026-03-04")

    # Find the volatility line and parse its number.
    vol_line = next(
        ln for ln in out.splitlines() if "daily_volatility_pct" in ln
    )
    vol_str = vol_line.split(":")[1].strip()
    vol = float(vol_str)
    assert vol > 0, "volatility must be positive for a moving series"
    assert vol < 100, "sanity: daily volatility above 100% means a bug"


def test_empty_dataframe_raises_no_market_data_without_stats():
    """Empty-data path must raise typed no-data before computing stats."""
    get_YFin_data_online = _import_target()

    class _EmptyTicker:
        def __init__(self, _s):
            pass
        def history(self, **_kw):
            return pd.DataFrame()

    with patch("tradingagents.dataflows.y_finance.yf.Ticker", _EmptyTicker), \
         patch("tradingagents.dataflows.y_finance.yf_retry",
               lambda fn: fn()), \
         pytest.raises(NoMarketDataError) as exc:
        get_YFin_data_online("XYZ", "2026-03-02", "2026-03-04")

    assert "no rows between 2026-03-02 and 2026-03-04" in str(exc.value)
