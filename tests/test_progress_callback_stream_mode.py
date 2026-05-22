from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from tradingagents.graph.trading_graph import TradingAgentsGraph


class _StubGraph:
    def __init__(self):
        self.stream_kwargs = None

    def stream(self, state, **kwargs):
        self.stream_kwargs = kwargs
        yield ("updates", {"Market Analyst": {}})
        yield ("updates", {"News Analyst": {}})
        yield ("values", {"final_trade_decision": "BUY"})

    def invoke(self, state, **kwargs):  # pragma: no cover - fallback should not run
        raise AssertionError("invoke fallback should not run when stream returns values")


def test_run_graph_progress_callback_overrides_stream_mode_without_type_error():
    graph = object.__new__(TradingAgentsGraph)
    graph.config = {"checkpoint_enabled": False}
    graph.debug = False
    graph.memory_log = SimpleNamespace(
        get_past_context=lambda ticker: "",
        store_decision=Mock(),
    )
    graph.propagator = SimpleNamespace(
        create_initial_state=lambda *args, **kwargs: {"messages": []},
        get_graph_args=lambda: {"stream_mode": "values", "config": {}},
    )
    graph.graph = _StubGraph()
    graph._log_state = Mock()
    graph.process_signal = lambda decision: decision
    seen_nodes = []

    final_state, decision = graph._run_graph(
        "AAPL",
        "2026-05-17",
        progress_callback=seen_nodes.append,
    )

    assert final_state == {"final_trade_decision": "BUY"}
    assert decision == "BUY"
    assert graph.graph.stream_kwargs["stream_mode"] == ["updates", "values"]
    assert len(seen_nodes) >= 2
