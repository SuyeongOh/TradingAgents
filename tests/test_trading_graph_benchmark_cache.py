from __future__ import annotations

import importlib
import sys
import types

import pandas as pd


def _install_trading_graph_import_stubs():
    sys.modules.setdefault("yfinance", types.SimpleNamespace(Ticker=None))

    langgraph = sys.modules.setdefault("langgraph", types.ModuleType("langgraph"))
    prebuilt = types.ModuleType("langgraph.prebuilt")

    class ToolNode:
        def __init__(self, *args, **kwargs):
            pass

    prebuilt.ToolNode = ToolNode
    sys.modules.setdefault("langgraph.prebuilt", prebuilt)

    llm_clients = types.ModuleType("tradingagents.llm_clients")
    llm_clients.create_llm_client = lambda *args, **kwargs: None
    sys.modules.setdefault("tradingagents.llm_clients", llm_clients)

    agents = types.ModuleType("tradingagents.agents")
    agents.__all__ = []
    sys.modules.setdefault("tradingagents.agents", agents)

    default_config = types.ModuleType("tradingagents.default_config")
    default_config.DEFAULT_CONFIG = {}
    sys.modules.setdefault("tradingagents.default_config", default_config)

    memory = types.ModuleType("tradingagents.agents.utils.memory")
    memory.TradingMemoryLog = object
    sys.modules.setdefault("tradingagents.agents.utils.memory", memory)

    dataflows_utils = types.ModuleType("tradingagents.dataflows.utils")
    dataflows_utils.safe_ticker_component = lambda ticker: ticker
    sys.modules.setdefault("tradingagents.dataflows.utils", dataflows_utils)

    states = types.ModuleType("tradingagents.agents.utils.agent_states")
    states.AgentState = dict
    states.InvestDebateState = dict
    states.RiskDebateState = dict
    sys.modules.setdefault("tradingagents.agents.utils.agent_states", states)

    config = types.ModuleType("tradingagents.dataflows.config")
    config.set_config = lambda cfg: None
    sys.modules.setdefault("tradingagents.dataflows.config", config)

    agent_utils = types.ModuleType("tradingagents.agents.utils.agent_utils")
    for name in (
        "get_stock_data",
        "get_indicators",
        "get_fundamentals",
        "get_balance_sheet",
        "get_cashflow",
        "get_income_statement",
        "get_news",
        "get_insider_transactions",
        "get_global_news",
    ):
        setattr(agent_utils, name, lambda *args, **kwargs: None)
    sys.modules.setdefault("tradingagents.agents.utils.agent_utils", agent_utils)

    checkpointer = types.ModuleType("tradingagents.graph.checkpointer")
    checkpointer.checkpoint_step = lambda *args, **kwargs: None
    checkpointer.clear_checkpoint = lambda *args, **kwargs: None
    checkpointer.get_checkpointer = lambda *args, **kwargs: None
    checkpointer.thread_id = lambda *args, **kwargs: None
    sys.modules.setdefault("tradingagents.graph.checkpointer", checkpointer)

    conditional_logic = types.ModuleType("tradingagents.graph.conditional_logic")
    conditional_logic.ConditionalLogic = object
    sys.modules.setdefault("tradingagents.graph.conditional_logic", conditional_logic)

    setup = types.ModuleType("tradingagents.graph.setup")
    setup.GraphSetup = object
    sys.modules.setdefault("tradingagents.graph.setup", setup)

    propagation = types.ModuleType("tradingagents.graph.propagation")
    propagation.Propagator = object
    sys.modules.setdefault("tradingagents.graph.propagation", propagation)

    reflection = types.ModuleType("tradingagents.graph.reflection")
    reflection.Reflector = object
    sys.modules.setdefault("tradingagents.graph.reflection", reflection)

    signal_processing = types.ModuleType("tradingagents.graph.signal_processing")
    signal_processing.SignalProcessor = object
    sys.modules.setdefault("tradingagents.graph.signal_processing", signal_processing)


try:
    trading_graph = importlib.import_module("tradingagents.graph.trading_graph")
except ModuleNotFoundError:
    _install_trading_graph_import_stubs()
    for module_name in ("tradingagents.graph", "tradingagents.graph.trading_graph"):
        sys.modules.pop(module_name, None)
    trading_graph = importlib.import_module("tradingagents.graph.trading_graph")

TradingAgentsGraph = trading_graph.TradingAgentsGraph


class _MemoryLog:
    def __init__(self):
        self.updates = []

    def get_pending_entries(self):
        return [
            {"ticker": "SOXX", "date": "2026-05-01", "decision": "Buy"},
            {"ticker": "SOXX", "date": "2026-05-01", "decision": "Hold"},
            {"ticker": "SOXX", "date": "2026-05-01", "decision": "Sell"},
        ]

    def batch_update_with_outcomes(self, updates):
        self.updates = updates


class _Reflector:
    def reflect_on_final_decision(
        self, final_decision, raw_return, alpha_return, benchmark_name
    ):
        return f"{benchmark_name}: {final_decision}"


def _price_df(values):
    return pd.DataFrame({"Close": values})


def test_resolve_pending_entries_fetches_benchmark_history_once(monkeypatch):
    calls = []

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, start, end):
            calls.append((self.symbol, start, end))
            if self.symbol == "SOXX":
                return _price_df([100.0, 103.0])
            return _price_df([100.0, 101.0])

    monkeypatch.setattr(trading_graph.yf, "Ticker", FakeTicker)

    graph = object.__new__(TradingAgentsGraph)
    graph.config = {"benchmark_map": {"": "SPY"}}
    graph.memory_log = _MemoryLog()
    graph.reflector = _Reflector()

    graph._resolve_pending_entries("SOXX")

    benchmark_calls = [call for call in calls if call[0] == "SPY"]
    stock_calls = [call for call in calls if call[0] == "SOXX"]
    assert len(benchmark_calls) == 1
    assert len(stock_calls) == 3
    assert len(graph.memory_log.updates) == 3
