"""Regression guards for the analyst_concurrency_limit wiring chain.

Protects the 3-layer wiring that merge c652198 silently dropped:
DEFAULT_CONFIG -> TradingAgentsGraph -> GraphSetup.__init__ ->
build_analyst_execution_plan(concurrency_limit=...).
"""

import inspect
import unittest
from unittest.mock import MagicMock, patch

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph import setup as graph_setup_module
from tradingagents.graph.setup import GraphSetup


class DefaultConfigAnalystConcurrencyTests(unittest.TestCase):
    def test_default_config_exposes_analyst_concurrency_limit_key(self):
        # Removing this key silently falls back to concurrency=1 — no error,
        # no signal. Regression guard against silent drops like c652198.
        self.assertIn("analyst_concurrency_limit", DEFAULT_CONFIG)
        value = DEFAULT_CONFIG["analyst_concurrency_limit"]
        self.assertIsInstance(value, int)
        self.assertGreaterEqual(value, 1)


class GraphSetupCtorWiringTests(unittest.TestCase):
    def test_init_accepts_analyst_concurrency_limit_kwarg(self):
        params = inspect.signature(GraphSetup.__init__).parameters
        self.assertIn("analyst_concurrency_limit", params)
        default = params["analyst_concurrency_limit"].default
        self.assertIsInstance(default, int)
        self.assertGreaterEqual(default, 1)


class GraphSetupConcurrencyPropagationTests(unittest.TestCase):
    def test_setup_graph_forwards_non_default_concurrency_to_plan_builder(self):
        gs = GraphSetup(
            quick_thinking_llm=MagicMock(),
            deep_thinking_llm=MagicMock(),
            tool_nodes={"market": MagicMock(), "news": MagicMock()},
            conditional_logic=MagicMock(),
            analyst_concurrency_limit=4,
        )

        with patch.object(graph_setup_module, "build_analyst_execution_plan") as mock_build:
            # Short-circuit downstream wiring — only the plan-builder call
            # site is under test.
            mock_build.side_effect = RuntimeError("stop after plan build")
            with self.assertRaises(RuntimeError):
                gs.setup_graph(selected_analysts=("market", "news"))

        mock_build.assert_called_once()
        _, kwargs = mock_build.call_args
        if "concurrency_limit" in kwargs:
            self.assertEqual(kwargs["concurrency_limit"], 4)
        else:
            args = mock_build.call_args.args
            self.assertGreaterEqual(len(args), 2)
            self.assertEqual(args[1], 4)


if __name__ == "__main__":
    unittest.main()
