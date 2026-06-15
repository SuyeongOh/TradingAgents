from __future__ import annotations

import json
import urllib.error

import pytest

from tests.api_server_test_utils import import_api_server

server = import_api_server()


class _Response:
    def __init__(self, payload: dict):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


def test_validate_ollama_setup_passes_when_required_models_exist(monkeypatch):
    monkeypatch.delenv("TRADINGAGENTS_SKIP_OLLAMA_VALIDATION", raising=False)
    monkeypatch.setattr(server, "DEFAULT_OLLAMA_QUICK_MODEL", "qwen3:8b")
    monkeypatch.setattr(server, "DEFAULT_OLLAMA_DEEP_MODEL", "qwen3:14b")
    monkeypatch.setattr(
        server.urllib.request,
        "urlopen",
        lambda request, timeout: _Response(
            {"models": [{"name": "qwen3:8b"}, {"model": "qwen3:14b"}]}
        ),
    )

    server.validate_ollama_setup()


def test_validate_ollama_setup_raises_when_model_missing(monkeypatch):
    monkeypatch.delenv("TRADINGAGENTS_SKIP_OLLAMA_VALIDATION", raising=False)
    monkeypatch.setattr(server, "DEFAULT_OLLAMA_QUICK_MODEL", "qwen3:8b")
    monkeypatch.setattr(server, "DEFAULT_OLLAMA_DEEP_MODEL", "qwen3:14b")
    monkeypatch.setattr(
        server.urllib.request,
        "urlopen",
        lambda request, timeout: _Response({"models": [{"name": "qwen3:8b"}]}),
    )

    with pytest.raises(RuntimeError, match="qwen3:14b"):
        server.validate_ollama_setup()


def test_validate_ollama_setup_raises_on_connection_failure(monkeypatch):
    monkeypatch.delenv("TRADINGAGENTS_SKIP_OLLAMA_VALIDATION", raising=False)

    def fail(request, timeout):
        raise urllib.error.URLError(ConnectionRefusedError("connection refused"))

    monkeypatch.setattr(server.urllib.request, "urlopen", fail)

    with pytest.raises(RuntimeError, match="Ollama validation failed"):
        server.validate_ollama_setup()


def test_validate_ollama_setup_can_be_skipped(monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_SKIP_OLLAMA_VALIDATION", "1")

    def fail_if_called(request, timeout):
        raise AssertionError("urlopen should not be called")

    monkeypatch.setattr(server.urllib.request, "urlopen", fail_if_called)

    server.validate_ollama_setup()
