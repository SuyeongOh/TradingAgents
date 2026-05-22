from __future__ import annotations

import json
import os

from tradingagents.llm_clients import ollama_lifecycle


def _read_state(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_register_records_current_pid(tmp_path, monkeypatch):
    state_path = tmp_path / "lifecycle.json"
    monkeypatch.setattr(ollama_lifecycle, "_STATE_PATH", state_path)

    ollama_lifecycle.register_run_start(["qwen3:latest"])

    state = _read_state(state_path)
    assert state["active_run_pids"]["qwen3:latest"] == [os.getpid()]
    assert state["models"]["qwen3:latest"]["active_runs"] == 1


def test_register_gc_removes_dead_pid_and_keeps_counter_consistent(tmp_path, monkeypatch):
    state_path = tmp_path / "lifecycle.json"
    state_path.write_text(
        json.dumps(
            {
                "models": {"qwen3:latest": {"active_runs": 1}},
                "active_run_pids": {"qwen3:latest": [99999999]},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(ollama_lifecycle, "_STATE_PATH", state_path)

    ollama_lifecycle.register_run_start(["qwen3:latest"])

    state = _read_state(state_path)
    assert state["active_run_pids"]["qwen3:latest"] == [os.getpid()]
    assert state["models"]["qwen3:latest"]["active_runs"] == 1


def test_finish_removes_current_pid_and_decrements_counter(tmp_path, monkeypatch):
    state_path = tmp_path / "lifecycle.json"
    monkeypatch.setattr(ollama_lifecycle, "_STATE_PATH", state_path)
    monkeypatch.setattr(ollama_lifecycle, "stop_model", lambda model: None)
    monkeypatch.setattr(ollama_lifecycle.time, "sleep", lambda seconds: None)

    ollama_lifecycle.register_run_start(["qwen3:latest"])
    ollama_lifecycle.finish_run_and_stop_when_idle(["qwen3:latest"])

    state = _read_state(state_path)
    assert state["active_run_pids"]["qwen3:latest"] == []
    assert state["models"]["qwen3:latest"]["active_runs"] == 0


def test_repeated_register_finish_keeps_pid_list_and_counter_consistent(
    tmp_path,
    monkeypatch,
):
    state_path = tmp_path / "lifecycle.json"
    monkeypatch.setattr(ollama_lifecycle, "_STATE_PATH", state_path)
    monkeypatch.setattr(ollama_lifecycle, "stop_model", lambda model: None)
    monkeypatch.setattr(ollama_lifecycle.time, "sleep", lambda seconds: None)

    ollama_lifecycle.register_run_start(["qwen3:latest"])
    ollama_lifecycle.register_run_start(["qwen3:latest"])
    state = _read_state(state_path)
    assert len(state["active_run_pids"]["qwen3:latest"]) == 2
    assert state["models"]["qwen3:latest"]["active_runs"] == 2

    ollama_lifecycle.finish_run_and_stop_when_idle(["qwen3:latest"])
    state = _read_state(state_path)
    assert len(state["active_run_pids"]["qwen3:latest"]) == 1
    assert state["models"]["qwen3:latest"]["active_runs"] == 1

    ollama_lifecycle.finish_run_and_stop_when_idle(["qwen3:latest"])
    state = _read_state(state_path)
    assert state["active_run_pids"]["qwen3:latest"] == []
    assert state["models"]["qwen3:latest"]["active_runs"] == 0


def test_v1_state_without_pid_map_migrates(tmp_path, monkeypatch):
    state_path = tmp_path / "lifecycle.json"
    state_path.write_text(
        json.dumps({"models": {"qwen3:latest": {"active_runs": 3}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(ollama_lifecycle, "_STATE_PATH", state_path)

    ollama_lifecycle.register_run_start(["qwen3:latest"])

    state = _read_state(state_path)
    assert state["active_run_pids"]["qwen3:latest"] == [os.getpid()]
    assert state["models"]["qwen3:latest"]["active_runs"] == 1
