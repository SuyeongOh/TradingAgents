from __future__ import annotations

import fcntl
import json
import logging
import os
import subprocess
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

_STATE_PATH = Path(
    os.getenv(
        "TRADINGAGENTS_OLLAMA_LIFECYCLE_STATE",
        "/tmp/tradingagents_ollama_lifecycle.json",
    )
)
_DEFAULT_LOAD_SECONDS = float(os.getenv("TRADINGAGENTS_OLLAMA_DEFAULT_LOAD_SECONDS", "5"))
_MIN_IDLE_SECONDS = float(os.getenv("TRADINGAGENTS_OLLAMA_MIN_IDLE_SECONDS", "1"))
_MAX_IDLE_SECONDS = float(os.getenv("TRADINGAGENTS_OLLAMA_MAX_IDLE_SECONDS", "300"))
# TRADINGAGENTS_OLLAMA_BASE_URL wins; OLLAMA_BASE_URL is kept for Ollama tooling compatibility.
_OLLAMA_BASE_URL = (
    os.getenv("TRADINGAGENTS_OLLAMA_BASE_URL")
    or os.getenv("OLLAMA_BASE_URL")
    or "http://localhost:11434"
)


def cleanup_disabled() -> bool:
    """Return True when the user intentionally wants Ollama models kept loaded."""
    return os.getenv("TRADINGAGENTS_OLLAMA_KEEP_LOADED") == "1"


def normalize_models(models: Iterable[str | None]) -> list[str]:
    """Return stable, unique Ollama model names."""
    return sorted({model for model in models if model})


@contextmanager
def _locked_state():
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _STATE_PATH.open("a+", encoding="utf-8") as state_file:
        fcntl.flock(state_file.fileno(), fcntl.LOCK_EX)
        state_file.seek(0)
        try:
            state = json.load(state_file)
        except json.JSONDecodeError:
            state = {}

        state.setdefault("models", {})
        state.setdefault("active_run_pids", {})
        try:
            yield state
        finally:
            state_file.seek(0)
            state_file.truncate()
            json.dump(state, state_file, indent=2, sort_keys=True)
            state_file.flush()
            os.fsync(state_file.fileno())
            fcntl.flock(state_file.fileno(), fcntl.LOCK_UN)


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _model_entry(state: dict, model: str) -> dict:
    return state["models"].setdefault(
        model,
        {"active_runs": 0, "last_load_seconds": _DEFAULT_LOAD_SECONDS},
    )


def _gc_model_pids(state: dict, model: str) -> None:
    pids_by_model = state.setdefault("active_run_pids", {})
    raw_pids = pids_by_model.setdefault(model, [])
    live_pids: list[int] = []
    for raw_pid in raw_pids:
        try:
            pid = int(raw_pid)
        except (TypeError, ValueError):
            continue
        if pid > 0 and _is_pid_alive(pid):
            live_pids.append(pid)
    pids_by_model[model] = live_pids
    _model_entry(state, model)["active_runs"] = len(live_pids)


def register_run_start(models: Iterable[str | None]) -> list[str]:
    """Register that this process is about to use the given Ollama models."""
    model_names = normalize_models(models)
    if cleanup_disabled() or not model_names:
        return model_names

    pid = os.getpid()
    with _locked_state() as state:
        for model in model_names:
            _gc_model_pids(state, model)
            state["active_run_pids"].setdefault(model, []).append(pid)
            entry = _model_entry(state, model)
            entry["active_runs"] = len(state["active_run_pids"][model])
    return model_names


def _cached_load_seconds(model: str) -> float:
    with _locked_state() as state:
        entry = state["models"].setdefault(
            model,
            {"active_runs": 0, "last_load_seconds": _DEFAULT_LOAD_SECONDS},
        )
        return float(entry.get("last_load_seconds") or _DEFAULT_LOAD_SECONDS)


def _record_load_seconds(model: str, seconds: float) -> None:
    bounded = max(_MIN_IDLE_SECONDS, min(_MAX_IDLE_SECONDS, seconds))
    with _locked_state() as state:
        entry = state["models"].setdefault(
            model,
            {"active_runs": 0, "last_load_seconds": _DEFAULT_LOAD_SECONDS},
        )
        entry["last_load_seconds"] = bounded
        entry["last_load_measured_at"] = time.time()


def measure_model_load_seconds(model: str) -> float:
    """Preload the model and record the latest observed Ollama load duration.

    Ollama's native API returns ``load_duration`` in nanoseconds. When the model
    is already warm that value can be near zero, so we keep the previous cold
    load measurement instead of shrinking the idle window to zero.
    """
    if cleanup_disabled():
        return _cached_load_seconds(model)

    payload = json.dumps(
        {
            "model": model,
            "prompt": " ",
            "stream": False,
            "options": {"num_predict": 1},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{_OLLAMA_BASE_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            body = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        logger.warning("Could not measure Ollama load time for %s: %s", model, exc)
        return _cached_load_seconds(model)

    elapsed = time.monotonic() - started
    load_seconds = float(body.get("load_duration") or 0) / 1_000_000_000
    if load_seconds < _MIN_IDLE_SECONDS:
        return _cached_load_seconds(model)

    # Include the API round-trip overhead; it is small, but makes the idle
    # window match the user-visible cold-start cost more closely.
    measured = max(load_seconds, elapsed)
    _record_load_seconds(model, measured)
    return measured


def measure_models_load_seconds(models: Iterable[str | None]) -> float:
    model_names = normalize_models(models)
    if not model_names:
        return _DEFAULT_LOAD_SECONDS
    return max(measure_model_load_seconds(model) for model in model_names)


def finish_run_and_stop_when_idle(models: Iterable[str | None]) -> None:
    """Mark a run complete and unload models after load-time-sized idle grace."""
    model_names = normalize_models(models)
    if cleanup_disabled() or not model_names:
        return

    pid = os.getpid()
    idle_seconds = _DEFAULT_LOAD_SECONDS
    with _locked_state() as state:
        for model in model_names:
            _gc_model_pids(state, model)
            pids = state["active_run_pids"].setdefault(model, [])
            if pid in pids:
                pids.remove(pid)
            entry = _model_entry(state, model)
            entry["active_runs"] = len(pids)
            idle_seconds = max(
                idle_seconds,
                float(entry.get("last_load_seconds") or _DEFAULT_LOAD_SECONDS),
            )

        if any(
            int(state["models"].get(model, {}).get("active_runs", 0)) > 0
            for model in model_names
        ):
            return

    idle_seconds = max(_MIN_IDLE_SECONDS, min(_MAX_IDLE_SECONDS, idle_seconds))
    logger.info("Ollama idle cleanup waiting %.2fs before unload", idle_seconds)
    time.sleep(idle_seconds)

    with _locked_state() as state:
        if any(
            int(state["models"].get(model, {}).get("active_runs", 0)) > 0
            for model in model_names
        ):
            return

    for model in model_names:
        stop_model(model)


def stop_model(model: str) -> None:
    """Unload one Ollama model while keeping the Ollama server process alive."""
    try:
        result = subprocess.run(
            ["ollama", "stop", model],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        logger.warning("Could not stop Ollama model %s: %s", model, exc)
        return

    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        logger.debug(
            "Ollama stop returned %s for model %s: %s",
            result.returncode,
            model,
            detail,
        )
