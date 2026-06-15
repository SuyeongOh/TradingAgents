from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def import_api_server():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    _install_fastapi_stub()
    import api.server as server

    return importlib.reload(server)


def _install_fastapi_stub() -> None:
    if "fastapi" in sys.modules:
        return

    fastapi = types.ModuleType("fastapi")
    middleware = types.ModuleType("fastapi.middleware")
    cors = types.ModuleType("fastapi.middleware.cors")
    responses = types.ModuleType("fastapi.responses")

    class _FastAPI:
        def __init__(self, *args, **kwargs):
            self.startup_handlers = []

        def add_middleware(self, *args, **kwargs):
            return None

        def add_event_handler(self, event_type, func):
            if event_type == "startup":
                self.startup_handlers.append(func)

        def get(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

        def post(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

    class _HTTPException(Exception):
        def __init__(self, status_code, detail):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    def _query(default, **kwargs):
        return default

    class _CORSMiddleware:
        pass

    class _StreamingResponse:
        def __init__(self, content, media_type=None, headers=None):
            self.content = content
            self.media_type = media_type
            self.headers = headers or {}

    fastapi.FastAPI = _FastAPI
    fastapi.HTTPException = _HTTPException
    fastapi.Query = _query
    cors.CORSMiddleware = _CORSMiddleware
    responses.StreamingResponse = _StreamingResponse

    sys.modules["fastapi"] = fastapi
    sys.modules["fastapi.middleware"] = middleware
    sys.modules["fastapi.middleware.cors"] = cors
    sys.modules["fastapi.responses"] = responses
