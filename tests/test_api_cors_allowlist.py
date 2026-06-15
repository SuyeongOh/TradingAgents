from __future__ import annotations

from tests.api_server_test_utils import import_api_server


def test_cors_allow_origins_default_is_empty_and_localhost_regex_remains(
    monkeypatch,
):
    monkeypatch.delenv("CORS_ALLOW_ORIGINS", raising=False)

    server = import_api_server()

    assert server._cors_allow_origins() == []
    assert "localhost" in server.DEFAULT_CORS_ALLOW_ORIGIN_REGEX
    assert "127" in server.DEFAULT_CORS_ALLOW_ORIGIN_REGEX


def test_cors_allow_origins_parses_comma_separated_netlify_origins(monkeypatch):
    monkeypatch.setenv(
        "CORS_ALLOW_ORIGINS",
        " https://tradinghelper.netlify.app/ , https://deploy-preview.netlify.app ",
    )

    server = import_api_server()

    assert server._cors_allow_origins() == [
        "https://tradinghelper.netlify.app",
        "https://deploy-preview.netlify.app",
    ]
