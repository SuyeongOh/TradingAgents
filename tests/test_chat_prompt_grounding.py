from __future__ import annotations

from tests.api_server_test_utils import import_api_server


def test_ko_chat_prompt_handles_korean_particle_attached_tickers():
    server = import_api_server()

    prompt = server._build_chat_system_prompt("ko-KR")

    assert "'bnai에 대해'는 symbol='BNAI'" in prompt
    assert "'crcl은'은 symbol='CRCL'" in prompt
    assert "'005930.KS는'은 symbol='005930.KS'" in prompt
    assert "낯선 ticker여도 먼저 도구로 확인하세요" in prompt


def test_ko_chat_prompt_forbids_unverified_nonexistent_claims():
    server = import_api_server()

    prompt = server._build_chat_system_prompt("ko-KR")

    assert "존재하지 않는다'고" in prompt
    assert "단정하지 말고" in prompt
    assert "현재 조회 도구로 확인하지 못했다" in prompt
    assert "ticker, 회사명, 거래소 suffix 확인" in prompt


def test_ko_chat_prompt_softens_investment_drive_without_becoming_evasive():
    server = import_api_server()

    prompt = server._build_chat_system_prompt("ko-KR")

    assert "확인된 데이터가 뒷받침하는 범위" in prompt
    assert "억지로 매수/매도 결론" in prompt
    assert "확인된 사실과 불확실성" in prompt


def test_get_stock_quote_tool_schema_allows_ticker_cleanup_not_substitution():
    server = import_api_server()

    get_quote_tool = next(
        tool
        for tool in server.CHAT_TOOLS_SCHEMA
        if tool["function"]["name"] == "get_stock_quote"
    )
    description = get_quote_tool["function"]["parameters"]["properties"]["symbol"][
        "description"
    ]

    assert "'bnai에 대해' -> symbol='BNAI'" in description
    assert "한국어 조사/구두점만 정리" in description
    assert "다른 symbol로" in description
    assert "추측 대체하지 않는다" in description
