from __future__ import annotations

import asyncio
import threading
import time


from tests.api_server_test_utils import import_api_server


server = import_api_server()


def test_stream_cancel_signals_executor_worker(monkeypatch):
    worker_done = threading.Event()

    def stub_run_graph(symbol, locale, emit_progress, cancel_event):
        while True:
            time.sleep(1)
            emit_progress("stub")
            if cancel_event.is_set():
                worker_done.set()
                raise server.RunCancelled("cancelled")

    async def run_case():
        monkeypatch.setattr(server, "_run_trading_graph", stub_run_graph)
        stream = server._stream_analysis("AAPL", "en")
        await stream.__anext__()
        await stream.__anext__()

        started = time.monotonic()
        pending_next = asyncio.create_task(stream.__anext__())
        await asyncio.sleep(0.5)
        pending_next.cancel()
        try:
            await pending_next
        except asyncio.CancelledError:
            pass

        assert await asyncio.to_thread(worker_done.wait, 1.2)
        assert time.monotonic() - started < 1.5

    asyncio.run(run_case())


def test_safe_put_keeps_stream_queue_bounded():
    queue: asyncio.Queue[tuple[str, int]] = asyncio.Queue(maxsize=256)

    for i in range(300):
        server._safe_put(queue, ("progress", i))

    assert queue.qsize() == 256
    assert queue.get_nowait() == ("progress", 44)
