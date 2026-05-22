"""Reddit search fetcher for ticker-specific discussion posts.

Uses Reddit's public JSON endpoints (``reddit.com/r/{sub}/search.json``)
which do not require an API key. Public throughput is ~10 requests per
minute per IP, well within budget for a single agent run that queries
a handful of finance subreddits per ticker.

Returns formatted plaintext blocks ready for prompt injection. Degrades
gracefully — returns a placeholder string rather than raising, so callers
never have to special-case missing data.
"""

from __future__ import annotations

import json
import logging
import concurrent.futures
import threading
import time
from dataclasses import dataclass
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_API = "https://www.reddit.com/r/{sub}/search.json?{qs}"
_UA = "tradingagents/0.2 (+https://github.com/TauricResearch/TradingAgents)"

# Default subreddits ordered roughly by signal density for ticker-specific
# discussion. wallstreetbets has the most volume but most noise; stocks /
# investing trend more measured. Caller can override.
DEFAULT_SUBREDDITS = ("wallstreetbets", "stocks", "investing")


@dataclass(frozen=True)
class _FetchError:
    status: int
    message: str


def _is_unavailable_status(status: int) -> bool:
    return status in {403, 429} or status >= 500


def _unavailable_placeholder(source: str, status: int, message: str) -> str:
    return (
        f"<reddit unavailable: {source} HTTP {status} {message}. "
        f"Data source temporarily unavailable (HTTP {status}). "
        "Treat sentiment signal as missing, not neutral.>"
    )


def _fetch_subreddit(
    ticker: str,
    sub: str,
    limit: int,
    timeout: float,
) -> list[dict] | _FetchError:
    qs = urlencode({
        "q": ticker,
        "restrict_sr": "on",
        "sort": "new",
        "t": "week",  # last 7 days
        "limit": limit,
    })
    url = _API.format(sub=sub, qs=qs)
    req = Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
    except HTTPError as exc:
        status = int(exc.code)
        if _is_unavailable_status(status):
            message = "rate-limited or blocked" if status in {403, 429} else "server error"
            logger.warning("reddit r/%s blocked (status=%d)", sub, status)
            return _FetchError(status=status, message=message)
        logger.warning("Reddit fetch failed for r/%s · %s: %s", sub, ticker, exc)
        return []
    except (URLError, json.JSONDecodeError, TimeoutError) as exc:
        logger.warning("Reddit fetch failed for r/%s · %s: %s", sub, ticker, exc)
        return []
    children = (payload.get("data") or {}).get("children") or []
    return [c.get("data", {}) for c in children if isinstance(c, dict)]


def fetch_reddit_posts(
    ticker: str,
    subreddits: Iterable[str] = DEFAULT_SUBREDDITS,
    limit_per_sub: int = 5,
    timeout: float = 10.0,
    inter_request_delay: float = 0.4,
) -> str:
    """Fetch recent Reddit posts mentioning ``ticker`` across finance
    subreddits and return them as a formatted plaintext block.

    ``inter_request_delay`` is retained for backward compatibility; subreddit
    fetches now run in a small bounded pool so the default three sources do not
    add serial sleep latency.
    """
    subreddit_list = list(subreddits)
    if not subreddit_list:
        return f"<no Reddit posts found mentioning {ticker.upper()} across  in the past 7 days>"

    blocks = []
    total_posts = 0
    had_unavailable = False
    results: dict[str, list[dict] | _FetchError] = {}
    stop_event = threading.Event()

    def _fetch_one(sub: str) -> tuple[str, list[dict] | _FetchError]:
        if stop_event.is_set():
            return sub, []
        posts = _fetch_subreddit(ticker, sub, limit_per_sub, timeout)
        if isinstance(posts, _FetchError):
            stop_event.set()
        return sub, posts

    if len(subreddit_list) == 1:
        sub = subreddit_list[0]
        results[sub] = _fetch_subreddit(ticker, sub, limit_per_sub, timeout)
    else:
        max_workers = min(3, len(subreddit_list))
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="reddit",
        ) as executor:
            futures = {
                executor.submit(_fetch_one, sub): sub
                for sub in subreddit_list
            }
            for future in concurrent.futures.as_completed(futures):
                sub, posts = future.result()
                results[sub] = posts

    for sub in subreddit_list:
        posts = results.get(sub, [])
        if isinstance(posts, _FetchError):
            had_unavailable = True
            blocks.append(_unavailable_placeholder(f"r/{sub}", posts.status, posts.message))
            continue
        total_posts += len(posts)
        if not posts:
            blocks.append(f"r/{sub}: <no posts found mentioning {ticker.upper()} in the past 7 days>")
            continue

        lines = [f"r/{sub} — {len(posts)} recent posts mentioning {ticker.upper()}:"]
        for p in posts:
            title = (p.get("title") or "").replace("\n", " ").strip()
            score = p.get("score", 0)
            comments = p.get("num_comments", 0)
            created = p.get("created_utc")
            created_str = (
                time.strftime("%Y-%m-%d", time.gmtime(created)) if created else "?"
            )
            selftext = (p.get("selftext") or "").replace("\n", " ").strip()
            if len(selftext) > 240:
                selftext = selftext[:240] + "…"
            lines.append(
                f"  [{created_str} · {score:>4}↑ · {comments:>3}c] {title}"
                + (f"\n    body excerpt: {selftext}" if selftext else "")
            )
        blocks.append("\n".join(lines))

    if total_posts == 0 and not had_unavailable:
        return (
            f"<no Reddit posts found mentioning {ticker.upper()} across "
            f"{', '.join(f'r/{s}' for s in subreddit_list)} in the past 7 days>"
        )
    return "\n\n".join(blocks)
