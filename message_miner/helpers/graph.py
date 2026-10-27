"""
helpers/graph.py

Graph API request helpers for message-miner.
Exports:
  graph_get(token, endpoint, params)  -> dict
  graph_get_all(token, endpoint, params) -> list

All Graph API calls in this project go through these two functions.
Handles HTTP 429 rate limiting (Retry-After header) and @odata.nextLink pagination.
"""

import json as _json
import time

import requests

_GRAPH_BASE = "https://graph.microsoft.com/v1.0"
_MAX_RETRIES = 5
_JSON_RETRIES = 3
_JSON_RETRY_DELAY = 5


def graph_get(token: str, endpoint: str, params: dict | None = None) -> dict:
    """
    GET a single page from the Graph API.

    endpoint: either a full URL (used for @odata.nextLink) or a path like "/me/chats".
    Returns the parsed JSON response body.
    Raises RuntimeError after MAX_RETRIES 429 responses, or on any other HTTP error.
    Retries up to JSON_RETRIES times (5s delay) on malformed JSON responses.
    """
    url = endpoint if endpoint.startswith("http") else f"{_GRAPH_BASE}{endpoint}"
    headers = {"Authorization": f"Bearer {token}"}
    json_attempts = 0

    for attempt in range(1, _MAX_RETRIES + 1):
        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", "30"))
            print(
                f"  [rate limit] waiting {retry_after}s "
                f"(attempt {attempt}/{_MAX_RETRIES})..."
            )
            time.sleep(retry_after)
            params = None  # params are baked into the URL for retries
            continue

        response.raise_for_status()

        try:
            return response.json()
        except _json.JSONDecodeError:
            json_attempts += 1
            if json_attempts < _JSON_RETRIES:
                print(
                    f"  [json error] malformed response, retrying in {_JSON_RETRY_DELAY}s "
                    f"(attempt {json_attempts}/{_JSON_RETRIES})..."
                )
                time.sleep(_JSON_RETRY_DELAY)
                params = None
                continue
            raise

    raise RuntimeError(
        f"Max retries ({_MAX_RETRIES}) exceeded for endpoint: {endpoint}"
    )


def graph_get_all(
    token: str,
    endpoint: str,
    params: dict | None = None,
    since: str | None = None,
    on_page: "callable | None" = None,
) -> list:
    """
    GET all pages from a paginated Graph API endpoint.

    Follows @odata.nextLink chains until exhausted.
    Returns a flat list of all items from all pages' "value" arrays.

    If *since* is an ISO 8601 datetime string (e.g. "2025-01-01T00:00:00Z"),
    only items whose ``createdDateTime`` is >= *since* are returned and
    pagination stops early once a page contains items older than the cutoff
    (the Graph API returns chat messages newest-first).

    If *on_page* is provided it is called after each page with (total_so_far,)
    so callers can display progress.
    """
    results = []

    data = graph_get(token, endpoint, params)
    page_items = data.get("value", [])

    if since:
        filtered = [m for m in page_items if (m.get("createdDateTime") or "") >= since]
        results.extend(filtered)
        if on_page:
            on_page(len(results))
        if len(filtered) < len(page_items):
            return results
    else:
        results.extend(page_items)
        if on_page:
            on_page(len(results))

    while "@odata.nextLink" in data:
        # nextLink is a full URL with $skiptoken already embedded; pass no params
        data = graph_get(token, data["@odata.nextLink"])
        page_items = data.get("value", [])

        if since:
            # Graph API returns messages newest-first, so the first page containing
            # an item older than `since` means all subsequent pages are also older —
            # safe to stop pagination early.
            filtered = [m for m in page_items if (m.get("createdDateTime") or "") >= since]
            results.extend(filtered)
            if on_page:
                on_page(len(results))
            if len(filtered) < len(page_items):
                return results
        else:
            results.extend(page_items)
            if on_page:
                on_page(len(results))

    return results
