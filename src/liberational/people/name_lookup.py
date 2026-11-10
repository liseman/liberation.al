"""User-entered text search through Google Custom Search or Brave Web Search.

Web pages are evidence, not inferred identities. Credentials come only from the
service environment or dedicated .env.search. No media, agent configuration, or stored notes are read.
Wikidata remains available as a separate supplementary lookup.
"""

import json
import os
import re
import time
import urllib.parse
from typing import Any

import requests

from liberational import config as cfg

API = "https://www.wikidata.org/w/api.php"
MAX_RESPONSE = 2 * 1024 * 1024
_STOP = frozenset(
    [
        "a",
        "an",
        "the",
        "of",
        "in",
        "on",
        "at",
        "and",
        "or",
        "from",
        "born",
        "died",
        "person",
        "historical",
        "who",
        "is",
        "was",
        "with",
    ]
)


def _get(parameters: dict[str, str]) -> dict[str, Any]:
    deadline = time.monotonic() + 25
    with requests.get(
        API,
        params={"format": "json", **parameters},
        headers={"User-Agent": "Liberational-PeopleLookup/1.0 (explicit public name lookup)"},
        timeout=(5, 15),
        allow_redirects=False,
        stream=True,
    ) as response:
        if response.status_code != 200:
            raise ValueError(f"Wikidata returned HTTP {response.status_code}. Retry later or add manually.")
        raw = bytearray()
        for chunk in response.iter_content(65536):
            if time.monotonic() > deadline or len(raw) + len(chunk) > MAX_RESPONSE:
                raise ValueError("Wikidata response exceeded its time/size limit. Refine the name and retry.")
            raw.extend(chunk)
    result = json.loads(raw)
    if not isinstance(result, dict) or "error" in result:
        raise ValueError("Wikidata returned an unusable response. Retry later or add manually.")
    return result


def tokens(value: str) -> set[str]:
    return {word for word in re.findall(r"\w+", value.casefold()) if word not in _STOP}


def web_candidates(rows: list[Any], provider: str, name: str, context: str) -> list[dict[str, Any]]:
    """Keep provider text and URLs; deduplicate pages, not ambiguous people."""
    found: dict[str, dict[str, Any]] = {}
    wanted = tokens(context)
    for row in rows:
        if not isinstance(row, dict):
            continue
        title = row.get("title")
        snippet = row.get("snippet" if provider == "Google Custom Search" else "description", "")
        source = row.get("link" if provider == "Google Custom Search" else "url")
        if not isinstance(title, str) or not title or not isinstance(snippet, str) or not isinstance(source, str):
            continue
        try:
            url = urllib.parse.urlsplit(source)
            if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password:
                continue
        except ValueError:
            continue
        key = urllib.parse.urlunsplit((url.scheme, url.netloc.lower(), url.path.rstrip("/"), url.query, ""))
        if key in found:
            continue
        evidence = tokens(title + " " + snippet)
        found[key] = {
            "name": name,
            "detail": snippet,
            "url": source,
            "source_title": title,
            "source_snippet": snippet,
            "provider": provider,
            "matched_context": sorted(wanted & evidence),
        }
    return sorted(found.values(), key=lambda item: -len(item["matched_context"]))[:10]


def search_credentials() -> dict[str, str]:
    """Dedicated gitignored KEY=value file, matching the app's phone env convention."""
    values: dict[str, str] = {}
    path = cfg.ROOT / ".env.search"
    try:
        for line in path.read_text().splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and key in {"GOOGLE_CSE_API_KEY", "GOOGLE_CSE_CX", "BRAVE_API_KEY"}:
                values[key] = value.strip().strip('"').strip("'")
    except OSError:
        pass
    for key in ("GOOGLE_CSE_API_KEY", "GOOGLE_CSE_CX", "BRAVE_API_KEY"):
        if os.environ.get(key):
            values[key] = os.environ[key]
    return values


def lookup(name: str, context: str) -> dict[str, Any]:
    credentials = search_credentials()
    google_key = credentials.get("GOOGLE_CSE_API_KEY", "")
    google_cx = credentials.get("GOOGLE_CSE_CX", "")
    brave_key = credentials.get("BRAVE_API_KEY", "")
    query = f"{name.strip()} {context.strip()}".strip()
    headers = {"User-Agent": "Liberational-PeopleLookup/1.0"}
    if google_key and google_cx:
        label = "Google Custom Search"
        endpoint = "https://www.googleapis.com/customsearch/v1"
        params = {"key": google_key, "cx": google_cx, "q": query, "num": "10"}
    elif brave_key:
        label = "Brave Web Search"
        endpoint = "https://api.search.brave.com/res/v1/web/search"
        params = {"q": query, "count": "10", "text_decorations": "false"}
        headers["X-Subscription-Token"] = brave_key
    else:
        return {
            "error": "General web search needs BRAVE_API_KEY or GOOGLE_CSE_API_KEY and GOOGLE_CSE_CX in .env.search or the dashboard service environment. Add manually meanwhile."
        }
    try:
        deadline = time.monotonic() + 20
        with requests.get(
            endpoint, params=params, headers=headers, timeout=(5, 10), allow_redirects=False, stream=True
        ) as response:
            if response.status_code != 200:
                return {"error": f"{label} returned HTTP {response.status_code}. Retry later or add manually."}
            raw = bytearray()
            for chunk in response.iter_content(65536):
                if time.monotonic() > deadline or len(raw) + len(chunk) > MAX_RESPONSE:
                    raise ValueError("response limit")
                raw.extend(chunk)
        data = json.loads(raw)
        rows = data.get("items", []) if label == "Google Custom Search" else data.get("web", {}).get("results", [])
        if not isinstance(rows, list) or "error" in data:
            raise ValueError("invalid results")
        return {
            "candidates": web_candidates(rows, label, name, context),
            "supplementary_url": "https://www.wikidata.org/w/index.php?" + urllib.parse.urlencode({"search": name}),
            "provider": label,
            "note": f"{label}: general web pages, not verified identities. Only your entered name and context are sent. Context refines the query and ranks matching snippets, without excluding pages with incomplete snippets. Preview and check sources, or add manually.",
        }
    except requests.RequestException, ValueError, TypeError, AttributeError:
        # Exceptions can contain a credential-bearing Google URL: never return them.
        return {"error": f"{label} unavailable or returned an unusable response. Retry later or add manually."}
