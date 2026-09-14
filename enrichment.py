"""Optional bonus stage — search-engine fallback for missing LinkedIn URLs.

``key_leadership`` entries carry a ``linkedin_url`` only when that URL appears
verbatim in the crawled page text, which is rare. This module fills the gap by
querying a search API for the person's profile.

It is **entirely optional and disabled by default**. Without a provider key in
the environment the functions are a no-op, so the core pipeline never depends on
a third-party service being configured, reachable, or in credit.

Enable by setting one of the following in ``.env``:

    TAVILY_API_KEY=tvly-...      # https://tavily.com  (free tier)
    SERPAPI_API_KEY=...          # https://serpapi.com

A search result is only accepted when the profile URL or title plausibly matches
the person's name. Without that check the fallback would happily attach a
stranger's profile to a name — which is worse than leaving the field null, and
would undermine the whole point of the confidence score.
"""

from __future__ import annotations

import os
import re
from typing import List, Optional
from urllib.parse import urlparse

import requests

from schema import Person

SEARCH_TIMEOUT_S = 10
MAX_RESULTS = 5

# Matches a personal LinkedIn profile, not a company page or post.
LINKEDIN_PROFILE_RE = re.compile(r"https?://(?:[a-z]{2,3}\.)?linkedin\.com/in/[\w\-%.]+", re.I)


def _provider() -> Optional[str]:
    """Return the configured search provider, or ``None`` if the bonus is off."""
    if os.environ.get("TAVILY_API_KEY"):
        return "tavily"
    if os.environ.get("SERPAPI_API_KEY"):
        return "serpapi"
    return None


def _search_tavily(query: str) -> List[str]:
    """Query Tavily, returning result URLs. Returns ``[]`` on any failure."""
    response = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": os.environ["TAVILY_API_KEY"],
            "query": query,
            "max_results": MAX_RESULTS,
        },
        timeout=SEARCH_TIMEOUT_S,
    )
    response.raise_for_status()
    return [r.get("url", "") for r in response.json().get("results", [])]


def _search_serpapi(query: str) -> List[str]:
    """Query SerpAPI, returning result URLs. Returns ``[]`` on any failure."""
    response = requests.get(
        "https://serpapi.com/search",
        params={"q": query, "api_key": os.environ["SERPAPI_API_KEY"], "num": MAX_RESULTS},
        timeout=SEARCH_TIMEOUT_S,
    )
    response.raise_for_status()
    return [r.get("link", "") for r in response.json().get("organic_results", [])]


def _search(query: str) -> List[str]:
    """Dispatch to the configured provider, swallowing provider errors.

    A search failure must never take down a domain — the LinkedIn URL is a bonus
    field, so the correct behaviour on error is to leave it null and move on.
    """
    provider = _provider()
    try:
        if provider == "tavily":
            return _search_tavily(query)
        if provider == "serpapi":
            return _search_serpapi(query)
    except Exception as exc:
        print(f"  [warn] LinkedIn search failed ({provider}): {type(exc).__name__}: {exc}")
    return []


def _name_matches_profile(name: str, url: str) -> bool:
    """Check that a LinkedIn profile URL plausibly belongs to ``name``.

    LinkedIn slugs are derived from the member's name, so requiring every name
    part of length 3+ to appear in the slug is a cheap, high-precision guard. It
    rejects the common failure where a search for a little-known executive
    returns a well-known stranger's profile.

    Args:
        name: The person's name as extracted from the site.
        url: A candidate ``linkedin.com/in/...`` URL.

    Returns:
        ``True`` only if every substantial name part appears in the URL slug.
    """
    slug = urlparse(url).path.lower().replace("-", "").replace("_", "")
    parts = [p for p in re.split(r"[^\w]+", name.lower()) if len(p) >= 3]
    return bool(parts) and all(part in slug for part in parts)


def find_linkedin_url(name: str, domain: str) -> Optional[str]:
    """Search for one person's LinkedIn profile URL.

    Args:
        name: Person's full name.
        domain: Their company's domain, used to disambiguate common names.

    Returns:
        A validated profile URL, or ``None`` if the bonus is disabled, the search
        failed, or no result passed the name check.
    """
    if not _provider():
        return None

    for url in _search(f'"{name}" {domain} linkedin'):
        match = LINKEDIN_PROFILE_RE.search(url or "")
        if match and _name_matches_profile(name, match.group(0)):
            return match.group(0)
    return None


def enrich_linkedin_urls(people: List[Person], domain: str) -> List[Person]:
    """Fill in missing ``linkedin_url`` values via search, where possible.

    People who already have a URL from the page text are left untouched — text
    found on the company's own site is stronger evidence than a search hit.

    Args:
        people: Leadership entries, post-filtering.
        domain: The company domain being enriched.

    Returns:
        The same list, with some ``linkedin_url`` fields possibly populated. This
        function never raises and never removes anyone.
    """
    if not _provider():
        return people

    for person in people:
        if person.linkedin_url:
            continue
        found = find_linkedin_url(person.name, domain)
        if found:
            person.linkedin_url = found
            print(f"  [enrich] found LinkedIn for {person.name}")
    return people
