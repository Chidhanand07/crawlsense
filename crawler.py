"""Stage 1 — automated browsing and subpage discovery via headless Chromium.

Playwright is used rather than ``requests`` because most modern marketing sites
render their copy client-side; a plain HTTP GET on many of them returns an empty
``<div id="root">``. The browser executes the page's JavaScript first, so the
text we hand downstream is the text a human would actually see.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from playwright.async_api import Browser, Page, async_playwright

# Subpage keywords, most valuable first. Ordering matters: with a budget of only
# a handful of subpages, an /about page is worth more to lead enrichment than a
# /pricing page, so higher-priority matches win the limited slots.
TARGET_KEYWORDS: tuple[str, ...] = (
    "about",
    "team",
    "leadership",
    "founders",
    "company",
    "contact",
    "pricing",
)

MAX_SUBPAGES = 5
NAV_TIMEOUT_MS = 20_000
RENDER_SETTLE_MS = 1_500

# Path segments that reliably contain no company intelligence. Matched per
# segment rather than as substrings, so "/company/careers" is excluded (it is not
# "/careers/") while "/about-careers-at-x" is not accidentally caught.
EXCLUDED_SEGMENTS: frozenset[str] = frozenset(
    {
        "blog",
        "docs",
        "documentation",
        "careers",
        "jobs",
        "legal",
        "privacy",
        "terms",
        "login",
        "signup",
        "status",
        "support",
        "press-media",
        "customers",
        "case-studies",
        "case-study",
        "testimonials",
    }
)

# A realistic desktop UA. Marketing sites frequently serve a challenge page to
# obvious bot user-agents, which would leave us with nothing to extract.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

# Images, fonts and media are pure cost: they slow every page load and contain no
# text. Aborting them typically halves crawl time.
BLOCKED_RESOURCE_TYPES = {"image", "media", "font"}


async def _block_heavy_resources(route) -> None:  # pragma: no cover - thin shim
    """Playwright route handler that aborts non-textual resource requests."""
    if route.request.resource_type in BLOCKED_RESOURCE_TYPES:
        await route.abort()
    else:
        await route.continue_()


async def fetch_page(page: Page, url: str) -> Optional[str]:
    """Navigate to ``url`` and return its rendered HTML, or ``None`` on failure.

    Every failure mode here — DNS error, TLS error, timeout, navigation abort —
    is caught and reported as ``None``. A single unreachable subpage must never
    take down the run.
    """
    try:
        response = await page.goto(url, timeout=NAV_TIMEOUT_MS, wait_until="domcontentloaded")
        if response is not None and response.status >= 400:
            print(f"  [warn] {url} returned HTTP {response.status}")
            return None
        # Give client-side frameworks a moment to paint their content.
        await page.wait_for_timeout(RENDER_SETTLE_MS)
        return await page.content()
    except Exception as exc:
        print(f"  [warn] failed to fetch {url}: {type(exc).__name__}: {exc}")
        return None


# Where a keyword matched, in descending order of trustworthiness. A keyword in
# the URL path is strong evidence about the page's subject; a keyword in the link
# text is a weak hint that is frequently incidental ("Learn about Dependabot"
# pointing at a product page). Kept as an explicit rank so the two never compete
# on equal footing.
_MATCH_PATH_EXACT = 0    # /about
_MATCH_PATH_SEGMENT = 1  # /company/about
_MATCH_PATH_PARTIAL = 2  # /company/about-postman
_MATCH_TEXT_ONLY = 3     # <a href="/security/...">Learn about ...</a>

# Sentinel score meaning "no keyword matched"; sorts after every real candidate.
_NO_MATCH_SCORE = (2, len(TARGET_KEYWORDS), _MATCH_TEXT_ONLY + 1)


def _score_link(path: str, link_text: str) -> tuple[int, int, int]:
    """Rank one candidate link as ``(text_only, keyword_priority, specificity)``.

    Lower is better on every axis, compared left to right:

    1. ``text_only`` — whether the keyword was found *only* in the anchor text
       rather than anywhere in the URL path. This dominates, because incidental
       phrasing like "Learn about Dependabot" otherwise scores as highly as a
       real ``/about`` page. Observed on github.com, where three tangential
       ``/security/...`` pages displaced ``/pricing`` entirely.
    2. ``keyword_priority`` — position in ``TARGET_KEYWORDS``. An ``/about`` page
       is worth more to lead enrichment than a ``/pricing`` page, and that stays
       true however precisely each one matched.
    3. ``specificity`` — exact path, then whole segment, then substring. Only a
       tiebreak between pages of equal keyword value, so ``/company/about-us``
       still outranks ``/pricing``.

    Args:
        path: Lowercased URL path of the candidate.
        link_text: Lowercased visible text of the anchor.

    Returns:
        ``_NO_MATCH_SCORE`` when no keyword matches at all.
    """
    segments = [segment for segment in path.split("/") if segment]
    best = _NO_MATCH_SCORE

    for priority, keyword in enumerate(TARGET_KEYWORDS):
        if segments == [keyword]:
            candidate = (0, priority, _MATCH_PATH_EXACT)
        elif keyword in segments:
            candidate = (0, priority, _MATCH_PATH_SEGMENT)
        elif any(keyword in segment for segment in segments):
            candidate = (0, priority, _MATCH_PATH_PARTIAL)
        elif keyword in link_text:
            candidate = (1, priority, _MATCH_TEXT_ONLY)
        else:
            continue

        best = min(best, candidate)

    return best


def discover_links(html: str, base_url: str, limit: int = MAX_SUBPAGES) -> List[str]:
    """Find on-domain subpages worth crawling, ranked by match quality.

    Candidates are ordered by, in turn: whether the match came from the path or
    only the link text, the keyword's own priority (``/about`` beats
    ``/pricing``), how specifically the path matched, path shallowness, then URL
    length as a stable tiebreak.

    Args:
        html: Rendered homepage HTML.
        base_url: The homepage URL, used to resolve relative hrefs and to pin
            the crawl to a single host.
        limit: Maximum number of subpages to return.

    Returns:
        Up to ``limit`` absolute URLs, best-first, with fragments stripped and
        duplicates removed.
    """
    soup = BeautifulSoup(html, "html.parser")
    base_domain = urlparse(base_url).netloc

    # url -> (text_only, keyword_priority, specificity, path_depth, url_length)
    ranked: Dict[str, tuple[int, int, int, int, int]] = {}

    for anchor in soup.find_all("a", href=True):
        full_url = urljoin(base_url, anchor["href"]).split("#")[0].rstrip("/")
        parsed = urlparse(full_url)

        if parsed.scheme not in ("http", "https"):
            continue
        if parsed.netloc != base_domain:
            continue
        if full_url == base_url.rstrip("/"):
            continue

        path_lower = parsed.path.lower()
        segments = [segment for segment in path_lower.split("/") if segment]
        if set(segments) & EXCLUDED_SEGMENTS:
            continue

        match = _score_link(path_lower, (anchor.get_text() or "").lower())
        if match == _NO_MATCH_SCORE:
            continue

        score = (*match, len(segments), len(full_url))
        # The same URL can appear under several anchors; keep its best score.
        if score < ranked.get(full_url, (*_NO_MATCH_SCORE, 99, 99)):
            ranked[full_url] = score

    return sorted(ranked, key=lambda url: ranked[url])[:limit]


async def _crawl_with_browser(browser: Browser, domain: str) -> Dict[str, str]:
    """Crawl one domain using an already-running browser instance."""
    context = await browser.new_context(user_agent=USER_AGENT, viewport={"width": 1280, "height": 900})
    await context.route("**/*", _block_heavy_resources)
    page = await context.new_page()
    pages: Dict[str, str] = {}

    try:
        # Try https, then the www. variant, then plain http before giving up.
        candidates = [f"https://{domain}", f"https://www.{domain}", f"http://{domain}"]
        home_url, home_html = "", None
        for candidate in candidates:
            home_html = await fetch_page(page, candidate)
            if home_html:
                home_url = candidate
                break

        if not home_html:
            return pages

        pages[home_url] = home_html

        for url in discover_links(home_html, home_url):
            html = await fetch_page(page, url)
            if html:
                pages[url] = html
    finally:
        await context.close()

    return pages


async def crawl_domain(domain: str) -> Dict[str, str]:
    """Crawl ``domain``'s homepage plus its highest-value subpages.

    Args:
        domain: A bare domain such as ``"postman.com"`` (no scheme).

    Returns:
        A mapping of ``{url: rendered_html}``. Empty if the site was entirely
        unreachable — the caller treats that as a failed domain.
    """
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            return await _crawl_with_browser(browser, domain)
        finally:
            await browser.close()


if __name__ == "__main__":
    # Smoke test: python crawler.py
    async def _demo() -> None:
        for test_domain in ("postman.com",):
            result = await crawl_domain(test_domain)
            print(f"\n{test_domain}: {len(result)} pages")
            for url, html in result.items():
                print(f"  {url}  ({len(html):,} bytes)")

    asyncio.run(_demo())
