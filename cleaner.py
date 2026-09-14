"""Stage 2 — token optimization: rendered HTML down to clean text.

This is the stage that makes the pipeline affordable. A modern marketing page is
roughly 300KB of HTML, the overwhelming majority of which is markup, inline CSS,
JSON state blobs and navigation boilerplate. ``trafilatura`` uses text-density
heuristics to isolate the main content, typically discarding 95%+ of the bytes.

Nothing downstream of here ever sees raw HTML — that is an explicit requirement
of the assignment, and it is enforced structurally by ``build_context`` being the
only path from crawler output to the extractor.
"""

from __future__ import annotations

import re
from typing import Dict, List, Set

import trafilatura

# Per-page character budget. Roughly 4 chars per token, so 4000 chars is about
# 1000 tokens per page — with 6 pages that keeps the prompt near 6K tokens, well
# inside the free-tier context and rate limits.
MAX_CHARS_PER_PAGE = 4_000

# Total budget across all pages, as a second safety net for sites with many
# long subpages.
MAX_TOTAL_CHARS = 20_000

# Matches the generic public mailboxes the assignment asks for. Deliberately
# narrow: personal addresses are not the target, and a broad pattern picks up
# asset filenames and tracking pixels.
GENERIC_EMAIL_RE = re.compile(
    r"\b(?:contact|sales|support|hello|info|press|partnerships|careers|help)"
    r"@[a-z0-9.-]+\.[a-z]{2,}\b",
    re.IGNORECASE,
)


def clean_html(html: str, max_chars: int = MAX_CHARS_PER_PAGE) -> str:
    """Extract the main textual content from a rendered HTML page.

    Args:
        html: Rendered page HTML.
        max_chars: Hard truncation limit applied after extraction.

    Returns:
        Clean plain text, truncated to ``max_chars``. Empty string if
        trafilatura found no meaningful content (e.g. a pure-JS shell or a
        challenge page).
    """
    if not html:
        return ""

    extracted = trafilatura.extract(
        html,
        include_links=False,
        include_formatting=False,
        include_comments=False,
        include_tables=True,
        favor_recall=True,
    )
    if not extracted:
        return ""

    # Collapse runs of blank lines that survive extraction.
    normalized = re.sub(r"\n{3,}", "\n\n", extracted).strip()
    return normalized[:max_chars]


def find_emails(html_pages: Dict[str, str]) -> List[str]:
    """Scrape generic contact emails from raw HTML via regex.

    This runs against the *raw* HTML rather than the cleaned text on purpose:
    contact addresses commonly live in ``mailto:`` hrefs and footers, exactly the
    boilerplate that trafilatura strips. Doing it deterministically in code means
    these addresses are found rather than guessed by the LLM.

    Args:
        html_pages: ``{url: raw_html}`` as returned by the crawler.

    Returns:
        Sorted, de-duplicated, lowercased email addresses.
    """
    found: Set[str] = set()
    for html in html_pages.values():
        for match in GENERIC_EMAIL_RE.findall(html or ""):
            found.add(match.lower())
    return sorted(found)


def build_context(html_pages: Dict[str, str]) -> tuple[str, List[str]]:
    """Turn crawler output into a single clean text blob for the LLM.

    Args:
        html_pages: ``{url: raw_html}`` as returned by the crawler.

    Returns:
        A tuple of ``(context_text, usable_urls)`` where ``usable_urls`` lists
        only the pages that actually yielded text. A page that rendered but
        produced nothing extractable is not counted as crawled evidence, which
        keeps the confidence score honest.
    """
    chunks: List[str] = []
    usable_urls: List[str] = []
    total = 0

    for url, html in html_pages.items():
        cleaned = clean_html(html)
        if not cleaned:
            continue

        remaining = MAX_TOTAL_CHARS - total
        if remaining <= 0:
            break
        cleaned = cleaned[:remaining]

        chunks.append(f"--- SOURCE: {url} ---\n{cleaned}")
        usable_urls.append(url)
        total += len(cleaned)

    return "\n\n".join(chunks), usable_urls


if __name__ == "__main__":
    # Smoke test: crawl one domain and report the compression achieved.
    import asyncio

    from crawler import crawl_domain

    async def _demo() -> None:
        pages = await crawl_domain("postman.com")
        raw_bytes = sum(len(h) for h in pages.values())
        context, urls = build_context(pages)
        print(f"raw HTML:    {raw_bytes:,} chars across {len(pages)} pages")
        print(f"clean text:  {len(context):,} chars across {len(urls)} usable pages")
        print(f"reduction:   {100 * (1 - len(context) / max(raw_bytes, 1)):.1f}%")
        print(f"emails:      {find_emails(pages)}")
        print(f"\n--- first 600 chars ---\n{context[:600]}")

    asyncio.run(_demo())
