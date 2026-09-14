"""CrawlSense — autonomous lead enrichment agent.

Orchestrates crawl -> clean -> extract across a list of domains and writes
``output.json``.

Resilience contract: a failure anywhere in a single domain's pipeline produces a
zero-confidence record carrying an ``error`` field, and the run continues to the
next domain. The process exits non-zero only if it could not write its output at
all.

Usage:
    python main.py
    python main.py --domains postman.com supabase.com vapi.ai
    python main.py --output results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# Load .env before importing modules that read GROQ_API_KEY at call time.
load_dotenv()

from cleaner import build_context, find_emails  # noqa: E402
from enrichment import enrich_linkedin_urls  # noqa: E402
from crawler import crawl_domain  # noqa: E402
from extractor import extract_intel  # noqa: E402
from schema import CompanyIntel, compute_confidence  # noqa: E402

DEFAULT_DOMAINS: List[str] = ["postman.com", "supabase.com", "vapi.ai"]


async def process_domain(domain: str) -> CompanyIntel:
    """Run the full pipeline for one domain, converting any failure into a record.

    Args:
        domain: Bare domain, e.g. ``"supabase.com"``.

    Returns:
        A populated ``CompanyIntel`` on success, or a zero-confidence record with
        ``error`` set on any failure. This function does not raise.
    """
    print(f"\n{'=' * 60}\nProcessing {domain}\n{'=' * 60}")
    started = time.monotonic()

    try:
        html_pages = await crawl_domain(domain)
        if not html_pages:
            raise RuntimeError(
                "No pages could be crawled (site unreachable, blocked, or timed out)."
            )
        print(f"  crawled {len(html_pages)} page(s)")

        context, usable_urls = build_context(html_pages)
        if not context.strip():
            raise RuntimeError(
                "Pages were fetched but no readable text could be extracted."
            )
        print(f"  cleaned to {len(context):,} chars from {len(usable_urls)} usable page(s)")

        emails = find_emails(html_pages)
        extracted, tokens, cost = extract_intel(domain, context, known_emails=emails)

        # Optional bonus stage; a no-op unless a search provider key is configured.
        extracted.key_leadership = enrich_linkedin_urls(extracted.key_leadership, domain)

        record = CompanyIntel.from_extraction(
            domain=domain,
            extracted=extracted,
            pages_crawled=usable_urls,
            confidence=compute_confidence(extracted, usable_urls),
        )
        record.tokens_used = tokens
        record.estimated_cost_usd = cost

        elapsed = time.monotonic() - started
        print(
            f"  done in {elapsed:.1f}s | confidence={record.data_confidence_score:.2f} "
            f"| tokens={tokens} | leadership={len(record.key_leadership)}"
        )
        return record

    except Exception as exc:
        elapsed = time.monotonic() - started
        print(f"  [error] {domain} failed after {elapsed:.1f}s: {type(exc).__name__}: {exc}")
        return CompanyIntel.failed(domain, f"{type(exc).__name__}: {exc}")


async def run(domains: List[str], output_path: str) -> List[CompanyIntel]:
    """Process every domain sequentially and write the results to ``output_path``.

    Domains are processed one at a time rather than concurrently: Groq's free
    tier is rate-limited, and a burst of parallel calls trades a small speedup
    for a high chance of 429s on the calls that matter.
    """
    records: List[CompanyIntel] = []
    for domain in domains:
        records.append(await process_domain(domain))

    write_output(records, output_path)
    return records


def write_output(records: List[CompanyIntel], output_path: str) -> bool:
    """Serialize records to ``output_path``, reporting I/O failure legibly.

    Writes to a temporary file in the same directory and then atomically renames
    it, so an interrupted or out-of-space write cannot leave a half-written
    output.json behind masquerading as a good result.

    Returns:
        ``True`` on success, ``False`` if the file could not be written. A failure
        here is reported clearly rather than as a traceback — the most likely
        cause is a full disk, which is worth saying out loud.
    """
    payload = json.dumps(
        [r.model_dump() for r in records], indent=2, ensure_ascii=False
    )
    target = Path(output_path)
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(target)
        return True
    except OSError as exc:
        # Cleanup must never raise from inside the handler — a bad path makes
        # unlink() itself fail, which would defeat the point of catching at all.
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        print(
            f"\n[FATAL] Could not write {output_path}: {exc}\n"
            f"        The crawl and extraction succeeded; only the file write failed.\n"
            f"        If this is ENOSPC, free disk space and re-run.",
            file=sys.stderr,
        )
        return False


def summarize(records: List[CompanyIntel], output_path: str) -> None:
    """Print a short run report so a reader can judge quality without opening JSON."""
    succeeded = [r for r in records if r.error is None]
    total_tokens = sum(r.tokens_used or 0 for r in records)
    total_cost = sum(r.estimated_cost_usd or 0.0 for r in records)

    print(f"\n{'=' * 60}\nRun summary\n{'=' * 60}")
    for record in records:
        status = "ok " if record.error is None else "ERR"
        print(
            f"  [{status}] {record.domain:<18} confidence={record.data_confidence_score:.2f} "
            f"pages={len(record.pages_crawled)} leadership={len(record.key_leadership)}"
        )
    print(f"\n  {len(succeeded)}/{len(records)} domains succeeded")
    print(f"  {total_tokens:,} tokens | ~${total_cost:.6f}")
    print(f"  Wrote {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CrawlSense — autonomous lead enrichment agent"
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=DEFAULT_DOMAINS,
        help="Company domains to enrich (bare domains, no scheme).",
    )
    parser.add_argument(
        "--output",
        default="output.json",
        help="Path to write the JSON results (default: output.json).",
    )
    args = parser.parse_args()

    records = asyncio.run(run(args.domains, args.output))
    summarize(records, args.output)
    # Non-zero only if output could not be persisted at all; individual domain
    # failures are a normal, recorded outcome rather than a run failure.
    return 0 if Path(args.output).exists() else 1


if __name__ == "__main__":
    sys.exit(main())
