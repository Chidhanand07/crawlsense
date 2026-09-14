"""Test suite for CrawlSense.

Covers the pure functions in isolation plus the failure paths that are hard to
trigger against live sites. Network-dependent tests are marked ``integration``
and skipped by default:

    pytest                      # fast unit tests only
    pytest -m integration       # live crawl tests (needs network)
    pytest -m ""                # everything
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from cleaner import build_context, clean_html, find_emails
from crawler import discover_links
from extractor import _is_testimonial_attribution, filter_leadership
from main import write_output
from schema import CompanyIntel, ExtractedIntel, Person, compute_confidence

# --------------------------------------------------------------------------
# compute_confidence
# --------------------------------------------------------------------------

FULL = ExtractedIntel(
    company_overview="A" * 60,
    target_audience_icp="B" * 30,
    contact_points=["contact@example.com"],
    key_leadership=[Person(name="Ada", role="CEO", linkedin_url="https://li/ada")],
)


def test_confidence_full_extraction_broad_crawl_is_max():
    assert compute_confidence(FULL, ["a", "b", "c", "d"]) == 1.0


def test_confidence_single_page_is_capped_below_max():
    """One page means leadership was probably never reachable — never claim 1.0."""
    assert compute_confidence(FULL, ["a"]) == 0.7


def test_confidence_empty_extraction_is_zero():
    assert compute_confidence(ExtractedIntel(), ["a", "b", "c", "d"]) == 0.0


def test_confidence_no_pages_is_zero_regardless_of_content():
    """Content with no crawled pages is incoherent; refuse to score it."""
    assert compute_confidence(FULL, []) == 0.0


def test_confidence_rejects_too_short_overview():
    """A three-word 'overview' is not evidence and must not earn its weight."""
    stub = ExtractedIntel(company_overview="We do things.", target_audience_icp="B" * 30)
    assert compute_confidence(stub, ["a", "b", "c", "d"]) == 0.25


def test_confidence_always_within_bounds():
    """The caller feeds this into a Pydantic field with ge=0.0, le=1.0."""
    for pages in ([], ["a"], ["a"] * 50):
        for extracted in (FULL, ExtractedIntel()):
            score = compute_confidence(extracted, pages)
            assert 0.0 <= score <= 1.0


def test_confidence_is_monotonic_in_crawl_breadth():
    scores = [compute_confidence(FULL, ["p"] * n) for n in range(1, 6)]
    assert scores == sorted(scores)


# --------------------------------------------------------------------------
# discover_links
# --------------------------------------------------------------------------

BASE = "https://example.com"

LINKS_HTML = """
<html><body>
  <a href="/about">About us</a>
  <a href="/company/team">Team</a>
  <a href="/pricing">Pricing</a>
  <a href="/blog/some-post">Blog post</a>
  <a href="/company/careers">Careers</a>
  <a href="https://twitter.com/example/about">Twitter</a>
  <a href="/about#section">About anchor</a>
  <a href="mailto:hi@example.com">Email</a>
  <a href="/">Home</a>
  <a href="/random">Random</a>
</body></html>
"""


def test_discover_links_finds_keyword_pages():
    found = discover_links(LINKS_HTML, BASE)
    assert "https://example.com/about" in found
    assert "https://example.com/company/team" in found


def test_discover_links_excludes_offsite():
    assert not any("twitter.com" in u for u in discover_links(LINKS_HTML, BASE))


def test_discover_links_excludes_blog_and_careers():
    """Regression: '/company/careers' has no trailing slash and once slipped through."""
    found = discover_links(LINKS_HTML, BASE)
    assert not any("careers" in u for u in found)
    assert not any("/blog/" in u for u in found)


def test_discover_links_strips_fragments_and_dedupes():
    found = discover_links(LINKS_HTML, BASE)
    assert not any("#" in u for u in found)
    assert len(found) == len(set(found))


def test_discover_links_skips_non_http_schemes():
    assert not any(u.startswith("mailto:") for u in discover_links(LINKS_HTML, BASE))


def test_discover_links_excludes_homepage_itself():
    assert BASE not in discover_links(LINKS_HTML, BASE)


def test_discover_links_ranks_about_above_pricing():
    """Priority ordering must survive the limit — /about beats /pricing."""
    found = discover_links(LINKS_HTML, BASE, limit=1)
    assert found == ["https://example.com/about"]


def test_discover_links_respects_limit():
    assert len(discover_links(LINKS_HTML, BASE, limit=2)) <= 2


def test_discover_links_handles_empty_and_malformed_html():
    assert discover_links("", BASE) == []
    assert discover_links("<html><body><a>no href</a>", BASE) == []
    assert discover_links("<<<not html>>>", BASE) == []


# --------------------------------------------------------------------------
# cleaner
# --------------------------------------------------------------------------

ARTICLE_HTML = """
<html><head><style>.x{color:red}</style><script>var a=1;</script></head>
<body><nav>Home About Contact</nav>
<article><h1>Acme Corp</h1>
<p>Acme Corp builds developer tooling for engineering teams at scale.
We help companies ship software faster with reliable infrastructure and observability.</p>
<p>Our platform is trusted by thousands of engineering organisations worldwide.</p>
</article><footer>(c) Acme</footer></body></html>
"""


def test_clean_html_strips_scripts_and_styles():
    out = clean_html(ARTICLE_HTML)
    assert "var a=1" not in out
    assert "color:red" not in out
    assert "Acme Corp builds developer tooling" in out


def test_clean_html_handles_empty_and_junk_input():
    assert clean_html("") == ""
    assert clean_html("<html></html>") == ""
    assert clean_html("not html at all") == ""


def test_clean_html_respects_char_cap():
    big = "<html><body><article>" + "<p>word word word.</p>" * 5000 + "</article></body></html>"
    assert len(clean_html(big, max_chars=500)) <= 500


def test_build_context_labels_sources_and_reports_usable_urls():
    context, urls = build_context({"https://a.com": ARTICLE_HTML})
    assert "--- SOURCE: https://a.com ---" in context
    assert urls == ["https://a.com"]


def test_build_context_excludes_pages_with_no_extractable_text():
    """A page that renders but yields nothing is not crawl evidence."""
    context, urls = build_context({"https://a.com": ARTICLE_HTML, "https://b.com": "<html></html>"})
    assert urls == ["https://a.com"]
    assert "b.com" not in context


def test_build_context_on_empty_input():
    assert build_context({}) == ("", [])


def test_find_emails_matches_generic_mailboxes_only():
    html = """<a href="mailto:contact@acme.com">x</a> sales@acme.com
              personal.name@acme.com hero@acme.com support@acme.com"""
    found = find_emails({"u": html})
    assert "contact@acme.com" in found
    assert "sales@acme.com" in found
    assert "support@acme.com" in found
    assert "personal.name@acme.com" not in found


def test_find_emails_dedupes_case_insensitively():
    found = find_emails({"a": "Contact@Acme.com", "b": "contact@acme.com"})
    assert found == ["contact@acme.com"]


def test_find_emails_on_empty_input():
    assert find_emails({}) == []
    assert find_emails({"u": ""}) == []


# --------------------------------------------------------------------------
# testimonial filter
# --------------------------------------------------------------------------

VAPI_CTX = 'CSAT scores have improved."\nJason Mitura\nVP of Software Development'
SUPABASE_CTX = 'gone with Supabase from the beginning."\nJakob Steinn Co-founder & Tech Lead'
LEGAL_CTX = "Grievance Officer\nAttn: Tracy Lane\nTracy Lane, General Counsel of Supabase, Inc."


@pytest.mark.parametrize(
    "name,context,expected",
    [
        ("Jason Mitura", VAPI_CTX, True),       # real regression from vapi.ai
        ("Jakob Steinn", SUPABASE_CTX, True),   # real regression from supabase.com
        ("Tracy Lane", LEGAL_CTX, False),       # genuine executive, must survive
        ("Never Mentioned", LEGAL_CTX, True),   # hallucinated name
    ],
)
def test_testimonial_detection(name, context, expected):
    assert _is_testimonial_attribution(name, context) is expected


def test_testimonial_detection_handles_curly_quotes():
    assert _is_testimonial_attribution("Jane Doe", "great product”\nJane Doe\nCTO") is True


def test_executive_quoted_once_but_named_elsewhere_is_kept():
    """A real executive who also gives a quote must not be filtered out."""
    context = 'we built this."\nAda Lovelace\nCEO ... Our CEO Ada Lovelace founded the company.'
    assert _is_testimonial_attribution("Ada Lovelace", context) is False


def test_filter_leadership_removes_only_attributions():
    people = [Person(name="Jason Mitura"), Person(name="Tracy Lane")]
    kept = filter_leadership(people, VAPI_CTX + "\n" + LEGAL_CTX)
    assert [p.name for p in kept] == ["Tracy Lane"]


def test_filter_leadership_on_empty_list():
    assert filter_leadership([], "anything") == []


# --------------------------------------------------------------------------
# schema composition
# --------------------------------------------------------------------------

def test_failed_record_is_zero_confidence_and_carries_error():
    record = CompanyIntel.failed("broken.com", "TimeoutError: took too long")
    assert record.data_confidence_score == 0.0
    assert "TimeoutError" in record.error
    assert record.company_overview == ""
    assert record.key_leadership == []


def test_from_extraction_merges_llm_and_orchestrator_fields():
    record = CompanyIntel.from_extraction("acme.com", FULL, ["https://acme.com"], 0.8)
    assert record.domain == "acme.com"
    assert record.pages_crawled == ["https://acme.com"]
    assert record.data_confidence_score == 0.8
    assert record.company_overview == FULL.company_overview


def test_confidence_field_rejects_out_of_range():
    """The Pydantic bound is the contract compute_confidence must honour."""
    with pytest.raises(Exception):
        CompanyIntel(domain="x", data_confidence_score=1.5)
    with pytest.raises(Exception):
        CompanyIntel(domain="x", data_confidence_score=-0.1)


def test_extracted_intel_never_leaks_orchestrator_fields_to_the_llm():
    """The LLM-facing schema must not ask the model for bookkeeping it cannot know."""
    fields = set(ExtractedIntel.model_fields)
    assert not fields & {"pages_crawled", "tokens_used", "estimated_cost_usd", "error", "domain"}


# --------------------------------------------------------------------------
# output writing
# --------------------------------------------------------------------------

def test_write_output_produces_valid_json(tmp_path):
    target = tmp_path / "out.json"
    assert write_output([CompanyIntel.failed("a.com", "err")], str(target)) is True
    data = json.loads(target.read_text())
    assert data[0]["domain"] == "a.com"


def test_write_output_leaves_no_temp_file_behind(tmp_path):
    target = tmp_path / "out.json"
    write_output([CompanyIntel.failed("a.com", "err")], str(target))
    assert list(tmp_path.iterdir()) == [target]


def test_write_output_reports_failure_instead_of_raising(tmp_path):
    """A full disk or bad path must produce False, not a traceback."""
    unwritable = tmp_path / "no_such_dir" / "out.json"
    assert write_output([CompanyIntel.failed("a.com", "err")], str(unwritable)) is False


def test_write_output_does_not_clobber_on_failure(tmp_path):
    """Atomic write: a failed run must leave the previous good output intact."""
    target = tmp_path / "out.json"
    write_output([CompanyIntel.failed("good.com", "e")], str(target))
    before = target.read_text()
    target.chmod(0o444)
    try:
        write_output([CompanyIntel.failed("new.com", "e")], str(target / "sub" / "x.json"))
    finally:
        target.chmod(0o644)
    assert target.read_text() == before


# --------------------------------------------------------------------------
# extractor failure paths
# --------------------------------------------------------------------------

def test_missing_api_key_raises_actionable_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    from extractor import _build_client

    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        _build_client()


def test_invalid_api_key_surfaces_as_exception(monkeypatch):
    """The orchestrator relies on this raising so it can record an error record."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_invalid_key_for_testing")
    from extractor import extract_intel

    with pytest.raises(Exception):
        extract_intel("example.com", "Some cleaned website text about a company.")


# --------------------------------------------------------------------------
# integration (live network)
# --------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_crawl_unreachable_domain_returns_empty_not_raise():
    from crawler import crawl_domain

    assert await crawl_domain("this-domain-does-not-exist-9f3k2x.invalid") == {}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_crawl_real_site_returns_multiple_pages():
    from crawler import crawl_domain

    pages = await crawl_domain("postman.com")
    assert len(pages) >= 2
    assert all(html for html in pages.values())


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pipeline_compresses_html_by_over_ninety_percent():
    from crawler import crawl_domain

    pages = await crawl_domain("postman.com")
    raw = sum(len(h) for h in pages.values())
    context, _ = build_context(pages)
    assert len(context) < raw * 0.1


# --------------------------------------------------------------------------
# link ranking (regression: github.com /security pages displacing /pricing)
# --------------------------------------------------------------------------

GITHUB_LIKE_HTML = """
<html><body>
  <a href="/security/advanced-security/code-security">Learn about code security</a>
  <a href="/security/advanced-security/secret-protection">Learn about secret protection</a>
  <a href="/security/advanced-security/software-supply-chain">Learn about Dependabot</a>
  <a href="/pricing">Plans</a>
  <a href="/team">Enterprise</a>
  <a href="/about">About</a>
</body></html>
"""


def test_text_only_matches_never_displace_path_matches():
    """Regression: 'Learn about X' links crowded out /pricing on github.com.

    Three text-only matches compete with three path matches for a budget of
    three. Every path match must win; the text-only links rank below them.
    """
    found = discover_links(GITHUB_LIKE_HTML, BASE, limit=3)
    assert set(found) == {
        "https://example.com/about",
        "https://example.com/team",
        "https://example.com/pricing",
    }
    assert not any("/security/" in u for u in found)


def test_text_only_matches_rank_below_every_path_match():
    """They are demoted, not banned — order is what matters."""
    found = discover_links(GITHUB_LIKE_HTML, BASE, limit=6)
    first_text_only = next(i for i, u in enumerate(found) if "/security/" in u)
    last_path_match = max(i for i, u in enumerate(found) if "/security/" not in u)
    assert first_text_only > last_path_match


def test_path_match_ordering_follows_keyword_priority():
    found = discover_links(GITHUB_LIKE_HTML, BASE, limit=3)
    assert found == [
        "https://example.com/about",
        "https://example.com/team",
        "https://example.com/pricing",
    ]


def test_about_subpage_outranks_exact_pricing_match():
    """An /about page beats /pricing for lead enrichment however it matched."""
    html = '<a href="/company/about-postman">About</a><a href="/pricing">Pricing</a>'
    assert discover_links(html, BASE, limit=1) == ["https://example.com/company/about-postman"]


def test_exact_path_outranks_partial_at_equal_priority():
    html = '<a href="/about">About</a><a href="/all-about-us-and-more">About us</a>'
    assert discover_links(html, BASE, limit=1) == ["https://example.com/about"]


def test_shallower_path_wins_at_equal_priority_and_specificity():
    html = '<a href="/about/x/y">About deep</a><a href="/about">About</a>'
    assert discover_links(html, BASE)[0] == "https://example.com/about"


def test_text_only_match_still_included_when_budget_allows():
    """Demoted, not discarded — a weak hint is better than an empty slot."""
    html = '<a href="/pricing">Plans</a><a href="/whatever">Learn about us</a>'
    found = discover_links(html, BASE, limit=5)
    assert found == ["https://example.com/pricing", "https://example.com/whatever"]


# --------------------------------------------------------------------------
# LinkedIn enrichment (optional bonus stage)
# --------------------------------------------------------------------------

def test_enrichment_is_noop_without_provider_key(monkeypatch):
    """The core pipeline must never depend on the bonus being configured."""
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    from enrichment import enrich_linkedin_urls, find_linkedin_url

    people = [Person(name="Ada Lovelace")]
    assert enrich_linkedin_urls(people, "acme.com")[0].linkedin_url is None
    assert find_linkedin_url("Ada Lovelace", "acme.com") is None


def test_name_match_accepts_matching_slug():
    from enrichment import _name_matches_profile

    assert _name_matches_profile("Ada Lovelace", "https://linkedin.com/in/ada-lovelace")
    assert _name_matches_profile("Ada Lovelace", "https://www.linkedin.com/in/adalovelace1")


def test_name_match_rejects_stranger_profile():
    """The guard that stops a search from attaching the wrong person's profile."""
    from enrichment import _name_matches_profile

    assert not _name_matches_profile("Ada Lovelace", "https://linkedin.com/in/grace-hopper")
    assert not _name_matches_profile("Ada Lovelace", "https://linkedin.com/in/ada-smith")


def test_linkedin_regex_rejects_company_pages_and_posts():
    from enrichment import LINKEDIN_PROFILE_RE

    assert not LINKEDIN_PROFILE_RE.search("https://linkedin.com/company/acme")
    assert not LINKEDIN_PROFILE_RE.search("https://linkedin.com/posts/abc")
    assert LINKEDIN_PROFILE_RE.search("https://uk.linkedin.com/in/ada-lovelace")


def test_enrichment_accepts_validated_search_hit(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    import enrichment

    monkeypatch.setattr(
        enrichment, "_search", lambda q: ["https://www.linkedin.com/in/ada-lovelace"]
    )
    people = enrichment.enrich_linkedin_urls([Person(name="Ada Lovelace")], "acme.com")
    assert people[0].linkedin_url == "https://www.linkedin.com/in/ada-lovelace"


def test_enrichment_rejects_mismatched_search_hit(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    import enrichment

    monkeypatch.setattr(
        enrichment, "_search", lambda q: ["https://www.linkedin.com/in/grace-hopper"]
    )
    people = enrichment.enrich_linkedin_urls([Person(name="Ada Lovelace")], "acme.com")
    assert people[0].linkedin_url is None


def test_enrichment_preserves_url_already_found_on_site(monkeypatch):
    """Text on the company's own site outranks a search hit."""
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    import enrichment

    monkeypatch.setattr(enrichment, "_search", lambda q: ["https://linkedin.com/in/ada-lovelace"])
    existing = Person(name="Ada Lovelace", linkedin_url="https://linkedin.com/in/from-the-site")
    people = enrichment.enrich_linkedin_urls([existing], "acme.com")
    assert people[0].linkedin_url == "https://linkedin.com/in/from-the-site"


def test_enrichment_survives_search_provider_failure(monkeypatch):
    """A dead search API must not take down the domain."""
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    import enrichment

    def boom(query):
        raise ConnectionError("search API unreachable")

    monkeypatch.setattr(enrichment, "_search_tavily", boom)
    people = enrichment.enrich_linkedin_urls([Person(name="Ada Lovelace")], "acme.com")
    assert people[0].linkedin_url is None
