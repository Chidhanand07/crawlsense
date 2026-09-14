"""Structured output contract for CrawlSense.

Two-tier design:

* ``ExtractedIntel`` is the *only* model handed to the LLM. It contains just the
  fields the model can legitimately infer from page text.
* ``CompanyIntel`` is the full output record. It wraps ``ExtractedIntel`` with
  fields the orchestrator knows for a fact (which pages were crawled, token
  usage, errors) plus a confidence score computed from evidence.

Keeping them separate matters: if the LLM were handed the full model it would be
asked to invent ``pages_crawled`` and ``tokens_used``, wasting tokens and giving
it license to fabricate. The model only ever sees what it is qualified to answer.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Person(BaseModel):
    """A named individual in company leadership."""

    name: str = Field(..., description="Full name exactly as written on the site.")
    role: Optional[str] = Field(
        None, description="Job title, e.g. 'CEO' or 'Head of Engineering'."
    )
    linkedin_url: Optional[str] = Field(
        None,
        description="LinkedIn profile URL, only if it appears in the text. Never guess it.",
    )


class ExtractedIntel(BaseModel):
    """The LLM-facing schema — fields derivable from website text alone."""

    company_overview: str = Field(
        "",
        description=(
            "Concise two-sentence summary of what the company does. "
            "Empty string if the text does not say."
        ),
    )
    target_audience_icp: str = Field(
        "",
        description=(
            "The ideal customer profile: who the product is built for. "
            "Empty string if the text does not say."
        ),
    )
    contact_points: List[str] = Field(
        default_factory=list,
        description=(
            "Generic public email addresses found verbatim in the text "
            "(contact@, sales@, support@, hello@). Do not construct addresses."
        ),
    )
    key_leadership: List[Person] = Field(
        default_factory=list,
        description="Named leadership figures explicitly mentioned in the text.",
    )


class CompanyIntel(BaseModel):
    """The complete per-domain output record written to output.json."""

    domain: str
    company_overview: str = ""
    target_audience_icp: str = ""
    contact_points: List[str] = Field(default_factory=list)
    key_leadership: List[Person] = Field(default_factory=list)
    data_confidence_score: float = Field(..., ge=0.0, le=1.0)
    pages_crawled: List[str] = Field(default_factory=list)
    tokens_used: Optional[int] = None
    estimated_cost_usd: Optional[float] = None
    error: Optional[str] = None

    @classmethod
    def from_extraction(
        cls,
        domain: str,
        extracted: ExtractedIntel,
        pages_crawled: List[str],
        confidence: float,
    ) -> "CompanyIntel":
        """Compose a full record from an LLM extraction plus orchestrator-known facts."""
        return cls(
            domain=domain,
            **extracted.model_dump(),
            data_confidence_score=confidence,
            pages_crawled=pages_crawled,
        )

    @classmethod
    def failed(cls, domain: str, error: str, pages_crawled: Optional[List[str]] = None) -> "CompanyIntel":
        """Build the zero-confidence record used when a domain cannot be processed."""
        return cls(
            domain=domain,
            data_confidence_score=0.0,
            pages_crawled=pages_crawled or [],
            error=error,
        )


def compute_confidence(extracted: ExtractedIntel, pages_crawled: List[str]) -> float:
    """Score 0.0–1.0 for how much real data was actually recovered for this domain.

    This score is computed in code, deliberately *not* asked of the LLM — a model
    asked to rate its own output will happily return 0.9 for a near-empty record.
    Here the score is a function of observable evidence only.

    Args:
        extracted: What the LLM returned for this domain.
        pages_crawled: URLs that were successfully fetched and cleaned.

    Returns:
        A float in [0.0, 1.0]. Must be clamped — the caller trusts the range.

    The formula is weighted field coverage, scaled by crawl breadth:

    * Fields are weighted by how findable they actually are. A company overview
      is on virtually every homepage; a LinkedIn URL is rare, so it is a small
      bonus rather than a large penalty when absent.
    * Text fields must clear a minimum length — a three-word "overview" is not
      evidence, and counting it would inflate the score exactly as a
      self-reporting LLM would.
    * The breadth factor caps a single-page crawl below 1.0. If only the homepage
      loaded, leadership data was probably never reachable, so even a full-looking
      extraction should not claim maximum confidence.
    """
    score = 0.0

    if len(extracted.company_overview.strip()) >= 40:
        score += 0.30
    if len(extracted.target_audience_icp.strip()) >= 20:
        score += 0.25
    if extracted.contact_points:
        score += 0.15
    if extracted.key_leadership:
        score += 0.20
    if any(person.linkedin_url for person in extracted.key_leadership):
        score += 0.10

    # 1 page -> 0.70, 2 -> 0.80, 3 -> 0.90, 4+ -> 1.00. No pages -> no confidence.
    breadth = min(1.0, 0.60 + 0.10 * len(pages_crawled)) if pages_crawled else 0.0

    return round(max(0.0, min(1.0, score * breadth)), 2)
