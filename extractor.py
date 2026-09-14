"""Stage 3 — structured extraction via Groq + instructor.

``instructor`` patches the Groq client so that a Pydantic model can be passed as
``response_model``. It converts the model into a JSON schema the LLM must satisfy
and, crucially, re-prompts with the validation errors when the response does not
parse. That retry loop is what makes structured output *reliable* rather than
merely likely.
"""

from __future__ import annotations

import os
import re
from typing import List, Optional, Tuple

import instructor
from groq import Groq

from schema import ExtractedIntel, Person

# Verify current names at https://console.groq.com/docs/models — Groq deprecates
# model IDs periodically (the llama-3.x IDs this project originally targeted have
# since been retired). The fallback is used when the primary is rate-limited or
# decommissioned; run `python extractor.py --list-models` to see what is live.
PRIMARY_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODEL = "openai/gpt-oss-20b"

# Groq free-tier pricing is subject to change; this is a rough blended estimate
# used only for the optional cost-tracking bonus field.
USD_PER_TOKEN = 0.00000059

SYSTEM_PROMPT = """You are a precise research analyst extracting structured company \
intelligence from website text.

Rules you must follow:
1. Use ONLY information present in the provided text. Never use outside knowledge \
about the company, even if you recognise it.
2. If a field is not supported by the text, leave it empty ("" or []). An empty \
field is correct; an invented one is a failure.
3. For key_leadership, include a person ONLY if the text shows they are a founder, \
executive, or senior leader OF THE COMPANY AT THIS DOMAIN. Marketing pages are full \
of names that are NOT this company's leadership — exclude all of them:
   - Authors of customer testimonials and quotes (they work at a *customer* company)
   - Names in case studies, customer stories, or logos-and-quotes sections
   - Investors, advisors, board members of other firms, and blog post authors
   If a name appears next to a quote praising this company, that person almost \
certainly works elsewhere — leave them out. An empty key_leadership list is far \
better than a list of the wrong company's staff.
   Only include a linkedin_url if that exact URL appears in the text. Never \
construct a LinkedIn URL from a name.
4. For contact_points, only include email addresses written verbatim in the text. \
Never assemble an address from a domain name.
5. company_overview must be exactly two sentences describing what the company does.
6. target_audience_icp describes who the product is built for — the customer \
profile, not a list of features."""

USER_PROMPT_TEMPLATE = """Domain: {domain}

Extract structured company intelligence from the website content below.

WEBSITE CONTENT:
{context}"""


# Testimonial blocks on marketing pages follow a rigid shape: a quoted endorsement,
# then the speaker's name, then their title. The speaker works at a *customer*, so
# harvesting them as leadership attributes the wrong company's staff. Prompting
# alone does not reliably suppress this — the name/role adjacency is too strong a
# signal — so it is filtered deterministically after extraction.
_CLOSING_QUOTE_RE = re.compile(r"[\"\u201d\u2019\u00bb]\s*[\u2014\-]?\s*$")


def _is_testimonial_attribution(name: str, context: str) -> bool:
    """True if every mention of ``name`` directly follows a closing quotation mark.

    Requiring *all* occurrences to look like attributions means a genuine
    executive who also happens to be quoted once is still kept.
    """
    occurrences = list(re.finditer(re.escape(name), context))
    if not occurrences:
        # Name is not in the source text at all: the model invented it.
        return True
    return all(
        _CLOSING_QUOTE_RE.search(context[max(0, m.start() - 20) : m.start()])
        for m in occurrences
    )


def filter_leadership(people: List[Person], context: str) -> List[Person]:
    """Drop names that are quote attributions or absent from the source text."""
    kept: List[Person] = []
    for person in people:
        if _is_testimonial_attribution(person.name, context):
            print(f"  [filter] dropped '{person.name}' (testimonial or not in source)")
            continue
        kept.append(person)
    return kept


def _build_client() -> instructor.Instructor:
    """Create the instructor-patched Groq client.

    Raises:
        RuntimeError: If ``GROQ_API_KEY`` is absent, with an actionable message
            rather than a bare ``KeyError`` from ``os.environ``.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key "
            "from https://console.groq.com (API Keys -> Create Key)."
        )
    return instructor.from_groq(Groq(api_key=api_key), mode=instructor.Mode.JSON)


def extract_intel(
    domain: str,
    context: str,
    known_emails: Optional[List[str]] = None,
) -> Tuple[ExtractedIntel, Optional[int], Optional[float]]:
    """Extract structured intelligence from cleaned website text.

    Args:
        domain: The domain being processed, included in the prompt for context.
        context: Cleaned text blob from ``cleaner.build_context``. Never raw HTML.
        known_emails: Emails found deterministically by regex. These are merged
            into the result so that real addresses are never lost to the model's
            discretion.

    Returns:
        ``(extracted, tokens_used, estimated_cost_usd)``. The token/cost values
        are ``None`` if the provider did not report usage.

    Raises:
        RuntimeError: If the API key is missing.
        Exception: Any provider error that survives the fallback attempt. The
            orchestrator is responsible for catching these per domain.
    """
    client = _build_client()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(domain=domain, context=context),
        },
    ]

    last_error: Optional[Exception] = None
    result: Optional[ExtractedIntel] = None
    completion = None

    for model in (PRIMARY_MODEL, FALLBACK_MODEL):
        try:
            result, completion = client.chat.completions.create_with_completion(
                model=model,
                response_model=ExtractedIntel,
                max_retries=2,
                temperature=0.1,  # near-deterministic: this is extraction, not writing
                messages=messages,
            )
            break
        except Exception as exc:
            last_error = exc
            print(f"  [warn] model {model} failed: {type(exc).__name__}: {exc}")

    if result is None:
        raise RuntimeError(f"All models failed for {domain}") from last_error

    # Discard leadership entries that are really customer-testimonial bylines.
    result.key_leadership = filter_leadership(result.key_leadership, context)

    # Union the regex-found emails with anything the model spotted in body text.
    if known_emails:
        merged = {email.lower() for email in result.contact_points} | set(known_emails)
        result.contact_points = sorted(merged)

    tokens_used: Optional[int] = None
    estimated_cost: Optional[float] = None
    usage = getattr(completion, "usage", None)
    if usage is not None and getattr(usage, "total_tokens", None):
        tokens_used = usage.total_tokens
        estimated_cost = round(tokens_used * USD_PER_TOKEN, 6)

    return result, tokens_used, estimated_cost
