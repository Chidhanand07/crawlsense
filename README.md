<div align="center">

# CrawlSense

**An autonomous lead-enrichment agent.**

Give it company domains. It crawls their public websites with a headless browser,
strips the pages to clean text, and extracts structured company intelligence into a
strict schema — reporting honestly on what it could and could not find.

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)]()
[![Tests](https://img.shields.io/badge/tests-61%20passing-brightgreen)]()
[![LLM](https://img.shields.io/badge/LLM-Groq-orange)]()

[Setup](#setup) · [Architecture](#architecture) · [Design Decisions](#design-decisions) · [Testing](#testing) · [Troubleshooting](#troubleshooting)

</div>

---

## Setup

Requires **Python 3.11+** and about 700 MB of free disk (Chromium is ~450 MB unpacked).

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
playwright install chromium       # required — pip installs the bindings, not the browser

cp .env.example .env              # then add your Groq key to .env
python main.py
```

Get a free Groq API key at **[console.groq.com](https://console.groq.com)** → API Keys →
Create Key. No card required.

> **Put the key in `.env`, not `.env.example`.** `.env` is gitignored; `.env.example` is
> the committed template documenting which variables exist. The filenames differ by eight
> characters, and getting it wrong commits your key.

Verify the setup:

```bash
python -c "
import os; from dotenv import load_dotenv; load_dotenv()
from groq import Groq
print('key OK —', len(Groq(api_key=os.environ['GROQ_API_KEY']).models.list().data), 'models')
"
```

### Running it

```bash
python main.py                                              # the three default domains
python main.py --domains postman.com supabase.com vapi.ai   # explicit list
python main.py --domains stripe.com --output results.json   # custom target and output
```

Roughly 60 seconds and ~12,000 tokens for three domains. Results go to `output.json`.

### Optional: LinkedIn enrichment

Add **one** of these to `.env` to enable the search fallback for `key_leadership`:

```
TAVILY_API_KEY=tvly-...      # tavily.com   — free tier, 1000 searches/month
SERPAPI_API_KEY=...          # serpapi.com  — free tier, 100 searches/month
```

With neither set the stage is skipped entirely — it is a bonus, not a dependency.

---

## What it produces

```jsonc
{
  "domain": "postman.com",
  "company_overview": "Postman is an API platform for building and using APIs. Postman simplifies each step of the API lifecycle and streamlines collaboration so you can create better APIs—faster.",
  "target_audience_icp": "Engineers and development teams in organizations of any size that need to design, test, manage, and distribute APIs.",
  "contact_points": ["info-jp@postman.com", "info@postman.com"],
  "key_leadership": [
    { "name": "Abhinav Asthana", "role": "CEO and co-founder", "linkedin_url": null },
    { "name": "Ankit Sobti",     "role": "Co-founder",         "linkedin_url": null },
    { "name": "Abhijit Kane",    "role": "Co-founder",         "linkedin_url": null }
  ],
  "data_confidence_score": 0.9,
  "pages_crawled": ["https://postman.com", "https://postman.com/company/about-postman", "…"],
  "tokens_used": 4364,
  "estimated_cost_usd": 0.002575,
  "error": null
}
```

### A real run

```
  [ok ] postman.com        confidence=0.90 pages=4 leadership=3
  [ok ] supabase.com       confidence=0.90 pages=6 leadership=1
  [ok ] vapi.ai            confidence=0.50 pages=3 leadership=0

  3/3 domains succeeded
  12,413 tokens | ~$0.007324
```

`vapi.ai` scores **0.50** because no leadership or contact emails were recoverable from
its public pages. That is the point: the score reports what was actually found, rather
than being inflated to look good.

---

## Architecture

```
  domains
     │
     ▼
┌─────────────┐   rendered HTML   ┌─────────────┐   clean text   ┌──────────────┐
│  crawler.py │ ────────────────► │  cleaner.py │ ─────────────► │ extractor.py │
│  Playwright │    ~1,950,000 ch  │ trafilatura │     ~8,800 ch  │   Groq LLM   │
│ headless    │                   │   −99.5%    │                │ + instructor │
│ Chromium    │                   └─────────────┘                └──────────────┘
└─────────────┘                          │                              │
     │                                   │ regex: emails                │ ExtractedIntel
     │  homepage + ranked subpages       ▼                              ▼
     │                            ┌──────────────────────────────────────────┐
     └───────────────────────────►│                 main.py                  │
        pages_crawled             │  orchestration · per-domain try/except   │
                                  │  confidence scoring · atomic write       │
                                  └──────────────────┬───────────────────────┘
                                                     ▼
                                                output.json
```

| Stage | Module | Responsibility |
|---|---|---|
| **1. Crawl** | `crawler.py` | Headless Chromium fetches the homepage, then discovers and fetches up to 5 high-value subpages (`/about`, `/team`, `/company`, `/contact`, `/pricing`) ranked by match trustworthiness. |
| **2. Clean** | `cleaner.py` | `trafilatura` isolates main content, discarding markup, scripts and nav chrome. Per-page and total character caps bound prompt size. Emails scraped from raw HTML by regex. |
| **3. Extract** | `extractor.py` | Cleaned text → Groq via `instructor`, which enforces the `ExtractedIntel` schema and re-prompts on validation failure. Testimonial bylines filtered out afterward. |
| **4. Orchestrate** | `main.py` | Per-domain isolation, confidence scoring, run summary, atomic output write. |

---

## Design decisions

### A headless browser, not `requests`

All three test sites render their copy client-side. A plain HTTP GET returns an empty
app shell; Playwright returns the text a human would actually read.

### Raw HTML never reaches the LLM

Enforced *structurally*, not by convention: `build_context()` is the only path from
crawler output to extractor input, and it returns `trafilatura` output exclusively.

On `postman.com` this is a **99.5% reduction** — 1,954,421 characters of HTML become
8,812 characters of text. Roughly 490K tokens down to 2.2K, which is the difference
between "impossible and expensive" and one cheap call.

### Two schemas, so the model is only asked what it can know

`ExtractedIntel` is the **only** model the LLM ever sees — four fields, all inferable
from page text. `CompanyIntel` wraps it with facts the orchestrator already knows
(`pages_crawled`, `tokens_used`, `error`).

Handing the model the full schema would ask it to invent bookkeeping fields, waste
tokens generating them, and widen the surface for fabrication.

### Emails are found by regex, not by the LLM

Contact addresses live in `mailto:` hrefs and footers — exactly the boilerplate
`trafilatura` strips. A targeted regex over raw HTML finds them deterministically, so
`contact_points` is *evidence*, never a plausible guess.

> Postman's real address is `info@postman.com`. A model guessing from the domain would
> almost certainly have produced `contact@postman.com` — confidently, and wrongly.

### Confidence is computed, never self-reported

An LLM asked to rate its own output returns a high number nearly regardless of quality.
`compute_confidence()` scores weighted field coverage, scaled by crawl breadth:

| Signal | Weight | Rationale |
|---|---:|---|
| `company_overview` ≥ 40 chars | 0.30 | Nearly always findable; a length floor stops a three-word stub from earning it |
| `target_audience_icp` ≥ 20 chars | 0.25 | Usually inferable from homepage positioning |
| any `contact_points` | 0.15 | Regex-verified, so this weight is fully trustworthy |
| any `key_leadership` | 0.20 | Requires a team or about page |
| any `linkedin_url` | 0.10 | Rare — a bonus, not a penalty when missing |

The subtotal is multiplied by a breadth factor (1 page → 0.70, 4+ pages → 1.00): a
single-page crawl means leadership data was probably never *reachable*, so even a
full-looking extraction shouldn't claim certainty. A failed domain scores 0.00 and
carries an `error`.

### Subpage discovery ranks by match trustworthiness

A keyword in the URL path is strong evidence about a page's subject. A keyword in
the anchor text is a weak hint, and frequently incidental.

Treating them equally breaks badly. On `github.com`, three links reading *"Learn about
code security"*, *"Learn about secret protection"* and *"Learn about Dependabot"* all
matched on `about` — the top-priority keyword — and displaced `/pricing` and `/team`
from the crawl budget entirely:

| | Before | After |
|---|---|---|
| 1 | `/about` | `/about` |
| 2 | `/about/diversity` | `/about/diversity` |
| 3 | `/security/advanced-security/code-security` | **`/team`** |
| 4 | `/security/advanced-security/secret-protection` | **`/pricing`** |
| 5 | `/security/advanced-security/software-supply-chain` | `/security/advanced-security/code-security` |

`_score_link()` now ranks candidates on three axes, compared left to right:

1. **Path match vs text-only** — dominates, so incidental phrasing can never outrank a
   real page.
2. **Keyword priority** — `/about` beats `/pricing` however precisely each matched. This
   axis stays *above* specificity deliberately: making exactness dominant over-corrected
   and pushed `/company/about-postman` below `/pricing` on postman.com.
3. **Specificity** — exact path, then whole segment, then substring. A tiebreak only.

Text-only matches are demoted, not discarded: a weak hint still beats an empty slot in
the budget.

### Testimonial bylines are filtered out

This one came from watching the agent get it wrong.

Marketing pages quote customers in a rigid shape — a quoted endorsement, then the
speaker's name, then their title. The name/role adjacency is such a strong signal that
the model harvested these as company leadership, attributing **the wrong company's
staff**:

| Extracted as leadership | Reality |
|---|---|
| Jason Mitura, *VP of Software Development* (vapi.ai) | ❌ A customer, quoted: *"…100% of our inbound volume now runs through Vapi."* |
| Jakob Steinn, *Co-founder & Tech Lead* (supabase.com) | ❌ A customer, quoted: *"My biggest regret is not having gone with Supabase from the beginning."* |
| Tracy Lane, *General Counsel* (supabase.com) | ✅ Genuine — named as Supabase's Grievance Officer |

**Strengthening the prompt did not fix it.** So `filter_leadership()` handles it
deterministically: drop any name whose *every* mention directly follows a closing
quotation mark, and any name absent from the source text entirely. Requiring *all*
mentions to look like attributions means a real executive who also gives a quote is
still kept — which is exactly why Tracy Lane survives the filter.

---

### LinkedIn enrichment is optional and off by default

`enrichment.py` fills missing `linkedin_url` values by querying Tavily or SerpAPI.
Without a provider key in the environment it is a **no-op** — the core pipeline never
depends on a third-party service being configured, reachable, or in credit.

When enabled, a search hit is accepted only if every substantial part of the person's
name appears in the LinkedIn slug. Without that guard, a search for a little-known
executive cheerfully returns a *stranger's* profile — worse than leaving the field null,
and corrosive to the confidence score's meaning. URLs already found in the site's own
text are never overwritten.

```bash
# .env — optional
TAVILY_API_KEY=tvly-...
```

## Error handling

Every domain is processed inside its own `try/except`. Any failure — DNS error, timeout,
bot challenge, HTTP 4xx/5xx, empty extraction, LLM error — produces a zero-confidence
record with a populated `error` field, and **the run continues to the next domain**.

Verified against an unresolvable domain:

```console
$ python main.py --domains this-domain-definitely-does-not-exist-9f3k.com postman.com
  [ERR] this-domain-…-9f3k.com  confidence=0.00 pages=0 leadership=0
  [ok ] postman.com             confidence=0.90 pages=4 leadership=3
  1/2 domains succeeded
$ echo $?
0
```

| Layer | Measure |
|---|---|
| Network | `https://` → `https://www.` → `http://` fallback chain before giving up |
| Page | One unreachable subpage never aborts the domain; HTTP ≥ 400 is skipped |
| Model | If the primary Groq model is rate-limited or retired, retries on a smaller one |
| Schema | `instructor` re-prompts with validation errors, up to twice |
| Output | Written to a temp file then atomically renamed — an interrupted or out-of-disk write cannot leave a truncated `output.json` |
| Process | Exits non-zero **only** if output could not be persisted; domain failures are a recorded outcome, not a run failure |

---

## Testing

```bash
pytest                  # 58 unit tests    · ~2s  · no network, no API key
pytest -m integration   #  3 live crawls   · ~35s · needs network
pytest -m ""            # all 61
```

Coverage spans confidence scoring (boundary clamping, monotonicity), link discovery
(offsite, fragments, exclusions, ranking, malformed HTML), content cleaning and email
extraction, the testimonial filter, LinkedIn enrichment (including the name-mismatch
rejection and search-provider failure), schema composition, output writing, and the
missing/invalid API-key paths.

Both real-world regressions are pinned as fixtures: the github.com ranking collapse and
the two testimonial bylines.

> The suite earned its keep immediately: it caught a bug where `write_output`'s own
> cleanup call could raise from inside its `except` block — defeating the purpose of
> catching the error at all.

---

## Models

Groq retires model IDs periodically. Current targets:

```python
PRIMARY_MODEL  = "openai/gpt-oss-120b"
FALLBACK_MODEL = "openai/gpt-oss-20b"
```

The `llama-3.x` IDs originally planned for this project are **no longer served by Groq**.
If extraction starts failing with model-not-found errors, see
[Troubleshooting](#troubleshooting).

---

## Performance

- **~15–22 seconds and ~4K tokens per domain**, end to end.
- Domains run **sequentially** — Groq's free tier is rate-limited, and parallel requests
  trade a small speedup for a high chance of 429s on the calls that matter.
- Images, fonts and media are aborted at the network layer during crawling: pure cost,
  no text.

---

## Project layout

```
CrawlSense/
├── README.md              # this file
├── requirements.txt       # pinned (pip freeze)
├── .env.example           # template — copy to .env and add your key
├── .gitignore
│
├── schema.py              # Person · ExtractedIntel · CompanyIntel · compute_confidence
├── crawler.py             # Playwright crawl + ranked subpage discovery
├── cleaner.py             # trafilatura extraction + email scraping
├── extractor.py           # Groq + instructor extraction + leadership filter
├── enrichment.py          # optional LinkedIn search fallback (off by default)
├── main.py                # orchestration · error handling · atomic output
│
├── test_crawlsense.py     # 61 tests
├── pytest.ini
└── output.json            # generated results
```

Each stage is runnable standalone for debugging:

```bash
python crawler.py     # crawl postman.com, print pages and byte counts
python cleaner.py     # crawl + clean, print the compression ratio achieved
```

---

## Troubleshooting

#### `Executable doesn't exist at .../chrome-mac/...`

The Chromium binary was never downloaded, or its cache was cleared:

```bash
playwright install chromium
```

#### `RuntimeError: GROQ_API_KEY is not set`

Either `.env` doesn't exist, or the key went into `.env.example` by mistake:

```bash
ls -la .env && cat .env      # must exist and hold a real gsk_... value
```

#### `model_not_found` / `decommissioned` errors

Groq retires model IDs periodically — the `llama-3.x` models this project originally
targeted are already gone. List what is currently live:

```bash
python -c "
import os; from dotenv import load_dotenv; load_dotenv()
from groq import Groq
[print(' ', m.id) for m in Groq(api_key=os.environ['GROQ_API_KEY']).models.list().data]
"
```

Then update `PRIMARY_MODEL` / `FALLBACK_MODEL` at the top of `extractor.py`.

#### `429 Too Many Requests`

Groq free-tier rate limit. Wait a minute or run fewer domains. The extractor already
falls back to a smaller model automatically, and domains are processed sequentially by
design — do not parallelise them.

#### `ENOSPC: no space left on device`

Playwright needs temp space to launch a browser. Free a few GB and retry:

```bash
pip cache purge
rm -rf ~/Library/Caches/ms-playwright   # then: playwright install chromium
```

#### A domain returns `confidence=0.00` with an `error`

Working as designed — unreachable, blocked, or nothing extractable. The run continues to
the next domain; the `error` field carries the cause.

#### Crawl succeeds but `key_leadership` is empty

Often correct. Many companies do not publish a team page, and CrawlSense deliberately
filters out customer-testimonial bylines that naive extractors misreport as leadership.

#### Wrong Python version

3.11–3.13 are supported. 3.10 and older will not run the code. On 3.14 several
dependencies in the `httpx`/`lxml` chain still lack stable wheels.

```bash
brew install python@3.11 && python3.11 -m venv venv     # macOS
```

---

## Known limitations

- **`linkedin_url` is usually `null`** unless the optional search fallback is configured.
  From page text alone the URL is rarely present.
- **Discovery is one level deep.** Only homepage links are considered; a team page buried
  two clicks down won't be found.
- **The testimonial filter keys on quotation marks.** A testimonial formatted without them
  would slip through.
- **No `robots.txt` parsing.** Fine for a handful of public marketing pages; it would need
  adding before running this at any real volume.
