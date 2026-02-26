"""
SSED Layer 2: Narrative Signal Layer

Two data sources, two analysis tiers:
  1. News sentiment — NewsAPI feed → GPT-4.1-nano bulk scoring
  2. SEC filing analysis — EDGAR API → o4-mini deep analysis of risk factor changes

All narrative signals are structured via Pydantic for Layer 3 consumption.
"""

import json
import os
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Optional

import requests
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

SEC_USER_AGENT = "SSED Research ssed@example.com"

# CIK lookup for common tickers
TICKER_TO_CIK = {
    "NVDA": "0001045810",
    "MSFT": "0000789019",
    "AAPL": "0000320193",
    "GOOGL": "0001652044",
    "META": "0001326801",
    "AMZN": "0001018724",
    "CHGG": "0001364954",
    "TSLA": "0001318605",
}


# ============================================================
# PYDANTIC MODELS
# ============================================================

class ArticleSentiment(BaseModel):
    """Sentiment analysis of a single news article."""
    title: str
    source: str
    published_at: str
    sentiment_score: float  # -1.0 to 1.0
    relevance: str  # how relevant to the event
    novel_themes: list[str]  # new concepts not seen historically


class NewsSentimentSignals(BaseModel):
    """Aggregated news sentiment for a topic/period."""
    query: str
    period: str
    article_count: int
    avg_sentiment: float
    sentiment_trend: str  # "improving", "deteriorating", "stable"
    novel_theme_counts: dict[str, int]  # theme -> frequency
    top_articles: list[ArticleSentiment]


class FilingDiff(BaseModel):
    """Analysis of changes between two SEC filings."""
    company: str
    ticker: str
    filing_type: str
    before_date: str
    after_date: str
    new_risk_factors: list[str]  # risks that appear in the newer filing but not the older
    removed_risk_factors: list[str]
    language_shift_summary: str
    sample_space_signal: bool  # does this suggest the investment universe changed?
    signal_reasoning: str


class NarrativeSignals(BaseModel):
    """Complete Layer 2 output — structured for Layer 3 consumption."""
    news: Optional[NewsSentimentSignals] = None
    filing_diff: Optional[FilingDiff] = None
    generated_at: str


# ============================================================
# NEWS SENTIMENT (NewsAPI + OpenAI)
# ============================================================

def fetch_news_articles(
    query: str,
    from_date: str,
    to_date: str,
    max_articles: int = 20,
) -> list[dict]:
    """
    Fetch news articles from NewsAPI.

    Free tier: 100 requests/day, historical up to 1 month back.
    For older dates, falls back to cached/mock data for demo purposes.
    """
    api_key = os.environ.get("NEWSAPI_KEY")

    if api_key:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "from": from_date,
            "to": to_date,
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": min(max_articles, 100),
            "apiKey": api_key,
        }
        resp = requests.get(url, params=params)
        if resp.status_code == 200:
            data = resp.json()
            return [
                {
                    "title": a.get("title", ""),
                    "source": a.get("source", {}).get("name", "Unknown"),
                    "published_at": a.get("publishedAt", ""),
                    "description": a.get("description", ""),
                    "content": (a.get("content") or "")[:500],
                }
                for a in data.get("articles", [])[:max_articles]
            ]

    # Fallback: curated demo articles for ChatGPT launch period
    # These are real headlines from the period — used when no API key
    return _get_demo_articles(query, from_date, to_date)


def _get_demo_articles(query: str, from_date: str, to_date: str) -> list[dict]:
    """Curated real headlines for demo when NewsAPI key is unavailable."""
    demo_articles = [
        {
            "title": "ChatGPT reaches 100 million users in two months, fastest-growing consumer app ever",
            "source": "Reuters",
            "published_at": "2023-02-01",
            "description": "OpenAI's ChatGPT chatbot has reached 100 million monthly active users.",
            "content": "ChatGPT set a record for fastest-growing user base of a consumer application.",
        },
        {
            "title": "Nvidia reports explosive AI demand, revenue forecasts shatter expectations",
            "source": "CNBC",
            "published_at": "2023-05-24",
            "description": "Nvidia forecasts Q2 revenue of $11B vs $7.2B expected, driven by AI chip demand.",
            "content": "CEO Jensen Huang says the company is seeing 'surging demand' for AI infrastructure.",
        },
        {
            "title": "Chegg shares collapse 50% after CEO admits ChatGPT is hurting growth",
            "source": "Bloomberg",
            "published_at": "2023-05-02",
            "description": "Education company Chegg saw significant student adoption of ChatGPT.",
            "content": "CEO Dan Rosensweig said ChatGPT impact on new customer growth was significant.",
        },
        {
            "title": "Microsoft invests $10 billion in OpenAI, integrating AI across products",
            "source": "Wall Street Journal",
            "published_at": "2023-01-23",
            "description": "Microsoft extends partnership with OpenAI with multibillion-dollar investment.",
            "content": "The investment cements Microsoft's position as the leading platform for AI development.",
        },
        {
            "title": "AI chip shortage drives data center spending to record levels",
            "source": "Financial Times",
            "published_at": "2023-07-15",
            "description": "Companies scramble for GPU capacity as AI workloads surge.",
            "content": "AI infrastructure has become a new category of mandatory enterprise spending.",
        },
        {
            "title": "Goldman Sachs: AI could drive a 7% increase in global GDP",
            "source": "Goldman Sachs Research",
            "published_at": "2023-03-26",
            "description": "Generative AI could raise global GDP by 7% over a 10-year period.",
            "content": "The report highlights AI infrastructure as a new investable asset class.",
        },
        {
            "title": "S&P 500 concentration hits 50-year high as Magnificent Seven dominate",
            "source": "Barron's",
            "published_at": "2023-11-20",
            "description": "The top 7 stocks now account for over 30% of S&P 500 market cap.",
            "content": "Market concentration driven by AI-related investment themes.",
        },
        {
            "title": "Education stocks face existential threat from generative AI",
            "source": "Seeking Alpha",
            "published_at": "2023-06-10",
            "description": "Multiple education companies warn of ChatGPT impact on business models.",
            "content": "Analysts downgrade education sector citing structural disruption from AI tutors.",
        },
    ]
    return demo_articles


def score_articles_with_openai(
    articles: list[dict],
    event_context: str,
    model: str = "gpt-4.1-nano",
) -> list[ArticleSentiment]:
    """
    Score article sentiment using OpenAI in a single batched call.

    All articles are sent in one prompt; the model returns a JSON array
    with one scored object per article.  This replaces the old per-article
    loop (N API calls → 1 API call), cutting latency and cost proportionally.
    Falls back to heuristic scoring if OpenAI is unavailable or the response
    cannot be parsed.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return _score_articles_heuristic(articles)

    from openai import OpenAI

    client = OpenAI()

    # Build a numbered list of articles to include in a single prompt.
    articles_text = ""
    for i, article in enumerate(articles, start=1):
        line = f"[{i}] Title: {article['title']}"
        if article.get("description"):
            line += f" | {article['description']}"
        if article.get("content"):
            line += f" | {article['content'][:200]}"
        articles_text += line + "\n"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a financial analyst scoring news sentiment. "
                        "You will receive a numbered list of articles. "
                        "Return ONLY a valid JSON array — one object per article in order — "
                        "where each object has exactly these fields:\n"
                        '  "sentiment_score": float from -1.0 (very negative) to 1.0 (very positive),\n'
                        '  "relevance": one of "high", "medium", or "low",\n'
                        '  "novel_themes": list of strings naming NEW market categories or asset classes\n'
                        "    (e.g. 'AI infrastructure', 'GPU shortage', 'AI-native companies').\n"
                        "    Only include genuinely novel themes — omit this list if none exist.\n"
                        "The array must contain exactly as many objects as there are articles.\n"
                        "Do not wrap the array in any outer object."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Event context: {event_context}\n\n"
                        f"Score all {len(articles)} articles below:\n\n"
                        f"{articles_text}"
                    ),
                },
            ],
            response_format={"type": "json_object"},
        )

        raw = json.loads(response.choices[0].message.content)

        # The model may wrap the array under a key (e.g. {"articles": [...]})
        # even when asked not to. Find the array regardless of wrapper.
        if isinstance(raw, list):
            results = raw
        else:
            # Look for the first list value in the response dict
            results = next((v for v in raw.values() if isinstance(v, list)), [])

        scored = []
        for i, article in enumerate(articles):
            if i < len(results) and isinstance(results[i], dict):
                r = results[i]
                scored.append(ArticleSentiment(
                    title=article["title"],
                    source=article.get("source", "Unknown"),
                    published_at=article.get("published_at", ""),
                    sentiment_score=float(r.get("sentiment_score", 0)),
                    relevance=r.get("relevance", "medium"),
                    novel_themes=r.get("novel_themes", []),
                ))
            else:
                # Article missing from response — fall back to heuristic for it
                scored.append(_score_single_heuristic(article))

        return scored

    except Exception:
        # Full batch failed — fall back to heuristic for all articles
        return _score_articles_heuristic(articles)


def _score_articles_heuristic(articles: list[dict]) -> list[ArticleSentiment]:
    """Heuristic sentiment scoring when OpenAI is unavailable."""
    return [_score_single_heuristic(a) for a in articles]


def _score_single_heuristic(article: dict) -> ArticleSentiment:
    """Simple keyword-based sentiment for demo fallback."""
    title = (article.get("title") or "").lower()
    desc = (article.get("description") or "").lower()
    text = title + " " + desc

    # Simple keyword scoring
    positive = ["surge", "record", "soar", "explosive", "boom", "invest", "growth", "raises"]
    negative = ["collapse", "crash", "plunge", "threat", "hurt", "warning", "decline", "fear"]

    pos_count = sum(1 for w in positive if w in text)
    neg_count = sum(1 for w in negative if w in text)
    total = pos_count + neg_count
    score = (pos_count - neg_count) / max(total, 1) * 0.8

    # Detect novel themes
    novel = []
    theme_keywords = {
        "AI infrastructure": ["ai infrastructure", "ai chip", "gpu", "data center"],
        "AI asset class": ["asset class", "new category", "ai investment"],
        "creative destruction": ["disruption", "obsolete", "existential", "threat"],
        "market concentration": ["concentration", "magnificent seven", "mag 7", "dominat"],
    }
    for theme, keywords in theme_keywords.items():
        if any(k in text for k in keywords):
            novel.append(theme)

    return ArticleSentiment(
        title=article.get("title", ""),
        source=article.get("source", "Unknown"),
        published_at=article.get("published_at", ""),
        sentiment_score=round(score, 2),
        relevance="high" if novel else "medium",
        novel_themes=novel,
    )


def compute_news_signals(
    query: str,
    from_date: str,
    to_date: str,
    event_context: str,
    model: str = "gpt-4.1-nano",
) -> NewsSentimentSignals:
    """Full news sentiment pipeline: fetch → score → aggregate."""
    print(f"[Layer 2] Fetching news for '{query}'...")
    articles = fetch_news_articles(query, from_date, to_date)
    print(f"  Found {len(articles)} articles")

    print(f"[Layer 2] Scoring sentiment (model: {model})...")
    scored = score_articles_with_openai(articles, event_context, model)

    # Aggregate
    scores = [a.sentiment_score for a in scored]
    avg_sent = sum(scores) / len(scores) if scores else 0.0

    # Trend: compare first half vs second half
    mid = len(scores) // 2
    if mid > 0:
        first_half = sum(scores[:mid]) / mid
        second_half = sum(scores[mid:]) / (len(scores) - mid)
        if second_half > first_half + 0.1:
            trend = "improving"
        elif second_half < first_half - 0.1:
            trend = "deteriorating"
        else:
            trend = "stable"
    else:
        trend = "stable"

    # Count novel themes
    theme_counts: dict[str, int] = {}
    for a in scored:
        for theme in a.novel_themes:
            theme_counts[theme] = theme_counts.get(theme, 0) + 1

    return NewsSentimentSignals(
        query=query,
        period=f"{from_date} to {to_date}",
        article_count=len(scored),
        avg_sentiment=round(avg_sent, 3),
        sentiment_trend=trend,
        novel_theme_counts=theme_counts,
        top_articles=sorted(scored, key=lambda a: abs(a.sentiment_score), reverse=True)[:5],
    )


# ============================================================
# SEC FILING ANALYSIS (Direct EDGAR API)
# ============================================================

def get_company_filings(
    ticker: str,
    form_type: str = "10-K",
) -> list[dict]:
    """
    Get filing metadata from SEC EDGAR submissions API.
    Free, no key needed — just requires User-Agent.
    """
    cik = TICKER_TO_CIK.get(ticker.upper())
    if not cik:
        raise ValueError(f"Unknown ticker: {ticker}. Add CIK to TICKER_TO_CIK dict.")

    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    headers = {"User-Agent": SEC_USER_AGENT}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()

    data = resp.json()
    recent = data["filings"]["recent"]

    filings = []
    for i in range(len(recent["form"])):
        if recent["form"][i] == form_type:
            accession = recent["accessionNumber"][i].replace("-", "")
            filings.append({
                "form": recent["form"][i],
                "filing_date": recent["filingDate"][i],
                "accession": recent["accessionNumber"][i],
                "primary_doc": recent["primaryDocument"][i],
                "url": (
                    f"https://www.sec.gov/Archives/edgar/data/"
                    f"{cik.lstrip('0')}/{accession}/{recent['primaryDocument'][i]}"
                ),
            })

    return filings


def fetch_filing_text(filing_url: str, max_chars: int = 50000) -> str:
    """
    Fetch the text content of a filing.
    Strips HTML, truncates to max_chars for LLM consumption.
    """
    headers = {"User-Agent": SEC_USER_AGENT}
    resp = requests.get(filing_url, headers=headers)
    resp.raise_for_status()

    text = resp.text
    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode common HTML entities
    text = text.replace("&#160;", " ").replace("&nbsp;", " ")
    text = text.replace("&#8220;", '"').replace("&#8221;", '"')
    text = text.replace("&#8217;", "'").replace("&amp;", "&")
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text[:max_chars]


def extract_risk_factors_section(filing_text: str) -> str:
    """
    Extract Item 1A (Risk Factors) from a 10-K filing text.
    Returns the risk factors section or a truncated portion.
    """
    # Common markers for Item 1A
    patterns = [
        r"item\s*1a[\.\s]*risk\s*factors",
        r"ITEM\s*1A[\.\s]*RISK\s*FACTORS",
    ]

    text_lower = filing_text.lower()
    start = -1
    for pat in patterns:
        match = re.search(pat, text_lower)
        if match:
            start = match.start()
            break

    if start == -1:
        # Couldn't find risk factors — return first chunk as fallback
        return filing_text[:5000]

    # Find the end (next Item marker)
    end_patterns = [r"item\s*1b", r"item\s*2[\.\s]", r"ITEM\s*1B", r"ITEM\s*2[\.\s]"]
    end = len(filing_text)
    for pat in end_patterns:
        match = re.search(pat, text_lower[start + 100:])
        if match:
            end = start + 100 + match.start()
            break

    section = filing_text[start:end]
    return section[:12000]  # Cap for LLM context


def analyze_filing_diff_with_openai(
    ticker: str,
    before_text: str,
    after_text: str,
    before_date: str,
    after_date: str,
    model: str = "o4-mini",
) -> FilingDiff:
    """
    Use OpenAI to compare two 10-K risk factor sections.
    Identifies NEW risk categories — the signal for sample space expansion.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return _analyze_filing_diff_heuristic(
            ticker, before_text, after_text, before_date, after_date
        )

    from openai import OpenAI

    client = OpenAI()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a financial analyst comparing SEC 10-K risk factor sections "
                    "across two years. Your job is to identify NEW risk categories that "
                    "appear in the later filing but were absent in the earlier one.\n\n"
                    "This is critical for detecting 'sample space expansion' — when the "
                    "investment universe itself changes. New risk categories that didn't "
                    "exist before (e.g., 'AI infrastructure dependency', 'generative AI "
                    "competition') signal that the sample space has expanded.\n\n"
                    "Return ONLY valid JSON with these fields:\n"
                    '{"new_risk_factors": ["risk1", ...], '
                    '"removed_risk_factors": ["risk1", ...], '
                    '"language_shift_summary": "brief summary of how language changed", '
                    '"sample_space_signal": true/false, '
                    '"signal_reasoning": "why this does/doesn\'t signal sample space expansion"}'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Company: {ticker}\n"
                    f"BEFORE ({before_date}) Risk Factors (excerpt):\n"
                    f"{before_text[:3000]}\n\n"
                    f"AFTER ({after_date}) Risk Factors (excerpt):\n"
                    f"{after_text[:3000]}\n\n"
                    "What new risk categories appeared? Does this signal sample space expansion?"
                ),
            },
        ],
        response_format={"type": "json_object"},
    )

    result = json.loads(response.choices[0].message.content)

    return FilingDiff(
        company=ticker,
        ticker=ticker,
        filing_type="10-K",
        before_date=before_date,
        after_date=after_date,
        new_risk_factors=result.get("new_risk_factors", []),
        removed_risk_factors=result.get("removed_risk_factors", []),
        language_shift_summary=result.get("language_shift_summary", ""),
        sample_space_signal=result.get("sample_space_signal", False),
        signal_reasoning=result.get("signal_reasoning", ""),
    )


def _analyze_filing_diff_heuristic(
    ticker: str,
    before_text: str,
    after_text: str,
    before_date: str,
    after_date: str,
) -> FilingDiff:
    """Keyword-based filing diff when OpenAI is unavailable."""
    ai_keywords = [
        "artificial intelligence", "generative ai", "large language model",
        "machine learning", "ai infrastructure", "gpu", "chatgpt",
        "ai competition", "ai regulation", "ai safety",
    ]

    before_lower = before_text.lower()
    after_lower = after_text.lower()

    new_risks = []
    for kw in ai_keywords:
        if kw in after_lower and kw not in before_lower:
            new_risks.append(f"New mention of '{kw}' in risk factors")

    has_signal = len(new_risks) >= 2

    return FilingDiff(
        company=ticker,
        ticker=ticker,
        filing_type="10-K",
        before_date=before_date,
        after_date=after_date,
        new_risk_factors=new_risks,
        removed_risk_factors=[],
        language_shift_summary=(
            f"Found {len(new_risks)} new AI-related risk mentions in {after_date} "
            f"filing that were absent in {before_date} filing."
        ),
        sample_space_signal=has_signal,
        signal_reasoning=(
            "Multiple new AI-related risk categories suggest the investment "
            "universe has expanded to include AI as a distinct risk/opportunity axis."
            if has_signal
            else "Insufficient new risk categories to signal sample space expansion."
        ),
    )


def compute_filing_diff(
    ticker: str,
    before_year: int = 2022,
    after_year: int = 2024,
    model: str = "o4-mini",
) -> FilingDiff:
    """
    Full SEC filing diff pipeline:
    fetch two 10-Ks → extract risk factors → compare with LLM.
    """
    print(f"[Layer 2] Fetching {ticker} 10-K filings from EDGAR...")
    filings = get_company_filings(ticker, "10-K")

    # Find filings closest to before_year and after_year
    before_filing = None
    after_filing = None
    for f in filings:
        year = int(f["filing_date"][:4])
        if year <= before_year and before_filing is None:
            before_filing = f
        if year >= after_year and (after_filing is None or year <= int(after_filing["filing_date"][:4])):
            after_filing = f

    # Sort to get closest matches
    before_candidates = [f for f in filings if int(f["filing_date"][:4]) <= before_year]
    after_candidates = [f for f in filings if int(f["filing_date"][:4]) >= after_year]

    if not before_candidates or not after_candidates:
        raise ValueError(
            f"Could not find 10-K filings for {ticker} "
            f"around years {before_year} and {after_year}. "
            f"Available: {[f['filing_date'] for f in filings]}"
        )

    before_filing = before_candidates[0]  # Most recent before cutoff
    after_filing = after_candidates[-1]  # Earliest after cutoff

    print(f"  Before: {before_filing['filing_date']}")
    print(f"  After:  {after_filing['filing_date']}")

    # Fetch and extract risk factors
    print(f"[Layer 2] Fetching filing text...")
    before_text = fetch_filing_text(before_filing["url"])
    time.sleep(0.2)  # SEC rate limit courtesy
    after_text = fetch_filing_text(after_filing["url"])

    print(f"  Before text: {len(before_text)} chars")
    print(f"  After text:  {len(after_text)} chars")

    before_risks = extract_risk_factors_section(before_text)
    after_risks = extract_risk_factors_section(after_text)
    print(f"  Risk factors extracted: {len(before_risks)} / {len(after_risks)} chars")

    # Analyze diff
    print(f"[Layer 2] Analyzing risk factor changes (model: {model})...")
    diff = analyze_filing_diff_with_openai(
        ticker, before_risks, after_risks,
        before_filing["filing_date"], after_filing["filing_date"],
        model=model,
    )

    print(f"  New risk factors: {len(diff.new_risk_factors)}")
    print(f"  Sample space signal: {diff.sample_space_signal}")

    return diff


# ============================================================
# FULL PIPELINE
# ============================================================

def run_narrative_signals(
    news_query: str = "ChatGPT AI market impact",
    news_from: str = "2022-12-01",
    news_to: str = "2023-12-31",
    event_context: str = "ChatGPT launched Nov 30, 2022, disrupting education and accelerating AI infrastructure investment",
    filing_ticker: str = "NVDA",
    filing_before_year: int = 2022,
    filing_after_year: int = 2024,
    sentiment_model: str = "gpt-4.1-nano",
    filing_model: str = "o4-mini",
) -> NarrativeSignals:
    """Run the full Layer 2 narrative signal pipeline."""

    print("=" * 60)
    print("SSED Layer 2: Narrative Signal Engine")
    print("=" * 60)

    # News sentiment
    news = compute_news_signals(
        news_query, news_from, news_to, event_context, sentiment_model
    )

    # SEC filing diff
    try:
        filing_diff = compute_filing_diff(
            filing_ticker, filing_before_year, filing_after_year, filing_model
        )
    except Exception as e:
        print(f"[Layer 2] SEC filing analysis error: {e}")
        filing_diff = None

    return NarrativeSignals(
        news=news,
        filing_diff=filing_diff,
        generated_at=datetime.now().isoformat(),
    )


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SSED Layer 2: Narrative Signal Engine")
    print("Demo: ChatGPT Launch Period")
    print("=" * 60)

    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_newsapi = bool(os.environ.get("NEWSAPI_KEY"))

    if not has_openai:
        print("\nNote: OPENAI_API_KEY not set — using heuristic scoring")
    if not has_newsapi:
        print("Note: NEWSAPI_KEY not set — using curated demo articles")

    signals = run_narrative_signals()

    print("\n" + "=" * 60)
    print("NEWS SENTIMENT SIGNALS")
    print("=" * 60)

    if signals.news:
        news = signals.news
        print(f"Query: {news.query}")
        print(f"Articles analyzed: {news.article_count}")
        print(f"Average sentiment: {news.avg_sentiment:+.3f}")
        print(f"Trend: {news.sentiment_trend}")
        print(f"\nNovel themes detected:")
        for theme, count in sorted(news.novel_theme_counts.items(), key=lambda x: -x[1]):
            print(f"  {theme}: {count} mentions")
        print(f"\nTop articles:")
        for a in news.top_articles[:3]:
            print(f"  [{a.sentiment_score:+.2f}] {a.title}")
            if a.novel_themes:
                print(f"         Themes: {', '.join(a.novel_themes)}")

    print("\n" + "=" * 60)
    print("SEC FILING DIFF SIGNALS")
    print("=" * 60)

    if signals.filing_diff:
        diff = signals.filing_diff
        print(f"Company: {diff.ticker}")
        print(f"Compared: {diff.before_date} vs {diff.after_date}")
        print(f"Sample space signal: {'YES' if diff.sample_space_signal else 'NO'}")
        print(f"\nNew risk factors:")
        for rf in diff.new_risk_factors:
            print(f"  + {rf}")
        if diff.removed_risk_factors:
            print(f"\nRemoved risk factors:")
            for rf in diff.removed_risk_factors:
                print(f"  - {rf}")
        print(f"\nLanguage shift: {diff.language_shift_summary}")
        print(f"Reasoning: {diff.signal_reasoning}")
    else:
        print("No filing diff available")

    # Output as JSON
    print("\n" + "=" * 60)
    print("STRUCTURED OUTPUT (for Layer 3)")
    print("=" * 60)
    output = signals.model_dump()
    # Truncate for display
    if output.get("news") and output["news"].get("top_articles"):
        output["news"]["top_articles"] = output["news"]["top_articles"][:2]
    print(json.dumps(output, indent=2, default=str))
