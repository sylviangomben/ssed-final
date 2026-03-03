# Sample Space Expansion Detector (SSED)

**Detecting when the investment universe itself changes — not just probabilities within it.**

MGMT 69000 — Mastering AI for Finance | Purdue University

[![CI](https://github.com/sylviangomben/ssed-final/actions/workflows/ci.yml/badge.svg)](https://github.com/sylviangomben/ssed-final/actions/workflows/ci.yml)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sylviangomben-used.streamlit.app)

---

## Project Artifacts

| Deliverable | Link |
|-------------|------|
| Live Streamlit App | [sylviangomben-used.streamlit.app](https://sylviangomben-used.streamlit.app) |
| AI Operator Log (DRIVER framework) | [AI_LOG.md](https://github.com/sylviangomben/ssed-final/blob/main/AI_LOG.md) |
| Validation Report (8/8 checks passed) | [VALIDATION_REPORT.md](https://github.com/sylviangomben/ssed-final/blob/main/VALIDATION_REPORT.md) |
| Substack Post | [substack.com/home/post/p-189488220](https://substack.com/home/post/p-189488220) |
| Slides | [SSED_Presentation_v2-2.pptx](https://purdue0-my.sharepoint.com/:p:/r/personal/cngomben_purdue_edu/Documents/SSED_Presentation_v2-2.pptx?d=w9c599df9a9d64b00b1fdd186e874b6d6&csf=1&web=1&e=hg6VPn) |

---

## The Problem

Traditional regime detection tools like Hidden Markov Models (HMMs) catch probability shifts within a known investment universe — what we call **P changes**. But they completely miss when the investment universe *itself* changes — when new asset classes emerge and old ones die. We call this **X changes**, or **sample space expansion**.

**No existing open-source project fuses quantitative regime detection with LLM-powered narrative analysis to make this distinction.** FinGPT does sentiment. hmmlearn does regimes. Nobody combines them to tell you whether the game changed, or just the odds.

### Case Study: ChatGPT's Launch (Nov 30, 2022)

| What Happened | Before | After |
|---------------|--------|-------|
| NVDA | Gaming GPU company | AI infrastructure backbone (+800%) |
| CHGG | Education market leader | Disrupted incumbent (-90%) |
| Investment universe | Tech = FAANG | Tech = FAANG + "AI Infrastructure" |
| Portfolio question | "What's your tech allocation?" | "What's your AI exposure?" |

This wasn't a regime shift. **The game itself changed.**

---

## Architecture

SSED uses a three-layer architecture where each layer builds on the previous one:

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: AI Fusion & Classification (OpenAI o4-mini)       │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  6 function-calling tools (strict: True)            │    │
│  │  → Calls into Layer 1 & Layer 2 for real data       │    │
│  │  → Structured output: RegimeClassification          │    │
│  │  → Classification: P change vs X change             │    │
│  │  → Confidence + evidence chain + reasoning          │    │
│  └─────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: Narrative Signals (OpenAI GPT-4.1-nano + APIs)    │
│  ┌──────────────────────┐  ┌────────────────────────────┐   │
│  │  News Sentiment       │  │  Novel Theme Detection     │   │
│  │  • NewsAPI feed       │  │  • Unprecedented terms     │   │
│  │  • GPT-4.1-nano       │  │  • Category emergence      │   │
│  │    bulk scoring       │  │  • Narrative shift signal   │   │
│  └──────────────────────┘  └────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: Quantitative Signals (deterministic, local)       │
│  ┌────────────┐ ┌──────────┐ ┌────────────┐ ┌───────────┐  │
│  │ HMM Regime │ │ Shannon  │ │ Winner/    │ │ HHI       │  │
│  │ Detection  │ │ Entropy  │ │ Loser      │ │ Concen-   │  │
│  │ (3 states) │ │ (rolling)│ │ Divergence │ │ tration   │  │
│  └────────────┘ └──────────┘ └────────────┘ └───────────┘  │
├─────────────────────────────────────────────────────────────┤
│  Data Layer                                                  │
│  yfinance (prices)             │ NewsAPI (news) │ SEC EDGAR │
└─────────────────────────────────────────────────────────────┘
```

### Layer 1: Quantitative Signals (`quant_signals.py`)

All deterministic — no LLM involvement. Runs entirely locally.

| Signal | Method | What It Detects |
|--------|--------|-----------------|
| **HMM Regime** | Gaussian HMM (hmmlearn), 3 states sorted by volatility | Volatility regime transitions, model fit deterioration |
| **Shannon Entropy** | Rolling entropy of return distributions with z-score | Concentration/novelty — low entropy = few stocks dominating |
| **Divergence** | Winner vs loser spread with velocity + acceleration | Creative destruction — winners pulling away from losers |
| **HHI Concentration** | Herfindahl-Hirschman Index before/after comparison | Market structure shifts — increasing concentration |

### Layer 2: Narrative Signals (`narrative_signals.py`)

Bridges quantitative data with human narrative using LLMs.

| Source | Model | What It Detects |
|--------|-------|-----------------|
| **NewsAPI** | GPT-4.1-nano (bulk sentiment scoring) | Sentiment shifts, novel theme emergence |
| **Demo Articles** | Curated real headlines (fallback) | Works without API keys |

- News articles are scored for sentiment (-1 to +1) and scanned for novel themes
- Themes like "artificial intelligence", "generative AI", "large language model" are flagged as novel category indicators
- Falls back to keyword-based heuristic scoring when OpenAI is unavailable

### Layer 3: Fusion & Classification (`openai_core.py`)

The core innovation — OpenAI o4-mini reasons across all signals using function calling.

**6 Tools (all with `strict: True`):**

| Tool | Returns | Purpose |
|------|---------|---------|
| `get_hmm_regime` | Regime state, probability, transition matrix | Current volatility regime |
| `get_entropy_signals` | Entropy values, z-score, rolling history | Concentration anomalies |
| `get_divergence` | Winner/loser returns, spread, velocity | Creative destruction magnitude |
| `get_concentration` | HHI before/after, top-N weights | Market structure change |
| `get_news_sentiment` | Article count, avg sentiment, themes | Narrative shift detection |
| `get_sec_filing_diff` | New/removed risk factors, language shift | Regulatory awareness of change |

**How it works:**
1. User describes an event (e.g., "ChatGPT Launch")
2. o4-mini receives a system prompt teaching the P vs X framework
3. The model calls tools to gather real data (never generates numbers)
4. After collecting evidence, it produces a **structured `RegimeClassification`** via Pydantic:
   - `classification`: `regime_shift` | `sample_space_expansion` | `mean_reversion` | `inconclusive`
   - `confidence`: `high` | `medium` | `low`
   - `key_evidence`: list of specific data points
   - `reasoning`: natural language explanation

---

## Features

### Event Analysis (Non-Deterministic AI)
Run the full 3-layer analysis on any market event. The AI classification (Layer 3, o4-mini) and AI market narrative summary (GPT-4.1-nano) run **automatically** as part of every analysis — producing different reasoning each time (non-deterministic). Configure winner/loser tickers, event date, and analysis period. Includes 6 preset scenarios:
- ChatGPT Launch (NVDA vs CHGG)
- iPhone Revolution (AAPL vs NOK)
- Streaming Wars (NFLX vs DIS)
- EV Disruption (TSLA vs F)
- Cloud Computing (AMZN vs IBM)
- Social Media Shift (META vs SNAP)

### Long-Short Portfolio Backtest
Dollar-neutral strategy validating the SSED thesis:
- **Long leg:** AI infrastructure winners (configurable tickers)
- **Short leg:** Disrupted incumbents
- **Metrics:** Total return, alpha, Sharpe ratio, max drawdown, volatility
- **Visualization:** Equity curves for portfolio, long leg, short leg, and benchmark

| Metric (ChatGPT case) | Value |
|------------------------|-------|
| Total Return | +243% |
| Alpha vs SPY | +191% |
| Sharpe Ratio | 2.35 |
| Max Drawdown | -17.3% |

### Multi-Event Comparison
Compare expansion signals across multiple historical disruptions side by side:
- Winner/loser return bar charts
- Divergence comparison across events
- Identifies which events show true sample space expansion vs normal regime shifts

### Live Market Scanner & Sector Heatmap
Real-time scan across all 11 S&P 500 sectors (55 stocks):
- **Expansion score** (0-1) per sector combining divergence, entropy, and momentum
- Color-coded heatmap (green/yellow/red)
- Top winners and losers across all tracked stocks
- Alerts when sectors show elevated expansion signals

### AI Chat
Ask questions about the analysis powered by GPT-4.1-nano with full context from the current analysis data.

### Export Report
One-click downloadable report with all metrics, signals, and classification results.

---

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/sylviangomben/used.git
cd used
pip install -r requirements.txt
```

### 2. Run the Dashboard

```bash
streamlit run ssed/dashboard.py
```

The dashboard works immediately in **demo mode** — no API keys required.

### 3. (Optional) Enable AI Features

```bash
cp .env.example .env
```

Edit `.env` with your keys:

```
OPENAI_API_KEY=sk-...              # Enables Layer 2+3 AI analysis
NEWSAPI_KEY=...                    # Enables live news feed (free at newsapi.org)
```

---

## Three Operating Modes

| Mode | API Keys Needed | What Works |
|------|----------------|------------|
| **Demo** | None (yfinance is free) | Quant signals, backtest, sector scanner, heuristic classification, multi-event comparison |
| **OpenAI** | `OPENAI_API_KEY` | + AI classification (o4-mini), sentiment scoring (GPT-4.1-nano), AI chat, AI narrative summary |
| **Full** | + `NEWSAPI_KEY` | + Live news feed from NewsAPI |

The system gracefully degrades — every AI feature has a heuristic fallback that runs without API keys.

---

## Project Structure

```
project2/
├── ssed/                        # Core package
│   ├── __init__.py
│   ├── quant_signals.py         # Layer 1: HMM, entropy, divergence, HHI
│   ├── narrative_signals.py     # Layer 2: news sentiment, novel theme detection
│   ├── openai_core.py           # Layer 3: function calling + fusion classifier
│   ├── backtest.py              # Long-short portfolio backtest engine
│   ├── sector_scanner.py        # Live market scanner across S&P 500 sectors
│   └── dashboard.py             # Streamlit UI — all features integrated
├── portfolio_analyzer.py        # Original entropy/portfolio analysis (foundation)
├── validate_thesis.py           # Original divergence validation (foundation)
├── starter_template.py          # DRIVER framework cascade mapping (foundation)
├── requirements.txt             # Python dependencies
├── .env.example                 # API key template
├── .streamlit/config.toml       # Streamlit Cloud deployment config
└── product/
    ├── product-overview.md      # Product definition & research findings
    └── product-roadmap.md       # Build roadmap (4 sections)
```

---

## APIs Used

### OpenAI API
- **o4-mini** (`reasoning_effort="high"`) — Layer 3 fusion and classification. Receives tool call results from all signals and produces a structured `RegimeClassification` via `client.beta.chat.completions.parse()`.
- **GPT-4.1-nano** — Layer 2 bulk sentiment scoring of news articles. Fast and cheap for batch processing. Also powers the AI chat feature.
- **Function calling** with `strict: True` on all 6 tool definitions ensures the model only calls tools with valid parameters.
- **Structured Outputs** via Pydantic models — every LLM response is typed, never free-form text parsing.
- **Chat Completions API** (not Assistants) — full control over context, multi-model routing, lower latency.

### NewsAPI
- Free tier (100 requests/day) for financial news headlines
- Queries by keyword + date range
- Articles are scored for sentiment and scanned for novel theme emergence

### yfinance
- Open-source Python library for Yahoo Finance market data
- Used across Layer 1 (quant signals), backtest, and sector scanner
- No API key required

### SEC EDGAR
- Direct API calls to `data.sec.gov/submissions/` for company filing metadata
- Full-text filing retrieval for risk factor extraction
- CIK lookup for ticker-to-company mapping

---

## Key Design Decisions

1. **LLM as orchestrator, not calculator** — All numbers come from deterministic code (numpy, pandas, hmmlearn). OpenAI interprets and reasons over results, never generates financial data.

2. **Function calling with `strict: True`** — The LLM calls tools that return real data. This prevents hallucination of financial metrics and ensures reproducibility.

3. **Structured outputs via Pydantic** — Every LLM response is typed (`RegimeClassification`, `ArticleSentiment`, etc.). No regex parsing of free-form text.

4. **Graceful degradation** — Every AI feature has a heuristic fallback. The dashboard is fully functional without any API keys using rule-based classification (signal convergence counting).

5. **Direct APIs over frameworks** — OpenAI function calling directly (no LangChain abstraction), SEC EDGAR API directly (no heavy wrappers). Simpler code, fewer dependencies, full control.

6. **Two-tier model routing** — GPT-4.1-nano for high-volume, low-complexity tasks (sentiment scoring). o4-mini for low-volume, high-complexity reasoning (fusion classification). Optimizes cost and quality.

---

## The Thesis

**Sample space expansion** is fundamentally different from a regime shift:

| Concept | Regime Shift (P changed) | Sample Space Expansion (X changed) |
|---------|-------------------------|-------------------------------------|
| What changes | Probabilities within known universe | The universe itself |
| Example | Fed rate hike changes sector rotation | ChatGPT creates "AI infrastructure" as new category |
| HMM detects? | Yes | No — model fit deteriorates |
| Detection requires | Quantitative signals alone | Quant + narrative fusion |
| Portfolio impact | Rebalance within existing framework | Framework itself needs updating |

The detection signal: **when both statistical measures (entropy anomaly, HMM deterioration, divergence spike) and narrative measures (novel themes, unprecedented terminology) simultaneously indicate the historical model is breaking down.**

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **UI** | Streamlit | Interactive dashboard, charts, sidebar config |
| **Visualization** | Plotly | Equity curves, heatmaps, regime timelines |
| **Quant** | pandas, numpy, scipy | Data manipulation, statistical calculations |
| **HMM** | hmmlearn | Gaussian HMM regime detection (3 volatility states) |
| **ML** | scikit-learn | Supporting ML utilities |
| **AI** | OpenAI API | Function calling, structured outputs, sentiment |
| **Structured Data** | Pydantic | Typed LLM outputs, data validation |
| **Market Data** | yfinance | Yahoo Finance price data, no API key required |
| **News** | NewsAPI | Financial news headlines and articles |
| **Filings** | SEC EDGAR API | Company filing metadata and text |
| **Config** | python-dotenv | API key management via `.env` |

---

## AI Development Workflow (DRIVER Framework)

SSED was built using **Claude Code as a structured AI co-worker**, following the DRIVER framework to systematically move from research through implementation. The AI wasn't just a code generator — it was a collaborator at every stage.

### How AI Was Used at Each Phase

| Phase | What the AI Did | Human Role |
|-------|----------------|------------|
| **Discover** | Researched existing tools (FinGPT, hmmlearn, OpenBB), identified the gap: no tool fuses quant regime detection with LLM narrative analysis | Defined the thesis (P vs X changes) |
| **Define** | Wrote `product-overview.md` — problem statement, architecture, tech stack decisions, tiered build plan | Validated scope, chose what to cut |
| **Represent** | Designed the 3-layer architecture, defined data models (`QuantSignals`, `RegimeClassification`), planned the `product-roadmap.md` with 4 buildable sections | Approved architecture, set priorities |
| **Implement** | Built all code across 4 sections: quant engine, OpenAI function calling core, narrative layer, fusion dashboard | Tested each section, provided feedback |
| **Validate** | Ran backtests, verified signal convergence, tested API integrations (financialdatasets.ai, OpenAI, NewsAPI) | Confirmed results matched thesis |
| **Evolve** | Migrated yfinance to financialdatasets.ai, made AI classification automatic (non-deterministic), added AI narrative summary | Relayed professor feedback, directed changes |

### Non-Deterministic AI Features

The professor's requirement for non-deterministic AI/LLM usage is addressed in two places:

1. **AI Classification (Layer 3)** — OpenAI o4-mini reasons over all quantitative and narrative signals via function calling, producing a typed `RegimeClassification`. This runs **automatically** on every analysis and produces **different reasoning each time** because LLM inference is inherently non-deterministic.

2. **AI Market Narrative Summary** — GPT-4.1-nano generates a 3-4 sentence interpretation of all signals at the top of every analysis. Each run produces a **unique narrative** — same data, different phrasing and emphasis.

Both features are clearly labeled "NON-DETERMINISTIC" in the UI to make the LLM involvement visible.

### DRIVER as Structured AI Collaboration

The DRIVER plugin provided structure that prevented the common failure mode of AI-assisted development: jumping straight to code without planning. Concretely:

- **Product overview first** (`product/product-overview.md`) — Forced articulation of the problem, gap analysis, and architecture *before* writing any code
- **Roadmap with sections** (`product/product-roadmap.md`) — Broke the build into 4 independent sections (Quant Engine → OpenAI Core → Narrative Layer → Dashboard), each with a clear demo deliverable
- **Iterative evolution** — When professor feedback arrived (use financialdatasets.ai, make AI more prominent), the structured approach made it clear which files to change and why, without disrupting the architecture

The AI acted as co-worker by: writing initial implementations, proposing architecture decisions, handling API integration details, and executing systematic refactors across multiple files — while the human directed strategy, validated results, and made product decisions.

---

## References

- Shannon, C. E. (1948). "A Mathematical Theory of Communication." *Bell System Technical Journal*, 27(3), 379-423.
- Rabiner, L. R. (1989). "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition." *Proceedings of the IEEE*, 77(2), 257-286.
- Herfindahl-Hirschman Index (HHI) — U.S. Department of Justice, Market Concentration Measurement.
- OpenAI Function Calling — [platform.openai.com/docs/guides/function-calling](https://platform.openai.com/docs/guides/function-calling)
- OpenAI Structured Outputs — [platform.openai.com/docs/guides/structured-outputs](https://platform.openai.com/docs/guides/structured-outputs)

---

## License

MIT

---

*Built for MGMT 69000: Mastering AI for Finance — Purdue University*
