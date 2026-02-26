"""
SSED Layer 2/3: OpenAI Function Calling Core

Wires o4-mini to reason over quant signals via tool calling.
The LLM never generates numbers — it calls tools that return real data,
then reasons over the results to classify regime type.

Pattern: Chat Completions + tools (strict: True) + structured output (Pydantic)
"""

import json
import os
from enum import Enum

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from ssed.quant_signals import (
    compute_hmm_signals,
    compute_entropy_signals,
    compute_divergence_signals,
    compute_concentration_signals,
    fetch_prices,
    SP500_WEIGHTS_NOV_2022,
    SP500_WEIGHTS_NOV_2024,
)
from ssed.narrative_signals import (
    compute_news_signals,
    compute_filing_diff,
)

load_dotenv()


# ============================================================
# PYDANTIC MODELS — Structured Output Types
# ============================================================

class RegimeType(str, Enum):
    REGIME_SHIFT = "regime_shift"
    SAMPLE_SPACE_EXPANSION = "sample_space_expansion"
    MEAN_REVERSION = "mean_reversion"
    INCONCLUSIVE = "inconclusive"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RegimeClassification(BaseModel):
    """The final typed output from Layer 3 fusion."""
    classification: RegimeType
    confidence: ConfidenceLevel
    reasoning: str
    key_evidence: list[str]
    entropy_interpretation: str
    divergence_interpretation: str
    hmm_interpretation: str
    what_changed: str  # "P changed" or "X changed" — the core distinction


# ============================================================
# TOOL DEFINITIONS — strict: True for guaranteed schema conformance
# ============================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_hmm_regime",
            "description": (
                "Run Hidden Markov Model regime detection on a benchmark index. "
                "Returns current regime state (low/medium/high volatility), "
                "transition probabilities, and model log-likelihood. "
                "Deteriorating log-likelihood suggests the model is seeing data "
                "it wasn't trained on — potential sample space expansion."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Benchmark ticker to analyze (e.g., SPY, QQQ)"
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Start date for HMM training data (YYYY-MM-DD)"
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date for analysis (YYYY-MM-DD)"
                    },
                    "n_regimes": {
                        "type": "integer",
                        "description": "Number of HMM states to fit (typically 3)"
                    }
                },
                "required": ["ticker", "start_date", "end_date", "n_regimes"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_entropy_signals",
            "description": (
                "Calculate Shannon entropy of return distributions over rolling windows. "
                "Compares current entropy to a pre-event baseline. "
                "Decreasing entropy = increasing concentration. "
                "A large negative z-score means concentration is unusually high."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker to analyze"
                    },
                    "event_date": {
                        "type": "string",
                        "description": "The event date that may have caused a regime change (YYYY-MM-DD)"
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Start of data for baseline calculation (YYYY-MM-DD)"
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End of analysis period (YYYY-MM-DD)"
                    }
                },
                "required": ["ticker", "event_date", "start_date", "end_date"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_divergence",
            "description": (
                "Calculate the divergence between a winner and loser stock since an event. "
                "Returns total returns for each, the spread, and the velocity of divergence. "
                "Large divergence (>500%) with acceleration suggests creative destruction, "
                "not normal sector rotation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "winner_ticker": {
                        "type": "string",
                        "description": "Ticker of the winning stock (e.g., NVDA)"
                    },
                    "loser_ticker": {
                        "type": "string",
                        "description": "Ticker of the losing stock (e.g., CHGG)"
                    },
                    "benchmark_ticker": {
                        "type": "string",
                        "description": "Benchmark for comparison (e.g., SPY)"
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Start date (YYYY-MM-DD)"
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date (YYYY-MM-DD)"
                    }
                },
                "required": ["winner_ticker", "loser_ticker", "benchmark_ticker", "start_date", "end_date"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_concentration",
            "description": (
                "Compare market concentration (HHI and top-N weight) between two periods. "
                "Uses S&P 500 sector weights. Increasing HHI means the market is becoming "
                "more dominated by fewer sectors — a signature of sample space expansion "
                "where new winners concentrate capital."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "period_label": {
                        "type": "string",
                        "description": "Label for this comparison (e.g., 'Nov 2022 vs Nov 2024')"
                    }
                },
                "required": ["period_label"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_news_sentiment",
            "description": (
                "Fetch and score news articles for a topic/period. "
                "Returns aggregate sentiment, trend, and novel themes detected. "
                "Novel themes (concepts not seen historically) are a key signal "
                "for sample space expansion — e.g., 'AI infrastructure' as a new "
                "investment category."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "News search query (e.g., 'ChatGPT AI market impact')"
                    },
                    "from_date": {
                        "type": "string",
                        "description": "Start date for news search (YYYY-MM-DD)"
                    },
                    "to_date": {
                        "type": "string",
                        "description": "End date for news search (YYYY-MM-DD)"
                    },
                    "event_context": {
                        "type": "string",
                        "description": "Brief context about the event being analyzed"
                    }
                },
                "required": ["query", "from_date", "to_date", "event_context"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sec_filing_diff",
            "description": (
                "Compare SEC 10-K risk factors between two filing years for a company. "
                "Identifies NEW risk categories that appeared in the later filing but "
                "were absent earlier. When multiple companies simultaneously add new "
                "risk categories (e.g., 'AI disruption'), this signals sample space "
                "expansion — the investment universe itself is changing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Company ticker (e.g., CHGG, NVDA, MSFT)"
                    },
                    "before_year": {
                        "type": "integer",
                        "description": "Year of the earlier 10-K filing"
                    },
                    "after_year": {
                        "type": "integer",
                        "description": "Year of the later 10-K filing"
                    }
                },
                "required": ["ticker", "before_year", "after_year"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
]


# ============================================================
# TOOL DISPATCH — Execute tools with real data
# ============================================================

# Cache fetched prices to avoid redundant API calls
_price_cache: dict = {}


def _get_prices(tickers: list, start: str, end: str):
    """Fetch and cache prices."""
    key = (tuple(sorted(tickers)), start, end)
    if key not in _price_cache:
        _price_cache[key] = fetch_prices(tickers, start, end)
    return _price_cache[key]


def execute_tool(name: str, args: dict) -> dict:
    """Dispatch a tool call to the real implementation."""

    if name == "get_hmm_regime":
        prices = _get_prices([args["ticker"]], args["start_date"], args["end_date"])
        signals = compute_hmm_signals(prices[args["ticker"]], n_regimes=args["n_regimes"])
        return {
            "current_regime": signals.regime_label,
            "regime_probability": signals.regime_probability,
            "transition_matrix": signals.transition_matrix,
            "log_likelihood": signals.log_likelihood,
            "n_regimes": signals.n_regimes,
            "recent_regime_history": signals.regime_history[-20:],
        }

    elif name == "get_entropy_signals":
        prices = _get_prices([args["ticker"]], args["start_date"], args["end_date"])
        signals = compute_entropy_signals(
            prices[args["ticker"]], args["event_date"]
        )
        return {
            "current_entropy": signals.current_entropy,
            "baseline_entropy": signals.baseline_entropy,
            "entropy_change": signals.entropy_change,
            "entropy_zscore": signals.entropy_zscore,
        }

    elif name == "get_divergence":
        tickers = [args["winner_ticker"], args["loser_ticker"], args["benchmark_ticker"]]
        prices = _get_prices(tickers, args["start_date"], args["end_date"])
        signals = compute_divergence_signals(
            prices, args["winner_ticker"], args["loser_ticker"], args["benchmark_ticker"]
        )
        return {
            "winner_ticker": signals.winner_ticker,
            "winner_return_pct": signals.winner_return_pct,
            "loser_ticker": signals.loser_ticker,
            "loser_return_pct": signals.loser_return_pct,
            "benchmark_return_pct": signals.benchmark_return_pct,
            "total_divergence_pct": signals.total_divergence_pct,
            "divergence_velocity": signals.divergence_velocity,
        }

    elif name == "get_concentration":
        signals = compute_concentration_signals(
            SP500_WEIGHTS_NOV_2022, SP500_WEIGHTS_NOV_2024
        )
        return {
            "period": args["period_label"],
            "hhi_baseline": signals.hhi_baseline,
            "hhi_current": signals.hhi_current,
            "hhi_change": signals.hhi_change,
            "top3_weight_baseline": signals.top_n_weight_baseline,
            "top3_weight_current": signals.top_n_weight_current,
            "top3_weight_change": signals.top_n_weight_change,
        }

    elif name == "get_news_sentiment":
        signals = compute_news_signals(
            query=args["query"],
            from_date=args["from_date"],
            to_date=args["to_date"],
            event_context=args["event_context"],
        )
        return {
            "article_count": signals.article_count,
            "avg_sentiment": signals.avg_sentiment,
            "sentiment_trend": signals.sentiment_trend,
            "novel_theme_counts": signals.novel_theme_counts,
            "top_articles": [
                {
                    "title": a.title,
                    "sentiment": a.sentiment_score,
                    "novel_themes": a.novel_themes,
                }
                for a in signals.top_articles[:5]
            ],
        }

    elif name == "get_sec_filing_diff":
        diff = compute_filing_diff(
            ticker=args["ticker"],
            before_year=args["before_year"],
            after_year=args["after_year"],
        )
        return {
            "ticker": diff.ticker,
            "before_date": diff.before_date,
            "after_date": diff.after_date,
            "new_risk_factors": diff.new_risk_factors,
            "removed_risk_factors": diff.removed_risk_factors,
            "language_shift_summary": diff.language_shift_summary,
            "sample_space_signal": diff.sample_space_signal,
            "signal_reasoning": diff.signal_reasoning,
        }

    else:
        return {"error": f"Unknown tool: {name}"}


# ============================================================
# THE SYSTEM PROMPT — Teaches o4-mini the P vs X framework
# ============================================================

SYSTEM_PROMPT = """You are a quantitative financial analyst specializing in regime detection.
You have access to tools that return REAL market data. Use them to gather evidence before
forming conclusions. Never guess numbers — always call tools first.

YOUR CORE FRAMEWORK:
You distinguish between two types of market change:

1. REGIME SHIFT (P changed): The probabilities within the existing investment universe changed.
   - Same assets, same categories, different dynamics
   - Example: A tariff shock changes sector rotation patterns
   - HMM detects this well — it's a transition between known states
   - Entropy may fluctuate but within historical norms

2. SAMPLE SPACE EXPANSION (X changed): The investment universe ITSELF changed.
   - New asset class or category emerges
   - Old categories may become obsolete
   - HMM struggles because it's seeing states it wasn't trained on
   - Entropy shows unusual patterns (often DECREASING despite more options — the paradox)
   - Massive winner/loser divergence (>500%) suggests creative destruction, not rotation
   - New terminology appears in financial discourse

DETECTION SIGNALS for sample space expansion:
- Divergence >500% between winners and losers (creative destruction)
- Entropy z-score < -2 (unusual concentration increase)
- HHI increasing significantly (few players dominate new category)
- HMM log-likelihood deteriorating (model doesn't fit new reality)
- Novel themes in news (new investment categories like "AI infrastructure")
- New risk factors in SEC filings that didn't exist before
- These signals appearing SIMULTANEOUSLY across BOTH quantitative and narrative data

MULTI-MODAL FUSION:
The strongest signal for sample space expansion is when BOTH layers agree:
- Layer 1 (Quant): entropy spike + divergence + concentration increase
- Layer 2 (Narrative): novel themes in news + new risk categories in 10-K filings
When quant AND narrative signals converge, confidence should be HIGH.

IMPORTANT: Call multiple tools in parallel when possible to gather evidence efficiently.
Base your classification on the CONVERGENCE of multiple signals, not any single metric.
Use BOTH quantitative tools AND narrative tools for the most complete picture."""


# ============================================================
# ANALYSIS ENGINE — Tool-calling loop
# ============================================================

def analyze_event(
    query: str,
    model: str = "o4-mini",
    verbose: bool = True,
) -> RegimeClassification:
    """
    Run the full analysis: user asks a question, o4-mini calls tools,
    reasons over results, returns a typed RegimeClassification.
    """
    client = OpenAI()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    if verbose:
        print(f"\n[Layer 3] Query: {query}")
        print(f"[Layer 3] Model: {model}")

    # Tool-calling loop
    iteration = 0
    max_iterations = 5  # safety limit

    while iteration < max_iterations:
        iteration += 1

        if verbose:
            print(f"\n[Layer 3] Round {iteration} — calling {model}...")

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        msg = response.choices[0].message
        # Convert to dict to avoid Pydantic serialization bugs in openai 2.21.0
        msg_dict = {"role": msg.role, "content": msg.content}
        if msg.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        messages.append(msg_dict)

        # If no tool calls, the model is ready to give its answer
        if not msg.tool_calls:
            if verbose:
                print(f"[Layer 3] Model finished reasoning (no more tool calls)")
            break

        # Execute all tool calls (may be parallel)
        if verbose:
            print(f"[Layer 3] Model requested {len(msg.tool_calls)} tool call(s):")

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)

            if verbose:
                print(f"  -> {fn_name}({json.dumps(fn_args, separators=(',', ':'))})")

            result = execute_tool(fn_name, fn_args)

            if verbose:
                # Compact display of result
                display = json.dumps(result, separators=(",", ":"))
                if len(display) > 200:
                    display = display[:200] + "..."
                print(f"  <- {display}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result),
            })

    # Now get the structured classification
    if verbose:
        print(f"\n[Layer 3] Requesting structured classification...")

    messages.append({
        "role": "user",
        "content": (
            "Based on all the evidence you've gathered from the tools, "
            "provide your final classification. Was this event a regime shift "
            "(P changed — probabilities shifted within the existing universe) "
            "or a sample space expansion (X changed — the investment universe "
            "itself changed)? Provide your structured assessment."
        ),
    })

    # Use client.beta.chat.completions.parse() for true Pydantic structured output.
    # This sends the JSON schema to the API and guarantees field names/types match
    # RegimeClassification — no manual json.loads + model_validate needed.
    try:
        classification_response = client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=RegimeClassification,
        )
        result = classification_response.choices[0].message.parsed
        if result is None:
            raise ValidationError("parse() returned None — model refused to produce output")
    except (ValidationError, Exception) as exc:
        if verbose:
            print(f"[Layer 3] Structured output failed: {exc}")
        result = RegimeClassification(
            classification=RegimeType.INCONCLUSIVE,
            confidence=ConfidenceLevel.LOW,
            reasoning=f"Classification could not be produced: {exc}",
            key_evidence=["Structured output parsing failed — see reasoning for details."],
            entropy_interpretation="unavailable",
            divergence_interpretation="unavailable",
            hmm_interpretation="unavailable",
            what_changed="unknown — classification error",
        )

    if verbose:
        print(f"\n{'='*60}")
        print(f"CLASSIFICATION: {result.classification.value}")
        print(f"CONFIDENCE: {result.confidence.value}")
        print(f"WHAT CHANGED: {result.what_changed}")
        print(f"\nREASONING: {result.reasoning}")
        print(f"\nKEY EVIDENCE:")
        for e in result.key_evidence:
            print(f"  - {e}")
        print(f"\nENTROPY: {result.entropy_interpretation}")
        print(f"DIVERGENCE: {result.divergence_interpretation}")
        print(f"HMM: {result.hmm_interpretation}")
        print(f"{'='*60}")

    return result


# ============================================================
# CONVENIENCE: Run full pipeline (Layer 1 + Layer 3)
# ============================================================

def classify_event(
    event_description: str,
    event_date: str = "2022-11-30",
    analysis_end: str = "2024-12-01",
    winner: str = "NVDA",
    loser: str = "CHGG",
    model: str = "o4-mini",
) -> RegimeClassification:
    """
    High-level API: describe an event, get a classification.

    Constructs a detailed query with context and runs the analysis.
    """
    query = f"""Analyze the following market event and classify it:

EVENT: {event_description}
EVENT DATE: {event_date}
ANALYSIS PERIOD: {event_date} to {analysis_end}
SUSPECTED WINNER: {winner}
SUSPECTED LOSER: {loser}
BENCHMARK: SPY

Please use ALL available tools to gather evidence from BOTH quantitative and narrative sources:

QUANTITATIVE (Layer 1):
1. Check HMM regime states for SPY (use data from 2021-01-01 for baseline)
2. Check entropy signals for SPY around the event date
3. Check divergence between {winner} and {loser}
4. Check market concentration changes

NARRATIVE (Layer 2):
5. Check news sentiment for the event period (search for relevant news)
6. Check SEC 10-K filing changes for {loser} (compare pre-event vs post-event years)

Call as many tools in parallel as possible for efficiency.
Then classify: Is this a REGIME SHIFT (P changed) or SAMPLE SPACE EXPANSION (X changed)?"""

    return analyze_event(query, model=model)


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("=" * 60)
        print("SSED Layer 3: OpenAI Function Calling Core")
        print("=" * 60)
        print()
        print("ERROR: OPENAI_API_KEY not set.")
        print()
        print("To run this demo:")
        print("  1. Copy .env.example to .env")
        print("  2. Add your OpenAI API key")
        print("  3. Run: python -m ssed.openai_core")
        print()
        print("Or set it inline:")
        print("  OPENAI_API_KEY=sk-... python -m ssed.openai_core")

        # Show what WOULD happen with a mock
        print()
        print("-" * 60)
        print("PREVIEW: What the tool-calling flow looks like")
        print("-" * 60)
        print()
        print("[Layer 3] Query: Analyze ChatGPT launch (Nov 30, 2022)...")
        print("[Layer 3] Model: o4-mini")
        print("[Layer 3] Round 1 — model requests tools:")
        print("  -> get_hmm_regime(ticker=SPY, ...)")
        print("  -> get_entropy_signals(ticker=SPY, ...)")
        print("  -> get_divergence(winner=NVDA, loser=CHGG, ...)")
        print("  -> get_concentration(...)")
        print("[Layer 3] Round 2 — model reasons over data")
        print("[Layer 3] Structured output: RegimeClassification")
        print()
        print("Expected result:")
        print("  classification: sample_space_expansion")
        print("  confidence: high")
        print('  what_changed: "X changed — the investment universe expanded"')
    else:
        print("=" * 60)
        print("SSED Layer 3: OpenAI Function Calling Core")
        print("Demo: ChatGPT Launch Classification")
        print("=" * 60)

        result = classify_event(
            event_description=(
                "ChatGPT launched on November 30, 2022, reaching 100 million users "
                "in 2 months. This triggered massive investment in AI infrastructure "
                "(Nvidia GPUs, cloud compute) while disrupting knowledge work sectors "
                "(education, tutoring, content creation)."
            ),
            event_date="2022-11-30",
            analysis_end="2024-12-01",
            winner="NVDA",
            loser="CHGG",
        )

        print("\n\nFull structured output:")
        print(json.dumps(result.model_dump(), indent=2))
