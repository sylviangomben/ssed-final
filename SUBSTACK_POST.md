# I Built an AI That Detects When the Investment Universe Changes

*How I combined Hidden Markov Models, OpenAI function calling, and the DRIVER framework to distinguish regime shifts from something more dangerous.*

---

On November 30, 2022, Chegg was a $30 stock and Nvidia was known for gaming GPUs. Two years later, Chegg had fallen 93% and Nvidia had risen 717% — an 810-point spread. Standard risk models didn't flag this coming. For a while I couldn't figure out why, until I realized I was asking the wrong question. I was asking *how the odds changed*. The real question was: *did the game change?*

That's the difference between a regime shift and what I now call **sample space expansion**. It's the problem I built SSED to detect.

---

## P Changes vs X Changes — and Why It Matters

In probability, a sample space is the set of all possible outcomes. Traditional regime detection — Hidden Markov Models, volatility clustering, factor rotation — operates *within* a fixed sample space. The odds shift, allocations rebalance, but the investable universe stays the same. That's a **P change**.

Sample space expansion is different. Before ChatGPT launched, "AI infrastructure" wasn't a portfolio category. It didn't exist as a question you asked your advisor. After ChatGPT, it was mandatory — you either had AI exposure or you explained why you didn't. NVDA went from gaming GPU company to the backbone of a new asset class. CHGG went from education leader to cautionary tale. That's an **X change**: the universe itself changed.

No HMM catches this. The model's log-likelihood deteriorates — it can't explain the data — but it can't tell you *why*. That requires language understanding. SSED fuses quantitative signals with LLM narrative analysis to answer the question every quant tool ignores.

---

## Building It with Claude Code: 3 Real Prompts, 3 Real Decisions

I used Claude Code as a structured co-worker following the DRIVER framework — research, define, represent, implement, validate, evolve — before writing a single line of code.

**Prompt 1:** *"Explicitly document what we're NOT building and why. I need to explain to my professor why I chose direct OpenAI function calling over LangChain."*

Claude produced a clean exclusion list. My decision: I added a design principle Claude hadn't named — *"LLM as orchestrator, not calculator."* All numbers in SSED come from deterministic code. The AI interprets; it never generates a financial figure. That's what makes the tool trustworthy.

**Prompt 2:** *"Build the OpenAI function calling core. The classification must use `client.beta.chat.completions.parse()` with a Pydantic model — not json_object mode."*

Claude's first pass used `json_object` mode with manual parsing. I caught it in review and directed the fix. The difference matters: `parse()` with a Pydantic model enforces the schema at the API level. The classifier is constrained to return only `{regime_shift, sample_space_expansion, mean_reversion, inconclusive}`. On failure, it falls into a typed `INCONCLUSIVE` result — never a crash.

**Prompt 3:** *"Fix these three issues in priority order: [1] backtest short leg P&L can go below zero. [2] openai_core.py uses json_object mode. [3] narrative_signals.py scores articles one at a time."*

I set the order — correctness before performance. On Fix 3, I specified the wrapper-key guard: OpenAI's JSON mode sometimes wraps arrays in a dict like `{"articles": [...]}` rather than a bare array. Without that guard, scoring silently returns empty results. Claude wouldn't have added it without being told.

---

## The Technical Core: o4-mini as Orchestrator

Layer 1 is entirely deterministic: a 3-state Gaussian HMM on benchmark returns, Shannon entropy on a 60-day rolling window, winner/loser divergence with velocity, and HHI concentration shift. No AI. Every number is reproducible.

Layer 2 is GPT-4.1-nano scoring news articles in a single batched API call, flagging novel terminology — terms like "AI infrastructure" that appear post-event but weren't in pre-event financial language.

Layer 3 is where reasoning happens. o4-mini receives all signals as tool results — six tools, all with `strict: True` — and produces a structured `RegimeClassification`: typed classification, confidence level, per-signal interpretation, and a `what_changed` field. It never generates numbers. It reasons over them. Every run produces different language on the same data. That's not a bug. That's what reasoning looks like.

---

## Results

Running the ChatGPT launch case (Nov 2022 – Dec 2024):

- **NVDA:** +717.5% | **CHGG:** −92.9% | **Divergence:** 810%
- **Long-short backtest** (long NVDA + MSFT, short CHGG): **+243% total return**, alpha +191% vs SPY, **Sharpe ratio 2.35**, max drawdown −17.3%

The live sector heatmap scans 55 stocks across all 11 S&P 500 sectors and flags where expansion signals are currently elevated. An automated validation report — 8/8 checks passed on live market data — is in the repo.

---

## What I'd Build Next

The most important missing piece is real-time SEC filing analysis: flagging the moment companies use terminology in their 10-Ks that didn't exist in prior filings. "AI infrastructure" appearing in risk factors for the first time is a *leading* signal, not lagging. That's the difference between detecting expansion as it happens versus after the market has priced it in.

---

## Try It

The live app is at **[sylviangomben-used.streamlit.app](https://sylviangomben-used.streamlit.app)** — no API keys needed for the quant signals, backtest, and heuristic classification.

Full source, AI operator log, and validation report: **[github.com/sylviangomben/ssed-final](https://github.com/sylviangomben/ssed-final)**

If you're thinking about how to structure human-AI collaboration on a technical project, the `AI_LOG.md` documents every real prompt I gave and every decision I made after. That's where the actual operator work lives. Start there.

---

*Built for MGMT 69000: Mastering AI for Finance — Purdue University*
