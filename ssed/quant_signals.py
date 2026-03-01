"""
SSED Layer 1: Quantitative Signal Engine

Combines HMM regime detection with entropy, divergence, and concentration
metrics to produce structured signals for the fusion layer.

All numbers are deterministic — no LLM involvement in this layer.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from scipy.stats import entropy as scipy_entropy
from dataclasses import dataclass, asdict
from typing import Optional
from datetime import datetime


# ============================================================
# DATA TYPES
# ============================================================

@dataclass
class HMMRegimeState:
    """Current HMM regime detection results."""
    current_regime: int  # 0, 1, 2 (mapped to low/medium/high vol)
    regime_label: str  # "low_volatility", "medium_volatility", "high_volatility"
    regime_probability: float  # confidence in current state
    transition_matrix: list  # P(regime_i -> regime_j)
    log_likelihood: float  # model fit — deterioration signals novelty
    n_regimes: int
    regime_history: list  # regime labels over time


@dataclass
class EntropySignals:
    """Rolling entropy analysis results."""
    current_entropy: float  # Shannon entropy of recent returns distribution
    baseline_entropy: float  # entropy of pre-event period
    entropy_change: float  # current - baseline
    entropy_zscore: float  # how unusual is current entropy vs history
    rolling_entropy: list  # time series of entropy values
    rolling_dates: list  # corresponding dates


@dataclass
class DivergenceSignals:
    """Winner/loser divergence metrics."""
    total_divergence_pct: float  # winner return - loser return
    divergence_velocity: float  # rate of divergence change (acceleration)
    winner_return_pct: float
    loser_return_pct: float
    benchmark_return_pct: float
    winner_ticker: str
    loser_ticker: str


@dataclass
class ConcentrationSignals:
    """Market concentration metrics."""
    hhi_current: float  # Herfindahl-Hirschman Index
    hhi_baseline: float
    hhi_change: float
    top_n_weight_current: float  # top-N concentration ratio
    top_n_weight_baseline: float
    top_n_weight_change: float


@dataclass
class QuantSignals:
    """Complete Layer 1 output — structured for Layer 3 consumption."""
    hmm: HMMRegimeState
    entropy: EntropySignals
    divergence: DivergenceSignals
    concentration: ConcentrationSignals
    analysis_period: str  # e.g. "2022-11-30 to 2024-12-01"
    event_date: str
    generated_at: str

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================
# DATA FETCHING
# ============================================================

def fetch_prices(tickers: list, start: str, end: str) -> pd.DataFrame:
    """Fetch daily close prices via yfinance. Returns DataFrame with tickers as columns."""
    print(f"  [yfinance] Fetching {tickers} ({start} to {end})...")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    # yfinance returns a MultiIndex when multiple tickers are requested
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        # Single ticker — raw is a plain DataFrame; extract Close and name the column
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    # Drop timezone info if present
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)

    return prices.dropna()


# ============================================================
# HMM REGIME DETECTION
# ============================================================

def fit_hmm_regimes(
    returns: np.ndarray,
    n_regimes: int = 3,
    n_iter: int = 100,
    random_state: int = 42,
) -> tuple[GaussianHMM, np.ndarray]:
    """
    Fit Gaussian HMM to return series.

    Returns the fitted model and the decoded regime sequence.
    Regimes are sorted by volatility (0=lowest, 2=highest).
    """
    X = returns.reshape(-1, 1)

    model = GaussianHMM(
        n_components=n_regimes,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_state,
    )
    model.fit(X)

    hidden_states = model.predict(X)
    log_likelihood = model.score(X)

    # Sort regimes by variance (low vol = 0, high vol = n-1)
    state_vols = []
    for i in range(n_regimes):
        state_vols.append(np.sqrt(model.covars_[i][0, 0]))
    vol_order = np.argsort(state_vols)

    # Remap states
    remap = {old: new for new, old in enumerate(vol_order)}
    remapped_states = np.array([remap[s] for s in hidden_states])

    # Remap transition matrix
    transmat = model.transmat_[vol_order][:, vol_order]

    return model, remapped_states, transmat, log_likelihood


REGIME_LABELS = {0: "low_volatility", 1: "medium_volatility", 2: "high_volatility"}


def compute_hmm_signals(
    prices: pd.Series,
    n_regimes: int = 3,
) -> HMMRegimeState:
    """Compute HMM regime signals from a price series."""
    returns = prices.pct_change().dropna().values

    model, states, transmat, log_ll = fit_hmm_regimes(returns, n_regimes)

    current_regime = int(states[-1])
    # Get posterior probabilities — remap columns to match vol-sorted regimes
    _, posteriors = model.score_samples(returns.reshape(-1, 1))
    state_vols = [np.sqrt(model.covars_[i][0, 0]) for i in range(n_regimes)]
    vol_order = np.argsort(state_vols)
    posteriors_remapped = posteriors[:, vol_order]
    regime_prob = float(posteriors_remapped[-1, current_regime])

    regime_history = [REGIME_LABELS.get(s, f"regime_{s}") for s in states]

    return HMMRegimeState(
        current_regime=current_regime,
        regime_label=REGIME_LABELS.get(current_regime, f"regime_{current_regime}"),
        regime_probability=round(regime_prob, 4),
        transition_matrix=transmat.round(4).tolist(),
        log_likelihood=round(float(log_ll), 4),
        n_regimes=n_regimes,
        regime_history=regime_history,
    )


# ============================================================
# ENTROPY ANALYSIS
# ============================================================

def shannon_entropy(values: np.ndarray, bins: int = 50) -> float:
    """Shannon entropy of a distribution estimated via histogram."""
    hist, _ = np.histogram(values, bins=bins, density=True)
    hist = hist[hist > 0]
    # Normalize to probabilities
    hist = hist / hist.sum()
    return float(-np.sum(hist * np.log2(hist)))


def compute_entropy_signals(
    prices: pd.Series,
    event_date: str,
    window: int = 60,  # trading days for rolling window
    baseline_end: Optional[str] = None,
) -> EntropySignals:
    """
    Compute rolling entropy of returns distribution.

    Compares current entropy to a pre-event baseline.
    """
    returns = prices.pct_change().dropna()

    if baseline_end is None:
        baseline_end = event_date

    # Baseline: everything before event
    baseline_returns = returns[returns.index < baseline_end]
    baseline_ent = shannon_entropy(baseline_returns.values) if len(baseline_returns) > window else 0.0

    # Rolling entropy
    rolling_ent = []
    rolling_dates = []
    for i in range(window, len(returns)):
        window_returns = returns.iloc[i - window : i].values
        ent = shannon_entropy(window_returns)
        rolling_ent.append(ent)
        rolling_dates.append(returns.index[i].strftime("%Y-%m-%d"))

    current_ent = rolling_ent[-1] if rolling_ent else 0.0

    # Z-score: how unusual is current entropy vs full history
    ent_arr = np.array(rolling_ent)
    if ent_arr.std() > 0:
        zscore = float((current_ent - ent_arr.mean()) / ent_arr.std())
    else:
        zscore = 0.0

    return EntropySignals(
        current_entropy=round(current_ent, 4),
        baseline_entropy=round(baseline_ent, 4),
        entropy_change=round(current_ent - baseline_ent, 4),
        entropy_zscore=round(zscore, 4),
        rolling_entropy=[round(e, 4) for e in rolling_ent],
        rolling_dates=rolling_dates,
    )


# ============================================================
# DIVERGENCE METRICS
# ============================================================

def compute_divergence_signals(
    prices: pd.DataFrame,
    winner: str,
    loser: str,
    benchmark: str = "SPY",
) -> DivergenceSignals:
    """Compute winner/loser divergence and acceleration."""
    def total_return(col):
        return float((col.iloc[-1] / col.iloc[0] - 1) * 100)

    winner_ret = total_return(prices[winner])
    loser_ret = total_return(prices[loser])
    bench_ret = total_return(prices[benchmark]) if benchmark in prices.columns else 0.0
    divergence = winner_ret - loser_ret

    # Divergence velocity: rate of change in trailing 60-day divergence
    w_norm = prices[winner] / prices[winner].iloc[0]
    l_norm = prices[loser] / prices[loser].iloc[0]
    spread = (w_norm - l_norm) * 100
    velocity = float(spread.diff(periods=60).iloc[-1]) if len(spread) > 60 else 0.0

    return DivergenceSignals(
        total_divergence_pct=round(divergence, 2),
        divergence_velocity=round(velocity, 2),
        winner_return_pct=round(winner_ret, 2),
        loser_return_pct=round(loser_ret, 2),
        benchmark_return_pct=round(bench_ret, 2),
        winner_ticker=winner,
        loser_ticker=loser,
    )


# ============================================================
# CONCENTRATION (HHI)
# ============================================================

def herfindahl_index(weights: np.ndarray) -> float:
    """HHI: sum of squared weights. Range [1/n, 1]."""
    w = np.array(weights)
    w = w[w > 0]
    w = w / w.sum()
    return float(np.sum(w ** 2))


def compute_concentration_signals(
    sector_weights_before: dict,
    sector_weights_after: dict,
    top_n: int = 3,
) -> ConcentrationSignals:
    """Compare concentration between two periods."""
    w_before = np.array(list(sector_weights_before.values()))
    w_after = np.array(list(sector_weights_after.values()))

    hhi_before = herfindahl_index(w_before)
    hhi_after = herfindahl_index(w_after)

    def top_n_weight(w):
        return float(np.sort(w)[::-1][:top_n].sum() / w.sum())

    return ConcentrationSignals(
        hhi_current=round(hhi_after, 6),
        hhi_baseline=round(hhi_before, 6),
        hhi_change=round(hhi_after - hhi_before, 6),
        top_n_weight_current=round(top_n_weight(w_after), 4),
        top_n_weight_baseline=round(top_n_weight(w_before), 4),
        top_n_weight_change=round(top_n_weight(w_after) - top_n_weight(w_before), 4),
    )


# ============================================================
# FULL PIPELINE
# ============================================================

# Documented sector weight snapshots (public S&P 500 data).
# Used as fallbacks and interpolation anchors by fetch_sector_weights().
SP500_WEIGHTS_NOV_2022 = {
    "Technology": 0.20, "Healthcare": 0.15, "Financials": 0.12,
    "Consumer Discretionary": 0.10, "Communication Services": 0.08,
    "Industrials": 0.08, "Consumer Staples": 0.07, "Energy": 0.05,
    "Utilities": 0.03, "Materials": 0.03, "Real Estate": 0.03, "Other": 0.06,
}

SP500_WEIGHTS_NOV_2024 = {
    "Technology": 0.32, "Healthcare": 0.12, "Financials": 0.10,
    "Consumer Discretionary": 0.08, "Communication Services": 0.07,
    "Industrials": 0.08, "Consumer Staples": 0.06, "Energy": 0.04,
    "Utilities": 0.03, "Materials": 0.02, "Real Estate": 0.02, "Other": 0.06,
}

# SPDR sector ETFs used to estimate live sector weights via AUM
_SECTOR_WEIGHT_ETFS = {
    "Technology": "XLK", "Healthcare": "XLV", "Financials": "XLF",
    "Consumer Discretionary": "XLY", "Communication Services": "XLC",
    "Industrials": "XLI", "Consumer Staples": "XLP", "Energy": "XLE",
    "Utilities": "XLU", "Materials": "XLB", "Real Estate": "XLRE",
}

_WEIGHT_SNAPSHOTS = {
    pd.Timestamp("2022-11-30"): SP500_WEIGHTS_NOV_2022,
    pd.Timestamp("2024-12-01"): SP500_WEIGHTS_NOV_2024,
}


def fetch_sector_weights(as_of_date: str) -> dict:
    """
    Return approximate S&P 500 sector weights for a given date.

    Strategy (in order):
      1. Recent dates (within 90 days of today): fetch live ETF total assets
         from yfinance and normalise across the 11 SPDR sector ETFs.
      2. Historical dates: linearly interpolate between the two documented
         snapshots (Nov 2022, Dec 2024).
      3. Any failure: fall back to the nearest snapshot.
    """
    target = pd.Timestamp(as_of_date)
    today = pd.Timestamp.now().normalize()
    snap_ts = sorted(_WEIGHT_SNAPSHOTS.keys())

    # ── 1. Live fetch for recent dates ───────────────────────
    if (today - target).days <= 90:
        try:
            aum = {}
            for sector, etf in _SECTOR_WEIGHT_ETFS.items():
                info = yf.Ticker(etf).info
                assets = info.get("totalAssets", 0)
                if assets and assets > 0:
                    aum[sector] = float(assets)
            if len(aum) >= 8:
                total = sum(aum.values())
                print(f"  [sector weights] Live AUM fetched for {len(aum)} sectors")
                return {s: round(v / total, 4) for s, v in aum.items()}
        except Exception as e:
            print(f"  [sector weights] Live fetch failed ({e}), falling back to interpolation")

    # ── 2. Interpolate between snapshots ────────────────────
    if target <= snap_ts[0]:
        return _WEIGHT_SNAPSHOTS[snap_ts[0]]
    if target >= snap_ts[-1]:
        return _WEIGHT_SNAPSHOTS[snap_ts[-1]]

    t0, t1 = snap_ts[0], snap_ts[-1]
    frac = (target - t0) / (t1 - t0)
    w0, w1 = _WEIGHT_SNAPSHOTS[t0], _WEIGHT_SNAPSHOTS[t1]
    all_sectors = set(w0) | set(w1)
    blended = {s: w0.get(s, 0) * (1 - frac) + w1.get(s, 0) * frac for s in all_sectors}
    total = sum(blended.values())
    return {s: round(v / total, 4) for s, v in blended.items()} if total > 0 else w0


def run_quant_signals(
    event_date: str = "2022-11-30",
    analysis_end: str = "2024-12-01",
    winner: str = "NVDA",
    loser: str = "CHGG",
    benchmark: str = "SPY",
    pre_event_lookback: str = "2021-01-01",
) -> QuantSignals:
    """
    Run the full Layer 1 quant signal pipeline.

    Fetches data, computes all signals, returns structured output.
    """
    print(f"[Layer 1] Fetching price data...")
    tickers = list(set([winner, loser, benchmark]))
    prices = fetch_prices(tickers, pre_event_lookback, analysis_end)
    print(f"  Loaded {len(prices)} trading days for {tickers}")

    # Scope analysis to event period
    event_prices = prices[prices.index >= event_date]

    print(f"[Layer 1] Computing HMM regimes on {benchmark}...")
    hmm_signals = compute_hmm_signals(prices[benchmark], n_regimes=3)
    print(f"  Current regime: {hmm_signals.regime_label} (p={hmm_signals.regime_probability})")

    print(f"[Layer 1] Computing entropy signals on {benchmark}...")
    entropy_signals = compute_entropy_signals(prices[benchmark], event_date)
    print(f"  Entropy change: {entropy_signals.entropy_change:+.4f} (z={entropy_signals.entropy_zscore:.2f})")

    print(f"[Layer 1] Computing divergence: {winner} vs {loser}...")
    divergence_signals = compute_divergence_signals(event_prices, winner, loser, benchmark)
    print(f"  Divergence: {divergence_signals.total_divergence_pct:+.1f}%")
    print(f"    {winner}: {divergence_signals.winner_return_pct:+.1f}%")
    print(f"    {loser}: {divergence_signals.loser_return_pct:+.1f}%")

    print(f"[Layer 1] Computing concentration (HHI)...")
    weights_before = fetch_sector_weights(event_date)
    weights_after = fetch_sector_weights(analysis_end)
    concentration_signals = compute_concentration_signals(weights_before, weights_after)
    print(f"  HHI change: {concentration_signals.hhi_change:+.6f}")

    return QuantSignals(
        hmm=hmm_signals,
        entropy=entropy_signals,
        divergence=divergence_signals,
        concentration=concentration_signals,
        analysis_period=f"{event_date} to {analysis_end}",
        event_date=event_date,
        generated_at=datetime.now().isoformat(),
    )


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SSED Layer 1: Quant Signal Engine")
    print("Demo: ChatGPT Launch (Nov 30, 2022)")
    print("=" * 60)

    signals = run_quant_signals()

    print("\n" + "=" * 60)
    print("STRUCTURED OUTPUT (for Layer 3 consumption)")
    print("=" * 60)

    import json
    output = signals.to_dict()
    # Truncate long lists for display
    output["hmm"]["regime_history"] = output["hmm"]["regime_history"][-10:] + ["...truncated"]
    output["entropy"]["rolling_entropy"] = output["entropy"]["rolling_entropy"][-5:] + ["...truncated"]
    output["entropy"]["rolling_dates"] = output["entropy"]["rolling_dates"][-5:] + ["...truncated"]
    print(json.dumps(output, indent=2))
