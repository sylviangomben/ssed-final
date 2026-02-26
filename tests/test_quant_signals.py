"""
Smoke tests for ssed/quant_signals.py — Layer 1 deterministic functions.

All tests use synthetic price data generated with numpy seed 42.
No API calls are made anywhere in this file.
"""

import numpy as np
import pandas as pd
import pytest

from ssed.quant_signals import (
    shannon_entropy,
    herfindahl_index,
    compute_hmm_signals,
    compute_entropy_signals,
    compute_divergence_signals,
    compute_concentration_signals,
)


# ── Shared synthetic data helpers ──────────────────────────────────────────────

def _make_price_series(n: int = 450) -> pd.Series:
    """
    Synthetic price series with three distinct volatility regimes so that
    hmmlearn can reliably find three separate states without degenerate
    transition rows.

    Segments (each ~150 trading days):
      - Low vol:    sigma = 0.007  (calm market)
      - High vol:   sigma = 0.040  (crisis period)
      - Medium vol: sigma = 0.018  (recovery)
    """
    rng = np.random.default_rng(42)
    seg = n // 3
    returns = np.concatenate([
        rng.normal(0.0003, 0.007, seg),   # low vol
        rng.normal(0.0000, 0.040, seg),   # high vol
        rng.normal(0.0003, 0.018, n - 2 * seg),  # medium vol
    ])
    prices = 100.0 * np.exp(np.cumsum(returns))
    dates = pd.date_range("2021-01-04", periods=n, freq="B")
    return pd.Series(prices, index=dates, name="SYNTH")


def _make_price_df(n: int = 400) -> pd.DataFrame:
    """
    DataFrame with WINNER (upward drift), LOSER (downward drift), SPY (flat).
    Columns match what compute_divergence_signals expects.
    """
    rng = np.random.default_rng(42)
    dates = pd.date_range("2021-01-04", periods=n, freq="B")
    winner = 100.0 * np.exp(np.cumsum(rng.normal(+0.001, 0.02, n)))
    loser  = 100.0 * np.exp(np.cumsum(rng.normal(-0.001, 0.02, n)))
    spy    = 100.0 * np.exp(np.cumsum(rng.normal( 0.000, 0.01, n)))
    return pd.DataFrame({"WINNER": winner, "LOSER": loser, "SPY": spy}, index=dates)


# ── Test 1: Shannon entropy is positive for non-degenerate data ────────────────

def test_shannon_entropy_positive():
    """
    Entropy of a sample drawn from a normal distribution should be strictly
    positive.  A degenerate distribution (all identical values) is the only
    case where entropy can be zero.
    """
    rng = np.random.default_rng(42)
    values = rng.normal(0, 1, 500)
    result = shannon_entropy(values)
    assert result > 0, f"Expected positive entropy, got {result}"


# ── Test 2: HHI is exactly 1/n for a uniform portfolio ────────────────────────

def test_herfindahl_equal_weights():
    """
    When all n sectors carry the same weight, the Herfindahl-Hirschman Index
    equals exactly 1/n (minimum possible concentration for n assets).
    """
    n = 10
    weights = np.ones(n) / n
    result = herfindahl_index(weights)
    assert abs(result - 1.0 / n) < 1e-9, (
        f"HHI for uniform weights should be 1/{n}={1/n:.6f}, got {result:.6f}"
    )


# ── Test 3: HMM regime label is always one of the three valid strings ──────────

def test_hmm_signals_valid_regime_label():
    """
    compute_hmm_signals fits a 3-state Gaussian HMM and remaps states by
    volatility.  The returned regime_label must always be one of the three
    canonical strings regardless of the input data.
    """
    prices = _make_price_series()
    result = compute_hmm_signals(prices, n_regimes=3)

    valid_labels = {"low_volatility", "medium_volatility", "high_volatility"}
    assert result.regime_label in valid_labels, (
        f"Unexpected regime label: {result.regime_label!r}"
    )
    assert 0.0 <= result.regime_probability <= 1.0, (
        f"regime_probability out of [0,1]: {result.regime_probability}"
    )
    assert result.n_regimes == 3


# ── Test 4: Entropy signals fields are correctly typed and bounded ─────────────

def test_entropy_signals_structure():
    """
    compute_entropy_signals must return an EntropySignals dataclass where:
      - current_entropy and baseline_entropy are non-negative floats
      - rolling_entropy and rolling_dates have equal length
      - entropy_zscore is a finite float
    """
    prices = _make_price_series()
    event_date = "2022-01-03"   # well within the series range
    result = compute_entropy_signals(prices, event_date=event_date, window=60)

    assert result.current_entropy >= 0.0
    assert result.baseline_entropy >= 0.0
    assert len(result.rolling_entropy) == len(result.rolling_dates)
    assert np.isfinite(result.entropy_zscore), (
        f"entropy_zscore should be finite, got {result.entropy_zscore}"
    )


# ── Test 5: Divergence is positive when winner drifts up and loser drifts down ─

def test_divergence_signals_direction():
    """
    With a strongly upward-drifting WINNER (+0.1%/day) and a downward-drifting
    LOSER (-0.1%/day), total_divergence_pct must be positive and
    winner_return_pct > loser_return_pct.
    """
    prices = _make_price_df()
    result = compute_divergence_signals(
        prices,
        winner="WINNER",
        loser="LOSER",
        benchmark="SPY",
    )

    assert result.total_divergence_pct > 0, (
        f"Expected positive divergence, got {result.total_divergence_pct}"
    )
    assert result.winner_return_pct > result.loser_return_pct, (
        f"Winner ({result.winner_return_pct:.1f}%) should outperform "
        f"loser ({result.loser_return_pct:.1f}%)"
    )
    assert result.winner_ticker == "WINNER"
    assert result.loser_ticker == "LOSER"
