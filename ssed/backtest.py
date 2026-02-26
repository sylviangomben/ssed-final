"""
SSED Long-Short Portfolio Backtest

Professor's feedback: "Build the long-short portfolio: go long winners
(NVDA, MSFT) and short losers (CHGG) post-ChatGPT, then backtest performance."

Strategy:
  - Long: NVDA, MSFT (AI infrastructure winners)
  - Short: CHGG (disrupted education)
  - Equal weight within each leg
  - Rebalance: buy-and-hold (no rebalancing) to show raw creative destruction
  - Benchmark: SPY (passive market exposure)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from ssed.quant_signals import fetch_prices


@dataclass
class BacktestResult:
    """Complete backtest output."""
    # Portfolio series
    portfolio_values: pd.Series  # normalized to $100
    long_values: pd.Series
    short_values: pd.Series
    benchmark_values: pd.Series

    # Performance metrics
    total_return_pct: float
    annualized_return_pct: float
    benchmark_return_pct: float
    alpha_pct: float  # portfolio - benchmark
    sharpe_ratio: float
    max_drawdown_pct: float
    volatility_pct: float

    # Long leg
    long_return_pct: float
    long_tickers: list

    # Short leg
    short_return_pct: float  # profit from short (positive = short worked)
    short_tickers: list

    # Metadata
    start_date: str
    end_date: str
    trading_days: int


def run_backtest(
    long_tickers: list = None,
    short_tickers: list = None,
    benchmark: str = "SPY",
    start_date: str = "2022-11-30",
    end_date: str = "2024-12-01",
    initial_capital: float = 100.0,
) -> BacktestResult:
    """
    Run long-short portfolio backtest.

    Long leg: equal weight across long_tickers
    Short leg: equal weight across short_tickers
    Portfolio: 50% long, 50% short (dollar-neutral)
    """
    if long_tickers is None:
        long_tickers = ["NVDA", "MSFT"]
    if short_tickers is None:
        short_tickers = ["CHGG"]

    all_tickers = list(set(long_tickers + short_tickers + [benchmark]))
    prices = fetch_prices(all_tickers, start_date, end_date)

    # Normalize to 1.0 at start
    norm_prices = prices / prices.iloc[0]

    # Long leg: equal weight, average returns
    long_cols = [t for t in long_tickers if t in norm_prices.columns]
    long_portfolio = norm_prices[long_cols].mean(axis=1)

    # Short leg: profit when price drops.
    # A short position started at $1 earns 1 - norm_price per dollar short.
    # Total short account value = initial stake (1.0) + unrealised P&L:
    #   short_value = 1.0 + (1.0 - norm_price)
    #              = 2.0 - norm_price          ← algebraically identical to the old line,
    #   but capped at 0.0 to reflect a margin-call / maximum-loss scenario
    #   (a short can never produce a value below zero in a realistic account).
    short_cols = [t for t in short_tickers if t in norm_prices.columns]
    norm_short = norm_prices[short_cols].mean(axis=1)
    short_portfolio = (1.0 + (1.0 - norm_short)).clip(lower=0.0)

    # Combined: 50% long, 50% short (dollar-neutral)
    portfolio = (long_portfolio * 0.5 + short_portfolio * 0.5)

    # Scale to initial capital
    portfolio_values = portfolio * initial_capital
    long_values = long_portfolio * initial_capital
    short_values = short_portfolio * initial_capital
    benchmark_values = norm_prices[benchmark] * initial_capital if benchmark in norm_prices.columns else None

    # Performance metrics
    total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1) * 100

    days = (prices.index[-1] - prices.index[0]).days
    years = days / 365.25
    annualized = ((portfolio_values.iloc[-1] / portfolio_values.iloc[0]) ** (1 / years) - 1) * 100 if years > 0 else 0

    bench_return = (benchmark_values.iloc[-1] / benchmark_values.iloc[0] - 1) * 100 if benchmark_values is not None else 0

    # Sharpe ratio (annualized, assuming risk-free = 4.5%)
    daily_returns = portfolio_values.pct_change().dropna()
    rf_daily = 0.045 / 252
    excess_returns = daily_returns - rf_daily
    sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0

    # Max drawdown
    cummax = portfolio_values.cummax()
    drawdown = (portfolio_values - cummax) / cummax
    max_dd = drawdown.min() * 100

    # Volatility
    vol = daily_returns.std() * np.sqrt(252) * 100

    # Leg returns
    long_ret = (long_values.iloc[-1] / long_values.iloc[0] - 1) * 100
    short_ret = (short_values.iloc[-1] / short_values.iloc[0] - 1) * 100

    return BacktestResult(
        portfolio_values=portfolio_values,
        long_values=long_values,
        short_values=short_values,
        benchmark_values=benchmark_values,
        total_return_pct=round(float(total_return), 2),
        annualized_return_pct=round(float(annualized), 2),
        benchmark_return_pct=round(float(bench_return), 2),
        alpha_pct=round(float(total_return - bench_return), 2),
        sharpe_ratio=round(float(sharpe), 2),
        max_drawdown_pct=round(float(max_dd), 2),
        volatility_pct=round(float(vol), 2),
        long_return_pct=round(float(long_ret), 2),
        long_tickers=long_tickers,
        short_return_pct=round(float(short_ret), 2),
        short_tickers=short_tickers,
        start_date=start_date,
        end_date=end_date,
        trading_days=len(prices),
    )


if __name__ == "__main__":
    print("=" * 60)
    print("SSED Long-Short Portfolio Backtest")
    print("Strategy: Long NVDA+MSFT / Short CHGG")
    print("Period: ChatGPT Launch to Dec 2024")
    print("=" * 60)

    result = run_backtest()

    print(f"\n{'PERFORMANCE SUMMARY':=^60}")
    print(f"  Total Return:      {result.total_return_pct:+.1f}%")
    print(f"  Annualized Return: {result.annualized_return_pct:+.1f}%")
    print(f"  Benchmark (SPY):   {result.benchmark_return_pct:+.1f}%")
    print(f"  Alpha:             {result.alpha_pct:+.1f}%")
    print(f"  Sharpe Ratio:      {result.sharpe_ratio:.2f}")
    print(f"  Max Drawdown:      {result.max_drawdown_pct:.1f}%")
    print(f"  Volatility:        {result.volatility_pct:.1f}%")

    print(f"\n{'LEG BREAKDOWN':=^60}")
    print(f"  Long ({', '.join(result.long_tickers)}):  {result.long_return_pct:+.1f}%")
    print(f"  Short ({', '.join(result.short_tickers)}): {result.short_return_pct:+.1f}%")

    print(f"\n{'METADATA':=^60}")
    print(f"  Period:       {result.start_date} to {result.end_date}")
    print(f"  Trading Days: {result.trading_days}")
