"""
SSED Long-Short Portfolio Backtest

Two strategies:

1. run_backtest() — illustrated hindsight backtest.
   Tickers are selected by the analyst knowing the outcome (NVDA won, CHGG lost).
   Useful for illustrating the thesis, but NOT a valid test of predictive power.

2. run_forward_looking_backtest() — bias-free walk-forward backtest.
   At each monthly rebalance date T, stocks are ranked by trailing returns using
   ONLY data available at T. The system selects long/short legs — no future
   knowledge. This is the honest test: can the system detect expansion signals
   before you know the answer?
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
    transaction_cost_pct: float  # one-way cost per leg (e.g. 0.001 = 10bps)


def run_backtest(
    long_tickers: list = None,
    short_tickers: list = None,
    benchmark: str = "SPY",
    start_date: str = "2022-11-30",
    end_date: str = "2024-12-01",
    initial_capital: float = 100.0,
    transaction_cost_pct: float = 0.001,  # 10bps one-way per leg (realistic for institutional)
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

    # Apply one-time entry transaction cost (one-way per leg at inception).
    # Both the long and short legs incur execution costs when entered,
    # so effective starting capital is reduced by 2 × tc.
    after_cost = 1.0 - 2 * transaction_cost_pct
    portfolio_values = portfolio * initial_capital * after_cost
    long_values = long_portfolio * initial_capital * (1.0 - transaction_cost_pct)
    short_values = short_portfolio * initial_capital * (1.0 - transaction_cost_pct)
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
        transaction_cost_pct=transaction_cost_pct,
    )


@dataclass
class ForwardLookingBacktestResult:
    """Walk-forward backtest: no look-forward bias in ticker selection."""
    portfolio_values: pd.Series
    long_values: pd.Series
    short_values: pd.Series
    benchmark_values: pd.Series

    total_return_pct: float
    annualized_return_pct: float
    benchmark_return_pct: float
    alpha_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    volatility_pct: float

    rebalance_count: int
    universe: list
    start_date: str
    end_date: str
    trading_days: int
    transaction_cost_pct: float
    note: str


# Default universe: the 12 stocks that appear across the 6 preset scenarios
_DEFAULT_UNIVERSE = [
    "NVDA", "MSFT", "AAPL", "META", "AMZN",
    "NFLX", "TSLA", "CHGG", "NOK", "F", "IBM", "DIS",
]


def run_forward_looking_backtest(
    universe: list = None,
    benchmark: str = "SPY",
    start_date: str = "2022-11-30",
    end_date: str = "2024-12-01",
    lookback_days: int = 60,   # trailing window for momentum signal
    rebalance_days: int = 21,  # ~monthly
    long_n: int = 2,
    short_n: int = 1,
    initial_capital: float = 100.0,
    transaction_cost_pct: float = 0.001,
) -> ForwardLookingBacktestResult:
    """
    Walk-forward long-short backtest with no look-forward bias.

    At each rebalance date T:
      - Rank universe stocks by trailing `lookback_days` return using ONLY data < T
      - Go long top `long_n`, short bottom `short_n`
      - Hold `rebalance_days` trading days, then repeat

    Ticker selection is made by the system at each T — not by the analyst in
    hindsight. This tests whether trailing divergence signals (the same signals
    SSED detects) predict forward returns.
    """
    if universe is None:
        universe = _DEFAULT_UNIVERSE

    all_tickers = list(set(universe + [benchmark]))

    # Fetch extra lookback buffer so the first rebalance has enough history
    buffer_start = (
        pd.Timestamp(start_date) - pd.DateOffset(days=lookback_days * 2)
    ).strftime("%Y-%m-%d")
    prices = fetch_prices(all_tickers, buffer_start, end_date)

    available = [t for t in universe if t in prices.columns]
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    trading_dates = prices.index[(prices.index >= start_ts) & (prices.index <= end_ts)]

    if len(trading_dates) < rebalance_days * 2:
        raise ValueError("Insufficient trading days for walk-forward backtest")

    # Build rebalance schedule: indices into trading_dates
    rebal_indices = list(range(0, len(trading_dates), rebalance_days))

    pf_values = {}
    long_vals = {}
    short_vals = {}

    portfolio_nav = initial_capital
    current_long: list = []
    current_short: list = []

    for seg_num, seg_start_i in enumerate(rebal_indices):
        seg_end_i = rebal_indices[seg_num + 1] if seg_num + 1 < len(rebal_indices) else len(trading_dates)
        seg_dates = trading_dates[seg_start_i:seg_end_i]
        rebal_date = trading_dates[seg_start_i]

        # Select tickers using ONLY data strictly before rebal_date
        past = prices.loc[prices.index < rebal_date, available]
        if len(past) >= lookback_days:
            trailing = (past.iloc[-1] / past.iloc[-lookback_days] - 1).dropna()
            trailing = trailing.sort_values(ascending=False)
            current_long = list(trailing.head(long_n).index)
            current_short = [t for t in trailing.tail(short_n).index if t not in current_long]

        # Apply one-time transaction cost at entry
        seg_nav = portfolio_nav * (1 - 2 * transaction_cost_pct)
        long_nav = seg_nav / 2
        short_nav = seg_nav / 2

        # Day-by-day values for this segment (normalized to rebal_date prices)
        seg_prices = prices.loc[seg_dates]
        entry_prices = prices.loc[rebal_date]

        for date in seg_dates:
            # Long leg
            long_mults = [
                seg_prices.loc[date, t] / entry_prices[t]
                for t in current_long
                if t in seg_prices.columns and entry_prices.get(t, 0) > 0
            ]
            long_mult = float(np.mean(long_mults)) if long_mults else 1.0

            # Short leg (profit when price drops)
            short_mults = [
                seg_prices.loc[date, t] / entry_prices[t]
                for t in current_short
                if t in seg_prices.columns and entry_prices.get(t, 0) > 0
            ]
            short_raw = float(np.mean(short_mults)) if short_mults else 1.0
            short_mult = max(0.0, 2.0 - short_raw)

            pf_values[date] = long_nav * long_mult + short_nav * short_mult
            long_vals[date] = long_nav * long_mult
            short_vals[date] = short_nav * short_mult

        # Carry forward the ending NAV to the next segment
        if seg_dates.any():
            portfolio_nav = pf_values[seg_dates[-1]]

    pf_series = pd.Series(pf_values)
    long_series = pd.Series(long_vals)
    short_series = pd.Series(short_vals)

    bench_series = None
    if benchmark in prices.columns:
        bench_prices = prices.loc[trading_dates, benchmark]
        bench_series = bench_prices * (initial_capital / bench_prices.iloc[0])

    # Performance metrics
    total_return = (pf_series.iloc[-1] / pf_series.iloc[0] - 1) * 100
    days = (trading_dates[-1] - trading_dates[0]).days
    years = max(days / 365.25, 0.01)
    annualized = ((pf_series.iloc[-1] / pf_series.iloc[0]) ** (1 / years) - 1) * 100
    bench_return = (bench_series.iloc[-1] / bench_series.iloc[0] - 1) * 100 if bench_series is not None else 0.0

    daily_returns = pf_series.pct_change().dropna()
    rf_daily = 0.045 / 252
    excess = daily_returns - rf_daily
    sharpe = (excess.mean() / excess.std()) * np.sqrt(252) if excess.std() > 0 else 0.0
    max_dd = ((pf_series - pf_series.cummax()) / pf_series.cummax()).min() * 100
    vol = daily_returns.std() * np.sqrt(252) * 100

    return ForwardLookingBacktestResult(
        portfolio_values=pf_series,
        long_values=long_series,
        short_values=short_series,
        benchmark_values=bench_series,
        total_return_pct=round(float(total_return), 2),
        annualized_return_pct=round(float(annualized), 2),
        benchmark_return_pct=round(float(bench_return), 2),
        alpha_pct=round(float(total_return - bench_return), 2),
        sharpe_ratio=round(float(sharpe), 2),
        max_drawdown_pct=round(float(max_dd), 2),
        volatility_pct=round(float(vol), 2),
        rebalance_count=len(rebal_indices),
        universe=available,
        start_date=start_date,
        end_date=end_date,
        trading_days=len(trading_dates),
        transaction_cost_pct=transaction_cost_pct,
        note=(
            f"Walk-forward: {lookback_days}-day trailing momentum ranking, "
            f"~{rebalance_days}-day rebalance, {long_n} long / {short_n} short. "
            "No look-forward bias — ticker selection made at each T using only data available at T."
        ),
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
