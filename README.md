# S&P 500 Mean-Reversion Backtest

Backtests a simple mean-reversion strategy on S&P 500 stocks (Oct 2023 – Mar 2024): buy after a large one-day drop, short after a large one-day jump, and hold for N trading days.

`backtest.py` expects `hdf`, a long-format DataFrame of daily prices (columns: `Date`, `Ticker`, `Adj Close`), e.g. downloaded with `yfinance`.

## What it does
- Trade-level backtest for any return band and holding period, with round-trip trading costs
- Equal-weight portfolio returns for long, short, and long-short versions
- Grid search over thresholds and holding periods, ranked by Sharpe ratio
- Comparison against an equal-weight buy-and-hold benchmark, with a growth-of-$1 chart

## Caveats
- Survivorship bias: the universe is today's S&P 500 constituents
- Entry at the same close that triggers the signal is slightly optimistic
- About 5.5 months of data, so results are not statistically robust
