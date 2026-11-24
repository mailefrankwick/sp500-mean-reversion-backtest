import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

prices = hdf.pivot(index='Date', columns='Ticker', values='Adj Close').sort_index()
rets = prices.pct_change()


def backtest_trades(prices, low=-0.10, high=-0.05, hold_days=1, cost=0.0):
    """Buy at the close on any day a stock's return lands in (low, high),
    sell at the close `hold_days` trading days later. `cost` = round-trip cost (0.001 = 10 bps)."""
    rets = prices.pct_change()
    exit_px = prices.shift(-hold_days)            # shift the FULL price series, not the filtered one
    signal = (rets > low) & (rets < high) & exit_px.notna()

    trade_ret = (exit_px / prices - 1 - cost)[signal]
    trades = trade_ret.stack().rename('trade_ret').reset_index()

    r = trades['trade_ret']
    return trades, {
        'low': low, 'high': high, 'hold_days': hold_days,
        'n_trades': len(trades),                  # len(), not .size (which is rows x columns)
        'n_tickers': trades['Ticker'].nunique(),
        'mean_ret': r.mean(),
        'median_ret': r.median(),
        'win_rate': (r > 0).mean(),
        't_stat': r.mean() / (r.std() / np.sqrt(len(r))) if len(r) > 1 else np.nan,
    }


def strategy_returns(rets, signal, hold_days, side=1):
    """Daily returns of a portfolio that holds every stock for `hold_days` days
    after its signal fires. side=1 for long, side=-1 for short. Cash (0%) on days with no positions."""
    held = sum(signal.shift(k, fill_value=False).astype(int) for k in range(1, hold_days + 1)) > 0
    n_held = held.sum(axis=1)
    daily = (rets.where(held).sum(axis=1) / n_held.replace(0, np.nan)).fillna(0)
    return side * daily


def perf(daily):
    cum = (1 + daily).cumprod()
    drawdown = cum / cum.cummax() - 1
    return {
        'total_return': cum.iloc[-1] - 1,
        'ann_vol': daily.std() * np.sqrt(252),
        'sharpe': daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else np.nan,
        'max_drawdown': drawdown.min(),
    }


ranges = [(-0.10, -0.05), (-1.0, -0.05), (-0.10, -0.02), (-0.05, -0.02)]
holds = [1, 2, 3, 4, 5, 6, 10]

rows = []
for low, high in ranges:
    for h in holds:
        _, stats = backtest_trades(prices, low, high, h, cost=0.001)
        signal = (rets > low) & (rets < high)
        stats.update(perf(strategy_returns(rets, signal, h)))
        rows.append(stats)

results = pd.DataFrame(rows)
pd.set_option('display.float_format', '{:.4f}'.format)
display(results.sort_values('sharpe', ascending=False))


benchmark = rets.mean(axis=1).fillna(0)           # equal-weight S&P 500 universe, buy and hold

LOW, HIGH, H = -0.10, -0.05, 5                    # pick from the grid above
long_sig = (rets > LOW) & (rets < HIGH)           # buy after a 5-10% drop
short_sig = (rets > -HIGH) & (rets < -LOW)        # short after a 5-10% jump (mean reversion)

long_ret = strategy_returns(rets, long_sig, H, side=1)
short_ret = strategy_returns(rets, short_sig, H, side=-1)
ls_ret = 0.5 * long_ret + 0.5 * short_ret

summary = pd.DataFrame({
    'Long only': perf(long_ret),
    'Short only': perf(short_ret),
    'Long-short': perf(ls_ret),
    'Benchmark (EW)': perf(benchmark),
}).T
display(summary)


fig, ax = plt.subplots(figsize=(10, 5))
for name, s in [('Long only', long_ret), ('Short only', short_ret),
                ('Long-short', ls_ret), ('Benchmark (EW)', benchmark)]:
    ax.plot((1 + s).cumprod(), label=name)
ax.axhline(1, color='gray', lw=0.8)
ax.set_title(f'Mean-reversion strategy: {abs(HIGH):.0%}-{abs(LOW):.0%} moves, {H}-day hold')
ax.set_ylabel('Growth of $1')
ax.legend()
plt.show()

# Caveats to mention in the write-up:
# - Survivorship bias: universe is TODAY's S&P 500 constituents, applied to 2023-24.
# - Entry at the same close that triggers the signal is slightly optimistic.
# - Only ~5.5 months of data, so none of these results are statistically robust.
