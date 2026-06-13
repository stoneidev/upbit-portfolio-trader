"""
Post-hoc analysis on top of backtest_strategy.py results:
- Reconstruct equity curves for fixed and WFO modes for each market.
- Build 50:50 portfolio (XRP+ETH) and 33:33:33 sanity check.
- Show per-quarter performance (regime sensitivity).
- Stress test: what if no trade actually compounded (i.e. correlation across windows)?
"""
import os
import json
from dataclasses import asdict

import pandas as pd
import numpy as np

import backtest_strategy as bs


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def equity_curve_for_market(market: str, bars: int, mode: str):
    df = bs.fetch_candles(market, bars)
    cfg = bs.StrategyConfig()
    if mode == "fixed":
        result = bs.run_fixed(df, cfg)
    else:
        result = bs.run_wfo(df, cfg)

    eq = pd.DataFrame(result["equity_curve"], columns=["time", "equity"])
    eq["time"] = pd.to_datetime(eq["time"])
    eq = eq.set_index("time")
    eq = eq[~eq.index.duplicated(keep="last")]
    trades = pd.DataFrame([asdict(t) for t in result["trades"]])
    return df, eq, trades


def per_quarter(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    t = trades.copy()
    t["exit_time"] = pd.to_datetime(t["exit_time"])
    t["q"] = t["exit_time"].dt.to_period("Q")
    g = t.groupby("q").agg(
        trades=("pnl_pct", "size"),
        win_rate=("pnl_pct", lambda s: (s > 0).mean() * 100),
        avg_pnl=("pnl_pct", "mean"),
        cum_pnl=("pnl_pct", "sum"),
        max_loss=("pnl_pct", "min"),
        sl_count=("exit_reason", lambda s: (s == "SL").sum()),
        levels_avg=("levels_filled", "mean"),
    ).round(3)
    return g


def main():
    bars = 35040
    print("Loading equity curves...")
    xrp_df, xrp_eq_w, xrp_tr_w = equity_curve_for_market("KRW-XRP", bars, "wfo")
    eth_df, eth_eq_w, eth_tr_w = equity_curve_for_market("KRW-ETH", bars, "wfo")
    btc_df, btc_eq_w, btc_tr_w = equity_curve_for_market("KRW-BTC", bars, "wfo")

    # Align indexes for portfolio
    common_idx = xrp_eq_w.index.intersection(eth_eq_w.index)
    portfolio = (xrp_eq_w.loc[common_idx]["equity"] + eth_eq_w.loc[common_idx]["equity"])
    portfolio_initial = 1_000_000.0
    portfolio_final = portfolio.iloc[-1]
    portfolio_return = (portfolio_final / portfolio_initial - 1) * 100

    running_max = portfolio.cummax()
    dd = (portfolio / running_max - 1) * 100
    mdd = dd.min()

    daily = portfolio.resample("1D").last().ffill()
    daily_ret = daily.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(365) if daily_ret.std() > 0 else 0

    print("\n=== 50:50 PORTFOLIO (XRP+ETH, WFO) ===")
    print(f"Initial: 1,000,000  Final: {portfolio_final:,.0f}")
    print(f"Total Return: {portfolio_return:+.2f}%")
    print(f"MDD: {mdd:.2f}%  Sharpe: {sharpe:.2f}")
    span_days = (portfolio.index[-1] - portfolio.index[0]).days
    print(f"Window: {portfolio.index[0]} → {portfolio.index[-1]} ({span_days} days)")

    bh_xrp = xrp_df[xrp_df["time"].between(portfolio.index[0], portfolio.index[-1])]
    bh_eth = eth_df[eth_df["time"].between(portfolio.index[0], portfolio.index[-1])]
    bh_xrp_ret = (bh_xrp["close"].iloc[-1] / bh_xrp["close"].iloc[0] - 1) * 100
    bh_eth_ret = (bh_eth["close"].iloc[-1] / bh_eth["close"].iloc[0] - 1) * 100
    bh_portfolio = 0.5 * bh_xrp_ret + 0.5 * bh_eth_ret
    print(f"Buy&Hold 50:50 same window: {bh_portfolio:+.2f}%  (XRP {bh_xrp_ret:+.2f}, ETH {bh_eth_ret:+.2f})")
    print(f"Alpha: {portfolio_return - bh_portfolio:+.2f}%")

    print("\n=== PER-QUARTER (XRP WFO) ===")
    print(per_quarter(xrp_tr_w))
    print("\n=== PER-QUARTER (ETH WFO) ===")
    print(per_quarter(eth_tr_w))
    print("\n=== PER-QUARTER (BTC WFO) ===")
    print(per_quarter(btc_tr_w))

    # Loss anatomy
    for name, tr in [("XRP", xrp_tr_w), ("ETH", eth_tr_w), ("BTC", btc_tr_w)]:
        if tr.empty:
            continue
        sl = tr[tr["exit_reason"] == "SL"]
        tp = tr[tr["exit_reason"] == "TRAIL"]
        print(f"\n--- {name} WFO trade anatomy ---")
        print(f"Total: {len(tr)}  SL: {len(sl)}  TRAIL: {len(tp)}")
        print(f"Avg SL pct: {sl['pnl_pct'].mean():.3f}  Avg TRAIL pct: {tp['pnl_pct'].mean():.3f}")
        # By levels filled
        print(tr.groupby("levels_filled").agg(n=("pnl_pct", "size"), avg_pnl=("pnl_pct", "mean")).round(3))


if __name__ == "__main__":
    main()
