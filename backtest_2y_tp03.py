"""
2-year backtest with tp=0.3% / trail=0.12%, WFO (4w IS / 4w OOS).
Starting capital 1,000,000 KRW split 50:50 between KRW-XRP and KRW-ETH
(matching upbit_portfolio_trader.py's portfolio model).
Also reports KRW-BTC standalone for reference.
"""
import os, json, time
from dataclasses import asdict

import numpy as np
import pandas as pd

import backtest_strategy as bs


BARS_2Y = 70080  # 2 years of 15-min bars


def run(market, bars):
    df = bs.fetch_candles(market, bars)
    cfg = bs.StrategyConfig(
        tp_pct=0.3, trail_pct=0.12, sl_pct=-2.0,
        slippage_per_side=0.0005,
        initial_balance=500_000.0,  # 50% of 1M total
    )
    r = bs.run_wfo(df, cfg)
    m = bs.compute_metrics(r, df, cfg)
    m["market"] = market
    m["chosen_zs"] = r.get("chosen_zs")
    return df, r, m


def main():
    print(f"Loading 2-year data ({BARS_2Y} bars per market)...")
    xrp_df, xrp_r, xrp_m = run("KRW-XRP", BARS_2Y)
    eth_df, eth_r, eth_m = run("KRW-ETH", BARS_2Y)
    btc_df, btc_r, btc_m = run("KRW-BTC", BARS_2Y)

    print("\n=== 2-YEAR WFO RESULTS (tp=0.3, trail=0.12, sl=-2.0) ===")
    for m in [xrp_m, eth_m, btc_m]:
        print(f"  {m['market']}: ret={m['total_return_pct']:+.2f}%  "
              f"final={m['final_balance']:>12,.0f}  bh={m['buy_and_hold_pct']:+.2f}%  "
              f"alpha={m['alpha_vs_bh_pct']:+.2f}%  MDD={m['mdd_pct']:.2f}%  "
              f"Sharpe={m['sharpe']:.2f}  trades={m['trades']}  win={m['win_rate_pct']:.1f}%")

    # 50:50 portfolio (XRP + ETH), each 500k
    xrp_eq = pd.DataFrame(xrp_r["equity_curve"], columns=["time", "equity"])
    eth_eq = pd.DataFrame(eth_r["equity_curve"], columns=["time", "equity"])
    xrp_eq["time"] = pd.to_datetime(xrp_eq["time"])
    eth_eq["time"] = pd.to_datetime(eth_eq["time"])
    xrp_eq = xrp_eq.set_index("time")
    eth_eq = eth_eq.set_index("time")
    xrp_eq = xrp_eq[~xrp_eq.index.duplicated(keep="last")]
    eth_eq = eth_eq[~eth_eq.index.duplicated(keep="last")]

    common = xrp_eq.index.intersection(eth_eq.index)
    portfolio = xrp_eq.loc[common]["equity"] + eth_eq.loc[common]["equity"]
    initial_total = 1_000_000.0
    final_total = portfolio.iloc[-1]
    span_days = (portfolio.index[-1] - portfolio.index[0]).days
    years = span_days / 365.0
    cagr = (final_total / initial_total) ** (1 / years) - 1 if years > 0 else 0
    cagr *= 100
    total_ret = (final_total / initial_total - 1) * 100

    rolling_max = portfolio.cummax()
    dd = (portfolio / rolling_max - 1) * 100
    mdd = dd.min()

    daily = portfolio.resample("1D").last().ffill()
    daily_ret = daily.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(365) if daily_ret.std() > 0 else 0

    print("\n=== 50:50 PORTFOLIO (KRW-XRP + KRW-ETH, 2-year) ===")
    print(f"Window: {portfolio.index[0]} → {portfolio.index[-1]} ({span_days} days, {years:.2f}y)")
    print(f"Initial: 1,000,000 KRW")
    print(f"Final:   {final_total:,.0f} KRW")
    print(f"Total Return: {total_ret:+.2f}%")
    print(f"CAGR:    {cagr:+.2f}%")
    print(f"MDD:     {mdd:.2f}%")
    print(f"Sharpe:  {sharpe:.2f}")

    # B&H comparison (50:50 of XRP+ETH over the same window)
    bh_xrp = xrp_df[(xrp_df["time"] >= portfolio.index[0]) & (xrp_df["time"] <= portfolio.index[-1])]
    bh_eth = eth_df[(eth_df["time"] >= portfolio.index[0]) & (eth_df["time"] <= portfolio.index[-1])]
    bh_xrp_ret = (bh_xrp["close"].iloc[-1] / bh_xrp["close"].iloc[0] - 1) * 100
    bh_eth_ret = (bh_eth["close"].iloc[-1] / bh_eth["close"].iloc[0] - 1) * 100
    bh_50_50 = 0.5 * bh_xrp_ret + 0.5 * bh_eth_ret
    bh_final = 1_000_000 * (1 + bh_50_50 / 100)
    print(f"\nBuy & Hold 50:50 over same window:")
    print(f"  XRP B&H: {bh_xrp_ret:+.2f}%")
    print(f"  ETH B&H: {bh_eth_ret:+.2f}%")
    print(f"  50:50 B&H final: {bh_final:,.0f} KRW ({bh_50_50:+.2f}%)")
    print(f"  Alpha: {total_ret - bh_50_50:+.2f}%")

    # Yearly snapshots
    print("\n=== YEARLY SNAPSHOTS (50:50 portfolio) ===")
    portfolio_resampled = portfolio.resample("YE").last()
    print(portfolio_resampled.to_string())

    # Stress: try slippage 0.10% per side
    print("\n=== SLIPPAGE STRESS: 0.10% per side ===")
    for market, df in [("KRW-XRP", xrp_df), ("KRW-ETH", eth_df), ("KRW-BTC", btc_df)]:
        cfg = bs.StrategyConfig(
            tp_pct=0.3, trail_pct=0.12, sl_pct=-2.0,
            slippage_per_side=0.0010,
            initial_balance=500_000.0,
        )
        r = bs.run_wfo(df, cfg)
        m = bs.compute_metrics(r, df, cfg)
        print(f"  {market}: ret={m['total_return_pct']:+.2f}%  "
              f"final={m['final_balance']:>12,.0f}  MDD={m['mdd_pct']:.2f}%  "
              f"Sharpe={m['sharpe']:.2f}  trades={m['trades']}")

    out = os.path.join(bs.RESULTS_DIR, f"twoyear_tp03_{int(time.time())}.json")
    with open(out, "w") as f:
        json.dump({
            "xrp": xrp_m, "eth": eth_m, "btc": btc_m,
            "portfolio": {
                "initial": initial_total, "final": final_total,
                "total_return_pct": total_ret, "cagr_pct": cagr,
                "mdd_pct": float(mdd), "sharpe": float(sharpe),
                "span_days": span_days, "years": years,
                "bh_50_50_pct": bh_50_50, "alpha_vs_bh_pct": total_ret - bh_50_50,
            }
        }, f, indent=2, default=str)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
