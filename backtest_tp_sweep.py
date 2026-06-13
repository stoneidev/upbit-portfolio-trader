"""
Sweep TP activation level (and optionally trail width) to see if raising the
profit target improves the strategy. Tests on KRW-XRP, KRW-ETH, KRW-BTC with
both fixed and WFO-style fitted z thresholds.
"""
import os
import json
import time
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

import backtest_strategy as bs


def run_one(df, tp_pct, trail_pct, z_thresh=None, sl_pct=-2.0):
    cfg = bs.StrategyConfig(
        z_thresh=z_thresh if z_thresh is not None else -1.2,
        tp_pct=tp_pct,
        trail_pct=trail_pct,
        sl_pct=sl_pct,
    )
    df_ind = bs.compute_indicators(df)
    res = bs.simulate(df_ind, cfg, start_idx=100)
    m = bs.compute_metrics(res, df, cfg)
    m["tp_pct"] = tp_pct
    m["trail_pct"] = trail_pct
    m["sl_pct"] = sl_pct
    m["z_thresh"] = cfg.z_thresh
    return m


def sweep_market(market, bars=35040):
    df = bs.fetch_candles(market, bars)
    rows = []

    # TP grid (trailing activation), trail width as fraction of TP (40%)
    tp_values = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    trail_ratio = 0.4  # trail width = 40% of tp activation

    for tp in tp_values:
        trail = max(0.1, round(tp * trail_ratio, 2))
        # Test each TP across multiple z thresholds (mimics WFO surface)
        best = None
        for z in [-1.0, -1.2, -1.5, -1.8, -2.0]:
            m = run_one(df, tp, trail, z_thresh=z)
            m["market"] = market
            if best is None or m["total_return_pct"] > best["total_return_pct"]:
                best = m
        rows.append(best)

    # Also explore widening SL together with raising TP (better R:R symmetry)
    rr_pairs = [(0.5, -1.0), (1.0, -2.0), (1.5, -2.0), (2.0, -3.0), (3.0, -3.0), (3.0, -4.0)]
    for tp, sl in rr_pairs:
        trail = max(0.1, round(tp * trail_ratio, 2))
        best = None
        for z in [-1.0, -1.2, -1.5, -1.8, -2.0]:
            m = run_one(df, tp, trail, z_thresh=z, sl_pct=sl)
            m["market"] = market
            m["scenario"] = f"tp={tp}/sl={sl}"
            if best is None or m["total_return_pct"] > best["total_return_pct"]:
                best = m
        rows.append(best)

    return rows


def main():
    all_rows = []
    for market in ["KRW-XRP", "KRW-ETH", "KRW-BTC"]:
        print(f"\nSweeping {market}...")
        all_rows.extend(sweep_market(market))

    df = pd.DataFrame(all_rows)
    cols = ["market", "tp_pct", "trail_pct", "sl_pct", "z_thresh",
            "total_return_pct", "buy_and_hold_pct", "alpha_vs_bh_pct",
            "mdd_pct", "sharpe", "trades", "win_rate_pct", "profit_factor",
            "avg_win_pct", "avg_loss_pct"]
    df_view = df[cols].sort_values(["market", "tp_pct", "sl_pct"]).round(2)
    print("\n=== TP / SL SWEEP (best z per row) ===")
    for market in df_view["market"].unique():
        print(f"\n--- {market} ---")
        sub = df_view[df_view["market"] == market]
        print(sub.to_string(index=False))

    # Best per market
    print("\n=== BEST CONFIG PER MARKET (by total return) ===")
    best_per = df.loc[df.groupby("market")["total_return_pct"].idxmax()]
    print(best_per[cols].round(2).to_string(index=False))

    print("\n=== BEST CONFIG PER MARKET (by Sharpe) ===")
    best_sharpe = df.loc[df.groupby("market")["sharpe"].idxmax()]
    print(best_sharpe[cols].round(2).to_string(index=False))

    # Save
    os.makedirs(bs.RESULTS_DIR, exist_ok=True)
    out = os.path.join(bs.RESULTS_DIR, f"tp_sweep_{int(time.time())}.json")
    with open(out, "w") as f:
        json.dump(all_rows, f, indent=2, default=str)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
