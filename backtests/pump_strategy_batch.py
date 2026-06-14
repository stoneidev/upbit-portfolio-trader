"""
Stage 2: Apply 5 long-only intraday strategies to every (ticker, date) pump
candidate from pump_candidates.csv. Reports per-strategy distributions.
"""
from __future__ import annotations

import os
import time
import json
import math
from dataclasses import asdict

import numpy as np
import pandas as pd
import yfinance as yf

import backtest_pump_long_only as longs


CANDIDATES_CSV = "/Users/stoni/Projects/AI/backtest_results/pump_candidates.csv"
OUT_DIR = "/Users/stoni/Projects/AI/backtest_results"


STRATS = {
    "A. ORB Breakout (30m)":        lambda df: longs.strat_orb_breakout(df, 30),
    "B. VWAP Reclaim":              longs.strat_vwap_reclaim,
    "C. Lower-Low Reversal (>60m)": lambda df: longs.strat_lower_low_reversal(df, 60),
    "D. First Higher-High (>60m)":  lambda df: longs.strat_higher_high_reversal(df, 60),
    "E. Filtered ORB (vol×2,15m)":  lambda df: longs.strat_orb_breakout_vol_filter(df, 15),
}


def run_one(symbol: str, day: str) -> list[dict]:
    rows = []
    try:
        df = longs.load_intraday(symbol, day)
    except Exception as e:
        return [{"symbol": symbol, "date": day, "strategy": s, "triggered": False,
                 "pnl_pct": None, "reason": f"load error: {e}"}
                for s in STRATS]
    if df.empty or len(df) < 20:
        return [{"symbol": symbol, "date": day, "strategy": s, "triggered": False,
                 "pnl_pct": None, "reason": "no data"}
                for s in STRATS]
    for name, fn in STRATS.items():
        try:
            t = fn(df)
        except Exception as e:
            rows.append({"symbol": symbol, "date": day, "strategy": name,
                         "triggered": False, "pnl_pct": None,
                         "reason": f"strategy error: {e}"})
            continue
        if t is None:
            rows.append({"symbol": symbol, "date": day, "strategy": name,
                         "triggered": False, "pnl_pct": None,
                         "reason": "no setup"})
        else:
            rows.append({
                "symbol": symbol, "date": day, "strategy": name,
                "triggered": True,
                "pnl_pct": t.pnl_pct_net,
                "bars_held": t.bars_held,
                "halt_risk": t.halt_risk,
                "entry_time": t.entry_time, "entry_price": t.entry_price,
                "exit_time": t.exit_time, "exit_price": t.exit_price,
                "reason": t.reason,
            })
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strat in df["strategy"].unique():
        s = df[df["strategy"] == strat]
        triggered = s[s["triggered"]]
        n_total = len(s)
        n_trig = len(triggered)
        if n_trig == 0:
            rows.append({"strategy": strat, "n_days": n_total, "n_triggered": 0,
                         "trigger_rate%": 0.0, "win_rate%": 0.0,
                         "mean_pnl%": 0.0, "median_pnl%": 0.0,
                         "p25_pnl%": 0.0, "p75_pnl%": 0.0,
                         "max_pnl%": 0.0, "min_pnl%": 0.0,
                         "halt_rate%": 0.0})
            continue
        wins = (triggered["pnl_pct"] > 0).sum()
        rows.append({
            "strategy": strat,
            "n_days": n_total,
            "n_triggered": n_trig,
            "trigger_rate%": round(n_trig / n_total * 100, 1),
            "win_rate%": round(wins / n_trig * 100, 1),
            "mean_pnl%": round(triggered["pnl_pct"].mean(), 2),
            "median_pnl%": round(triggered["pnl_pct"].median(), 2),
            "p25_pnl%": round(triggered["pnl_pct"].quantile(0.25), 2),
            "p75_pnl%": round(triggered["pnl_pct"].quantile(0.75), 2),
            "max_pnl%": round(triggered["pnl_pct"].max(), 2),
            "min_pnl%": round(triggered["pnl_pct"].min(), 2),
            "halt_rate%": round(triggered["halt_risk"].mean() * 100, 1) if "halt_risk" in triggered.columns else 0,
        })
    return pd.DataFrame(rows)


def main():
    cands = pd.read_csv(CANDIDATES_CSV)
    print(f"Loaded {len(cands)} pump candidates.")
    print(f"Unique symbols: {cands['symbol'].nunique()}")
    print(f"Date range: {cands['date'].min()} → {cands['date'].max()}")

    all_rows = []
    for i, c in cands.iterrows():
        if (i + 1) % 25 == 0 or i == len(cands) - 1:
            print(f"  {i+1}/{len(cands)}  {c['symbol']} {c['date']}", flush=True)
        all_rows.extend(run_one(c["symbol"], c["date"]))
        time.sleep(0.15)

    res_df = pd.DataFrame(all_rows)
    out_full = os.path.join(OUT_DIR, "pump_strategy_results.csv")
    res_df.to_csv(out_full, index=False)
    print(f"\nSaved → {out_full}  ({len(res_df)} rows)")

    summary = summarize(res_df)
    print("\n=== Per-strategy summary ===")
    print(summary.to_string(index=False))
    out_sum = os.path.join(OUT_DIR, "pump_strategy_summary.csv")
    summary.to_csv(out_sum, index=False)
    print(f"Saved → {out_sum}")


if __name__ == "__main__":
    main()
