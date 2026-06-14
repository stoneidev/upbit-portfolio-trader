"""
Stage 1: Screen NASDAQ tickers for pump-day candidates.

Pulls full NASDAQ-listed symbols, downloads 60 days of daily OHLCV in batches,
then identifies (ticker, date) pairs matching:
  - day_open >= prev_close * 1.30   (gap up >= 30%)
  - day_volume >= 10 × 10-day avg vol
  - day high/low spread > 30% of open  (intraday volatility — true pump shape)

Saves results to backtest_results/pump_candidates.csv.
"""
from __future__ import annotations

import os
import time
import pandas as pd
import requests
import yfinance as yf
from io import StringIO

OUT_DIR = "/Users/stoni/Projects/AI/backtest_results"
os.makedirs(OUT_DIR, exist_ok=True)
LIST_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"

GAP_PCT = 0.30
VOL_MULT = 10
RANGE_PCT = 0.30


def fetch_symbols() -> list[str]:
    r = requests.get(LIST_URL, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text), sep="|")
    df = df.dropna(subset=["Symbol"])
    df = df[(df["Test Issue"] == "N") & (df["ETF"] == "N")]
    # Skip warrants/units/preferred (suffix W, U, P, R) and obvious non-tradeables
    bad_suffix = ("W", "WS", "U", "P", "R", "Z")
    df = df[~df["Symbol"].astype(str).str.endswith(bad_suffix)]
    syms = sorted({s for s in df["Symbol"].astype(str).tolist()
                   if "$" not in s and "." not in s})
    return syms


def screen_batch(symbols: list[str], days: int = 60) -> pd.DataFrame:
    """Download daily bars (last `days` days) for symbols in one batch and detect pump days."""
    rows = []
    BATCH = 200
    for i in range(0, len(symbols), BATCH):
        batch = symbols[i:i + BATCH]
        print(f"  batch {i//BATCH + 1}/{(len(symbols)-1)//BATCH + 1}  ({len(batch)} syms)...", flush=True)
        try:
            data = yf.download(batch, period=f"{days}d", interval="1d",
                               progress=False, auto_adjust=False, threads=True,
                               group_by="ticker")
        except Exception as e:
            print(f"    batch error: {e}")
            continue
        # data is a multi-index DataFrame: columns level 0 = ticker
        for sym in batch:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    sub = data[sym].dropna()
                else:
                    sub = data.dropna()
                if len(sub) < 11:
                    continue
                sub = sub.reset_index().rename(columns={
                    "Date": "date", "Open": "o", "High": "h", "Low": "l",
                    "Close": "c", "Volume": "v"
                })
                sub["prev_c"] = sub["c"].shift(1)
                sub["avg_v_10d"] = sub["v"].rolling(10).mean().shift(1)
                sub = sub.dropna()
                # Pump conditions
                cond_gap = sub["o"] >= sub["prev_c"] * (1 + GAP_PCT)
                cond_vol = sub["v"] >= sub["avg_v_10d"] * VOL_MULT
                cond_rng = (sub["h"] - sub["l"]) >= sub["o"] * RANGE_PCT
                hits = sub[cond_gap & cond_vol & cond_rng]
                if hits.empty:
                    continue
                for _, h in hits.iterrows():
                    rows.append({
                        "symbol": sym,
                        "date": str(h["date"].date()) if hasattr(h["date"], "date") else str(h["date"])[:10],
                        "prev_close": float(h["prev_c"]),
                        "open": float(h["o"]),
                        "high": float(h["h"]),
                        "low": float(h["l"]),
                        "close": float(h["c"]),
                        "volume": float(h["v"]),
                        "avg_vol_10d": float(h["avg_v_10d"]),
                        "gap_pct": float((h["o"] / h["prev_c"] - 1) * 100),
                        "vol_mult": float(h["v"] / h["avg_v_10d"]),
                        "intra_range_pct": float((h["h"] - h["l"]) / h["o"] * 100),
                        "open_to_close_pct": float((h["c"] / h["o"] - 1) * 100),
                    })
            except Exception:
                continue
        time.sleep(0.5)  # gentle on yfinance
    return pd.DataFrame(rows)


def main():
    print("Fetching NASDAQ symbols...")
    syms = fetch_symbols()
    print(f"  {len(syms)} symbols after filtering")
    print("Downloading 60 days of daily OHLCV and screening...")
    df = screen_batch(syms, days=60)
    if df.empty:
        print("No pump days found.")
        return
    df = df.sort_values(["date", "vol_mult"], ascending=[True, False])
    out = os.path.join(OUT_DIR, "pump_candidates.csv")
    df.to_csv(out, index=False)
    print(f"\nFound {len(df)} pump days across {df['symbol'].nunique()} unique symbols.")
    print(f"Date range: {df['date'].min()} → {df['date'].max()}")
    print(f"Saved → {out}")
    print("\nTop 20 by volume multiple:")
    cols = ["symbol", "date", "prev_close", "open", "high", "low", "close",
            "gap_pct", "vol_mult", "intra_range_pct", "open_to_close_pct"]
    print(df.nlargest(20, "vol_mult")[cols].to_string(index=False))


if __name__ == "__main__":
    main()
