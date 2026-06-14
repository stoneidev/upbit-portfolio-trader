"""
Sweep trailing-stop width for ORB Breakout strategy on the 183 pump
candidates. Question: does a wider trailing stop let us capture more of the
large pump moves at the cost of more SL hits?

Variations:
  - trail_pct: 2% (current), 3%, 5%, 8%, 12%, 20%
  - hard_sl_pct fixed at 3%
  - tp_activation: as soon as in-trade
  - exit at trailing stop OR EOD 15:50
"""
import os
import time
import pandas as pd
import yfinance as yf

import backtest_pump_long_only as longs


CANDIDATES_CSV = "/Users/stoni/Projects/AI/backtest_results/pump_candidates.csv"
OUT_DIR = "/Users/stoni/Projects/AI/backtest_results"


def manage_long_custom(df, entry_idx, trail_pct, hard_sl_pct=3.0, slippage=0.003):
    raw_entry = df.iloc[entry_idx]["c"]
    entry_price = raw_entry * (1 + slippage)
    sl_price = entry_price * (1 - hard_sl_pct / 100)
    peak = entry_price
    halt = False
    trail_active = False
    EOD = pd.Timestamp("15:50").time()

    for i in range(entry_idx + 1, len(df)):
        bar = df.iloc[i]
        if bar["intra_range_pct"] > 10:
            halt = True
        if bar["h"] > peak:
            peak = bar["h"]
            if (peak / entry_price - 1) * 100 >= trail_pct:
                trail_active = True
        if bar["time"].time() >= EOD:
            return {"exit_price": bar["c"]*(1-slippage), "bars": i-entry_idx,
                    "reason": "EOD", "halt": halt, "entry": entry_price}
        if trail_active:
            stop = peak * (1 - trail_pct/100)
            if bar["l"] <= stop:
                return {"exit_price": stop*(1-slippage), "bars": i-entry_idx,
                        "reason": "TRAIL", "halt": halt, "entry": entry_price}
        else:
            if bar["l"] <= sl_price:
                return {"exit_price": sl_price*(1-slippage), "bars": i-entry_idx,
                        "reason": "SL", "halt": halt, "entry": entry_price}
    last = df.iloc[-1]
    return {"exit_price": last["c"]*(1-slippage), "bars": len(df)-1-entry_idx,
            "reason": "DATA_END", "halt": halt, "entry": entry_price}


def find_orb_breakout(df, range_minutes=30):
    range_end = df.iloc[0]["time"] + pd.Timedelta(minutes=range_minutes)
    rng = df[df["time"] < range_end]
    if rng.empty:
        return None
    or_high = rng["h"].max()
    rest = df[df["time"] >= range_end].reset_index(drop=False)
    for _, row in rest.iterrows():
        is_green = row["c"] > row["o"]
        if row["c"] > or_high and is_green:
            return int(row["index"])
    return None


def main():
    cands = pd.read_csv(CANDIDATES_CSV)
    print(f"Loaded {len(cands)} pump candidates.")
    trail_widths = [2.0, 3.0, 5.0, 8.0, 12.0, 20.0]
    rows = []

    for i, c in cands.iterrows():
        if (i + 1) % 25 == 0 or i == len(cands) - 1:
            print(f"  {i+1}/{len(cands)}", flush=True)
        try:
            df = longs.load_intraday(c["symbol"], c["date"])
        except Exception:
            continue
        if df.empty or len(df) < 20:
            continue
        entry_idx = find_orb_breakout(df, 30)
        if entry_idx is None:
            continue
        for tw in trail_widths:
            r = manage_long_custom(df, entry_idx, trail_pct=tw)
            pnl = (r["exit_price"] - r["entry"]) / r["entry"] * 100
            rows.append({
                "symbol": c["symbol"], "date": c["date"],
                "trail_pct": tw, "pnl_pct": pnl,
                "bars_held": r["bars"], "exit_reason": r["reason"],
                "halt_risk": r["halt"],
            })
        time.sleep(0.05)

    res = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR, "pump_trail_sweep.csv")
    res.to_csv(out, index=False)
    print(f"\nSaved → {out}  ({len(res)} rows)")

    # Summary per trail_pct
    summ = res.groupby("trail_pct").agg(
        n=("pnl_pct", "size"),
        win_rate=("pnl_pct", lambda s: (s > 0).mean() * 100),
        mean_pnl=("pnl_pct", "mean"),
        median_pnl=("pnl_pct", "median"),
        p25=("pnl_pct", lambda s: s.quantile(0.25)),
        p75=("pnl_pct", lambda s: s.quantile(0.75)),
        max_pnl=("pnl_pct", "max"),
        min_pnl=("pnl_pct", "min"),
        sl_rate=("exit_reason", lambda s: (s == "SL").mean() * 100),
        eod_rate=("exit_reason", lambda s: (s == "EOD").mean() * 100),
        avg_bars=("bars_held", "mean"),
    ).round(2)
    print("\n=== Per trail-width summary ===")
    print(summ.to_string())

    # Top winners per trail
    print("\n=== Top 5 winners per trail width ===")
    for tw in trail_widths:
        sub = res[res["trail_pct"] == tw].nlargest(5, "pnl_pct")[
            ["symbol", "date", "pnl_pct", "bars_held", "exit_reason"]]
        print(f"\n--- trail = {tw}% ---")
        print(sub.to_string(index=False))


if __name__ == "__main__":
    main()
