"""
Long-only intraday strategies for pump-and-dump days.

Tested on SLGB June 9, 2026 (5-min bars). Korean brokers can't short US
equities, so we restrict to long entries only. All trades close by 15:50 ET.

Strategies:
  A) Opening Range Breakout — buy when price breaks above first 30-min high
     with a green bar, EOD close, trailing stop, or hard SL.
  B) VWAP Reclaim Long — after at least one bar below VWAP, buy when price
     closes back above VWAP with a green bar.
  C) Lower-Low Reversal — after the day's low has been set and price posts
     a green bar with close > prior bar's high, buy.
  D) Higher-High Reversal — after the day's low, wait for first bar that
     breaks the prior bar's high (momentum confirmation), buy.
  E) Pre-Pump Accumulation (control) — Opening Range Breakout with stricter
     filter: only enter if first-15min vol > 2x average of next 60min so far.

Risk model:
  - Slippage 0.3% per side (micro-cap pump days have wide spreads).
  - Hard SL at entry × (1 - 3%).
  - Trailing stop: 2% from peak after entry.
  - Force close at 15:50 ET.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional

import pandas as pd
import yfinance as yf


SLIPPAGE = 0.003           # 0.3% per side
HARD_SL_PCT = 3.0          # 3% stop loss
TRAIL_PCT = 2.0            # 2% trailing stop
EOD_CUTOFF = pd.Timestamp("15:50").time()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_intraday(symbol: str, day: str) -> pd.DataFrame:
    raw = yf.download(symbol, start=day,
                      end=pd.Timestamp(day) + pd.Timedelta(days=1),
                      interval="5m", progress=False, auto_adjust=False)
    raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
    df = raw.reset_index().rename(columns={
        "Datetime": "time", "Open": "o", "High": "h", "Low": "l",
        "Close": "c", "Volume": "v"
    })
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("US/Eastern")
    df = df[(df["time"].dt.time >= pd.Timestamp("09:30").time())
            & (df["time"].dt.time <= pd.Timestamp("16:00").time())]
    df = df.reset_index(drop=True)

    typical = (df["h"] + df["l"] + df["c"]) / 3
    cum_vol = df["v"].cumsum().replace(0, 1)
    df["vwap"] = (typical * df["v"]).cumsum() / cum_vol
    df["bar_change_pct"] = (df["c"] / df["c"].shift(1) - 1) * 100
    df["intra_range_pct"] = (df["h"] - df["l"]) / df["l"] * 100
    return df


# ---------------------------------------------------------------------------
# Trade record + manager
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    strategy: str
    entry_time: str
    entry_price: float
    exit_time: str
    exit_price: float
    bars_held: int
    pnl_pct_gross: float
    pnl_pct_net: float
    reason: str
    halt_risk: bool


def manage_long(df: pd.DataFrame, entry_idx: int, strategy: str, setup_reason: str) -> Trade:
    """Hold a long with hard SL + trailing stop, force close at 15:50."""
    raw_entry = df.iloc[entry_idx]["c"]
    entry_price = raw_entry * (1 + SLIPPAGE)  # pessimistic fill
    sl_price = entry_price * (1 - HARD_SL_PCT / 100)
    peak = entry_price
    halt = False
    trail_active = False  # only after price moves > TRAIL_PCT in our favor

    for i in range(entry_idx + 1, len(df)):
        bar = df.iloc[i]
        if bar["intra_range_pct"] > 10:
            halt = True
        # Update peak / trail
        if bar["h"] > peak:
            peak = bar["h"]
            if (peak / entry_price - 1) * 100 >= TRAIL_PCT:
                trail_active = True

        # Force close at 15:50
        if bar["time"].time() >= EOD_CUTOFF:
            exit_price = bar["c"] * (1 - SLIPPAGE)
            return _build(strategy, entry_idx, entry_price, i, exit_price,
                          "Force close 15:50", df, halt, setup_reason)

        # Trailing stop check
        if trail_active:
            stop = peak * (1 - TRAIL_PCT / 100)
            if bar["l"] <= stop:
                exit_price = stop * (1 - SLIPPAGE)
                return _build(strategy, entry_idx, entry_price, i, exit_price,
                              f"Trailing stop @ ${stop:.3f}", df, halt, setup_reason)
        # Hard SL (only when trail not yet active)
        else:
            if bar["l"] <= sl_price:
                exit_price = sl_price * (1 - SLIPPAGE)
                return _build(strategy, entry_idx, entry_price, i, exit_price,
                              f"Hard SL @ ${sl_price:.3f}", df, halt, setup_reason)

    # Ran out of bars (data ended before 15:50) — close at last bar
    last = df.iloc[-1]
    return _build(strategy, entry_idx, entry_price, len(df) - 1,
                  last["c"] * (1 - SLIPPAGE),
                  "Data ended (last bar)", df, halt, setup_reason)


def _build(strategy: str, ei: int, ep: float, xi: int, xp: float,
           reason: str, df: pd.DataFrame, halt: bool, setup: str) -> Trade:
    bars = xi - ei
    gross = (xp - ep) / ep * 100
    return Trade(
        strategy=strategy,
        entry_time=str(df.iloc[ei]["time"]),
        entry_price=ep,
        exit_time=str(df.iloc[xi]["time"]),
        exit_price=xp,
        bars_held=bars,
        pnl_pct_gross=gross,
        pnl_pct_net=gross,  # slippage already baked into ep / xp
        reason=f"Setup: {setup}; Exit: {reason}",
        halt_risk=halt,
    )


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def strat_orb_breakout(df: pd.DataFrame, range_minutes: int = 30) -> Optional[Trade]:
    range_end = df.iloc[0]["time"] + pd.Timedelta(minutes=range_minutes)
    rng = df[df["time"] < range_end]
    if rng.empty:
        return None
    or_high = rng["h"].max()
    rest = df[df["time"] >= range_end].reset_index(drop=False)
    for _, row in rest.iterrows():
        is_green = row["c"] > row["o"]
        if row["c"] > or_high and is_green:
            entry_idx = int(row["index"])
            return manage_long(df, entry_idx, "A. ORB Breakout",
                               f"Closed >${or_high:.3f} (OR high) with green bar")
    return None


def strat_vwap_reclaim(df: pd.DataFrame) -> Optional[Trade]:
    saw_below = False
    for i in range(1, len(df)):
        bar = df.iloc[i]
        if bar["c"] < bar["vwap"]:
            saw_below = True
        elif saw_below and bar["c"] > bar["vwap"] and bar["c"] > bar["o"]:
            return manage_long(df, i, "B. VWAP Reclaim",
                               f"Reclaimed VWAP ${bar['vwap']:.3f} with green bar")
    return None


def strat_lower_low_reversal(df: pd.DataFrame, after_minutes: int = 60) -> Optional[Trade]:
    """Wait for at least `after_minutes` to find LOD context, then buy first
    green bar that closes above the prior bar's high after the LOD has been set."""
    after = df.iloc[0]["time"] + pd.Timedelta(minutes=after_minutes)
    lod = float("inf")
    lod_idx = None
    for i in range(len(df)):
        if df.iloc[i]["l"] < lod:
            lod = df.iloc[i]["l"]
            lod_idx = i
        if df.iloc[i]["time"] < after or lod_idx is None:
            continue
        if i <= lod_idx + 1:
            continue
        bar = df.iloc[i]
        prev = df.iloc[i - 1]
        is_green = bar["c"] > bar["o"]
        if is_green and bar["c"] > prev["h"]:
            return manage_long(df, i, "C. Lower-Low Reversal",
                               f"After LOD ${lod:.3f} at {df.iloc[lod_idx]['time']}, "
                               f"green bar closes above prior bar high")
    return None


def strat_higher_high_reversal(df: pd.DataFrame, after_minutes: int = 60) -> Optional[Trade]:
    """After at least `after_minutes`, find first bar whose high exceeds prior
    bar's high (momentum confirmation) — used as long entry."""
    after = df.iloc[0]["time"] + pd.Timedelta(minutes=after_minutes)
    for i in range(1, len(df)):
        if df.iloc[i]["time"] < after:
            continue
        bar = df.iloc[i]
        prev = df.iloc[i - 1]
        if bar["h"] > prev["h"] and bar["c"] > bar["o"]:
            return manage_long(df, i, "D. First Higher-High",
                               f"First higher-high after {after.time()}")
    return None


def strat_orb_breakout_vol_filter(df: pd.DataFrame, range_minutes: int = 15) -> Optional[Trade]:
    """ORB Breakout but only enter when total volume in first `range_minutes`
    > 2× average of the next 60 minutes."""
    range_end = df.iloc[0]["time"] + pd.Timedelta(minutes=range_minutes)
    rng = df[df["time"] < range_end]
    if rng.empty:
        return None
    or_high = rng["h"].max()
    or_vol = rng["v"].sum()

    next_window = df[(df["time"] >= range_end)
                     & (df["time"] < range_end + pd.Timedelta(minutes=60))]
    if next_window.empty:
        return None
    avg_vol_next = next_window["v"].mean()
    if avg_vol_next <= 0 or or_vol < 2 * avg_vol_next * len(rng):
        return None

    rest = df[df["time"] >= range_end].reset_index(drop=False)
    for _, row in rest.iterrows():
        is_green = row["c"] > row["o"]
        if row["c"] > or_high and is_green:
            return manage_long(df, int(row["index"]),
                               "E. Filtered ORB Breakout",
                               f"OR vol {or_vol:.0f} > 2× avg next-60 vol; broke ${or_high:.3f} green")
    return None


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def main():
    symbol = "SLGB"
    day = "2026-06-09"
    df = load_intraday(symbol, day)
    print(f"\n=== {symbol} {day} — LONG-ONLY pump-day strategies ===")
    print(f"Bars: {len(df)}  Open: ${df['o'].iloc[0]:.3f}  HOD: ${df['h'].max():.3f}  "
          f"LOD: ${df['l'].min():.3f}  Close: ${df['c'].iloc[-1]:.3f}")
    print(f"Bars with >10% intra-range (LULD halt risk): "
          f"{(df['intra_range_pct'] > 10).sum()}")
    print(f"Risk model: slip 0.3%/side, HardSL -3%, Trailing -2% after +2%, EOD 15:50")

    strategies = {
        "A. ORB Breakout (30m)":        strat_orb_breakout(df, 30),
        "B. VWAP Reclaim":              strat_vwap_reclaim(df),
        "C. Lower-Low Reversal (>60m)": strat_lower_low_reversal(df, 60),
        "D. First Higher-High (>60m)":  strat_higher_high_reversal(df, 60),
        "E. Filtered ORB (vol×2, 15m)": strat_orb_breakout_vol_filter(df, 15),
    }

    for name, trade in strategies.items():
        print(f"\n--- {name} ---")
        if trade is None:
            print("  No setup triggered today.")
            continue
        print(f"  Entry:     {trade.entry_time}  @ ${trade.entry_price:.3f}")
        print(f"  Exit:      {trade.exit_time}  @ ${trade.exit_price:.3f}")
        print(f"  Bars held: {trade.bars_held} ({trade.bars_held*5} min)")
        print(f"  Net PnL:   {trade.pnl_pct_net:+.2f}%   Halt-risk: "
              f"{'YES' if trade.halt_risk else 'no'}")
        print(f"  Reason:    {trade.reason}")

    print("\n--- Summary ---")
    print(f"{'Strategy':<35} {'Net PnL':>9} {'Bars':>5} {'Halt':>5}")
    for name, trade in strategies.items():
        if trade is None:
            print(f"{name:<35} {'no setup':>9} {'—':>5} {'—':>5}")
        else:
            print(f"{name:<35} {trade.pnl_pct_net:>+8.2f}% "
                  f"{trade.bars_held:>5} {'YES' if trade.halt_risk else 'no':>5}")


if __name__ == "__main__":
    main()
