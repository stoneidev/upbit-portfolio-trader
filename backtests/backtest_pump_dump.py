"""
Pump-and-dump intraday strategies on SLGB June 9, 2026.

Implements four patterns from the previous answer:
  A) Opening Range Breakdown Fade (SHORT) — fade the breakdown of 09:30-10:00
     range; cover at EOD or trailing stop.
  B) First Red Candle Fade (SHORT) — after parabolic up, short the bar AFTER the
     first red candle that follows a series of green ones; cover at EOD/trail.
  C) HOD Failure (SHORT) — once the high-of-day has not been broken for N bars
     (and price < HOD), short; cover at EOD/trail.
  D) VWAP Reclaim Long (LONG) — after the dump, when price reclaims VWAP from
     below with a green bar, go long; exit at EOD or trailing stop.

Limitations modeled:
  - 5-min bar resolution (entry/exit at bar close, with optional next-bar open).
  - Slippage 0.3% per side (pump-day micro-cap spreads are wide).
  - Borrow-rate fee for shorts: 5% annualized intraday cost (rough estimate).
  - No halts modeled — flag in output if the strategy would have been
    interrupted by an LULD-like event (>10% intra-bar move).
  - All trades closed by 15:50 ET (10 min before close, intraday-only).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional, List

import pandas as pd
import yfinance as yf


SLIPPAGE = 0.003          # 0.3% per side
SHORT_BORROW_APR = 0.05   # 5% annualized for the half-day held
EOD_CUTOFF = pd.Timestamp("15:50").time()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_intraday(symbol: str, day: str) -> pd.DataFrame:
    raw = yf.download(symbol, start=day, end=pd.Timestamp(day) + pd.Timedelta(days=1),
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

    # Cumulative VWAP from session open
    typical = (df["h"] + df["l"] + df["c"]) / 3
    cum_vol = df["v"].cumsum().replace(0, 1)
    df["vwap"] = (typical * df["v"]).cumsum() / cum_vol
    df["bar_change_pct"] = (df["c"] / df["c"].shift(1) - 1) * 100
    df["intra_range_pct"] = (df["h"] - df["l"]) / df["l"] * 100
    return df


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    strategy: str
    side: str              # "SHORT" or "LONG"
    entry_time: str
    entry_price: float
    exit_time: str
    exit_price: float
    bars_held: int
    pnl_pct: float         # net of slippage + borrow
    reason: str            # exit reason
    halt_risk: bool        # True if any held bar moved >10% intra-bar


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bar_index_at(df: pd.DataFrame, t: pd.Timestamp) -> int:
    return df["time"].searchsorted(t)


def gross_pct(side: str, entry: float, exit_: float) -> float:
    if side == "SHORT":
        return (entry - exit_) / entry * 100
    return (exit_ - entry) / entry * 100


def net_pct(side: str, entry: float, exit_: float, bars_held: int,
            slip: float = SLIPPAGE, borrow_apr: float = SHORT_BORROW_APR) -> float:
    g = gross_pct(side, entry, exit_)
    # Slippage on both fills (5-min bar simulator → optimistic model already)
    g -= slip * 100 * 2
    # Borrow fee (only short). 5min bar = 5/60/24/365 of a year per bar.
    if side == "SHORT":
        years_held = bars_held * 5 / 60 / 24 / 365
        g -= borrow_apr * years_held * 100
    return g


def force_close_eod(df: pd.DataFrame, side: str, entry_idx: int,
                    entry_price: float, strategy: str, reason: str) -> Trade:
    eod_mask = df["time"].dt.time >= EOD_CUTOFF
    if eod_mask.any():
        exit_idx = df.index[eod_mask][0]
    else:
        exit_idx = len(df) - 1
    exit_price = df.iloc[exit_idx]["c"]
    bars_held = exit_idx - entry_idx
    held = df.iloc[entry_idx:exit_idx + 1]
    halt = bool((held["intra_range_pct"] > 10).any())
    return Trade(
        strategy=strategy, side=side,
        entry_time=str(df.iloc[entry_idx]["time"]),
        entry_price=entry_price,
        exit_time=str(df.iloc[exit_idx]["time"]),
        exit_price=exit_price,
        bars_held=bars_held,
        pnl_pct=net_pct(side, entry_price, exit_price, bars_held),
        reason=reason,
        halt_risk=halt,
    )


# ---------------------------------------------------------------------------
# Strategy A: Opening Range Breakdown Fade (SHORT)
# ---------------------------------------------------------------------------

def strat_orb_breakdown(df: pd.DataFrame, range_minutes: int = 30) -> Optional[Trade]:
    range_end = df.iloc[0]["time"] + pd.Timedelta(minutes=range_minutes)
    rng = df[df["time"] < range_end]
    if rng.empty:
        return None
    or_low = rng["l"].min()
    or_high = rng["h"].max()
    rest = df[df["time"] >= range_end].reset_index(drop=True)
    # Look for first close below or_low
    for i, row in rest.iterrows():
        if row["c"] < or_low:
            entry_idx = bar_index_at(df, row["time"])
            entry_price = row["c"] * (1 - SLIPPAGE)  # short fill (pessimistic)
            return force_close_eod(df, "SHORT", entry_idx, entry_price,
                                   "ORB Breakdown Fade",
                                   f"Closed below OR low ${or_low:.3f}; ORH=${or_high:.3f}; covered EOD")
    return None


# ---------------------------------------------------------------------------
# Strategy B: First Red Candle Fade after parabolic up
# ---------------------------------------------------------------------------

def strat_first_red_fade(df: pd.DataFrame, min_consec_green: int = 3) -> Optional[Trade]:
    """Find a parabolic run (≥N green bars) then short the bar AFTER first red bar."""
    consec_green = 0
    for i in range(1, len(df)):
        bar = df.iloc[i]
        is_green = bar["c"] > bar["o"]
        is_red = bar["c"] < bar["o"]
        if is_green:
            consec_green += 1
        elif is_red and consec_green >= min_consec_green:
            # Short on the open of the next bar
            if i + 1 >= len(df):
                return None
            entry_bar = df.iloc[i + 1]
            entry_idx = i + 1
            entry_price = entry_bar["o"] * (1 - SLIPPAGE)
            return force_close_eod(df, "SHORT", entry_idx, entry_price,
                                   "First Red Fade",
                                   f"After {consec_green} green bars, first red at {df.iloc[i]['time']}")
        else:
            consec_green = 0
    return None


# ---------------------------------------------------------------------------
# Strategy C: HOD Failure (SHORT)
# ---------------------------------------------------------------------------

def strat_hod_failure(df: pd.DataFrame, fail_bars: int = 6) -> Optional[Trade]:
    """Short once HOD has not been retaken for `fail_bars` consecutive bars."""
    hod_idx = 0
    bars_since_hod = 0
    for i in range(1, len(df)):
        if df.iloc[i]["h"] > df.iloc[hod_idx]["h"]:
            hod_idx = i
            bars_since_hod = 0
        else:
            bars_since_hod += 1
        if bars_since_hod == fail_bars:
            # Confirm price is below HOD (it always is by construction)
            entry_idx = i
            entry_price = df.iloc[i]["c"] * (1 - SLIPPAGE)
            return force_close_eod(df, "SHORT", entry_idx, entry_price,
                                   "HOD Failure",
                                   f"HOD ${df.iloc[hod_idx]['h']:.3f} at {df.iloc[hod_idx]['time']}, "
                                   f"failed for {fail_bars} bars")
    return None


# ---------------------------------------------------------------------------
# Strategy D: VWAP Reclaim Long
# ---------------------------------------------------------------------------

def strat_vwap_reclaim_long(df: pd.DataFrame) -> Optional[Trade]:
    """After at least one bar below VWAP, enter long when price closes back above
    VWAP with a green bar."""
    saw_below = False
    for i in range(1, len(df)):
        bar = df.iloc[i]
        if bar["c"] < bar["vwap"]:
            saw_below = True
        elif saw_below and bar["c"] > bar["vwap"] and bar["c"] > bar["o"]:
            entry_idx = i
            entry_price = bar["c"] * (1 + SLIPPAGE)
            return force_close_eod(df, "LONG", entry_idx, entry_price,
                                   "VWAP Reclaim",
                                   f"Reclaimed VWAP ${bar['vwap']:.3f} with green bar at {bar['time']}")
    return None


# ---------------------------------------------------------------------------
# Run on SLGB June 9
# ---------------------------------------------------------------------------

def main():
    symbol = "SLGB"
    day = "2026-06-09"
    df = load_intraday(symbol, day)
    print(f"\n=== {symbol} {day} — pump/dump intraday strategy comparison ===")
    print(f"Bars: {len(df)}  Open: ${df['o'].iloc[0]:.3f}  HOD: ${df['h'].max():.3f}  "
          f"LOD: ${df['l'].min():.3f}  Close: ${df['c'].iloc[-1]:.3f}")
    big_bars = (df["intra_range_pct"] > 10).sum()
    print(f"Bars with >10% intra-range (LULD halt risk): {big_bars}")

    strategies = {
        "A. ORB Breakdown Fade (SHORT)": strat_orb_breakdown(df, 30),
        "B. First Red Candle Fade (SHORT)": strat_first_red_fade(df, 3),
        "C. HOD Failure (SHORT, 6 bars)": strat_hod_failure(df, 6),
        "D. VWAP Reclaim Long":           strat_vwap_reclaim_long(df),
    }

    rows = []
    for name, trade in strategies.items():
        if trade is None:
            print(f"\n--- {name} ---")
            print("  No setup triggered today.")
            rows.append({"strategy": name, "triggered": False})
            continue
        print(f"\n--- {name} ---")
        print(f"  Side:        {trade.side}")
        print(f"  Entry:       {trade.entry_time}  @ ${trade.entry_price:.3f}")
        print(f"  Exit:        {trade.exit_time}  @ ${trade.exit_price:.3f}")
        print(f"  Bars held:   {trade.bars_held} ({trade.bars_held*5} min)")
        gross = gross_pct(trade.side, trade.entry_price, trade.exit_price)
        print(f"  Gross PnL:   {gross:+.2f}%")
        print(f"  Net PnL:     {trade.pnl_pct:+.2f}%  (slip {SLIPPAGE*100:.1f}%/side both fills"
              + (", + 5% APR borrow)" if trade.side == "SHORT" else ")"))
        print(f"  Halt risk:   {'YES — held bar moved >10% intra-bar' if trade.halt_risk else 'no'}")
        print(f"  Reason:      {trade.reason}")
        rows.append({"strategy": name, "triggered": True, **asdict(trade)})

    print("\n--- Summary ---")
    print(f"{'Strategy':<40} {'Side':<6} {'Net PnL':>9} {'Halt':>5}")
    for r in rows:
        if r["triggered"]:
            print(f"{r['strategy']:<40} {r['side']:<6} {r['pnl_pct']:>+8.2f}% {'YES' if r['halt_risk'] else 'no':>5}")
        else:
            print(f"{r['strategy']:<40} {'—':<6} {'no setup':>9} {'—':>5}")

    return rows


if __name__ == "__main__":
    main()
