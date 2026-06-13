"""
Rigorous backtest for the strategy implemented in upbit_portfolio_trader.py.

Goals:
1. Replicate the live entry rules as faithfully as possible from candle data alone
   (vol_power is real-time only; we therefore document its absence as a caveat).
2. Apply round-trip fees (entry + exit) at the correct rate (0.05% per side).
3. Backtest a full year of 15-min candles for KRW-XRP, KRW-ETH, KRW-BTC.
4. Run two regimes: fixed parameters and rolling WFO (4w in-sample / 4w out-of-sample).
5. Compare to buy-and-hold and report robust metrics (CAGR-equivalent, MDD, Sharpe,
   Profit Factor, Win Rate, trade count).

Caveats baked into the report:
- vol_power filter cannot be reconstructed from historical candles, so it is
  reported separately. Live edge could be smaller or larger than backtest.
- We model fills at the next bar's open after a signal, and use intra-bar high/low
  for grid fills, TP activation, and stop-outs. Same-bar TP+SL ties resolved
  pessimistically (assume SL hit first).
- Slippage is modeled as a small additional cost on each fill (configurable).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
import requests


UPBIT_API_URL = "https://api.upbit.com/v1"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "backtest_data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "backtest_results")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def fetch_candles(market: str, total: int, interval: str = "15") -> pd.DataFrame:
    os.makedirs(DATA_DIR, exist_ok=True)
    cache = os.path.join(DATA_DIR, f"{market}_{interval}m_{total}.csv")
    if os.path.exists(cache):
        df = pd.read_csv(cache)
        df["time"] = pd.to_datetime(df["time"])
        return df

    url = f"{UPBIT_API_URL}/candles/minutes/{interval}"
    all_rows = []
    to_time = None
    pulled = 0
    while pulled < total:
        params = {"market": market, "count": min(200, total - pulled)}
        if to_time:
            params["to"] = to_time
        for attempt in range(5):
            try:
                resp = requests.get(url, params=params, timeout=10)
                if resp.status_code == 429:
                    time.sleep(0.5)
                    continue
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as exc:
                print(f"[{market}] API error attempt {attempt+1}: {exc}", flush=True)
                time.sleep(1.0)
        else:
            raise RuntimeError(f"Failed to fetch candles for {market}")

        if not data:
            break
        all_rows.extend(data)
        oldest = data[-1]["candle_date_time_utc"]
        to_time = oldest.replace("T", " ")
        pulled = len(all_rows)
        print(f"[{market}] pulled={pulled}/{total} oldest={oldest}", flush=True)
        time.sleep(0.12)

    rows = []
    for c in all_rows:
        rows.append({
            "time": c["candle_date_time_kst"],
            "open": float(c["opening_price"]),
            "high": float(c["high_price"]),
            "low": float(c["low_price"]),
            "close": float(c["trade_price"]),
            "volume": float(c["candle_acc_trade_volume"]),
        })
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    df.to_csv(cache, index=False)
    return df


# ---------------------------------------------------------------------------
# Strategy simulator
# ---------------------------------------------------------------------------

@dataclass
class StrategyConfig:
    z_thresh: float = -1.2
    tp_pct: float = 0.5          # trailing activation
    trail_pct: float = 0.2       # trailing width
    sl_pct: float = -2.0         # stop loss vs avg
    grid_l2_pct: float = -1.0    # grid level 2 trigger relative to L1
    grid_l3_pct: float = -2.0    # grid level 3 trigger relative to L1
    fee_per_side: float = 0.0005 # Upbit standard fee
    slippage_per_side: float = 0.0005  # 0.05% slippage per fill (conservative)
    weights: tuple = (0.30, 0.30, 0.40)
    initial_balance: float = 500_000.0


@dataclass
class TradeRecord:
    entry_time: str
    exit_time: str
    entry_price: float
    avg_price: float
    exit_price: float
    weight_used: float            # 0.3 / 0.6 / 1.0
    levels_filled: int            # 1, 2, or 3
    exit_reason: str              # "TRAIL", "SL"
    pnl_pct: float                # net of fees, on capital actually deployed
    bars_held: int


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ma_100"] = df["close"].rolling(100).mean()
    df["std_100"] = df["close"].rolling(100).std()
    df["z_score"] = (df["close"] - df["ma_100"]) / (df["std_100"] + 1e-9)
    return df


def simulate(df: pd.DataFrame, cfg: StrategyConfig, start_idx: int = 100) -> dict:
    """Run a single bar-by-bar simulation. Entry at next bar open after signal."""
    closes = df["close"].values
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    times = df["time"].astype(str).values
    z = df["z_score"].values

    balance = cfg.initial_balance
    equity_curve = []  # (time, equity)
    trades: List[TradeRecord] = []

    in_trade = False
    entry_idx = 0
    level_1_price = 0.0
    weights_filled = [0.0, 0.0, 0.0]   # capital fraction filled at each level
    cost_basis = 0.0  # weighted price * fraction
    total_weight = 0.0
    trailing_active = False
    peak = 0.0

    for i in range(start_idx, len(df) - 1):
        # Mark to market (rough — uses close even if in trade)
        equity_curve.append((times[i], balance))

        if not in_trade:
            if not math.isnan(z[i]) and z[i] < cfg.z_thresh:
                fill = opens[i + 1] * (1 + cfg.slippage_per_side)
                level_1_price = fill
                weights_filled = [cfg.weights[0], 0.0, 0.0]
                cost_basis = fill * cfg.weights[0]
                total_weight = cfg.weights[0]
                in_trade = True
                entry_idx = i + 1
                trailing_active = False
                peak = 0.0
            continue

        # Skip the entry bar itself (we already booked the fill at its open)
        if i < entry_idx:
            continue

        l2_trigger = level_1_price * (1 + cfg.grid_l2_pct / 100.0)
        l3_trigger = level_1_price * (1 + cfg.grid_l3_pct / 100.0)

        if weights_filled[1] == 0.0 and lows[i] <= l2_trigger:
            fill = l2_trigger * (1 + cfg.slippage_per_side)
            cost_basis += fill * cfg.weights[1]
            weights_filled[1] = cfg.weights[1]
            total_weight += cfg.weights[1]
        if weights_filled[2] == 0.0 and lows[i] <= l3_trigger:
            fill = l3_trigger * (1 + cfg.slippage_per_side)
            cost_basis += fill * cfg.weights[2]
            weights_filled[2] = cfg.weights[2]
            total_weight += cfg.weights[2]

        avg_price = cost_basis / total_weight
        sl_price = avg_price * (1 + cfg.sl_pct / 100.0)
        tp_activation = avg_price * (1 + cfg.tp_pct / 100.0)

        bar_high = highs[i]
        bar_low = lows[i]

        exit_price = None
        exit_reason = None

        if not trailing_active:
            # Pessimistic: SL takes priority within the same bar
            if bar_low <= sl_price:
                exit_price = sl_price
                exit_reason = "SL"
            elif bar_high >= tp_activation:
                trailing_active = True
                peak = max(bar_high, tp_activation)
        else:
            peak = max(peak, bar_high)
            stop = peak * (1 - cfg.trail_pct / 100.0)
            floor = avg_price * (1 + (cfg.tp_pct - cfg.trail_pct) / 100.0)
            stop = max(stop, floor)
            if bar_low <= stop:
                exit_price = stop
                exit_reason = "TRAIL"

        if exit_price is not None:
            sell_price = exit_price * (1 - cfg.slippage_per_side)
            ret = (sell_price - avg_price) / avg_price
            fee_total = cfg.fee_per_side * 2.0  # buy + sell
            net_ret = ret - fee_total
            # Compound on deployed fraction
            balance = balance * (1 - total_weight) + balance * total_weight * (1 + net_ret)

            levels = sum(1 for w in weights_filled if w > 0)
            trades.append(TradeRecord(
                entry_time=str(times[entry_idx]),
                exit_time=str(times[i]),
                entry_price=level_1_price,
                avg_price=avg_price,
                exit_price=sell_price,
                weight_used=total_weight,
                levels_filled=levels,
                exit_reason=exit_reason,
                pnl_pct=net_ret * 100.0,
                bars_held=i - entry_idx,
            ))
            in_trade = False
            trailing_active = False
            peak = 0.0
            weights_filled = [0.0, 0.0, 0.0]
            total_weight = 0.0
            cost_basis = 0.0

    equity_curve.append((times[-1], balance))
    return {
        "final_balance": balance,
        "trades": trades,
        "equity_curve": equity_curve,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(result: dict, df: pd.DataFrame, cfg: StrategyConfig) -> dict:
    trades = result["trades"]
    eq = pd.DataFrame(result["equity_curve"], columns=["time", "equity"])
    eq["time"] = pd.to_datetime(eq["time"])
    eq = eq.set_index("time")

    final = result["final_balance"]
    initial = cfg.initial_balance
    total_return = (final / initial - 1) * 100.0

    # Time span in years (15-min bars)
    if len(eq) > 1:
        span_days = (eq.index[-1] - eq.index[0]).total_seconds() / 86400
    else:
        span_days = 0
    years = span_days / 365.0 if span_days > 0 else 1.0
    cagr = (final / initial) ** (1 / years) - 1 if years > 0 else 0
    cagr *= 100

    # Drawdown
    running_max = eq["equity"].cummax()
    dd = (eq["equity"] / running_max - 1) * 100.0
    mdd = dd.min() if len(dd) else 0.0

    # Daily returns from equity curve resampled
    daily_eq = eq["equity"].resample("1D").last().ffill()
    daily_ret = daily_eq.pct_change().dropna()
    if len(daily_ret) > 1 and daily_ret.std() > 0:
        sharpe = (daily_ret.mean() / daily_ret.std()) * math.sqrt(365)
    else:
        sharpe = 0.0

    # Trade-level
    if trades:
        wins = [t for t in trades if t.pnl_pct > 0]
        losses = [t for t in trades if t.pnl_pct <= 0]
        win_rate = len(wins) / len(trades) * 100
        avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0
        gross_profit = sum(t.pnl_pct for t in wins)
        gross_loss = -sum(t.pnl_pct for t in losses)
        pf = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
        avg_levels = np.mean([t.levels_filled for t in trades])
        avg_hold = np.mean([t.bars_held for t in trades])
    else:
        win_rate = avg_win = avg_loss = pf = avg_levels = avg_hold = 0

    # Buy & hold benchmark over the same window
    df_window = df[df["time"].between(eq.index[0], eq.index[-1])]
    if len(df_window) >= 2:
        bh_return = (df_window["close"].iloc[-1] / df_window["close"].iloc[0] - 1) * 100
    else:
        bh_return = 0.0

    return {
        "trades": len(trades),
        "final_balance": final,
        "total_return_pct": total_return,
        "cagr_pct": cagr,
        "mdd_pct": float(mdd),
        "sharpe": float(sharpe),
        "win_rate_pct": float(win_rate),
        "avg_win_pct": float(avg_win),
        "avg_loss_pct": float(avg_loss),
        "profit_factor": float(pf) if pf != float("inf") else 999.0,
        "avg_levels_filled": float(avg_levels),
        "avg_bars_held": float(avg_hold),
        "buy_and_hold_pct": float(bh_return),
        "alpha_vs_bh_pct": float(total_return - bh_return),
        "span_days": float(span_days),
    }


# ---------------------------------------------------------------------------
# WFO runner
# ---------------------------------------------------------------------------

def run_fixed(df: pd.DataFrame, cfg: StrategyConfig) -> dict:
    df_ind = compute_indicators(df)
    return simulate(df_ind, cfg)


def run_wfo(
    df: pd.DataFrame,
    base_cfg: StrategyConfig,
    is_bars: int = 4 * 7 * 96,   # 4 weeks of 15-min bars
    oos_bars: int = 4 * 7 * 96,
    z_candidates: tuple = (-1.0, -1.2, -1.5, -1.8, -2.0),
) -> dict:
    """Walk-forward: every oos_bars, refit on previous is_bars, then trade oos_bars."""
    df_ind = compute_indicators(df).reset_index(drop=True)
    n = len(df_ind)
    start = 200  # need indicator warmup
    pos = start + is_bars

    # Run a stitched simulation: we mutate cfg.z_thresh in segments and replay.
    # Simpler approach: run independent simulations per OOS window, then chain
    # by carrying balance forward.
    balance = base_cfg.initial_balance
    all_trades: List[TradeRecord] = []
    equity_curve = []
    chosen_zs = []

    while pos + oos_bars <= n:
        is_window = df_ind.iloc[pos - is_bars: pos].reset_index(drop=True)
        oos_window = df_ind.iloc[pos: pos + oos_bars].reset_index(drop=True)

        # Fit z on IS
        best_z = base_cfg.z_thresh
        best_ret = -1e9
        for z in z_candidates:
            cfg_try = StrategyConfig(**{**base_cfg.__dict__, "z_thresh": z, "initial_balance": 500_000.0})
            r = simulate(is_window, cfg_try, start_idx=100)
            ret = r["final_balance"] - cfg_try.initial_balance
            if ret > best_ret:
                best_ret = ret
                best_z = z
        chosen_zs.append(best_z)

        # Trade OOS with chosen z, carrying balance
        cfg_oos = StrategyConfig(**{**base_cfg.__dict__, "z_thresh": best_z, "initial_balance": balance})
        # OOS still needs warmup bars of indicators — we already computed them,
        # but iloc reset means start_idx=100 inside oos window misses earlier data.
        # Use a smaller start_idx since indicators are already populated.
        r = simulate(oos_window, cfg_oos, start_idx=0)
        balance = r["final_balance"]
        all_trades.extend(r["trades"])
        equity_curve.extend(r["equity_curve"])

        pos += oos_bars

    return {
        "final_balance": balance,
        "trades": all_trades,
        "equity_curve": equity_curve,
        "chosen_zs": chosen_zs,
    }


# ---------------------------------------------------------------------------
# CLI / entry
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markets", nargs="+", default=["KRW-XRP", "KRW-ETH", "KRW-BTC"])
    parser.add_argument("--bars", type=int, default=35040)  # ~1 year of 15m
    parser.add_argument("--mode", choices=["fixed", "wfo", "both"], default="both")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_rows = []

    for market in args.markets:
        print(f"\n=========== {market} ===========")
        df = fetch_candles(market, args.bars)
        print(f"Loaded {len(df)} bars: {df['time'].iloc[0]} → {df['time'].iloc[-1]}")

        base_cfg = StrategyConfig()

        if args.mode in ("fixed", "both"):
            r = run_fixed(df, base_cfg)
            m = compute_metrics(r, df, base_cfg)
            m["market"] = market
            m["mode"] = "fixed"
            summary_rows.append(m)
            print(f"[FIXED]  Return={m['total_return_pct']:+.2f}% B&H={m['buy_and_hold_pct']:+.2f}% "
                  f"Alpha={m['alpha_vs_bh_pct']:+.2f}% MDD={m['mdd_pct']:.2f}% "
                  f"Sharpe={m['sharpe']:.2f} Trades={m['trades']} Win={m['win_rate_pct']:.1f}% "
                  f"PF={m['profit_factor']:.2f}")

        if args.mode in ("wfo", "both"):
            r = run_wfo(df, base_cfg)
            m = compute_metrics(r, df, base_cfg)
            m["market"] = market
            m["mode"] = "wfo"
            m["chosen_zs"] = r.get("chosen_zs")
            summary_rows.append(m)
            print(f"[WFO]    Return={m['total_return_pct']:+.2f}% B&H={m['buy_and_hold_pct']:+.2f}% "
                  f"Alpha={m['alpha_vs_bh_pct']:+.2f}% MDD={m['mdd_pct']:.2f}% "
                  f"Sharpe={m['sharpe']:.2f} Trades={m['trades']} Win={m['win_rate_pct']:.1f}% "
                  f"PF={m['profit_factor']:.2f}  ZsPicked={r.get('chosen_zs')}")

    # Save
    out_path = os.path.join(RESULTS_DIR, f"summary_{int(time.time())}.json")
    with open(out_path, "w") as f:
        json.dump(summary_rows, f, indent=2, default=str)
    print(f"\nSaved summary → {out_path}")

    # Pretty table
    print("\n=== SUMMARY ===")
    cols = ["market", "mode", "total_return_pct", "buy_and_hold_pct", "alpha_vs_bh_pct",
            "mdd_pct", "sharpe", "trades", "win_rate_pct", "profit_factor"]
    df_summary = pd.DataFrame(summary_rows)[cols]
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()
