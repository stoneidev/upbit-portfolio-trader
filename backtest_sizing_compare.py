"""
Test the user's hypothesis that grid DCA (30/30/40) is asymmetric:
- Up → only L1 (30%) deployed → small absolute gain
- Down → full deployment → large absolute loss

We compare four sizing strategies on the same signals:
  A) Current grid 30/30/40
  B) Reverse grid     60/30/10  (front-load on entry, less DCA)
  C) Full upfront    100/0/0    (no grids, all-in at L1)
  D) Equal grid       33/33/34

For each: report total return, MDD, Sharpe, win rate, avg ABS contribution
per trade by levels-filled bucket. The "abs contribution" is what actually
moves the equity curve, accounting for the deployed fraction.
"""
import os, math, time, json
from dataclasses import dataclass

import numpy as np
import pandas as pd

import backtest_strategy as bs


@dataclass
class SizeCfg:
    z_thresh: float = -1.2
    tp_pct: float = 0.5
    trail_pct: float = 0.2
    sl_pct: float = -2.0
    grid_l2_pct: float = -1.0
    grid_l3_pct: float = -2.0
    fee_per_side: float = 0.0005
    slippage_per_side: float = 0.0005
    weights: tuple = (0.30, 0.30, 0.40)
    initial_balance: float = 500_000.0


def simulate(df: pd.DataFrame, cfg: SizeCfg, start_idx: int = 100) -> dict:
    closes = df["close"].values
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    times = df["time"].astype(str).values
    z = df["z_score"].values

    fee, slip = cfg.fee_per_side, cfg.slippage_per_side
    balance = cfg.initial_balance
    eq = []
    trades = []
    in_trade = False
    entry_idx = 0
    L1 = 0.0
    wf = [0.0, 0.0, 0.0]
    cb = 0.0
    tw = 0.0
    trailing = False
    peak = 0.0

    for i in range(start_idx, len(df) - 1):
        eq.append((times[i], balance))
        if not in_trade:
            if not math.isnan(z[i]) and z[i] < cfg.z_thresh:
                fill = opens[i + 1] * (1 + slip) if cfg.weights[0] > 0 else opens[i + 1]
                # Use limit-style entry (no slip) for fairness across strategies:
                fill = opens[i + 1]
                L1 = fill
                wf = [cfg.weights[0], 0.0, 0.0]
                cb = fill * cfg.weights[0]
                tw = cfg.weights[0]
                in_trade = True
                entry_idx = i + 1
                trailing = False
                peak = 0.0
            continue
        if i < entry_idx:
            continue

        l2 = L1 * (1 + cfg.grid_l2_pct / 100.0)
        l3 = L1 * (1 + cfg.grid_l3_pct / 100.0)
        if cfg.weights[1] > 0 and wf[1] == 0.0 and lows[i] <= l2:
            cb += l2 * cfg.weights[1]
            wf[1] = cfg.weights[1]
            tw += cfg.weights[1]
        if cfg.weights[2] > 0 and wf[2] == 0.0 and lows[i] <= l3:
            cb += l3 * cfg.weights[2]
            wf[2] = cfg.weights[2]
            tw += cfg.weights[2]

        avg = cb / tw
        sl_p = avg * (1 + cfg.sl_pct / 100.0)
        tp_a = avg * (1 + cfg.tp_pct / 100.0)
        bh, bl = highs[i], lows[i]
        ep, reason, is_mkt = None, None, False
        if not trailing:
            if bl <= sl_p:
                ep = sl_p
                reason = "SL"
                is_mkt = True
            elif bh >= tp_a:
                trailing = True
                peak = max(bh, tp_a)
        else:
            peak = max(peak, bh)
            stop = peak * (1 - cfg.trail_pct / 100.0)
            floor = avg * (1 + (cfg.tp_pct - cfg.trail_pct) / 100.0)
            stop = max(stop, floor)
            if bl <= stop:
                ep = stop
                reason = "TRAIL"

        if ep is not None:
            sell = ep * (1 - slip) if is_mkt else ep
            ret = (sell - avg) / avg - fee * 2
            balance = balance * (1 - tw) + balance * tw * (1 + ret)
            levels = sum(1 for w in wf if w > 0)
            # Absolute contribution to total equity (in pct points)
            abs_contrib = tw * ret * 100
            trades.append({
                "exit_reason": reason, "pnl_on_deployed_pct": ret * 100,
                "weight_used": tw, "levels_filled": levels,
                "abs_contrib_pct": abs_contrib,  # what really moved equity
            })
            in_trade = False
            trailing = False
            peak = 0.0
            wf = [0.0, 0.0, 0.0]
            tw = 0.0
            cb = 0.0
    eq.append((times[-1], balance))
    return {"final": balance, "trades": trades, "equity_curve": eq}


def metrics(r, df, cfg):
    final = r["final"]
    initial = cfg.initial_balance
    eq = pd.DataFrame(r["equity_curve"], columns=["t", "e"])
    eq["t"] = pd.to_datetime(eq["t"])
    eq = eq.set_index("t")
    eq = eq[~eq.index.duplicated(keep="last")]
    rmax = eq["e"].cummax()
    dd = (eq["e"] / rmax - 1) * 100
    daily = eq["e"].resample("1D").last().ffill()
    dr = daily.pct_change().dropna()
    sharpe = (dr.mean() / dr.std()) * math.sqrt(365) if dr.std() > 0 else 0
    trades = r["trades"]
    wr = sum(1 for t in trades if t["abs_contrib_pct"] > 0) / len(trades) * 100 if trades else 0
    return {
        "final": final,
        "total_return_pct": (final / initial - 1) * 100,
        "mdd_pct": float(dd.min()) if len(dd) else 0.0,
        "sharpe": float(sharpe),
        "trades": len(trades),
        "win_rate_pct": wr,
    }


def by_levels(trades):
    rows = []
    for lvl in [1, 2, 3]:
        sub = [t for t in trades if t["levels_filled"] == lvl]
        if not sub:
            continue
        rows.append({
            "levels_filled": lvl,
            "n": len(sub),
            "avg_pnl_on_deployed_pct": np.mean([t["pnl_on_deployed_pct"] for t in sub]),
            "avg_abs_contrib_pct": np.mean([t["abs_contrib_pct"] for t in sub]),
            "total_abs_contrib_pct": np.sum([t["abs_contrib_pct"] for t in sub]),
            "win_rate_pct": sum(1 for t in sub if t["abs_contrib_pct"] > 0) / len(sub) * 100,
        })
    return pd.DataFrame(rows)


def main():
    df = bs.fetch_candles("KRW-XRP", 35040)
    df_ind = bs.compute_indicators(df)

    strategies = {
        "A. Grid 30/30/40 (current)": (0.30, 0.30, 0.40),
        "B. Reverse 60/30/10":         (0.60, 0.30, 0.10),
        "C. Full upfront 100/0/0":     (1.00, 0.00, 0.00),
        "D. Equal grid 33/33/34":      (0.33, 0.33, 0.34),
    }

    print("\n=== KRW-XRP, 1 year (15m), tp=0.5, sl=-2 ===\n")
    for name, weights in strategies.items():
        cfg = SizeCfg(weights=weights)
        r = simulate(df_ind, cfg)
        m = metrics(r, df, cfg)
        print(f"{name}")
        print(f"  Return={m['total_return_pct']:+.2f}%  Final={m['final']:>12,.0f}  "
              f"MDD={m['mdd_pct']:.2f}%  Sharpe={m['sharpe']:.2f}  "
              f"Trades={m['trades']}  Win={m['win_rate_pct']:.1f}%")
        bl = by_levels(r["trades"])
        if not bl.empty:
            print("  Per-level breakdown:")
            print(bl.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print()

    # Same for ETH and BTC
    for sym in ["KRW-ETH", "KRW-BTC"]:
        print(f"\n=== {sym}, 1 year (15m) ===\n")
        df = bs.fetch_candles(sym, 35040)
        df_ind = bs.compute_indicators(df)
        for name, weights in strategies.items():
            cfg = SizeCfg(weights=weights)
            r = simulate(df_ind, cfg)
            m = metrics(r, df, cfg)
            print(f"{name}: ret={m['total_return_pct']:+.2f}% MDD={m['mdd_pct']:.2f}% "
                  f"Sharpe={m['sharpe']:.2f} Trades={m['trades']} Win={m['win_rate_pct']:.1f}%")


if __name__ == "__main__":
    main()
