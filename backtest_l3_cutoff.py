"""
Add a time-based cutoff after Level-3 fill (worst-case grid).
Hypothesis: L3-filled trades currently lose -0.46%/trade on avg (XRP),
contributing -73%p total. If we cut them earlier — at BEP or small loss —
we should improve net return AND reduce MDD.

Cutoff rule:
  After L3 fill, start a timer. If price hasn't reached avg_price * (1 + bep_thresh)
  within `cutoff_bars` bars, force close at current price (market exit).

Sweep cutoff_bars ∈ {8, 16, 32, 64} and bep_thresh ∈ {-0.2%, 0%, +0.1%}.
Compare against baseline (no cutoff) on KRW-XRP, KRW-ETH, KRW-BTC.
"""
import os, math, time, json
from dataclasses import dataclass

import numpy as np
import pandas as pd

import backtest_strategy as bs


@dataclass
class L3Cfg:
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
    cutoff_bars: int = 0           # 0 = disabled
    bep_thresh_pct: float = 0.0    # exit threshold relative to avg (e.g. 0.0 = BEP)


def simulate(df: pd.DataFrame, cfg: L3Cfg, start_idx: int = 100) -> dict:
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
    l3_fill_idx = -1   # bar where L3 was filled

    for i in range(start_idx, len(df) - 1):
        eq.append((times[i], balance))
        if not in_trade:
            if not math.isnan(z[i]) and z[i] < cfg.z_thresh:
                fill = opens[i + 1]   # limit-style entry
                L1 = fill
                wf = [cfg.weights[0], 0.0, 0.0]
                cb = fill * cfg.weights[0]
                tw = cfg.weights[0]
                in_trade = True
                entry_idx = i + 1
                trailing = False
                peak = 0.0
                l3_fill_idx = -1
            continue
        if i < entry_idx:
            continue

        l2 = L1 * (1 + cfg.grid_l2_pct / 100.0)
        l3 = L1 * (1 + cfg.grid_l3_pct / 100.0)
        if wf[1] == 0.0 and lows[i] <= l2:
            cb += l2 * cfg.weights[1]
            wf[1] = cfg.weights[1]
            tw += cfg.weights[1]
        if wf[2] == 0.0 and lows[i] <= l3:
            cb += l3 * cfg.weights[2]
            wf[2] = cfg.weights[2]
            tw += cfg.weights[2]
            l3_fill_idx = i

        avg = cb / tw
        sl_p = avg * (1 + cfg.sl_pct / 100.0)
        tp_a = avg * (1 + cfg.tp_pct / 100.0)
        bh, bl = highs[i], lows[i]
        ep, reason, is_mkt = None, None, False

        # L3 cutoff check (only when L3 has been filled and trail not yet active)
        cutoff_triggered = False
        if (cfg.cutoff_bars > 0 and l3_fill_idx >= 0 and not trailing
                and (i - l3_fill_idx) >= cfg.cutoff_bars):
            cutoff_price = avg * (1 + cfg.bep_thresh_pct / 100.0)
            if closes[i] >= cutoff_price:
                # We've recovered to BEP+, force exit at current close
                ep = closes[i]
                reason = "L3_CUTOFF"
                is_mkt = True
                cutoff_triggered = True

        if not cutoff_triggered:
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
            trades.append({
                "exit_reason": reason,
                "pnl_on_deployed_pct": ret * 100,
                "weight_used": tw,
                "levels_filled": levels,
                "abs_contrib_pct": tw * ret * 100,
                "bars_held": i - entry_idx,
            })
            in_trade = False
            trailing = False
            peak = 0.0
            wf = [0.0, 0.0, 0.0]
            tw = 0.0
            cb = 0.0
            l3_fill_idx = -1
    eq.append((times[-1], balance))
    return {"final": balance, "trades": trades, "equity_curve": eq}


def metrics(r, cfg):
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
    l3_trades = [t for t in trades if t["levels_filled"] == 3]
    l3_avg = np.mean([t["abs_contrib_pct"] for t in l3_trades]) if l3_trades else 0
    cutoff_trades = [t for t in trades if t["exit_reason"] == "L3_CUTOFF"]
    cutoff_avg = np.mean([t["abs_contrib_pct"] for t in cutoff_trades]) if cutoff_trades else 0
    return {
        "total_return_pct": (final / initial - 1) * 100,
        "mdd_pct": float(dd.min()) if len(dd) else 0.0,
        "sharpe": float(sharpe),
        "trades": len(trades),
        "win_rate_pct": wr,
        "l3_count": len(l3_trades),
        "l3_avg_abs_contrib_pct": float(l3_avg),
        "cutoff_count": len(cutoff_trades),
        "cutoff_avg_abs_contrib_pct": float(cutoff_avg),
    }


def main():
    bars = 35040  # 1 year of 15m
    print(f"\n=== L3 cutoff sweep, 1y, tp=0.5, sl=-2 ===\n")

    for sym in ["KRW-XRP", "KRW-ETH", "KRW-BTC"]:
        df = bs.fetch_candles(sym, bars)
        df_ind = bs.compute_indicators(df)

        # Baseline (no cutoff)
        cfg_base = L3Cfg(cutoff_bars=0)
        r = simulate(df_ind, cfg_base)
        m_base = metrics(r, cfg_base)
        print(f"{sym} baseline (no cutoff): "
              f"ret={m_base['total_return_pct']:+.2f}% MDD={m_base['mdd_pct']:.2f}% "
              f"Sharpe={m_base['sharpe']:.2f} L3n={m_base['l3_count']} "
              f"L3avg={m_base['l3_avg_abs_contrib_pct']:+.3f}%")

        # Sweep
        rows = []
        for cb in [8, 16, 32, 64, 96]:
            for bep in [-0.2, 0.0, 0.1, 0.2]:
                cfg = L3Cfg(cutoff_bars=cb, bep_thresh_pct=bep)
                r = simulate(df_ind, cfg)
                m = metrics(r, cfg)
                m["cutoff_bars"] = cb
                m["bep_pct"] = bep
                rows.append(m)
        df_r = pd.DataFrame(rows)
        cols = ["cutoff_bars", "bep_pct", "total_return_pct", "mdd_pct", "sharpe",
                "trades", "win_rate_pct", "l3_count", "l3_avg_abs_contrib_pct",
                "cutoff_count", "cutoff_avg_abs_contrib_pct"]
        print(df_r[cols].round(3).to_string(index=False))

        # Best by sharpe
        best_sharpe = df_r.iloc[df_r["sharpe"].idxmax()]
        print(f"  Best by Sharpe: cutoff={best_sharpe['cutoff_bars']:.0f}bars  "
              f"bep={best_sharpe['bep_pct']:.2f}%  ret={best_sharpe['total_return_pct']:+.2f}%  "
              f"MDD={best_sharpe['mdd_pct']:.2f}%  Sharpe={best_sharpe['sharpe']:.2f}")
        # Best by return
        best_ret = df_r.iloc[df_r["total_return_pct"].idxmax()]
        print(f"  Best by Return: cutoff={best_ret['cutoff_bars']:.0f}bars  "
              f"bep={best_ret['bep_pct']:.2f}%  ret={best_ret['total_return_pct']:+.2f}%  "
              f"MDD={best_ret['mdd_pct']:.2f}%  Sharpe={best_ret['sharpe']:.2f}")
        print()


if __name__ == "__main__":
    main()
