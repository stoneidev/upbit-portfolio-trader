"""
Sweep WFO in-sample (IS) window length to test the user's hypothesis:
shorter IS = more market-sensitive learning. Trade-off: smaller sample
= noisier z choice.

Tests 1w / 2w / 4w(current) / 8w / 12w IS, with OOS = 4w fixed.
Reports total return, MDD, Sharpe, z-switching frequency, plus a focused
look at how each window adapted to June 2026.
"""
import os, math, time, json
from dataclasses import asdict

import numpy as np
import pandas as pd

import backtest_strategy as bs


def run_wfo_with_window(df, base_cfg, is_bars, oos_bars=4*7*96,
                        z_candidates=(-1.0, -1.2, -1.5, -1.8, -2.0)):
    df_ind = bs.compute_indicators(df).reset_index(drop=True)
    n = len(df_ind)
    pos = 200 + is_bars
    balance = base_cfg.initial_balance
    all_trades = []
    eqc = []
    chosen = []
    chosen_times = []
    while pos + oos_bars <= n:
        is_w = df_ind.iloc[pos - is_bars: pos].reset_index(drop=True)
        oos_w = df_ind.iloc[pos: pos + oos_bars].reset_index(drop=True)
        best_z, best_ret = base_cfg.z_thresh, -1e9
        for z in z_candidates:
            cfg_try = bs.StrategyConfig(**{**base_cfg.__dict__, "z_thresh": z, "initial_balance": 500_000.0})
            r = bs.simulate(is_w, cfg_try, start_idx=100)
            ret = r["final_balance"] - cfg_try.initial_balance
            if ret > best_ret:
                best_ret, best_z = ret, z
        chosen.append(best_z)
        chosen_times.append(str(df_ind.iloc[pos]["time"]))
        cfg_oos = bs.StrategyConfig(**{**base_cfg.__dict__, "z_thresh": best_z, "initial_balance": balance})
        r = bs.simulate(oos_w, cfg_oos, start_idx=0)
        balance = r["final_balance"]
        all_trades.extend(r["trades"])
        eqc.extend(r["equity_curve"])
        pos += oos_bars
    return {
        "final_balance": balance,
        "trades": all_trades,
        "equity_curve": eqc,
        "chosen_zs": chosen,
        "chosen_times": chosen_times,
    }


def metrics(result, df, cfg):
    eq = pd.DataFrame(result["equity_curve"], columns=["t", "e"])
    eq["t"] = pd.to_datetime(eq["t"])
    eq = eq.set_index("t")
    eq = eq[~eq.index.duplicated(keep="last")]
    final = result["final_balance"]
    initial = cfg.initial_balance
    total = (final / initial - 1) * 100
    rmax = eq["e"].cummax()
    dd = (eq["e"] / rmax - 1) * 100
    daily = eq["e"].resample("1D").last().ffill()
    dr = daily.pct_change().dropna()
    sharpe = (dr.mean() / dr.std()) * math.sqrt(365) if dr.std() > 0 else 0
    trades = result["trades"]
    wr = sum(1 for t in trades if t.pnl_pct > 0) / len(trades) * 100 if trades else 0
    zs = result["chosen_zs"]
    switches = sum(1 for i in range(1, len(zs)) if zs[i] != zs[i-1])
    return {
        "total_return_pct": total,
        "final": final,
        "mdd_pct": float(dd.min()),
        "sharpe": float(sharpe),
        "trades": len(trades),
        "win_rate_pct": wr,
        "n_windows": len(zs),
        "z_switches": switches,
        "z_switch_pct": switches / max(len(zs)-1, 1) * 100,
        "z_distribution": {z: zs.count(z) for z in sorted(set(zs))},
    }


def june_balance_change(result, initial):
    """Trace balance through June 2026."""
    bal = initial
    pre = bal
    last = bal
    trade_dates = []
    for t in result["trades"]:
        w = t.weight_used
        net = t.pnl_pct / 100
        bal = bal * (1 - w) + bal * w * (1 + net)
        et = pd.to_datetime(t.exit_time)
        if et < pd.Timestamp("2026-06-01"):
            pre = bal
        elif et < pd.Timestamp("2026-07-01"):
            last = bal
            trade_dates.append(et)
    if not trade_dates:
        return {"june_trades": 0, "balance_2026_06_01": pre, "balance_now": pre, "june_pnl_pct": 0}
    return {
        "june_trades": len(trade_dates),
        "balance_2026_06_01": pre,
        "balance_now": last,
        "june_pnl_pct": (last / pre - 1) * 100,
    }


def main():
    bars = 35040
    windows = {
        "1w IS":  1 * 7 * 96,
        "2w IS":  2 * 7 * 96,
        "4w IS (current)": 4 * 7 * 96,
        "8w IS":  8 * 7 * 96,
        "12w IS": 12 * 7 * 96,
    }

    for sym in ["KRW-XRP", "KRW-ETH"]:
        df = bs.fetch_candles(sym, bars)
        print(f"\n=== {sym}, 1y, OOS=4w fixed ===")
        rows = []
        for label, is_bars in windows.items():
            cfg = bs.StrategyConfig(tp_pct=0.5, trail_pct=0.2, sl_pct=-2.0, initial_balance=500_000.0)
            r = run_wfo_with_window(df, cfg, is_bars)
            m = metrics(r, df, cfg)
            j = june_balance_change(r, cfg.initial_balance)
            rows.append({
                "window": label,
                "ret%": round(m["total_return_pct"], 2),
                "mdd%": round(m["mdd_pct"], 2),
                "sharpe": round(m["sharpe"], 2),
                "trades": m["trades"],
                "win%": round(m["win_rate_pct"], 1),
                "windows": m["n_windows"],
                "z_switches": m["z_switches"],
                "z_switch%": round(m["z_switch_pct"], 1),
                "june_n": j["june_trades"],
                "june_pnl%": round(j["june_pnl_pct"], 2),
                "z_dist": m["z_distribution"],
            })
            print(f"  {label:<18}  ret={m['total_return_pct']:+7.2f}%  "
                  f"MDD={m['mdd_pct']:+6.2f}%  Sharpe={m['sharpe']:+5.2f}  "
                  f"trades={m['trades']:>4}  switches={m['z_switches']}/{m['n_windows']-1} ({m['z_switch_pct']:.0f}%)  "
                  f"June_pnl={j['june_pnl_pct']:+5.2f}%  z_dist={m['z_distribution']}")

        # Show the actual z-trajectory for current 4w + the shortest (1w)
        for label in ["1w IS", "4w IS (current)"]:
            is_bars = windows[label]
            cfg = bs.StrategyConfig(tp_pct=0.5, trail_pct=0.2, sl_pct=-2.0, initial_balance=500_000.0)
            r = run_wfo_with_window(df, cfg, is_bars)
            print(f"\n  {label} z-trajectory (last 12 windows):")
            for t, z in list(zip(r["chosen_times"], r["chosen_zs"]))[-12:]:
                print(f"    {t[:10]}  →  z = {z}")


if __name__ == "__main__":
    main()
