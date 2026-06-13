"""
Validate the TP=0.3% finding with two stress tests:
1. Slippage sensitivity (0.05%, 0.10%, 0.15% per side)
2. WFO with z fitted on rolling IS, applied to OOS — same as base WFO but with
   TP=0.3 and trail=0.12.
"""
import os, json, time
from dataclasses import asdict

import numpy as np
import pandas as pd

import backtest_strategy as bs


def slippage_sensitivity():
    print("\n=== TP=0.3 SLIPPAGE SENSITIVITY (z fit best per market, fixed) ===")
    results = []
    for slip in [0.0003, 0.0005, 0.0008, 0.0010, 0.0015]:
        for market in ["KRW-XRP", "KRW-ETH", "KRW-BTC"]:
            df = bs.fetch_candles(market, 35040)
            best = None
            for z in [-1.0, -1.2, -1.5, -1.8, -2.0]:
                cfg = bs.StrategyConfig(
                    z_thresh=z, tp_pct=0.3, trail_pct=0.12, sl_pct=-2.0,
                    slippage_per_side=slip,
                )
                df_ind = bs.compute_indicators(df)
                res = bs.simulate(df_ind, cfg, start_idx=100)
                m = bs.compute_metrics(res, df, cfg)
                m["market"] = market
                m["slippage"] = slip
                m["z_thresh"] = z
                if best is None or m["total_return_pct"] > best["total_return_pct"]:
                    best = m
            results.append(best)

    df_res = pd.DataFrame(results)
    cols = ["market", "slippage", "z_thresh", "total_return_pct", "alpha_vs_bh_pct",
            "mdd_pct", "sharpe", "trades", "win_rate_pct", "profit_factor"]
    print(df_res[cols].round(2).to_string(index=False))
    return df_res


def wfo_with_tp(tp_pct=0.3, trail_pct=0.12):
    """Run rolling 4w IS / 4w OOS WFO with the new TP."""
    print(f"\n=== WFO WITH tp={tp_pct} trail={trail_pct} (4w IS / 4w OOS) ===")
    results = []
    for market in ["KRW-XRP", "KRW-ETH", "KRW-BTC"]:
        df = bs.fetch_candles(market, 35040)
        cfg_base = bs.StrategyConfig(
            tp_pct=tp_pct, trail_pct=trail_pct, sl_pct=-2.0,
            slippage_per_side=0.0005,  # baseline
        )
        r = bs.run_wfo(df, cfg_base)
        m = bs.compute_metrics(r, df, cfg_base)
        m["market"] = market
        m["chosen_zs"] = r.get("chosen_zs")
        results.append(m)
        print(f"  {market}: ret={m['total_return_pct']:+.2f}%  bh={m['buy_and_hold_pct']:+.2f}%  "
              f"alpha={m['alpha_vs_bh_pct']:+.2f}%  MDD={m['mdd_pct']:.2f}%  "
              f"Sharpe={m['sharpe']:.2f}  trades={m['trades']}  win={m['win_rate_pct']:.1f}%  "
              f"PF={m['profit_factor']:.2f}  zs={r.get('chosen_zs')}")
    return results


def main():
    df_slip = slippage_sensitivity()
    wfo_results = wfo_with_tp(0.3, 0.12)

    # also compare WFO at tp=0.5 (status quo) for direct delta
    print("\n=== BASELINE WFO at tp=0.5 (status quo, for comparison) ===")
    base_results = []
    for market in ["KRW-XRP", "KRW-ETH", "KRW-BTC"]:
        df = bs.fetch_candles(market, 35040)
        cfg_base = bs.StrategyConfig(tp_pct=0.5, trail_pct=0.2, sl_pct=-2.0)
        r = bs.run_wfo(df, cfg_base)
        m = bs.compute_metrics(r, df, cfg_base)
        m["market"] = market
        base_results.append(m)
        print(f"  {market}: ret={m['total_return_pct']:+.2f}%  alpha={m['alpha_vs_bh_pct']:+.2f}%  "
              f"MDD={m['mdd_pct']:.2f}%  Sharpe={m['sharpe']:.2f}  trades={m['trades']}")

    print("\n=== DELTA: WFO@tp=0.3 vs WFO@tp=0.5 ===")
    for new, old in zip(wfo_results, base_results):
        d_ret = new["total_return_pct"] - old["total_return_pct"]
        d_mdd = new["mdd_pct"] - old["mdd_pct"]
        d_sh = new["sharpe"] - old["sharpe"]
        print(f"  {new['market']}: ΔReturn={d_ret:+.2f}%  ΔMDD={d_mdd:+.2f}%  ΔSharpe={d_sh:+.2f}")

    out = os.path.join(bs.RESULTS_DIR, f"tp03_validate_{int(time.time())}.json")
    with open(out, "w") as f:
        json.dump({
            "slippage": df_slip.to_dict(orient="records"),
            "wfo_tp03": wfo_results,
            "wfo_tp05_baseline": base_results,
        }, f, indent=2, default=str)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
