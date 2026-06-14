"""
Sweep L1 (initial entry) size while keeping the strategy fixed.
The user's hypothesis: since L1-only trades have ~100% win rate, putting
more capital at L1 should multiply absolute gains.

The counterforce: bigger L1 means
  (a) higher avg-price when grids fill (L2/L3 weights smaller, less DCA dilution)
  (b) bigger losses when SL hits (deployed % is the same in any all-grid case;
      it's the L1-only and L1+L2 cases where MORE capital is at risk)

Compares variants on KRW-XRP, KRW-ETH, KRW-BTC over 1y at tp=0.5.
"""
import math
import numpy as np
import pandas as pd

import backtest_strategy as bs
from backtest_sizing_compare import SizeCfg, simulate, metrics, by_levels


VARIANTS = {
    "30/30/40 (current)": (0.30, 0.30, 0.40),
    "40/30/30":           (0.40, 0.30, 0.30),
    "50/30/20":           (0.50, 0.30, 0.20),
    "50/25/25":           (0.50, 0.25, 0.25),
    "60/20/20":           (0.60, 0.20, 0.20),
    "70/20/10":           (0.70, 0.20, 0.10),
}


def main():
    for sym in ["KRW-XRP", "KRW-ETH", "KRW-BTC"]:
        df = bs.fetch_candles(sym, 35040)
        df_ind = bs.compute_indicators(df)
        print(f"\n=== {sym}, 1y, tp=0.5 ===\n")
        rows = []
        for name, w in VARIANTS.items():
            cfg = SizeCfg(weights=w)
            r = simulate(df_ind, cfg)
            m = metrics(r, df, cfg)
            bl = by_levels(r["trades"])
            l1 = bl[bl["levels_filled"] == 1].iloc[0] if (bl["levels_filled"] == 1).any() else None
            l2 = bl[bl["levels_filled"] == 2].iloc[0] if (bl["levels_filled"] == 2).any() else None
            l3 = bl[bl["levels_filled"] == 3].iloc[0] if (bl["levels_filled"] == 3).any() else None
            rows.append({
                "weights": name,
                "ret": m["total_return_pct"],
                "mdd": m["mdd_pct"],
                "sharpe": m["sharpe"],
                "trades": m["trades"],
                "win": m["win_rate_pct"],
                "L1_n": int(l1["n"]) if l1 is not None else 0,
                "L1_avg_abs": float(l1["avg_abs_contrib_pct"]) if l1 is not None else 0,
                "L1_total": float(l1["total_abs_contrib_pct"]) if l1 is not None else 0,
                "L3_n": int(l3["n"]) if l3 is not None else 0,
                "L3_avg_abs": float(l3["avg_abs_contrib_pct"]) if l3 is not None else 0,
                "L3_total": float(l3["total_abs_contrib_pct"]) if l3 is not None else 0,
            })
        df_r = pd.DataFrame(rows)
        print(df_r.to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
