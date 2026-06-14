"""
Per-day trade count for the last 1 year (KRW-XRP, KRW-ETH).
Uses the WFO simulator (matches live behavior) with hybrid orders.
"""
import os
import pandas as pd

import backtest_strategy as bs


def main():
    bars = 35040  # 1 year of 15m
    out_dir = os.path.join(bs.RESULTS_DIR)
    os.makedirs(out_dir, exist_ok=True)

    summary = {}
    for sym in ["KRW-XRP", "KRW-ETH"]:
        df = bs.fetch_candles(sym, bars)
        cfg = bs.StrategyConfig(tp_pct=0.5, trail_pct=0.2, sl_pct=-2.0)
        r = bs.run_wfo(df, cfg)
        trades = r["trades"]

        rows = pd.DataFrame([{
            "exit_time": pd.to_datetime(t.exit_time),
            "exit_reason": t.exit_reason,
            "levels": t.levels_filled,
            "pnl_pct": t.pnl_pct,
        } for t in trades])
        rows["date"] = rows["exit_time"].dt.date

        daily = rows.groupby("date").agg(
            trades=("pnl_pct", "size"),
            wins=("pnl_pct", lambda s: (s > 0).sum()),
            losses=("pnl_pct", lambda s: (s <= 0).sum()),
            sl_count=("exit_reason", lambda s: (s == "SL").sum()),
            avg_pnl=("pnl_pct", "mean"),
            cum_pnl=("pnl_pct", "sum"),
            avg_levels=("levels", "mean"),
        ).round(3)

        out_path = os.path.join(out_dir, f"daily_trades_{sym}_1y.csv")
        daily.to_csv(out_path)

        print(f"\n=== {sym}  daily trade summary  (1y, {len(trades)} trades) ===")
        print(f"Window: {daily.index.min()} → {daily.index.max()}  ({len(daily)} active days)")
        print(f"Mean trades/day: {daily['trades'].mean():.2f}   Median: {daily['trades'].median()}   Max: {daily['trades'].max()}")

        # Distribution of trades-per-day
        dist = daily["trades"].value_counts().sort_index()
        print("\nDistribution of daily trade counts:")
        print(dist.to_string())

        # Days with 0 signals (need to fill from full date range)
        all_days = pd.date_range(daily.index.min(), daily.index.max(), freq="D").date
        zero_days = len(all_days) - len(daily)
        print(f"\nDays with 0 trades: {zero_days} / {len(all_days)} ({zero_days/len(all_days)*100:.1f}%)")

        # Top 10 busiest days
        print("\nTop 10 busiest days:")
        print(daily.sort_values("trades", ascending=False).head(10).to_string())

        # Monthly aggregate
        monthly = rows.assign(month=rows["exit_time"].dt.to_period("M"))
        monthly_g = monthly.groupby("month").agg(
            trades=("pnl_pct", "size"),
            avg_pnl=("pnl_pct", "mean"),
            cum_pnl=("pnl_pct", "sum"),
            sl_count=("exit_reason", lambda s: (s == "SL").sum()),
        ).round(2)
        print("\nMonthly summary:")
        print(monthly_g.to_string())

        summary[sym] = {
            "csv": out_path,
            "total_trades": len(trades),
            "mean_per_day": float(daily["trades"].mean()),
        }
        print(f"\n  → CSV saved: {out_path}")

    print("\n=== Summary ===")
    for sym, s in summary.items():
        print(f"  {sym}: total {s['total_trades']} trades  ~{s['mean_per_day']:.1f}/day  → {s['csv']}")


if __name__ == "__main__":
    main()
