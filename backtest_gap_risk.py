"""
Quantify gap-risk impact: compare three intraday/overnight policies on the same
data. Specifically targets the previous overnight backtest's optimistic
assumption that SL always fills at the SL price, even when overnight gap blew
through it.

Modes per symbol:
  A) intraday_only        — force-close before session end, no overnight risk.
  B) overnight_naive      — allow overnight holds, SL fills at SL price (the
                            unrealistic baseline that produced +1,177% SOXL).
  C) overnight_realistic  — allow overnight holds, but if next session opens
                            BELOW the SL price, exit at the open price (worse
                            than SL). This models real gap-down loss.

Also reports a histogram of overnight gaps to show the asymmetry the user
worried about.
"""
import os, math, time, json
from dataclasses import dataclass

import numpy as np
import pandas as pd

import backtest_us_equities as us


@dataclass
class GapConfig:
    z_thresh: float = -1.2
    tp_pct: float = 0.5
    trail_pct: float = 0.2
    sl_pct: float = -2.0
    grid_l2_pct: float = -1.0
    grid_l3_pct: float = -2.0
    fee_per_side: float = 0.00005
    slippage_per_side: float = 0.0005
    weights: tuple = (0.30, 0.30, 0.40)
    initial_balance: float = 500_000.0
    entry_timeout_bars: int = 4
    mode: str = "intraday"   # "intraday" | "overnight_naive" | "overnight_realistic"


def simulate(df: pd.DataFrame, cfg: GapConfig, start_idx: int = 100) -> dict:
    df = df.copy()
    df["date"] = df["time"].dt.date
    df["session_first"] = df["date"] != df["date"].shift(1)
    df["session_last"] = df["date"] != df["date"].shift(-1)

    closes = df["close"].values
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    times = df["time"].values
    z = df["z_score"].values
    sf = df["session_first"].values
    sl_arr = df["session_last"].values

    times_pd = pd.to_datetime(df["time"])
    minutes_from_close = (16 * 60) - (times_pd.dt.hour * 60 + times_pd.dt.minute)
    force_close_arr = (minutes_from_close <= 30).values
    no_new_entry_arr = (minutes_from_close <= 90).values

    fee = cfg.fee_per_side
    slip = cfg.slippage_per_side
    balance = cfg.initial_balance

    in_trade = False
    pending_limit = None
    entry_idx = 0
    level_1_price = 0.0
    weights_filled = [0.0, 0.0, 0.0]
    cost_basis = 0.0
    total_weight = 0.0
    trailing_active = False
    peak = 0.0

    trades = []
    equity_curve = []
    forced = 0
    gap_overrides = 0       # times where realistic gap exit was worse than SL price
    gap_excess_loss = 0.0   # cumulative pnl difference (negative) from realistic vs naive

    for i in range(start_idx, len(df) - 1):
        equity_curve.append((str(times[i]), balance))

        # 1) Gap handling at session_first when in trade (overnight modes only)
        if in_trade and sf[i] and cfg.mode in ("overnight_naive", "overnight_realistic"):
            avg_price = cost_basis / total_weight
            sl_price = avg_price * (1 + cfg.sl_pct / 100.0)
            session_open = opens[i]
            if cfg.mode == "overnight_realistic" and session_open < sl_price:
                # Gap-down through SL: forced exit at session open (worse than SL)
                sell_price = session_open * (1 - slip)
                ret = (sell_price - avg_price) / avg_price
                naive_ret = (sl_price - avg_price) / avg_price - fee * 2.0
                real_ret_net = ret - fee * 2.0
                gap_overrides += 1
                gap_excess_loss += (real_ret_net - naive_ret) * total_weight * 100  # pct points
                balance = balance * (1 - total_weight) + balance * total_weight * (1 + real_ret_net)
                trades.append({
                    "entry_time": str(times[entry_idx]), "exit_time": str(times[i]),
                    "avg_price": avg_price, "exit_price": sell_price,
                    "weight_used": total_weight, "levels_filled": sum(1 for w in weights_filled if w > 0),
                    "exit_reason": "GAP_DOWN_SL", "pnl_pct": real_ret_net * 100,
                    "bars_held": i - entry_idx, "overnight": True,
                })
                in_trade = False
                trailing_active = False
                peak = 0.0
                weights_filled = [0.0, 0.0, 0.0]
                total_weight = 0.0
                cost_basis = 0.0
                pending_limit = None
                continue

        # 2) Intraday-only: force-close at session end
        if cfg.mode == "intraday":
            if (force_close_arr[i] or sl_arr[i]) and pending_limit is not None and not in_trade:
                pending_limit = None
            if in_trade and (force_close_arr[i] or sl_arr[i]):
                avg_price = cost_basis / total_weight
                sell_price = closes[i] * (1 - slip)
                ret = (sell_price - avg_price) / avg_price - fee * 2.0
                balance = balance * (1 - total_weight) + balance * total_weight * (1 + ret)
                trades.append({
                    "entry_time": str(times[entry_idx]), "exit_time": str(times[i]),
                    "avg_price": avg_price, "exit_price": sell_price,
                    "weight_used": total_weight, "levels_filled": sum(1 for w in weights_filled if w > 0),
                    "exit_reason": "FORCE_CLOSE", "pnl_pct": ret * 100,
                    "bars_held": i - entry_idx, "overnight": False,
                })
                forced += 1
                in_trade = False
                trailing_active = False
                peak = 0.0
                weights_filled = [0.0, 0.0, 0.0]
                total_weight = 0.0
                cost_basis = 0.0
                pending_limit = None
                continue
            if sf[i]:
                pending_limit = None
                continue

        # 3) Try fill pending limit
        if pending_limit is not None and not in_trade:
            limit_price, expire_idx = pending_limit
            if lows[i] <= limit_price:
                level_1_price = limit_price
                weights_filled = [cfg.weights[0], 0.0, 0.0]
                cost_basis = limit_price * cfg.weights[0]
                total_weight = cfg.weights[0]
                in_trade = True
                entry_idx = i
                trailing_active = False
                peak = 0.0
                pending_limit = None
            elif i >= expire_idx:
                pending_limit = None

        # 4) New signal
        block_new = (cfg.mode == "intraday" and (force_close_arr[i] or no_new_entry_arr[i]))
        if not in_trade and pending_limit is None and not block_new:
            if not math.isnan(z[i]) and z[i] < cfg.z_thresh:
                pending_limit = (closes[i], i + cfg.entry_timeout_bars)
            continue

        if not in_trade or i < entry_idx:
            continue

        # 5) Manage open trade — grids, TP trail, SL
        l2 = level_1_price * (1 + cfg.grid_l2_pct / 100.0)
        l3 = level_1_price * (1 + cfg.grid_l3_pct / 100.0)
        if weights_filled[1] == 0.0 and lows[i] <= l2:
            cost_basis += l2 * cfg.weights[1]
            weights_filled[1] = cfg.weights[1]
            total_weight += cfg.weights[1]
        if weights_filled[2] == 0.0 and lows[i] <= l3:
            cost_basis += l3 * cfg.weights[2]
            weights_filled[2] = cfg.weights[2]
            total_weight += cfg.weights[2]

        avg_price = cost_basis / total_weight
        sl_price = avg_price * (1 + cfg.sl_pct / 100.0)
        tp_act = avg_price * (1 + cfg.tp_pct / 100.0)
        bh, bl = highs[i], lows[i]
        exit_price = None
        is_market = False
        reason = None
        if not trailing_active:
            if bl <= sl_price:
                exit_price = sl_price
                is_market = True
                reason = "SL"
            elif bh >= tp_act:
                trailing_active = True
                peak = max(bh, tp_act)
        else:
            peak = max(peak, bh)
            stop = peak * (1 - cfg.trail_pct / 100.0)
            floor = avg_price * (1 + (cfg.tp_pct - cfg.trail_pct) / 100.0)
            stop = max(stop, floor)
            if bl <= stop:
                exit_price = stop
                reason = "TRAIL"

        if exit_price is not None:
            sell_price = exit_price * (1 - slip) if is_market else exit_price
            ret = (sell_price - avg_price) / avg_price - fee * 2.0
            balance = balance * (1 - total_weight) + balance * total_weight * (1 + ret)
            entry_date = pd.to_datetime(str(times[entry_idx])).date()
            exit_date = pd.to_datetime(str(times[i])).date()
            trades.append({
                "entry_time": str(times[entry_idx]), "exit_time": str(times[i]),
                "avg_price": avg_price, "exit_price": sell_price,
                "weight_used": total_weight, "levels_filled": sum(1 for w in weights_filled if w > 0),
                "exit_reason": reason, "pnl_pct": ret * 100,
                "bars_held": i - entry_idx, "overnight": entry_date != exit_date,
            })
            in_trade = False
            trailing_active = False
            peak = 0.0
            weights_filled = [0.0, 0.0, 0.0]
            total_weight = 0.0
            cost_basis = 0.0

    equity_curve.append((str(times[-1]), balance))
    return {
        "final_balance": balance, "trades": trades, "equity_curve": equity_curve,
        "forced_exits": forced, "gap_overrides": gap_overrides,
        "gap_excess_loss_pct": gap_excess_loss,
    }


def quick_metrics(result, cfg, df):
    final = result["final_balance"]
    initial = cfg.initial_balance
    total_ret = (final / initial - 1) * 100
    eq = pd.DataFrame(result["equity_curve"], columns=["time", "equity"])
    eq["time"] = pd.to_datetime(eq["time"], utc=True)
    eq = eq.set_index("time")
    eq = eq[~eq.index.duplicated(keep="last")]
    rmax = eq["equity"].cummax()
    dd = (eq["equity"] / rmax - 1) * 100
    daily = eq["equity"].resample("1D").last().ffill()
    dr = daily.pct_change().dropna()
    sharpe = (dr.mean() / dr.std()) * math.sqrt(252) if dr.std() > 0 else 0
    trades = result["trades"]
    if trades:
        wins = sum(1 for t in trades if t["pnl_pct"] > 0)
        wr = wins / len(trades) * 100
        overnights = sum(1 for t in trades if t.get("overnight"))
    else:
        wr = overnights = 0
    return {
        "final": final,
        "total_return_pct": total_ret,
        "mdd_pct": float(dd.min()) if len(dd) else 0.0,
        "sharpe": float(sharpe),
        "trades": len(trades),
        "win_rate_pct": wr,
        "overnight_trades": overnights,
        "forced_exits": result["forced_exits"],
        "gap_overrides": result["gap_overrides"],
        "gap_excess_loss_pct": result["gap_excess_loss_pct"],
    }


def session_gap_stats(df: pd.DataFrame) -> dict:
    df = df.copy().sort_values("time").reset_index(drop=True)
    df["date"] = df["time"].dt.date
    df["session_first"] = df["date"] != df["date"].shift(1)
    # Previous session close (last bar before each session_first)
    prev_close = df["close"].shift(1)
    gaps = (df.loc[df["session_first"], "open"].values
            / prev_close.loc[df["session_first"]].values - 1) * 100
    gaps = gaps[~np.isnan(gaps)]
    return {
        "n_sessions": len(gaps),
        "mean_gap_pct": float(np.mean(gaps)),
        "median_gap_pct": float(np.median(gaps)),
        "std_gap_pct": float(np.std(gaps)),
        "gap_down_count": int((gaps < -1.0).sum()),     # gap-down > 1%
        "gap_down_2pct": int((gaps < -2.0).sum()),
        "gap_down_5pct": int((gaps < -5.0).sum()),
        "min_gap_pct": float(np.min(gaps)),
        "gap_up_count": int((gaps > 1.0).sum()),
        "gap_up_2pct": int((gaps > 2.0).sum()),
        "max_gap_pct": float(np.max(gaps)),
    }


def main():
    rows = []
    print("\nLoading data...")
    for sym in ["NVDA", "TQQQ", "SOXL"]:
        df = us.fetch_yf(sym, "1h", "2y")
        df_ind = us.compute_indicators(df)

        gs = session_gap_stats(df)
        print(f"\n=== {sym} session gap stats (1h, 2y) ===")
        print(f"  sessions={gs['n_sessions']}  mean={gs['mean_gap_pct']:+.2f}%  "
              f"median={gs['median_gap_pct']:+.2f}%  std={gs['std_gap_pct']:.2f}%  "
              f"min={gs['min_gap_pct']:+.2f}%  max={gs['max_gap_pct']:+.2f}%")
        print(f"  Gap-DOWN >1%: {gs['gap_down_count']}  >2%: {gs['gap_down_2pct']}  >5%: {gs['gap_down_5pct']}")
        print(f"  Gap-UP   >1%: {gs['gap_up_count']}  >2%: {gs['gap_up_2pct']}")

        for mode in ["intraday", "overnight_naive", "overnight_realistic"]:
            cfg = GapConfig(mode=mode, tp_pct=0.5)
            r = simulate(df_ind, cfg)
            m = quick_metrics(r, cfg, df_ind)
            m["symbol"] = sym
            m["mode"] = mode
            rows.append(m)
            print(f"  [{mode:22}] ret={m['total_return_pct']:+10.2f}%  "
                  f"final={m['final']:>13,.0f}  MDD={m['mdd_pct']:.2f}%  "
                  f"Sharpe={m['sharpe']:.2f}  trades={m['trades']:>4}  "
                  f"win={m['win_rate_pct']:.1f}%  overnights={m['overnight_trades']:>4}  "
                  f"gap_overrides={m['gap_overrides']}  gap_excess={m['gap_excess_loss_pct']:+.2f}%")

    out = os.path.join(us.RESULTS_DIR, f"gap_risk_{int(time.time())}.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
