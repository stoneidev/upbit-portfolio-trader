"""
US equity backtest of the same mean-reversion strategy on NVDA and TQQQ.

Differences vs crypto version:
- Regular trading hours only (09:30-16:00 ET).
- Force-close any open position at 15:30 ET (avoid overnight gap risk).
- Skip the first bar of each session (overnight gap is not a mean-reversion signal).
- Indicator window 100 bars (~8 trading days for 1h, ~1.3 days for 5m).
- Hybrid order model: limit entries (no slip) + market SL (slip).
- Round-trip fees set low (most US brokers are commission-free for stocks/ETFs;
  we still model an SEC/TAF passthrough of ~0.001% per side as token cost).
- Slippage modeled per-side, default 0.05% (NVDA/TQQQ are deep liquid).

Two tests per symbol:
  A) 1-hour bars over ~1 year (yfinance allows it).
  B) 5-min bars over ~60 days (yfinance limit for 5m).
"""
import os, math, time, json
from dataclasses import dataclass, asdict
from typing import List

import numpy as np
import pandas as pd
import yfinance as yf

import backtest_strategy as bs


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "backtest_data_us")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "backtest_results")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def fetch_yf(symbol: str, interval: str, period: str) -> pd.DataFrame:
    cache = os.path.join(DATA_DIR, f"{symbol}_{interval}_{period}.csv")
    if os.path.exists(cache):
        df = pd.read_csv(cache)
        df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("US/Eastern")
        return df

    raw = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
    if raw.empty:
        raise RuntimeError(f"No data returned for {symbol}")
    raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
    raw = raw.reset_index()
    raw = raw.rename(columns={
        "Datetime": "time", "Date": "time",
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    df = raw[["time", "open", "high", "low", "close", "volume"]].copy()
    # Convert to US/Eastern for session filter
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("US/Eastern")
    # Regular trading hours only
    df["t"] = df["time"].dt.time
    df = df[(df["t"] >= pd.Timestamp("09:30").time()) & (df["t"] <= pd.Timestamp("16:00").time())]
    df = df.drop(columns=["t"])
    # Drop NaN
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    df.to_csv(cache, index=False)
    return df


@dataclass
class USConfig:
    z_thresh: float = -1.2
    tp_pct: float = 0.5
    trail_pct: float = 0.2
    sl_pct: float = -2.0
    grid_l2_pct: float = -1.0
    grid_l3_pct: float = -2.0
    fee_per_side: float = 0.00005     # ~SEC/TAF passthrough on sells; tiny
    slippage_per_side: float = 0.0005  # 0.05% per side (deep liquid)
    weights: tuple = (0.30, 0.30, 0.40)
    initial_balance: float = 500_000.0
    entry_timeout_bars: int = 4
    force_close_minutes_before_close: int = 30   # force-exit window starts at 15:30 ET
    no_new_entry_minutes_before_close: int = 90  # no new signals after 14:30 ET


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ma_100"] = df["close"].rolling(100).mean()
    df["std_100"] = df["close"].rolling(100).std()
    df["z_score"] = (df["close"] - df["ma_100"]) / (df["std_100"] + 1e-9)
    df["date"] = df["time"].dt.date
    df["session_first"] = df["date"] != df["date"].shift(1)
    df["session_last"] = df["date"] != df["date"].shift(-1)
    return df


def simulate_us(df: pd.DataFrame, cfg: USConfig, start_idx: int = 100) -> dict:
    closes = df["close"].values
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    times = df["time"].values
    z = df["z_score"].values
    session_first = df["session_first"].values
    session_last = df["session_last"].values

    # Pre-compute time-of-day masks
    times_pd = pd.to_datetime(df["time"])
    minutes_from_close = (16 * 60) - (times_pd.dt.hour * 60 + times_pd.dt.minute)
    force_close_arr = (minutes_from_close <= cfg.force_close_minutes_before_close).values
    no_new_entry_arr = (minutes_from_close <= cfg.no_new_entry_minutes_before_close).values

    fee = cfg.fee_per_side
    slip = cfg.slippage_per_side

    balance = cfg.initial_balance
    equity_curve = []
    trades = []
    in_trade = False
    pending_limit = None
    entry_idx = 0
    level_1_price = 0.0
    weights_filled = [0.0, 0.0, 0.0]
    cost_basis = 0.0
    total_weight = 0.0
    trailing_active = False
    peak = 0.0
    sig_total = sig_filled = sig_cancelled = 0
    forced_exits = 0

    for i in range(start_idx, len(df) - 1):
        equity_curve.append((str(times[i]), balance))

        # Cancel any pending limit when entering force-close window or session end
        if (force_close_arr[i] or session_last[i]) and pending_limit is not None and not in_trade:
            pending_limit = None
            sig_cancelled += 1

        # Force-close: at force_close_arr OR session_last (last-bar safety net)
        if in_trade and (force_close_arr[i] or session_last[i]):
            avg_price = cost_basis / total_weight
            sell_price = closes[i] * (1 - slip)  # market exit
            ret = (sell_price - avg_price) / avg_price
            net_ret = ret - fee * 2.0
            balance = balance * (1 - total_weight) + balance * total_weight * (1 + net_ret)
            levels = sum(1 for w in weights_filled if w > 0)
            trades.append({
                "entry_time": str(times[entry_idx]), "exit_time": str(times[i]),
                "avg_price": avg_price, "exit_price": sell_price,
                "weight_used": total_weight, "levels_filled": levels,
                "exit_reason": "FORCE_CLOSE", "pnl_pct": net_ret * 100.0,
                "bars_held": i - entry_idx,
            })
            forced_exits += 1
            in_trade = False
            trailing_active = False
            peak = 0.0
            weights_filled = [0.0, 0.0, 0.0]
            total_weight = 0.0
            cost_basis = 0.0
            pending_limit = None
            continue

        # Skip first bar of session (overnight gap distorts indicators)
        if session_first[i]:
            pending_limit = None  # cancel any leftover pending across sessions
            continue

        # Try fill pending limit
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
                sig_filled += 1
            elif i >= expire_idx:
                pending_limit = None
                sig_cancelled += 1

        # New signal — also blocked once we're in the no-new-entry window
        if (not in_trade and pending_limit is None
                and not force_close_arr[i] and not no_new_entry_arr[i]):
            if not math.isnan(z[i]) and z[i] < cfg.z_thresh:
                sig_total += 1
                pending_limit = (closes[i], i + cfg.entry_timeout_bars)
            continue

        if not in_trade:
            continue
        if i < entry_idx:
            continue

        # Manage open trade
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
        bar_h, bar_l = highs[i], lows[i]
        exit_price = None
        exit_reason = None
        is_market_exit = False
        if not trailing_active:
            if bar_l <= sl_price:
                exit_price = sl_price
                exit_reason = "SL"
                is_market_exit = True
            elif bar_h >= tp_act:
                trailing_active = True
                peak = max(bar_h, tp_act)
        else:
            peak = max(peak, bar_h)
            stop = peak * (1 - cfg.trail_pct / 100.0)
            floor = avg_price * (1 + (cfg.tp_pct - cfg.trail_pct) / 100.0)
            stop = max(stop, floor)
            if bar_l <= stop:
                exit_price = stop
                exit_reason = "TRAIL"

        if exit_price is not None:
            sell_price = exit_price * (1 - slip) if is_market_exit else exit_price
            ret = (sell_price - avg_price) / avg_price
            net_ret = ret - fee * 2.0
            balance = balance * (1 - total_weight) + balance * total_weight * (1 + net_ret)
            levels = sum(1 for w in weights_filled if w > 0)
            trades.append({
                "entry_time": str(times[entry_idx]), "exit_time": str(times[i]),
                "avg_price": avg_price, "exit_price": sell_price,
                "weight_used": total_weight, "levels_filled": levels,
                "exit_reason": exit_reason, "pnl_pct": net_ret * 100.0,
                "bars_held": i - entry_idx,
            })
            in_trade = False
            trailing_active = False
            peak = 0.0
            weights_filled = [0.0, 0.0, 0.0]
            total_weight = 0.0
            cost_basis = 0.0

    equity_curve.append((str(times[-1]), balance))
    return {
        "final_balance": balance,
        "trades": trades,
        "equity_curve": equity_curve,
        "signals": sig_total, "filled": sig_filled, "cancelled": sig_cancelled,
        "forced_exits": forced_exits,
    }


def assert_intraday_only(trades, label):
    bad = []
    for t in trades:
        e = pd.to_datetime(t["entry_time"]).date()
        x = pd.to_datetime(t["exit_time"]).date()
        if e != x:
            bad.append((t["entry_time"], t["exit_time"]))
    if bad:
        print(f"  ⚠️ {label}: {len(bad)} overnight trades found! e.g. {bad[:3]}")
    return len(bad) == 0


def metrics_for_us(result, df, cfg, label):
    eq = pd.DataFrame(result["equity_curve"], columns=["time", "equity"])
    eq["time"] = pd.to_datetime(eq["time"], utc=True)
    eq = eq.set_index("time")
    eq = eq[~eq.index.duplicated(keep="last")]
    final = result["final_balance"]
    initial = cfg.initial_balance
    total_return = (final / initial - 1) * 100
    if len(eq) > 1:
        span_days = (eq.index[-1] - eq.index[0]).total_seconds() / 86400
    else:
        span_days = 0
    years = span_days / 365.0 if span_days > 0 else 1.0
    cagr = ((final / initial) ** (1 / years) - 1) * 100 if years > 0 else 0
    rmax = eq["equity"].cummax()
    dd = (eq["equity"] / rmax - 1) * 100
    mdd = float(dd.min()) if len(dd) else 0.0
    daily = eq["equity"].resample("1D").last().ffill()
    dr = daily.pct_change().dropna()
    sharpe = (dr.mean() / dr.std()) * math.sqrt(252) if dr.std() > 0 else 0.0

    trades = result["trades"]
    if trades:
        wins = [t for t in trades if t["pnl_pct"] > 0]
        losses = [t for t in trades if t["pnl_pct"] <= 0]
        wr = len(wins) / len(trades) * 100
        avg_w = np.mean([t["pnl_pct"] for t in wins]) if wins else 0
        avg_l = np.mean([t["pnl_pct"] for t in losses]) if losses else 0
        gp = sum(t["pnl_pct"] for t in wins)
        gl = -sum(t["pnl_pct"] for t in losses)
        pf = gp / gl if gl > 0 else 999.0
    else:
        wr = avg_w = avg_l = pf = 0

    df_window = df[df["time"].between(eq.index[0], eq.index[-1])]
    bh = (df_window["close"].iloc[-1] / df_window["close"].iloc[0] - 1) * 100 if len(df_window) >= 2 else 0

    return {
        "label": label,
        "trades": len(trades),
        "final_balance": float(final),
        "total_return_pct": float(total_return),
        "cagr_pct": float(cagr),
        "mdd_pct": mdd,
        "sharpe": float(sharpe),
        "win_rate_pct": float(wr),
        "avg_win_pct": float(avg_w),
        "avg_loss_pct": float(avg_l),
        "profit_factor": float(pf),
        "buy_and_hold_pct": float(bh),
        "alpha_vs_bh_pct": float(total_return - bh),
        "span_days": float(span_days),
        "signals": result["signals"],
        "filled": result["filled"],
        "cancelled": result["cancelled"],
        "forced_exits": result["forced_exits"],
    }


def run_wfo(df, base_cfg, is_bars, oos_bars,
            z_candidates=(-1.0, -1.2, -1.5, -1.8, -2.0)):
    df_ind = compute_indicators(df).reset_index(drop=True)
    n = len(df_ind)
    pos = 200 + is_bars
    balance = base_cfg.initial_balance
    all_trades = []
    eqc = []
    chosen = []
    sig_t = sig_f = sig_c = forced = 0
    while pos + oos_bars <= n:
        is_w = df_ind.iloc[pos - is_bars: pos].reset_index(drop=True)
        oos_w = df_ind.iloc[pos: pos + oos_bars].reset_index(drop=True)
        best_z, best_ret = base_cfg.z_thresh, -1e9
        for z in z_candidates:
            cfg_try = USConfig(**{**base_cfg.__dict__, "z_thresh": z, "initial_balance": 500_000.0})
            r = simulate_us(is_w, cfg_try, start_idx=100)
            ret = r["final_balance"] - cfg_try.initial_balance
            if ret > best_ret:
                best_ret, best_z = ret, z
        chosen.append(best_z)
        cfg_oos = USConfig(**{**base_cfg.__dict__, "z_thresh": best_z, "initial_balance": balance})
        r = simulate_us(oos_w, cfg_oos, start_idx=0)
        balance = r["final_balance"]
        all_trades.extend(r["trades"])
        eqc.extend(r["equity_curve"])
        sig_t += r["signals"]; sig_f += r["filled"]; sig_c += r["cancelled"]
        forced += r["forced_exits"]
        pos += oos_bars
    return {
        "final_balance": balance, "trades": all_trades, "equity_curve": eqc,
        "chosen_zs": chosen, "signals": sig_t, "filled": sig_f, "cancelled": sig_c,
        "forced_exits": forced,
    }


def report_run(symbol, interval, period, bars_per_session, is_days, oos_days):
    print(f"\n=== {symbol} | {interval} bars over {period} ===")
    df = fetch_yf(symbol, interval, period)
    print(f"  Loaded {len(df)} bars: {df['time'].iloc[0]} → {df['time'].iloc[-1]}")
    is_bars = bars_per_session * is_days
    oos_bars = bars_per_session * oos_days

    # Fixed-param test
    cfg_fixed = USConfig(tp_pct=0.5, trail_pct=0.2, sl_pct=-2.0)
    df_ind = compute_indicators(df)
    r_fixed = simulate_us(df_ind, cfg_fixed, start_idx=100)
    m_fixed = metrics_for_us(r_fixed, df, cfg_fixed, f"{symbol}-{interval}-FIXED")
    assert_intraday_only(r_fixed["trades"], f"{symbol}-{interval}-FIXED")
    print(f"  [FIXED tp=0.5] ret={m_fixed['total_return_pct']:+.2f}% bh={m_fixed['buy_and_hold_pct']:+.2f}% "
          f"alpha={m_fixed['alpha_vs_bh_pct']:+.2f}% MDD={m_fixed['mdd_pct']:.2f}% "
          f"Sharpe={m_fixed['sharpe']:.2f} trades={m_fixed['trades']} win={m_fixed['win_rate_pct']:.1f}% "
          f"PF={m_fixed['profit_factor']:.2f} forced={m_fixed['forced_exits']}")

    # WFO test
    cfg_wfo = USConfig(tp_pct=0.5, trail_pct=0.2, sl_pct=-2.0)
    if is_bars + oos_bars + 200 < len(df):
        r_wfo = run_wfo(df, cfg_wfo, is_bars, oos_bars)
        m_wfo = metrics_for_us(r_wfo, df, cfg_wfo, f"{symbol}-{interval}-WFO")
        m_wfo["chosen_zs"] = r_wfo["chosen_zs"]
        assert_intraday_only(r_wfo["trades"], f"{symbol}-{interval}-WFO")
        print(f"  [WFO tp=0.5]   ret={m_wfo['total_return_pct']:+.2f}% bh={m_wfo['buy_and_hold_pct']:+.2f}% "
              f"alpha={m_wfo['alpha_vs_bh_pct']:+.2f}% MDD={m_wfo['mdd_pct']:.2f}% "
              f"Sharpe={m_wfo['sharpe']:.2f} trades={m_wfo['trades']} win={m_wfo['win_rate_pct']:.1f}% "
              f"PF={m_wfo['profit_factor']:.2f} zs={r_wfo['chosen_zs']}")
    else:
        m_wfo = None
        print(f"  [WFO] skipped — not enough bars (need {is_bars+oos_bars+200}, have {len(df)})")

    # Also tp=0.3 fixed for comparison
    cfg_03 = USConfig(tp_pct=0.3, trail_pct=0.12, sl_pct=-2.0)
    df_ind = compute_indicators(df)
    r_03 = simulate_us(df_ind, cfg_03, start_idx=100)
    m_03 = metrics_for_us(r_03, df, cfg_03, f"{symbol}-{interval}-FIXED-tp0.3")
    assert_intraday_only(r_03["trades"], f"{symbol}-{interval}-FIXED-tp0.3")
    print(f"  [FIXED tp=0.3] ret={m_03['total_return_pct']:+.2f}% alpha={m_03['alpha_vs_bh_pct']:+.2f}% "
          f"MDD={m_03['mdd_pct']:.2f}% Sharpe={m_03['sharpe']:.2f} trades={m_03['trades']} "
          f"win={m_03['win_rate_pct']:.1f}% PF={m_03['profit_factor']:.2f}")

    # Slippage stress
    print("  [SLIPPAGE STRESS @ tp=0.5 fixed]")
    for slip in [0.0003, 0.0005, 0.0010, 0.0020]:
        cfg_s = USConfig(tp_pct=0.5, slippage_per_side=slip)
        df_ind = compute_indicators(df)
        r = simulate_us(df_ind, cfg_s, start_idx=100)
        m = metrics_for_us(r, df, cfg_s, f"{symbol}-slip{slip}")
        print(f"    slip={slip*100:.2f}%/side  ret={m['total_return_pct']:+.2f}%  "
              f"MDD={m['mdd_pct']:.2f}%  Sharpe={m['sharpe']:.2f}  trades={m['trades']}")

    return m_fixed, m_wfo, m_03


def main():
    out_rows = []
    # 1-hour bars, 2 years
    for sym in ["NVDA", "TQQQ", "SOXL"]:
        # 1-hour: ~6.5 bars/session
        m_fix, m_wfo, m_03 = report_run(sym, "1h", "2y",
                                         bars_per_session=7,  # ~6.5 rounded up
                                         is_days=20, oos_days=20)
        if m_fix: out_rows.append(m_fix)
        if m_wfo: out_rows.append(m_wfo)
        if m_03:  out_rows.append(m_03)
        # 5-min bars, ~60 days max
        m_fix5, m_wfo5, m_03_5 = report_run(sym, "5m", "60d",
                                             bars_per_session=78,  # 6.5h * 60min/5
                                             is_days=10, oos_days=5)
        if m_fix5: out_rows.append(m_fix5)
        if m_wfo5: out_rows.append(m_wfo5)
        if m_03_5: out_rows.append(m_03_5)

    out = os.path.join(RESULTS_DIR, f"us_equities_{int(time.time())}.json")
    with open(out, "w") as f:
        json.dump(out_rows, f, indent=2, default=str)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
