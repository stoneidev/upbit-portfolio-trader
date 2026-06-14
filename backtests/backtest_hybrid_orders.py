"""
Hybrid order model:
- L1 entry: LIMIT at signal close. Fill only if a subsequent bar's low <= limit
  price within `entry_timeout_bars`. Otherwise cancel and skip the signal.
- L2/L3 grid: LIMIT at trigger prices (no slippage).
- TP trailing: LIMIT (no slippage).
- SL: MARKET stop (slippage applied).

Also includes the original "all-market" model for direct comparison.

Compares 2-year results (KRW-XRP, KRW-ETH, KRW-BTC) and the 50:50 portfolio.
"""
import os, math, json, time
from dataclasses import dataclass, asdict
from typing import List

import numpy as np
import pandas as pd

import backtest_strategy as bs


@dataclass
class HybridConfig:
    z_thresh: float = -1.2
    tp_pct: float = 0.3
    trail_pct: float = 0.12
    sl_pct: float = -2.0
    grid_l2_pct: float = -1.0
    grid_l3_pct: float = -2.0
    fee_per_side: float = 0.0005
    slippage_per_side: float = 0.0005
    weights: tuple = (0.30, 0.30, 0.40)
    initial_balance: float = 500_000.0
    entry_timeout_bars: int = 4   # 4 × 15min = 1 hour to fill limit order
    entry_mode: str = "limit"     # "limit" or "market"
    sl_mode: str = "market"       # always market for safety
    tp_mode: str = "limit"        # trailing as limit


def simulate_hybrid(df: pd.DataFrame, cfg: HybridConfig, start_idx: int = 100) -> dict:
    closes = df["close"].values
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    times = df["time"].astype(str).values
    z = df["z_score"].values

    fee = cfg.fee_per_side
    slip = cfg.slippage_per_side

    balance = cfg.initial_balance
    equity_curve = []
    trades: List[bs.TradeRecord] = []

    in_trade = False
    pending_limit = None    # (price, expire_idx)
    entry_idx = 0
    level_1_price = 0.0
    weights_filled = [0.0, 0.0, 0.0]
    cost_basis = 0.0
    total_weight = 0.0
    trailing_active = False
    peak = 0.0

    cancelled_signals = 0
    filled_signals = 0
    total_signals = 0

    for i in range(start_idx, len(df) - 1):
        equity_curve.append((times[i], balance))

        # 1. Try to fill pending limit entry first
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
                filled_signals += 1
            elif i >= expire_idx:
                pending_limit = None
                cancelled_signals += 1

        # 2. New signal handling
        if not in_trade and pending_limit is None:
            if not math.isnan(z[i]) and z[i] < cfg.z_thresh:
                total_signals += 1
                if cfg.entry_mode == "limit":
                    pending_limit = (closes[i], i + cfg.entry_timeout_bars)
                else:
                    # immediate market entry on next bar open with slip
                    fill = opens[i + 1] * (1 + slip)
                    level_1_price = fill
                    weights_filled = [cfg.weights[0], 0.0, 0.0]
                    cost_basis = fill * cfg.weights[0]
                    total_weight = cfg.weights[0]
                    in_trade = True
                    entry_idx = i + 1
                    trailing_active = False
                    peak = 0.0
                    filled_signals += 1
            continue

        if not in_trade:
            continue
        if i < entry_idx:
            continue

        # 3. Manage open trade
        l2_trigger = level_1_price * (1 + cfg.grid_l2_pct / 100.0)
        l3_trigger = level_1_price * (1 + cfg.grid_l3_pct / 100.0)

        if weights_filled[1] == 0.0 and lows[i] <= l2_trigger:
            cost_basis += l2_trigger * cfg.weights[1]
            weights_filled[1] = cfg.weights[1]
            total_weight += cfg.weights[1]
        if weights_filled[2] == 0.0 and lows[i] <= l3_trigger:
            cost_basis += l3_trigger * cfg.weights[2]
            weights_filled[2] = cfg.weights[2]
            total_weight += cfg.weights[2]

        avg_price = cost_basis / total_weight
        sl_price = avg_price * (1 + cfg.sl_pct / 100.0)
        tp_activation = avg_price * (1 + cfg.tp_pct / 100.0)

        bar_high = highs[i]
        bar_low = lows[i]
        exit_price = None
        exit_reason = None
        exit_is_market = False

        if not trailing_active:
            if bar_low <= sl_price:
                exit_price = sl_price
                exit_reason = "SL"
                exit_is_market = (cfg.sl_mode == "market")
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
                exit_is_market = (cfg.tp_mode == "market")

        if exit_price is not None:
            if exit_is_market:
                sell_price = exit_price * (1 - slip)
            else:
                sell_price = exit_price

            ret = (sell_price - avg_price) / avg_price
            fee_total = fee * 2.0
            net_ret = ret - fee_total
            balance = balance * (1 - total_weight) + balance * total_weight * (1 + net_ret)

            levels = sum(1 for w in weights_filled if w > 0)
            trades.append(bs.TradeRecord(
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
        "signals": total_signals,
        "filled": filled_signals,
        "cancelled": cancelled_signals,
    }


def run_wfo_hybrid(df, base_cfg: HybridConfig,
                   is_bars=4*7*96, oos_bars=4*7*96,
                   z_candidates=(-1.0, -1.2, -1.5, -1.8, -2.0)):
    df_ind = bs.compute_indicators(df).reset_index(drop=True)
    n = len(df_ind)
    start = 200
    pos = start + is_bars
    balance = base_cfg.initial_balance
    all_trades = []
    equity_curve = []
    chosen_zs = []
    sig_total = sig_filled = sig_cancelled = 0

    while pos + oos_bars <= n:
        is_window = df_ind.iloc[pos - is_bars: pos].reset_index(drop=True)
        oos_window = df_ind.iloc[pos: pos + oos_bars].reset_index(drop=True)

        best_z = base_cfg.z_thresh
        best_ret = -1e9
        for z in z_candidates:
            cfg_try = HybridConfig(**{**base_cfg.__dict__, "z_thresh": z, "initial_balance": 500_000.0})
            r = simulate_hybrid(is_window, cfg_try, start_idx=100)
            ret = r["final_balance"] - cfg_try.initial_balance
            if ret > best_ret:
                best_ret = ret
                best_z = z
        chosen_zs.append(best_z)

        cfg_oos = HybridConfig(**{**base_cfg.__dict__, "z_thresh": best_z, "initial_balance": balance})
        r = simulate_hybrid(oos_window, cfg_oos, start_idx=0)
        balance = r["final_balance"]
        all_trades.extend(r["trades"])
        equity_curve.extend(r["equity_curve"])
        sig_total += r["signals"]
        sig_filled += r["filled"]
        sig_cancelled += r["cancelled"]
        pos += oos_bars

    return {
        "final_balance": balance,
        "trades": all_trades,
        "equity_curve": equity_curve,
        "chosen_zs": chosen_zs,
        "signals": sig_total,
        "filled": sig_filled,
        "cancelled": sig_cancelled,
    }


def metrics_for(df, result, cfg, label):
    """Wrap bs.compute_metrics for our result format."""
    # bs.compute_metrics expects StrategyConfig but we made HybridConfig.
    # It only reads initial_balance — we can pass a stub.
    stub = bs.StrategyConfig(initial_balance=cfg.initial_balance)
    m = bs.compute_metrics(result, df, stub)
    m["label"] = label
    return m


def main():
    bars = 70080  # 2 years
    print(f"Loading {bars} bars per market...")
    markets = ["KRW-XRP", "KRW-ETH", "KRW-BTC"]
    dfs = {m: bs.fetch_candles(m, bars) for m in markets}

    rows = []
    eq_curves = {}

    for label, mode_cfg in [
        ("LIMIT-entry+MARKET-SL (hybrid)", dict(entry_mode="limit", sl_mode="market", tp_mode="limit")),
        ("ALL-MARKET (current trader)",     dict(entry_mode="market", sl_mode="market", tp_mode="market")),
    ]:
        eq_curves[label] = {}
        for market in markets:
            df = dfs[market]
            cfg = HybridConfig(
                tp_pct=0.3, trail_pct=0.12, sl_pct=-2.0,
                slippage_per_side=0.0005,
                **mode_cfg,
            )
            r = run_wfo_hybrid(df, cfg)
            m = metrics_for(df, r, cfg, label)
            m["market"] = market
            m["signals"] = r["signals"]
            m["filled"] = r["filled"]
            m["cancelled"] = r["cancelled"]
            m["fill_rate"] = r["filled"] / r["signals"] * 100 if r["signals"] else 0
            rows.append(m)
            eq_curves[label][market] = r["equity_curve"]
            print(f"[{label}] {market}: ret={m['total_return_pct']:+.2f}%  "
                  f"final={m['final_balance']:>13,.0f}  MDD={m['mdd_pct']:.2f}%  "
                  f"Sharpe={m['sharpe']:.2f}  trades={m['trades']}  win={m['win_rate_pct']:.1f}%  "
                  f"signals={r['signals']} filled={r['filled']}({m['fill_rate']:.0f}%) "
                  f"cancelled={r['cancelled']}")

    # Build 50:50 portfolios
    print("\n=== 50:50 PORTFOLIOS (XRP+ETH, 2Y) ===")
    for label in ["LIMIT-entry+MARKET-SL (hybrid)", "ALL-MARKET (current trader)"]:
        xrp_eq = pd.DataFrame(eq_curves[label]["KRW-XRP"], columns=["time","equity"])
        eth_eq = pd.DataFrame(eq_curves[label]["KRW-ETH"], columns=["time","equity"])
        xrp_eq["time"] = pd.to_datetime(xrp_eq["time"])
        eth_eq["time"] = pd.to_datetime(eth_eq["time"])
        xrp_eq = xrp_eq.set_index("time"); xrp_eq = xrp_eq[~xrp_eq.index.duplicated(keep="last")]
        eth_eq = eth_eq.set_index("time"); eth_eq = eth_eq[~eth_eq.index.duplicated(keep="last")]
        common = xrp_eq.index.intersection(eth_eq.index)
        port = xrp_eq.loc[common]["equity"] + eth_eq.loc[common]["equity"]
        rmax = port.cummax(); dd = (port/rmax-1)*100
        daily = port.resample("1D").last().ffill()
        dret = daily.pct_change().dropna()
        sharpe = dret.mean()/dret.std()*np.sqrt(365) if dret.std()>0 else 0
        days = (port.index[-1]-port.index[0]).days
        years = days/365.0
        cagr = ((port.iloc[-1]/1_000_000)**(1/years)-1)*100 if years>0 else 0
        print(f"\n{label}")
        print(f"  Initial: 1,000,000  Final: {port.iloc[-1]:,.0f}")
        print(f"  Total: {(port.iloc[-1]/1_000_000-1)*100:+.2f}%  CAGR: {cagr:+.2f}%  "
              f"MDD: {dd.min():.2f}%  Sharpe: {sharpe:.2f}  Days: {days}")

    # Slippage stress
    print("\n=== HYBRID SLIPPAGE STRESS (only SL is slipped) ===")
    for slip in [0.0005, 0.0010, 0.0020]:
        results = []
        for market in markets:
            df = dfs[market]
            cfg = HybridConfig(
                tp_pct=0.3, trail_pct=0.12, sl_pct=-2.0,
                slippage_per_side=slip,
                entry_mode="limit", sl_mode="market", tp_mode="limit",
            )
            r = run_wfo_hybrid(df, cfg)
            m = metrics_for(df, r, cfg, f"hybrid-slip{slip}")
            m["market"] = market
            print(f"  slip={slip*100:.2f}%/side  {market}: ret={m['total_return_pct']:+.2f}%  "
                  f"final={m['final_balance']:>13,.0f}  MDD={m['mdd_pct']:.2f}%  "
                  f"Sharpe={m['sharpe']:.2f}  trades={m['trades']}")

    out = os.path.join(bs.RESULTS_DIR, f"hybrid_orders_{int(time.time())}.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
