import pandas as pd
import numpy as np
import os
import time

def load_data():
    csv_path = "/Users/stoni/Projects/AI/xrp_2y_data.csv"
    df = pd.read_csv(csv_path)
    df["open_time"] = pd.to_datetime(df["open_time"])
    cols = ["open", "high", "low", "close", "volume", "taker_buy_volume"]
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def calculate_indicators(df):
    df["ma_100"] = df["close"].rolling(window=100).mean()
    df["std_100"] = df["close"].rolling(window=100).std()
    df["z_score"] = (df["close"] - df["ma_100"]) / (df["std_100"] + 1e-9)
    
    df["taker_buy_ratio"] = df["taker_buy_volume"] / (df["volume"] + 1e-9)
    df["taker_buy_ratio_smoothed"] = df["taker_buy_ratio"].rolling(window=3).mean()
    
    return df

def run_scalper_sim_fast(lows, highs, closes, z_scores, tb_ratios, z_thresh, tb_thresh, tp_pct, sl_pct, fee_rate):
    """
    Fast simulation of 3-step grid scalping on XRP.
    - 1차: Buy 30% at close
    - 2차: Buy 30% at 1차 - 1.0%
    - 3차: Buy 40% at 1차 - 2.0%
    - TP: avg_price + tp_pct%
    - SL: avg_price - sl_pct% (sl_pct is negative, e.g. -3.0)
    """
    trades_profit = []
    trades_weight = []
    trades_win = []
    
    in_trade = False
    level_1_price = 0.0
    level_2_filled = False
    level_3_filled = False
    
    n = len(closes)
    
    for i in range(n):
        if np.isnan(z_scores[i]):
            continue
            
        low = lows[i]
        high = highs[i]
        close = closes[i]
        
        if not in_trade:
            # Entry signal
            c1 = z_scores[i] < z_thresh
            c2 = tb_ratios[i] > tb_thresh
            
            if c1 and c2:
                level_1_price = close
                level_2_filled = False
                level_3_filled = False
                in_trade = True
        else:
            # Grid levels based on fixed percentages
            target_level_2 = level_1_price * 0.990  # -1.0%
            target_level_3 = level_1_price * 0.980  # -2.0%
            
            if not level_2_filled and low <= target_level_2:
                level_2_filled = True
            if not level_3_filled and low <= target_level_3:
                level_3_filled = True
                
            # Average price and weight calculation
            if level_3_filled:
                avg_price = (level_1_price * 0.3 + target_level_2 * 0.3 + target_level_3 * 0.4)
                total_weight = 1.0
            elif level_2_filled:
                avg_price = (level_1_price * 0.3 + target_level_2 * 0.3) / 0.6
                total_weight = 0.6
            else:
                avg_price = level_1_price
                total_weight = 0.3
                
            tp_price = avg_price * (1 + tp_pct / 100.0)
            sl_price = avg_price * (1 + sl_pct / 100.0)
            
            hit_tp = high >= tp_price
            hit_sl = low <= sl_price
            
            if hit_tp and hit_sl:
                profit = sl_pct
                trades_profit.append(profit)
                trades_weight.append(total_weight)
                trades_win.append(False)
                in_trade = False
            elif hit_tp:
                profit = tp_pct
                trades_profit.append(profit)
                trades_weight.append(total_weight)
                trades_win.append(True)
                in_trade = False
            elif hit_sl:
                profit = sl_pct
                trades_profit.append(profit)
                trades_weight.append(total_weight)
                trades_win.append(False)
                in_trade = False
                
    total_trades = len(trades_win)
    if total_trades == 0:
        return 0, 0.0, 1000000.0
        
    wins = sum(1 for w in trades_win if w)
    win_rate = wins / total_trades * 100
    
    capital = 1000000.0
    for p, w in zip(trades_profit, trades_weight):
        ret = (w * p) / 100.0
        fee = w * fee_rate
        capital *= (1 + ret - fee)
        
    return total_trades, win_rate, capital

if __name__ == "__main__":
    df = load_data()
    df = calculate_indicators(df)
    
    print("Preparing optimized numpy arrays...")
    lows = df["low"].to_numpy()
    highs = df["high"].to_numpy()
    closes = df["close"].to_numpy()
    z_scores = df["z_score"].to_numpy()
    tb_ratios = df["taker_buy_ratio_smoothed"].to_numpy()
    
    print("Optimizing XRP Scalper Strategy ('작게 많이 먹기')...")
    print("Filters: Trades >= 300, Win Rate >= 80%, Profit > 0 (after 0.04% Maker fees)\n")
    
    # Scalper search space
    z_space = [-1.8, -1.6, -1.4, -1.2, -1.0]
    tb_space = [0.49, 0.50, 0.51]
    tp_space = [0.5, 0.6, 0.7, 0.8, 1.0] # 작게 먹기 (0.5% ~ 1.0%)
    sl_space = [-2.0, -2.5, -3.0, -4.0]  # 적절한 손절선
    
    results = []
    
    start = time.time()
    for z in z_space:
        for tb in tb_space:
            for tp in tp_space:
                for sl in sl_space:
                    # Run with 0.04% maker fee
                    total_trd, wr, cap_maker = run_scalper_sim_fast(
                        lows, highs, closes, z_scores, tb_ratios,
                        z, tb, tp, sl, fee_rate=0.0004
                    )
                    net_ret_maker = (cap_maker - 1000000.0) / 1000000.0 * 100
                    
                    if total_trd >= 300 and wr >= 80.0 and net_ret_maker > 0.0:
                        # Also calculate with 0.1% taker fee for comparison
                        _, _, cap_taker = run_scalper_sim_fast(
                            lows, highs, closes, z_scores, tb_ratios,
                            z, tb, tp, sl, fee_rate=0.001
                        )
                        net_ret_taker = (cap_taker - 1000000.0) / 1000000.0 * 100
                        
                        results.append({
                            "z": z,
                            "tb": tb,
                            "tp": tp,
                            "sl": sl,
                            "trades": total_trd,
                            "win_rate": wr,
                            "cap_maker": cap_maker,
                            "ret_maker": net_ret_maker,
                            "ret_taker": net_ret_taker
                        })
                        
    end = time.time()
    print(f"Grid search completed in {end - start:.2f} seconds.")
    
    # Sort by Maker Return
    results = sorted(results, key=lambda x: x["ret_maker"], reverse=True)
    
    if len(results) == 0:
        print("No combinations met the criteria: Trades >= 300, Win Rate >= 80%, and Net Profit > 0 (after 0.04% fees).")
    else:
        print(f"Found {len(results)} matching combinations. Top 15:")
        print(f"{'Z-Thresh':<8} | {'TB-Thresh':<9} | {'TP (%)':<6} | {'SL (%)':<6} | {'Trades':<6} | {'Win Rate':<10} | {'Return (0.04% Maker)':<20} | {'Return (0.1% Taker)':<20}")
        print("-" * 105)
        for r in results[:15]:
            print(f"{r['z']:<8} | {r['tb']:<9} | {r['tp']:<6.1f} | {r['sl']:<6.1f} | {r['trades']:<6} | {r['win_rate']:<9.2f}% | {r['ret_maker']:<18.2f}% | {r['ret_taker']:<18.2f}%")
