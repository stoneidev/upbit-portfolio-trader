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

def run_trailing_stop_sim_fast(lows, highs, closes, z_scores, tb_ratios, z_thresh, tb_thresh, activation_pct, trailing_pct, sl_pct):
    """
    Lookahead-free trailing stop simulation.
    - If trailing_pct == 0.0: standard fixed TP exit at exactly activation_pct%.
    - If trailing_pct > 0.0:
      - Trailing activates when High >= activation_price.
      - Once active, peak_price = max(peak_price, High).
      - Exit occurs on the NEXT bar if Low <= peak_price * (1 - trailing_pct/100).
    """
    trades = []
    in_trade = False
    
    level_1_price = 0.0
    level_2_filled = False
    level_3_filled = False
    
    trailing_active = False
    peak_price = 0.0
    
    n = len(closes)
    
    for i in range(n):
        if np.isnan(z_scores[i]):
            continue
            
        low = lows[i]
        high = highs[i]
        close = closes[i]
        
        if not in_trade:
            c1 = z_scores[i] < z_thresh
            c2 = tb_ratios[i] > tb_thresh
            
            if c1 and c2:
                level_1_price = close
                level_2_filled = False
                level_3_filled = False
                trailing_active = False
                peak_price = 0.0
                in_trade = True
        else:
            # Grid fills
            target_level_2 = level_1_price * 0.990
            target_level_3 = level_1_price * 0.980
            
            if not level_2_filled and low <= target_level_2:
                level_2_filled = True
            if not level_3_filled and low <= target_level_3:
                level_3_filled = True
                
            # Average entry price
            if level_3_filled:
                avg_price = (level_1_price * 0.3 + target_level_2 * 0.3 + target_level_3 * 0.4)
                total_weight = 1.0
            elif level_2_filled:
                avg_price = (level_1_price * 0.3 + target_level_2 * 0.3) / 0.6
                total_weight = 0.6
            else:
                avg_price = level_1_price
                total_weight = 0.3
                
            # Exits
            sl_price = avg_price * (1 + sl_pct / 100.0)
            tp_price_fixed = avg_price * (1 + activation_pct / 100.0)
            
            if trailing_pct == 0.0:
                # Standard fixed TP
                hit_tp = high >= tp_price_fixed
                hit_sl = low <= sl_price
                
                if hit_tp and hit_sl:
                    trades.append({'weight': total_weight, 'profit_pct': sl_pct, 'win': False})
                    in_trade = False
                elif hit_tp:
                    trades.append({'weight': total_weight, 'profit_pct': activation_pct, 'win': True})
                    in_trade = False
                elif hit_sl:
                    trades.append({'weight': total_weight, 'profit_pct': sl_pct, 'win': False})
                    in_trade = False
            else:
                # Trailing stop mode (lookahead-free)
                if not trailing_active:
                    # Check if trailing activates in the current bar
                    if high >= tp_price_fixed:
                        trailing_active = True
                        peak_price = max(high, tp_price_fixed)
                        
                    # Also check if stop loss hit before trailing activated
                    if low <= sl_price:
                        trades.append({'weight': total_weight, 'profit_pct': sl_pct, 'win': False})
                        in_trade = False
                else:
                    # Trailing is already active. Check exit based on previous peak
                    stop_price = peak_price * (1 - trailing_pct / 100.0)
                    
                    # If low goes below stop price, we exit at stop_price
                    if low <= stop_price:
                        # Lock in at least some profit
                        exit_price = max(stop_price, avg_price * (1 + (activation_pct - trailing_pct)/100.0))
                        profit_pct = (exit_price - avg_price) / avg_price * 100
                        trades.append({'weight': total_weight, 'profit_pct': profit_pct, 'win': True})
                        in_trade = False
                    else:
                        # Otherwise, update peak price with the current bar's high
                        peak_price = max(peak_price, high)
                        
    return trades

if __name__ == "__main__":
    df = load_data()
    df = calculate_indicators(df)
    
    lows = df["low"].to_numpy()
    highs = df["high"].to_numpy()
    closes = df["close"].to_numpy()
    z_scores = df["z_score"].to_numpy()
    tb_ratios = df["taker_buy_ratio_smoothed"].to_numpy()
    
    print("=========================================================")
    print("      XRP Trailing Stop Simulation (2 Years - NO BIAS)")
    print("=========================================================")
    print("Settings:")
    print(" - Z-Thresh: -1.2, Taker Buy Thresh: 0.49")
    print(" - 3-Step Grid scaling-in")
    print(" - Stop Loss: -2.0%")
    print(" - Fee Rate: 0.04% (Maker Fee)\n")
    
    # 1. Fixed TP 0.5% (No Trailing)
    trades_fixed = run_trailing_stop_sim_fast(
        lows, highs, closes, z_scores, tb_ratios,
        z_thresh=-1.2, tb_thresh=0.49, activation_pct=0.5, trailing_pct=0.0, sl_pct=-2.0
    )
    
    # 2. Trailing Stop (Activates at 0.5%, trails by 0.2%)
    trades_trailing = run_trailing_stop_sim_fast(
        lows, highs, closes, z_scores, tb_ratios,
        z_thresh=-1.2, tb_thresh=0.49, activation_pct=0.5, trailing_pct=0.2, sl_pct=-2.0
    )
    
    # 3. Trailing Stop (Activates at 1.0%, trails by 0.3%)
    trades_trailing_large = run_trailing_stop_sim_fast(
        lows, highs, closes, z_scores, tb_ratios,
        z_thresh=-1.2, tb_thresh=0.49, activation_pct=1.0, trailing_pct=0.3, sl_pct=-2.0
    )
    
    def evaluate(trades, name):
        total = len(trades)
        if total == 0:
            print(f"[{name}] No trades.")
            return
        wins = sum(1 for t in trades if t['win'])
        wr = wins / total * 100
        
        cap = 1000000.0
        avg_profit = []
        for t in trades:
            ret = (t['weight'] * t['profit_pct']) / 100.0
            fee = t['weight'] * 0.0004
            cap *= (1 + ret - fee)
            avg_profit.append(t['profit_pct'])
            
        print(f"Strategy: {name}")
        print(f" Trades: {total} | Win Rate: {wr:.2f}%")
        print(f" Avg Trade Profit: {np.mean(avg_profit):.2f}%")
        print(f" Final Capital: {cap:,.0f} KRW (Return: {(cap-1000000)/1000000*100:.2f}%)")
        print("-" * 55)

    evaluate(trades_fixed, "1. Fixed TP 0.5% (Baseline)")
    evaluate(trades_trailing, "2. Trailing Stop (Activate +0.5%, Trail 0.2%)")
    evaluate(trades_trailing_large, "3. Trailing Stop (Activate +1.0%, Trail 0.3%)")
