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
    entry_idx = 0
    
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
                entry_idx = i
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
                    trades.append({'weight': total_weight, 'profit_pct': sl_pct, 'win': False, 'idx': i})
                    in_trade = False
                elif hit_tp:
                    trades.append({'weight': total_weight, 'profit_pct': activation_pct, 'win': True, 'idx': i})
                    in_trade = False
                elif hit_sl:
                    trades.append({'weight': total_weight, 'profit_pct': sl_pct, 'win': False, 'idx': i})
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
                        trades.append({'weight': total_weight, 'profit_pct': sl_pct, 'win': False, 'idx': i})
                        in_trade = False
                else:
                    # Trailing is already active. Check exit based on previous peak
                    stop_price = peak_price * (1 - trailing_pct / 100.0)
                    
                    if low <= stop_price:
                        # Lock in at least some profit
                        exit_price = max(stop_price, avg_price * (1 + (activation_pct - trailing_pct)/100.0))
                        profit_pct = (exit_price - avg_price) / avg_price * 100
                        trades.append({'weight': total_weight, 'profit_pct': profit_pct, 'win': True, 'idx': i})
                        in_trade = False
                    else:
                        # Update peak price
                        peak_price = max(peak_price, high)
                        
    return trades

def find_best_params_trailing(lows_s, highs_s, closes_s, z_scores_s, tb_ratios_s, fee_rate=0.0004):
    """
    Finds best parameters on rolling lookback window.
    Search space is small for speed.
    """
    z_space = [-1.5, -1.2, -1.0]
    tb_space = [0.49, 0.50]
    activation_space = [0.5, 0.8]
    trailing_space = [0.2] # fixed trailing pct
    sl_space = [-2.0, -3.0]
    
    best_cap = 1000000.0
    best_p = (-1.2, 0.49, 0.5, 0.2, -2.0)
    
    for z in z_space:
        for tb in tb_space:
            for act in activation_space:
                for trail in trailing_space:
                    for sl in sl_space:
                        trades = run_trailing_stop_sim_fast(
                            lows_s, highs_s, closes_s, z_scores_s, tb_ratios_s,
                            z, tb, act, trail, sl
                        )
                        if len(trades) == 0:
                            continue
                        cap = 1000000.0
                        for t in trades:
                            ret = (t['weight'] * t['profit_pct']) / 100.0
                            fee = t['weight'] * fee_rate
                            cap *= (1 + ret - fee)
                        if cap > best_cap:
                            best_cap = cap
                            best_p = (z, tb, act, trail, sl)
    return best_p

if __name__ == "__main__":
    df = load_data()
    df = calculate_indicators(df)
    
    lows = df["low"].to_numpy()
    highs = df["high"].to_numpy()
    closes = df["close"].to_numpy()
    z_scores = df["z_score"].to_numpy()
    tb_ratios = df["taker_buy_ratio_smoothed"].to_numpy()
    open_times = df["open_time"].to_numpy()
    
    print("=========================================================")
    print("  Walk-Forward Trailing Stop Simulation (XRP 2 Years)")
    print("=========================================================")
    print("D-60 Lookback, Weekly Parameter Update, Maker Fee 0.04%\n")
    
    lookback_size = 60 * 96
    step_size = 7 * 96
    n_candles = len(df)
    
    capital = 1000000.0
    total_trades_count = 0
    wins_count = 0
    all_executed_trades = []
    
    start_idx = lookback_size
    start_time = time.time()
    
    while start_idx < n_candles:
        opt_start = start_idx - lookback_size
        opt_end = start_idx
        
        # Optimize on last 60 days
        best_z, best_tb, best_act, best_trail, best_sl = find_best_params_trailing(
            lows[opt_start:opt_end], highs[opt_start:opt_end], closes[opt_start:opt_end],
            z_scores[opt_start:opt_end], tb_ratios[opt_start:opt_end]
        )
        
        # Test on next 7 days
        test_end = min(start_idx + step_size, n_candles)
        test_trades = run_trailing_stop_sim_fast(
            lows[start_idx:test_end], highs[start_idx:test_end], closes[start_idx:test_end],
            z_scores[start_idx:test_end], tb_ratios[start_idx:test_end],
            best_z, best_tb, best_act, best_trail, best_sl
        )
        
        # Compound profits
        for t in test_trades:
            ret = (t['weight'] * t['profit_pct']) / 100.0
            fee = t['weight'] * 0.0004
            capital *= (1 + ret - fee)
            
            total_trades_count += 1
            if t['win']:
                wins_count += 1
                
            all_executed_trades.append({
                "time": open_times[start_idx + t['idx']],
                "profit_pct": t['profit_pct'],
                "weight": t['weight'],
                "win": t['win'],
                "capital": capital,
                "z": best_z,
                "tb": best_tb,
                "activation": best_act,
                "trailing": best_trail,
                "sl": best_sl
            })
            
        start_idx += step_size
        
    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds.")
    
    win_rate = (wins_count / total_trades_count * 100) if total_trades_count > 0 else 0.0
    net_return = (capital - 1000000.0) / 1000000.0 * 100
    
    print("\n=========================================================")
    print("          Walk-Forward Trailing Stop Results (XRP)")
    print("=========================================================")
    print(f" Total Trades  : {total_trades_count}")
    print(f" Wins / Losses : {wins_count} / {total_trades_count - wins_count}")
    print(f" Win Rate      : {win_rate:.2f}%")
    print(f" Initial Capital: 1,000,000 KRW")
    print(f" Final Capital  : {capital:,.0f} KRW")
    print(f" Real Net Return: {net_return:.2f}% (After 0.04% Maker Fees)")
    print("=========================================================")
    
    # Print sample timeline log
    if len(all_executed_trades) > 0:
        trades_df = pd.DataFrame(all_executed_trades)
        print("\nSample Walk-Forward Trailing Parameter Log:")
        print(trades_df.iloc[::max(1, len(trades_df)//10)].to_string(index=False))
