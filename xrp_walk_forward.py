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

def run_scalper_sim_fast(lows, highs, closes, z_scores, tb_ratios, z_thresh, tb_thresh, tp_pct, sl_pct):
    """
    Returns lists of trades (profit_pct, weight, win, index) for a specific slice of data.
    """
    trades = []
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
            c1 = z_scores[i] < z_thresh
            c2 = tb_ratios[i] > tb_thresh
            
            if c1 and c2:
                level_1_price = close
                level_2_filled = False
                level_3_filled = False
                in_trade = True
        else:
            target_level_2 = level_1_price * 0.990
            target_level_3 = level_1_price * 0.980
            
            if not level_2_filled and low <= target_level_2:
                level_2_filled = True
            if not level_3_filled and low <= target_level_3:
                level_3_filled = True
                
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
                trades.append({'weight': total_weight, 'profit_pct': sl_pct, 'win': False, 'idx': i})
                in_trade = False
            elif hit_tp:
                trades.append({'weight': total_weight, 'profit_pct': tp_pct, 'win': True, 'idx': i})
                in_trade = False
            elif hit_sl:
                trades.append({'weight': total_weight, 'profit_pct': sl_pct, 'win': False, 'idx': i})
                in_trade = False
                
    return trades

def find_best_params_fast(lows_slice, highs_slice, closes_slice, z_scores_slice, tb_ratios_slice, fee_rate=0.0004):
    """
    Finds the parameters that yield the highest final capital over the 60-day lookback slice.
    """
    # Grid search space
    z_space = [-1.8, -1.5, -1.2, -1.0]
    tb_space = [0.49, 0.50, 0.51]
    tp_space = [0.5, 0.8, 1.0]
    sl_space = [-2.0, -3.0, -4.0]
    
    best_cap = 1000000.0
    # Default parameters in case nothing is profitable
    best_params = (-1.2, 0.49, 0.5, -2.0)
    
    for z in z_space:
        for tb in tb_space:
            for tp in tp_space:
                for sl in sl_space:
                    trades = run_scalper_sim_fast(
                        lows_slice, highs_slice, closes_slice, z_scores_slice, tb_ratios_slice,
                        z, tb, tp, sl
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
                        best_params = (z, tb, tp, sl)
                        
    return best_params

if __name__ == "__main__":
    df = load_data()
    df = calculate_indicators(df)
    
    lows = df["low"].to_numpy()
    highs = df["high"].to_numpy()
    closes = df["close"].to_numpy()
    z_scores = df["z_score"].to_numpy()
    tb_ratios = df["taker_buy_ratio_smoothed"].to_numpy()
    open_times = df["open_time"].to_numpy()
    
    print("Running Walk-Forward Optimization (Rolling 60 Days Lookback, Weekly Update)...")
    print("Dataset: XRPUSDT 2 Years (15m candles)")
    print("Transaction Fee: 0.04% Maker Fee (Limit Order)\n")
    
    # 60 days = 60 * 96 = 5760 candles
    lookback_size = 60 * 96
    # 7 days = 7 * 96 = 672 candles
    step_size = 7 * 96
    
    n_candles = len(df)
    capital = 1000000.0
    total_trades_count = 0
    wins_count = 0
    all_executed_trades = []
    
    # Start trading from lookback_size onwards
    start_idx = lookback_size
    
    start_time = time.time()
    
    # Loop over week increments
    while start_idx < n_candles:
        # Lookback window for optimization: [start_idx - lookback_size, start_idx]
        opt_start = start_idx - lookback_size
        opt_end = start_idx
        
        # Slices
        lows_opt = lows[opt_start:opt_end]
        highs_opt = highs[opt_start:opt_end]
        closes_opt = closes[opt_start:opt_end]
        z_scores_opt = z_scores[opt_start:opt_end]
        tb_ratios_opt = tb_ratios[opt_start:opt_end]
        
        # Find best parameters based on D-60 days history
        best_z, best_tb, best_tp, best_sl = find_best_params_fast(
            lows_opt, highs_opt, closes_opt, z_scores_opt, tb_ratios_opt, fee_rate=0.0004
        )
        
        # Test window (Next 7 Days): [start_idx, min(start_idx + step_size, n_candles)]
        test_end = min(start_idx + step_size, n_candles)
        
        lows_test = lows[start_idx:test_end]
        highs_test = highs[start_idx:test_end]
        closes_test = closes[start_idx:test_end]
        z_scores_test = z_scores[start_idx:test_end]
        tb_ratios_test = tb_ratios[start_idx:test_end]
        
        # Run backtest for the next 7 days using the optimized parameters
        test_trades = run_scalper_sim_fast(
            lows_test, highs_test, closes_test, z_scores_test, tb_ratios_test,
            best_z, best_tb, best_tp, best_sl
        )
        
        # Apply trade profits to capital
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
                "tp": best_tp,
                "sl": best_sl
            })
            
        start_idx += step_size
        
    end_time = time.time()
    print(f"Walk-Forward Simulation completed in {end_time - start_time:.2f} seconds.")
    
    win_rate = (wins_count / total_trades_count * 100) if total_trades_count > 0 else 0.0
    net_return = (capital - 1000000.0) / 1000000.0 * 100
    
    print("\n=========================================================")
    print("          Walk-Forward Simulation Results (XRP)")
    print("=========================================================")
    print(f" Total Trades  : {total_trades_count}")
    print(f" Wins / Losses : {wins_count} / {total_trades_count - wins_count}")
    print(f" Win Rate      : {win_rate:.2f}%")
    print(f" Initial Capital: 1,000,000 KRW")
    print(f" Final Capital  : {capital:,.0f} KRW")
    print(f" Real Net Return: {net_return:.2f}% (After 0.04% Maker Fees)")
    print("=========================================================")
    
    # Print some sample parameter updates
    if len(all_executed_trades) > 0:
        trades_df = pd.DataFrame(all_executed_trades)
        print("\nSample Walk-Forward Parameter Log & Capital Curve:")
        # Show a sample representing parameter changes over time
        print(trades_df.iloc[::max(1, len(trades_df)//10)].to_string(index=False))
