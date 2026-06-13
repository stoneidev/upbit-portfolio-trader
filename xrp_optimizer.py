import pandas as pd
import numpy as np
import os

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
    
    df["sma_200"] = df["close"].rolling(window=200).mean()
    
    df["taker_buy_ratio"] = df["taker_buy_volume"] / (df["volume"] + 1e-9)
    df["taker_buy_ratio_smoothed"] = df["taker_buy_ratio"].rolling(window=3).mean()
    
    # ATR
    high = df["high"]
    low = df["low"]
    close_prev = df["close"].shift(1)
    tr = pd.concat([high - low, (high - close_prev).abs(), (low - close_prev).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(window=14).mean()
    
    return df

def run_sim_fast(lows, highs, closes, sma_200, atr, z_score, tb_ratio, z_thresh, tb_thresh, atr_tp_mult, atr_sl_mult, use_trend, fee_rate=0.001):
    trades_profit = []
    trades_weight = []
    trades_win = []
    
    in_trade = False
    level_1_price = 0.0
    atr_entry = 0.0
    level_2_filled = False
    level_3_filled = False
    
    n = len(closes)
    
    for i in range(n):
        # Skip NaNs
        if np.isnan(sma_200[i]) or np.isnan(atr[i]) or np.isnan(z_score[i]):
            continue
            
        low = lows[i]
        high = highs[i]
        close = closes[i]
        
        if not in_trade:
            # Check entry
            c1 = True if not use_trend else (close > sma_200[i])
            c2 = z_score[i] < z_thresh
            c3 = tb_ratio[i] > tb_thresh
            
            if c1 and c2 and c3:
                level_1_price = close
                atr_entry = atr[i]
                level_2_filled = False
                level_3_filled = False
                in_trade = True
        else:
            # Grid
            target_level_2 = level_1_price - (1.0 * atr_entry)
            target_level_3 = level_1_price - (2.0 * atr_entry)
            
            if not level_2_filled and low <= target_level_2:
                level_2_filled = True
            if not level_3_filled and low <= target_level_3:
                level_3_filled = True
                
            # Compute average price and weight
            # 1차: weight 0.3
            # 2차: weight 0.3
            # 3차: weight 0.4
            if level_3_filled: # level 2 must also be filled
                avg_price = (level_1_price * 0.3 + target_level_2 * 0.3 + target_level_3 * 0.4)
                total_weight = 1.0
            elif level_2_filled:
                avg_price = (level_1_price * 0.3 + target_level_2 * 0.3) / 0.6
                total_weight = 0.6
            else:
                avg_price = level_1_price
                total_weight = 0.3
                
            tp_price = avg_price + (atr_tp_mult * atr_entry)
            sl_price = avg_price - (atr_sl_mult * atr_entry)
            
            hit_tp = high >= tp_price
            hit_sl = low <= sl_price
            
            if hit_tp and hit_sl:
                # worst case
                profit = (sl_price - avg_price) / avg_price * 100
                trades_profit.append(profit)
                trades_weight.append(total_weight)
                trades_win.append(False)
                in_trade = False
            elif hit_tp:
                profit = (tp_price - avg_price) / avg_price * 100
                trades_profit.append(profit)
                trades_weight.append(total_weight)
                trades_win.append(True)
                in_trade = False
            elif hit_sl:
                profit = (sl_price - avg_price) / avg_price * 100
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
    sma_200 = df["sma_200"].to_numpy()
    atr = df["atr"].to_numpy()
    z_score = df["z_score"].to_numpy()
    tb_ratio = df["taker_buy_ratio_smoothed"].to_numpy()
    
    print("Searching for profitable conditions on XRPUSDT 2-year data (Fast Mode)...")
    print("Filters: Win Rate >= 80%, Trades >= 40, Profit > 0 (after 0.1% fees)\n")
    
    # Define search spaces
    z_space = [-2.2, -2.0, -1.8, -1.6, -1.4]
    tb_space = [0.49, 0.50, 0.51, 0.52]
    tp_mult_space = [1.0, 1.2, 1.5, 1.8, 2.0]
    sl_mult_space = [2.5, 3.0, 3.5, 4.0]
    trend_space = [True, False]
    
    valid_results = []
    
    # Run loop
    import time
    start = time.time()
    for z in z_space:
        for tb in tb_space:
            for tp in tp_mult_space:
                for sl in sl_mult_space:
                    for trend in trend_space:
                        total_trd, wr, cap = run_sim_fast(
                            lows, highs, closes, sma_200, atr, z_score, tb_ratio,
                            z, tb, tp, sl, trend, fee_rate=0.001
                        )
                        net_return = (cap - 1000000.0) / 1000000.0 * 100
                        
                        if wr >= 80.0 and total_trd >= 40 and net_return > 0.0:
                            valid_results.append({
                                "z": z,
                                "tb": tb,
                                "tp_mult": tp,
                                "sl_mult": sl,
                                "trend": trend,
                                "trades": total_trd,
                                "win_rate": wr,
                                "final_cap": cap,
                                "return": net_return
                            })
                            
    end = time.time()
    print(f"Grid search completed in {end - start:.2f} seconds.")
    
    # Print top 15 results
    valid_results = sorted(valid_results, key=lambda x: x["return"], reverse=True)
    
    if len(valid_results) == 0:
        print("No combinations met the criteria of Win Rate >= 80%, Trades >= 40, and Net Profit > 0.")
    else:
        print(f"Found {len(valid_results)} matching combinations. Top 15:")
        print(f"{'Z-Thresh':<8} | {'TB-Thresh':<9} | {'TP Mult':<7} | {'SL Mult':<7} | {'Trend':<5} | {'Trades':<6} | {'Win Rate':<10} | {'Final Capital':<18} | {'Net Return':<12}")
        print("-" * 105)
        for r in valid_results[:15]:
            print(f"{r['z']:<8} | {r['tb']:<9} | {r['tp_mult']:<7.1f} | {r['sl_mult']:<7.1f} | {str(r['trend']):<5} | {r['trades']:<6} | {r['win_rate']:<9.2f}% | {r['final_cap']:<16,.0f} KRW | {r['return']:<10.2f}%")
