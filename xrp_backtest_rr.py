import pandas as pd
import numpy as np
import os

def load_data():
    csv_path = "/Users/stoni/Projects/AI/xrp_2y_data.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError("Cached data not found. Run the previous script first.")
    df = pd.read_csv(csv_path)
    df["open_time"] = pd.to_datetime(df["open_time"])
    return df

def calculate_indicators(df):
    # Bollinger Bands (20, 2)
    df["bb_middle"] = df["close"].rolling(window=20).mean()
    df["bb_std"] = df["close"].rolling(window=20).std()
    df["bb_lower"] = df["bb_middle"] - (df["bb_std"] * 2)
    return df

def simulate_scaling_in(df, target_profit, stop_loss):
    """
    Simulates 3-step scaling-in on Bollinger Bands lower band touch.
    - 1차: Buy 30% at BB Lower touch
    - 2차: Buy 30% at 1차 entry * 0.985 (-1.5%)
    - 3차: Buy 40% at 1차 entry * 0.970 (-3.0%)
    - TP: avg_price * (1 + target_profit / 100)
    - SL: avg_price * (1 + stop_loss / 100)
    """
    trades = []
    in_trade = False
    
    level_1_price = 0.0
    level_2_filled = False
    level_3_filled = False
    
    for i in range(len(df)):
        if pd.isna(df.loc[i, "bb_lower"]):
            continue
            
        low = df.loc[i, "low"]
        high = df.loc[i, "high"]
        close = df.loc[i, "close"]
        
        if not in_trade:
            # 1차 Entry
            if close < df.loc[i, "bb_lower"]:
                level_1_price = close
                level_2_filled = False
                level_3_filled = False
                in_trade = True
        else:
            # Check for 2차 and 3차 fills during the trade
            target_level_2 = level_1_price * 0.985
            target_level_3 = level_1_price * 0.970
            
            if not level_2_filled and low <= target_level_2:
                level_2_filled = True
            if not level_3_filled and low <= target_level_3:
                level_3_filled = True
                
            # Compute current average price and weight
            prices = [level_1_price]
            weights = [0.3]
            if level_2_filled:
                prices.append(target_level_2)
                weights.append(0.3)
            if level_3_filled:
                prices.append(target_level_3)
                weights.append(0.4)
                
            avg_price = sum(p * w for p, w in zip(prices, weights)) / sum(weights)
            total_weight = sum(weights)
            
            # Check exit conditions
            pct_high = (high - avg_price) / avg_price * 100
            pct_low = (low - avg_price) / avg_price * 100
            
            hit_tp = pct_high >= target_profit
            hit_sl = pct_low <= stop_loss
            
            if hit_tp and hit_sl:
                # worst case: hit stop loss
                trades.append({'weight': total_weight, 'profit_pct': stop_loss, 'win': False})
                in_trade = False
            elif hit_tp:
                trades.append({'weight': total_weight, 'profit_pct': target_profit, 'win': True})
                in_trade = False
            elif hit_sl:
                trades.append({'weight': total_weight, 'profit_pct': stop_loss, 'win': False})
                in_trade = False
                
    return trades

def evaluate_performance(trades, fee_rate=0.001):
    if len(trades) == 0:
        return 0, 0.0, 1000000.0, 1000000.0
        
    total_trades = len(trades)
    wins = sum(1 for t in trades if t['win'])
    win_rate = wins / total_trades * 100
    
    # Capital simulation
    capital_no_fee = 1000000.0
    capital_with_fee = 1000000.0
    
    for t in trades:
        ret = (t['weight'] * t['profit_pct']) / 100.0
        
        # Compounding without fee
        capital_no_fee *= (1 + ret)
        
        # Compounding with fee (fee rate is round-trip, e.g. 0.1% on traded position size)
        fee_drag = t['weight'] * fee_rate
        capital_with_fee *= (1 + ret - fee_drag)
        
    return total_trades, win_rate, capital_no_fee, capital_with_fee

if __name__ == "__main__":
    df = load_data()
    df = calculate_indicators(df)
    
    print("Evaluating Risk-Reward Matrix for 3-Step Scaling In strategy...")
    print("Dataset: XRPUSDT 15m, 2 Years (June 2024 - June 2026)")
    print("Trading Fee: 0.1% (round-trip) on traded size\n")
    
    results = []
    
    # Test matrix of TP and SL
    tp_list = [1.0, 1.5, 2.0, 2.5, 3.0]
    sl_list = [-3.0, -4.0, -5.0]
    
    for sl in sl_list:
        for tp in tp_list:
            trades = simulate_scaling_in(df, target_profit=tp, stop_loss=sl)
            total_trd, wr, cap_nofee, cap_fee = evaluate_performance(trades, fee_rate=0.001)
            
            results.append({
                "TP": tp,
                "SL": sl,
                "Trades": total_trd,
                "WinRate": wr,
                "CapNoFee": cap_nofee,
                "CapWithFee": cap_fee,
                "ReturnFee": (cap_fee - 1000000.0) / 1000000.0 * 100
            })
            
    # Print results in a nice table
    print(f"{'TP (%)':<8} | {'SL (%)':<8} | {'Trades':<8} | {'Win Rate':<10} | {'Final Cap (No Fee)':<20} | {'Final Cap (With Fee)':<22} | {'Net Return':<12}")
    print("-" * 100)
    for r in results:
        print(f"{r['TP']:<8.1f} | {r['SL']:<8.1f} | {r['Trades']:<8} | {r['WinRate']:<9.2f}% | {r['CapNoFee']:<18,.0f} KRW | {r['CapWithFee']:<20,.0f} KRW | {r['ReturnFee']:<10.2f}%")
