import pandas as pd
import numpy as np
import os

def load_data():
    csv_path = "/Users/stoni/Projects/AI/xrp_2y_data.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError("Cached data not found. Run the previous script first.")
    df = pd.read_csv(csv_path)
    df["open_time"] = pd.to_datetime(df["open_time"])
    
    # Cast necessary columns to numeric
    cols_to_cast = ["open", "high", "low", "close", "volume", "taker_buy_volume"]
    for col in cols_to_cast:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
    return df

def calculate_quant_indicators(df):
    # 1. Bollinger Bands (20, 2)
    df["bb_middle"] = df["close"].rolling(window=20).mean()
    df["bb_std"] = df["close"].rolling(window=20).std()
    df["bb_lower"] = df["bb_middle"] - (df["bb_std"] * 2)
    
    # 2. 200 SMA for Trend Filter
    df["sma_200"] = df["close"].rolling(window=200).mean()
    
    # 3. Z-Score of Close Price relative to 100 MA
    df["ma_100"] = df["close"].rolling(window=100).mean()
    df["std_100"] = df["close"].rolling(window=100).std()
    df["z_score"] = (df["close"] - df["ma_100"]) / (df["std_100"] + 1e-9)
    
    # 4. Taker Buy Ratio
    df["taker_buy_ratio"] = df["taker_buy_volume"] / (df["volume"] + 1e-9)
    # Smooth the ratio slightly to avoid raw noise (say, 3-period SMA)
    df["taker_buy_ratio_smoothed"] = df["taker_buy_ratio"].rolling(window=3).mean()
    
    # 5. ATR (Average True Range) calculation
    high = df["high"]
    low = df["low"]
    close_prev = df["close"].shift(1)
    
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = tr.rolling(window=14).mean()
    
    return df

def run_quant_backtest(df, atr_tp_multiplier=1.2, atr_sl_multiplier=2.5, use_grid=True, fee_rate=0.001):
    """
    Advanced Quant Strategy:
    - Entry Signal:
      1. close > sma_200 (Long-term uptrend)
      2. z_score < -2.0 (Price is extreme outlier below 100 MA)
      3. taker_buy_ratio_smoothed > 0.51 (Aggressive buyers step in)
    
    - Dynamic ATR Exit:
      - ATR-based Take Profit and Stop Loss.
      
    - If use_grid is True:
      - 1차 Buy: 30% weight at Signal Close
      - 2차 Buy: 30% weight at 1차 - 1.0 * ATR
      - 3차 Buy: 40% weight at 1차 - 2.0 * ATR
      - TP: avg_price + (atr_tp_multiplier * ATR_entry)
      - SL: avg_price - (atr_sl_multiplier * ATR_entry)
    """
    trades = []
    in_trade = False
    
    # Trade state
    level_1_price = 0.0
    atr_entry = 0.0
    level_2_filled = False
    level_3_filled = False
    
    for i in range(len(df)):
        if pd.isna(df.loc[i, "sma_200"]) or pd.isna(df.loc[i, "atr"]) or pd.isna(df.loc[i, "z_score"]):
            continue
            
        low = df.loc[i, "low"]
        high = df.loc[i, "high"]
        close = df.loc[i, "close"]
        
        if not in_trade:
            # Check entry conditions
            c1 = close > df.loc[i, "sma_200"]
            c2 = df.loc[i, "z_score"] < -2.0
            c3 = df.loc[i, "taker_buy_ratio_smoothed"] > 0.51
            
            if c1 and c2 and c3:
                level_1_price = close
                atr_entry = df.loc[i, "atr"]
                level_2_filled = False
                level_3_filled = False
                in_trade = True
        else:
            if use_grid:
                # 3-Step Grid based on ATR
                target_level_2 = level_1_price - (1.0 * atr_entry)
                target_level_3 = level_1_price - (2.0 * atr_entry)
                
                if not level_2_filled and low <= target_level_2:
                    level_2_filled = True
                if not level_3_filled and low <= target_level_3:
                    level_3_filled = True
                    
                # Compute average price and weight
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
                
                # Exits based on Average Price and ATR
                tp_price = avg_price + (atr_tp_multiplier * atr_entry)
                sl_price = avg_price - (atr_sl_multiplier * atr_entry)
                
                hit_tp = high >= tp_price
                hit_sl = low <= sl_price
                
                if hit_tp and hit_sl:
                    # Worst case: hit stop loss
                    trades.append({'weight': total_weight, 'profit_pct': (sl_price - avg_price)/avg_price * 100, 'win': False})
                    in_trade = False
                elif hit_tp:
                    trades.append({'weight': total_weight, 'profit_pct': (tp_price - avg_price)/avg_price * 100, 'win': True})
                    in_trade = False
                elif hit_sl:
                    trades.append({'weight': total_weight, 'profit_pct': (sl_price - avg_price)/avg_price * 100, 'win': False})
                    in_trade = False
            else:
                # Single Entry (100% position size)
                tp_price = level_1_price + (atr_tp_multiplier * atr_entry)
                sl_price = level_1_price - (atr_sl_multiplier * atr_entry)
                
                hit_tp = high >= tp_price
                hit_sl = low <= sl_price
                
                if hit_tp and hit_sl:
                    trades.append({'weight': 1.0, 'profit_pct': (sl_price - level_1_price)/level_1_price * 100, 'win': False})
                    in_trade = False
                elif hit_tp:
                    trades.append({'weight': 1.0, 'profit_pct': (tp_price - level_1_price)/level_1_price * 100, 'win': True})
                    in_trade = False
                elif hit_sl:
                    trades.append({'weight': 1.0, 'profit_pct': (sl_price - level_1_price)/level_1_price * 100, 'win': False})
                    in_trade = False
                    
    return trades

def analyze_quant_results(trades, name, fee_rate=0.001):
    if len(trades) == 0:
        print(f"\n=========================================\n Strategy: {name}\n=========================================\n No trades executed.")
        return
        
    total_trades = len(trades)
    wins = sum(1 for t in trades if t['win'])
    win_rate = wins / total_trades * 100
    
    capital_no_fee = 1000000.0
    capital_with_fee = 1000000.0
    
    total_weight_sum = 0.0
    avg_profit_raw = []
    
    for t in trades:
        ret = (t['weight'] * t['profit_pct']) / 100.0
        capital_no_fee *= (1 + ret)
        
        fee_drag = t['weight'] * fee_rate
        capital_with_fee *= (1 + ret - fee_drag)
        
        total_weight_sum += t['weight']
        avg_profit_raw.append(t['profit_pct'])
        
    avg_weight = total_weight_sum / total_trades
    avg_profit = np.mean(avg_profit_raw)
    
    print(f"\n=========================================")
    print(f" Strategy: {name}")
    print(f"=========================================")
    print(f" Total Trades  : {total_trades}")
    print(f" Wins / Losses : {wins} / {total_trades - wins}")
    print(f" Win Rate      : {win_rate:.2f}%")
    print(f" Avg Position  : {avg_weight * 100:.1f}% of capital")
    print(f" Avg Trade Pct : {avg_profit:.2f}%")
    print(f" Final Cap (No Fee)  : {capital_no_fee:,.0f} KRW")
    print(f" Final Cap (0.1% Fee): {capital_with_fee:,.0f} KRW")
    print(f" Net Return (Real)   : {(capital_with_fee - 1000000.0) / 1000000.0 * 100:.2f}%")
    
    weights_list = [t['weight'] for t in trades]
    print(f" Grid Distribution: 1차만 ({weights_list.count(0.3)}회), 1+2차 ({weights_list.count(0.6)}회), 1+2+3차 ({weights_list.count(1.0)}회)")
    print(f"=========================================")

if __name__ == "__main__":
    df = load_data()
    df = calculate_quant_indicators(df)
    
    print("Running Advanced Quant Backtest on XRPUSDT 15m (2 Years)...")
    print("Signals applied:")
    print(" - SMA 200 Trend Filter (close > sma_200)")
    print(" - Z-Score Extreme Oversold (z_score < -2.0)")
    print(" - Taker Buy Ratio buyers filter (smoothed_ratio > 51%)")
    print(" - Exit: Dynamic ATR (TP = 1.2 * ATR, SL = 2.5 * ATR)")
    
    # Run Single Entry Quant Strategy (No Grid)
    trades_single = run_quant_backtest(df, atr_tp_multiplier=1.2, atr_sl_multiplier=2.5, use_grid=False)
    analyze_quant_results(trades_single, "Advanced Quant (Single Entry, No Grid)")
    
    # Run Grid Scaling In Quant Strategy
    trades_grid = run_quant_backtest(df, atr_tp_multiplier=1.2, atr_sl_multiplier=2.5, use_grid=True)
    analyze_quant_results(trades_grid, "Advanced Quant + ATR Dynamic Grid (3-Step)")
