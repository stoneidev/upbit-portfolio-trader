import requests
import pandas as pd
import numpy as np
import time
import os

def fetch_xrp_data(symbol="XRPUSDT", interval="15m", start_time_ms=1718236800000):
    """
    Fetches 2 years of XRP data starting from 2024-06-13 (1718236800000 ms) to present.
    Saves to CSV to cache the data.
    """
    csv_path = "/Users/stoni/Projects/AI/xrp_2y_data.csv"
    if os.path.exists(csv_path):
        print("Loading cached XRP 2-year data from CSV...")
        df = pd.read_csv(csv_path)
        df["open_time"] = pd.to_datetime(df["open_time"])
        return df

    print(f"Fetching 2 years of {symbol} ({interval}) from Binance...")
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    current_start = start_time_ms
    page = 1
    
    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "limit": 1000
        }
        try:
            response = requests.get(url, params=params)
            data = response.json()
            if not data or len(data) == 0:
                break
            
            all_data += data
            last_time = data[-1][0]
            current_start = last_time + 1
            
            if page % 10 == 0:
                print(f"Fetched {len(all_data)} candles (Page {page})...")
                
            # If we are close to current time, stop
            if last_time >= int(time.time() * 1000) - 15 * 60 * 1000:
                break
                
            page += 1
            time.sleep(0.05) # short polite delay
        except Exception as e:
            print(f"Error fetching: {e}")
            break
            
    df = pd.DataFrame(all_data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "count", "taker_buy_volume",
        "taker_buy_quote_volume", "ignore"
    ])
    
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])
        
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.sort_values("open_time").reset_index(drop=True)
    
    # Save cache
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} candles to {csv_path}")
    return df

def calculate_indicators(df):
    # Bollinger Bands (20, 2)
    df["bb_middle"] = df["close"].rolling(window=20).mean()
    df["bb_std"] = df["close"].rolling(window=20).std()
    df["bb_lower"] = df["bb_middle"] - (df["bb_std"] * 2)
    
    # Trend Filter (200 SMA on 15m)
    df["sma_200"] = df["close"].rolling(window=200).mean()
    
    # RSI (14)
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))
    
    return df

def run_standard_bb_backtest(df, use_trend_filter=False, stop_loss=-5.0):
    """
    Standard Bollinger Bands Pullback strategy.
    - Buy if Close < bb_lower
    - If use_trend_filter is True, also check if Close > sma_200
    - TP: +1.0%, SL: stop_loss%
    """
    trades = []
    in_trade = False
    entry_price = 0.0
    entry_time = None
    
    for i in range(len(df)):
        if pd.isna(df.loc[i, "bb_lower"]) or (use_trend_filter and pd.isna(df.loc[i, "sma_200"])):
            continue
            
        if not in_trade:
            # Signal
            signal = df.loc[i, "close"] < df.loc[i, "bb_lower"]
            if use_trend_filter:
                signal = signal and (df.loc[i, "close"] > df.loc[i, "sma_200"])
                
            if signal:
                entry_price = df.loc[i, "close"]
                entry_time = df.loc[i, "open_time"]
                in_trade = True
        else:
            high = df.loc[i, "high"]
            low = df.loc[i, "low"]
            current_time = df.loc[i, "open_time"]
            
            pct_high = (high - entry_price) / entry_price * 100
            pct_low = (low - entry_price) / entry_price * 100
            
            hit_tp = pct_high >= 1.0
            hit_sl = pct_low <= stop_loss
            
            if hit_tp and hit_sl:
                trades.append({"profit": stop_loss, "win": False})
                in_trade = False
            elif hit_tp:
                trades.append({"profit": 1.0, "win": True})
                in_trade = False
            elif hit_sl:
                trades.append({"profit": stop_loss, "win": False})
                in_trade = False
                
    return trades

def run_scaling_bb_backtest(df, stop_loss=-3.0):
    """
    Scaling In (3-step Grid) Strategy.
    - 1차 Buy: Close < bb_lower (30% weight)
    - 2차 Buy: Limit order at 1차 entry price - 1.5% (30% weight)
    - 3차 Buy: Limit order at 1차 entry price - 3.0% (40% weight)
    - TP: +1.0% of average entry price
    - SL: stop_loss% of average entry price
    """
    trades = []
    in_trade = False
    
    # State variables
    entry_prices = []
    weights = []
    avg_price = 0.0
    
    # Levels based on 1차 entry price
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
                entry_prices = [level_1_price]
                weights = [0.3]
                avg_price = level_1_price
                level_2_filled = False
                level_3_filled = False
                in_trade = True
        else:
            # Check 2차 / 3차 fills during the current bar
            target_level_2 = level_1_price * 0.985
            target_level_3 = level_1_price * 0.970
            
            # Since we only have high/low, we check if low went below target
            if not level_2_filled and low <= target_level_2:
                # 2차 filled (assume executed at target_level_2)
                entry_prices.append(target_level_2)
                weights.append(0.3)
                # recalculate average price
                avg_price = sum(p * w for p, w in zip(entry_prices, weights)) / sum(weights)
                level_2_filled = True
                
            if not level_3_filled and low <= target_level_3:
                # 3차 filled
                entry_prices.append(target_level_3)
                weights.append(0.4)
                avg_price = sum(p * w for p, w in zip(entry_prices, weights)) / sum(weights)
                level_3_filled = True
                
            # Check exit conditions based on average entry price
            pct_high = (high - avg_price) / avg_price * 100
            pct_low = (low - avg_price) / avg_price * 100
            
            hit_tp = pct_high >= 1.0
            hit_sl = pct_low <= stop_loss
            
            if hit_tp and hit_sl:
                # worst case
                trades.append({"profit": stop_loss, "win": False})
                in_trade = False
            elif hit_tp:
                trades.append({"profit": 1.0, "win": True})
                in_trade = False
            elif hit_sl:
                trades.append({"profit": stop_loss, "win": False})
                in_trade = False
                
    return trades

def run_extreme_rsi_backtest(df, rsi_threshold=20, stop_loss=-2.0):
    """
    Extreme oversold strategy.
    - Buy if RSI < rsi_threshold
    - TP: +1.0%, SL: stop_loss%
    """
    trades = []
    in_trade = False
    entry_price = 0.0
    
    for i in range(len(df)):
        if pd.isna(df.loc[i, "rsi"]):
            continue
            
        if not in_trade:
            if df.loc[i, "rsi"] < rsi_threshold:
                entry_price = df.loc[i, "close"]
                in_trade = True
        else:
            high = df.loc[i, "high"]
            low = df.loc[i, "low"]
            
            pct_high = (high - entry_price) / entry_price * 100
            pct_low = (low - entry_price) / entry_price * 100
            
            hit_tp = pct_high >= 1.0
            hit_sl = pct_low <= stop_loss
            
            if hit_tp and hit_sl:
                trades.append({"profit": stop_loss, "win": False})
                in_trade = False
            elif hit_tp:
                trades.append({"profit": 1.0, "win": True})
                in_trade = False
            elif hit_sl:
                trades.append({"profit": stop_loss, "win": False})
                in_trade = False
                
    return trades

def analyze_trades(trades, name):
    if len(trades) == 0:
        print(f"\n[{name}] No trades executed.")
        return
        
    df_trades = pd.DataFrame(trades)
    total = len(df_trades)
    wins = len(df_trades[df_trades["win"] == True])
    win_rate = wins / total * 100
    total_return = df_trades["profit"].sum()
    
    # Compound growth simulation
    capital = 1000000.0
    for t in trades:
        # Simple compound (position size is 100% of capital)
        # Note: for scaling in, the total weight is up to 1.0 (100% of capital)
        capital *= (1 + t["profit"] / 100.0)
        
    print(f"\n=========================================")
    print(f" Strategy: {name}")
    print(f"=========================================")
    print(f" Total Trades  : {total}")
    print(f" Wins / Losses : {wins} / {total - wins}")
    print(f" Win Rate      : {win_rate:.2f}%")
    print(f" Total Return  : {total_return:.2f}% (Arithmetic Sum)")
    print(f" Final Capital : {capital:,.0f} KRW (from 1,000,000 KRW)")
    print(f" Net Return (%) : {(capital - 1000000.0) / 1000000.0 * 100:.2f}%")
    print(f"=========================================")

if __name__ == "__main__":
    # Fetch 2 years XRP data (2024-06-13 to 2026-06-13)
    df = fetch_xrp_data()
    print(f"Calculating indicators for {len(df)} candles...")
    df = calculate_indicators(df)
    
    # 1. Baseline: BB Lower, TP 1%, SL -5%
    trades_baseline = run_standard_bb_backtest(df, use_trend_filter=False, stop_loss=-5.0)
    analyze_trades(trades_baseline, "1. Baseline BB Pullback (TP +1%, SL -5%)")
    
    # 2. Trend Filter: BB Lower + Above SMA 200, TP 1%, SL -3% (tightened SL because of trend filter)
    trades_trend = run_standard_bb_backtest(df, use_trend_filter=True, stop_loss=-3.0)
    analyze_trades(trades_trend, "2. BB Pullback + Trend Filter (TP +1%, SL -3%)")
    
    # 3. Scaling In: BB Lower + 3-step Grid (TP +1%, SL -3% from average price)
    trades_scale = run_scaling_bb_backtest(df, stop_loss=-3.0)
    analyze_trades(trades_scale, "3. BB Pullback + 3-Step Scaling In (TP +1%, SL -3% avg)")
    
    # 4. Extreme Panic Filter: RSI < 20, TP 1%, SL -2% (Tight SL)
    trades_rsi20 = run_extreme_rsi_backtest(df, rsi_threshold=20, stop_loss=-2.0)
    analyze_trades(trades_rsi20, "4. Extreme Panic (RSI < 20) (TP +1%, SL -2%)")
