import requests
import pandas as pd
import numpy as np
import time

def fetch_binance_klines_from_date(symbol="BTCUSDT", interval="15m", start_time_ms=1767225600000):
    """
    Fetches historical OHLCV data from Binance starting from a specific timestamp (in ms).
    Jan 1, 2026 00:00:00 UTC is 1767225600000 ms.
    """
    print(f"Fetching candles for {symbol} ({interval}) starting from 2026-01-01...")
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    current_start = start_time_ms
    
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
            # Set next startTime to the open time of the last candle fetched + 1 ms
            last_time = data[-1][0]
            current_start = last_time + 1
            
            # If we are close to current time, stop
            if last_time >= int(time.time() * 1000) - 15 * 60 * 1000:
                break
                
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
    return df

def calculate_bollinger_bands(df, period=20, std_dev=2):
    df["bb_middle"] = df["close"].rolling(window=period).mean()
    df["bb_std"] = df["close"].rolling(window=period).std()
    df["bb_lower"] = df["bb_middle"] - (df["bb_std"] * std_dev)
    return df

def run_backtest_trades(df, entry_signal_col, target_profit=1.0, stop_loss=-5.0):
    trades = []
    in_trade = False
    entry_price = 0.0
    entry_time = None
    
    for i in range(len(df)):
        if not in_trade:
            if df.loc[i, entry_signal_col] == True:
                entry_price = df.loc[i, "close"]
                entry_time = df.loc[i, "open_time"]
                in_trade = True
        else:
            high_price = df.loc[i, "high"]
            low_price = df.loc[i, "low"]
            current_time = df.loc[i, "open_time"]
            
            pct_high = (high_price - entry_price) / entry_price * 100
            pct_low = (low_price - entry_price) / entry_price * 100
            
            hit_tp = pct_high >= target_profit
            hit_sl = pct_low <= stop_loss
            
            if hit_tp and hit_sl:
                # worst case
                trades.append({"time": current_time, "profit_pct": stop_loss, "win": False})
                in_trade = False
            elif hit_tp:
                trades.append({"time": current_time, "profit_pct": target_profit, "win": True})
                in_trade = False
            elif hit_sl:
                trades.append({"time": current_time, "profit_pct": stop_loss, "win": False})
                in_trade = False
                
    if in_trade:
        last_idx = len(df) - 1
        close_price = df.loc[last_idx, "close"]
        pct = (close_price - entry_price) / entry_price * 100
        trades.append({"time": df.loc[last_idx, "open_time"], "profit_pct": pct, "win": pct >= target_profit})
        
    return trades

def simulate_equity(trades, initial_capital=1000000, risk_fraction=0.10, max_leverage=1.0):
    """
    Simulates compounding equity.
    risk_fraction: the fraction of capital we are willing to lose if stop loss is hit (-5%).
                   If risk_fraction is 10%, it means we lose 10% of our bankroll on a losing trade.
                   Since stop loss is -5%, the position size is: (risk_fraction / 0.05) * capital.
                   If max_leverage is 1.0, position size cannot exceed capital.
    """
    capital = initial_capital
    history = []
    
    for trade in trades:
        # Calculate position size
        # Max loss is 5% (i.e. -5%)
        # Position Size = Capital * risk_fraction / 0.05
        position_size = capital * (risk_fraction / 0.05)
        
        # Apply leverage limit
        if position_size > capital * max_leverage:
            position_size = capital * max_leverage
            
        trade_profit = position_size * (trade["profit_pct"] / 100)
        capital += trade_profit
        
        # Avoid bankrupt
        if capital <= 0:
            capital = 0
            
        history.append({
            "time": trade["time"],
            "profit_pct": trade["profit_pct"],
            "win": trade["win"],
            "capital": capital,
            "pos_size": position_size
        })
        
        if capital == 0:
            break
            
    return capital, history

if __name__ == "__main__":
    # Fetch 2026 data
    df = fetch_binance_klines_from_date(symbol="BTCUSDT", interval="15m", start_time_ms=1767225600000)
    print(f"Total candles fetched: {len(df)}")
    
    df = calculate_bollinger_bands(df)
    # Buy signal when close crosses below lower band
    df["bb_buy_signal"] = (df["close"] < df["bb_lower"]) & (df["close"].shift(1) >= df["bb_lower"].shift(1))
    
    trades = run_backtest_trades(df, "bb_buy_signal", target_profit=1.0, stop_loss=-5.0)
    total_trades = len(trades)
    wins = sum(1 for t in trades if t["win"])
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    print(f"\nTrade Statistics from Jan 1, 2026 to present:")
    print(f"Total Trades: {total_trades}")
    print(f"Wins: {wins} / Losses: {total_trades - wins}")
    print(f"Win Rate: {win_rate:.2f}%")
    
    # ----------------------------------------------------
    # Calculate Kelly Fraction
    # f* = p - q/b
    # For our trade setup:
    # win_profit = +1%
    # loss_loss = -5%
    # b = 1/5 = 0.2
    # p = win_rate / 100
    # q = 1 - p
    # ----------------------------------------------------
    p = win_rate / 100.0
    q = 1.0 - p
    b = 0.2
    kelly_fraction = p - (q / b)
    print(f"\nCalculated Kelly Fraction (f*): {kelly_fraction:.4f}")
    
    # Simulate different money management styles:
    # 1. No Money Management (All-in position size = Capital, meaning loss = 5% of capital)
    #    This corresponds to risk_fraction = 5% (since position size = Capital * 0.05/0.05 = Capital)
    cap1, hist1 = simulate_equity(trades, initial_capital=1000000, risk_fraction=0.05, max_leverage=1.0)
    
    # 2. Kelly Criterion (risk_fraction = kelly_fraction)
    #    Since kelly_fraction might be negative, we handle that case.
    cap2, hist2 = simulate_equity(trades, initial_capital=1000000, risk_fraction=max(0, kelly_fraction), max_leverage=1.0)
    
    # 3. Leveraged Kelly (allowing max_leverage = 3.0)
    cap3, hist3 = simulate_equity(trades, initial_capital=1000000, risk_fraction=max(0, kelly_fraction), max_leverage=3.0)
    
    # 4. Conservative Half-Kelly (risk_fraction = kelly_fraction / 2)
    cap4, hist4 = simulate_equity(trades, initial_capital=1000000, risk_fraction=max(0, kelly_fraction / 2), max_leverage=1.0)
    
    # 5. Over-aggressive (Risking 20% of capital per trade)
    cap5, hist5 = simulate_equity(trades, initial_capital=1000000, risk_fraction=0.20, max_leverage=4.0)

    print(f"\nSimulation Results (Initial Capital: 1,000,000 KRW):")
    print(f"1. Standard All-in (No leverage, 100% position size): {cap1:,.0f} KRW")
    if kelly_fraction > 0:
        print(f"2. Kelly Criterion (f*={kelly_fraction:.2f}, max 1x leverage): {cap2:,.0f} KRW")
        print(f"3. Kelly Criterion (f*={kelly_fraction:.2f}, max 3x leverage): {cap3:,.0f} KRW")
        print(f"4. Half-Kelly (f*={kelly_fraction/2:.2f}, max 1x leverage): {cap4:,.0f} KRW")
    else:
        print("Kelly Criterion suggested not playing (f* <= 0).")
    print(f"5. Aggressive (Risk 20% per trade, max 4x leverage): {cap5:,.0f} KRW")
