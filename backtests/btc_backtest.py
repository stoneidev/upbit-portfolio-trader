import requests
import pandas as pd
import numpy as np
import time

def fetch_binance_klines(symbol="BTCUSDT", interval="15m", limit=1000, pages=5):
    """
    Fetches historical OHLCV data from Binance.
    Each page returns 'limit' number of candles.
    """
    print(f"Fetching {pages * limit} candles of {symbol} ({interval})...")
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    end_time = None
    
    for i in range(pages):
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        if end_time:
            params["endTime"] = end_time - 1
            
        try:
            response = requests.get(url, params=params)
            data = response.json()
            if not data or len(data) == 0:
                break
            all_data = data + all_data
            # Set endTime to the open time of the oldest candle fetched
            end_time = data[0][0]
            time.sleep(0.1) # polite delay
        except Exception as e:
            print(f"Error fetching page {i+1}: {e}")
            break
            
    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "count", "taker_buy_volume",
        "taker_buy_quote_volume", "ignore"
    ])
    
    # Cast to numeric
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])
        
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.sort_values("open_time").reset_index(drop=True)
    return df

def calculate_rsi(df, period=14):
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / (loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))
    return df

def calculate_bollinger_bands(df, period=20, std_dev=2):
    df["bb_middle"] = df["close"].rolling(window=period).mean()
    df["bb_std"] = df["close"].rolling(window=period).std()
    df["bb_upper"] = df["bb_middle"] + (df["bb_std"] * std_dev)
    df["bb_lower"] = df["bb_middle"] - (df["bb_std"] * std_dev)
    return df

def run_backtest(df, entry_signal_col, target_profit=1.0, stop_loss=-1.0):
    """
    Runs a simple backtest where:
    - df: DataFrame with columns ['open', 'high', 'low', 'close', entry_signal_col]
    - entry_signal_col: Boolean column where True means Buy
    - target_profit: Take profit percentage (e.g. 1.0 means +1.0%)
    - stop_loss: Stop loss percentage (e.g. -1.0 means -1.0%). If None, no stop loss.
    """
    trades = []
    in_trade = False
    entry_price = 0.0
    entry_time = None
    
    for i in range(len(df)):
        if not in_trade:
            # Check for buy signal
            if df.loc[i, entry_signal_col] == True:
                entry_price = df.loc[i, "close"]
                entry_time = df.loc[i, "open_time"]
                in_trade = True
        else:
            # We are in a trade, check exit conditions using High and Low of the current candle
            high_price = df.loc[i, "high"]
            low_price = df.loc[i, "low"]
            close_price = df.loc[i, "close"]
            current_time = df.loc[i, "open_time"]
            
            pct_high = (high_price - entry_price) / entry_price * 100
            pct_low = (low_price - entry_price) / entry_price * 100
            
            # Check if both hit in the same candle (worst case scenario: assume stop loss hit first)
            hit_tp = pct_high >= target_profit
            hit_sl = stop_loss is not None and pct_low <= stop_loss
            
            if hit_tp and hit_sl:
                # Conservative assumption: hit SL first
                exit_price = entry_price * (1 + stop_loss / 100)
                trades.append({
                    "entry_time": entry_time,
                    "exit_time": current_time,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "profit_pct": stop_loss,
                    "win": False
                })
                in_trade = False
            elif hit_tp:
                exit_price = entry_price * (1 + target_profit / 100)
                trades.append({
                    "entry_time": entry_time,
                    "exit_time": current_time,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "profit_pct": target_profit,
                    "win": True
                })
                in_trade = False
            elif hit_sl:
                exit_price = entry_price * (1 + stop_loss / 100)
                trades.append({
                    "entry_time": entry_time,
                    "exit_time": current_time,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "profit_pct": stop_loss,
                    "win": False
                })
                in_trade = False
                
    # If still in trade at the end of data, close at last candle's close
    if in_trade:
        last_idx = len(df) - 1
        close_price = df.loc[last_idx, "close"]
        pct = (close_price - entry_price) / entry_price * 100
        trades.append({
            "entry_time": entry_time,
            "exit_time": df.loc[last_idx, "open_time"],
            "entry_price": entry_price,
            "exit_price": close_price,
            "profit_pct": pct,
            "win": pct >= target_profit
        })
        
    # Summarize results
    if len(trades) == 0:
        return {
            "total_trades": 0, "win_rate": 0, "total_return": 0,
            "avg_profit": 0, "max_drawdown": 0, "trades": []
        }
        
    trades_df = pd.DataFrame(trades)
    winning_trades = trades_df[trades_df["win"] == True]
    losing_trades = trades_df[trades_df["win"] == False]
    
    win_rate = len(winning_trades) / len(trades_df) * 100
    total_return = trades_df["profit_pct"].sum() # simple sum for simplicity
    avg_profit = trades_df["profit_pct"].mean()
    
    # Calculate Max Drawdown based on cumulative equity curve (simple sum)
    trades_df["cum_return"] = trades_df["profit_pct"].cumsum()
    cum_max = trades_df["cum_return"].cummax()
    drawdown = cum_max - trades_df["cum_return"]
    max_drawdown = drawdown.max()
    
    return {
        "total_trades": len(trades_df),
        "win_rate": win_rate,
        "total_return": total_return,
        "avg_profit": avg_profit,
        "max_drawdown": max_drawdown,
        "trades": trades
    }

def print_results(name, results, stop_loss):
    sl_str = f"{stop_loss}%" if stop_loss is not None else "None"
    print(f"\n=========================================")
    print(f" Strategy: {name}")
    print(f" Settings: Take Profit +1.0%, Stop Loss: {sl_str}")
    print(f"=========================================")
    print(f" Total Trades  : {results['total_trades']}")
    print(f" Win Rate      : {results['win_rate']:.2f}%")
    print(f" Total Return  : {results['total_return']:.2f}%")
    print(f" Avg Return/Trd: {results['avg_profit']:.2f}%")
    print(f" Max Drawdown  : {results['max_drawdown']:.2f}%")
    print(f"=========================================")

# Main execution
if __name__ == "__main__":
    # Fetch 5000 candles of 15m (approx 52 days)
    df = fetch_binance_klines(symbol="BTCUSDT", interval="15m", limit=1000, pages=5)
    
    print(f"Calculating indicators...")
    df = calculate_rsi(df)
    df = calculate_bollinger_bands(df)
    
    # Define signals
    # Strategy 1: RSI Overbought/Oversold (Buy when RSI < 30)
    # Avoid buying on consecutive bars to simulate realistic trigger
    df["rsi_buy_signal"] = (df["rsi"] < 30) & (df["rsi"].shift(1) >= 30)
    
    # Strategy 2: Bollinger Bands Lower Band touch (Buy when close goes below lower band)
    df["bb_buy_signal"] = (df["close"] < df["bb_lower"]) & (df["close"].shift(1) >= df["bb_lower"].shift(1))
    
    print("\nStarting Backtests (Take Profit = +1.0%)...")
    
    # Run backtests for different stop losses
    for sl in [-1.0, -2.0, -3.0, -5.0, None]:
        results_rsi = run_backtest(df, "rsi_buy_signal", target_profit=1.0, stop_loss=sl)
        print_results("RSI Mean Reversion (RSI < 30)", results_rsi, sl)
        
    for sl in [-1.0, -2.0, -3.0, -5.0, None]:
        results_bb = run_backtest(df, "bb_buy_signal", target_profit=1.0, stop_loss=sl)
        print_results("Bollinger Bands Pullback (Close < Lower Band)", results_bb, sl)
