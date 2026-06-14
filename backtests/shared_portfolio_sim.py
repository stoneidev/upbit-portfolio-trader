import pandas as pd
import numpy as np
import os

TOP_10_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "BNBUSDT", "LINKUSDT", "LTCUSDT", "NEARUSDT"
]

def load_and_prep_all_data():
    aligned_dfs = {}
    
    # First, load and calculate indicators for all symbols
    for symbol in TOP_10_SYMBOLS:
        csv_path = f"/Users/stoni/Projects/AI/{symbol.lower()}_2y_data.csv"
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Cache for {symbol} not found.")
            
        df = pd.read_csv(csv_path)
        df["open_time"] = pd.to_datetime(df["open_time"])
        
        # Cast numeric
        cols = ["open", "high", "low", "close", "volume", "taker_buy_volume"]
        for col in cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
        # Calculate indicators
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
        
        # Generate Signal: Z < -2.2, Taker Buy > 0.49
        df["signal"] = (df["z_score"] < -2.2) & (df["taker_buy_ratio_smoothed"] > 0.49)
        
        # Keep only required columns
        df = df[["open_time", "open", "high", "low", "close", "atr", "signal"]].copy()
        
        # Set index for alignment
        df.set_index("open_time", inplace=True)
        aligned_dfs[symbol] = df
        
    return aligned_dfs

def run_shared_portfolio_simulation(aligned_dfs, initial_capital=1000000.0, max_active_trades=3, allocation_per_trade=0.30, fee_rate=0.001):
    """
    Simulates a Shared Capital Pool of 1,000,000 KRW.
    - Up to 3 active trades at the same time.
    - Each trade gets allocated 30% of current capital.
    - Grid rules apply on the allocated sub-capital.
    """
    # Get all common timestamps
    common_timestamps = sorted(list(aligned_dfs["BTCUSDT"].index))
    
    capital = initial_capital
    active_trades = {} # key: symbol -> trade details
    trade_history = []
    capital_history = []
    
    print(f"Starting simulation over {len(common_timestamps)} candles...")
    
    for t_idx, timestamp in enumerate(common_timestamps):
        # 1. Update active trades with the new candle
        finished_symbols = []
        for symbol, trade in active_trades.items():
            df = aligned_dfs[symbol]
            if timestamp not in df.index:
                continue
                
            candle = df.loc[timestamp]
            if pd.isna(candle["atr"]):
                continue
                
            low = candle["low"]
            high = candle["high"]
            
            # Grid levels
            target_level_2 = trade["level_1_price"] - (1.0 * trade["atr_entry"])
            target_level_3 = trade["level_1_price"] - (2.0 * trade["atr_entry"])
            
            if not trade["level_2_filled"] and low <= target_level_2:
                trade["level_2_filled"] = True
            if not trade["level_3_filled"] and low <= target_level_3:
                trade["level_3_filled"] = True
                
            # Recalculate average price and weight
            prices = [trade["level_1_price"]]
            weights = [0.3]
            if trade["level_2_filled"]:
                prices.append(target_level_2)
                weights.append(0.3)
            if trade["level_3_filled"]:
                prices.append(target_level_3)
                weights.append(0.4)
                
            avg_price = sum(p * w for p, w in zip(prices, weights)) / sum(weights)
            total_weight = sum(weights)
            
            tp_price = avg_price + (1.0 * trade["atr_entry"])
            sl_price = avg_price - (3.0 * trade["atr_entry"])
            
            hit_tp = high >= tp_price
            hit_sl = low <= sl_price
            
            exit_profit_pct = 0.0
            exited = False
            win = False
            
            if hit_tp and hit_sl:
                # worst case
                exit_profit_pct = (sl_price - avg_price) / avg_price * 100
                exited = True
                win = False
            elif hit_tp:
                exit_profit_pct = (tp_price - avg_price) / avg_price * 100
                exited = True
                win = True
            elif hit_sl:
                exit_profit_pct = (sl_price - avg_price) / avg_price * 100
                exited = True
                win = False
                
            if exited:
                # Calculate return on the trade allocation
                trade_allocated_cap = trade["allocated_capital"]
                trade_volume_weight = total_weight
                
                # Profit/Loss on this trade
                trade_return = trade_allocated_cap * trade_volume_weight * (exit_profit_pct / 100.0)
                # Fee drag
                fee = trade_allocated_cap * trade_volume_weight * fee_rate
                
                # Update main capital
                capital += (trade_return - fee)
                
                trade_history.append({
                    "symbol": symbol,
                    "entry_time": trade["entry_time"],
                    "exit_time": timestamp,
                    "profit_pct": exit_profit_pct,
                    "weight": total_weight,
                    "win": win,
                    "net_profit": trade_return - fee
                })
                finished_symbols.append(symbol)
                
        # Remove finished trades
        for symbol in finished_symbols:
            del active_trades[symbol]
            
        # 2. Check for new signals
        if len(active_trades) < max_active_trades:
            # Look at all symbols for signals
            for symbol in TOP_10_SYMBOLS:
                if symbol in active_trades:
                    continue # already in trade
                    
                df = aligned_dfs[symbol]
                if timestamp not in df.index:
                    continue
                    
                candle = df.loc[timestamp]
                if candle["signal"] == True and not pd.isna(candle["atr"]):
                    # Check if we still have room
                    if len(active_trades) < max_active_trades:
                        # Open trade: allocate 30% of current capital
                        allocated_cap = capital * allocation_per_trade
                        active_trades[symbol] = {
                            "entry_time": timestamp,
                            "level_1_price": candle["close"],
                            "atr_entry": candle["atr"],
                            "allocated_capital": allocated_cap,
                            "level_2_filled": False,
                            "level_3_filled": False
                        }
                        
        # Record capital history (say, every day/96 candles to keep it light)
        if t_idx % 96 == 0:
            capital_history.append({"time": timestamp, "capital": capital})
            
    return capital, trade_history, capital_history

if __name__ == "__main__":
    print("Loading Top 10 data...")
    aligned_dfs = load_and_prep_all_data()
    
    print("\nRunning Shared Capital Pool Simulation...")
    print("Settings:")
    print(" - Initial Capital: 1,000,000 KRW")
    print(" - Max Active Concurrent Trades: 3")
    print(" - Capital Allocation per Trade: 30%")
    print(" - Trade Fee: 0.1% (round-trip)")
    print(" - Strategy Parameters (XRP Top 1 Optimization):")
    print("   - Z-Thresh: -2.2, Taker Buy Thresh: 0.49")
    print("   - TP = 1.0 * ATR, SL = 3.0 * ATR")
    
    final_cap, history, cap_hist = run_shared_portfolio_simulation(aligned_dfs)
    
    total_trades = len(history)
    wins = sum(1 for t in history if t["win"])
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
    
    print("\n=========================================================")
    print("              Shared Pool Simulation Results")
    print("=========================================================")
    print(f" Total Trades  : {total_trades}")
    print(f" Wins / Losses : {wins} / {total_trades - wins}")
    print(f" Win Rate      : {win_rate:.2f}%")
    print(f" Initial Capital: 1,000,000 KRW")
    print(f" Final Capital  : {final_cap:,.0f} KRW")
    print(f" Net Return (%) : {(final_cap - 1000000.0) / 1000000.0 * 100:.2f}%")
    
    # Calculate Max Drawdown based on capital history
    cap_df = pd.DataFrame(cap_hist)
    cap_df["cum_max"] = cap_df["capital"].cummax()
    cap_df["drawdown"] = (cap_df["cum_max"] - cap_df["capital"]) / cap_df["cum_max"] * 100
    max_dd = cap_df["drawdown"].max()
    print(f" Max Drawdown  : {max_dd:.2f}%")
    print("=========================================================")
    
    # Print a few sample trades
    if total_trades > 0:
        print("\nSample Trades:")
        sample_df = pd.DataFrame(history)
        print(sample_df.head(10).to_string(index=False))
