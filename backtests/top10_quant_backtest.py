import requests
import pandas as pd
import numpy as np
import time
import os

TOP_10_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "BNBUSDT", "LINKUSDT", "LTCUSDT", "NEARUSDT"
]

def fetch_data_for_symbol(symbol, start_time_ms=1718236800000):
    """
    Fetches 2 years of 15m data for a symbol and caches it to CSV.
    """
    csv_path = f"/Users/stoni/Projects/AI/{symbol.lower()}_2y_data.csv"
    
    # If XRP, try to use the existing cache to save time
    if symbol == "XRPUSDT" and os.path.exists("/Users/stoni/Projects/AI/xrp_2y_data.csv") and not os.path.exists(csv_path):
        import shutil
        shutil.copyfile("/Users/stoni/Projects/AI/xrp_2y_data.csv", csv_path)
        
    if os.path.exists(csv_path):
        print(f"[{symbol}] Loading cached data...")
        df = pd.read_csv(csv_path)
        df["open_time"] = pd.to_datetime(df["open_time"])
        return df

    print(f"[{symbol}] Fetching 2 years from Binance...")
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    current_start = start_time_ms
    page = 1
    
    while True:
        params = {
            "symbol": symbol,
            "interval": "15m",
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
            
            if last_time >= int(time.time() * 1000) - 15 * 60 * 1000:
                break
                
            page += 1
            time.sleep(0.02) # very short delay to avoid rate limit but run fast
        except Exception as e:
            print(f"[{symbol}] Error fetching: {e}")
            break
            
    df = pd.DataFrame(all_data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "count", "taker_buy_volume",
        "taker_buy_quote_volume", "ignore"
    ])
    
    numeric_cols = ["open", "high", "low", "close", "volume", "taker_buy_volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.sort_values("open_time").reset_index(drop=True)
    
    df.to_csv(csv_path, index=False)
    print(f"[{symbol}] Saved {len(df)} candles to cache.")
    return df

def calculate_quant_indicators(df):
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

def run_grid_backtest(df, atr_tp_multiplier=1.2, atr_sl_multiplier=2.5):
    trades = []
    in_trade = False
    
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
            # Strictly strict entry
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
            # 3-Step Grid
            target_level_2 = level_1_price - (1.0 * atr_entry)
            target_level_3 = level_1_price - (2.0 * atr_entry)
            
            if not level_2_filled and low <= target_level_2:
                level_2_filled = True
            if not level_3_filled and low <= target_level_3:
                level_3_filled = True
                
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
            
            tp_price = avg_price + (atr_tp_multiplier * atr_entry)
            sl_price = avg_price - (atr_sl_multiplier * atr_entry)
            
            hit_tp = high >= tp_price
            hit_sl = low <= sl_price
            
            if hit_tp and hit_sl:
                trades.append({'weight': total_weight, 'profit_pct': (sl_price - avg_price)/avg_price * 100, 'win': False})
                in_trade = False
            elif hit_tp:
                trades.append({'weight': total_weight, 'profit_pct': (tp_price - avg_price)/avg_price * 100, 'win': True})
                in_trade = False
            elif hit_sl:
                trades.append({'weight': total_weight, 'profit_pct': (sl_price - avg_price)/avg_price * 100, 'win': False})
                in_trade = False
                
    return trades

def simulate_capital(trades, initial_capital=100000, fee_rate=0.001):
    capital = initial_capital
    for t in trades:
        ret = (t['weight'] * t['profit_pct']) / 100.0
        fee = t['weight'] * fee_rate
        capital *= (1 + ret - fee)
    return capital

if __name__ == "__main__":
    print("=========================================================")
    print("      Top 10 Coins 2-Year Portfolio Quant Backtest")
    print("=========================================================")
    
    portfolio_results = []
    initial_total_capital = 1000000.0 # 100만 원
    coin_allocation = initial_total_capital / len(TOP_10_SYMBOLS) # 코인당 10만 원씩 균등 할당
    
    final_total_capital = 0.0
    
    for symbol in TOP_10_SYMBOLS:
        try:
            df = fetch_data_for_symbol(symbol)
            df = calculate_quant_indicators(df)
            trades = run_grid_backtest(df)
            
            # Simulate with 10% capital
            final_cap = simulate_capital(trades, initial_capital=coin_allocation, fee_rate=0.001)
            final_total_capital += final_cap
            
            total_trades = len(trades)
            wins = sum(1 for t in trades if t['win'])
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
            net_ret = (final_cap - coin_allocation) / coin_allocation * 100
            
            portfolio_results.append({
                "symbol": symbol,
                "trades": total_trades,
                "win_rate": win_rate,
                "initial": coin_allocation,
                "final": final_cap,
                "return": net_ret
            })
            
        except Exception as e:
            print(f"[{symbol}] Failed to backtest: {e}")
            final_total_capital += coin_allocation # assume cash held
            
    print("\n=========================================================")
    print("                Individual Coin Performance")
    print("=========================================================")
    print(f"{'Coin':<10} | {'Trades':<8} | {'Win Rate':<10} | {'Initial Cap':<15} | {'Final Cap':<15} | {'Net Return':<12}")
    print("-" * 85)
    for r in portfolio_results:
        print(f"{r['symbol']:<10} | {r['trades']:<8} | {r['win_rate']:<9.2f}% | {r['initial']:<13,.0f}원 | {r['final']:<13,.0f}원 | {r['return']:<10.2f}%")
        
    total_net_return = (final_total_capital - initial_total_capital) / initial_total_capital * 100
    print("=========================================================")
    print("                  Portfolio Summary")
    print("=========================================================")
    print(f" Total Initial Capital: {initial_total_capital:,.0f} KRW")
    print(f" Total Final Capital  : {final_total_capital:,.0f} KRW")
    print(f" Portfolio Net Return : {total_net_return:.2f}% (After 0.1% Fees)")
    print("=========================================================")
