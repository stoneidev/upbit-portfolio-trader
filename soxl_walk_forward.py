import requests
import pandas as pd
import numpy as np
import os
import time

def fetch_soxl_data():
    """
    Fetches 2 years of SOXL 1h data from Yahoo Finance chart API.
    """
    csv_path = "/Users/stoni/Projects/AI/soxl_1h_data.csv"
    if os.path.exists(csv_path):
        print("Loading cached SOXL 1h data from CSV...")
        df = pd.read_csv(csv_path)
        df["open_time"] = pd.to_datetime(df["open_time"])
        return df

    print("Fetching 2 years of SOXL (1h) from Yahoo Finance...")
    url = "https://query1.finance.yahoo.com/v8/finance/chart/SOXL"
    params = {"interval": "1h", "range": "2y"}
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        response = requests.get(url, params=params, headers=headers)
        res = response.json()
        result = res["chart"]["result"][0]
        quote = result["indicators"]["quote"][0]
        timestamps = result["timestamp"]
        
        df = pd.DataFrame({
            "open_time": pd.to_datetime(timestamps, unit="s"),
            "open": quote["open"],
            "high": quote["high"],
            "low": quote["low"],
            "close": quote["close"],
            "volume": quote["volume"]
        })
        
        # Drop any rows with NaN in OHLCV
        df = df.dropna().reset_index(drop=True)
        df.to_csv(csv_path, index=False)
        print(f"Saved {len(df)} candles to {csv_path}")
        return df
    except Exception as e:
        print(f"Error fetching SOXL data: {e}")
        raise e

def calculate_indicators(df):
    # 1. Standard MA and Z-score of Close Price
    df["ma_100"] = df["close"].rolling(window=100).mean()
    df["std_100"] = df["close"].rolling(window=100).std()
    df["z_score"] = (df["close"] - df["ma_100"]) / (df["std_100"] + 1e-9)
    
    # 2. Volume 20 MA (for volume capitulation filter)
    df["volume_ma_20"] = df["volume"].rolling(window=20).mean()
    
    # 3. ATR
    high = df["high"]
    low = df["low"]
    close_prev = df["close"].shift(1)
    tr = pd.concat([high - low, (high - close_prev).abs(), (low - close_prev).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(window=14).mean()
    
    return df

def run_soxl_sim_fast(lows, highs, closes, opens, z_scores, volumes, vol_mas, atrs, z_thresh, activation_pct, trailing_pct, sl_pct):
    """
    Lookahead-free trailing stop simulation on SOXL.
    Grid rules (for 3x leverage ETF):
    - 1차: Buy 30%
    - 2차: Buy 30% at 1차 - 2.0% (wider grid for highly volatile SOXL)
    - 3차: Buy 40% at 1차 - 4.0%
    
    - Entry conditions:
      - z_score < z_thresh (oversold outlier)
      - volume > volume_ma_20 (above average capitulation volume)
      - close > open (green bar reversal confirmation)
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
        if np.isnan(z_scores[i]) or np.isnan(vol_mas[i]):
            continue
            
        low = lows[i]
        high = highs[i]
        close = closes[i]
        open_p = opens[i]
        
        if not in_trade:
            # Signal
            c1 = z_scores[i] < z_thresh
            c2 = volumes[i] > vol_mas[i]
            c3 = close > open_p
            
            if c1 and c2 and c3:
                level_1_price = close
                level_2_filled = False
                level_3_filled = False
                trailing_active = False
                peak_price = 0.0
                entry_idx = i
                in_trade = True
        else:
            # Grid levels (SOXL specific: 2% and 4% spacing)
            target_level_2 = level_1_price * 0.980 # -2.0%
            target_level_3 = level_1_price * 0.960 # -4.0%
            
            if not level_2_filled and low <= target_level_2:
                level_2_filled = True
            if not level_3_filled and low <= target_level_3:
                level_3_filled = True
                
            # Average Price
            if level_3_filled:
                avg_price = (level_1_price * 0.3 + target_level_2 * 0.3 + target_level_3 * 0.4)
                total_weight = 1.0
            elif level_2_filled:
                avg_price = (level_1_price * 0.3 + target_level_2 * 0.3) / 0.6
                total_weight = 0.6
            else:
                avg_price = level_1_price
                total_weight = 0.3
                
            sl_price = avg_price * (1 + sl_pct / 100.0)
            tp_price_fixed = avg_price * (1 + activation_pct / 100.0)
            
            if trailing_pct == 0.0:
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
                if not trailing_active:
                    if high >= tp_price_fixed:
                        trailing_active = True
                        peak_price = max(high, tp_price_fixed)
                    if low <= sl_price:
                        trades.append({'weight': total_weight, 'profit_pct': sl_pct, 'win': False, 'idx': i})
                        in_trade = False
                else:
                    stop_price = peak_price * (1 - trailing_pct / 100.0)
                    if low <= stop_price:
                        exit_price = max(stop_price, avg_price * (1 + (activation_pct - trailing_pct)/100.0))
                        profit_pct = (exit_price - avg_price) / avg_price * 100
                        trades.append({'weight': total_weight, 'profit_pct': profit_pct, 'win': True, 'idx': i})
                        in_trade = False
                    else:
                        peak_price = max(peak_price, high)
                        
    return trades

def find_best_params_soxl(lows_s, highs_s, closes_s, opens_s, z_scores_s, volumes_s, vol_mas_s, atrs_s, fee_rate=0.0004):
    z_space = [-2.0, -1.5, -1.0]
    activation_space = [1.0, 1.5, 2.0]
    trailing_space = [0.3, 0.5]
    sl_space = [-4.0, -6.0]
    
    best_cap = 1000000.0
    best_p = (-1.5, 1.5, 0.3, -4.0)
    
    for z in z_space:
        for act in activation_space:
            for trail in trailing_space:
                for sl in sl_space:
                    trades = run_soxl_sim_fast(
                        lows_s, highs_s, closes_s, opens_s, z_scores_s, volumes_s, vol_mas_s, atrs_s,
                        z, act, trail, sl
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
                        best_p = (z, act, trail, sl)
    return best_p

if __name__ == "__main__":
    df = fetch_soxl_data()
    df = calculate_indicators(df)
    
    lows = df["low"].to_numpy()
    highs = df["high"].to_numpy()
    closes = df["close"].to_numpy()
    opens = df["open"].to_numpy()
    z_scores = df["z_score"].to_numpy()
    volumes = df["volume"].to_numpy()
    vol_mas = df["volume_ma_20"].to_numpy()
    atrs = df["atr"].to_numpy()
    open_times = df["open_time"].to_numpy()
    
    print("\n=========================================================")
    print("      SOXL (Semiconductor 3X) Walk-Forward Backtest")
    print("=========================================================")
    print("Lookback: 60 Trading Days (approx 390 hours)")
    print("Update: 5 Trading Days (approx 32 hours)")
    print("Fee Rate: 0.04% (Maker/Limit Order Fee)\n")
    
    # 60 trading days = 60 * 6.5 = 390 hours
    lookback_size = 390
    # 5 trading days = 5 * 6.5 = 32 hours
    step_size = 32
    n_candles = len(df)
    
    capital = 1000000.0
    total_trades = 0
    wins = 0
    executed_trades = []
    
    start_idx = lookback_size
    start_time = time.time()
    
    while start_idx < n_candles:
        opt_start = start_idx - lookback_size
        opt_end = start_idx
        
        # Optimize parameters on lookback slice
        best_z, best_act, best_trail, best_sl = find_best_params_soxl(
            lows[opt_start:opt_end], highs[opt_start:opt_end], closes[opt_start:opt_end], opens[opt_start:opt_end],
            z_scores[opt_start:opt_end], volumes[opt_start:opt_end], vol_mas[opt_start:opt_end], atrs[opt_start:opt_end]
        )
        
        # Trade on next week
        test_end = min(start_idx + step_size, n_candles)
        test_trades = run_soxl_sim_fast(
            lows[start_idx:test_end], highs[start_idx:test_end], closes[start_idx:test_end], opens[start_idx:test_end],
            z_scores[start_idx:test_end], volumes[start_idx:test_end], vol_mas[start_idx:test_end], atrs[start_idx:test_end],
            best_z, best_act, best_trail, best_sl
        )
        
        for t in test_trades:
            ret = (t['weight'] * t['profit_pct']) / 100.0
            fee = t['weight'] * 0.0004
            capital *= (1 + ret - fee)
            
            total_trades += 1
            if t['win']:
                wins += 1
                
            executed_trades.append({
                "time": open_times[start_idx + t['idx']],
                "profit_pct": t['profit_pct'],
                "weight": t['weight'],
                "win": t['win'],
                "capital": capital,
                "z": best_z,
                "activation": best_act,
                "trailing": best_trail,
                "sl": best_sl
            })
            
        start_idx += step_size
        
    end_time = time.time()
    print(f"SOXL Walk-Forward completed in {end_time - start_time:.2f} seconds.")
    
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
    net_return = (capital - 1000000.0) / 1000000.0 * 100
    
    print("\n=========================================================")
    print("             SOXL Walk-Forward Results (1H)")
    print("=========================================================")
    print(f" Total Trades  : {total_trades}")
    print(f" Wins / Losses : {wins} / {total_trades - wins}")
    print(f" Win Rate      : {win_rate:.2f}%")
    print(f" Initial Capital: 1,000,000 KRW")
    print(f" Final Capital  : {capital:,.0f} KRW")
    print(f" Real Net Return: {net_return:.2f}% (After 0.04% Fees)")
    print("=========================================================")
    
    if len(executed_trades) > 0:
        trades_df = pd.DataFrame(executed_trades)
        print("\nSample SOXL Parameter Log & Capital Curve:")
        print(trades_df.iloc[::max(1, len(trades_df)//10)].to_string(index=False))
