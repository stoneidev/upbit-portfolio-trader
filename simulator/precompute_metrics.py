import os
import sqlite3
import pandas as pd
import numpy as np
import json
import sys

# Add current folder to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import engine

SIMULATOR_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SIMULATOR_DIR, "data", "nasdaq_simulator.db")

def precompute():
    print(f"Connecting to database: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    
    # Load daily prices
    df_all = pd.read_sql("SELECT * FROM daily_prices ORDER BY Ticker, Date ASC", conn)
    print(f"Loaded {len(df_all)} rows from daily_prices.")
    
    tickers = df_all["Ticker"].unique()
    all_metrics = []
    
    for idx, ticker in enumerate(tickers):
        print(f"[{idx+1}/{len(tickers)}] Precomputing for {ticker}...")
        df_t = df_all[df_all["Ticker"] == ticker].copy().reset_index(drop=True)
        
        # Base indicators
        df_t["MA20"] = df_t["Close"].rolling(20).mean()
        df_t["MA50"] = df_t["Close"].rolling(50).mean()
        df_t["MA150"] = df_t["Close"].rolling(150).mean()
        df_t["MA200"] = df_t["Close"].rolling(200).mean()
        
        df_t["High52W"] = df_t["High"].rolling(252, min_periods=120).max()
        df_t["Low52W"] = df_t["Low"].rolling(252, min_periods=120).min()
        
        df_t["STD20"] = df_t["Close"].rolling(20).std()
        df_t["Z_Score"] = (df_t["Close"] - df_t["MA20"]) / (df_t["STD20"] + 1e-9)
        
        df_t["RSI"] = engine.calculate_rsi(df_t["Close"], 14)
        df_t["ATR"] = engine.calculate_atr(df_t, 14)
        df_t["ATR_Pct"] = df_t["ATR"] / df_t["Close"]
        
        df_t["Momentum_3M"] = df_t["Close"].pct_change(60)
        
        # Loop through rows where we have enough history (index >= 252)
        for i in range(len(df_t)):
            if i < 252:
                # We can't compute Minervini/Phase correctly without 252 days history
                continue
                
            close_val = float(df_t.loc[i, "Close"])
            ma20_val = float(df_t.loc[i, "MA20"])
            ma50_val = float(df_t.loc[i, "MA50"])
            ma150_val = float(df_t.loc[i, "MA150"])
            ma200_val = float(df_t.loc[i, "MA200"])
            
            high_52w = float(df_t.loc[i, "High52W"])
            low_52w = float(df_t.loc[i, "Low52W"])
            
            z_val = float(df_t.loc[i, "Z_Score"])
            rsi_val = float(df_t.loc[i, "RSI"]) if pd.notna(df_t.loc[i, "RSI"]) else None
            atr_pct = float(df_t.loc[i, "ATR_Pct"]) if pd.notna(df_t.loc[i, "ATR_Pct"]) else None
            mom_val = float(df_t.loc[i, "Momentum_3M"]) if pd.notna(df_t.loc[i, "Momentum_3M"]) else None
            
            # MA series for slopes
            ma50_series = df_t.loc[i-19:i, "MA50"]
            ma200_series = df_t.loc[i-19:i, "MA200"]
            close_series = df_t.loc[i-39:i, "Close"]
            volume_series = df_t.loc[i-20:i, "Volume"]
            
            slope_50 = engine.calculate_slope(ma50_series, 20)
            slope_200 = engine.calculate_slope(ma200_series, 20)
            vol_data = engine.detect_volatility_contraction(close_series, 20)
            volume_ratio = engine.calculate_volume_ratio(volume_series, 20)
            distance_50 = engine.calculate_distance_from_sma(close_val, ma50_val)
            
            phase, trend_str = engine.classify_phase(
                close_val, ma50_val, ma200_val, slope_50, slope_200, vol_data, volume_ratio, distance_50
            )
            
            minervini_res = engine.validate_minervini_trend_template(
                close_val, ma50_val, ma150_val, ma200_val, df_t.loc[i-19:i, "MA200"], high_52w, low_52w, phase
            )
            
            # Append row to metrics
            all_metrics.append({
                "Ticker": ticker,
                "Date": df_t.loc[i, "Date"],
                "Open": float(df_t.loc[i, "Open"]),
                "High": float(df_t.loc[i, "High"]),
                "Low": float(df_t.loc[i, "Low"]),
                "Close": close_val,
                "Volume": int(df_t.loc[i, "Volume"]),
                "MA20": ma20_val,
                "MA50": ma50_val,
                "MA150": ma150_val,
                "MA200": ma200_val,
                "Z_Score": z_val,
                "RSI": rsi_val,
                "ATR_Pct": atr_pct,
                "Momentum_3M": mom_val,
                "Trend": trend_str,
                "Minervini_Score": int(minervini_res["criteria_passed"]),
                "Minervini_Pass": 1 if minervini_res["passes_template"] else 0,
                "Minervini_Details": json.dumps(minervini_res["details"])
            })
            
    df_metrics = pd.DataFrame(all_metrics)
    
    # Save to SQLite
    print("Writing metrics to SQLite...")
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS daily_metrics")
    conn.commit()
    
    df_metrics.to_sql("daily_metrics", conn, if_exists="replace", index=False)
    
    print("Creating indexes on daily_metrics...")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_ticker_date ON daily_metrics (Ticker, Date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_date ON daily_metrics (Date)")
    conn.commit()
    
    # Verify row count
    cursor.execute("SELECT COUNT(*) FROM daily_metrics")
    count = cursor.fetchone()[0]
    conn.close()
    
    print(f"Successfully computed and saved {count} rows in daily_metrics.")

if __name__ == "__main__":
    precompute()
