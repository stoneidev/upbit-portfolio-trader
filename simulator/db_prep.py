import os
import sqlite3
import datetime
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

# Define directories
SIMULATOR_DIR = "/Users/stoni/Projects/AI/simulator"
DATA_DIR = os.path.join(SIMULATOR_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "nasdaq_simulator.db")

# Fallback list of top Nasdaq-100 components by market cap (as of mid-2026)
TOP50_TICKERS = [
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "AVGO", "TSLA", "COST",
    "NFLX", "AMD", "ASML", "PEP", "AZN", "LIN", "ADBE", "CSCO", "QCOM", "TMUS",
    "TXN", "AMGN", "INTU", "ISRG", "HON", "AMAT", "BKNG", "MU", "MDLZ", "REGN",
    "LRCX", "VRTX", "ADP", "PANW", "GILD", "MELI", "SNPS", "CDNS", "KLAC", "CSX",
    "MAR", "PDD", "INTC", "PYPL", "ORLY", "ADI", "WMT", "NXPI", "CRWD", "WDAY"
]

def download_ticker_data(ticker, start_date, end_date):
    """Downloads daily data for a single ticker."""
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df.empty:
            return ticker, None
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
            
        df = df.reset_index()
        df["Ticker"] = ticker
        
        # Select and order required columns
        cols = ["Ticker", "Date", "Open", "High", "Low", "Close", "Volume"]
        df = df[[c for c in cols if c in df.columns]]
        
        # Format Date column to YYYY-MM-DD
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
        
        return ticker, df
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
        return ticker, None

def build_database():
    start_date = "2023-11-01"  # 14-month buffer before 2025-01-02 to calculate 200 SMA and 52-week High/Low
    end_date = datetime.date.today().strftime("%Y-%m-%d")
    
    print(f"Connecting to SQLite database at: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Drop table if exists to start fresh
    cursor.execute("DROP TABLE IF EXISTS daily_prices")
    conn.commit()
    
    print(f"Downloading daily data from {start_date} to {end_date} for {len(TOP50_TICKERS)} tickers...")
    
    # Download data sequentially to avoid yfinance thread-safety issues
    all_data = []
    for idx, t in enumerate(TOP50_TICKERS):
        ticker, df = download_ticker_data(t, start_date, end_date)
        if df is not None:
            all_data.append(df)
            print(f"[{idx+1}/{len(TOP50_TICKERS)}] Successfully downloaded {ticker} ({len(df)} rows)")
        else:
            print(f"[{idx+1}/{len(TOP50_TICKERS)}] ❌ Failed to download {ticker}")
                
    if not all_data:
        print("Error: No data downloaded.")
        conn.close()
        return
        
    # Combine and save to SQLite
    print("Writing data to database...")
    df_combined = pd.concat(all_data).reset_index(drop=True)
    df_combined.to_sql("daily_prices", conn, if_exists="append", index=False)
    
    # Create index on Ticker and Date
    print("Creating indexes on Ticker and Date...")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ticker_date ON daily_prices (Ticker, Date)")
    conn.commit()
    
    # Verify table row count
    cursor.execute("SELECT COUNT(*) FROM daily_prices")
    total_rows = cursor.fetchone()[0]
    
    conn.close()
    
    print("\n" + "="*50)
    print("SIMULATOR DATABASE BUILD COMPLETED")
    print(f"Total Tickers Imported: {len(all_data)}")
    print(f"Total Rows Inserted: {total_rows}")
    print(f"SQLite DB File: {DB_PATH}")
    print("="*50 + "\n")

if __name__ == "__main__":
    build_database()
