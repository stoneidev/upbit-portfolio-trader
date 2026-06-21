import os
import sqlite3
import pandas as pd
import numpy as np

SIMULATOR_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SIMULATOR_DIR, "data", "nasdaq_simulator.db")

TOP50_TICKERS = [
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "AVGO", "TSLA", "COST",
    "NFLX", "AMD", "ASML", "PEP", "AZN", "LIN", "ADBE", "CSCO", "QCOM", "TMUS",
    "TXN", "AMGN", "INTU", "ISRG", "HON", "AMAT", "BKNG", "MU", "MDLZ", "REGN",
    "LRCX", "VRTX", "ADP", "PANW", "GILD", "MELI", "SNPS", "CDNS", "KLAC", "CSX",
    "MAR", "PDD", "INTC", "PYPL", "ORLY", "ADI", "WMT", "NXPI", "CRWD", "WDAY"
]

import json

SHARES_PATH = os.path.join(SIMULATOR_DIR, "data", "shares_outstanding.json")

def load_shares_outstanding():
    if os.path.exists(SHARES_PATH):
        try:
            with open(SHARES_PATH, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading shares JSON: {e}")
    # Default fallback values for shares (roughly mid-2026 outstanding shares)
    return {
        "NVDA": 24221000000, "AAPL": 14687356000, "MSFT": 7428434704, "AMZN": 10757109436,
        "GOOGL": 5867155790, "GOOG": 5499638298, "META": 2196045588, "AVGO": 4757580198,
        "TSLA": 3755723871, "COST": 443478804, "NFLX": 4210798528, "AMD": 1630600639,
        "ASML": 385417665, "PEP": 1366768315, "AZN": 1550862203, "LIN": 462347310,
        "ADBE": 397500000, "CSCO": 3941434665, "QCOM": 1054000000, "TMUS": 1082204717,
        "TXN": 910092791, "AMGN": 539708274, "INTU": 273537000, "ISRG": 354162842,
        "HON": 633653119, "AMAT": 793959430, "BKNG": 774878436, "MU": 1127734051,
        "MDLZ": 1283649766, "REGN": 103021886, "LRCX": 1250571000, "VRTX": 253805417,
        "ADP": 399734282, "PANW": 815000000, "GILD": 1241569874, "MELI": 50697182,
        "SNPS": 191479325, "CDNS": 275816000, "KLAC": 1306275210, "CSX": 1858138856,
        "MAR": 263688623, "PDD": 1423396462, "INTC": 5026000000, "PYPL": 882105493,
        "ORLY": 828715079, "ADI": 487087040, "WMT": 7958079155, "NXPI": 252471079,
        "CRWD": 254564820, "WDAY": 201000000
    }

def calculate_rsi(prices, window=14):
    """Calculates standard 14-period RSI using Wilder's EMA smoothing."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.ewm(com=window-1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window-1, min_periods=window).mean()
    
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(df, window=14):
    """Calculates standard 14-period ATR."""
    high = df['High']
    low = df['Low']
    close_prev = df['Close'].shift(1)
    
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr

def calculate_slope(series: pd.Series, periods: int = 20) -> float:
    """Calculates linear regression slope over recent periods as percentage change per day."""
    if len(series) < periods or series.isna().all():
        return 0.0
    
    recent = series.iloc[-periods:].dropna()
    if len(recent) < 2:
        return 0.0
        
    x = np.arange(len(recent))
    y = recent.values
    
    if np.std(x) == 0:
        return 0.0
        
    slope = np.polyfit(x, y, 1)[0]
    avg_val = np.mean(y)
    if avg_val == 0:
        return 0.0
        
    slope_pct = (slope / avg_val) * 100
    return slope_pct

def detect_volatility_contraction(prices: pd.Series, window: int = 20) -> dict:
    """Detects volatility contraction (squeeze ratio)."""
    if len(prices) < window * 2:
        return {
            'is_contracting': False,
            'contraction_quality': 0.0,
            'current_volatility': 0.0
        }
        
    volatility = prices.rolling(window=window).std()
    if len(volatility.dropna()) < 2:
        return {
            'is_contracting': False,
            'contraction_quality': 0.0,
            'current_volatility': 0.0
        }
        
    current_vol = volatility.iloc[-1]
    avg_vol = volatility.iloc[-window*2:-window].mean()
    contraction_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
    is_contracting = contraction_ratio < 0.7
    quality = max(0, min(100, (1 - contraction_ratio) * 100))
    
    return {
        'is_contracting': is_contracting,
        'contraction_quality': round(quality, 2),
        'current_volatility': round(current_vol, 2),
        'contraction_ratio': round(contraction_ratio, 2)
    }

def calculate_volume_ratio(volumes: pd.Series, period: int = 20) -> float:
    """Calculates current volume relative to its moving average."""
    if len(volumes) < period + 1:
        return 1.0
    current = volumes.iloc[-1]
    avg = volumes.iloc[-period-1:-1].mean()
    if avg == 0:
        return 1.0
    return float(current / avg)

def calculate_distance_from_sma(price: float, sma: float) -> float:
    """Calculates percentage distance of price from SMA."""
    if sma == 0:
        return 0.0
    return ((price - sma) / sma) * 100

def classify_phase(
    current_price: float,
    sma_50_val: float,
    sma_200_val: float,
    slope_50: float,
    slope_200: float,
    vol_data: dict,
    volume_ratio: float,
    distance_50: float
) -> tuple:
    """Classifies current trend into one of the 4 Stages."""
    # Downtrend (Stage 4)
    if current_price < sma_50_val and current_price < sma_200_val and sma_50_val < sma_200_val:
        return 4, "STAGE 4 (DOWN)"
    # Uptrend / Breakout (Stage 2)
    elif current_price > sma_50_val and sma_50_val > sma_200_val and slope_50 > 0:
        return 2, "STAGE 2 (UP)"
    # Distribution / Top (Stage 3)
    elif current_price > sma_50_val and distance_50 > 25:
        return 3, "STAGE 3 (TOP)"
    # Base Building (Stage 1)
    else:
        return 1, "STAGE 1 (BASE)"

def validate_minervini_trend_template(
    current_price: float,
    sma_50: float,
    sma_150: float,
    sma_200: float,
    sma_200_series: pd.Series,
    week_52_high: float,
    week_52_low: float,
    phase: int
) -> dict:
    """Validates the 8 criteria for Minervini Trend Template."""
    criteria = {}
    passed_count = 0

    # Criterion 1: Price > 150 SMA AND 200 SMA
    c1 = bool(current_price > sma_150 and current_price > sma_200)
    criteria['price_above_150_200'] = c1
    if c1: passed_count += 1

    # Criterion 2: 150 SMA > 200 SMA
    c2 = bool(sma_150 > sma_200)
    criteria['sma_150_above_200'] = c2
    if c2: passed_count += 1

    # Criterion 3: 200 SMA trending up for at least 1 month
    if len(sma_200_series) >= 20:
        sma_200_1mo_ago = sma_200_series.iloc[-20]
        sma_200_now = sma_200_series.iloc[-1]
        sma_200_rising = bool(sma_200_now > sma_200_1mo_ago)
    else:
        sma_200_rising = False
    
    c3 = sma_200_rising
    criteria['sma_200_rising'] = c3
    if c3: passed_count += 1

    # Criterion 4: 50 SMA > 150 SMA
    c4 = bool(sma_50 > sma_150)
    criteria['sma_50_above_150'] = c4
    if c4: passed_count += 1

    # Criterion 5: Price > 50 SMA
    c5 = bool(current_price > sma_50)
    criteria['price_above_50'] = c5
    if c5: passed_count += 1

    # Criterion 6: Price at least 30% above 52-week low
    if week_52_low > 0:
        distance_from_52w_low = float(((current_price - week_52_low) / week_52_low) * 100)
        c6 = bool(distance_from_52w_low >= 30)
    else:
        c6 = False
        distance_from_52w_low = 0.0
    criteria['price_30pct_above_52w_low'] = c6
    criteria['distance_from_52w_low_pct'] = float(round(distance_from_52w_low, 1))

    if c6: passed_count += 1

    # Criterion 7: Price within 25% of 52-week high
    if week_52_high > 0:
        distance_from_52w_high = float(((week_52_high - current_price) / week_52_high) * 100)
        c7 = bool(distance_from_52w_high <= 25)
    else:
        c7 = False
        distance_from_52w_high = 100.0
    criteria['price_near_52w_high'] = c7
    criteria['distance_from_52w_high_pct'] = float(round(distance_from_52w_high, 1))

    if c7: passed_count += 1

    # Criterion 8: Phase must be 2
    c8 = bool(phase == 2)
    criteria['confirmed_stage_2'] = c8
    if c8: passed_count += 1

    template_score = int((passed_count / 8) * 100)
    passes_template = bool(passed_count >= 7)

    return {
        'passes_template': passes_template,
        'criteria_passed': int(passed_count),
        'criteria_total': 8,
        'template_score': template_score,
        'details': criteria
    }

def get_trading_dates():
    """Returns a sorted list of all unique trading dates from 2025-01-01 onwards."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT Date FROM daily_prices 
        WHERE Date >= '2025-01-01' 
        ORDER BY Date ASC
    """)
    dates = [r[0] for r in cursor.fetchall()]
    conn.close()
    return dates

def get_market_state(target_date):
    """
    Computes all technical and WTA indicators for all tickers up to target_date.
    Ensures zero look-ahead bias.
    """
    shares_dict = load_shares_outstanding()
    conn = sqlite3.connect(DB_PATH)
    
    # Load all data up to target_date
    query = """
        SELECT Ticker, Date, Open, High, Low, Close, Volume 
        FROM daily_prices 
        WHERE Date <= ?
        ORDER BY Ticker, Date ASC
    """
    df_all = pd.read_sql(query, conn, params=(target_date,))
    conn.close()
    
    if df_all.empty:
        return None, {}
        
    tickers = df_all["Ticker"].unique()
    rows = []
    
    for ticker in tickers:
        df_t = df_all[df_all["Ticker"] == ticker].copy().reset_index(drop=True)
        # Need at least 252 trading days for 52-week rolling high/low and 200 SMA
        if len(df_t) < 252:
            continue
            
        # Calculate indicators
        df_t["MA20"] = df_t["Close"].rolling(20).mean()
        df_t["MA50"] = df_t["Close"].rolling(50).mean()
        df_t["MA150"] = df_t["Close"].rolling(150).mean()
        df_t["MA200"] = df_t["Close"].rolling(200).mean()
        
        df_t["High52W"] = df_t["High"].rolling(252, min_periods=120).max()
        df_t["Low52W"] = df_t["Low"].rolling(252, min_periods=120).min()
        
        df_t["STD20"] = df_t["Close"].rolling(20).std()
        df_t["Z"] = (df_t["Close"] - df_t["MA20"]) / (df_t["STD20"] + 1e-9)
        
        df_t["RSI"] = calculate_rsi(df_t["Close"], 14)
        df_t["ATR"] = calculate_atr(df_t, 14)
        df_t["ATR_Pct"] = df_t["ATR"] / df_t["Close"]
        
        # Momentum (60-day return, approx 3 months)
        df_t["Mom_60d"] = df_t["Close"].pct_change(60)
        
        # Get target day record (the last row because we filtered <= target_date)
        last_idx = len(df_t) - 1
        
        # Verify the last record is indeed target_date
        if df_t.loc[last_idx, "Date"] != target_date:
            continue
            
        close_val = df_t.loc[last_idx, "Close"]
        shares = shares_dict.get(ticker, 1e9)
        market_cap = float(close_val * shares)
        ma20_val = df_t.loc[last_idx, "MA20"]
        ma50_val = df_t.loc[last_idx, "MA50"]
        ma150_val = df_t.loc[last_idx, "MA150"]
        ma200_val = df_t.loc[last_idx, "MA200"]
        
        high_52w = df_t.loc[last_idx, "High52W"]
        low_52w = df_t.loc[last_idx, "Low52W"]
        
        z_val = df_t.loc[last_idx, "Z"]
        rsi_val = df_t.loc[last_idx, "RSI"]
        atr_pct = df_t.loc[last_idx, "ATR_Pct"]
        mom_val = df_t.loc[last_idx, "Mom_60d"]
        volume = df_t.loc[last_idx, "Volume"]
        open_val = df_t.loc[last_idx, "Open"]
        high_val = df_t.loc[last_idx, "High"]
        low_val = df_t.loc[last_idx, "Low"]
        
        # Calculate slopes and details
        slope_50 = calculate_slope(df_t["MA50"], 20)
        slope_200 = calculate_slope(df_t["MA200"], 20)
        vol_data = detect_volatility_contraction(df_t["Close"], 20)
        volume_ratio = calculate_volume_ratio(df_t["Volume"], 20)
        distance_50 = calculate_distance_from_sma(close_val, ma50_val)
        
        # Classify trend phase (Stage 1-4)
        phase, trend_str = classify_phase(
            close_val,
            ma50_val,
            ma200_val,
            slope_50,
            slope_200,
            vol_data,
            volume_ratio,
            distance_50
        )
        
        # Validate Minervini Trend Template
        minervini_res = validate_minervini_trend_template(
            close_val,
            ma50_val,
            ma150_val,
            ma200_val,
            df_t["MA200"],
            high_52w,
            low_52w,
            phase
        )
            
        rows.append({
            "Ticker": ticker,
            "Open": open_val,
            "High": high_val,
            "Low": low_val,
            "Close": close_val,
            "MA20": ma20_val,
            "MA50": ma50_val,
            "MA150": ma150_val,
            "MA200": ma200_val,
            "Z-Score": z_val,
            "RSI": rsi_val,
            "ATR_Pct": atr_pct,
            "Momentum_3M": mom_val,
            "Trend": trend_str,
            "Volume": volume,
            "Market_Cap": market_cap,
            "Minervini_Score": minervini_res["criteria_passed"],
            "Minervini_Pass": minervini_res["passes_template"],
            "Minervini_Details": minervini_res["details"]
        })
        
    if not rows:
        return None, {}
        
    df_metrics = pd.DataFrame(rows)
    
    # Rank by Market Cap dynamically
    df_metrics["MCap_Rank"] = df_metrics["Market_Cap"].rank(ascending=False, method="min").astype(int)
    
    # Rank by momentum
    df_metrics["Mom_Rank"] = df_metrics["Momentum_3M"].rank(ascending=False, method="min").astype(int)
    
    # Calculate Momentum Z-Score
    mom_mean = df_metrics["Momentum_3M"].mean()
    mom_std = df_metrics["Momentum_3M"].std() + 1e-9
    df_metrics["Mom_Z"] = (df_metrics["Momentum_3M"] - mom_mean) / mom_std
    
    # WTA Status
    df_metrics["WTA_Status"] = "HOLD"
    df_metrics.loc[df_metrics["Mom_Rank"] == 1, "WTA_Status"] = "WTA-Champion"
    df_metrics.loc[(df_metrics["Mom_Rank"] > 1) & (df_metrics["Mom_Rank"] <= 5), "WTA_Status"] = "WTA-Leader"
    
    # WTA Concentration Index
    pos_returns = df_metrics.loc[df_metrics["Momentum_3M"] > 0, "Momentum_3M"]
    top_5_sum = df_metrics.sort_values(by="Momentum_3M", ascending=False).head(5)["Momentum_3M"].sum()
    total_pos_sum = pos_returns.sum() if len(pos_returns) > 0 else 1e-9
    wta_index = top_5_sum / total_pos_sum
    
    # Market Breadth
    total_stocks = len(df_metrics)
    above_ma20 = sum(df_metrics["Close"] > df_metrics["MA20"]) / total_stocks
    above_ma50 = sum(df_metrics["Close"] > df_metrics["MA50"]) / total_stocks
    
    market_regime = "NEUTRAL"
    if above_ma20 > 0.60 and above_ma50 > 0.60:
        market_regime = "BULLISH"
    elif above_ma20 < 0.40 and above_ma50 < 0.40:
        market_regime = "BEARISH"
        
    summary = {
        "date": target_date,
        "above_ma20": above_ma20,
        "above_ma50": above_ma50,
        "wta_index": wta_index,
        "market_regime": market_regime,
        "avg_rsi": df_metrics["RSI"].mean(),
        "avg_z": df_metrics["Z-Score"].mean()
    }
    
    return df_metrics, summary
