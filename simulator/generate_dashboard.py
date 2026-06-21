import os
import sys
import sqlite3
import argparse
import datetime
import numpy as np
import pandas as pd

# Paths
DB_PATH = "/Users/stoni/Projects/AI/simulator/data/nasdaq_simulator.db"
ARTIFACT_DIR = "/Users/stoni/.gemini/antigravity/brain/78b9864b-3618-46a6-ad91-15dcb23e1b46"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def calculate_rsi(prices, window=14):
    """Calculates standard 14-period RSI."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Use exponential moving average (standard Wilder's RSI smoothing)
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

def get_market_data(target_date):
    """Loads all data up to target_date and computes metrics using engine.py."""
    # Ensure local directory is in path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import engine
    
    conn = sqlite3.connect(DB_PATH)
    
    # Check if target_date exists in db
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT Date FROM daily_prices ORDER BY Date DESC")
    available_dates = [r[0] for r in cursor.fetchall()]
    conn.close()
    
    if not available_dates:
        print("Error: No data found in database.")
        return None, None
        
    # If no date specified or invalid, default to latest
    if not target_date or target_date not in available_dates:
        if target_date:
            print(f"Warning: Target date {target_date} not found. Defaulting to latest available date.")
        # Find nearest date that is <= target_date if date was specified
        if target_date:
            valid_dates = [d for d in available_dates if d <= target_date]
            target_date = valid_dates[0] if valid_dates else available_dates[0]
        else:
            target_date = available_dates[0]
            
    print(f"Generating dashboard for Date: {target_date}")
    
    df_metrics, _ = engine.get_market_state(target_date)
    return df_metrics, target_date

def generate_report(df_metrics, target_date):
    """Generates the Markdown report artifact with WTA metrics."""
    # 1. Market Breadth Calculations
    total_stocks = len(df_metrics)
    above_ma20 = sum(df_metrics["Close"] > df_metrics["MA20"]) / total_stocks
    above_ma50 = sum(df_metrics["Close"] > df_metrics["MA50"]) / total_stocks
    avg_z = df_metrics["Z-Score"].mean()
    avg_rsi = df_metrics["RSI"].mean()
    
    # 2. Winner-Take-All Concentration Index (WTA Index)
    # Sum of 3M return of top 5 stocks / Sum of 3M return of all positive-momentum stocks
    pos_returns = df_metrics.loc[df_metrics["Momentum_3M"] > 0, "Momentum_3M"]
    top_5_sum = df_metrics.sort_values(by="Momentum_3M", ascending=False).head(5)["Momentum_3M"].sum()
    total_pos_sum = pos_returns.sum() if len(pos_returns) > 0 else 1e-9
    wta_index = top_5_sum / total_pos_sum
    
    # Classify market regime
    market_regime = "NEUTRAL"
    if above_ma20 > 0.60 and above_ma50 > 0.60:
        market_regime = "BULLISH"
    elif above_ma20 < 0.40 and above_ma50 < 0.40:
        market_regime = "BEARISH"
        
    # Sort for top tables
    top_mom = df_metrics.sort_values(by="Momentum_3M", ascending=False).head(5)
    oversold = df_metrics[df_metrics["RSI"].notna()].sort_values(by="RSI", ascending=True).head(5)
    overbought = df_metrics[df_metrics["RSI"].notna()].sort_values(by="RSI", ascending=False).head(5)
    
    # Format and save report
    markdown_path = os.path.join(ARTIFACT_DIR, "nasdaq_trading_dashboard.md")
    
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write(f"# 📊 Nasdaq Top 50 Trading Dashboard ({target_date})\n\n")
        f.write("이 대시보드는 모의투자 연습을 돕기 위해 주요 시장 지표, 개별 종목의 모멘텀 순위, Z-Score, 과매수/과매도(RSI), 변동성 및 **승자독식(Winner-Take-All) 지표**를 분석하여 제공합니다.\n\n")
        
        # Market Regime Alert
        if market_regime == "BULLISH":
            f.write("> [!NOTE]\n")
            f.write(f"> **시장 국면: 강세장 (BULLISH)**\n")
            f.write(f"> 현재 시총 상위 종목의 {above_ma20:.1%}가 20일선 위에, {above_ma50:.1%}가 50일선 위에 있어 시장 전반의 상승 탄력이 강합니다. 돌파 매매 및 모멘텀 추종 전략이 유리합니다.\n\n")
        elif market_regime == "BEARISH":
            f.write("> [!WARNING]\n")
            f.write(f"> **시장 국면: 약세장 (BEARISH)**\n")
            f.write(f"> 현재 시총 상위 종목의 {above_ma20:.1%}만이 20일선 위에 있습니다. 하락 압력이 매우 강하므로 매수 진입 시 손절을 타이트하게 잡거나 현금 비중을 극대화해야 합니다.\n\n")
        else:
            f.write("> [!IMPORTANT]\n")
            f.write(f"> **시장 국면: 혼조/횡보장 (NEUTRAL)**\n")
            f.write(f"> 시장 참여도가 중간 수준(20일선 위 {above_ma20:.1%})입니다. 종목별 차별화 장세이거나 박스권 흐름이 예상되므로 밴드 매매 및 평균회귀 전략이 적합할 수 있습니다.\n\n")
            
        f.write("## 📈 시장 지표 요약 (Market Breadth & WTA Index)\n\n")
        f.write(f"* **평가 일자**: `{target_date}`\n")
        f.write(f"* **단기 시장 Breadth (Close > 20 SMA)**: `{above_ma20:.1%}`\n")
        f.write(f"* **중기 시장 Breadth (Close > 50 SMA)**: `{above_ma50:.1%}`\n")
        f.write(f"* **시장 평균 Z-Score (20일)**: `{avg_z:.2f}`\n")
        f.write(f"* **시장 평균 RSI (14일)**: `{avg_rsi:.1f}`\n")
        
        # WTA Index explanation
        wta_alert_type = "[!NOTE]"
        wta_status_str = "보통 (Broad-market Participation)"
        if wta_index > 0.50:
            wta_alert_type = "[!IMPORTANT]"
            wta_status_str = "매우 높음 (Extreme Winner-Take-All)"
        f.write(f"* **승자독식 지수 (WTA Concentration Index)**: `{wta_index:.1%}` (상위 5개 종목이 전체 상승 동력의 해당 비율을 차지)\n\n")
        
        f.write(f"> {wta_alert_type}\n")
        f.write(f"> **WTA Concentration 상태: {wta_status_str}**\n")
        f.write(f"> WTA 지수가 {wta_index:.1%}입니다. 지수가 50%를 초과하는 경우, 시장 상승세가 소수 주도주에 과도하게 쏠려있는 '승자독식 장세'입니다. 이 경우 무조건 모멘텀 상위 1~5위 주도주에만 자금을 집중하는 것이 수익률을 극대화하는 비결입니다.\n\n")
        
        # Carousels for trading setups
        f.write("## 🎯 모의투자 추천 전략 포커스 그룹\n\n")
        f.write("````carousel\n")
        
        # Slide 1: Momentum Leaders & WTA Targets
        f.write("### 🏆 승자독식(WTA) 모멘텀 리더 (추세 추종용)\n")
        f.write("최근 3개월간 시장을 지배한 주도주군입니다. WTA 전략 적용 시 우선 매수 후보입니다.\n\n")
        f.write("| Ticker | Mom Rank | 3M Return | Mom Z-Score | WTA Status | Trend | Minervini |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for _, r in top_mom.iterrows():
            f.write(f"| **{r['Ticker']}** | **{r['Mom_Rank']}** | {r['Momentum_3M']:.1%} | {r['Mom_Z']:.2f} | `{r['WTA_Status']}` | `{r['Trend']}` | `{r['Minervini_Score']}/8` |\n")
            
        f.write("\n<!-- slide -->\n")
        
        # Slide 2: Oversold Mean Reversion
        f.write("### 🛡️ 과매도 낙폭과대주 (평균회귀용)\n")
        f.write("RSI가 낮고 Z-Score가 마이너스로 내려간 종목들로, 단기 반등을 노리는 역추세 매매에 적합합니다.\n\n")
        f.write("| Ticker | RSI | Z-Score | ATR% (변동성) | Trend |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for _, r in oversold.iterrows():
            f.write(f"| **{r['Ticker']}** | **{r['RSI']:.1f}** | {r['Z-Score']:.2f} | {r['ATR_Pct']:.1%} | `{r['Trend']}` |\n")
            
        f.write("\n<!-- slide -->\n")
        
        # Slide 3: Overbought Risk Warning
        f.write("### ⚠️ 과매수 경계주 (보유자 영역/추격 매수 금지)\n")
        f.write("RSI가 70에 근접하거나 초과하였고, Z-score가 높아 단기 조정 리스크가 큰 종목들입니다.\n\n")
        f.write("| Ticker | RSI | Z-Score | Trend | 3M Return |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for _, r in overbought.iterrows():
            f.write(f"| **{r['Ticker']}** | **{r['RSI']:.1f}** | {r['Z-Score']:.2f} | `{r['Trend']}` | {r['Momentum_3M']:.1%} |\n")
            
        f.write("\n````\n\n")
        
        # Detailed table
        f.write("## 📋 나스닥 시총 상위 50 종목 종합 스크리너\n\n")
        f.write("| Ticker | Close | Trend | Mom Rank | 3M Return | Mom Z-Score | WTA Status | Minervini | Z-Score (20d) | RSI (14d) | ATR% |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        
        # Sort by ticker for the screening list
        df_sorted = df_metrics.sort_values(by="Ticker")
        for _, r in df_sorted.iterrows():
            mom_str = f"{r['Momentum_3M']:.1%}" if pd.notna(r['Momentum_3M']) else "-"
            mom_z_str = f"{r['Mom_Z']:.2f}" if pd.notna(r['Mom_Z']) else "-"
            rsi_str = f"{r['RSI']:.1f}" if pd.notna(r['RSI']) else "-"
            z_str = f"{r['Z-Score']:.2f}" if pd.notna(r['Z-Score']) else "-"
            f.write(f"| **{r['Ticker']}** | {r['Close']:.2f} | `{r['Trend']}` | {r['Mom_Rank']} | {mom_str} | {mom_z_str} | `{r['WTA_Status']}` | `{r['Minervini_Score']}/8` | {z_str} | {rsi_str} | {r['ATR_Pct']:.1%} |\n")
            
    print(f"Successfully generated trading dashboard report at: {markdown_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Nasdaq 50 Trading Practice Dashboard")
    parser.add_argument("--date", type=str, help="Target date in YYYY-MM-DD format", default=None)
    args = parser.parse_args()
    
    df, final_date = get_market_data(args.date)
    if df is not None:
        generate_report(df, final_date)
        
        print("\n" + "="*50)
        print(f"📊 SUMMARY OF TOP MOMENTUM (Date: {final_date})")
        for idx, row in df.sort_values(by="Momentum_3M", ascending=False).head(5).iterrows():
            print(f"- {row['Ticker']}: 3M Return = {row['Momentum_3M']:.1%}, Mom Z-Score = {row['Mom_Z']:.2f}, WTA Status = {row['WTA_Status']}")
        print("="*50 + "\n")
