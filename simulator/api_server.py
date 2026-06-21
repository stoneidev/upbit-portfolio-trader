import os
import json
import sqlite3
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add simulator folder to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import engine

app = FastAPI(title="Nasdaq 50 Trading Simulator API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATE_FILE = "/Users/stoni/Projects/AI/simulator/data/simulator_state.json"
INITIAL_CASH = 100000.0
FEE_RATE = 0.0015  # 0.15% commission

class TradeRequest(BaseModel):
    ticker: str
    action: str  # "BUY" or "SELL"
    shares: int

class StepRequest(BaseModel):
    days: int

# In-memory session state fallback (persisted in JSON)
class SimState:
    def __init__(self):
        self.trading_dates = engine.get_trading_dates()
        self.reset()
        self.load()

    def reset(self):
        self.cash = INITIAL_CASH
        self.portfolio = {}  # Ticker -> {"shares": int, "avg_price": float}
        self.transactions = []
        self.date_idx = 0
        # Initialize equity curve with starting cash
        initial_date = self.trading_dates[0] if self.trading_dates else "2025-01-02"
        self.equity_curve = [{"Date": initial_date, "Value": INITIAL_CASH}]
        self.save()

    def save(self):
        state = {
            "cash": self.cash,
            "portfolio": self.portfolio,
            "transactions": self.transactions,
            "date_idx": self.date_idx,
            "equity_curve": self.equity_curve
        }
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def load(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r", encoding="utf-8") as f:
                    state = json.load(f)
                self.cash = state.get("cash", INITIAL_CASH)
                self.portfolio = state.get("portfolio", {})
                self.transactions = state.get("transactions", [])
                self.date_idx = state.get("date_idx", 0)
                self.equity_curve = state.get("equity_curve", [])
            except Exception as e:
                print(f"Error loading state file: {e}. Starting fresh.")
                self.reset()

    def get_current_date(self):
        if not self.trading_dates:
            return "2025-01-02"
        return self.trading_dates[self.date_idx]

    def get_holdings_valuation(self, df_market):
        total_val = 0.0
        for ticker, info in self.portfolio.items():
            shares = info["shares"]
            # Find current price
            row = df_market[df_market["Ticker"] == ticker]
            price = row.iloc[0]["Close"] if not row.empty else info["avg_price"]
            total_val += shares * price
        return total_val

state = SimState()

@app.get("/api/state")
def get_state():
    current_date = state.get_current_date()
    df, summary = engine.get_market_state(current_date)
    if df is None:
        raise HTTPException(status_code=500, detail="Failed to calculate market metrics.")
    
    # Calculate top 30 entries compared to previous day
    entered_top30 = []
    exited_top30 = []
    if state.date_idx > 0:
        try:
            prev_date = state.trading_dates[state.date_idx - 1]
            df_prev, _ = engine.get_market_state(prev_date)
            if df_prev is not None:
                # Map ticker -> Mom_Rank on previous day
                prev_ranks = dict(zip(df_prev["Ticker"], df_prev["Mom_Rank"]))
                # Map ticker -> Mom_Rank on current day
                curr_ranks = dict(zip(df["Ticker"], df["Mom_Rank"]))
                
                for ticker, curr_rank in curr_ranks.items():
                    prev_rank = prev_ranks.get(ticker)
                    if curr_rank <= 30 and prev_rank is not None and prev_rank > 30:
                        entered_top30.append({
                            "ticker": ticker,
                            "prev_rank": int(prev_rank),
                            "curr_rank": int(curr_rank)
                        })
                    elif (prev_rank is not None and prev_rank <= 30) and curr_rank > 30:
                        exited_top30.append({
                            "ticker": ticker,
                            "prev_rank": int(prev_rank),
                            "curr_rank": int(curr_rank)
                        })
        except Exception as e:
            print(f"Error calculating rank changes: {e}")
            
    # Convert DataFrame to records dict
    screener = df.to_dict(orient="records")
    return {
        "current_date": current_date,
        "date_index": state.date_idx,
        "total_days": len(state.trading_dates),
        "market_summary": summary,
        "screener": screener,
        "entered_top30": entered_top30,
        "exited_top30": exited_top30
    }


@app.get("/api/portfolio")
def get_portfolio():
    current_date = state.get_current_date()
    df_market, _ = engine.get_market_state(current_date)
    if df_market is None:
        raise HTTPException(status_code=500, detail="Failed to load market data.")
        
    holdings_val = state.get_holdings_valuation(df_market)
    total_val = state.cash + holdings_val
    return_pct = ((total_val - INITIAL_CASH) / INITIAL_CASH) * 100
    
    holdings_list = []
    for ticker, info in state.portfolio.items():
        shares = info["shares"]
        avg_price = info["avg_price"]
        row = df_market[df_market["Ticker"] == ticker]
        curr_price = row.iloc[0]["Close"] if not row.empty else avg_price
        val = shares * curr_price
        pnl = val - (shares * avg_price)
        pnl_pct = (pnl / (shares * avg_price)) * 100 if avg_price > 0 else 0.0
        
        holdings_list.append({
            "ticker": ticker,
            "shares": shares,
            "avg_price": avg_price,
            "current_price": curr_price,
            "value": val,
            "pnl": pnl,
            "pnl_pct": pnl_pct
        })
        
    total_realized_pnl = sum(t.get("Realized_PnL", 0.0) for t in state.transactions if t.get("Type") == "SELL")
    return {
        "cash": state.cash,
        "holdings_value": holdings_val,
        "total_value": total_val,
        "return_pct": return_pct,
        "total_realized_pnl": total_realized_pnl,
        "holdings": holdings_list,
        "transactions": state.transactions,
        "equity_curve": state.equity_curve
    }

@app.post("/api/trade")
def execute_trade(req: TradeRequest):
    current_date = state.get_current_date()
    df_market, _ = engine.get_market_state(current_date)
    if df_market is None:
        raise HTTPException(status_code=500, detail="Market data unavailable.")
        
    ticker = req.ticker.upper()
    action = req.action.upper()
    shares = req.shares
    
    if shares <= 0:
        raise HTTPException(status_code=400, detail="Shares count must be positive.")
        
    row = df_market[df_market["Ticker"] == ticker]
    if row.empty:
        raise HTTPException(status_code=400, detail=f"Ticker {ticker} not found in Top 50.")
        
    price = row.iloc[0]["Close"]
    cost = shares * price
    fee = cost * FEE_RATE
    
    if action == "BUY":
        total_cost = cost + fee
        if state.cash < total_cost:
            raise HTTPException(status_code=400, detail=f"Insufficient cash. Need ${total_cost:,.2f}, have ${state.cash:,.2f}")
            
        state.cash -= total_cost
        if ticker in state.portfolio:
            curr_shares = state.portfolio[ticker]["shares"]
            curr_avg = state.portfolio[ticker]["avg_price"]
            new_shares = curr_shares + shares
            new_avg = ((curr_shares * curr_avg) + cost) / new_shares
            state.portfolio[ticker] = {"shares": new_shares, "avg_price": new_avg}
        else:
            state.portfolio[ticker] = {"shares": shares, "avg_price": price}
            
        state.transactions.append({
            "Date": current_date,
            "Ticker": ticker,
            "Type": "BUY",
            "Shares": shares,
            "Price": price,
            "Fee": fee,
            "Net Cash Flow": -total_cost
        })
        
    elif action == "SELL":
        if ticker not in state.portfolio or state.portfolio[ticker]["shares"] < shares:
            raise HTTPException(status_code=400, detail="Insufficient shares owned.")
            
        avg_price = state.portfolio[ticker]["avg_price"]
        cost_basis = shares * avg_price
        
        # Calculate fees: 0.15% on buy, 0.15% on sell
        buy_fee = cost_basis * FEE_RATE
        sell_fee = fee
        
        realized_pnl = cost - cost_basis - buy_fee - sell_fee
        realized_pnl_pct = (realized_pnl / cost_basis) * 100 if cost_basis > 0 else 0.0
        
        net_revenue = cost - fee
        state.cash += net_revenue
        state.portfolio[ticker]["shares"] -= shares
        if state.portfolio[ticker]["shares"] == 0:
            del state.portfolio[ticker]
            
        state.transactions.append({
            "Date": current_date,
            "Ticker": ticker,
            "Type": "SELL",
            "Shares": shares,
            "Price": price,
            "Fee": fee,
            "Net Cash Flow": net_revenue,
            "Realized_PnL": realized_pnl,
            "Realized_PnL_Pct": realized_pnl_pct
        })
    else:
        raise HTTPException(status_code=400, detail="Invalid trade action. Use BUY or SELL.")
        
    state.save()
    return {"status": "success", "message": f"Successfully executed {action} of {shares} {ticker}"}

@app.post("/api/step")
def progress_time(req: StepRequest):
    new_idx = state.date_idx + req.days
    if new_idx >= len(state.trading_dates):
        state.date_idx = len(state.trading_dates) - 1
    elif new_idx < 0:
        state.date_idx = 0
    else:
        state.date_idx = new_idx
        
    current_date = state.get_current_date()
    df_market, _ = engine.get_market_state(current_date)
    
    # Calculate new portfolio value to append to equity curve
    holdings_val = state.get_holdings_valuation(df_market)
    total_val = state.cash + holdings_val
    
    # Erase future equity curve entries if we travel back in time
    state.equity_curve = [entry for entry in state.equity_curve if entry["Date"] <= current_date]
    
    # Check if we already have an entry for this date in equity curve (to avoid duplicates)
    if not state.equity_curve or state.equity_curve[-1]["Date"] != current_date:
        state.equity_curve.append({"Date": current_date, "Value": total_val})
        
    state.save()
    return {"status": "success", "current_date": current_date}

@app.get("/api/chart/{ticker}")
def get_stock_chart(ticker: str):
    current_date = state.get_current_date()
    ticker = ticker.upper()
    conn = sqlite3.connect(engine.DB_PATH)
    cursor = conn.cursor()
    # Query last 60 trading days of data up to current_date for the chart
    cursor.execute("""
        SELECT Date, Open, High, Low, Close, Volume
        FROM daily_prices
        WHERE Ticker = ? AND Date <= ?
        ORDER BY Date DESC
        LIMIT 60
    """, (ticker, current_date))
    rows = cursor.fetchall()
    conn.close()
    
    # Reverse rows to have chronological order
    chart_data = []
    for r in reversed(rows):
        chart_data.append({
            "Date": r[0],
            "Open": r[1],
            "High": r[2],
            "Low": r[3],
            "Close": r[4],
            "Volume": r[5]
        })
    return chart_data

@app.post("/api/reset")
def reset_simulation():
    state.reset()
    return {"status": "success", "message": "Simulation reset successfully."}
