import os
import sys
import argparse
import pandas as pd
import numpy as np

# Add simulator folder to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import engine

INITIAL_CASH = 100_000.0
TRANSACTION_FEE_RATE = 0.0015  # 0.15% transaction fee (commission)

class Simulator:
    def __init__(self):
        self.cash = INITIAL_CASH
        self.portfolio = {}  # Ticker -> {"shares": int, "avg_price": float}
        self.transactions = []  # List of dicts
        self.trading_dates = engine.get_trading_dates()
        
        if not self.trading_dates:
            print("Error: No trading dates available in the database from 2025-01-01. Did you run db_prep.py?")
            sys.exit(1)
            
        self.current_date_idx = 0
        self.current_date = self.trading_dates[self.current_date_idx]
        
        # Load initial market metrics
        self.update_market_data()

    def update_market_data(self):
        self.current_date = self.trading_dates[self.current_date_idx]
        df, summary = engine.get_market_state(self.current_date)
        self.df_market = df
        self.market_summary = summary

    def get_portfolio_value(self):
        total_holdings_val = 0.0
        portfolio_details = []
        
        for ticker, info in self.portfolio.items():
            shares = info["shares"]
            avg_price = info["avg_price"]
            
            # Get current price
            curr_row = self.df_market[self.df_market["Ticker"] == ticker]
            if not curr_row.empty:
                curr_price = curr_row.iloc[0]["Close"]
            else:
                curr_price = avg_price  # Fallback if ticker data is missing
                
            val = shares * curr_price
            pnl = val - (shares * avg_price)
            pnl_pct = (pnl / (shares * avg_price)) * 100 if avg_price > 0 else 0.0
            
            total_holdings_val += val
            portfolio_details.append({
                "Ticker": ticker,
                "Shares": shares,
                "Avg Price": avg_price,
                "Current Price": curr_price,
                "Value": val,
                "PnL": pnl,
                "PnL%": pnl_pct
            })
            
        return total_holdings_val, portfolio_details

    def print_dashboard(self, limit=15):
        if self.df_market is None:
            print(f"No market data available for {self.current_date}")
            return
            
        print("\n" + "="*80)
        print(f"📊 NASDAQ TOP 50 MARKET DASHBOARD | DATE: {self.current_date}")
        print("="*80)
        
        summary = self.market_summary
        print(f"• Market Breadth (20 SMA): {summary['above_ma20']:.1%} | (50 SMA): {summary['above_ma50']:.1%}")
        print(f"• Market Avg RSI (14d): {summary['avg_rsi']:.1f} | Market Avg Z-Score: {summary['avg_z']:.2f}")
        print(f"• WTA Index: {summary['wta_index']:.1%} | Market Regime: {summary['market_regime']}")
        print("-"*80)
        
        # Display WTA Leaders
        wta_leaders = self.df_market.sort_values(by="Momentum_3M", ascending=False).head(5)
        print("🏆 WINNER-TAKE-ALL (WTA) LEADERS:")
        for _, r in wta_leaders.iterrows():
            print(f"  [{r['Mom_Rank']}] {r['Ticker']:<5} | Price: {r['Close']:7.2f} | 3M Ret: {r['Momentum_3M']:7.1%} | Mom Z: {r['Mom_Z']:5.2f} | {r['WTA_Status']}")
            
        print("-"*80)
        # Display Screen table
        print(f"📋 TOP {limit} SCREENER (Sorted by Ticker):")
        print(f"{'Ticker':<7} | {'Price':<8} | {'Trend':<8} | {'Mom Rank':<8} | {'3M Ret':<7} | {'Mom Z':<5} | {'Z-Score':<7} | {'RSI':<5} | {'ATR%':<5}")
        print("-" * 80)
        
        df_sorted = self.df_market.sort_values(by="Ticker").head(limit)
        for _, r in df_sorted.iterrows():
            mom_str = f"{r['Momentum_3M']:.1%}"
            print(f"{r['Ticker']:<7} | {r['Close']:8.2f} | {r['Trend']:<8} | {r['Mom_Rank']:<8} | {mom_str:<7} | {r['Mom_Z']:5.2f} | {r['Z-Score']:7.2f} | {r['RSI']:5.1f} | {r['ATR_Pct']:5.1%}")
        
        if len(self.df_market) > limit:
            print(f"... and {len(self.df_market) - limit} more stocks. Open nasdaq_trading_dashboard.md to view all.")
            
        # Also write the full markdown report to the artifact directory for the user to open
        self.save_markdown_report()

    def save_markdown_report(self):
        """Saves a rich markdown report of the current state to the artifact directory."""
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        import generate_dashboard
        generate_dashboard.generate_report(self.df_market, self.current_date)

    def print_portfolio(self):
        holdings_val, details = self.get_portfolio_value()
        total_val = self.cash + holdings_val
        total_pnl = total_val - INITIAL_CASH
        total_pnl_pct = (total_pnl / INITIAL_CASH) * 100
        
        print("\n" + "="*80)
        print("💼 MY PORTFOLIO STATUS")
        print("="*80)
        print(f"• Available Cash       : ${self.cash:,.2f}")
        print(f"• Holdings Valuation   : ${holdings_val:,.2f}")
        print(f"• Total Portfolio Value: ${total_val:,.2f}")
        print(f"• Total Cumulative Return: {total_pnl_pct:+.2f}% (${total_pnl:+,.2f})")
        print("-"*80)
        
        if not details:
            print("No holdings. Your portfolio is currently 100% Cash.")
        else:
            print(f"{'Ticker':<7} | {'Shares':<7} | {'Avg Price':<10} | {'Curr Price':<10} | {'Current Value':<13} | {'Unrealized PnL':<15}")
            print("-" * 80)
            for d in details:
                pnl_str = f"${d['PnL']:+,.2f} ({d['PnL%']:+.2f}%)"
                print(f"{d['Ticker']:<7} | {d['Shares']:<7} | ${d['Avg Price']:9.2f} | ${d['Current Price']:9.2f} | ${d['Value']:12,.2f} | {pnl_str:<15}")
        print("="*80 + "\n")

    def buy_stock(self, ticker, shares):
        ticker = ticker.upper()
        if self.df_market is None:
            print("Error: Market data not loaded.")
            return
            
        row = self.df_market[self.df_market["Ticker"] == ticker]
        if row.empty:
            print(f"Error: Ticker {ticker} is not available in the Nasdaq Top 50.")
            return
            
        price = row.iloc[0]["Close"]
        cost = shares * price
        fee = cost * TRANSACTION_FEE_RATE
        total_cost = cost + fee
        
        if self.cash < total_cost:
            print(f"❌ Purchase failed: Insufficient cash. Required: ${total_cost:,.2f}, Available: ${self.cash:,.2f}")
            return
            
        self.cash -= total_cost
        
        # Update portfolio
        if ticker in self.portfolio:
            curr_shares = self.portfolio[ticker]["shares"]
            curr_avg = self.portfolio[ticker]["avg_price"]
            new_shares = curr_shares + shares
            new_avg = ((curr_shares * curr_avg) + cost) / new_shares
            self.portfolio[ticker] = {"shares": new_shares, "avg_price": new_avg}
        else:
            self.portfolio[ticker] = {"shares": shares, "avg_price": price}
            
        # Log transaction
        self.transactions.append({
            "Date": self.current_date,
            "Ticker": ticker,
            "Type": "BUY",
            "Shares": shares,
            "Price": price,
            "Fee": fee,
            "Net Cash Flow": -total_cost
        })
        
        print(f"✅ Successfully bought {shares} shares of {ticker} @ ${price:.2f} (Total Cost: ${total_cost:,.2f})")

    def sell_stock(self, ticker, shares):
        ticker = ticker.upper()
        if ticker not in self.portfolio or self.portfolio[ticker]["shares"] < shares:
            curr_shares = self.portfolio[ticker]["shares"] if ticker in self.portfolio else 0
            print(f"❌ Sale failed: Insufficient shares. Trying to sell {shares}, but you hold {curr_shares}.")
            return
            
        row = self.df_market[self.df_market["Ticker"] == ticker]
        price = row.iloc[0]["Close"] if not row.empty else self.portfolio[ticker]["avg_price"]
        
        revenue = shares * price
        fee = revenue * TRANSACTION_FEE_RATE
        net_cash = revenue - fee
        
        self.cash += net_cash
        
        # Update portfolio
        self.portfolio[ticker]["shares"] -= shares
        if self.portfolio[ticker]["shares"] == 0:
            del self.portfolio[ticker]
            
        # Log transaction
        self.transactions.append({
            "Date": self.current_date,
            "Ticker": ticker,
            "Type": "SELL",
            "Shares": shares,
            "Price": price,
            "Fee": fee,
            "Net Cash Flow": net_cash
        })
        
        print(f"✅ Successfully sold {shares} shares of {ticker} @ ${price:.2f} (Net Cash Received: ${net_cash:,.2f})")

    def print_history(self):
        print("\n" + "="*80)
        print("📜 TRANSACTION HISTORY LOG")
        print("="*80)
        if not self.transactions:
            print("No transactions executed yet.")
        else:
            print(f"{'Date':<10} | {'Ticker':<6} | {'Type':<5} | {'Shares':<6} | {'Price':<8} | {'Fee':<6} | {'Net Cash Flow':<15}")
            print("-" * 80)
            for t in self.transactions:
                flow_str = f"${t['Net Cash Flow']:+,.2f}"
                print(f"{t['Date']:<10} | {t['Ticker']:<6} | {t['Type']:<5} | {t['Shares']:<6} | ${t['Price']:7.2f} | ${t['Fee']:5.2f} | {flow_str:<15}")
        print("="*80 + "\n")

    def next_day(self, days=1):
        target_idx = self.current_date_idx + days
        if target_idx >= len(self.trading_dates):
            print("⚠️ Cannot step forward: Reached end of available historical simulation data.")
            self.current_date_idx = len(self.trading_dates) - 1
        else:
            self.current_date_idx = target_idx
            
        self.update_market_data()
        print(f"\n⏩ Stepped forward {days} trading day(s) to Date: {self.current_date}")

def print_help():
    print("\n💡 AVAILABLE COMMANDS:")
    print("  d, dashboard      : Show market metrics, WTA leaders, and stock screener")
    print("  p, portfolio      : Show portfolio cash, holdings valuation, and cumulative PnL")
    print("  b <tick> <shares> : Buy stock (e.g. 'buy NVDA 100')")
    print("  s <tick> <shares> : Sell stock (e.g. 'sell AAPL 50')")
    print("  n, next           : Progress time by 1 trading day")
    print("  nw, next_week     : Progress time by 1 week (5 trading days)")
    print("  nm, next_month    : Progress time by 1 month (20 trading days)")
    print("  h, history        : View transaction logs")
    print("  help              : Show this help list")
    print("  q, quit           : Quit simulation and show final report\n")

def run_interactive():
    print("="*80)
    print("🚀 NASDAQ TOP 50 INTERACTIVE TRADING SIMULATOR v1.0")
    print("="*80)
    print("Practice Winner-Take-All and Z-Score strategies starting from January 2, 2025.")
    print("Look-ahead bias is fully prevented. You only see indicators computed from historical data.")
    print("="*80)
    
    sim = Simulator()
    print(f"Simulator initialized. Current Simulation Date: {sim.current_date}")
    print_help()
    
    # Auto print dashboard at start
    sim.print_dashboard()
    
    while True:
        try:
            cmd_line = input(f"[{sim.current_date} | Cash: ${sim.cash:,.2f}] Command > ").strip().split()
            if not cmd_line:
                continue
                
            cmd = cmd_line[0].lower()
            
            if cmd in ["q", "quit", "exit"]:
                print("\nQuitting simulation...")
                sim.print_portfolio()
                print("Thank you for playing!")
                break
                
            elif cmd in ["h", "help"]:
                print_help()
                
            elif cmd in ["d", "dashboard"]:
                limit = 15
                if len(cmd_line) > 1:
                    try:
                        limit = int(cmd_line[1])
                    except ValueError:
                        pass
                sim.print_dashboard(limit)
                
            elif cmd in ["p", "portfolio"]:
                sim.print_portfolio()
                
            elif cmd in ["n", "next"]:
                sim.next_day(1)
                sim.print_dashboard()
                
            elif cmd in ["nw", "next_week"]:
                sim.next_day(5)
                sim.print_dashboard()
                
            elif cmd in ["nm", "next_month"]:
                sim.next_day(20)
                sim.print_dashboard()
                
            elif cmd in ["b", "buy"]:
                if len(cmd_line) < 3:
                    print("Usage: buy <ticker> <shares> (e.g. 'buy NVDA 100')")
                    continue
                ticker = cmd_line[1]
                try:
                    shares = int(cmd_line[2])
                    if shares <= 0:
                        raise ValueError()
                except ValueError:
                    print("Error: Shares must be a positive integer.")
                    continue
                sim.buy_stock(ticker, shares)
                
            elif cmd in ["s", "sell"]:
                if len(cmd_line) < 3:
                    print("Usage: sell <ticker> <shares> (e.g. 'sell AAPL 50')")
                    continue
                ticker = cmd_line[1]
                try:
                    shares = int(cmd_line[2])
                    if shares <= 0:
                        raise ValueError()
                except ValueError:
                    print("Error: Shares must be a positive integer.")
                    continue
                sim.sell_stock(ticker, shares)
                
            elif cmd in ["history"]:
                sim.print_history()
                
            else:
                print(f"Unknown command: '{cmd}'. Type 'help' to see list of valid commands.")
                
        except KeyboardInterrupt:
            print("\nType 'quit' or 'q' to exit the simulator.")
        except Exception as e:
            print(f"Error executing command: {e}")

if __name__ == "__main__":
    run_interactive()
