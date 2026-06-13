import requests
import pandas as pd
import numpy as np
import time
import datetime
import os

# Binance API URL
BINANCE_API_URL = "https://api.binance.com"

class BTCAutoTrader:
    def __init__(self, mode="paper", initial_balance=1000000.0, symbol="BTCUSDT"):
        """
        BTC Auto Trader.
        mode: 'paper' (Mock trading) or 'live' (Real trading - placeholder)
        """
        self.mode = mode
        self.symbol = symbol
        self.interval = "15m"
        self.fee_rate = 0.0004 # 0.04% Maker fee
        
        # Simulated Wallet (Paper Mode)
        self.balance = initial_balance
        self.position_size = 0.0
        self.position_avg_price = 0.0
        
        # Default Strategy parameters (XRP/BTC optimized sweet-spot)
        self.z_thresh = -1.2
        self.tb_thresh = 0.49
        self.tp_pct = 0.5
        self.sl_pct = -2.0
        
        # Grid parameters
        self.allocated_capital = 0.0
        self.level_1_price = 0.0
        self.level_2_filled = False
        self.level_3_filled = False
        
        # Trailing stop parameters
        self.in_trade = False
        self.trailing_active = False
        self.peak_price = 0.0
        
        # Logging
        self.log_file = "btc_trader_log.txt"
        self.write_log(f"=== Bot Started at {datetime.datetime.now()} ===")
        self.write_log(f"Mode: {self.mode.upper()} | Initial Capital: {self.balance:,.0f} KRW")
        self.write_log(f"Parameters: Z-Thresh={self.z_thresh}, TB-Thresh={self.tb_thresh}, TP={self.tp_pct}%, SL={self.sl_pct}%")

    def write_log(self, msg):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {msg}\n"
        print(log_line.strip())
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_line)

    def fetch_recent_klines(self, limit=120):
        """
        Fetches the latest limit OHLCV candles from Binance.
        """
        url = f"{BINANCE_API_URL}/api/v3/klines"
        params = {
            "symbol": self.symbol,
            "interval": self.interval,
            "limit": limit
        }
        try:
            response = requests.get(url, params=params)
            data = response.json()
            df = pd.DataFrame(data, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "count", "taker_buy_volume",
                "taker_buy_quote_volume", "ignore"
            ])
            # cast numeric
            for col in ["open", "high", "low", "close", "volume", "taker_buy_volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            return df
        except Exception as e:
            self.write_log(f"Error fetching candles: {e}")
            return None

    def fetch_ticker_price(self):
        """
        Fetches the current real-time price of BTCUSDT.
        """
        url = f"{BINANCE_API_URL}/api/v3/ticker/price"
        params = {"symbol": self.symbol}
        try:
            res = requests.get(url, params=params).json()
            return float(res["price"])
        except Exception as e:
            self.write_log(f"Error fetching ticker price: {e}")
            return None

    def calculate_indicators(self, df):
        if df is None or len(df) < 100:
            return None
        
        # 1. Z-Score (100 MA)
        df["ma_100"] = df["close"].rolling(window=100).mean()
        df["std_100"] = df["close"].rolling(window=100).std()
        df["z_score"] = (df["close"] - df["ma_100"]) / (df["std_100"] + 1e-9)
        
        # 2. Smoothed Taker Buy Ratio
        df["taker_buy_ratio"] = df["taker_buy_volume"] / (df["volume"] + 1e-9)
        df["taker_buy_ratio_smoothed"] = df["taker_buy_ratio"].rolling(window=3).mean()
        
        # Get last values
        last_row = df.iloc[-1]
        return {
            "z_score": last_row["z_score"],
            "tb_ratio": last_row["taker_buy_ratio_smoothed"],
            "close": last_row["close"]
        }

    def execute_trade_cycle(self):
        """
        Main execution loop. Run every 30 seconds.
        """
        # 1. Fetch current live price
        current_price = self.fetch_ticker_price()
        if not current_price:
            return

        # 2. Fetch 15m candles to compute indicators
        # Note: we exclude the current incomplete candle by fetching the last 101 candles
        # and dropping the very last row (which is the live moving candle).
        df = self.fetch_recent_klines(limit=101)
        if df is None:
            return
        
        # Drop the last candle (incomplete live candle) to ensure we calculate indicators on CLOSED candles
        df_closed = df.iloc[:-1].copy()
        indicators = self.calculate_indicators(df_closed)
        if not indicators:
            return
            
        z_score = indicators["z_score"]
        tb_ratio = indicators["tb_ratio"]
        last_closed_close = indicators["close"]

        # 3. Decision Logic
        if not self.in_trade:
            # Check entry conditions
            c1 = z_score < self.z_thresh
            c2 = tb_ratio > self.tb_thresh
            
            # Print status periodically
            print(f"\r[SCAN] BTC Live Price: ${current_price:,.2f} | Z-Score: {z_score:.2f} | Taker Buy Ratio: {tb_ratio*100:.1f}%", end="")
            
            if c1 and c2:
                # 1차 Entry
                self.in_trade = True
                self.level_1_price = current_price
                self.allocated_capital = self.balance * 0.30  # Allocate 30% of total capital
                
                # Deduct 1차 capital from balance
                self.balance -= self.allocated_capital
                
                # 1차 Buy (30% weight of allocated capital)
                self.position_size = (self.allocated_capital * 0.30) / current_price
                self.position_avg_price = current_price
                
                self.level_2_filled = False
                self.level_3_filled = False
                self.trailing_active = False
                self.peak_price = 0.0
                
                self.write_log(f"--- [BUY 1차] BTC Price: ${current_price:,.2f} ---")
                self.write_log(f"Allocated Cap: {self.allocated_capital:,.0f} KRW | Pos Size: {self.position_size:.6f} BTC | Avg Price: ${self.position_avg_price:,.2f}")
        else:
            # We are currently in a trade, monitoring exits & grid fills using live ticker price
            target_level_2 = self.level_1_price * 0.990 # -1.0%
            target_level_3 = self.level_1_price * 0.980 # -2.0%
            
            # Print trade status
            cur_return = (current_price - self.position_avg_price) / self.position_avg_price * 100
            print(f"\r[MONITOR] BTC Price: ${current_price:,.2f} | Avg Entry: ${self.position_avg_price:,.2f} | Return: {cur_return:+.2f}%", end="")
            
            # 1. Check Grid Fills
            if not self.level_2_filled and current_price <= target_level_2:
                # 2차 buy (30% weight of allocated capital)
                buy_amount = self.allocated_capital * 0.30
                new_tokens = buy_amount / current_price
                
                # Recalculate average price
                self.position_avg_price = (self.position_size * self.position_avg_price + buy_amount) / (self.position_size + new_tokens)
                self.position_size += new_tokens
                self.level_2_filled = True
                self.write_log(f"--- [GRID FILL 2차] BTC Price: ${current_price:,.2f} | New Avg Price: ${self.position_avg_price:,.2f} ---")
                
            if not self.level_3_filled and current_price <= target_level_3:
                # 3차 buy (40% weight of allocated capital)
                buy_amount = self.allocated_capital * 0.40
                new_tokens = buy_amount / current_price
                
                # Recalculate average price
                self.position_avg_price = (self.position_size * self.position_avg_price + buy_amount) / (self.position_size + new_tokens)
                self.position_size += new_tokens
                self.level_3_filled = True
                self.write_log(f"--- [GRID FILL 3차] BTC Price: ${current_price:,.2f} | New Avg Price: ${self.position_avg_price:,.2f} ---")
                
            # 2. Check Exits (Trailing Stop or Stop Loss)
            sl_price = self.position_avg_price * (1 + self.sl_pct / 100.0)
            tp_activation_price = self.position_avg_price * (1 + self.tp_pct / 100.0)
            
            # Trailing Stop Peak update
            if not self.trailing_active:
                if current_price >= tp_activation_price:
                    self.trailing_active = True
                    self.peak_price = max(current_price, tp_activation_price)
                    self.write_log(f"--- [TRAILING ACTIVATED] BTC Price: ${current_price:,.2f} ---")
            else:
                if current_price > self.peak_price:
                    self.peak_price = current_price
                    
            # Check exits
            if self.trailing_active:
                # Trail by 0.2%
                stop_price = self.peak_price * 0.998
                if current_price <= stop_price:
                    # Trailing Stop Triggered! Exit.
                    exit_price = max(stop_price, self.position_avg_price * (1 + (self.tp_pct - 0.2)/100.0))
                    
                    # Calculate final return
                    gross_value = self.position_size * exit_price
                    total_weight = 0.3
                    if self.level_2_filled: total_weight += 0.3
                    if self.level_3_filled: total_weight += 0.4
                    
                    # Deduct Maker fee
                    fee = self.allocated_capital * total_weight * self.fee_rate
                    net_return_amount = gross_value - fee
                    
                    # Add remaining sub-capital + trade proceeds back to total balance
                    self.balance += (self.allocated_capital * (1 - total_weight)) + net_return_amount
                    
                    profit_pct = (exit_price - self.position_avg_price) / self.position_avg_price * 100
                    self.write_log(f"=== [EXIT: TRAILING STOP] Price: ${exit_price:,.2f} | Profit: {profit_pct:+.2f}% | Fee: {fee:,.0f}원 ===")
                    self.write_log(f"New Wallet Balance: {self.balance:,.0f} KRW")
                    self.reset_trade_state()
            else:
                # Check Stop Loss
                if current_price <= sl_price:
                    # Stop Loss Triggered! Exit at loss.
                    total_weight = 0.3
                    if self.level_2_filled: total_weight += 0.3
                    if self.level_3_filled: total_weight += 0.4
                    
                    gross_value = self.position_size * current_price
                    fee = self.allocated_capital * total_weight * self.fee_rate
                    net_return_amount = gross_value - fee
                    
                    # Refund unused grid cash + loss proceeds
                    self.balance += (self.allocated_capital * (1 - total_weight)) + net_return_amount
                    
                    profit_pct = (current_price - self.position_avg_price) / self.position_avg_price * 100
                    self.write_log(f"=== [EXIT: STOP LOSS] Price: ${current_price:,.2f} | Profit: {profit_pct:+.2f}% | Fee: {fee:,.0f}원 ===")
                    self.write_log(f"New Wallet Balance: {self.balance:,.0f} KRW")
                    self.reset_trade_state()

    def reset_trade_state(self):
        self.in_trade = False
        self.position_size = 0.0
        self.position_avg_price = 0.0
        self.allocated_capital = 0.0
        self.level_1_price = 0.0
        self.level_2_filled = False
        self.level_3_filled = False
        self.trailing_active = False
        self.peak_price = 0.0

if __name__ == "__main__":
    # Initialize trader in Paper Trading Mode
    trader = BTCAutoTrader(mode="paper", initial_balance=1000000.0)
    
    print("\nRunning live-monitoring loop (Mock trading mode)...")
    print("Press Ctrl+C to exit.")
    
    try:
        while True:
            trader.execute_trade_cycle()
            time.sleep(30) # Poll every 30 seconds
    except KeyboardInterrupt:
        print("\nBot stopped by user.")
