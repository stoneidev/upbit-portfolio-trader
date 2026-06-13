import requests
import pandas as pd
import numpy as np
import time
import datetime
import os

UPBIT_API_URL = "https://api.upbit.com/v1"

class UpbitAutoTrader:
    def __init__(self, mode="paper", initial_balance=1000000.0, market="KRW-XRP"):
        """
        Upbit Auto Trader for XRP/KRW.
        mode: 'paper' (Mock trading) or 'live' (Real trading using Upbit keys)
        """
        self.mode = mode
        self.market = market
        self.interval = "15" # 15 minutes
        self.fee_rate = 0.0005 # Upbit standard fee: 0.05%
        
        # Simulated Wallet (Paper Mode)
        self.balance = initial_balance
        self.position_size = 0.0
        self.position_avg_price = 0.0
        
        # Strategy parameters
        self.z_thresh = -1.2
        self.vol_power_thresh = 50.0  # Upbit Volume Power (Taker Buy Ratio proxy) > 50%
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
        self.log_file = "upbit_trader_log.txt"
        self.write_log(f"=== Upbit XRP Bot Started at {datetime.datetime.now()} ===")
        self.write_log(f"Mode: {self.mode.upper()} | Initial Capital: {self.balance:,.0f} 원")
        self.write_log(f"Parameters: Z-Thresh={self.z_thresh}, VolPower-Thresh={self.vol_power_thresh}%, TP={self.tp_pct}%, SL={self.sl_pct}%")

    def write_log(self, msg):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {msg}\n"
        print(log_line.strip())
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_line)

    def fetch_upbit_candles(self, count=101):
        """
        Fetches the latest count candles from Upbit.
        Upbit returns newest first, so we reverse it to match standard chronological order.
        """
        url = f"{UPBIT_API_URL}/candles/minutes/{self.interval}"
        params = {
            "market": self.market,
            "count": count
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            # Reverse list to chronological order (oldest first)
            df = pd.DataFrame(data).iloc[::-1].reset_index(drop=True)
            
            # Rename columns to standard names
            df = df.rename(columns={
                "trade_price": "close",
                "opening_price": "open",
                "high_price": "high",
                "low_price": "low",
                "candle_acc_trade_volume": "volume",
                "candle_date_time_kst": "time"
            })
            
            # Cast to numeric
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                
            return df
        except Exception as e:
            self.write_log(f"Error fetching Upbit candles: {e}")
            return None

    def fetch_upbit_ticker(self):
        """
        Fetches the current live price of KRW-XRP from Upbit.
        """
        url = f"{UPBIT_API_URL}/ticker"
        params = {"markets": self.market}
        try:
            res = requests.get(url, params=params, timeout=10).json()
            return float(res[0]["trade_price"])
        except Exception as e:
            self.write_log(f"Error fetching Upbit ticker: {e}")
            return None

    def fetch_upbit_volume_power(self, tick_count=100):
        """
        Calculates real-time Taker Buy Ratio (Volume Power) from Upbit ticks.
        ask_bid: 'BID' (매수 체결 - buyer aggressively hits ask = taker buy)
                 'ASK' (매도 체결 - seller aggressively hits bid = taker sell)
        Volume Power (%) = (BID Volume / Total Volume) * 100
        """
        url = f"{UPBIT_API_URL}/trades/ticks"
        params = {
            "market": self.market,
            "count": tick_count
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            df = pd.DataFrame(data)
            
            df["trade_volume"] = pd.to_numeric(df["trade_volume"])
            total_vol = df["trade_volume"].sum()
            
            # Sum volume of buyer-initiated ticks (BID)
            bid_vol = df[df["ask_bid"] == "BID"]["trade_volume"].sum()
            
            volume_power = (bid_vol / (total_vol + 1e-9)) * 100
            return volume_power
        except Exception as e:
            self.write_log(f"Error calculating Upbit Volume Power: {e}")
            return 50.0 # return neutral

    def calculate_indicators(self, df):
        if df is None or len(df) < 100:
            return None
        
        # Z-Score (100 MA)
        df["ma_100"] = df["close"].rolling(window=100).mean()
        df["std_100"] = df["close"].rolling(window=100).std()
        df["z_score"] = (df["close"] - df["ma_100"]) / (df["std_100"] + 1e-9)
        
        last_row = df.iloc[-1]
        return {
            "z_score": last_row["z_score"],
            "close": last_row["close"]
        }

    def execute_trade_cycle(self):
        """
        Main execution loop. Runs every 30 seconds.
        """
        # 1. Fetch current live price
        current_price = self.fetch_upbit_ticker()
        if not current_price:
            return

        # 2. Fetch candles for Z-score (excluding current incomplete candle)
        df = self.fetch_upbit_candles(count=101)
        if df is None:
            return
        
        df_closed = df.iloc[:-1].copy()
        indicators = self.calculate_indicators(df_closed)
        if not indicators:
            return
            
        z_score = indicators["z_score"]
        
        # 3. Fetch real-time Volume Power (Taker Buy proxy)
        vol_power = self.fetch_upbit_volume_power(tick_count=100)

        # 4. Decision Logic
        if not self.in_trade:
            # Check entry conditions
            c1 = z_score < self.z_thresh
            c2 = vol_power > self.vol_power_thresh
            
            print(f"\r[SCAN] XRP Price: {current_price:,.0f}원 | Z-Score: {z_score:.2f} | Vol Power: {vol_power:.1f}%", end="", flush=True)
            
            if c1 and c2:
                # 1차 Entry
                self.in_trade = True
                self.level_1_price = current_price
                self.allocated_capital = self.balance * 0.30 # Allocate 30% of total cash
                
                self.balance -= self.allocated_capital
                
                # 1차 Buy (30% weight)
                self.position_size = (self.allocated_capital * 0.30) / current_price
                self.position_avg_price = current_price
                
                self.level_2_filled = False
                self.level_3_filled = False
                self.trailing_active = False
                self.peak_price = 0.0
                
                self.write_log(f"--- [BUY 1차] XRP Price: {current_price:,.0f}원 ---")
                self.write_log(f"Allocated Cap: {self.allocated_capital:,.0f}원 | Pos Size: {self.position_size:.2f} XRP | Avg Price: {self.position_avg_price:,.0f}원")
        else:
            # Monitoring exits & grid fills
            target_level_2 = self.level_1_price * 0.990 # -1.0%
            target_level_3 = self.level_1_price * 0.980 # -2.0%
            
            cur_return = (current_price - self.position_avg_price) / self.position_avg_price * 100
            print(f"\r[MONITOR] XRP Price: {current_price:,.0f}원 | Avg Entry: {self.position_avg_price:,.0f}원 | Return: {cur_return:+.2f}%", end="", flush=True)
            
            # Check Grid Fills
            if not self.level_2_filled and current_price <= target_level_2:
                # 2차 buy (30% weight)
                buy_amount = self.allocated_capital * 0.30
                new_tokens = buy_amount / current_price
                self.position_avg_price = (self.position_size * self.position_avg_price + buy_amount) / (self.position_size + new_tokens)
                self.position_size += new_tokens
                self.level_2_filled = True
                self.write_log(f"--- [GRID FILL 2차] XRP Price: {current_price:,.0f}원 | Avg Price: {self.position_avg_price:,.0f}원 ---")
                
            if not self.level_3_filled and current_price <= target_level_3:
                # 3차 buy (40% weight)
                buy_amount = self.allocated_capital * 0.40
                new_tokens = buy_amount / current_price
                self.position_avg_price = (self.position_size * self.position_avg_price + buy_amount) / (self.position_size + new_tokens)
                self.position_size += new_tokens
                self.level_3_filled = True
                self.write_log(f"--- [GRID FILL 3차] XRP Price: {current_price:,.0f}원 | Avg Price: {self.position_avg_price:,.0f}원 ---")
                
            # Check Exits
            sl_price = self.position_avg_price * (1 + self.sl_pct / 100.0)
            tp_activation_price = self.position_avg_price * (1 + self.tp_pct / 100.0)
            
            if not self.trailing_active:
                if current_price >= tp_activation_price:
                    self.trailing_active = True
                    self.peak_price = max(current_price, tp_activation_price)
                    self.write_log(f"--- [TRAILING ACTIVATED] XRP Price: {current_price:,.0f}원 ---")
            else:
                if current_price > self.peak_price:
                    self.peak_price = current_price
                    
            # Exits execution
            if self.trailing_active:
                # Trail by 0.2%
                stop_price = self.peak_price * 0.998
                if current_price <= stop_price:
                    exit_price = max(stop_price, self.position_avg_price * (1 + (self.tp_pct - 0.2)/100.0))
                    
                    gross_value = self.position_size * exit_price
                    total_weight = 0.3
                    if self.level_2_filled: total_weight += 0.3
                    if self.level_3_filled: total_weight += 0.4
                    
                    fee = self.allocated_capital * total_weight * self.fee_rate
                    net_return_amount = gross_value - fee
                    
                    self.balance += (self.allocated_capital * (1 - total_weight)) + net_return_amount
                    
                    profit_pct = (exit_price - self.position_avg_price) / self.position_avg_price * 100
                    self.write_log(f"=== [EXIT: TRAILING STOP] Price: {exit_price:,.0f}원 | Profit: {profit_pct:+.2f}% | Fee: {fee:,.0f}원 ===")
                    self.write_log(f"New Wallet Balance: {self.balance:,.0f} 원")
                    self.reset_trade_state()
            else:
                if current_price <= sl_price:
                    total_weight = 0.3
                    if self.level_2_filled: total_weight += 0.3
                    if self.level_3_filled: total_weight += 0.4
                    
                    gross_value = self.position_size * current_price
                    fee = self.allocated_capital * total_weight * self.fee_rate
                    net_return_amount = gross_value - fee
                    
                    self.balance += (self.allocated_capital * (1 - total_weight)) + net_return_amount
                    
                    profit_pct = (current_price - self.position_avg_price) / self.position_avg_price * 100
                    self.write_log(f"=== [EXIT: STOP LOSS] Price: {current_price:,.0f}원 | Profit: {profit_pct:+.2f}% | Fee: {fee:,.0f}원 ===")
                    self.write_log(f"New Wallet Balance: {self.balance:,.0f} 원")
                    self.reset_trade_state()

        # Save dashboard state for the HTML page
        self.save_dashboard_data(current_price, z_score, vol_power, df)

    def get_recent_logs(self, limit=15):
        if not os.path.exists(self.log_file):
            return []
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                return [line.strip() for line in lines[-limit:] if line.strip()]
        except Exception as e:
            return [f"Error reading logs: {e}"]

    def save_dashboard_data(self, current_price, z_score, vol_power, df_candles):
        import json
        dashboard_file = "dashboard_data.json"
        
        # Format candles to match UI requirements (only keeping necessary keys)
        candle_list = []
        if df_candles is not None:
            for _, r in df_candles.tail(40).iterrows():
                candle_list.append({
                    "time": r.get("time"),
                    "open": float(r.get("open", 0)),
                    "high": float(r.get("high", 0)),
                    "low": float(r.get("low", 0)),
                    "close": float(r.get("close", 0)),
                    "volume": float(r.get("volume", 0))
                })
        
        data = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": self.mode,
            "market": self.market,
            "balance": self.balance,
            "position_size": self.position_size,
            "position_avg_price": self.position_avg_price,
            "current_price": current_price,
            "z_score": z_score,
            "vol_power": vol_power,
            "in_trade": self.in_trade,
            "allocated_capital": self.allocated_capital,
            "level_1_price": self.level_1_price,
            "level_2_filled": self.level_2_filled,
            "level_3_filled": self.level_3_filled,
            "trailing_active": self.trailing_active,
            "peak_price": self.peak_price,
            "candles": candle_list,
            "logs": self.get_recent_logs(15)
        }
        try:
            with open(dashboard_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            self.write_log(f"Error saving dashboard data: {e}")

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
    # Initialize trader for Upbit XRP/KRW
    trader = UpbitAutoTrader(mode="paper", initial_balance=1000000.0)
    
    print("\nRunning Upbit real-time monitoring loop (Mock trading mode)...")
    print("Press Ctrl+C to exit.")
    
    try:
        while True:
            trader.execute_trade_cycle()
            time.sleep(30) # Poll every 30 seconds
    except KeyboardInterrupt:
        print("\nUpbit Bot stopped by user.")
