import requests
import pandas as pd
import numpy as np
import time
import datetime
import os
import json

UPBIT_API_URL = "https://api.upbit.com/v1"

# Load local credentials from config.json (ignores on git)
TELEGRAM_TOKEN = ""
TELEGRAM_CHAT_ID = ""
CF_WORKER_TOKEN = "upbit-portfolio-secret-key-2026"

try:
    # Use relative or script-directory path for config.json
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            TELEGRAM_TOKEN = cfg.get("TELEGRAM_TOKEN", "")
            TELEGRAM_CHAT_ID = cfg.get("TELEGRAM_CHAT_ID", "")
            CF_WORKER_TOKEN = cfg.get("CF_WORKER_TOKEN", CF_WORKER_TOKEN)
except Exception as e:
    print(f"Error loading config.json: {e}")

def send_telegram_message(msg):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": msg
    }
    try:
        import requests
        res = requests.post(url, json=payload, timeout=5)
        if res.status_code != 200:
            print(f"Telegram send failed: {res.text}")
    except Exception as e:
        print(f"Telegram connection error: {e}")

class UpbitAutoTrader:
    def __init__(self, market, mode="paper", initial_balance=500000.0):
        """
        Upbit Auto Trader instance for a specific market (e.g. KRW-XRP, KRW-ETH).
        """
        self.mode = mode
        self.market = market
        self.interval = "15" # 15 minutes
        self.fee_rate = 0.0005 # Upbit standard fee: 0.05%
        self.slippage_per_side = 0.0005  # Applied only to market-order exits (SL)

        # Simulated Wallet (Paper Mode)
        self.balance = initial_balance
        self.position_size = 0.0
        self.position_avg_price = 0.0

        # Strategy parameters (Optimized dynamically by WFO)
        self.z_thresh = -1.2
        self.vol_power_thresh = 50.0  # Kept constant at 50% for volume power filter
        self.tp_pct = 0.5
        self.sl_pct = -2.0

        # Hybrid order model:
        #   L1 entry, L2/L3 grids, trailing TP -> LIMIT (no slippage)
        #   Stop-loss -> MARKET (slippage applied)
        # Pending limit entry state
        self.pending_entry_price = 0.0      # 0 means no pending entry
        self.pending_entry_expires = 0.0    # epoch seconds
        self.entry_timeout_sec = 60 * 60    # 1 hour to fill the L1 limit order

        # L3 time cutoff: if L3 grid was filled and price hasn't recovered to
        # avg * (1 + l3_bep_thresh_pct/100) within l3_cutoff_sec, force close.
        # Caps the worst-case drawdown of full-grid trades.
        self.l3_fill_time = 0.0
        self.l3_cutoff_sec = 16 * 15 * 60       # 16 × 15-min bars = 4 hours
        self.l3_bep_thresh_pct = -0.2           # accept up to 0.2% loss to exit

        # WFO Tracking
        self.last_opt_time = 0.0

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
        self.log_file = "upbit_portfolio_log.txt"
        self.write_log(f"=== [{self.market}] Instance Initialized | Capital: {self.balance:,.0f} 원 ===")

    def write_log(self, msg):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] [{self.market}] {msg}\n"
        print(log_line.strip())
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_line)

    def record_transaction(self, side, level, price, size, profit_pct=None):
        import json
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tx_path = os.path.join(script_dir, "trade_history.json")
        
        history = []
        if os.path.exists(tx_path):
            try:
                with open(tx_path, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except Exception as e:
                print(f"Error loading trade_history.json: {e}")
                history = []
                
        tx = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "market": self.market.replace("KRW-", ""),
            "side": side,
            "level": level,
            "price": float(price),
            "size": float(size),
            "value": float(price * size),
            "profit_pct": float(profit_pct) if profit_pct is not None else None,
            "balance": float(self.balance)
        }
        history.append(tx)
        
        if len(history) > 100:
            history = history[-100:]
            
        try:
            with open(tx_path, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving trade_history.json: {e}")

    def fetch_upbit_candles(self, count=100):
        """
        Fetches the latest count candles from Upbit for self.market.
        """
        url = f"{UPBIT_API_URL}/candles/minutes/{self.interval}"
        params = {
            "market": self.market,
            "count": count
        }
        try:
            res = requests.get(url, params=params, timeout=10)
            data = res.json()
            if not isinstance(data, list) or len(data) == 0:
                self.write_log(f"Error fetching candles: Invalid response structure.")
                return None
            
            # Upbit returns newest first, reverse it to chronological
            data.reverse()
            
            # Format to DataFrame
            candles = []
            for c in data:
                candles.append({
                    "time": c["candle_date_time_kst"],
                    "open": float(c["opening_price"]),
                    "high": float(c["high_price"]),
                    "low": float(c["low_price"]),
                    "close": float(c["trade_price"]),
                    "volume": float(c["candle_acc_trade_volume"])
                })
            
            df = pd.DataFrame(candles)
            return df
        except Exception as e:
            self.write_log(f"API Connection error (candles): {e}")
            return None

    def fetch_historical_candles(self, count=5760):
        """
        Fetches 'count' historical candles from Upbit by paging backwards.
        """
        url = f"{UPBIT_API_URL}/candles/minutes/{self.interval}"
        all_candles = []
        to_time = None
        
        self.write_log(f"Fetching {count} historical candles for optimization...")
        
        while len(all_candles) < count:
            params = {
                "market": self.market,
                "count": min(200, count - len(all_candles))
            }
            if to_time:
                params["to"] = to_time
                
            try:
                res = requests.get(url, params=params, timeout=10)
                data = res.json()
                if not isinstance(data, list) or len(data) == 0:
                    break
                
                all_candles.extend(data)
                
                # Paging cursor: oldest candle's KST time
                oldest_candle_kst = data[-1]["candle_date_time_kst"]
                to_time = oldest_candle_kst.replace("T", " ")
                time.sleep(0.05) # Polite API throttling
            except Exception as e:
                self.write_log(f"Error in historical fetch: {e}")
                time.sleep(1)
                
        if len(all_candles) == 0:
            return None
            
        all_candles.reverse()
        
        candles = []
        for c in all_candles:
            candles.append({
                "time": c["candle_date_time_kst"],
                "open": float(c["opening_price"]),
                "high": float(c["high_price"]),
                "low": float(c["low_price"]),
                "close": float(c["trade_price"]),
                "volume": float(c["candle_acc_trade_volume"])
            })
        return pd.DataFrame(candles)

    def fetch_volume_power(self):
        """
        Fetches the real-time Volume Power (체결강도) for self.market from Upbit.
        """
        url = f"{UPBIT_API_URL}/trades/ticks"
        params = {
            "market": self.market,
            "count": 100
        }
        try:
            res = requests.get(url, params=params, timeout=10)
            data = res.json()
            if not isinstance(data, list) or len(data) == 0:
                return 50.0
                
            ask_vol = 0.0 
            bid_vol = 0.0 
            
            for t in data:
                vol = float(t["trade_volume"])
                if t["ask_bid"] == "BID": 
                    bid_vol += vol
                else: 
                    ask_vol += vol
                    
            if ask_vol == 0:
                return 100.0 if bid_vol > 0 else 50.0
            
            vol_power = (bid_vol / ask_vol) * 100
            return vol_power
        except Exception as e:
            self.write_log(f"API Connection error (volume power): {e}")
            return 50.0

    def calculate_indicators(self, df):
        if df is None or len(df) < 100:
            return None, None
            
        df["ma_100"] = df["close"].rolling(window=100).mean()
        df["std_100"] = df["close"].rolling(window=100).std()
        df["z_score"] = (df["close"] - df["ma_100"]) / (df["std_100"] + 1e-9)
        
        current_z = df["z_score"].iloc[-1]
        return current_z, df

    # Sub-simulation over past data for WFO
    def simulate_sub_period(self, df_sub, z_thresh):
        """
        Runs WFO sub-period simulation.
        Since historical candles don't have volume power, we optimize solely on Z-Score.
        Hybrid order model: limit entry/grids/trailing (no slippage),
        market stop-loss (slippage applied).
        """
        signals = df_sub["z_score"].values < z_thresh
        closes = df_sub["close"].values
        highs = df_sub["high"].values
        lows = df_sub["low"].values

        fee_rate = 0.0005          # Upbit fee, applied per side
        slippage_per_side = 0.0005  # SL only (limit orders have no slippage)
        tp_activation_pct = self.tp_pct
        trail_pct = 0.2
        sl_pct = -2.0
        l3_cutoff_bars = self.l3_cutoff_sec // (15 * 60)  # 16 bars = 4h
        l3_bep_thresh_pct = self.l3_bep_thresh_pct

        in_trade = False
        level_1_price = 0.0
        level_2_filled = False
        level_3_filled = False
        trailing_active = False
        peak_price = 0.0
        balance = 500000.0
        l3_fill_idx = -1

        for i in range(len(df_sub)):
            if not in_trade:
                if signals[i]:
                    in_trade = True
                    level_1_price = closes[i]
                    level_2_filled = False
                    level_3_filled = False
                    trailing_active = False
                    peak_price = 0.0
                    l3_fill_idx = -1
            else:
                level_2_price = level_1_price * 0.990
                level_3_price = level_1_price * 0.980

                curr_low = lows[i]
                curr_high = highs[i]
                curr_close = closes[i]

                if not level_2_filled and curr_low <= level_2_price:
                    level_2_filled = True
                if not level_3_filled and curr_low <= level_3_price:
                    level_3_filled = True
                    l3_fill_idx = i

                if level_3_filled:
                    avg_price = (level_1_price * 0.3 + level_2_price * 0.3 + level_3_price * 0.4)
                    total_weight = 1.0
                elif level_2_filled:
                    avg_price = (level_1_price * 0.3 + level_2_price * 0.3) / 0.6
                    total_weight = 0.6
                else:
                    avg_price = level_1_price
                    total_weight = 0.3

                sl_price = avg_price * (1 + sl_pct/100.0)
                tp_activation_price = avg_price * (1 + tp_activation_pct/100.0)

                if not trailing_active:
                    if curr_high >= tp_activation_price:
                        trailing_active = True
                        peak_price = max(curr_high, tp_activation_price)
                else:
                    peak_price = max(peak_price, curr_high)

                exited = False
                exit_price = 0.0
                exit_is_market = False

                # L3 time cutoff (only after L3 fill, before trailing)
                if (level_3_filled and not trailing_active and l3_fill_idx >= 0
                        and (i - l3_fill_idx) >= l3_cutoff_bars):
                    bep_target = avg_price * (1 + l3_bep_thresh_pct / 100.0)
                    if curr_close >= bep_target:
                        exit_price = curr_close
                        exited = True
                        exit_is_market = True

                if not exited and trailing_active:
                    stop_price = peak_price * (1 - trail_pct/100.0)
                    activation_floor = avg_price * (1 + (tp_activation_pct - trail_pct)/100.0)
                    stop_price = max(stop_price, activation_floor)

                    if curr_low <= stop_price:
                        exit_price = stop_price
                        exited = True
                elif not exited:
                    if curr_low <= sl_price:
                        exit_price = sl_price
                        exited = True
                        exit_is_market = True
                        
                if exited:
                    # Hybrid: TP=limit (no slip), SL/L3-cutoff=market (slip applied)
                    if exit_is_market:
                        sell_price = exit_price * (1 - slippage_per_side)
                    else:
                        sell_price = exit_price
                    profit_pct = (sell_price - avg_price) / avg_price * 100
                    ret = (total_weight * profit_pct) / 100.0
                    fee = total_weight * fee_rate * 2.0   # round-trip fee
                    balance *= (1 + ret - fee)
                    in_trade = False
                    l3_fill_idx = -1
                    
        return (balance - 500000.0) / 500000.0 * 100

    def reoptimize_parameters(self):
        """
        Loads 60 days of historical data, runs simulation across candidate Z-Score thresholds,
        and overrides self.z_thresh with the highest yield one.
        """
        self.write_log(">>> Starting 자율 학습 (WFO) Parameter Optimization...")
        df_opt = self.fetch_historical_candles(5760)
        
        if df_opt is None or len(df_opt) < 100:
            self.write_log("Optimization failed: Not enough historical candles.")
            return
            
        # Prep indicators on training set
        df_opt["ma_100"] = df_opt["close"].rolling(window=100).mean()
        df_opt["std_100"] = df_opt["close"].rolling(window=100).std()
        df_opt["z_score"] = (df_opt["close"] - df_opt["ma_100"]) / (df_opt["std_100"] + 1e-9)
        df_opt = df_opt.dropna().reset_index(drop=True)
        
        z_candidates = [-1.0, -1.2, -1.5, -1.8, -2.0]
        best_return = -999.0
        best_z = self.z_thresh
        
        for z in z_candidates:
            ret = self.simulate_sub_period(df_opt, z)
            if ret > best_return:
                best_return = ret
                best_z = z
                
        self.z_thresh = best_z
        msg = f">>> Optimization Complete! Selected Z-Thresh: {self.z_thresh} | Simulated Yield: {best_return:+.2f}%"
        self.write_log(msg)
        send_telegram_message(f"⚙️ [{self.market}] WFO 자율 학습 완료\n- 최적 Z-Score: {self.z_thresh}\n- 백테스트 수익률: {best_return:+.2f}%")

    def execute_trade_cycle(self):
        df_candles = self.fetch_upbit_candles()
        z_score, df = self.calculate_indicators(df_candles)
        vol_power = self.fetch_volume_power()

        if z_score is None:
            self.write_log("Skipping trade cycle: Not enough candle data.")
            return None, 50.0, None

        current_price = df["close"].iloc[-1]

        # Hybrid order model:
        #   L1 entry / L2 / L3 / trailing TP -> LIMIT (no slippage)
        #   Stop-loss -> MARKET (slippage applied on exit price)
        # Pending limit entries time out after self.entry_timeout_sec.

        # 1) Try to fill a pending L1 limit entry
        if not self.in_trade and self.pending_entry_price > 0:
            now = time.time()
            if current_price <= self.pending_entry_price:
                # Limit fill
                fill_price = self.pending_entry_price
                self.in_trade = True
                self.allocated_capital = self.balance
                self.level_1_price = fill_price
                self.position_size = (self.allocated_capital * 0.30) / fill_price
                self.position_avg_price = fill_price
                self.level_2_filled = False
                self.level_3_filled = False
                self.trailing_active = False
                self.peak_price = 0.0
                self.pending_entry_price = 0.0
                self.pending_entry_expires = 0.0
                self.record_transaction("BUY", "L1", fill_price, self.position_size)
                msg_text = f"=== [ENTRY: LEVEL 1 LIMIT FILLED] Price: {fill_price:,.4f}원 | Alloc Capital: {self.allocated_capital:,.0f}원 | Active Z-Thresh: {self.z_thresh} ==="
                self.write_log(msg_text)
                send_telegram_message(f"🚀 [{self.market}] 1차 매수 체결 (Limit)\n- 진입가: {fill_price:,.4f}원\n- 할당 자금: {self.allocated_capital:,.0f}원\n- Z-Thresh: {self.z_thresh}")
            elif now >= self.pending_entry_expires:
                # Cancel timed-out limit
                self.write_log(f"=== [ENTRY: LEVEL 1 LIMIT CANCELLED] Limit {self.pending_entry_price:,.4f}원 not filled within {self.entry_timeout_sec//60}분 ===")
                self.pending_entry_price = 0.0
                self.pending_entry_expires = 0.0

        # 2) Place a new pending limit entry on fresh signal
        if not self.in_trade and self.pending_entry_price == 0:
            c1 = z_score < self.z_thresh
            c2 = vol_power > self.vol_power_thresh
            if c1:
                if c2:
                    self.pending_entry_price = current_price
                    self.pending_entry_expires = time.time() + self.entry_timeout_sec
                    msg_text = f"=== [ENTRY: LIMIT QUEUED] Price: {current_price:,.4f}원 | Z={z_score:.2f} | VP={vol_power:.1f}% | Expires: {self.entry_timeout_sec//60}분 ==="
                    self.write_log(msg_text)
                    send_telegram_message(f"📥 [{self.market}] 1차 매수 지정가 등록\n- 지정가: {current_price:,.4f}원\n- Z-Score: {z_score:.2f}\n- 체결강도: {vol_power:.1f}%")
                else:
                    msg_text = f"[NEAR-MISS] Z-Score met (Z={z_score:.2f} < {self.z_thresh}) but Volume Power too low (VP={vol_power:.1f}% <= {self.vol_power_thresh}%)"
                    self.write_log(msg_text)

        # 3) Manage open trade (grids + exits)
        if self.in_trade:
            level_2_price = self.level_1_price * 0.990
            level_3_price = self.level_1_price * 0.980

            if not self.level_2_filled and current_price <= level_2_price:
                buy_amount = self.allocated_capital * 0.30
                new_tokens = buy_amount / level_2_price
                self.position_avg_price = (self.position_size * self.position_avg_price + buy_amount) / (self.position_size + new_tokens)
                self.position_size += new_tokens
                self.level_2_filled = True
                self.record_transaction("BUY", "L2", level_2_price, new_tokens)
                msg_text = f"=== [GRID FILL: LEVEL 2 LIMIT] Price: {level_2_price:,.4f}원 | New Avg: {self.position_avg_price:,.4f}원 ==="
                self.write_log(msg_text)
                send_telegram_message(f"➕ [{self.market}] 2차 분할매수 체결 (Level 2)\n- 매수가: {level_2_price:,.4f}원\n- 새로운 평단가: {self.position_avg_price:,.4f}원")

            if not self.level_3_filled and current_price <= level_3_price:
                buy_amount = self.allocated_capital * 0.40
                new_tokens = buy_amount / level_3_price
                self.position_avg_price = (self.position_size * self.position_avg_price + buy_amount) / (self.position_size + new_tokens)
                self.position_size += new_tokens
                self.level_3_filled = True
                self.l3_fill_time = time.time()
                self.record_transaction("BUY", "L3", level_3_price, new_tokens)
                msg_text = f"=== [GRID FILL: LEVEL 3 LIMIT] Price: {level_3_price:,.4f}원 | New Avg: {self.position_avg_price:,.4f}원 ==="
                self.write_log(msg_text)
                send_telegram_message(f"🔥 [{self.market}] 3차 분할매수 체결 (Level 3 - 최종)\n- 매수가: {level_3_price:,.4f}원\n- 새로운 평단가: {self.position_avg_price:,.4f}원")

            sl_price = self.position_avg_price * (1 + self.sl_pct/100.0)
            tp_activation_price = self.position_avg_price * (1 + self.tp_pct/100.0)

            # L3 time cutoff: applies only after L3 fill, before trailing activation.
            # If price has recovered above (avg + bep_thresh) after the timer expires,
            # force-close at current_price (market) to cap worst-case drawdown.
            if (self.level_3_filled and not self.trailing_active
                    and self.l3_fill_time > 0
                    and time.time() - self.l3_fill_time >= self.l3_cutoff_sec):
                bep_target = self.position_avg_price * (1 + self.l3_bep_thresh_pct / 100.0)
                if current_price >= bep_target:
                    total_weight = 0.3 + 0.3 + 0.4   # full grid by definition
                    sell_price = current_price * (1 - self.slippage_per_side)
                    gross_value = self.position_size * sell_price
                    fee = self.allocated_capital * total_weight * self.fee_rate * 2.0
                    net_return_amount = gross_value - fee
                    self.balance += (self.allocated_capital * (1 - total_weight)) + net_return_amount - self.allocated_capital
                    profit_pct = (sell_price - self.position_avg_price) / self.position_avg_price * 100
                    elapsed_min = (time.time() - self.l3_fill_time) / 60
                    msg_text = f"=== [EXIT: L3 CUTOFF] Price: {sell_price:,.4f}원 | Profit: {profit_pct:+.2f}% | Held: {elapsed_min:.0f}분 | Balance: {self.balance:,.0f}원 ==="
                    self.write_log(msg_text)
                    send_telegram_message(f"⏱ [{self.market}] L3 시간 컷오프 청산 (BEP 회복)\n- 청산가: {sell_price:,.4f}원\n- 거래 수익률: {profit_pct:+.2f}%\n- L3 보유: {elapsed_min:.0f}분\n- 잔고: {self.balance:,.0f}원")
                    self.record_transaction("SELL", "EXIT_CUTOFF", sell_price, self.position_size, profit_pct)
                    self.reset_trade_state()
                    return current_price, z_score, vol_power, df

            if not self.trailing_active:
                if current_price >= tp_activation_price:
                    self.trailing_active = True
                    self.peak_price = max(current_price, tp_activation_price)
                    self.write_log(f"=== [TRAILING ACTIVATED] Peak: {self.peak_price:,.4f}원 ===")
            else:
                self.peak_price = max(self.peak_price, current_price)

            if self.trailing_active:
                stop_price = self.peak_price * (1 - 0.2/100.0)
                activation_floor = self.position_avg_price * (1 + (self.tp_pct - 0.2)/100.0)
                stop_price = max(stop_price, activation_floor)

                if current_price <= stop_price:
                    # Trailing TP exit -> LIMIT (no slippage)
                    total_weight = 0.3
                    if self.level_2_filled: total_weight += 0.3
                    if self.level_3_filled: total_weight += 0.4

                    sell_price = stop_price  # limit fill, no slip
                    gross_value = self.position_size * sell_price
                    fee = self.allocated_capital * total_weight * self.fee_rate * 2.0  # round-trip
                    net_return_amount = gross_value - fee

                    self.balance += (self.allocated_capital * (1 - total_weight)) + net_return_amount - self.allocated_capital

                    profit_pct = (sell_price - self.position_avg_price) / self.position_avg_price * 100
                    msg_text = f"=== [EXIT: TRAILING STOP LIMIT] Price: {sell_price:,.4f}원 | Profit: {profit_pct:+.2f}% | Balance: {self.balance:,.0f}원 ==="
                    self.write_log(msg_text)
                    send_telegram_message(f"💰 [{self.market}] 익절 청산 완료 (Trailing/Limit)\n- 청산가: {sell_price:,.4f}원\n- 거래 수익률: {profit_pct:+.2f}%\n- 잔고: {self.balance:,.0f}원")
                    self.record_transaction("SELL", "EXIT_TP", sell_price, self.position_size, profit_pct)
                    self.reset_trade_state()
            else:
                if current_price <= sl_price:
                    # Stop loss -> MARKET (slippage applied)
                    total_weight = 0.3
                    if self.level_2_filled: total_weight += 0.3
                    if self.level_3_filled: total_weight += 0.4

                    sell_price = sl_price * (1 - self.slippage_per_side)
                    gross_value = self.position_size * sell_price
                    fee = self.allocated_capital * total_weight * self.fee_rate * 2.0  # round-trip
                    net_return_amount = gross_value - fee

                    self.balance += (self.allocated_capital * (1 - total_weight)) + net_return_amount - self.allocated_capital

                    profit_pct = (sell_price - self.position_avg_price) / self.position_avg_price * 100
                    msg_text = f"=== [EXIT: STOP LOSS MARKET] Price: {sell_price:,.4f}원 | Profit: {profit_pct:+.2f}% | Balance: {self.balance:,.0f}원 ==="
                    self.write_log(msg_text)
                    send_telegram_message(f"🚨 [{self.market}] 손절 청산 완료 (Stop Loss/Market)\n- 청산가: {sell_price:,.4f}원\n- 거래 수익률: {profit_pct:+.2f}%\n- 잔고: {self.balance:,.0f}원")
                    self.record_transaction("SELL", "EXIT_SL", sell_price, self.position_size, profit_pct)
                    self.reset_trade_state()

        return current_price, z_score, vol_power, df

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
        self.pending_entry_price = 0.0
        self.pending_entry_expires = 0.0
        self.l3_fill_time = 0.0

    def get_recent_logs(self, limit=15):
        if not os.path.exists(self.log_file):
            return []
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                return [line.strip() for line in lines[-limit:] if line.strip()]
        except:
            return ["Error reading logs."]

def save_merged_dashboard_data(trader_xrp, state_xrp, trader_eth, state_eth):
    """
    Merges state dictionaries of XRP and ETH traders, updates equity history, and writes to dashboard_data.json.
    """
    import json
    import os
    
    # helper to format candles
    def format_candles(df_candles):
        candle_list = []
        if df_candles is not None:
            for _, r in df_candles.tail(100).iterrows():
                candle_list.append({
                    "time": r.get("time"),
                    "open": float(r.get("open", 0)),
                    "high": float(r.get("high", 0)),
                    "low": float(r.get("low", 0)),
                    "close": float(r.get("close", 0))
                })
        return candle_list

    # Unpack states
    cur_xrp_price, z_xrp, vp_xrp, df_xrp = state_xrp
    cur_eth_price, z_eth, vp_eth, df_eth = state_eth
    
    total_balance = trader_xrp.balance + trader_eth.balance
    
    # 2-year Daily Equity History Management
    # One data point per day (YYYY-MM-DD), updated every cycle with latest balance.
    # New point appended when date changes. Max 730 days (2 years).
    script_dir = os.path.dirname(os.path.abspath(__file__))
    history_path = os.path.join(script_dir, "equity_history.json")
    
    history_data = []
    if os.path.exists(history_path):
        try:
            with open(history_path, "r", encoding="utf-8") as hf:
                history_data = json.load(hf)
        except Exception as e:
            print(f"Error loading equity_history.json: {e}")
            history_data = []
            
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    
    if not history_data:
        history_data.append({
            "time": today_str,
            "value": total_balance
        })
    else:
        last_entry = history_data[-1]
        if last_entry.get("time") != today_str:
            # New day → append
            history_data.append({
                "time": today_str,
                "value": total_balance
            })
            if len(history_data) > 730:
                history_data = history_data[-730:]
        else:
            # Same day → overwrite with latest balance
            last_entry["value"] = total_balance
            
    try:
        with open(history_path, "w", encoding="utf-8") as hf:
            json.dump(history_data, hf, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving equity_history.json: {e}")

    # Load trade history
    tx_path = os.path.join(script_dir, "trade_history.json")
    trade_history = []
    if os.path.exists(tx_path):
        try:
            with open(tx_path, "r", encoding="utf-8") as f:
                trade_history = json.load(f)
        except Exception as e:
            print(f"Error loading trade_history.json: {e}")

    # Unified output data
    data = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": trader_xrp.mode,
        "total_balance": total_balance,
        "trade_history": trade_history,
        
        "xrp": {
            "market": trader_xrp.market,
            "balance": trader_xrp.balance,
            "in_trade": trader_xrp.in_trade,
            "position_size": trader_xrp.position_size,
            "position_avg_price": trader_xrp.position_avg_price,
            "current_price": cur_xrp_price,
            "z_score": z_xrp,
            "vol_power": vp_xrp,
            "allocated_capital": trader_xrp.allocated_capital,
            "level_2_filled": trader_xrp.level_2_filled,
            "level_3_filled": trader_xrp.level_3_filled,
            "trailing_active": trader_xrp.trailing_active,
            "peak_price": trader_xrp.peak_price,
            "z_thresh": trader_xrp.z_thresh,
            "vol_power_thresh": trader_xrp.vol_power_thresh,
            "pending_entry_price": trader_xrp.pending_entry_price,
            "pending_entry_expires": trader_xrp.pending_entry_expires,
            "l3_fill_time": trader_xrp.l3_fill_time,
            "tp_pct": trader_xrp.tp_pct,
            "candles": format_candles(df_xrp)
        },
        "eth": {
            "market": trader_eth.market,
            "balance": trader_eth.balance,
            "in_trade": trader_eth.in_trade,
            "position_size": trader_eth.position_size,
            "position_avg_price": trader_eth.position_avg_price,
            "current_price": cur_eth_price,
            "z_score": z_eth,
            "vol_power": vp_eth,
            "allocated_capital": trader_eth.allocated_capital,
            "level_2_filled": trader_eth.level_2_filled,
            "level_3_filled": trader_eth.level_3_filled,
            "trailing_active": trader_eth.trailing_active,
            "peak_price": trader_eth.peak_price,
            "z_thresh": trader_eth.z_thresh,
            "vol_power_thresh": trader_eth.vol_power_thresh,
            "pending_entry_price": trader_eth.pending_entry_price,
            "pending_entry_expires": trader_eth.pending_entry_expires,
            "l3_fill_time": trader_eth.l3_fill_time,
            "tp_pct": trader_eth.tp_pct,
            "candles": format_candles(df_eth)
        },
        "equity_history": history_data,
        "logs": trader_xrp.get_recent_logs(15)
    }
    
    try:
        with open("dashboard_data.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving merged dashboard data: {e}")

    # Upload to Cloudflare Workers KV
    cf_url = "https://upbit-portfolio-worker.nijin39.workers.dev/update"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {CF_WORKER_TOKEN}"
    }
    try:
        res = requests.post(cf_url, json=data, headers=headers, timeout=5)
        if res.status_code != 200:
            print(f"Cloudflare upload failed. Status: {res.status_code}, Response: {res.text}")
    except Exception as e:
        print(f"Cloudflare connection error: {e}")

if __name__ == "__main__":
    # Initialize both traders with 50% of the total 1,000,000 KRW capital
    trader_xrp = UpbitAutoTrader(market="KRW-XRP", mode="paper", initial_balance=500000.0)
    trader_eth = UpbitAutoTrader(market="KRW-ETH", mode="paper", initial_balance=500000.0)
    
    print("\nRunning Upbit real-time 50:50 Portfolio loop (Mock trading mode)...")
    print("Initializing WFO Self-Learning Engine...")
    
    # 1. Immediate Initial WFO Run on Startup
    trader_xrp.reoptimize_parameters()
    trader_xrp.last_opt_time = time.time()
    
    trader_eth.reoptimize_parameters()
    trader_eth.last_opt_time = time.time()
    
    print("Self-Learning Engine Initialized. Entering real-time monitoring loop...")
    print("XRP + ETH parallel scanning active. Press Ctrl+C to exit.")
    send_telegram_message("🔔 [Upbit Portfolio Bot] XRP & ETH 실시간 자율학습형 포트폴리오 트레이더 가동을 시작했습니다.")
    
    cycle_count = 0
    try:
        while True:
            # 2. Check if 7 days (604,800 seconds) have elapsed for re-optimization
            if time.time() - trader_xrp.last_opt_time >= 7 * 24 * 3600:
                print("\n>>> [SCHEDULED] Running Weekly WFO Self-Learning Parameter Optimization...")
                trader_xrp.reoptimize_parameters()
                trader_eth.reoptimize_parameters()
                trader_xrp.last_opt_time = time.time()
                trader_eth.last_opt_time = time.time()
                
            # 3. Execute XRP Cycle
            state_xrp = trader_xrp.execute_trade_cycle()
            
            # 4. Execute ETH Cycle
            state_eth = trader_eth.execute_trade_cycle()
            
            # 5. Save combined dashboard data (Every 10th cycle = 5 minutes)
            if cycle_count % 10 == 0:
                if state_xrp and state_eth and state_xrp[0] is not None and state_eth[0] is not None:
                    save_merged_dashboard_data(trader_xrp, state_xrp, trader_eth, state_eth)
                
            cycle_count += 1
            time.sleep(30) # Poll every 30 seconds
    except KeyboardInterrupt:
        print("\nUpbit Portfolio Bot stopped by user.")
