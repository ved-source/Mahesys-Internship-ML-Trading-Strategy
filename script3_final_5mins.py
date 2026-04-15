#!/usr/bin/env python3
"""
SCRIPT 3: Paper Trading via Binance Public WebSocket (NO API KEY NEEDED)
Install: pip install websocket-client numpy pandas scikit-learn xgboost joblib

Persistence:
  - paper_trade_log.csv     : every closed trade with full ML + PnL details
  - candle_log.csv          : every closed 5m candle (bar buffer state)
  - signal_log.csv          : every Renko+ML signal event (fired or skipped)
  - open_positions.json     : open positions saved after every update
  - buffer_state.json       : last N bars saved so warmup resumes on restart
  All files are appended/updated continuously — not just on Ctrl+C.
  On restart, open positions and buffer are restored automatically.
"""

import json
import time
import threading
import warnings
import joblib
import numpy as np
import pandas as pd
import websocket
from pathlib import Path
from collections import deque
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
WS_URL          = "wss://stream.binance.com:9443/ws/btcusdt@kline_5m"

BASE_DIR        = Path(r"C:\Users\saive\OneDrive\Desktop\Desktop\all folders\self-projects\placement\ML-trading")
MODEL_DIR       = BASE_DIR / "MODELS"
LOGS_DIR        = BASE_DIR / "PAPER_TRADE_LOGS_5M"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH      = MODEL_DIR / "final_renko_ml_model.pkl"
SCALER_PATH     = MODEL_DIR / "scaler.pkl"
FEAT_PATH       = MODEL_DIR / "feature_cols.pkl"

# Log file paths
TRADE_LOG_CSV   = LOGS_DIR / "paper_trade_log.csv"
CANDLE_LOG_CSV  = LOGS_DIR / "candle_log.csv"
SIGNAL_LOG_CSV  = LOGS_DIR / "signal_log.csv"
OPEN_POS_JSON   = LOGS_DIR / "open_positions.json"
BUFFER_JSON     = LOGS_DIR / "buffer_state.json"
CAPITAL_JSON    = LOGS_DIR / "capital_state.json"

# Trading params
INITIAL_CAPITAL   = 10_000.0
RISK_PER_TRADE    = 0.01
SL_ATR_MULT       = 1.5
RR_RATIO          = 2.0
ML_CONF_THRESHOLD = 0.60
MAX_OPEN_TRADES   = 2

ATR_PERIOD        = 14
ATR_MULTIPLIER    = 1.5
EMA_SHORT         = 9
EMA_LONG          = 21
BUFFER_SIZE       = 300

LABEL_NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}

# ─────────────────────────────────────────────────────────────
# PERSISTENCE HELPERS
# ─────────────────────────────────────────────────────────────

def append_csv(path, row_dict):
    df = pd.DataFrame([row_dict])
    df.to_csv(path, mode="a", header=not path.exists(), index=False)

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)

def load_json(path, default):
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return default
    return default

# ─────────────────────────────────────────────────────────────
# RENKO FUNCTIONS
# ─────────────────────────────────────────────────────────────

def compute_atr(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def build_renko_bricks(df, brick_size):
    prices     = df["close"].values
    timestamps = df["timestamp"].values
    volumes    = df["volume"].values
    atrs       = df["atr"].values

    bricks       = []
    current_base = np.floor(prices[0] / brick_size) * brick_size
    direction    = None
    vol_accum    = 0.0

    for i, price in enumerate(prices):
        vol_accum += volumes[i]
        curr_atr   = atrs[i]

        if direction is None or direction == 1:
            if price >= current_base + brick_size:
                new_base = current_base + brick_size
                bricks.append({"timestamp": timestamps[i], "open": current_base,
                                "close": new_base, "direction": 1,
                                "brick_size": brick_size, "volume": vol_accum, "atr": curr_atr})
                current_base = new_base; direction = 1; vol_accum = 0.0
            elif direction == 1 and price <= current_base - brick_size:
                new_base = current_base - brick_size
                bricks.append({"timestamp": timestamps[i], "open": current_base,
                                "close": new_base, "direction": -1,
                                "brick_size": brick_size, "volume": vol_accum, "atr": curr_atr})
                current_base = new_base; direction = -1; vol_accum = 0.0

        if direction == -1:
            if price <= current_base - brick_size:
                new_base = current_base - brick_size
                bricks.append({"timestamp": timestamps[i], "open": current_base,
                                "close": new_base, "direction": -1,
                                "brick_size": brick_size, "volume": vol_accum, "atr": curr_atr})
                current_base = new_base; vol_accum = 0.0
            elif price >= current_base + brick_size:
                new_base = current_base + brick_size
                bricks.append({"timestamp": timestamps[i], "open": current_base,
                                "close": new_base, "direction": 1,
                                "brick_size": brick_size, "volume": vol_accum, "atr": curr_atr})
                current_base = new_base; direction = 1; vol_accum = 0.0

    return bricks

def add_renko_signals(rdf):
    rdf = rdf.copy()
    rdf["ema_short"] = rdf["close"].ewm(span=EMA_SHORT, adjust=False).mean()
    rdf["ema_long"]  = rdf["close"].ewm(span=EMA_LONG,  adjust=False).mean()

    prev_dir   = rdf["direction"].shift(1)
    flip_up    = (prev_dir == -1) & (rdf["direction"] ==  1)
    flip_down  = (prev_dir ==  1) & (rdf["direction"] == -1)
    trend_up   = rdf["close"] > rdf["ema_short"]
    trend_down = rdf["close"] < rdf["ema_short"]

    rdf["signal"] = 0
    rdf.loc[flip_up   & trend_up,   "signal"] =  1
    rdf.loc[flip_down & trend_down, "signal"] = -1

    rdf["stop_loss"] = np.where(
        rdf["signal"] ==  1, rdf["close"] - SL_ATR_MULT * rdf["atr"],
        np.where(rdf["signal"] == -1, rdf["close"] + SL_ATR_MULT * rdf["atr"], np.nan))
    rdf["take_profit"] = np.where(
        rdf["signal"] ==  1, rdf["close"] + SL_ATR_MULT * rdf["atr"] * RR_RATIO,
        np.where(rdf["signal"] == -1, rdf["close"] - SL_ATR_MULT * rdf["atr"] * RR_RATIO, np.nan))
    return rdf

# ─────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

def engineer_row_features(rdf, feat_cols):
    rdf = rdf.copy()

    ts = pd.to_datetime(rdf["timestamp"].iloc[-1], errors="coerce")
    rdf["hour"]        = ts.hour      if pd.notna(ts) else 12
    rdf["day_of_week"] = ts.dayofweek if pd.notna(ts) else 0

    rdf["ema_diff"]          = rdf["ema_short"] - rdf["ema_long"]
    rdf["ema_diff_pct"]      = rdf["ema_diff"] / (rdf["ema_long"].abs() + 1e-9) * 100
    rdf["renko_momentum_3"]  = rdf["direction"].rolling(3, min_periods=1).sum()
    rdf["renko_momentum_5"]  = rdf["direction"].rolling(5, min_periods=1).sum()
    rdf["close_lag1"]        = rdf["close"].shift(1)
    rdf["close_lag2"]        = rdf["close"].shift(2)
    rdf["close_lag3"]        = rdf["close"].shift(3)
    rdf["close_roll_mean_5"] = rdf["close"].rolling(5, min_periods=1).mean()
    rdf["close_roll_std_5"]  = rdf["close"].rolling(5, min_periods=1).std().fillna(0)
    rdf["price_range_pct"]   = (rdf["close"] - rdf["open"]).abs() / (rdf["open"] + 1e-9) * 100
    rdf["volume_ma5"]        = rdf["volume"].rolling(5, min_periods=1).mean()
    rdf["vol_ratio"]         = rdf["volume"] / (rdf["volume_ma5"] + 1e-9)

    last = rdf.iloc[-1]
    vec  = []
    for c in feat_cols:
        try:
            vec.append(float(last[c]))
        except Exception:
            vec.append(0.0)
    return np.array(vec, dtype=float)

# ─────────────────────────────────────────────────────────────
# PAPER TRADE JOURNAL
# ─────────────────────────────────────────────────────────────

class PaperTradeJournal:
    def __init__(self):
        state = load_json(CAPITAL_JSON, {"capital": INITIAL_CAPITAL, "trade_counter": 0})
        self.capital       = state["capital"]
        self.initial       = INITIAL_CAPITAL
        self.trade_counter = state["trade_counter"]
        self.equity_curve  = [self.capital]

        raw_open = load_json(OPEN_POS_JSON, [])
        self.open_pos = raw_open

        self.closed_trades = []

        if self.trade_counter > 0:
            print("[RESTORED] Previous session found.")
            print("           Capital: $" + str(round(self.capital, 2)))
            print("           Trade counter: " + str(self.trade_counter))
            print("           Open positions restored: " + str(len(self.open_pos)))

    def _save_capital(self):
        save_json(CAPITAL_JSON, {
            "capital":       self.capital,
            "trade_counter": self.trade_counter,
        })

    def _save_open_positions(self):
        save_json(OPEN_POS_JSON, self.open_pos)

    def open_trade(self, side, entry, sl, tp, ml_label, ml_conf,
                   ml_hold_prob, ml_buy_prob, ml_sell_prob, renko_signal, ts):
        sl_dist = abs(entry - sl)
        qty     = (RISK_PER_TRADE * self.capital) / (sl_dist + 1e-9)
        self.trade_counter += 1

        trade = {
            "id":            self.trade_counter,
            "side":          side,
            "entry":         round(entry, 4),
            "sl":            round(sl, 4),
            "tp":            round(tp, 4),
            "qty":           round(qty, 6),
            "ml_label":      ml_label,
            "ml_conf":       round(ml_conf, 4),
            "ml_hold_prob":  round(ml_hold_prob, 4),
            "ml_buy_prob":   round(ml_buy_prob, 4),
            "ml_sell_prob":  round(ml_sell_prob, 4),
            "renko_signal":  renko_signal,
            "open_ts":       ts,
            "status":        "OPEN",
            "exit":          None,
            "pnl":           None,
            "close_ts":      None,
        }
        self.open_pos.append(trade)

        self._save_open_positions()
        self._save_capital()

        append_csv(SIGNAL_LOG_CSV, {
            "event":         "TRADE_OPENED",
            "timestamp":     ts,
            "side":          side,
            "entry":         round(entry, 4),
            "sl":            round(sl, 4),
            "tp":            round(tp, 4),
            "qty":           round(qty, 6),
            "ml_label":      ml_label,
            "ml_conf":       round(ml_conf, 4),
            "ml_hold_prob":  round(ml_hold_prob, 4),
            "ml_buy_prob":   round(ml_buy_prob, 4),
            "ml_sell_prob":  round(ml_sell_prob, 4),
            "renko_signal":  renko_signal,
            "trade_id":      self.trade_counter,
            "capital_before": round(self.capital, 2),
        })

        sep = "-" * 70
        print("")
        print(sep)
        print("  NEW PAPER TRADE #" + str(trade["id"]))
        print("  Side         : " + side)
        print("  Entry Price  : " + str(round(entry, 4)))
        print("  Stop-Loss    : " + str(round(sl, 4)))
        print("  Take-Profit  : " + str(round(tp, 4)) + "  (RR=" + str(RR_RATIO) + ":1)")
        print("  Quantity     : " + str(round(qty, 6)) + " BTC")
        print("  ML Label     : " + ml_label + "  (" + str(round(ml_conf*100, 1)) + "% confidence)")
        print("  ML Probs     : HOLD=" + str(round(ml_hold_prob*100,1)) + "%  BUY=" + str(round(ml_buy_prob*100,1)) + "%  SELL=" + str(round(ml_sell_prob*100,1)) + "%")
        print("  Time         : " + ts)
        print(sep)

    def update_positions(self, live_price, ts):
        remaining = []
        for t in self.open_pos:
            hit_sl = False
            hit_tp = False
            if t["side"] == "BUY":
                if live_price <= t["sl"]:   hit_sl = True
                elif live_price >= t["tp"]: hit_tp = True
            else:
                if live_price >= t["sl"]:   hit_sl = True
                elif live_price <= t["tp"]: hit_tp = True

            if hit_sl or hit_tp:
                exit_price = t["sl"] if hit_sl else t["tp"]
                if t["side"] == "BUY":
                    pnl = (exit_price - t["entry"]) * t["qty"]
                else:
                    pnl = (t["entry"] - exit_price) * t["qty"]

                self.capital += pnl
                self.equity_curve.append(self.capital)

                t["exit"]     = round(exit_price, 4)
                t["pnl"]      = round(pnl, 4)
                t["close_ts"] = ts
                t["status"]   = "TP" if hit_tp else "SL"
                self.closed_trades.append(t)

                append_csv(TRADE_LOG_CSV, t)

                append_csv(SIGNAL_LOG_CSV, {
                    "event":          "TRADE_CLOSED",
                    "timestamp":      ts,
                    "side":           t["side"],
                    "entry":          t["entry"],
                    "exit":           t["exit"],
                    "sl":             t["sl"],
                    "tp":             t["tp"],
                    "qty":            t["qty"],
                    "pnl":            t["pnl"],
                    "status":         t["status"],
                    "ml_label":       t["ml_label"],
                    "ml_conf":        t["ml_conf"],
                    "ml_hold_prob":   t["ml_hold_prob"],
                    "ml_buy_prob":    t["ml_buy_prob"],
                    "ml_sell_prob":   t["ml_sell_prob"],
                    "renko_signal":   t["renko_signal"],
                    "trade_id":       t["id"],
                    "capital_after":  round(self.capital, 2),
                })

                outcome = "WIN" if pnl > 0 else "LOSS"
                print("  [CLOSED #" + str(t["id"]) + "]  " + t["side"] +
                      "  Exit=" + str(exit_price) +
                      "  PnL=$" + str(round(pnl, 2)) +
                      "  (" + t["status"] + ")  " + outcome +
                      "  Capital=$" + str(round(self.capital, 2)))
            else:
                remaining.append(t)

        self.open_pos = remaining
        self._save_open_positions()
        self._save_capital()

    def compute_stats(self):
        if TRADE_LOG_CSV.exists():
            try:
                all_trades = pd.read_csv(TRADE_LOG_CSV)
                pnls = all_trades["pnl"].dropna().values
            except Exception:
                pnls = np.array([t["pnl"] for t in self.closed_trades])
        else:
            pnls = np.array([t["pnl"] for t in self.closed_trades])

        n = len(pnls)
        if n == 0:
            return {}

        wins   = pnls[pnls > 0]
        losses = pnls[pnls <= 0]

        win_rate     = len(wins) / n
        gross_profit = float(wins.sum())        if len(wins)   else 0.0
        gross_loss   = float(abs(losses.sum())) if len(losses) else 1e-9
        pf           = gross_profit / gross_loss
        net_pnl      = float(pnls.sum())
        avg_win      = float(wins.mean())        if len(wins)   else 0.0
        avg_loss     = float(losses.mean())      if len(losses) else 0.0
        expectancy   = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        eq     = np.array(self.equity_curve)
        peak   = np.maximum.accumulate(eq)
        dd     = (peak - eq) / (peak + 1e-9)
        max_dd = float(dd.max())

        if len(eq) > 2:
            rets   = np.diff(eq) / (eq[:-1] + 1e-9)
            sharpe = (rets.mean() / (rets.std() + 1e-9)) * np.sqrt(252 * 1440)
        else:
            sharpe = 0.0

        return {
            "total_closed_trades" : n,
            "open_positions"      : len(self.open_pos),
            "win_rate_pct"        : round(win_rate * 100, 2),
            "profit_factor"       : round(pf, 4),
            "net_pnl_usd"         : round(net_pnl, 2),
            "gross_profit_usd"    : round(gross_profit, 2),
            "gross_loss_usd"      : round(float(-abs(losses.sum())), 2) if len(losses) else 0.0,
            "avg_win_usd"         : round(avg_win, 2),
            "avg_loss_usd"        : round(avg_loss, 2),
            "expectancy_usd"      : round(expectancy, 4),
            "max_drawdown_pct"    : round(max_dd * 100, 2),
            "sharpe_ratio"        : round(float(sharpe), 4),
            "initial_capital"     : round(self.initial, 2),
            "current_capital"     : round(self.capital, 2),
            "total_return_pct"    : round((self.capital - self.initial) / self.initial * 100, 4),
        }

    def print_summary(self):
        stats = self.compute_stats()
        if not stats:
            print("  No closed trades yet.")
            return
        sep = "=" * 60
        print("")
        print(sep)
        print("  PAPER TRADING LIVE STATISTICS")
        print(sep)
        for k, v in stats.items():
            label = k.replace("_", " ").upper()
            if "USD" in label or "CAPITAL" in label:
                print("  " + label.ljust(30) + ": $" + str(v))
            else:
                print("  " + label.ljust(30) + ": " + str(v))
        print(sep)

# ─────────────────────────────────────────────────────────────
# MAIN TRADER CLASS
# ─────────────────────────────────────────────────────────────

class RenkoMLPaperTrader:
    def __init__(self):
        if not MODEL_PATH.exists():
            print("[ERROR] Model not found at: " + str(MODEL_PATH))
            print("        Run script2_ml_training.py first.")
            raise SystemExit(1)

        self.model     = joblib.load(MODEL_PATH)
        self.scaler    = joblib.load(SCALER_PATH)
        self.feat_cols = joblib.load(FEAT_PATH)
        print("[OK] ML model loaded: " + str(MODEL_PATH))

        self.buffer = deque(maxlen=BUFFER_SIZE)
        saved_buffer = load_json(BUFFER_JSON, [])
        if saved_buffer:
            for bar in saved_buffer:
                bar["timestamp"] = pd.Timestamp(bar["timestamp"])
                self.buffer.append(bar)
            print("[RESTORED] Buffer: " + str(len(self.buffer)) + " bars loaded from previous session.")

        self.journal               = PaperTradeJournal()
        self.prev_n_bricks         = 0
        self.last_signal_brick_idx = -1
        self._lock                 = threading.Lock()

    def _save_buffer(self):
        bars = []
        for bar in self.buffer:
            b = dict(bar)
            b["timestamp"] = str(b["timestamp"])
            bars.append(b)
        save_json(BUFFER_JSON, bars)

    def on_open(self, ws):
        print("[WS] Connected to Binance public stream.")
        if len(self.buffer) >= 30:
            print("[WS] Buffer already warmed up (" + str(len(self.buffer)) + " bars). Starting immediately.")
        else:
            print("[WS] Waiting for closed 5m candles...")
        print("")

    def on_error(self, ws, error):
        print("[WS ERROR] " + str(error))

    def on_close(self, ws, close_status_code, close_msg):
        print("[WS] Connection closed.")
        self._save_buffer()
        self.journal.print_summary()
        print("[SAVED] All logs are in: " + str(LOGS_DIR))

    def on_message(self, ws, message):
        try:
            data  = json.loads(message)
            kline = data["k"]

            if not kline["x"]:
                return

            bar = {
                "timestamp": pd.Timestamp(int(kline["T"]), unit="ms", tz="UTC"),
                "open":      float(kline["o"]),
                "high":      float(kline["h"]),
                "low":       float(kline["l"]),
                "close":     float(kline["c"]),
                "volume":    float(kline["v"]),
            }

            with self._lock:
                self.buffer.append(bar)
                self._save_buffer()

                append_csv(CANDLE_LOG_CSV, {
                    "timestamp": str(bar["timestamp"]),
                    "open":      bar["open"],
                    "high":      bar["high"],
                    "low":       bar["low"],
                    "close":     bar["close"],
                    "volume":    bar["volume"],
                })

                self._process()

        except Exception as e:
            print("[MSG ERROR] " + str(e))

    def _process(self):
        if len(self.buffer) < 30:
            print("  Warming up... (" + str(len(self.buffer)) + "/" + str(BUFFER_SIZE) + " bars)")
            return

        df = pd.DataFrame(list(self.buffer))
        df["atr"] = compute_atr(df, ATR_PERIOD)

        median_atr = float(df["atr"].median())
        brick_size = median_atr * ATR_MULTIPLIER
        if brick_size <= 0:
            return

        live_price = float(df["close"].iloc[-1])
        ts_str     = str(df["timestamp"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S UTC"))

        self.journal.update_positions(live_price, ts_str)

        bricks = build_renko_bricks(df, brick_size)
        if len(bricks) < 5:
            return

        rdf        = pd.DataFrame(bricks)
        rdf        = add_renko_signals(rdf)
        n_bricks   = len(rdf)
        new_bricks = n_bricks - self.prev_n_bricks

        last_dir = int(rdf["direction"].iloc[-1])
        dir_str  = "UP" if last_dir == 1 else "DOWN"

        print("  [" + ts_str + "]  Price=" + str(round(live_price, 2)) +
              "  Bricks=" + str(n_bricks) + "(+" + str(max(new_bricks, 0)) + ")" +
              "  LastBrick=" + dir_str +
              "  Capital=$" + str(round(self.journal.capital, 2)) +
              "  Open=" + str(len(self.journal.open_pos)))

        if new_bricks <= 0:
            self.prev_n_bricks = n_bricks
            return

        self.prev_n_bricks = n_bricks

        last          = rdf.iloc[-1]
        renko_signal  = int(last["signal"])
        current_brick = n_bricks - 1

        if renko_signal == 0:
            return

        if current_brick == self.last_signal_brick_idx:
            return

        side_str = "BUY" if renko_signal == 1 else "SELL"
        print("  >> RENKO SIGNAL: " + side_str + " (brick #" + str(current_brick) + ")")

        feat_vec    = engineer_row_features(rdf, self.feat_cols)
        feat_vec    = np.nan_to_num(feat_vec.reshape(1, -1), nan=0.0, posinf=0.0, neginf=0.0)
        feat_scaled = self.scaler.transform(feat_vec)

        proba     = self.model.predict_proba(feat_scaled)[0]
        pred_idx  = int(np.argmax(proba))
        pred_conf = float(proba[pred_idx])
        pred_name = LABEL_NAMES[pred_idx]

        hold_p = float(proba[0])
        buy_p  = float(proba[1])
        sell_p = float(proba[2])

        print("  >> ML: HOLD=" + str(round(hold_p*100,1)) + "%  " +
              "BUY=" + str(round(buy_p*100,1)) + "%  " +
              "SELL=" + str(round(sell_p*100,1)) + "%  " +
              "=> " + pred_name + " (" + str(round(pred_conf*100,1)) + "%)")

        skip_reason = None
        renko_to_ml = {1: 1, -1: 2}
        if pred_idx != renko_to_ml[renko_signal]:
            skip_reason = "ML_DISAGREES"
        elif pred_conf < ML_CONF_THRESHOLD:
            skip_reason = "LOW_CONFIDENCE"
        elif len(self.journal.open_pos) >= MAX_OPEN_TRADES:
            skip_reason = "MAX_POSITIONS_REACHED"
        else:
            sl = float(last["stop_loss"])
            tp = float(last["take_profit"])
            if np.isnan(sl) or np.isnan(tp):
                skip_reason = "INVALID_SL_TP"

        append_csv(SIGNAL_LOG_CSV, {
            "event":          "SIGNAL_FIRED" if skip_reason is None else "SIGNAL_SKIPPED",
            "timestamp":      ts_str,
            "renko_signal":   renko_signal,
            "side":           side_str,
            "live_price":     round(live_price, 4),
            "ml_label":       pred_name,
            "ml_conf":        round(pred_conf, 4),
            "ml_hold_prob":   round(hold_p, 4),
            "ml_buy_prob":    round(buy_p, 4),
            "ml_sell_prob":   round(sell_p, 4),
            "skip_reason":    skip_reason if skip_reason else "NONE",
            "capital":        round(self.journal.capital, 2),
            "open_positions": len(self.journal.open_pos),
        })

        if skip_reason:
            print("  >> SKIP: " + skip_reason)
            return

        sl = float(last["stop_loss"])
        tp = float(last["take_profit"])

        self.journal.open_trade(
            side=side_str, entry=live_price, sl=sl, tp=tp,
            ml_label=pred_name, ml_conf=pred_conf,
            ml_hold_prob=hold_p, ml_buy_prob=buy_p, ml_sell_prob=sell_p,
            renko_signal=renko_signal, ts=ts_str
        )
        self.last_signal_brick_idx = current_brick

        n_closed = len(self.journal.closed_trades)
        if n_closed > 0 and n_closed % 5 == 0:
            self.journal.print_summary()

    def run(self):
        print("=" * 70)
        print("  SCRIPT 3: RENKO + ML PAPER TRADING (5m)")
        print("  Binance Public WebSocket — No API Key Required")
        print("=" * 70)
        print("  Stream   : " + WS_URL)
        print("  Logs Dir : " + str(LOGS_DIR))
        print("  Capital  : $" + str(INITIAL_CAPITAL) + "  (restored from state if exists)")
        print("  Files    : paper_trade_log.csv | candle_log.csv | signal_log.csv")
        print("             open_positions.json | buffer_state.json | capital_state.json")
        print("")
        print("  FLOW: Closed candle -> save candle -> Renko -> [signal only] -> ML -> paper trade")
        print("  All data saved after EVERY candle. Restart continues from last state.")
        print("  Press Ctrl+C to stop.")
        print("=" * 70)
        print("")

        websocket.enableTrace(False)
        ws_app = websocket.WebSocketApp(
            WS_URL,
            on_open    = self.on_open,
            on_message = self.on_message,
            on_error   = self.on_error,
            on_close   = self.on_close,
        )

        while True:
            try:
                ws_app.run_forever(ping_interval=30, ping_timeout=10)
                print("[WS] Disconnected. Reconnecting in 5 seconds...")
                time.sleep(5)
            except KeyboardInterrupt:
                print("")
                print("[STOPPED] Ctrl+C received.")
                self._save_buffer()
                self.journal.print_summary()
                print("[SAVED] All logs in: " + str(LOGS_DIR))
                break

# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    trader = RenkoMLPaperTrader()
    trader.run()