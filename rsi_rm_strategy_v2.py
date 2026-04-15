from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# CONFIG
# ============================================================

DATASETS_ROOT = Path(r"C:\Users\saive\OneDrive\Desktop\Desktop\all folders\self-projects\placement\ML-trading\DATASETS")
OUTPUT_ROOT = DATASETS_ROOT / "RSI_RM"

RSI_PERIOD   = 14
BUY_RSI      = 30    # RSI must cross ABOVE this (not just be below)
SELL_RSI     = 70
EMA_FAST     = 50
EMA_SLOW     = 200
FEE_RATE     = 0.0004   # 0.04% per side
ATR_PERIOD   = 14
ATR_MIN_MULT = 0.3      # skip entry if ATR < this * median ATR (low-volatility filter)

# Timeframe-specific SL/TP and cooldown candles
TF_PARAMS = {
    "scalp": {"sl": 0.003,  "tp": 0.006,  "cooldown": 5},
    "5min":  {"sl": 0.005,  "tp": 0.010,  "cooldown": 3},
    "15min": {"sl": 0.007,  "tp": 0.015,  "cooldown": 3},
    "30min": {"sl": 0.010,  "tp": 0.020,  "cooldown": 2},
    "60min": {"sl": 0.015,  "tp": 0.030,  "cooldown": 2},
    "swing": {"sl": 0.050,  "tp": 0.100,  "cooldown": 1},
}

# Annualization factors per timeframe for Sharpe
ANNUALIZATION = {
    "scalp": 525600,
    "5min":  105120,
    "15min": 35040,
    "30min": 17520,
    "60min": 8760,
    "swing": 365,
}

CLOSE_OPEN_TRADE_AT_EOD = True
SAVE_PROCESSED_SIGNALS  = True
SAVE_TRADE_LOG          = True
SAVE_SUMMARY            = True

TIMEFRAMES = {
    "scalp": "1min",
    "5min":  "5min",
    "15min": "15min",
    "30min": "30min",
    "60min": "60min",
    "swing": "1D",
}

# ============================================================
# DATA LOADING (unchanged from v2 -- fully robust)
# ============================================================

def safe_mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def get_bucket_name(tf_name: str) -> str:
    if tf_name == "scalp":  return "scalp"
    if tf_name == "swing":  return "swing"
    return "intraday"

def first_numeric_value(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return None if s.empty else float(s.iloc[0])

def detect_datetime_unit(series: pd.Series) -> str:
    v = first_numeric_value(series)
    if v is None:       return "unknown"
    if v >= 1e15:       return "us"
    if v >= 1e12:       return "ms"
    if v >= 1e9:        return "s"
    return "unknown"

def parse_datetime_column(series: pd.Series) -> pd.Series:
    unit    = detect_datetime_unit(series)
    numeric = pd.to_numeric(series, errors="coerce")
    if unit in {"us", "ms", "s"}:
        dt = pd.to_datetime(numeric, unit=unit, errors="coerce", utc=True)
    else:
        dt = pd.to_datetime(series, errors="coerce", utc=True)
    return dt.dt.tz_convert(None)

def has_header(csv_path: Path) -> bool:
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline().strip()
    if not first_line:
        return False
    tokens      = [x.strip() for x in first_line.split(",")]
    numeric_hits = 0
    for token in tokens[:6]:
        try:
            float(token)
            numeric_hits += 1
        except:
            pass
    return numeric_hits < min(4, len(tokens))

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().lower() for c in df.columns]
    rename_map = {
        "open time": "open_time", "open_time": "open_time",
        "timestamp": "open_time", "date": "open_time", "datetime": "open_time",
        "open": "open", "high": "high", "low": "low",
        "close": "close", "volume": "volume"
    }
    return df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

def read_ohlcv_csv(csv_path: Path) -> pd.DataFrame:
    if has_header(csv_path):
        df = pd.read_csv(csv_path)
        df = normalize_columns(df)
    else:
        df = pd.read_csv(csv_path, header=None)
        if df.shape[1] >= 12:
            df = df.iloc[:, :12].copy()
            df.columns = [
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
            ]
        elif df.shape[1] >= 6:
            df = df.iloc[:, :6].copy()
            df.columns = ["open_time", "open", "high", "low", "close", "volume"]
        else:
            return pd.DataFrame()

    required = {"open_time": ["open_time"], "open": ["open"], "high": ["high"],
                "low": ["low"], "close": ["close"], "volume": ["volume"]}
    final_cols = {}
    for target, candidates in required.items():
        found = next((c for c in candidates if c in df.columns), None)
        if found is None:
            return pd.DataFrame()
        final_cols[target] = found

    df = df[[final_cols[k] for k in ["open_time","open","high","low","close","volume"]]].copy()
    df.columns = ["open_time", "open", "high", "low", "close", "volume"]
    df["datetime"] = parse_datetime_column(df["open_time"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["datetime", "open", "high", "low", "close", "volume"])
    if df.empty:
        return df
    df = df.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last")
    df = df.set_index("datetime")
    return df[["open", "high", "low", "close", "volume"]]

def discover_month_sources(root: Path):
    sources = {}
    for year_dir in sorted([p for p in root.iterdir() if p.is_dir() and p.name.isdigit()], key=lambda x: x.name):
        year = year_dir.name
        for item in sorted(year_dir.iterdir(), key=lambda x: x.name):
            if item.is_file() and item.suffix.lower() == ".csv" and item.stem.startswith("BTCUSDT-1m-"):
                sources[(year, item.stem)] = item
            elif item.is_dir() and item.name.startswith("BTCUSDT-1m-"):
                csv_files = sorted(item.glob("*.csv"))
                if not csv_files:
                    continue
                exact_name = item.name + ".csv"
                preferred  = next((f for f in csv_files if f.name == exact_name), csv_files[0])
                sources[(year, item.name)] = preferred
    return [
        {"year": y, "month_name": m, "csv_path": p}
        for (y, m), p in sorted(sources.items(), key=lambda x: (x[0][0], x[0][1]))
    ]

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if rule == "1min":
        return df.copy()
    return (
        df.resample(rule)
          .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
          .dropna()
    )

# ============================================================
# INDICATORS
# ============================================================

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    rsi      = 100 - (100 / (1 + rs))
    return rsi.bfill()

def compute_ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False, min_periods=period).mean()

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high  = df["high"]
    low   = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

# ============================================================
# STRATEGY: RSI CROSSOVER + EMA TREND + ATR FILTER + COOLDOWN
# ============================================================

def apply_rsi_rm_strategy(
    df: pd.DataFrame,
    tf_name: str,
    fee_rate: float = FEE_RATE
):
    if df.empty:
        return df.copy(), pd.DataFrame()

    params       = TF_PARAMS[tf_name]
    sl_pct       = params["sl"]
    tp_pct       = params["tp"]
    cooldown_n   = params["cooldown"]

    data             = df.copy()
    data["rsi"]      = compute_rsi(data["close"], RSI_PERIOD)
    data["ema_fast"] = compute_ema(data["close"], EMA_FAST)
    data["ema_slow"] = compute_ema(data["close"], EMA_SLOW)
    data["atr"]      = compute_atr(data, ATR_PERIOD)

    # RSI crossover: True on the candle where RSI crosses ABOVE BUY_RSI
    rsi_prev             = data["rsi"].shift(1)
    data["rsi_cross_up"] = (rsi_prev <= BUY_RSI) & (data["rsi"] > BUY_RSI)

    # ATR median for low-volatility filter (ignore bottom 30% of ATR candles)
    atr_median = data["atr"].median()

    data["buy_signal"]  = 0
    data["sell_signal"] = 0
    data["position"]    = 0
    data["entry_price"] = np.nan
    data["exit_price"]  = np.nan
    data["exit_reason"] = ""

    trades       = []
    in_position  = False
    entry_time   = None
    entry_idx    = None
    entry_pg     = None   # entry price gross
    entry_pn     = None   # entry price net (after buy fee)
    stop_price   = None
    take_price   = None
    cooldown_rem = 0      # candles remaining in cooldown

    for i in range(len(data)):
        row = data.iloc[i]
        ts  = data.index[i]

        if pd.isna(row["rsi"]) or pd.isna(row["close"]) or \
           pd.isna(row["ema_fast"]) or pd.isna(row["ema_slow"]) or \
           pd.isna(row["atr"]):
            if in_position:
                data.iat[i, data.columns.get_loc("position")] = 1
            continue

        if not in_position:
            if cooldown_rem > 0:
                cooldown_rem -= 1
                continue

            trend_ok  = row["ema_fast"] > row["ema_slow"]           # bullish regime
            signal_ok = bool(row["rsi_cross_up"])                   # RSI crossover
            vol_ok    = row["atr"] >= ATR_MIN_MULT * atr_median     # not dead market

            if trend_ok and signal_ok and vol_ok:
                entry_time  = ts
                entry_idx   = i
                entry_pg    = float(row["close"])
                entry_pn    = entry_pg * (1 + fee_rate)
                stop_price  = entry_pg * (1 - sl_pct)
                take_price  = entry_pg * (1 + tp_pct)
                in_position = True

                data.iat[i, data.columns.get_loc("buy_signal")]  = 1
                data.iat[i, data.columns.get_loc("position")]    = 1
                data.iat[i, data.columns.get_loc("entry_price")] = entry_pg
        else:
            data.iat[i, data.columns.get_loc("position")] = 1

            low   = float(row["low"])
            high  = float(row["high"])
            close = float(row["close"])

            exit_reason   = None
            exit_pg       = None

            # SL checked before TP -- conservative
            if low <= stop_price:
                exit_reason = "stop_loss"
                exit_pg     = stop_price
            elif high >= take_price:
                exit_reason = "take_profit"
                exit_pg     = take_price
            elif row["rsi"] > SELL_RSI:
                exit_reason = "rsi_exit"
                exit_pg     = close

            if exit_reason is not None:
                exit_pn  = exit_pg * (1 - fee_rate)
                pnl_pct  = ((exit_pn / entry_pn) - 1) * 100.0
                pnl_abs  = exit_pn - entry_pn

                trades.append({
                    "entry_time":       entry_time,
                    "exit_time":        ts,
                    "entry_price_gross": entry_pg,
                    "exit_price_gross":  exit_pg,
                    "entry_price_net":   entry_pn,
                    "exit_price_net":    exit_pn,
                    "stop_price":        stop_price,
                    "take_price":        take_price,
                    "exit_reason":       exit_reason,
                    "bars_held":         i - entry_idx,
                    "pnl_pct":           pnl_pct,
                    "pnl_abs":           pnl_abs
                })

                data.iat[i, data.columns.get_loc("sell_signal")]  = 1
                data.iat[i, data.columns.get_loc("exit_price")]   = exit_pg
                data.iat[i, data.columns.get_loc("exit_reason")]  = exit_reason

                in_position  = False
                cooldown_rem = cooldown_n
                entry_time = entry_idx = entry_pg = entry_pn = stop_price = take_price = None

    # Force-close any open position at end of month
    if in_position and CLOSE_OPEN_TRADE_AT_EOD and len(data) > 0:
        last_i   = len(data) - 1
        last_ts  = data.index[last_i]
        exit_pg  = float(data.iloc[last_i]["close"])
        exit_pn  = exit_pg * (1 - fee_rate)
        pnl_pct  = ((exit_pn / entry_pn) - 1) * 100.0
        pnl_abs  = exit_pn - entry_pn

        trades.append({
            "entry_time":        entry_time,
            "exit_time":         last_ts,
            "entry_price_gross": entry_pg,
            "exit_price_gross":  exit_pg,
            "entry_price_net":   entry_pn,
            "exit_price_net":    exit_pn,
            "stop_price":        stop_price,
            "take_price":        take_price,
            "exit_reason":       "eod",
            "bars_held":         last_i - entry_idx,
            "pnl_pct":           pnl_pct,
            "pnl_abs":           pnl_abs
        })

        data.iat[last_i, data.columns.get_loc("sell_signal")]  = 1
        data.iat[last_i, data.columns.get_loc("exit_price")]   = exit_pg
        data.iat[last_i, data.columns.get_loc("exit_reason")]  = "eod"

    return data, pd.DataFrame(trades)

# ============================================================
# METRICS (equity-curve based, annualized Sharpe)
# ============================================================

def compute_metrics(trades_df: pd.DataFrame, tf_name: str) -> dict:
    empty = {
        "num_trades": 0, "win_count": 0, "loss_count": 0,
        "win_rate": 0.0, "total_pnl_pct": 0.0, "max_drawdown_pct": 0.0,
        "sharpe_ratio": 0.0, "avg_trade_pct": 0.0,
        "best_trade_pct": 0.0, "worst_trade_pct": 0.0
    }
    if trades_df.empty:
        return empty

    returns     = trades_df["pnl_pct"].astype(float) / 100.0
    equity      = (1 + returns).cumprod()
    running_max = equity.cummax()
    drawdown    = (equity / running_max - 1.0) * 100.0

    wins       = int((trades_df["pnl_pct"] > 0).sum())
    losses     = int((trades_df["pnl_pct"] <= 0).sum())
    num_trades = len(trades_df)
    win_rate   = (wins / num_trades) * 100.0 if num_trades > 0 else 0.0
    total_pnl  = (equity.iloc[-1] - 1.0) * 100.0

    ann_factor = ANNUALIZATION.get(tf_name, 252)
    std        = returns.std(ddof=1)
    if pd.isna(std) or std == 0 or num_trades < 2:
        sharpe = 0.0
    else:
        # Per-trade frequency relative to annualization period
        periods_per_trade = ann_factor / num_trades
        sharpe = (returns.mean() / std) * np.sqrt(ann_factor / max(periods_per_trade, 1))

    return {
        "num_trades":      num_trades,
        "win_count":       wins,
        "loss_count":      losses,
        "win_rate":        float(win_rate),
        "total_pnl_pct":   float(total_pnl),
        "max_drawdown_pct": float(abs(drawdown.min())) if len(drawdown) else 0.0,
        "sharpe_ratio":    float(np.clip(sharpe, -50, 50)),  # cap extreme values
        "avg_trade_pct":   float(trades_df["pnl_pct"].mean()),
        "best_trade_pct":  float(trades_df["pnl_pct"].max()),
        "worst_trade_pct": float(trades_df["pnl_pct"].min())
    }

# ============================================================
# OUTPUT HELPERS (unchanged)
# ============================================================

def build_output_dir(base_output: Path, timeframe_name: str, year: str, month_name: str) -> Path:
    bucket = get_bucket_name(timeframe_name)
    if bucket == "intraday":
        out_dir = base_output / bucket / timeframe_name / year / month_name
    else:
        out_dir = base_output / bucket / year / month_name
    safe_mkdir(out_dir)
    return out_dir

def save_outputs(out_dir: Path, processed_df: pd.DataFrame, trades_df: pd.DataFrame, summary_row: dict):
    if SAVE_PROCESSED_SIGNALS:
        processed_df.reset_index().rename(columns={"datetime": "timestamp"}).to_csv(
            out_dir / "processed_signals.csv", index=False)
    if SAVE_TRADE_LOG:
        trades_df.to_csv(out_dir / "trade_log.csv", index=False)
    if SAVE_SUMMARY:
        pd.DataFrame([summary_row]).to_csv(out_dir / "summary.csv", index=False)

def aggregate_by_timeframe(master_df: pd.DataFrame) -> pd.DataFrame:
    if master_df.empty:
        return pd.DataFrame()
    rows = []
    for tf, g in master_df.groupby("timeframe"):
        total_trades = int(g["num_trades"].sum())
        total_wins   = int(g["win_count"].sum())
        win_rate     = (total_wins / total_trades * 100.0) if total_trades > 0 else 0.0
        pnl_comp     = ((1 + g["total_pnl_pct"].fillna(0) / 100.0).prod() - 1.0) * 100.0
        max_dd       = g["max_drawdown_pct"].fillna(0).max()
        sharpe_vals  = g["sharpe_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
        sharpe       = sharpe_vals.mean() if not sharpe_vals.empty else 0.0
        rows.append({
            "timeframe":        tf,
            "total_trades":     total_trades,
            "win_rate":         round(win_rate, 3),
            "total_pnl_pct":    round(float(pnl_comp), 3),
            "max_drawdown_pct": round(float(max_dd), 3),
            "sharpe_ratio":     round(float(sharpe), 3)
        })
    return pd.DataFrame(rows).sort_values("timeframe").reset_index(drop=True)

# ============================================================
# MAIN
# ============================================================

def main():
    safe_mkdir(OUTPUT_ROOT)
    month_sources = discover_month_sources(DATASETS_ROOT)
    if not month_sources:
        print("No monthly source CSVs found.")
        return

    all_summary_rows = []
    print("=" * 60)
    print("RSI + RISK MANAGEMENT BACKTEST  (v3 -- full fixes)")
    print("=" * 60)

    for item in month_sources:
        year       = item["year"]
        month_name = item["month_name"]
        csv_path   = item["csv_path"]
        print(f"\n[{year}] {month_name}")

        try:
            raw_df = read_ohlcv_csv(csv_path)
        except Exception as e:
            print(f"  -> Read failed: {e}")
            continue

        if raw_df.empty:
            print("  -> No data, skipped.")
            continue

        for tf_name, rule in TIMEFRAMES.items():
            try:
                tf_df = resample_ohlcv(raw_df, rule)
                if tf_df.empty:
                    print(f"    [{tf_name:<6}] -> empty after resample, skipped.")
                    continue

                processed_df, trades_df = apply_rsi_rm_strategy(tf_df, tf_name)
                metrics = compute_metrics(trades_df, tf_name)

                summary_row = {
                    "year": year, "month": month_name,
                    "timeframe": tf_name, "source_file": str(csv_path),
                    "rows": len(tf_df), **metrics
                }

                out_dir = build_output_dir(OUTPUT_ROOT, tf_name, year, month_name)
                save_outputs(out_dir, processed_df, trades_df, summary_row)
                all_summary_rows.append(summary_row)

                print(
                    f"    [{tf_name:<6}] trades={metrics['num_trades']:4d}  "
                    f"wr={metrics['win_rate']:5.1f}%  "
                    f"pnl={metrics['total_pnl_pct']:8.3f}%  "
                    f"sharpe={metrics['sharpe_ratio']:6.3f}"
                )
            except Exception as e:
                print(f"    [{tf_name:<6}] failed: {e}")

    master_df      = pd.DataFrame(all_summary_rows)
    master_path    = OUTPUT_ROOT / "RSI_RM_master_summary.csv"
    aggregate_path = OUTPUT_ROOT / "RSI_RM_aggregate_summary.csv"

    if not master_df.empty:
        master_df.to_csv(master_path, index=False)
        agg_df = aggregate_by_timeframe(master_df)
        agg_df.to_csv(aggregate_path, index=False)

        print("\n" + "=" * 60)
        print("AGGREGATE RESULTS BY TIMEFRAME")
        print("=" * 60)
        print(agg_df.set_index("timeframe").to_string())
        print(f"\nMaster summary    -> {master_path}")
        print(f"Aggregate summary -> {aggregate_path}")
    else:
        print("\nNo summaries generated.")

if __name__ == "__main__":
    main()