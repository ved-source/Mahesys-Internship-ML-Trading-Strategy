#!/usr/bin/env python3
"""
NOTE: No logic changes required for Script 3 v2.
      All Renko/signal/feature logic is mirrored in Script 3 directly.

SCRIPT 1: Renko Label Generation + Backtesting with Risk Management
- Reads 1-minute BTCUSDT CSVs from DATASETS/ hierarchy
- Builds Renko bricks (ATR-based brick size)
- Generates BUY/SELL/HOLD signals with EMA confirmation
- Applies risk management (ATR stop-loss, 2:1 RR take-profit, position sizing)
- Backtests strategy and validates results
- Saves Renko CSVs mirroring original dataset hierarchy under DATASETS/RENKO_LABELS/
- Retries CSV + backtest generation until quality criteria are met
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(r"C:\Users\saive\OneDrive\Desktop\Desktop\all folders\self-projects\placement\ML-trading\DATASETS")
RENKO_OUT_DIR = BASE_DIR / "RENKO_LABELS"
RENKO_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Renko params
ATR_PERIOD       = 14          # ATR lookback for dynamic brick size
ATR_MULTIPLIER   = 1.5         # Brick size = ATR * multiplier
EMA_SHORT        = 9
EMA_LONG         = 21

# Risk management
RISK_PER_TRADE   = 0.01        # 1% risk per trade
INITIAL_CAPITAL  = 10_000.0    # USD
SL_ATR_MULT      = 1.5         # Stop-loss = entry ± SL_ATR_MULT * ATR
RR_RATIO         = 2.0         # Take-profit = SL_distance * RR_RATIO

# Quality gate thresholds
MIN_TRADES        = 30
MIN_WIN_RATE      = 0.40
MIN_PROFIT_FACTOR = 1.2
MAX_RETRY         = 3

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def build_renko_bricks(df: pd.DataFrame, brick_size: float):
    """
    Classic Renko from OHLCV 1-min bars.
    Returns a list of dicts: {open, close, direction, timestamp, volume_sum, atr}
    """
    bricks = []
    prices = df["close"].values
    timestamps = df["timestamp"].values if "timestamp" in df.columns else df.index.values
    volumes = df["volume"].values
    atrs = df["atr"].values

    # Seed first brick
    start_price = prices[0]
    # Round to nearest brick
    current_base = np.floor(start_price / brick_size) * brick_size
    direction = None   # None until first brick forms

    vol_accum = 0.0

    for i, price in enumerate(prices):
        vol_accum += volumes[i]
        curr_atr = atrs[i]

        if direction is None or direction == 1:  # looking for up OR continuing up
            if price >= current_base + brick_size:
                new_base = current_base + brick_size
                bricks.append({
                    "timestamp":   timestamps[i],
                    "open":        current_base,
                    "close":       new_base,
                    "direction":   1,          # bullish
                    "brick_size":  brick_size,
                    "volume":      vol_accum,
                    "atr":         curr_atr,
                })
                current_base = new_base
                direction = 1
                vol_accum = 0.0
            elif direction == 1 and price <= current_base - brick_size:
                new_base = current_base - brick_size
                bricks.append({
                    "timestamp":   timestamps[i],
                    "open":        current_base,
                    "close":       new_base,
                    "direction":   -1,
                    "brick_size":  brick_size,
                    "volume":      vol_accum,
                    "atr":         curr_atr,
                })
                current_base = new_base
                direction = -1
                vol_accum = 0.0


        if direction == -1:  # continuing down
            if price <= current_base - brick_size:
                new_base = current_base - brick_size
                bricks.append({
                    "timestamp":   timestamps[i],
                    "open":        current_base,
                    "close":       new_base,
                    "direction":   -1,
                    "brick_size":  brick_size,
                    "volume":      vol_accum,
                    "atr":         curr_atr,
                })
                current_base = new_base
                vol_accum = 0.0
            elif price >= current_base + brick_size:
                new_base = current_base + brick_size
                bricks.append({
                    "timestamp":   timestamps[i],
                    "open":        current_base,
                    "close":       new_base,
                    "direction":   1,
                    "brick_size":  brick_size,
                    "volume":      vol_accum,
                    "atr":         curr_atr,
                })
                current_base = new_base
                direction = 1
                vol_accum = 0.0

    return bricks


def add_renko_signals(rdf: pd.DataFrame) -> pd.DataFrame:
    """
    Signal logic:
      BUY  (1) : direction flips from -1 → +1  AND  close > EMA_SHORT > EMA_LONG
      SELL (-1): direction flips from +1 → -1  AND  close < EMA_SHORT < EMA_LONG
      HOLD (0) : otherwise
    """
    rdf = rdf.copy()
    rdf["ema_short"] = rdf["close"].ewm(span=EMA_SHORT, adjust=False).mean()
    rdf["ema_long"]  = rdf["close"].ewm(span=EMA_LONG,  adjust=False).mean()

    prev_dir = rdf["direction"].shift(1)
    flip_up   = (prev_dir == -1) & (rdf["direction"] == 1)
    flip_down = (prev_dir ==  1) & (rdf["direction"] == -1)

    trend_up   = rdf["close"] > rdf["ema_short"]
    trend_down = rdf["close"] < rdf["ema_short"]

    rdf["signal"] = 0
    rdf.loc[flip_up   & trend_up,   "signal"] =  1   # BUY
    rdf.loc[flip_down & trend_down, "signal"] = -1   # SELL

    # Risk management columns
    rdf["stop_loss"]    = np.where(
        rdf["signal"] ==  1, rdf["close"] - SL_ATR_MULT * rdf["atr"],
        np.where(rdf["signal"] == -1, rdf["close"] + SL_ATR_MULT * rdf["atr"], np.nan)
    )
    rdf["take_profit"]  = np.where(
        rdf["signal"] ==  1, rdf["close"] + SL_ATR_MULT * rdf["atr"] * RR_RATIO,
        np.where(rdf["signal"] == -1, rdf["close"] - SL_ATR_MULT * rdf["atr"] * RR_RATIO, np.nan)
    )
    sl_dist = (rdf["close"] - rdf["stop_loss"]).abs()
    rdf["position_size_pct"] = np.where(
        rdf["signal"] != 0,
        (RISK_PER_TRADE * INITIAL_CAPITAL) / (sl_dist + 1e-9),
        0.0
    )
    return rdf


def backtest(rdf: pd.DataFrame):
    """Simple event-driven backtest on Renko signal rows."""
    capital  = INITIAL_CAPITAL
    equity   = [capital]
    trades   = []
    in_trade = False
    entry_price = sl = tp = direction = 0

    for _, row in rdf.iterrows():
        price = row["close"]

        if in_trade:
            if direction == 1:
                if price <= sl:
                    pnl = -(entry_price - sl) * qty
                    capital += pnl
                    trades.append(pnl)
                    in_trade = False
                elif price >= tp:
                    pnl = (tp - entry_price) * qty
                    capital += pnl
                    trades.append(pnl)
                    in_trade = False
            else:
                if price >= sl:
                    pnl = -(sl - entry_price) * qty
                    capital += pnl
                    trades.append(pnl)
                    in_trade = False
                elif price <= tp:
                    pnl = (entry_price - tp) * qty
                    capital += pnl
                    trades.append(pnl)
                    in_trade = False
            equity.append(capital)

        if not in_trade and row["signal"] != 0:
            direction   = row["signal"]
            entry_price = price
            sl          = row["stop_loss"]
            tp          = row["take_profit"]
            sl_dist     = abs(entry_price - sl)
            qty         = (RISK_PER_TRADE * capital) / (sl_dist + 1e-9)
            in_trade    = True

    return trades, equity


def evaluate_backtest(trades, equity):
    if len(trades) < MIN_TRADES:
        return None, "Too few trades"

    trades = np.array(trades)
    wins   = trades[trades > 0]
    losses = trades[trades < 0]

    win_rate      = len(wins) / len(trades)
    gross_profit  = wins.sum()  if len(wins)   else 0
    gross_loss    = abs(losses.sum()) if len(losses) else 1e-9
    profit_factor = gross_profit / gross_loss
    net_pnl       = trades.sum()
    max_drawdown  = 0.0
    peak = equity[0]
    for e in equity:
        if e > peak: peak = e
        dd = (peak - e) / peak
        if dd > max_drawdown: max_drawdown = dd

    result = {
        "total_trades":   len(trades),
        "win_rate":       round(win_rate, 4),
        "profit_factor":  round(profit_factor, 4),
        "net_pnl":        round(net_pnl, 2),
        "max_drawdown_%": round(max_drawdown * 100, 2),
        "avg_win":        round(wins.mean(),   2) if len(wins)   else 0,
        "avg_loss":       round(losses.mean(), 2) if len(losses) else 0,
        "final_capital":  round(equity[-1], 2),
    }

    # Quality gate
    if win_rate < MIN_WIN_RATE:
        return result, f"Win rate {win_rate:.2%} < {MIN_WIN_RATE:.0%}"
    if profit_factor < MIN_PROFIT_FACTOR:
        return result, f"Profit factor {profit_factor:.2f} < {MIN_PROFIT_FACTOR}"
    return result, "OK"


def print_results(result, label=""):
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  BACKTEST RESULTS  {label}")
    print(sep)
    for k, v in result.items():
        print(f"  {k:<22}: {v}")
    print(sep)


# ─────────────────────────────────────────────────────────────
# CORE PROCESSING
# ─────────────────────────────────────────────────────────────

def detect_timestamp_unit(ts_val) -> str:
    """
    Auto-detect whether a Binance timestamp is in:
      - milliseconds  (13 digits):  e.g. 1746057600000    (2017-2024 data)
      - microseconds  (16 digits):  e.g. 1746057600000000 (2025+ data)
    Returns 'ms' or 'us'.
    """
    try:
        s = str(int(float(ts_val)))
        if len(s) >= 16:
            return "us"
        return "ms"
    except Exception:
        return "ms"


def load_year_month_csv(year_dir: Path, month_folder: str) -> pd.DataFrame | None:
    folder = year_dir / month_folder
    csvs   = list(folder.glob("*.csv"))
    if not csvs:
        return None
    frames = []
    for f in sorted(csvs):
        try:
            df = pd.read_csv(f, header=None)
            # Binance 1m format: timestamp,open,high,low,close,volume,...
            df = df.iloc[:, :6]
            df.columns = ["timestamp", "open", "high", "low", "close", "volume"]

            # Auto-detect timestamp unit from first valid value
            first_ts = df["timestamp"].dropna().iloc[0]
            ts_unit  = detect_timestamp_unit(first_ts)

            if ts_unit == "us":
                # Microseconds -> divide by 1000 to get ms, then parse
                df["timestamp"] = pd.to_datetime(
                    pd.to_numeric(df["timestamp"], errors="coerce") // 1000,
                    unit="ms", errors="coerce"
                )
            else:
                df["timestamp"] = pd.to_datetime(
                    pd.to_numeric(df["timestamp"], errors="coerce"),
                    unit="ms", errors="coerce"
                )

            df = df.dropna(subset=["timestamp"])
            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna()
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return None
    result = pd.concat(frames).sort_values("timestamp").reset_index(drop=True)
    return result


def process_month(df: pd.DataFrame, year: str, month: str, attempt: int = 0):
    """Build Renko, generate signals, backtest. Retry with looser ATR mult if needed."""
    # Possibly widen brick size on retries
    atr_mult = ATR_MULTIPLIER + attempt * 0.3

    df["atr"] = compute_atr(df, ATR_PERIOD)
    median_atr  = df["atr"].median()
    brick_size  = median_atr * atr_mult
    if brick_size <= 0:
        return None, None, "zero brick size"

    bricks = build_renko_bricks(df, brick_size)
    if len(bricks) < 20:
        return None, None, "too few bricks"

    rdf = pd.DataFrame(bricks)
    rdf = add_renko_signals(rdf)

    # Add extra features useful for ML (Script 2)
    rdf["brick_count"]     = range(len(rdf))
    rdf["consec_dir"]      = rdf["direction"].groupby(
        (rdf["direction"] != rdf["direction"].shift()).cumsum()
    ).cumcount() + 1
    rdf["volume_ma5"]      = rdf["volume"].rolling(5, min_periods=1).mean()
    rdf["vol_ratio"]       = rdf["volume"] / (rdf["volume_ma5"] + 1e-9)
    rdf["price_range_pct"] = (rdf["close"] - rdf["open"]).abs() / (rdf["open"] + 1e-9) * 100
    rdf["year"]  = year
    rdf["month"] = month

    trades, equity = backtest(rdf)
    result, status = evaluate_backtest(trades, equity)

    return rdf, result, status


def process_all():
    all_results = []
    year_dirs = sorted([d for d in BASE_DIR.iterdir() if d.is_dir() and d.name.isdigit()])

    if not year_dirs:
        print(f"[ERROR] No year directories found under {BASE_DIR}")
        sys.exit(1)

    print(f"Found {len(year_dirs)} year directories: {[d.name for d in year_dirs]}")

    total_months = 0
    failed       = 0

    for year_dir in year_dirs:
        year = year_dir.name
        out_year = RENKO_OUT_DIR / year
        out_year.mkdir(parents=True, exist_ok=True)

        month_folders = sorted([d.name for d in year_dir.iterdir() if d.is_dir()])
        for month_folder in month_folders:
            total_months += 1
            month = month_folder.split("-")[-1] if "-" in month_folder else month_folder

            df = load_year_month_csv(year_dir, month_folder)
            if df is None or len(df) < 100:
                print(f"[SKIP] {year}/{month_folder} - no data")
                failed += 1
                continue

            # Retry loop
            rdf = result = None
            status = "INIT"
            for attempt in range(MAX_RETRY):
                rdf, result, status = process_month(df, year, month, attempt)
                if status == "OK":
                    break
                print(f"  [RETRY {attempt+1}/{MAX_RETRY}] {year}/{month_folder}: {status}")

            if status != "OK" or rdf is None:
                print(f"[FAIL] {year}/{month_folder}: {status} after {MAX_RETRY} attempts")
                failed += 1
                if rdf is not None:  # still save even if quality gate fails
                    out_path = out_year / f"RENKO_{month_folder}.csv"
                    rdf.to_csv(out_path, index=False)
                continue

            out_path = out_year / f"RENKO_{month_folder}.csv"
            rdf.to_csv(out_path, index=False)
            print(f"[OK]   {year}/{month_folder} -> {len(rdf)} bricks | "
                  f"Trades={result['total_trades']} WR={result['win_rate']:.0%} "
                  f"PF={result['profit_factor']:.2f} NetPnL=${result['net_pnl']:.0f}")
            all_results.append({"year": year, "month": month, **result})

    # ── Aggregate backtest summary ──────────────────────────────
    if all_results:
        agg = pd.DataFrame(all_results)
        print("\n" + "=" * 70)
        print("  AGGREGATE BACKTEST SUMMARY ACROSS ALL MONTHS")
        print("=" * 70)
        numeric = agg.select_dtypes(include="number")
        print(numeric.describe().round(3).to_string())
        print("=" * 70)
        print(f"  Total months processed : {total_months}")
        print(f"  Successful (quality OK): {len(all_results)}")
        print(f"  Failed / skipped       : {failed}")
        total_trades = agg["total_trades"].sum()
        overall_wr   = (agg["win_rate"] * agg["total_trades"]).sum() / (total_trades + 1e-9)
        overall_pnl  = agg["net_pnl"].sum()
        print(f"  Combined trades        : {int(total_trades)}")
        print(f"  Weighted win rate      : {overall_wr:.2%}")
        print(f"  Total net PnL          : ${overall_pnl:,.2f}")
        print("=" * 70)

        # Save summary
        summary_path = RENKO_OUT_DIR / "backtest_summary.csv"
        agg.to_csv(summary_path, index=False)
        print(f"\nBacktest summary saved to: {summary_path}")
    else:
        print("[WARNING] No successful months processed. Check dataset path and file formats.")

    print(f"\nRenko CSVs saved to: {RENKO_OUT_DIR}")


if __name__ == "__main__":
    print("=" * 70)
    print("  STEP 1: RENKO LABEL GENERATION + BACKTESTING")
    print("=" * 70)
    process_all()
    print("\n[DONE] Script 1 complete.")
