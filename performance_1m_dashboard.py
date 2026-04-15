#!/usr/bin/env python3
"""
PERFORMANCE ANALYSIS DASHBOARD — 1-Minute Timeframe Only
Reads logs from PAPER_TRADE_LOGS
Generates a full HTML performance report
"""

import json
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
BASE_DIR   = Path(r"C:\Users\saive\OneDrive\Desktop\Desktop\all folders\self-projects\placement\ML-trading")
LOGS_DIR   = BASE_DIR / "PAPER_TRADE_LOGS"
OUTPUT_DIR = BASE_DIR / "PERFORMANCE_REPORT"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_HTML = OUTPUT_DIR / "performance_1m.html"

INITIAL_CAPITAL = 10_000.0

COLORS = {
    "primary": "#00d4aa",
    "win":     "#22c55e",
    "loss":    "#ef4444",
    "warn":    "#f59e0b",
    "bg":      "#0d1117",
    "card":    "#161b22",
    "border":  "#21262d",
    "text":    "#e6edf3",
    "muted":   "#8b949e",
    "grid":    "#21262d",
}

# ─────────────────────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────────────────────

def load_trades():
    p = LOGS_DIR / "paper_trade_log.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "pnl" not in df.columns:
        return pd.DataFrame()
    df = df[df["pnl"].notna()].copy()
    for col in ["open_ts", "close_ts"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return df

def load_signals():
    p = LOGS_DIR / "signal_log.csv"
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(p, on_bad_lines="skip")
    except TypeError:
        df = pd.read_csv(p, error_bad_lines=False)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    return df

def load_capital():
    p = LOGS_DIR / "capital_state.json"
    if not p.exists():
        return {"capital": INITIAL_CAPITAL, "trade_counter": 0}
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return {"capital": INITIAL_CAPITAL, "trade_counter": 0}

# ─────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────

def compute_all_metrics(trades):
    if trades.empty:
        return {}
    pnls   = trades["pnl"].values
    wins   = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    n      = len(pnls)

    win_rate     = len(wins) / n if n > 0 else 0
    gross_profit = float(wins.sum())        if len(wins)   else 0.0
    gross_loss   = float(abs(losses.sum())) if len(losses) else 1e-9
    pf           = gross_profit / gross_loss
    net_pnl      = float(pnls.sum())
    avg_win      = float(wins.mean())   if len(wins)   else 0.0
    avg_loss     = float(losses.mean()) if len(losses) else 0.0
    expectancy   = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    eq   = np.concatenate([[INITIAL_CAPITAL], INITIAL_CAPITAL + np.cumsum(pnls)])
    peak = np.maximum.accumulate(eq)
    dd   = (peak - eq) / (peak + 1e-9)
    max_dd = float(dd.max())

    if len(eq) > 2:
        rets   = np.diff(eq) / (eq[:-1] + 1e-9)
        sharpe = (rets.mean() / (rets.std() + 1e-9)) * np.sqrt(252 * 1440)
    else:
        sharpe = 0.0

    best_trade  = float(pnls.max())
    worst_trade = float(pnls.min())
    avg_trade   = float(pnls.mean())
    ml_conf     = trades["ml_conf"].mean() * 100 if "ml_conf" in trades.columns else 0.0
    consec_wins = 0
    consec_losses = 0
    cur_w = cur_l = 0
    for p in pnls:
        if p > 0: cur_w += 1; cur_l = 0
        else:     cur_l += 1; cur_w = 0
        consec_wins   = max(consec_wins,   cur_w)
        consec_losses = max(consec_losses, cur_l)

    return {
        "Total Trades":         n,
        "Wins":                 len(wins),
        "Losses":               len(losses),
        "Win Rate (%)":         round(win_rate * 100, 2),
        "Net PnL ($)":          round(net_pnl, 2),
        "Gross Profit ($)":     round(gross_profit, 2),
        "Gross Loss ($)":       round(-abs(losses.sum()), 2) if len(losses) else 0.0,
        "Profit Factor":        round(pf, 4),
        "Expectancy ($)":       round(expectancy, 4),
        "Avg Win ($)":          round(avg_win, 2),
        "Avg Loss ($)":         round(avg_loss, 2),
        "Best Trade ($)":       round(best_trade, 2),
        "Worst Trade ($)":      round(worst_trade, 2),
        "Avg Trade ($)":        round(avg_trade, 4),
        "Max Drawdown (%)":     round(max_dd * 100, 2),
        "Sharpe Ratio":         round(float(sharpe), 4),
        "Avg ML Conf (%)":      round(ml_conf, 2),
        "Max Consec. Wins":     consec_wins,
        "Max Consec. Losses":   consec_losses,
        "Final Capital ($)":    round(INITIAL_CAPITAL + net_pnl, 2),
        "Return (%)":           round(net_pnl / INITIAL_CAPITAL * 100, 4),
    }

# ─────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────
trades  = load_trades()
signals = load_signals()
cap     = load_capital()
metrics = compute_all_metrics(trades)
now     = datetime.now().strftime("%Y-%m-%d %H:%M")

current_cap = cap.get("capital", INITIAL_CAPITAL)
net         = current_cap - INITIAL_CAPITAL
ret         = net / INITIAL_CAPITAL * 100
n           = len(trades)
wins_n      = int((trades["pnl"] > 0).sum()) if not trades.empty else 0
wr          = round(wins_n / n * 100, 1) if n > 0 else 0

# ─────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────
LAYOUT = dict(
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["bg"],
    font=dict(color=COLORS["text"], family="Segoe UI"),
    margin=dict(l=40, r=20, t=60, b=40),
    legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
)

def fig_html(fig):
    return fig.to_html(full_html=False, include_plotlyjs=False)

# 1. Equity curve
fig_eq = go.Figure()
if not trades.empty:
    eq_df = trades.sort_values("close_ts").copy()
    eq_df["capital"] = INITIAL_CAPITAL + eq_df["pnl"].cumsum()
    fig_eq.add_trace(go.Scatter(
        x=eq_df["close_ts"], y=eq_df["capital"],
        mode="lines", name="Equity",
        line=dict(color=COLORS["primary"], width=2.5),
        fill="tozeroy", fillcolor="rgba(0,212,170,0.08)"
    ))
    fig_eq.add_hline(y=INITIAL_CAPITAL, line_dash="dash",
                     line_color=COLORS["muted"], annotation_text="Start $10,000")
fig_eq.update_layout(title="Equity Curve", **LAYOUT)
fig_eq.update_xaxes(title_text="Date", gridcolor=COLORS["grid"])
fig_eq.update_yaxes(title_text="Capital ($)", gridcolor=COLORS["grid"])

# 2. Drawdown
fig_dd = go.Figure()
if not trades.empty:
    pnls = trades["pnl"].values
    eq   = np.concatenate([[INITIAL_CAPITAL], INITIAL_CAPITAL + np.cumsum(pnls)])
    peak = np.maximum.accumulate(eq)
    dd   = (peak - eq) / (peak + 1e-9) * 100
    fig_dd.add_trace(go.Scatter(
        x=list(range(len(dd))), y=-dd,
        mode="lines", name="Drawdown",
        line=dict(color=COLORS["loss"], width=2),
        fill="tozeroy", fillcolor="rgba(239,68,68,0.1)"
    ))
fig_dd.update_layout(title="Drawdown Curve (%)", **LAYOUT)
fig_dd.update_xaxes(title_text="Trade #", gridcolor=COLORS["grid"])
fig_dd.update_yaxes(title_text="Drawdown (%)", gridcolor=COLORS["grid"])

# 3. PnL per trade
fig_pnl = go.Figure()
if not trades.empty:
    bar_colors = [COLORS["win"] if p > 0 else COLORS["loss"] for p in trades["pnl"]]
    fig_pnl.add_trace(go.Bar(
        x=list(range(1, len(trades)+1)),
        y=trades["pnl"].values,
        marker_color=bar_colors, opacity=0.9, name="PnL"
    ))
fig_pnl.update_layout(title="PnL per Trade", **LAYOUT)
fig_pnl.update_xaxes(title_text="Trade #", gridcolor=COLORS["grid"])
fig_pnl.update_yaxes(title_text="PnL ($)", gridcolor=COLORS["grid"])

# 4. Win/Loss pie
fig_pie = go.Figure()
if not trades.empty:
    fig_pie.add_trace(go.Pie(
        labels=["Win", "Loss"],
        values=[wins_n, n - wins_n],
        marker=dict(colors=[COLORS["win"], COLORS["loss"]]),
        hole=0.45, textinfo="percent+label",
        name="Win/Loss"
    ))
fig_pie.update_layout(title="Win / Loss Distribution", **LAYOUT,
                      uniformtext_minsize=13, uniformtext_mode="hide")

# 5. Daily PnL
fig_daily = go.Figure()
if not trades.empty:
    t2 = trades.copy()
    t2["date"] = pd.to_datetime(t2["close_ts"]).dt.date
    daily = t2.groupby("date")["pnl"].sum().reset_index()
    dc = [COLORS["win"] if p > 0 else COLORS["loss"] for p in daily["pnl"]]
    fig_daily.add_trace(go.Bar(
        x=daily["date"].astype(str), y=daily["pnl"],
        marker_color=dc, opacity=0.9, name="Daily PnL"
    ))
fig_daily.update_layout(title="Daily Net PnL ($)", **LAYOUT)
fig_daily.update_xaxes(title_text="Date", gridcolor=COLORS["grid"])
fig_daily.update_yaxes(title_text="PnL ($)", gridcolor=COLORS["grid"])

# 6. ML confidence distribution
fig_conf = go.Figure()
if not trades.empty and "ml_conf" in trades.columns:
    fig_conf.add_trace(go.Histogram(
        x=trades["ml_conf"] * 100,
        marker_color=COLORS["primary"], opacity=0.8,
        xbins=dict(start=50, end=100, size=2),
        name="ML Confidence"
    ))
fig_conf.update_layout(title="ML Confidence on Executed Trades", **LAYOUT)
fig_conf.update_xaxes(title_text="Confidence (%)", gridcolor=COLORS["grid"])
fig_conf.update_yaxes(title_text="Trade Count", gridcolor=COLORS["grid"])

# 7. Cumulative PnL
fig_cum = go.Figure()
if not trades.empty:
    cum = trades["pnl"].cumsum().values
    colors_cum = [COLORS["win"] if v > 0 else COLORS["loss"] for v in cum]
    fig_cum.add_trace(go.Scatter(
        x=list(range(1, len(cum)+1)), y=cum,
        mode="lines+markers",
        line=dict(color=COLORS["primary"], width=2),
        marker=dict(color=colors_cum, size=5),
        name="Cumulative PnL"
    ))
    fig_cum.add_hline(y=0, line_dash="dash", line_color=COLORS["muted"])
fig_cum.update_layout(title="Cumulative PnL ($)", **LAYOUT)
fig_cum.update_xaxes(title_text="Trade #", gridcolor=COLORS["grid"])
fig_cum.update_yaxes(title_text="Cumulative PnL ($)", gridcolor=COLORS["grid"])

# 8. Skip reasons
fig_skip = go.Figure()
if not signals.empty and "event" in signals.columns and "skip_reason" in signals.columns:
    skipped = signals[signals["event"] == "SIGNAL_SKIPPED"]
    if not skipped.empty:
        sc = skipped["skip_reason"].value_counts()
        fig_skip.add_trace(go.Bar(
            x=sc.index.tolist(), y=sc.values.tolist(),
            marker_color=COLORS["warn"], opacity=0.9, name="Skip Reasons"
        ))
fig_skip.update_layout(title="Signal Skip Reasons", **LAYOUT)
fig_skip.update_xaxes(title_text="Reason", gridcolor=COLORS["grid"])
fig_skip.update_yaxes(title_text="Count", gridcolor=COLORS["grid"])

# 9. Exit type breakdown
fig_exit = go.Figure()
if not trades.empty and "exit_reason" in trades.columns:
    ec = trades["exit_reason"].value_counts()
    exit_colors = []
    for r in ec.index:
        if "tp" in str(r).lower() or "take" in str(r).lower():
            exit_colors.append(COLORS["win"])
        elif "sl" in str(r).lower() or "stop" in str(r).lower():
            exit_colors.append(COLORS["loss"])
        else:
            exit_colors.append(COLORS["warn"])
    fig_exit.add_trace(go.Bar(
        x=ec.index.tolist(), y=ec.values.tolist(),
        marker_color=exit_colors, opacity=0.9, name="Exit Types"
    ))
fig_exit.update_layout(title="Trade Exit Type Breakdown", **LAYOUT)
fig_exit.update_xaxes(title_text="Exit Type", gridcolor=COLORS["grid"])
fig_exit.update_yaxes(title_text="Count", gridcolor=COLORS["grid"])

# ─────────────────────────────────────────────────────────────
# METRICS TABLE HTML
# ─────────────────────────────────────────────────────────────
def metrics_table_html(m):
    if not m:
        return "<p style='color:#6e7681'>No closed trades yet.</p>"
    rows = ""
    for k, v in m.items():
        color = ""
        if isinstance(v, (int, float)):
            if any(x in k for x in ["PnL","Return","Profit","Win","Best","Avg Trade","Expectancy"]):
                color = "color:#22c55e;" if v > 0 else "color:#ef4444;" if v < 0 else ""
            if "Drawdown" in k or "Loss" in k or "Worst" in k:
                color = "color:#ef4444;" if v != 0 else ""
        rows += f"""
        <tr>
          <td style="color:#8b949e;padding:9px 16px;border-bottom:1px solid #21262d;">{k}</td>
          <td style="{color}font-weight:600;padding:9px 16px;border-bottom:1px solid #21262d;">{v}</td>
        </tr>"""
    return f"<table style='width:100%;border-collapse:collapse;font-size:13px;'><tbody>{rows}</tbody></table>"

# ─────────────────────────────────────────────────────────────
# TRADE LOG TABLE
# ─────────────────────────────────────────────────────────────
def trade_log_html(df):
    if df.empty:
        return "<p style='color:#6e7681'>No trades recorded.</p>"
    display_cols = [c for c in ["close_ts","side","entry_price","exit_price","sl","tp","pnl","ml_conf","exit_reason"] if c in df.columns]
    sub = df[display_cols].tail(50).copy()
    if "ml_conf" in sub.columns:
        sub["ml_conf"] = (sub["ml_conf"] * 100).round(1).astype(str) + "%"
    if "pnl" in sub.columns:
        sub["pnl"] = sub["pnl"].round(2)
    header = "".join(f"<th style='padding:9px 12px;background:#21262d;color:#8b949e;text-align:left;'>{c}</th>" for c in sub.columns)
    body = ""
    for _, row in sub.iterrows():
        pnl_val = df.loc[row.name, "pnl"] if "pnl" in df.columns else 0
        row_color = "rgba(34,197,94,0.04)" if pnl_val > 0 else "rgba(239,68,68,0.04)"
        cells = ""
        for col, val in row.items():
            c = ""
            if col == "pnl":
                c = "color:#22c55e;font-weight:600;" if float(str(val)) > 0 else "color:#ef4444;font-weight:600;"
            cells += f"<td style='padding:8px 12px;border-bottom:1px solid #21262d;{c}'>{val}</td>"
        body += f"<tr style='background:{row_color};'>{cells}</tr>"
    return f"<div style='overflow-x:auto;'><table style='width:100%;border-collapse:collapse;font-size:12px;'><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>"

# ─────────────────────────────────────────────────────────────
# BUILD HTML
# ─────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>1-Minute Paper Trading — Performance Report</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{background:#0d1117;color:#e6edf3;font-family:'Segoe UI',sans-serif;padding:28px 32px;}}
  h1{{font-size:26px;font-weight:700;margin-bottom:4px;}}
  .sub{{color:#8b949e;font-size:13px;margin-bottom:36px;}}
  .sec{{font-size:16px;font-weight:600;margin:36px 0 14px;color:#00d4aa;border-left:3px solid #00d4aa;padding-left:10px;}}
  .card{{background:#161b22;border:1px solid #21262d;border-radius:12px;padding:22px;margin-bottom:16px;}}
  .chart-card{{background:#161b22;border:1px solid #21262d;border-radius:12px;padding:16px;margin-bottom:16px;}}
  .kpi-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:14px;margin-bottom:16px;}}
  .kpi{{background:#161b22;border:1px solid #21262d;border-radius:10px;padding:18px 20px;}}
  .kpi-label{{font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px;}}
  .kpi-value{{font-size:24px;font-weight:700;}}
  .green{{color:#22c55e;}} .red{{color:#ef4444;}} .neutral{{color:#e6edf3;}} .amber{{color:#f59e0b;}}
  .grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:16px;}}
  .grid-3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;}}
  .metrics-split{{display:grid;grid-template-columns:1fr 1fr;gap:0;}}
  .badge{{display:inline-block;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:600;background:#00d4aa22;color:#00d4aa;margin-left:8px;}}
  footer{{margin-top:52px;color:#6e7681;font-size:12px;text-align:center;padding-bottom:20px;}}
  @media(max-width:768px){{.grid-2,.grid-3,.metrics-split{{grid-template-columns:1fr;}}}}
</style>
</head>
<body>

<h1>Renko + ML Paper Trading &nbsp;<span class="badge">1-Minute</span></h1>
<div class="sub">BTCUSDT &nbsp;|&nbsp; Binance &nbsp;|&nbsp; Generated: {now} &nbsp;|&nbsp; Initial Capital: $10,000</div>

<!-- KPI STRIP -->
<div class="kpi-grid">
  <div class="kpi"><div class="kpi-label">Net PnL</div><div class="kpi-value {'green' if net>=0 else 'red'}">${net:+,.2f}</div></div>
  <div class="kpi"><div class="kpi-label">Return</div><div class="kpi-value {'green' if ret>=0 else 'red'}">{ret:+.2f}%</div></div>
  <div class="kpi"><div class="kpi-label">Capital</div><div class="kpi-value neutral">${current_cap:,.2f}</div></div>
  <div class="kpi"><div class="kpi-label">Total Trades</div><div class="kpi-value neutral">{n}</div></div>
  <div class="kpi"><div class="kpi-label">Win Rate</div><div class="kpi-value {'green' if wr>=50 else 'red'}">{wr}%</div></div>
  <div class="kpi"><div class="kpi-label">Wins / Losses</div><div class="kpi-value neutral">{wins_n} / {n-wins_n}</div></div>
  <div class="kpi"><div class="kpi-label">Profit Factor</div><div class="kpi-value {'green' if metrics.get('Profit Factor',0)>=1 else 'red'}">{metrics.get('Profit Factor','—')}</div></div>
  <div class="kpi"><div class="kpi-label">Max Drawdown</div><div class="kpi-value {'amber' if metrics.get('Max Drawdown (%)',0)>0 else 'green'}">{metrics.get('Max Drawdown (%)','—')}%</div></div>
</div>

<!-- EQUITY + DRAWDOWN -->
<div class="sec">Equity and Drawdown</div>
<div class="grid-2">
  <div class="chart-card">{fig_html(fig_eq)}</div>
  <div class="chart-card">{fig_html(fig_dd)}</div>
</div>

<!-- PNL CHARTS -->
<div class="sec">Trade PnL Analysis</div>
<div class="grid-2">
  <div class="chart-card">{fig_html(fig_pnl)}</div>
  <div class="chart-card">{fig_html(fig_cum)}</div>
</div>

<!-- WIN/LOSS + DAILY -->
<div class="sec">Win/Loss and Daily Performance</div>
<div class="grid-2">
  <div class="chart-card">{fig_html(fig_pie)}</div>
  <div class="chart-card">{fig_html(fig_daily)}</div>
</div>

<!-- EXIT TYPE + SKIP REASONS -->
<div class="sec">Signal Intelligence</div>
<div class="grid-2">
  <div class="chart-card">{fig_html(fig_exit)}</div>
  <div class="chart-card">{fig_html(fig_skip)}</div>
</div>

<!-- ML CONFIDENCE -->
<div class="sec">ML Confidence Distribution</div>
<div class="chart-card">{fig_html(fig_conf)}</div>

<!-- FULL METRICS TABLE -->
<div class="sec">Full Performance Metrics</div>
<div class="card">{metrics_table_html(metrics)}</div>

<!-- TRADE LOG -->
<div class="sec">Trade Log (Last 50 Trades)</div>
<div class="card">{trade_log_html(trades)}</div>

<footer>Renko + ML Paper Trading &nbsp;|&nbsp; BTCUSDT 1-Minute &nbsp;|&nbsp; {now}</footer>
</body>
</html>
"""

with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Saved: {OUTPUT_HTML}")
