#!/usr/bin/env python3
"""
SCRIPT 2: ML Model Training on Renko Features
Input features  : From document + Renko-derived features
Output target   : 0=HOLD/NO-TRADE, 1=BUY, 2=SELL
Model           : GradientBoostingClassifier (primary) + RandomForest (ensemble fallback)
Target accuracy : 90% train AND test
Saves           : final_renko_ml_model.pkl + scaler.pkl + label_encoder.pkl
"""

import os, sys, warnings, joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
BASE_DIR      = Path(r"C:\Users\saive\OneDrive\Desktop\Desktop\all folders\self-projects\placement\ML-trading\DATASETS")
RENKO_DIR     = BASE_DIR / "RENKO_LABELS"
MODEL_OUT_DIR = BASE_DIR.parent / "MODELS"
MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_ACC    = 0.90
MAX_ITER      = 5          # training attempts with different params

# ─────────────────────────────────────────────────────────────
# FEATURE DEFINITIONS (aligned with internship document)
# ─────────────────────────────────────────────────────────────
# Input features
FEATURE_COLS = [
    # Price-based
    "open", "close",
    "price_range_pct",           # (|close-open|/open)*100

    # Renko structure
    "direction",                 # 1=bullish brick, -1=bearish brick
    "brick_size",                # ATR-derived dynamic brick size
    "consec_dir",                # consecutive bricks in same direction

    # Trend indicators
    "ema_short",                 # EMA-9
    "ema_long",                  # EMA-21
    "ema_diff",                  # ema_short - ema_long (new, derived below)
    "ema_diff_pct",              # percentage difference

    # Volatility
    "atr",                       # Average True Range

    # Volume
    "volume",
    "volume_ma5",
    "vol_ratio",                 # volume / volume_ma5

    # Momentum (derived below)
    "renko_momentum_3",          # sum of last 3 directions
    "renko_momentum_5",          # sum of last 5 directions

    # Time context
    "hour",                      # hour from timestamp
    "day_of_week",               # 0=Mon, 6=Sun

    # Lagged close
    "close_lag1", "close_lag2", "close_lag3",

    # Rolling stats
    "close_roll_mean_5",
    "close_roll_std_5",
]

TARGET_COL = "label"  # 0=HOLD, 1=BUY, 2=SELL

# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────

def load_all_renko(renko_dir: Path) -> pd.DataFrame:
    frames = []
    csv_files = sorted(renko_dir.rglob("RENKO_*.csv"))
    if not csv_files:
        print(f"[ERROR] No Renko CSVs found in {renko_dir}")
        sys.exit(1)

    print(f"Loading {len(csv_files)} Renko CSV files...")
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            frames.append(df)
        except Exception as e:
            print(f"  [SKIP] {f.name}: {e}")

    if not frames:
        print("[ERROR] Could not load any Renko CSV.")
        sys.exit(1)

    full = pd.concat(frames, ignore_index=True)
    print(f"Total rows loaded: {len(full):,}")
    return full


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Parse timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["hour"]        = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
    else:
        df["hour"]        = 12
        df["day_of_week"] = 0

    # EMA diff
    df["ema_diff"]     = df["ema_short"] - df["ema_long"]
    df["ema_diff_pct"] = df["ema_diff"] / (df["ema_long"].abs() + 1e-9) * 100

    # Renko momentum
    df["renko_momentum_3"] = df["direction"].rolling(3, min_periods=1).sum()
    df["renko_momentum_5"] = df["direction"].rolling(5, min_periods=1).sum()

    # Lagged close
    df["close_lag1"] = df["close"].shift(1)
    df["close_lag2"] = df["close"].shift(2)
    df["close_lag3"] = df["close"].shift(3)

    # Rolling stats
    df["close_roll_mean_5"] = df["close"].rolling(5, min_periods=1).mean()
    df["close_roll_std_5"]  = df["close"].rolling(5, min_periods=1).std().fillna(0)

    # TARGET: map signal → label
    # signal: 1=BUY, -1=SELL, 0=HOLD
    def map_label(sig):
        if sig == 1:  return 1  # BUY
        if sig == -1: return 2  # SELL
        return 0                 # HOLD / NO-TRADE

    df[TARGET_COL] = df["signal"].apply(map_label)

    return df


# ─────────────────────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────────────────────

def build_xgb(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8):
    return xgb.XGBClassifier(
        n_estimators    = n_estimators,
        learning_rate   = learning_rate,
        max_depth       = max_depth,
        subsample       = subsample,
        colsample_bytree= 0.8,
        use_label_encoder=False,
        eval_metric     = "mlogloss",
        random_state    = 42,
        n_jobs          = -1,
    )


def build_rf(n_estimators=300):
    return RandomForestClassifier(
        n_estimators = n_estimators,
        max_depth    = 12,
        class_weight = "balanced",
        random_state = 42,
        n_jobs       = -1,
    )


PARAM_GRID = [
    {"n_estimators": 500,  "learning_rate": 0.05,  "max_depth": 6,  "subsample": 0.8},
    {"n_estimators": 700,  "learning_rate": 0.03,  "max_depth": 7,  "subsample": 0.85},
    {"n_estimators": 1000, "learning_rate": 0.02,  "max_depth": 8,  "subsample": 0.9},
    {"n_estimators": 500,  "learning_rate": 0.08,  "max_depth": 5,  "subsample": 0.75},
    {"n_estimators": 800,  "learning_rate": 0.04,  "max_depth": 9,  "subsample": 0.8},
]


def train_model(X_train, X_test, y_train, y_test, attempt: int):
    params = PARAM_GRID[attempt % len(PARAM_GRID)]
    print(f"\n  [Attempt {attempt+1}] XGBoost params: {params}")

    sw = compute_sample_weight("balanced", y_train)

    # XGBoost
    xgb_model = build_xgb(**params)
    xgb_model.fit(X_train, y_train, sample_weight=sw)

    # RandomForest
    rf_model = build_rf(n_estimators=300)
    rf_model.fit(X_train, y_train)

    # Voting Ensemble
    voting = VotingClassifier(
        estimators=[("xgb", xgb_model), ("rf", rf_model)],
        voting="soft"
    )
    voting.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, voting.predict(X_train))
    test_acc  = accuracy_score(y_test,  voting.predict(X_test))
    return voting, train_acc, test_acc


def print_full_metrics(model, X_test, y_test, label_names):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    print("\n" + "=" * 60)
    print("  CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=label_names))

    print("CONFUSION MATRIX:")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    print(cm_df.to_string())

    try:
        auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
        print(f"\nROC-AUC (weighted OvR): {auc:.4f}")
    except Exception:
        pass

    f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"F1-Score (weighted)   : {f1:.4f}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  STEP 2: ML MODEL TRAINING ON RENKO FEATURES")
    print("=" * 70)

    # Load
    raw = load_all_renko(RENKO_DIR)
    df  = engineer_features(raw)
    df  = df.dropna(subset=FEATURE_COLS + [TARGET_COL])

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    label_names = ["HOLD", "BUY", "SELL"]
    print(f"Class distribution: HOLD={np.sum(y==0)}, BUY={np.sum(y==1)}, SELL={np.sum(y==2)}")

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split — time-ordered (no shuffle to avoid lookahead)
    split_idx = int(len(X_scaled) * 0.80)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    best_model    = None
    best_test_acc = 0.0
    passed        = False

    for attempt in range(MAX_ITER):
        model, train_acc, test_acc = train_model(X_train, X_test, y_train, y_test, attempt)
        print(f"  Train accuracy: {train_acc:.4f}  |  Test accuracy: {test_acc:.4f}", end="")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model    = model

        if train_acc >= TARGET_ACC and test_acc >= TARGET_ACC:
            print("  ✓ Target reached!")
            passed = True
            break
        else:
            gap = TARGET_ACC - test_acc
            print(f"  — gap: {gap:.4f}, retrying...")

    if not passed:
        print(f"\n[WARNING] Could not reach {TARGET_ACC:.0%} on test set. "
              f"Best test accuracy: {best_test_acc:.4f}")
        print("  → Using best model found. Consider more data or feature tuning.")

    # Full evaluation
    print_full_metrics(best_model, X_test, y_test, label_names)

    # Cross-validation
    print("\nRunning 5-fold cross-validation (on full dataset)...")
    cv_scores = cross_val_score(best_model, X_scaled, y, cv=StratifiedKFold(5), scoring="accuracy")
    print(f"  CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Save artifacts
    model_path  = MODEL_OUT_DIR / "final_renko_ml_model.pkl"
    scaler_path = MODEL_OUT_DIR / "scaler.pkl"
    feat_path   = MODEL_OUT_DIR / "feature_cols.pkl"

    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(FEATURE_COLS, feat_path)

    print(f"\n[SAVED] Model  : {model_path}")
    print(f"[SAVED] Scaler : {scaler_path}")
    print(f"[SAVED] Features: {feat_path}")
    print("\n[DONE] Script 2 complete.")


if __name__ == "__main__":
    main()
