#!/usr/bin/env python3
"""
mlb_train_model.py
==================
Train calibrated logistic regression models for game-level hit and HR probability.

Each training row = one batter in one game (from mlb_data_collector.py).
Target labels are game-level: got_hit (1+ hits in game), hit_hr (1+ HR in game).

Usage:
    python mlb_train_model.py --data mlb_training_data.csv
    python mlb_train_model.py --data mlb_training_data.csv --model-dir mlb_models

Output:
    mlb_models/hit_model.joblib     calibrated pipeline
    mlb_models/hr_model.joblib      calibrated pipeline
    mlb_models/model_info.json      feature lists, means/scales, eval metrics
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import joblib

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Feature definitions ───────────────────────────────────────────────────────
# These must stay in sync with the inference code in mlb_app.py

HIT_FEATURES = [
    "xba",                  # Expected BA (Savant) — best single predictor
    "split_ba",             # Season BA vs this pitcher hand
    "barrel_pct",           # Barrel % (contact quality)
    "season_ba",            # Overall season BA
    "pitcher_avg_against",  # Pitcher BAA (season)
    "pitcher_k_per_9",      # Pitcher K/9 — suppresses hits
    "park_hit_factor",      # Park hit factor
    "lineup_spot",          # Batting order position (1=leadoff → more PA)
    "pitcher_hand_L",       # 1 = facing LHP, 0 = facing RHP
]

HR_FEATURES = [
    "xslg",                 # Expected SLG (Savant) — power proxy
    "barrel_pct",           # Barrel % — strongest HR predictor
    "exit_velo",            # Average exit velocity
    "split_hr_rate",        # Season HR/AB vs this pitcher hand
    "pitcher_hr_per_9",     # Pitcher HR/9 (season)
    "park_hr_factor",       # Park HR factor
    "lineup_spot",          # Batting order position
    "pitcher_hand_L",       # 1 = facing LHP, 0 = facing RHP
]

# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    log.info(f"Loading {path} …")
    df = pd.read_csv(path, low_memory=False)
    log.info(f"  {len(df):,} rows, {df['date'].nunique()} unique dates")

    # Encode pitcher hand
    df["pitcher_hand_L"] = (df["pitcher_hand"] == "L").astype(float)

    # Validate labels
    for col in ("got_hit", "hit_hr"):
        missing = df[col].isna().sum()
        if missing:
            log.warning(f"  {missing} rows missing label '{col}' — dropping")
            df = df.dropna(subset=[col])

    df["got_hit"] = df["got_hit"].astype(int)
    df["hit_hr"]  = df["hit_hr"].astype(int)

    log.info(f"  Hit rate: {df['got_hit'].mean():.3f}   HR rate: {df['hit_hr'].mean():.3f}")
    return df

# ── Chronological train / val / test split ────────────────────────────────────

def chrono_split(df: pd.DataFrame, val_frac=0.10, test_frac=0.10):
    """
    Split by date (not random) to prevent data leakage.
    Train on earlier games, validate + calibrate on middle games,
    evaluate on most-recent games.
    """
    df = df.sort_values("date").reset_index(drop=True)
    n  = len(df)
    i_val  = int(n * (1 - val_frac - test_frac))
    i_test = int(n * (1 - test_frac))
    train = df.iloc[:i_val]
    val   = df.iloc[i_val:i_test]
    test  = df.iloc[i_test:]
    log.info(f"  Train {len(train):,}  |  Val {len(val):,}  |  Test {len(test):,}")
    return train, val, test

# ── Model building ────────────────────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    """
    Imputer → scaler → logistic regression.
    Missing features are filled with the training-set column mean (safe for inference).
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
        ("lr",      LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                                       class_weight="balanced")),
    ])

def train_model(train: pd.DataFrame, val: pd.DataFrame,
                features: list, label: str) -> CalibratedClassifierCV:
    """
    Fit pipeline on train, calibrate with Platt scaling on val.
    Using cv='prefit' so we have one clean inner estimator for coefficient extraction.
    """
    X_train = train[features].values
    y_train = train[label].values
    X_val   = val[features].values
    y_val   = val[label].values

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    calibrated = CalibratedClassifierCV(pipe, method="sigmoid", cv="prefit")
    calibrated.fit(X_val, y_val)
    return calibrated

# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, X_test: np.ndarray, y_test: np.ndarray, label: str) -> dict:
    probs = model.predict_proba(X_test)[:, 1]
    bs    = brier_score_loss(y_test, probs)
    ll    = log_loss(y_test, probs)
    auc   = roc_auc_score(y_test, probs)

    # Calibration curve (10 bins)
    frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10)
    max_cal_err = float(np.max(np.abs(frac_pos - mean_pred)))

    log.info(f"\n── {label} ──────────────────────")
    log.info(f"  Brier score     : {bs:.4f}  (lower = better; baseline≈{y_test.mean()*(1-y_test.mean()):.4f})")
    log.info(f"  Log loss        : {ll:.4f}")
    log.info(f"  ROC-AUC         : {auc:.4f}")
    log.info(f"  Max cal. error  : {max_cal_err:.4f}")
    log.info(f"  Predicted range : [{probs.min():.3f}, {probs.max():.3f}]")
    log.info(f"  Actual pos rate : {y_test.mean():.3f}")

    return {"brier": round(bs, 4), "log_loss": round(ll, 4),
            "roc_auc": round(auc, 4), "max_cal_err": round(max_cal_err, 4)}

# ── Coefficient extraction (for waterfall chart in app) ──────────────────────

def extract_lr_info(calibrated: CalibratedClassifierCV, features: list) -> dict:
    """
    Pull the imputer means/stds and LR coefficients from inside the calibrated model.
    These are used in the app to compute per-feature contributions for the waterfall chart.
    coef_[i] is in standardised space, so contribution = coef_[i] * (x_raw - mean_) / scale_
    """
    inner_pipe = calibrated.calibrated_classifiers_[0].estimator
    imputer    = inner_pipe.named_steps["imputer"]
    scaler     = inner_pipe.named_steps["scaler"]
    lr         = inner_pipe.named_steps["lr"]

    return {
        "features":       features,
        "imputer_means":  imputer.statistics_.tolist(),
        "scaler_means":   scaler.mean_.tolist(),
        "scaler_scales":  scaler.scale_.tolist(),
        "lr_coef":        lr.coef_[0].tolist(),
        "lr_intercept":   float(lr.intercept_[0]),
    }

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train MLB hit/HR probability models.")
    parser.add_argument("--data",      required=True, help="Path to CSV from mlb_data_collector.py")
    parser.add_argument("--model-dir", default="mlb_models", help="Directory to save models")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    df = load_data(args.data)

    # ── Split ─────────────────────────────────────────────────────────────────
    log.info("\nSplitting data chronologically …")
    train, val, test = chrono_split(df)

    # ── Hit model ─────────────────────────────────────────────────────────────
    log.info("\nTraining hit model …")
    hit_model = train_model(train, val, HIT_FEATURES, "got_hit")

    hit_metrics = evaluate(
        hit_model,
        test[HIT_FEATURES].values,
        test["got_hit"].values,
        "Hit model — test set",
    )
    hit_lr_info = extract_lr_info(hit_model, HIT_FEATURES)

    log.info("\n  Hit model feature coefficients (standardised):")
    for feat, coef in zip(HIT_FEATURES, hit_lr_info["lr_coef"]):
        sign = "+" if coef >= 0 else ""
        log.info(f"    {feat:<28} {sign}{coef:.4f}")

    # ── HR model ──────────────────────────────────────────────────────────────
    log.info("\nTraining HR model …")
    hr_model = train_model(train, val, HR_FEATURES, "hit_hr")

    hr_metrics = evaluate(
        hr_model,
        test[HR_FEATURES].values,
        test["hit_hr"].values,
        "HR model — test set",
    )
    hr_lr_info = extract_lr_info(hr_model, HR_FEATURES)

    log.info("\n  HR model feature coefficients (standardised):")
    for feat, coef in zip(HR_FEATURES, hr_lr_info["lr_coef"]):
        sign = "+" if coef >= 0 else ""
        log.info(f"    {feat:<28} {sign}{coef:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    log.info("\nSaving models …")
    joblib.dump(hit_model, model_dir / "hit_model.joblib")
    joblib.dump(hr_model,  model_dir / "hr_model.joblib")

    model_info = {
        "trained_date":   datetime.utcnow().isoformat() + "Z",
        "n_train_rows":   len(train),
        "n_val_rows":     len(val),
        "n_test_rows":    len(test),
        "hit_features":   HIT_FEATURES,
        "hr_features":    HR_FEATURES,
        "hit_metrics":    hit_metrics,
        "hr_metrics":     hr_metrics,
        "hit_lr_info":    hit_lr_info,
        "hr_lr_info":     hr_lr_info,
    }
    (model_dir / "model_info.json").write_text(json.dumps(model_info, indent=2))

    log.info(f"\nSaved to {model_dir}/")
    log.info("  hit_model.joblib")
    log.info("  hr_model.joblib")
    log.info("  model_info.json")
    log.info("\nDeploy: copy mlb_models/ next to mlb_app.py and restart Streamlit.")

if __name__ == "__main__":
    main()
