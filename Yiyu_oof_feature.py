# -*- coding: utf-8 -*-
# ============================================================
# Wildfire | CatBoost OOF + History-safe Feature Engineering
# Pipeline: data input → feature engineering → CatBoost OOF → OOF-derived → export
# ============================================================

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, roc_auc_score
from catboost import CatBoostClassifier

# -----------------------------
# Path config
# -----------------------------
CSV_PATH  = "/Users/liuliuyiyu/Documents/senior/first quarter/sta160/Wildfire_CLEANED.csv"
OUT_PATH  = os.path.splitext(CSV_PATH)[0] + "_WITH_OOF.csv"  

# -----------------------------
# Load & basic cleaning
# -----------------------------
df = pd.read_csv(CSV_PATH)

# Normalize labels to 0/1
df["Wildfire"] = (
    df["Wildfire"].astype(str).str.strip().str.title()
      .map({"Yes": 1, "No": 0}).fillna(0).astype("int8")
)

# Time keys & sort (ensures "past-only" operations)
df["datetime"] = pd.to_datetime(df["datetime"])
df["year"]     = df["datetime"].dt.year
df = df.sort_values(["latitude","longitude","datetime"]).reset_index(drop=True)

# Coarse spatial bins (reduce overfitting/leakage on exact coordinates)
df["lat_bin"] = df["latitude"].round(1)
df["lon_bin"] = df["longitude"].round(1)
df["geo_id"]  = df["lat_bin"].astype(str) + "_" + df["lon_bin"].astype(str)

# Month features (cyclical encoding)
if "month" not in df.columns:
    df["month"] = df["datetime"].dt.month
df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
df["month_cos"] = np.cos(2*np.pi*df["month"]/12)

# Latitude/longitude trig transforms
df["lat_sin"] = np.sin(np.deg2rad(df["latitude"]))
df["lat_cos"] = np.cos(np.deg2rad(df["latitude"]))
df["lon_sin"] = np.sin(np.deg2rad(df["longitude"]))
df["lon_cos"] = np.cos(np.deg2rad(df["longitude"]))

# ------------------------------------------------------------
# History-safe feature engineering (uses only past information)
# ------------------------------------------------------------
group_cols = ["lat_bin", "lon_bin"]

def make_lag(df, cols, lags=(1,)):
    # Group-wise lagged features
    df = df.copy()
    for c in cols:
        g = df.groupby(group_cols)[c]
        for k in lags:
            df[f"{c}_lag{k}"] = g.shift(k).astype("float32")
    return df

def make_roll(df, cols_roll_mean_7=(), cols_roll_max_7=(), cols_roll_sum_7=(),
              cols_roll_mean_14=(), cols_roll_max_14=(), cols_roll_sum_14=()):
    # Rolling stats; shift(1) ensures no look-ahead
    df = df.copy()
    for c in cols_roll_mean_7:
        df[f"{c}_mean_7"] = df.groupby(group_cols)[c].shift(1).rolling(7, min_periods=1).mean()
    for c in cols_roll_max_7:
        df[f"{c}_max_7"]  = df.groupby(group_cols)[c].shift(1).rolling(7, min_periods=1).max()
    for c in cols_roll_sum_7:
        df[f"{c}_sum_7"]  = df.groupby(group_cols)[c].shift(1).rolling(7, min_periods=1).sum()

    for c in cols_roll_mean_14:
        df[f"{c}_mean_14"] = df.groupby(group_cols)[c].shift(1).rolling(14, min_periods=1).mean()
    for c in cols_roll_max_14:
        df[f"{c}_max_14"]  = df.groupby(group_cols)[c].shift(1).rolling(14, min_periods=1).max()
    for c in cols_roll_sum_14:
        df[f"{c}_sum_14"]  = df.groupby(group_cols)[c].shift(1).rolling(14, min_periods=1).sum()
    return df

def make_slope(df, cols, win=14):
    # Rolling linear trend (slope) with window=win; requires ≥3 values
    df = df.copy()
    for c in cols:
        def _slope(x):
            if np.sum(~np.isnan(x)) < 3:
                return np.nan
            y = np.array(x, dtype="float32")
            x_idx = np.arange(len(y), dtype="float32")
            mask = ~np.isnan(y)
            return np.polyfit(x_idx[mask], y[mask], 1)[0]
        df[f"{c}_slope_{win}"] = (
            df.groupby(group_cols)[c]
              .shift(1).rolling(win, min_periods=3).apply(_slope, raw=False)
        )
    return df

def make_climatology_anomaly(df, cols):
    # (grid, month) climatology → z-score anomaly
    df = df.copy()
    key = ["lat_bin","lon_bin","month"]
    for c in cols:
        stat = df.groupby(key)[c].agg(["mean","std"]).rename(columns={"mean":f"{c}_clim_mu","std":f"{c}_clim_sd"})
        df = df.join(stat, on=key)
        df[f"{c}_z"] = (df[c] - df[f"{c}_clim_mu"]) / (df[f"{c}_clim_sd"].replace(0, np.nan))
        df.drop(columns=[f"{c}_clim_mu", f"{c}_clim_sd"], inplace=True)
    return df

def make_fire_memory(df):
    # Fire recency & short-term frequency (past-only)
    df = df.copy()
    def _days_since(group):
        g = group.sort_values("datetime").reset_index()
        t = np.arange(len(g))
        fire_t = np.where(g["Wildfire"].values==1, t, np.nan)
        last_fire_t = pd.Series(fire_t).ffill().shift(1)
        dsf = t - last_fire_t
        return pd.Series(dsf.values, index=g["index"])
    df["days_since_fire"] = df.groupby(group_cols, group_keys=False).apply(_days_since)
    df["fire_count_30"] = df.groupby(group_cols)["Wildfire"].shift(1).rolling(30, min_periods=1).sum()
    return df

def make_historic_risk_priors(df):
    # Expanding-mean priors up to yesterday; backoff to global mean
    df = df.copy()
    for key, name in [(group_cols, "geo_risk_hist"),
                      (["region"], "region_risk_hist"),
                      (["lat_zone"], "lat_zone_risk_hist")]:
        g = df.groupby(key)["Wildfire"]
        cum_sum   = g.shift(1).cumsum()
        cum_count = g.cumcount()
        prior = cum_sum / np.where(cum_count>0, cum_count, np.nan)
        overall = df["Wildfire"].mean()
        df[name] = prior.fillna(overall).astype("float32")
    return df

# Variables to lag/roll (subset to existing columns)
met_vars = [c for c in ["pr","vpd","tmmx_c","tmmn_c","srad","vs","etr","pet","fm100","fm1000","erc"] if c in df.columns]

df = make_lag(df, met_vars, lags=(1,))  # lag1
df = make_roll(
    df,
    cols_roll_mean_7 = [c for c in ["vpd","tmmx_c","tmmn_c"] if c in df.columns],
    cols_roll_max_7  = [c for c in ["vpd","tmmx_c"] if c in df.columns],
    cols_roll_sum_7  = [c for c in ["pr"] if c in df.columns],
    cols_roll_mean_14= [c for c in ["vpd","tmmx_c"] if c in df.columns],
    cols_roll_max_14 = [c for c in ["vpd"] if c in df.columns],
    cols_roll_sum_14 = [c for c in ["pr"] if c in df.columns],
)
df = make_slope(df, [c for c in ["vpd","tmmx_c"] if c in df.columns], win=14)
df = make_climatology_anomaly(df, [c for c in ["vpd","tmmx_c","vs"] if c in df.columns])
df = make_fire_memory(df)
df = make_historic_risk_priors(df)

# Forward-fill within groups for early missing values (still past-only)
df[met_vars] = df[met_vars].astype("float32")
for col in df.columns:
    if col.endswith("_lag1") or any(col.endswith(suf) for suf in ["_mean_7","_max_7","_sum_7","_mean_14","_max_14","_sum_14","_slope_14"]):
        df[col] = df.groupby(group_cols)[col].ffill()

# ------------------------------------------------------------
# CatBoost OOF (expanding-window by year; no future leakage)
# ------------------------------------------------------------
# Feature set for CatBoost (must exclude any "future" info)
numeric_features = []
cand = [
    # Raw meteorology
    "pr","rmax","rmin","sph","srad","vs","etr","pet","vpd","tmmn_c","tmmx_c",
    # Time/geo
    "month","month_sin","month_cos","lat_sin","lat_cos","lon_sin","lon_cos",
    # Historic priors
    "geo_risk_hist","region_risk_hist","lat_zone_risk_hist",
    # lag/rolling/slope
    "pr_lag1","vpd_lag1","tmmx_c_lag1",
    "vpd_mean_7","tmmx_c_mean_7","tmmn_c_mean_7",
    "vpd_max_7","tmmx_c_max_7","pr_sum_7",
    "vpd_mean_14","tmmx_c_mean_14","pr_sum_14","vpd_max_14",
    "tmmx_c_slope_14","vpd_slope_14",
    # Anomaly & memory
    "vpd_z","tmmx_c_z","vs_z","days_since_fire","fire_count_30",
]
numeric_features = [c for c in cand if c in df.columns]

categorical_features = [c for c in ["region","lat_zone","season"] if c in df.columns]
feat_cols = numeric_features + categorical_features

X_all = df[feat_cols].copy()
y_all = df["Wildfire"].astype(int).copy()
cat_idx = [X_all.columns.get_loc(c) for c in categorical_features]

years = sorted(df["year"].unique())
oof = pd.Series(np.nan, index=df.index, dtype="float32")

print(f"[Info] Building OOF by expanding years: {years}")
for k in range(1, len(years)):
    train_years = years[:k]
    valid_year  = years[k]
    tr_idx = df.index[df["year"].isin(train_years)]
    va_idx = df.index[df["year"]==valid_year]

    if len(va_idx)==0 or len(tr_idx)==0:
        continue

    cb = CatBoostClassifier(
        iterations=1200,
        depth=8,
        learning_rate=0.05,
        l2_leaf_reg=3,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        class_weights=[1.0, 3.0],  # 稀有事件加权
        od_type="Iter",
        od_wait=40,
        verbose=False
    )
    cb.fit(X_all.loc[tr_idx], y_all.loc[tr_idx],
           eval_set=(X_all.loc[va_idx], y_all.loc[va_idx]),
           cat_features=cat_idx, use_best_model=True)

    oof.loc[va_idx] = cb.predict_proba(X_all.loc[va_idx])[:, 1].astype("float32")
    print(f"  Fold year={valid_year}: AUC={roc_auc_score(y_all.loc[va_idx], oof.loc[va_idx]):.4f} (valid)")

df["p_cb_oof"] = oof
# Remove rows with no OOF (earliest years) or keep as NaN (here we drop)
df = df.dropna(subset=["p_cb_oof"]).reset_index(drop=True)

# ------------------------------------------------------------
# OOF-derived features + hard-example weights (for RNN)
# ------------------------------------------------------------
eps = 1e-6
df["logit_cb"] = np.log(np.clip(df["p_cb_oof"], eps, 1-eps) / np.clip(1-df["p_cb_oof"], eps, 1-eps))

# Per-day percentile rank of OOF (relative risk within day)
df["rank_cb_day"] = df.groupby("datetime")["p_cb_oof"].rank(pct=True).astype("float32")

# Temporal dynamics of p_cb (past-only)
for k in range(1, 8):
    df[f"p_cb_lag{k}"] = df.groupby(group_cols)["p_cb_oof"].shift(k)
df["p_cb_roll7_mean"] = df.groupby(group_cols)["p_cb_oof"].shift(1).rolling(7, min_periods=1).mean()
df["p_cb_roll7_max"]  = df.groupby(group_cols)["p_cb_oof"].shift(1).rolling(7, min_periods=1).max()

# Threshold via best-F1 to label CB errors (can switch to fixed-FPR if desired)
prec, rec, thr = precision_recall_curve(df["Wildfire"], df["p_cb_oof"])
f1 = 2*prec*rec/(prec+rec+1e-9)
best_thr = thr[np.nanargmax(f1)] if len(thr)>0 else 0.5
df["cb_pred_bin"] = (df["p_cb_oof"] >= best_thr).astype("int8")

df["cb_error_type"] = "TN"
df.loc[(df["Wildfire"]==1)&(df["cb_pred_bin"]==1), "cb_error_type"] = "TP"
df.loc[(df["Wildfire"]==0)&(df["cb_pred_bin"]==1), "cb_error_type"] = "FP"
df.loc[(df["Wildfire"]==1)&(df["cb_pred_bin"]==0), "cb_error_type"] = "FN"

# Sample weights for RNN: emphasize CB mistakes
w = np.ones(len(df), dtype="float32")
w[df["cb_error_type"]=="FN"] = 3.0
w[df["cb_error_type"]=="FP"] = 2.0
df["rnn_weight"] = w

# One-hot error types (optional extra channels)
df = pd.get_dummies(df, columns=["cb_error_type"], prefix="cb")

# ------------------------------------------------------------
# Export dataset with OOF & derived features
# ------------------------------------------------------------
extra_cols = [
    "p_cb_oof","logit_cb","rank_cb_day",
    "p_cb_lag1","p_cb_lag2","p_cb_lag3","p_cb_lag4","p_cb_lag5","p_cb_lag6","p_cb_lag7",
    "p_cb_roll7_mean","p_cb_roll7_max",
    "rnn_weight"
] + [c for c in df.columns if c.startswith("cb_")]  # cb_error_type_* one-hot

base_keep = [
    "latitude","longitude","datetime","Wildfire","region","lat_zone","season",
    "month","lat_bin","lon_bin","geo_id"
]

# Final column set = base + model features + OOF-derived
final_cols = sorted(set(base_keep + feat_cols + extra_cols))
df_final = df[final_cols].copy()

print(f"[Info] Final rows: {df_final.shape[0]}, cols: {df_final.shape[1]}")
print(f"[Info] Saving to: {OUT_PATH}")
df_final.to_csv(OUT_PATH, index=False)

# Summary
print("\n=== Added OOF columns ===")
print([c for c in extra_cols if c in df_final.columns])

print("\nSample head:")
print(df_final.head(3))
