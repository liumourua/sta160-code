# -*- coding: utf-8 -*-
"""
Post-hoc reporting using existing GRU & Hazard outputs
- Read-only: consume CSV/Excel already produced by your GRU & Hazard workflows
- Produce both per-region and global summary figures/tables
- Normalize region label for display: map 'South' -> 'Coast'
"""

import os, re, glob, math, json, warnings
from collections import defaultdict
from typing import Dict, Tuple, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_recall_curve, roc_curve,
    average_precision_score, roc_auc_score, brier_score_loss,
    confusion_matrix
)

# ─────────────── Config ───────────────
GRU_PRED_DIR     = r"D:\桌面\大学\大四\第一学期\sta160\project\data2\gru"
GRU_PERREGION_XLS_DIR =  r"D:\桌面\大学\大四\第一学期\sta160\project\data2\gru\excel"  # optional
HAZARD_CSV_DIR   = r"D:\桌面\大学\大四\第一学期\sta160\project\data2\hazard"

SITE_EXPORT_ROOT = r"D:\桌面\大学\大四\第一学期\sta160\project\plot2"

# Representative region selection (based on positives for with_prior & k=14 test set)
N_REP_HIGH, N_REP_MID, N_REP_LOW = 5, 4, 4

# Budget curve settings
BUDGET_GRID  = np.linspace(0.0, 0.10, 21)       # coverage 0%–10%
FIXED_PRECISION_TARGET = 0.30                   # prefer fixed precision
TOP_K_PER_MILLE_FALLBACK = 1.0                  # else Top-1‰
K_FOR_BUDGET = 14

# PR/Calibration binning
CALIBRATION_BINS = 15

# Normalize region labels for display/tables
REGION_LABEL_MAP = {"South": "Coast"}  # others unchanged

# Lenient filename parsing for predictions
PRED_NAME_RE = re.compile(r"(?P<tag>.+?)_(?P<model>GRU_(?:with|no)_prior|prior_only?)_k(?P<k>\d+)_", re.I)

# ─────────────── Utils ───────────────
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def ensure_dirs(*paths: str):
    for p in paths:
        ensure_dir(p)

def safe_filename(s: str, max_len=140) -> str:
    s = re.sub(r'[<>:"/\\|?*]+', '_', s)
    s = re.sub(r'\s+', '_', s).strip('._')
    return s[:max_len] if s else "x"

def map_region_label(x: str) -> str:
    return REGION_LABEL_MAP.get(x, x)

def parse_tag_from_row(region: str, lat_zone: str) -> str:
    return f"{map_region_label(str(region))}_{str(lat_zone)}".replace(" ", "")

def write_table_both(df: pd.DataFrame, csv_path: str):
    """Write both CSV and Excel side-by-side."""
    ensure_dir(os.path.dirname(csv_path))
    xlsx_path = os.path.splitext(csv_path)[0] + ".xlsx"
    df.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl", mode="w") as xw:
        df.to_excel(xw, sheet_name="table", index=False)

def normalized_ap(y_true, y_prob):
    ap = average_precision_score(y_true, y_prob)
    base = float(np.mean(y_true))
    nap = 0.0 if base >= 1.0 else (ap - base) / max(1 - base, 1e-12)
    return ap, nap, base

def threshold_sweep(y_true, y_prob, n=CALIBRATION_BINS*2+1):
    ths = np.linspace(0, 1, n)
    rows=[]
    for t in ths:
        pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0,1]).ravel()
        prec = tp / max(tp+fp, 1)
        rec  = tp / max(tp+fn, 1)
        f1   = (2*prec*rec)/max(prec+rec, 1e-12)
        acc  = (tp+tn)/max(tn+fp+fn+tp,1)
        tpr  = rec
        tnr  = tn / max(tn+fp,1)
        fpr  = fp / max(fp+tn,1)
        rows.append(dict(threshold=t, tp=tp, fp=fp, fn=fn, tn=tn,
                         precision=prec, recall=rec, f1=f1, accuracy=acc,
                         tpr=tpr, tnr=tnr, fpr=fpr))
    return pd.DataFrame(rows)

def precision_at_topk(y_true, y_prob, frac):
    n = len(y_true)
    k = max(1, int(round(n * frac)))
    idx = np.argpartition(y_prob, -k)[-k:]
    return float(np.mean(y_true[idx]))

def budget_hits_curve(y_true, y_prob, covers):
    order = np.argsort(-y_prob); y_sorted = y_true[order]
    n = len(y_true); pos_total = max(1, int(y_true.sum()))
    hits=[]; recs=[]; precs=[]
    for c in covers:
        k = max(1,int(round(c*n))); sel = y_sorted[:k]
        h = int(sel.sum()); r = h/pos_total; p = h/max(k,1)
        hits.append(h); recs.append(r); precs.append(p)
    return np.array(hits), np.array(recs), np.array(precs)

def pr_roc_calibration_and_save(y_true, y_prob, title, out_prefix):
    # PR / ROC / Reliability + histogram + gain/lift
    ap, nap, base = normalized_ap(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    brier   = brier_score_loss(y_true, y_prob)

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    fpr, tpr, _  = roc_curve(y_true, y_prob)

    # reliability via equal-width bins
    bins = np.linspace(0,1,CALIBRATION_BINS+1)
    inds = np.digitize(y_prob, bins) - 1
    df_cal = pd.DataFrame({"bin":inds, "y":y_true, "p":y_prob})
    cal = df_cal.groupby("bin").agg(obs=("y","mean"), pred=("p","mean")).reset_index()
    cal = cal.dropna()

    # 3-panel summary
    fig, axes = plt.subplots(1,3, figsize=(16,4))

    # PR
    axes[0].plot(rec, prec)
    axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision")
    axes[0].set_title(f"PR | AP={ap:.3f} | nAP={nap:.3f} | base={base:.3f}")

    # ROC
    axes[1].plot(fpr, tpr); axes[1].plot([0,1],[0,1],'--', alpha=0.6)
    axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
    axes[1].set_title(f"ROC | AUC={roc_auc:.3f} | Brier={brier:.3f}")

    # Reliability
    axes[2].plot(cal["pred"], cal["obs"], marker="o")
    axes[2].plot([0,1], [0,1], '--', alpha=0.6)
    axes[2].set_xlabel("Predicted prob"); axes[2].set_ylabel("Observed freq")
    axes[2].set_title("Reliability")

    plt.suptitle(title, y=1.04, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_prefix + "_pr_roc_reliability.png", dpi=150)
    plt.close()

    # Score histogram
    plt.figure(figsize=(6,4))
    plt.hist(y_prob[y_true==0], bins=40, range=(0,1), alpha=0.6, label='Neg')
    plt.hist(y_prob[y_true==1], bins=40, range=(0,1), alpha=0.6, label='Pos')
    plt.xlabel("Predicted probability"); plt.ylabel("Count")
    plt.title(f"Score distribution — {title}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_prefix + "_score_hist.png", dpi=150)
    plt.close()

    # Gain & Lift
    order = np.argsort(-y_prob); y_sorted = y_true[order]
    cov = np.arange(1, len(y_true)+1) / len(y_true)
    rec2 = np.cumsum(y_sorted) / max(y_true.sum(), 1)
    lift = np.divide(rec2, np.maximum(cov, 1e-12))

    fig, ax = plt.subplots(1,2, figsize=(12,4))
    ax[0].plot(cov, rec2); ax[0].plot([0,1],[0,1],'--', alpha=0.6)
    ax[0].set_xlabel("Coverage"); ax[0].set_ylabel("Cumulative recall")
    ax[0].set_title("Cumulative Gain")
    ax[1].plot(cov, lift); ax[1].axhline(1.0, ls='--', alpha=0.6)
    ax[1].set_xlabel("Coverage"); ax[1].set_ylabel("Lift")
    ax[1].set_title("Lift curve")
    plt.tight_layout()
    plt.savefig(out_prefix + "_gain_lift.png", dpi=150)
    plt.close()

    # Threshold sweep + metrics
    df_th = threshold_sweep(y_true, y_prob)
    metr  = dict(AP=ap, nAP=nap, Base=base, ROC_AUC=roc_auc, Brier=brier)
    return metr, df_th

# ─────────────── Load GRU predictions ───────────────
def parse_pred_filename(path: str) -> Tuple[str, str, int]:
    base = os.path.basename(path)
    m = PRED_NAME_RE.search(base)
    if not m:
        df = pd.read_csv(path, nrows=1)
        model = str(df.get("model", [""])[0])
        k = int(df.get("k_days", [0])[0])
        tag = os.path.splitext(base)[0]
        return tag, model, k
    return m.group("tag"), m.group("model"), int(m.group("k"))

def load_all_gru_predictions(pred_dir: str) -> Dict[Tuple[str,str,int,str], Dict[str,Any]]:
    store = {}
    for path in glob.glob(os.path.join(pred_dir, "*.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        tag, model, k = parse_pred_filename(path)
        region = df.get("region", pd.Series([None])).iloc[0]
        latz   = df.get("lat_zone", pd.Series([None])).iloc[0]
        if region is None or pd.isna(region):
            parts = tag.split("_")
            if len(parts)>=2:
                region = parts[0]; latz = "_".join(parts[1:])
            else:
                region, latz = "Unknown", "Unknown"
        region = map_region_label(str(region))
        key = (region, str(latz), int(df.get("k_days", [k])[0]), str(df.get("model", [model])[0]))

        y = df["y_true"].to_numpy(np.float32)
        p_uncal = df["p_uncal"].to_numpy(np.float32) if "p_uncal" in df.columns else None
        p_cal   = df["p_cal"].to_numpy(np.float32)   if "p_cal"   in df.columns else None
        store[key] = dict(y=y, p_cal=p_cal, p_uncal=p_uncal, meta=df)
    return store

# ─────────────── Load Hazard CSVs ───────────────
def scan_hazard_curves(hazard_csv_dir: str):
    curves = {}
    waits  = {}
    trends = {}
    for path in glob.glob(os.path.join(hazard_csv_dir, "hazard_curve_*.csv")):
        tag = os.path.splitext(os.path.basename(path))[0].replace("hazard_curve_","")
        df = pd.read_csv(path)
        curves[tag] = df
    for path in glob.glob(os.path.join(hazard_csv_dir, "waiting_intervals_*.csv")):
        tag = os.path.splitext(os.path.basename(path))[0].replace("waiting_intervals_","")
        df = pd.read_csv(path)
        waits[tag] = df
    for path in glob.glob(os.path.join(hazard_csv_dir, "waiting_trend_*.csv")):
        tag = os.path.splitext(os.path.basename(path))[0].replace("waiting_trend_","")
        df = pd.read_csv(path)
        trends[tag] = df
    return curves, waits, trends

# ─────────────── Helpers for cross-region metrics & tiers ───────────────
def region_metrics_table(pred_store, model="GRU_with_prior", k=14, use_calibrated=True):
    """Aggregate by region×lat_zone: AP, ROC-AUC, positives, total, base rate."""
    rows=[]
    for (reg,lz,kk,mm), d in pred_store.items():
        if kk!=k or str(mm).lower()!=model.lower():
            continue
        y = d["y"]
        p = d["p_cal"] if (use_calibrated and d["p_cal"] is not None) else (
            d["p_uncal"] if d["p_uncal"] is not None else None
        )
        if p is None: 
            continue
        ap = average_precision_score(y,p)
        roc = roc_auc_score(y,p)
        base = float(np.mean(y))
        rows.append(dict(region=reg, lat_zone=lz, k=k, model=mm,
                         pos=int(y.sum()), n=len(y), base=base,
                         PR_AUC_cal=ap, ROC_AUC_cal=roc))
    df = pd.DataFrame(rows)
    return df

def split_tiers_by_activity(df_metrics, counts=(4,3,3)):
    """Label HIGH/MID/LOW tiers by descending positives (pos)."""
    if df_metrics.empty:
        return df_metrics
    df = df_metrics.sort_values("pos", ascending=False).reset_index(drop=True)
    h,m,l = counts
    df["tier"] = None
    df.loc[:h-1, "tier"] = "HIGH"
    df.loc[h:h+m-1, "tier"] = "MID"
    df.loc[h+m:h+m+l-1, "tier"] = "LOW"
    return df.dropna(subset=["tier"]).reset_index(drop=True)

def best_in_each_tier(df_tier, metric="PR_AUC_cal"):
    reps=[]
    for t in ["HIGH","MID","LOW"]:
        sub = df_tier[df_tier["tier"]==t]
        if sub.empty: 
            continue
        reps.append(sub.sort_values(metric, ascending=False).iloc[0])
    return pd.DataFrame(reps)

# ─────────────── Global paired plots ───────────────
def paired_plots_and_tables(pred_store, out_dir_global):
    figs_dir = os.path.join(out_dir_global,"figs")
    tabs_dir = os.path.join(out_dir_global,"tables")
    ensure_dirs(figs_dir, tabs_dir)

    def auc_of(key):
        d = pred_store.get(key)
        if not d: return np.nan
        p = d["p_cal"] if d["p_cal"] is not None else d["p_uncal"]
        return roc_auc_score(d["y"], p)

    # k=7 vs k=14 (with_prior)
    xs=[]; ys=[]; labels=[]
    for (reg,lz,_,_), _ in pred_store.items():
        k7  = (reg,lz,7,"GRU_with_prior")
        k14 = (reg,lz,14,"GRU_with_prior")
        if (k7 in pred_store) and (k14 in pred_store):
            xs.append(auc_of(k7)); ys.append(auc_of(k14)); labels.append(f"{reg}_{lz}")
    if xs:
        xs=np.array(xs, float); ys=np.array(ys, float)
        plt.figure(figsize=(6,6)); plt.scatter(xs,ys,alpha=0.7)
        lim=[min(np.nanmin(xs),np.nanmin(ys)), max(np.nanmax(xs),np.nanmax(ys))]
        plt.plot(lim,lim,'--',alpha=0.6); plt.xlabel("AUC (k=7)"); plt.ylabel("AUC (k=14)")
        plt.title("ROC-AUC: k=7 vs k=14 (GRU_with_prior)")
        plt.tight_layout(); plt.savefig(os.path.join(figs_dir,"paired_k7_vs_k14_withprior.png"), dpi=150); plt.close()

        plt.figure(figsize=(5,4)); plt.bar([0,1],[np.nanmean(xs), np.nanmean(ys)])
        plt.xticks([0,1],["k=7","k=14"]); plt.ylabel("Mean AUC")
        plt.title(f"Mean Δ = {np.nanmean(ys-xs):+.3f}")
        plt.tight_layout(); plt.savefig(os.path.join(figs_dir,"paired_mean_k7_vs_k14_withprior.png"), dpi=150); plt.close()

        pd.DataFrame([dict(n=len(xs),
                           mean_k7=float(np.nanmean(xs)),
                           mean_k14=float(np.nanmean(ys)),
                           delta_mean=float(np.nanmean(ys-xs)))])\
          .to_csv(os.path.join(tabs_dir,"paired_k7_vs_k14_withprior.csv"), index=False)

    # with_prior vs no_prior (k=14)
    xs=[]; ys=[]; labels=[]
    for (reg,lz,_,_), _ in pred_store.items():
        no = (reg,lz,14,"GRU_no_prior")
        yes= (reg,lz,14,"GRU_with_prior")
        if (no in pred_store) and (yes in pred_store):
            xs.append(auc_of(no)); ys.append(auc_of(yes)); labels.append(f"{reg}_{lz}")
    if xs:
        xs=np.array(xs,float); ys=np.array(ys,float)
        plt.figure(figsize=(6,6)); plt.scatter(xs,ys,alpha=0.7)
        lim=[min(np.nanmin(xs),np.nanmin(ys)), max(np.nanmax(xs),np.nanmax(ys))]
        plt.plot(lim,lim,'--',alpha=0.6); plt.xlabel("AUC (no_prior)"); plt.ylabel("AUC (with_prior)")
        plt.title("ROC-AUC: with_prior vs no_prior (k=14)")
        plt.tight_layout(); plt.savefig(os.path.join(figs_dir,"paired_nprior_vs_yprior_k14.png"), dpi=150); plt.close()

        plt.figure(figsize=(5,4)); plt.bar([0,1],[np.nanmean(xs), np.nanmean(ys)])
        plt.xticks([0,1],["no_prior","with_prior"]); plt.ylabel("Mean AUC")
        plt.title(f"Mean Δ = {np.nanmean(ys-xs):+.3f}")
        plt.tight_layout(); plt.savefig(os.path.join(figs_dir,"paired_mean_nprior_vs_yprior_k14.png"), dpi=150); plt.close()

        pd.DataFrame([dict(n=len(xs),
                           mean_no=float(np.nanmean(xs)),
                           mean_yes=float(np.nanmean(ys)),
                           delta_mean=float(np.nanmean(ys-xs)))])\
          .to_csv(os.path.join(tabs_dir,"paired_nprior_vs_yprior_k14.csv"), index=False)

# ─────────────── Delta-Value & Budget tables ───────────────
def delta_value_and_budget(pred_store, out_dir_global):
    tabs_dir = os.path.join(out_dir_global,"tables")
    ensure_dir(tabs_dir)

    rows=[]
    budget_rows=[]
    for (reg,lz,_,_), _ in pred_store.items():
        key_yes = (reg,lz,14,"GRU_with_prior")
        key_no  = (reg,lz,14,"GRU_no_prior")
        d_yes = pred_store.get(key_yes); d_no = pred_store.get(key_no)

        # ΔValue at Top-1/3/5%
        def topk(d):
            if d is None: return (np.nan,np.nan,np.nan)
            p = d["p_cal"] if d["p_cal"] is not None else d["p_uncal"]
            y = d["y"]
            return (precision_at_topk(y,p,0.01), precision_at_topk(y,p,0.03), precision_at_topk(y,p,0.05))

        row=dict(region=reg, lat_zone=lz)
        t_no = topk(d_no); t_yes = topk(d_yes)
        row.update(no_prior_top1=t_no[0], no_prior_top3=t_no[1], no_prior_top5=t_no[2],
                   with_prior_top1=t_yes[0], with_prior_top3=t_yes[1], with_prior_top5=t_yes[2])

        # Add prior_only if present
        key_prior = (reg,lz,14,"prior_only")
        if key_prior in pred_store:
            t_pr = topk(pred_store[key_prior])
            row.update(prior_only_top1=t_pr[0], prior_only_top3=t_pr[1], prior_only_top5=t_pr[2])
            row.update(gain_yes_vs_prior_top1=(row["with_prior_top1"]-row["prior_only_top1"]),
                       gain_yes_vs_prior_top3=(row["with_prior_top3"]-row["prior_only_top3"]),
                       gain_yes_vs_prior_top5=(row["with_prior_top5"]-row["prior_only_top5"]))
        row.update(gain_yes_vs_no_top1=(row["with_prior_top1"]-row["no_prior_top1"]),
                   gain_yes_vs_no_top3=(row["with_prior_top3"]-row["no_prior_top3"]),
                   gain_yes_vs_no_top5=(row["with_prior_top5"]-row["no_prior_top5"]))
        rows.append(row)

        # Budget recommendation (k=14, with_prior)
        if d_yes is not None:
            y = d_yes["y"]; p = d_yes["p_cal"] if d_yes["p_cal"] is not None else d_yes["p_uncal"]
            hits, recs, precs = budget_hits_curve(y,p,BUDGET_GRID)
            idx_ok = np.where(precs >= FIXED_PRECISION_TARGET)[0]
            i_star = (idx_ok[-1] if len(idx_ok)>0
                      else int(np.argmin(np.abs(BUDGET_GRID - TOP_K_PER_MILLE_FALLBACK/1000.0))))
            budget_rows.append(dict(region=reg, lat_zone=lz,
                                    cov=float(BUDGET_GRID[i_star]),
                                    hits=int(hits[i_star]),
                                    precision=float(precs[i_star]),
                                    recall=float(recs[i_star])))

    if rows:
        write_table_both(pd.DataFrame(rows), os.path.join(tabs_dir,"delta_value_k14.csv"))
    if budget_rows:
        write_table_both(pd.DataFrame(budget_rows), os.path.join(tabs_dir,"budget_recommend_k14.csv"))

# ─────────────── Tiered bar + Overlays (cross-region) ───────────────
def global_tiered_and_overlays(
    pred_store,
    out_dir_global,
    tier_counts=(4,3,3),
    bottom_extra=2,
    k=14,
    model="GRU_with_prior"
):
    """
    Outputs: tiered bar + PR/Gain overlays
    - Works for k=7 or 14
    - Uses calibrated probabilities when present
    """
    figs_dir = os.path.join(out_dir_global,"figs")
    tabs_dir = os.path.join(out_dir_global,"tables")
    ensure_dirs(figs_dir, tabs_dir)

    # 1) Collect metrics for specified k/model
    dfm = region_metrics_table(pred_store, model=model, k=k, use_calibrated=True)
    if dfm.empty:
        print(f"[WARN] No predictions for {model}, k={k}; skip global overlays.")
        return

    # 2) Tier by positives (HIGH/MID/LOW)
    dfm = dfm.sort_values("pos", ascending=False).reset_index(drop=True)
    want = sum(tier_counts)
    df_tier = split_tiers_by_activity(dfm.head(max(want, len(dfm))), tier_counts)

    # 3) Tiered horizontal bar
    order = {"HIGH":0,"MID":1,"LOW":2}
    df_plot = df_tier.copy()
    df_plot["ord"] = df_plot["tier"].map(order)
    df_plot = df_plot.sort_values(["ord","PR_AUC_cal"], ascending=[True, False])

    labels = [f"{r}_{z} [{t}]" for r,z,t in zip(df_plot["region"], df_plot["lat_zone"], df_plot["tier"])]
    vals   = df_plot["PR_AUC_cal"].values

    plt.figure(figsize=(8,5))
    plt.barh(labels, vals); plt.gca().invert_yaxis()
    plt.xlabel("PR_AUC (calibrated)")
    plt.title(f"Tiered regions — {model}, k={k}")
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, f"bar_tiered_with_prior_k{k}.png"), dpi=150)
    plt.close()

    write_table_both(
        df_plot.drop(columns=["ord"]),
        os.path.join(tabs_dir, f"tiered_regions_with_prior_k{k}.csv")
    )

    # 4) Overlay: best from each tier + a few low-fire examples
    def best_in_each_tier(df_tier, metric="PR_AUC_cal"):
        reps=[]
        for t in ["HIGH","MID","LOW"]:
            sub = df_tier[df_tier["tier"]==t]
            if sub.empty: 
                continue
            reps.append(sub.sort_values(metric, ascending=False).iloc[0])
        return pd.DataFrame(reps)

    best3 = best_in_each_tier(df_tier, metric="PR_AUC_cal")

    # Add bottom examples
    bottom = dfm.tail(bottom_extra*2)
    sel = [(r["region"], r["lat_zone"], r["tier"] if "tier" in r else "LOW-FIRE") for _,r in best3.iterrows()]
    for _, r in bottom.iterrows():
        p = (r["region"], r["lat_zone"], "LOW-FIRE")
        if p[:2] not in [(a[0],a[1]) for a in sel]:
            sel.append(p)
        if len(sel) >= len(best3)+bottom_extra:
            break

    # Helper to get y/p for a region
    def get_pred(reg,lz):
        key=(reg,lz,k,model)
        if key not in pred_store: return None,None
        d=pred_store[key]
        y=d["y"]; p=d["p_cal"] if d["p_cal"] is not None else d["p_uncal"]
        return y,p

    # 5) PR overlay
    plt.figure(figsize=(7,6))
    for (reg,lz,tier) in sel:
        y,p = get_pred(reg,lz)
        if y is None: continue
        prec, rec, _ = precision_recall_curve(y,p)
        lab = f"{tier} • {reg}_{lz}"
        plt.plot(rec, prec, label=lab)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR overlay — tiers + low-fire ({model}, k={k})")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, f"overlay_PR_tiers_lowfire_with_prior_k{k}.png"), dpi=150)
    plt.close()

    # 6) Gain overlay
    plt.figure(figsize=(7,6))
    for (reg,lz,tier) in sel:
        y,p = get_pred(reg,lz)
        if y is None: continue
        order = np.argsort(-p); y_sorted = y[order]
        cov = np.arange(1, len(y)+1)/len(y)
        rec2 = np.cumsum(y_sorted)/max(y.sum(),1)
        lab = f"{tier} • {reg}_{lz}"
        plt.plot(cov, rec2, label=lab)
    plt.plot([0,1],[0,1],'--', alpha=0.6)
    plt.xlabel("Coverage"); plt.ylabel("Cumulative recall")
    plt.title(f"Gain overlay — tiers + low-fire ({model}, k={k})")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, f"overlay_Gain_tiers_lowfire_with_prior_k{k}.png"), dpi=150)
    plt.close()


# ─────────────── Per-region outputs (curves/budget/ΔValue) ───────────────
def per_region_outputs(pred_store, hazard_curves, hazard_waits, hazard_trends, out_root):
    for (reg,lz,_,_), _ in pred_store.items():
        tag = f"{reg}_{lz}".replace(" ","")
        per_dir = os.path.join(out_root,"per_region", safe_filename(tag))
        figs_dir = os.path.join(per_dir,"figs"); tbl_dir = os.path.join(per_dir,"tables")
        ensure_dirs(figs_dir, tbl_dir)

        # All (k, model) combos for this region
        combos = sorted({(kk,mm) for (r2,l2,kk,mm) in pred_store.keys() if (r2==reg and l2==lz)})
        for kk, mm in combos:
            dd = pred_store[(reg,lz,kk,mm)]
            y  = dd["y"]; p = dd["p_cal"] if dd["p_cal"] is not None else dd["p_uncal"]
            title = f"{mm} | k={kk} | {reg}_{lz}"
            outp  = os.path.join(figs_dir, f"{safe_filename(tag)}_k{kk}_{mm}")
            metr, df_th = pr_roc_calibration_and_save(y, p, title, outp)
            one = dict(region=reg, lat_zone=lz, model=mm, horizon=kk, **metr)
            write_table_both(pd.DataFrame([one]), os.path.join(tbl_dir, f"metrics_k{kk}_{mm}.csv"))
            write_table_both(df_th, os.path.join(tbl_dir, f"threshold_sweep_k{kk}_{mm}.csv"))

        # Budget curves (k=K_FOR_BUDGET): overlay with vs without prior
        overlays={}
        for mm in ["GRU_with_prior","GRU_no_prior"]:
            key = (reg,lz,K_FOR_BUDGET,mm)
            if key in pred_store:
                dd = pred_store[key]; y=dd["y"]; p=dd["p_cal"] if dd["p_cal"] is not None else dd["p_uncal"]
                hits,recs,precs = budget_hits_curve(y,p,BUDGET_GRID)
                overlays[mm]=(hits,recs,precs)
        if overlays:
            plt.figure(figsize=(8,5))
            for lab,(h,r,p) in overlays.items():
                plt.plot(BUDGET_GRID, h, label=lab)
            if ("GRU_with_prior" in overlays):
                h,r,p = overlays["GRU_with_prior"]
                idx_ok = np.where(p >= FIXED_PRECISION_TARGET)[0]
                i_star = (idx_ok[-1] if len(idx_ok)>0
                          else int(np.argmin(np.abs(BUDGET_GRID - TOP_K_PER_MILLE_FALLBACK/1000.0))))
                plt.plot(BUDGET_GRID[i_star], h[i_star], 'o')
                plt.annotate("★Rec.", (BUDGET_GRID[i_star], h[i_star]), xytext=(5,5), textcoords='offset points', fontsize=8)
            plt.xlabel("Coverage (alert rate)"); plt.ylabel("Hits")
            plt.title(f"Budget curve (k={K_FOR_BUDGET}) — {reg}_{lz}")
            plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(figs_dir, f"{safe_filename(tag)}_budget_k{K_FOR_BUDGET}.png"), dpi=150); plt.close()

        # ΔValue per region (k=14)
        def topk_row(key):
            d=pred_store.get(key)
            if d is None: return (np.nan,np.nan,np.nan)
            p=d["p_cal"] if d["p_cal"] is not None else d["p_uncal"]
            y=d["y"]; return (precision_at_topk(y,p,0.01), precision_at_topk(y,p,0.03), precision_at_topk(y,p,0.05))
        row=dict(region=reg, lat_zone=lz)
        row["no_prior_top1"],row["no_prior_top3"],row["no_prior_top5"] = topk_row((reg,lz,14,"GRU_no_prior"))
        row["with_prior_top1"],row["with_prior_top3"],row["with_prior_top5"] = topk_row((reg,lz,14,"GRU_with_prior"))
        write_table_both(pd.DataFrame([row]), os.path.join(tbl_dir,"delta_value_k14.csv"))

        # Hazard plots/tables (if CSVs available)
        htag = safe_filename(tag)
        hz = hazard_curves.get(htag)
        if isinstance(hz, pd.DataFrame) and not hz.empty:
            plt.figure(figsize=(6,4))
            plt.plot(hz["day"], hz["cum_prob"])
            plt.xlabel("Days since last fire"); plt.ylabel("P(T ≤ d)")
            plt.title(f"Hazard cumulative — {reg}_{lz}")
            plt.tight_layout()
            plt.savefig(os.path.join(figs_dir, f"{safe_filename(tag)}_hazard_cumprob.png"), dpi=150); plt.close()
            write_table_both(hz, os.path.join(tbl_dir, "hazard_curve.csv"))
        wt = hazard_waits.get(htag)
        if isinstance(wt, pd.DataFrame) and not wt.empty and ("gap_days" in wt.columns):
            gaps = wt["gap_days"].values
            plt.figure(figsize=(6,4))
            plt.hist(np.clip(gaps,1,60), bins=np.arange(1,61), edgecolor="k", alpha=0.7)
            plt.xlabel("Days between consecutive fires"); plt.ylabel("Count")
            plt.title(f"Waiting histogram — {reg}_{lz}")
            plt.tight_layout()
            plt.savefig(os.path.join(figs_dir, f"{safe_filename(tag)}_waiting_hist.png"), dpi=150); plt.close()
            write_table_both(wt, os.path.join(tbl_dir, "waiting_intervals.csv"))
        tr = hazard_trends.get(htag)
        if isinstance(tr, pd.DataFrame) and not tr.empty and ("median" in tr.columns):
            plt.figure(figsize=(6,4))
            plt.plot(tr["start_year"], tr["median"], marker="o", label="median")
            if "p25" in tr.columns and "p75" in tr.columns:
                plt.fill_between(tr["start_year"], tr["p25"], tr["p75"], alpha=0.2, label="IQR")
            plt.xlabel("Start year"); plt.ylabel("Days between fires")
            plt.title(f"Waiting trend — {reg}_{lz}")
            plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(figs_dir, f"{safe_filename(tag)}_waiting_trend.png"), dpi=150); plt.close()
            write_table_both(tr, os.path.join(tbl_dir, "waiting_trend.csv"))

# ─────────────── Global hazard summary (if available) ───────────────
def global_hazard_summary(hazard_curves, hazard_waits, out_dir_global, top_n=12):
    """
    Global hazard summary from waiting_intervals_* and hazard_curve_*:
      - Aggregate: n_intervals, gap_median
      - From curves: cum30 / cum60
      - Overlays:
          1) Top-N by n_intervals (most active)
          2) Top-N by cum60 (fastest reignite)
      - Export CSV + Excel
    """
    ensure_dir(os.path.join(out_dir_global, "figs"))
    ensure_dir(os.path.join(out_dir_global, "tables"))

    # Robust curve lookup with region-label normalization
    def lookup_curve(curves_dict: dict, display_tag: str):
        candidates = [
            display_tag,
            display_tag.replace("Coast_", "South_"),
            display_tag.replace("South_", "Coast_"),
        ]
        for k in candidates:
            v = curves_dict.get(k, None)
            if isinstance(v, pd.DataFrame):
                return v
        def norm(s): return re.sub(r'[^A-Za-z0-9]+', '', str(s)).lower()
        ndisp = norm(display_tag)
        for k, v in curves_dict.items():
            if norm(k) == ndisp and isinstance(v, pd.DataFrame):
                return v
        return None

    def map_tag(tag):
        parts = tag.split("_")
        region = parts[0]
        latz = "_".join(parts[1:]) if len(parts) > 1 else ""
        if region == "South":
            region = "Coast"
        return f"{region}_{latz}", region, latz

    def cum_at_day(dfh, day):
        if dfh is None or dfh.empty or ("day" not in dfh.columns):
            return np.nan
        sub = dfh[dfh["day"] <= day]
        return float(sub["cum_prob"].max()) if not sub.empty else np.nan

    # Summary table
    rows = []
    for tag, wt in hazard_waits.items():
        if wt is None or wt.empty or ("gap_days" not in wt.columns):
            continue
        gaps = wt["gap_days"].dropna().values
        if len(gaps) == 0:
            continue
        full_tag, region, latz = map_tag(tag)
        dfh = lookup_curve(hazard_curves, full_tag)

        rows.append(dict(
            tag=full_tag,
            region=region,
            lat_zone=latz,
            n_intervals=int(len(gaps)),
            gap_median=float(np.median(gaps)),
            P_le_30=float((gaps <= 30).mean()),
            cum30=cum_at_day(dfh, 30),
            cum60=cum_at_day(dfh, 60),
        ))

    if not rows:
        print("[GLOBAL] hazard_summary: 没有可汇总的 waiting_intervals。")
        return

    df = pd.DataFrame(rows)

    # Save summary (CSV + Excel)
    tbl_dir = os.path.join(out_dir_global, "tables")
    ensure_dir(tbl_dir)
    csv_path = os.path.join(tbl_dir, "hazard_summary_GLOBAL.csv")
    xlsx_path = os.path.join(tbl_dir, "hazard_summary_GLOBAL.xlsx")
    df.sort_values("n_intervals", ascending=False).to_csv(csv_path, index=False)
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
            df.sort_values("n_intervals", ascending=False).to_excel(xw, sheet_name="by_activity", index=False)
            df.sort_values("cum60", ascending=False).to_excel(xw, sheet_name="by_fast_reignite", index=False)
    except Exception as e:
        print(f"[WARN] 写入 Excel 失败：{e}")

    # Overlay #1: Top-N by activity
    top_by_activity = df.sort_values("n_intervals", ascending=False).head(top_n)
    plt.figure(figsize=(7, 5))
    any_curve = False
    for disp_tag in top_by_activity["tag"].tolist():
        dfh = lookup_curve(hazard_curves, disp_tag)
        if dfh is None or dfh.empty:
            continue
        any_curve = True
        plt.plot(dfh["day"], dfh["cum_prob"], label=disp_tag[:24])
    if any_curve:
        plt.xlabel("Days since last fire")
        plt.ylabel("P(T ≤ d)")
        plt.title("Post-fire cumulative probability — Top-N by activity")
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir_global, "figs", "global_hazard_cumprob_overlay_top_activity.png"), dpi=150)
    plt.close()

    # Overlay #2: Top-N by fastest reignite (cum60)
    top_by_fast = df.sort_values("cum60", ascending=False).head(top_n)
    plt.figure(figsize=(7, 5))
    any_curve = False
    for disp_tag in top_by_fast["tag"].tolist():
        dfh = lookup_curve(hazard_curves, disp_tag)
        if dfh is None or dfh.empty:
            continue
        any_curve = True
        plt.plot(dfh["day"], dfh["cum_prob"], label=disp_tag[:24])
    if any_curve:
        plt.xlabel("Days since last fire")
        plt.ylabel("P(T ≤ d)")
        plt.title("Post-fire cumulative probability — Top-N by fast reignite")
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir_global, "figs", "global_hazard_cumprob_overlay_top_fast.png"), dpi=150)
    plt.close()

    # Median gap bar (sorted by activity)
    df_bar = df.sort_values(["n_intervals", "gap_median"], ascending=[False, True])
    labels = [f"{r}_{z}" for r, z in zip(df_bar["region"], df_bar["lat_zone"])]
    plt.figure(figsize=(8, 6))
    plt.barh(labels, df_bar["gap_median"].values)
    plt.xlabel("Median days between fires")
    plt.title("Median inter-fire waiting time (sorted by activity)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir_global, "figs", "global_median_gap_bar.png"), dpi=150)
    plt.close()

    print(f"[GLOBAL] hazard summary -> {csv_path}")

# ─────────────── Per-region operating point (accuracy, etc.) ───────────────
def per_region_operating_point(pred_store, out_dir_global):
    """At fixed precision 30% (else Top-1‰), output region-wise accuracy/recall/F1/etc. for k=14 with_prior."""
    tabs_dir = os.path.join(out_dir_global,"tables")
    ensure_dir(tabs_dir)

    rows=[]
    for (reg,lz,kk,mm), d in pred_store.items():
        if kk!=14 or mm!="GRU_with_prior": 
            continue
        y = d["y"]; p = d["p_cal"] if d["p_cal"] is not None else d["p_uncal"]
        df_th = threshold_sweep(y,p)
        cand = df_th[df_th["precision"] >= FIXED_PRECISION_TARGET]
        if len(cand)>0:
            t = float(cand.loc[cand["recall"].idxmax(),"threshold"])
            rule = "fixed_precision"
        else:
            k = max(1, int(len(y) * (TOP_K_PER_MILLE_FALLBACK/1000.0)))
            t = float(np.partition(p, -k)[-k])
            rule = "top_k_per_mille"
        row = df_th.iloc[(np.abs(df_th["threshold"]-t)).argmin()].to_dict()
        row.update(region=reg, lat_zone=lz, threshold=row["threshold"], rule=rule,
                   coverage=float((p>=t).mean()))
        rows.append(row)
    if rows:
        df = pd.DataFrame(rows)
        cols_order = ["region","lat_zone","rule","threshold","coverage","precision","recall","f1",
                      "accuracy","tpr","tnr","fpr","tp","fp","fn","tn"]
        for c in cols_order:
            if c not in df.columns: df[c]=np.nan
        write_table_both(df[cols_order], os.path.join(tabs_dir,"operating_point_k14_with_prior.csv"))

# ─────────────── MAIN ───────────────
if __name__ == "__main__":
    # 1) Load GRU predictions & Hazard outputs
    pred_store = load_all_gru_predictions(GRU_PRED_DIR)
    curves, waits, trends = scan_hazard_curves(HAZARD_CSV_DIR)

    # Export roots
    OUT_GRU_GLOBAL   = os.path.join(SITE_EXPORT_ROOT, "gru", "global")
    OUT_GRU_PERROOT  = os.path.join(SITE_EXPORT_ROOT, "gru")
    OUT_HAZ_GLOBAL   = os.path.join(SITE_EXPORT_ROOT, "hazard", "global")

    ensure_dirs(OUT_GRU_GLOBAL, OUT_HAZ_GLOBAL)

    # 2) Global comparisons (k=7 vs 14; with vs without prior)
    paired_plots_and_tables(pred_store, OUT_GRU_GLOBAL)

    # 3) ΔValue & budget recommendation tables
    delta_value_and_budget(pred_store, OUT_GRU_GLOBAL)

    # Cross-region comparisons at k=14 (existing)
    global_tiered_and_overlays(pred_store, OUT_GRU_GLOBAL, tier_counts=(4,3,3), bottom_extra=2, k=14)
    
    # Cross-region comparisons at k=7 (new)
    global_tiered_and_overlays(pred_store, OUT_GRU_GLOBAL, tier_counts=(4,3,3), bottom_extra=2, k=7)

    # 5) Per-region outputs (GRU + available Hazard)
    per_region_outputs(pred_store, curves, waits, trends, OUT_GRU_PERROOT)

    # 6) Global hazard summary (if curves present)
    global_hazard_summary(curves, waits, OUT_HAZ_GLOBAL)

    # 7) Per-region operating point (accuracy, etc.)
    per_region_operating_point(pred_store, OUT_GRU_GLOBAL)

    print("\n[DONE] Outputs written under:", SITE_EXPORT_ROOT)
    print("      - GRU/global: paired plots, tiered bar & overlays, delta-value & budget tables, operating-point table")
    print("      - GRU/per_region/<tag>: PR/ROC/Calibration/Gain-Lift, budget, delta tables")
    print("      - Hazard/global: median bar & overlay (if hazard CSVs present)")
    print("      - Region labels normalized: 'South' -> 'Coast'")
