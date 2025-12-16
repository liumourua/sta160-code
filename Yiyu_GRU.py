# -*- coding: utf-8 -*-
"""
Wildfire GRU — region × lat_zone (resumable & auto-rerun on small windows)

"""

import os, re, gc, math, random, time
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss,
    precision_recall_curve, roc_curve, confusion_matrix
)
from sklearn.calibration import calibration_curve

import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.utils import Sequence

# =========================
# Paths & basic config
# =========================
PARQUET_ROOT = r'D:\桌面\大学\大四\第一学期\sta160\project\Wildfire_CLEANED_WITH_OOF_LITE_parquet_region_latzone'
OUT_ROOT      = r'D:\桌面\大学\大四\第一学期\sta160\project\rnn2'
FIG_DIR       = os.path.join(OUT_ROOT, "figs")
PRED_OUT_DIR  = os.path.join(OUT_ROOT, "predictions")
CMP_DIR       = os.path.join(FIG_DIR, "_comparisons")
os.makedirs(FIG_DIR, exist_ok=True); os.makedirs(PRED_OUT_DIR, exist_ok=True); os.makedirs(CMP_DIR, exist_ok=True)

RUN_PLAN = 'full'               # 'smoke' / 'full'
REGION_OVERRIDE = None          # e.g., 'West'
LATZONE_OVERRIDE = None         # e.g., 'Mid'

RUN_MODE = 'both'               # 'phys' / 'prior' / 'both'
PRED_K_LIST = [7, 14]           # run horizons

# Training mode: per region×lat_zone vs predefined "combined" groups
TRAIN_MODE = "combined"         # or "standard"

# Resume / rerun
RESUME_MODE = "auto"            # skip if prediction CSV exists
PRED_FILE_PATTERN = "{tag}_{model}_k{k}_{plan}.csv"

# Comparison plots
COMPARISON_MODEL_PREF = "GRU_with_prior"
COMPARISON_K_PREF     = 14

# Hyperparameters
DEBUG_MODE = False
if DEBUG_MODE:
    SEQ_LEN = 7;  EPOCHS = 6;  PATIENCE = 2
    LEARNING_RATE = 1e-3; GRU_UNITS = 32; DENSE_UNITS = 16
else:
    SEQ_LEN = 28; EPOCHS = 20; PATIENCE = 4
    LEARNING_RATE = 5e-4; GRU_UNITS = 64; DENSE_UNITS = 32

BATCH_SIZE = 1024               # baseline for steps/epoch check
PRED_BATCH_SIZE = 1536
FAST_MODE = False
SHOW_FIG  = False

if FAST_MODE:
    EPOCHS = 12; PATIENCE = 2
    STEPS_PER_EPOCH_CAP = 1500; VAL_STEPS_CAP = 200
    CALIBRATION_BINS = 10; THRESH_SWEEP_N = 51
    TRAIN_METRICS = []; USE_MASKING = False; DROPOUT_RATE = 0.2
else:
    STEPS_PER_EPOCH_CAP = None; VAL_STEPS_CAP = None
    CALIBRATION_BINS = 15; THRESH_SWEEP_N = 101
    TRAIN_METRICS = [tf.keras.metrics.AUC(curve="PR", name="PR_AUC"),
                     tf.keras.metrics.AUC(curve="ROC", name="ROC_AUC")]
    USE_MASKING = True; DROPOUT_RATE = 0.3

# Time split (strict, label-based) + gap to avoid window crossing splits
TRAIN_MAX_YEAR = 2021
VAL_YEAR       = 2022
TEST_MIN_YEAR  = 2023
ENFORCE_GAP    = True
GAP_DAYS       = 7

# Combined region definitions (names must match values in parquet)
COMBINED_REGION_GROUPS: Dict[str, List[Tuple[str, str]]] = {
    "NorthEast_NorthBand": [
        ("North East", "Far North"),
        ("North East", "North"),
    ],
    "East_NorthBand": [
        ("East", "North"),
        ("East", "Far North"),
    ],
    "CentralWest_Mid+South": [
        ("Central West", "Mid"),
        ("Central West", "South"),
    ],
}
COMBINED_LATZONE_TAG = "COMBINED"

# Loss/optimization
USE_FOCAL_LOSS = False
POS_WEIGHT     = None

# Misc
ENABLE_MIXED_PRECISION = True
SEED = 2024
np.random.seed(SEED); random.seed(SEED); tf.random.set_seed(SEED)

# Threshold / budget policy
FIXED_PRECISION_TARGET = 0.30
TOP_K_PER_MILLE = 1.0
BUDGET_GRID   = np.linspace(0.0, 0.10, 21)
K_FOR_BUDGET  = 14

SAVE_PREDICTIONS = True
CLEAR_GPU_BETWEEN_JOBS = True
WAIT_BETWEEN_REGIONS_SEC = 1

# Auto-rerun trigger (too few steps/epoch)
AUTO_RERUN_IF_FEW_STEPS = True
TARGET_MIN_STEPS = 60
TRAIN_BATCH_MIN = 128
VAL_BATCH_MAX   = 1536
TEST_BATCH_MAX  = 4096

# =========================
# GPU / AMP
# =========================
if ENABLE_MIXED_PRECISION:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')
tf.config.optimizer.set_jit(False)
os.environ.pop("TF_XLA_FLAGS", None)
for g in tf.config.list_physical_devices('GPU'):
    try: tf.config.experimental.set_memory_growth(g, True)
    except Exception: pass

# =========================
# Utilities
# =========================
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def safe_filename(s, max_len=140):
    s = re.sub(r'[<>:"/\\|?*]+','_',s); s = re.sub(r'\s+','_',s)
    return s[:max_len].strip('._')

def safe_sheetname(s): return safe_filename(s,31) or "sheet"

def write_region_excel(excel_path, sheets: Dict[str,pd.DataFrame]):
    if not sheets: return
    with pd.ExcelWriter(excel_path, engine="openpyxl", mode="w") as xw:
        for k, df in sheets.items():
            df.to_excel(xw, sheet_name=safe_sheetname(str(k)), index=False)

def focal_loss(alpha=0.75, gamma=2.0):
    def loss(y_true, y_pred):
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.-eps)
        pt = tf.where(tf.equal(y_true,1.), y_pred, 1-y_pred)
        w  = tf.where(tf.equal(y_true,1.), alpha, 1-alpha)
        return tf.reduce_mean(-w * tf.pow(1-pt, gamma) * tf.math.log(pt))
    return loss

def get_loss_and_metrics(y_tr=None):
    # Weighted BCE by default; focal optional
    if USE_FOCAL_LOSS:
        loss_obj = focal_loss()
    else:
        if POS_WEIGHT is None and y_tr is not None:
            pos = float(np.sum(y_tr)); neg = float(len(y_tr)-pos)
            pw = (neg/max(pos,1.0))
        else:
            pw = POS_WEIGHT if POS_WEIGHT is not None else 1.0
        @tf.function
        def weighted_bce(y_true, y_pred):
            eps = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, eps, 1.-eps)
            return tf.reduce_mean(- y_true*tf.math.log(y_pred)*pw - (1-y_true)*tf.math.log(1-y_pred))
        loss_obj = weighted_bce
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
    return loss_obj, TRAIN_METRICS, optimizer

def sigmoid_to_logit(p, eps=1e-7):
    p = np.clip(p, eps, 1-p); return np.log(p/(1-p))

def fit_temperature(y_val, p_val):
    # Temperature scaling by minimizing NLL on validation
    z = sigmoid_to_logit(p_val)
    def nll(t):
        q = 1/(1+np.exp(-z/t)); q = np.clip(q,1e-7,1-1e-7)
        return -np.mean(y_val*np.log(q) + (1-y_val)*np.log(1-q))
    grid = np.linspace(0.25, 4.0, 30)
    t0 = grid[int(np.argmin([nll(t) for t in grid]))]
    fine = np.linspace(max(0.1,t0-0.5), t0+0.5, 40)
    return float(fine[int(np.argmin([nll(t) for t in fine]))])

def apply_temperature(p, T):
    z = sigmoid_to_logit(p); return (1/(1+np.exp(-z/T))).astype("float32")

def iso_f1_lines(ax, f1s=(0.2,0.4,0.6,0.8)):
    r = np.linspace(0.01,1,200)
    for f in f1s:
        p = (f*r)/(2*r - f + 1e-9); p[(2*r-f)<=0] = np.nan
        ax.plot(r,p,'--',alpha=0.35)
    ax.text(0.98,0.98,"F1 iso-lines",ha='right',va='top',fontsize=8,alpha=0.6)

def normalized_ap(y_true, y_prob):
    ap = average_precision_score(y_true, y_prob); base = float(np.mean(y_true))
    nap = 0.0 if base>=1.0 else (ap - base)/max(1-base,1e-12)
    return ap, nap, base

def precision_at_topk(y_true, y_prob, frac):
    n = len(y_true); k = max(1, int(round(n*frac)))
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

# =========================
# Data I/O / feature columns
# =========================
def auto_list_all_regions(root_path):
    dataset = ds.dataset(root_path, format="parquet", partitioning="hive")
    tbl = dataset.to_table(columns=["region","lat_zone","Wildfire"])
    df_idx = tbl.to_pandas()
    stats = (df_idx.groupby(["region","lat_zone"])["Wildfire"]
             .agg(["sum","count"]).rename(columns={"sum":"n_fire","count":"n_total"}))
    stats["fire_rate"] = stats["n_fire"]/stats["n_total"]
    stats = stats.reset_index()
    print("\n[AUTO] 按火灾数降序（前10）:")
    print(stats.sort_values("n_fire", ascending=False).head(10))
    pairs = [(r, z) for r,z in stats[["region","lat_zone"]].itertuples(index=False)]
    return pairs, stats

def load_region_latzone_df(root_path, region, lat_zone):
    dataset = ds.dataset(root_path, format="parquet", partitioning="hive")
    filt = (ds.field("region")==region) & (ds.field("lat_zone")==lat_zone)
    table = dataset.to_table(filter=filt)
    df = table.to_pandas()
    if df.empty:
        print(f"[LOAD] region={region}, lat_zone={lat_zone}, rows=0 → 跳过")
        return df
    df["datetime"] = pd.to_datetime(df["datetime"])
    if "year" not in df.columns: df["year"] = df["datetime"].dt.year
    df = df.sort_values(["lat_bin","lon_bin","datetime"]).reset_index(drop=True)
    print(f"[LOAD] region={region}, lat_zone={lat_zone}, rows={len(df)}, fires={df['Wildfire'].sum()}")
    return df

def choose_feature_columns(df):
    # Split features into physical vs prior-based
    all_cols = df.columns.tolist()
    meta_cols = {"latitude","longitude","datetime","year","month",
                 "lat_bin","lon_bin","geo_bin","region","lat_zone","season",
                 "Wildfire","rnn_weight"}
    meta_cols |= {c for c in all_cols if c.startswith("cb_")}
    feat_all = [c for c in all_cols if c not in meta_cols]
    prior_cols = [c for c in feat_all if c.startswith("p_cb") or c in ["logit_cb","rank_cb_day"]]
    phys_cols  = [c for c in feat_all if c not in prior_cols]
    print(f"[FEATS] phys={len(phys_cols)}, prior={len(prior_cols)}")
    return phys_cols, prior_cols

# =========================
# Time split + gap (label-based)
# =========================
def split_masks_for_k(df, seq_len, k_days, train_max_year, val_year, test_min_year, enforce_gap=True):
    """
    Strict label-date splits:
      - train: label_dt ≤ TRAIN_MAX_YEAR end; reserve GAP_DAYS before year end
      - val:   label_dt in VAL_YEAR; reserve GAP_DAYS at both ends
      - test:  label_dt ≥ TEST_MIN_YEAR start; reserve GAP_DAYS after year start
    """
    jan1_val   = pd.Timestamp(val_year, 1, 1)
    dec31_tr   = pd.Timestamp(train_max_year, 12, 31)
    dec31_val  = pd.Timestamp(val_year, 12, 31)
    jan1_test  = pd.Timestamp(test_min_year, 1, 1)

    if enforce_gap and GAP_DAYS > 0:
        gap = pd.Timedelta(days=GAP_DAYS)
        train_latest   = dec31_tr - gap
        val_earliest   = jan1_val + gap
        val_latest     = dec31_val - gap
        test_earliest  = jan1_test + gap
    else:
        train_latest   = dec31_tr
        val_earliest   = jan1_val
        val_latest     = dec31_val
        test_earliest  = jan1_test

    def per_group(g: pd.DataFrame):
        g = g.sort_values("datetime").reset_index(drop=True)
        n = len(g); idx_tr=[]; idx_va=[]; idx_te=[]
        if n <= seq_len + k_days:
            return idx_tr, idx_va, idx_te
        dts = g["datetime"].values
        for t in range(seq_len, n - k_days):
            # label date = end of window + k days
            label_dt = pd.Timestamp(dts[t + k_days])
            if label_dt <= train_latest:
                idx_tr.append(t)
            elif val_earliest <= label_dt <= val_latest:
                idx_va.append(t)
            elif label_dt >= test_earliest:
                idx_te.append(t)
        return idx_tr, idx_va, idx_te

    return per_group

class KWindowSequence(Sequence):
    # Sliding windows per grid cell; label is "any fire in next k_days"
    def __init__(self, df, feature_cols, seq_len, k_days, t_pairs, batch_size, shuffle, name="train"):
        self.seq_len = seq_len; self.k_days=k_days; self.batch_size=batch_size
        self.shuffle = shuffle; self.name=name
        groups = list(df.groupby(["lat_bin","lon_bin"]).groups.items())
        self.index_pairs = t_pairs[:]
        self.grp_X=[]; self.grp_y=[]; self.grp_w=[]; self.grp_dt=[]
        for _, rows in groups:
            g = df.iloc[rows]
            X = g[feature_cols].to_numpy(dtype="float16", copy=False)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            y = g["Wildfire"].to_numpy(dtype="float32", copy=False)
            w = (g["rnn_weight"].to_numpy(dtype="float32", copy=False)
                 if "rnn_weight" in g.columns else np.ones_like(y, dtype="float32"))
            dt = g["datetime"].to_numpy()
            self.grp_X.append(X); self.grp_y.append(y); self.grp_w.append(w); self.grp_dt.append(dt)
        self.on_epoch_end()
    def __len__(self): return math.ceil(len(self.index_pairs)/self.batch_size)
    def on_epoch_end(self):
        if self.shuffle: random.shuffle(self.index_pairs)
    def __getitem__(self, idx):
        start = idx*self.batch_size; end = min((idx+1)*self.batch_size,len(self.index_pairs))
        cur = self.index_pairs[start:end]; L=self.seq_len; bs=len(cur); F=self.grp_X[0].shape[1]
        Xb = np.empty((bs,L,F),dtype="float16"); yb=np.empty((bs,),dtype="float32"); wb=np.empty((bs,),dtype="float32")
        for i,(gid,t) in enumerate(cur):
            Xg,yg,wg = self.grp_X[gid], self.grp_y[gid], self.grp_w[gid]
            Xb[i] = Xg[t-L:t,:]
            yb[i] = 1.0 if yg[t+1:t+1+self.k_days].max() > 0 else 0.0
            wb[i] = wg[t+self.k_days] if (t+self.k_days) < len(wg) else 1.0
        return Xb, yb, wb

def build_index_lists_for_split(df, seq_len, k_days, train_max, val_year, test_min, enforce_gap):
    """Expand per-group masks into global (gid, t) lists."""
    per_group = split_masks_for_k(df, seq_len, k_days, train_max, val_year, test_min, enforce_gap)
    gitems = list(df.groupby(["lat_bin","lon_bin"]).groups.items())
    idx_tr=[]; idx_va=[]; idx_te=[]
    for gid, (_, rows) in enumerate(gitems):
        g = df.iloc[rows]
        t_tr, t_va, t_te = per_group(g)
        idx_tr.extend([(gid,t) for t in t_tr]); idx_va.extend([(gid,t) for t in t_va]); idx_te.extend([(gid,t) for t in t_te])
    return idx_tr, idx_va, idx_te

def maybe_subsample_indices(idx_list, max_n, name):
    if (max_n is not None) and (len(idx_list) > max_n):
        rnd = np.random.RandomState(SEED)
        keep = rnd.choice(len(idx_list), size=max_n, replace=False)
        idx_list = [idx_list[i] for i in keep]
        print(f"[SUBSAMPLE] {name}: kept {len(keep)} samples (cap={max_n})")
    return idx_list

# =========================
# Model
# =========================
def build_gru_backbone(input_tensor):
    x = input_tensor
    if USE_MASKING: x = layers.Masking()(x)
    x = layers.GRU(GRU_UNITS, return_sequences=False)(x)
    if DROPOUT_RATE and DROPOUT_RATE>0: x = layers.Dropout(DROPOUT_RATE)(x)
    return x

def build_gru_model(input_shape, loss_obj, metrics, optimizer):
    inp = layers.Input(shape=input_shape, dtype='float16')
    x = build_gru_backbone(inp); x = layers.Dense(DENSE_UNITS, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid", dtype="float32")(x)
    model = models.Model(inp, out)
    model.compile(optimizer=optimizer, loss=loss_obj, metrics=metrics, weighted_metrics=[])
    return model

def build_gru_with_prior(input_shape, feature_cols, prior_cols, loss_obj, metrics, optimizer):
    # Concatenate GRU state with last-step logit_cb (prior)
    if "logit_cb" not in prior_cols: raise ValueError("prior_cols 必须包含 'logit_cb'")
    inp = layers.Input(shape=input_shape, dtype='float16')
    h = build_gru_backbone(inp)
    logit_idx = feature_cols.index("logit_cb")
    logit_cb_last = layers.Lambda(lambda t: tf.cast(t[:, -1, logit_idx:logit_idx+1], 'float32'),
                                  name="logit_cb_last")(inp)
    x = layers.Concatenate()([h, logit_cb_last]); x = layers.Dense(DENSE_UNITS, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid", dtype="float32")(x)
    model = models.Model(inp, out)
    model.compile(optimizer=optimizer, loss=loss_obj, metrics=metrics, weighted_metrics=[])
    return model

# =========================
# Evaluation / plotting
# =========================
def threshold_sweep(y_true, y_prob, n=101):
    ths = np.linspace(0,1,n); rows=[]
    for t in ths:
        pred = (y_prob>=t).astype(int)
        tn,fp,fn,tp = confusion_matrix(y_true,pred,labels=[0,1]).ravel()
        prec = tp/max(tp+fp,1); rec = tp/max(tp+fn,1)
        f1 = (2*prec*rec)/max(prec+rec,1e-12); brier=np.mean((y_prob-y_true)**2)
        rows.append(dict(threshold=t,tp=tp,fp=fp,fn=fn,tn=tn,precision=prec,recall=rec,f1=f1,brier=brier))
    return pd.DataFrame(rows)

def auto_threshold_fixed_precision(df_th, target_p=0.3):
    sub=df_th[df_th["precision"]>=target_p]
    return None if sub.empty else float(sub.loc[sub["recall"].idxmax(),"threshold"])

def auto_threshold_topk(y_prob, top_k_per_mille=1.0):
    k=max(1,int(len(y_prob)*(top_k_per_mille/1000.0))); return float(np.partition(y_prob,-k)[-k])

def evaluate_and_plot_all(fig_prefix, name, y_true, y_prob, save_dir):
    ensure_dir(save_dir)
    sp = safe_filename(fig_prefix); sn = safe_filename(name)
    ap, nap, base = normalized_ap(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    brier   = brier_score_loss(y_true, y_prob)
    bss     = 1.0 - brier / brier_score_loss(y_true, np.full_like(y_true, base))
    prec,rec,_ = precision_recall_curve(y_true,y_prob); fpr,tpr,_ = roc_curve(y_true,y_prob)
    prob_true, prob_pred = calibration_curve(y_true,y_prob,n_bins=CALIBRATION_BINS,strategy='uniform')

    fig, ax = plt.subplots(1,3,figsize=(16,4))
    ax0=ax[0]; ax0.plot(rec,prec,label=f"{name} (AP={ap:.3f} | nAP={nap:.3f} | base={base:.3f})")
    iso_f1_lines(ax0); ax0.set_xlabel("Recall"); ax0.set_ylabel("Precision"); ax0.set_title("Precision-Recall"); ax0.legend()

    ax1=ax[1]; ax1.plot(fpr,tpr,label=f"{name} (ROC-AUC={roc_auc:.3f})"); ax1.plot([0,1],[0,1],'--',alpha=0.6)
    ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR"); ax1.set_title("ROC"); ax1.legend()

    ax2=ax[2]; ax2.plot(prob_pred,prob_true,marker="o"); ax2.plot([0,1],[0,1],'--',alpha=0.6)
    ax2.set_xlabel("Predicted prob"); ax2.set_ylabel("Observed freq"); ax2.set_title("Reliability")
    hist_ax=ax2.twinx(); counts,bins=np.histogram(y_prob,bins=CALIBRATION_BINS,range=(0,1)); centers=0.5*(bins[1:]+bins[:-1])
    if counts.max()>0: hist_ax.bar(centers, counts/max(counts.max(),1), width=(bins[1]-bins[0])*0.85, alpha=0.3)
    hist_ax.set_ylabel("Relative bin count")
    plt.tight_layout(); plt.savefig(os.path.join(save_dir,f"{sp}_{sn}_pr_roc_reliability.png"),dpi=150)
    if SHOW_FIG: plt.show(); plt.close('all')

    plt.figure(figsize=(6,4))
    plt.hist(y_prob[y_true==0],bins=40,range=(0,1),alpha=0.6,label='Neg')
    plt.hist(y_prob[y_true==1],bins=40,range=(0,1),alpha=0.6,label='Pos')
    plt.xlabel("Predicted probability"); plt.ylabel("Count"); plt.title(f"Score distribution - {name}"); plt.legend()
    plt.savefig(os.path.join(save_dir,f"{sp}_{sn}_score_hist.png"),dpi=150)
    if SHOW_FIG: plt.show(); plt.close('all')

    order=np.argsort(-y_prob); y_sorted=y_true[order]; cov=np.arange(1,len(y_true)+1)/len(y_true)
    rec2=np.cumsum(y_sorted)/max(y_true.sum(),1); lift=np.divide(rec2,np.maximum(cov,1e-12))
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    ax[0].plot(cov,rec2,label=name); ax[0].plot([0,1],[0,1],'--',alpha=0.6)
    ax[0].set_xlabel("Coverage"); ax[0].set_ylabel("Cumulative recall"); ax[0].set_title("Cumulative Gain"); ax[0].legend()
    ax[1].plot(cov,lift,label=name); ax[1].axhline(1.0,ls='--',alpha=0.6)
    ax[1].set_xlabel("Coverage"); ax[1].set_ylabel("Lift"); ax[1].set_title("Lift curve"); ax[1].legend()
    plt.tight_layout(); plt.savefig(os.path.join(save_dir,f"{sp}_{sn}_gain_lift.png"),dpi=150)
    if SHOW_FIG: plt.show(); plt.close('all')

    df_th = threshold_sweep(y_true, y_prob, n=CALIBRATION_BINS*2+1)
    metr = dict(PR_AUC=ap,NAP=nap,Base=base,ROC_AUC=roc_auc,Brier=brier,BSS=bss)
    return metr, df_th

def report_base_rate(gen, name):
    tot=0; pos=0
    for _, yb, _ in gen: tot+=len(yb); pos+=float(yb.sum())
    if tot>0: print(f"[BASE] {name} pos_rate={pos/tot:.4f} ({int(pos)}/{tot})")

# —— Comparison helpers —— #
def paired_scatter(xs, ys, labels, title, xlab, ylab, out_png):
    plt.figure(figsize=(6,6)); plt.scatter(xs,ys,alpha=0.7)
    lim=[min(xs.min(),ys.min()), max(xs.max(),ys.max())]; plt.plot(lim,lim,'--',alpha=0.5)
    for (x,y,lbl) in zip(xs,ys,labels): plt.annotate(lbl,(x,y),xytext=(3,3),textcoords='offset points',fontsize=8,alpha=0.6)
    plt.xlabel(xlab); plt.ylabel(ylab); plt.title(title)
    plt.tight_layout(); plt.savefig(out_png,dpi=150); plt.close('all'); print(f"[PLOT] {out_png}")

def diff_bar(mean_a, mean_b, a_name, b_name, out_png):
    plt.figure(figsize=(5,4)); plt.bar([0,1],[mean_a,mean_b]); plt.xticks([0,1],[a_name,b_name])
    plt.ylabel("Mean metric"); plt.title(f"Mean: {b_name} - {a_name} = {mean_b-mean_a:+.3f}")
    plt.tight_layout(); plt.savefig(out_png,dpi=150); plt.close('all'); print(f"[PLOT] {out_png}")

def plot_budget_curves(budget_dict, covers, title, out_png, mark_points=None):
    plt.figure(figsize=(8,5))
    for lbl,(hits,recs,precs) in budget_dict.items(): plt.plot(covers,hits,label=lbl)
    if mark_points:
        for mp in mark_points:
            plt.plot(mp['cov'], mp['hits'], 'o')
            plt.annotate(mp['label'], (mp['cov'], mp['hits']), xytext=(5,5), textcoords='offset points', fontsize=8)
    plt.xlabel("Coverage (alert rate)"); plt.ylabel("Hits in selection")
    plt.title(title); plt.legend(); plt.tight_layout(); plt.savefig(out_png,dpi=150); plt.close('all'); print(f"[PLOT] {out_png}")

# —— Resume I/O —— #
def pred_csv_path(tag, model_name, k_days, plan):
    fname = PRED_FILE_PATTERN.format(tag=safe_filename(tag), model=model_name, k=k_days, plan=plan)
    return os.path.join(PRED_OUT_DIR, fname)

def has_pred_file(tag, model_name, k_days, plan): return os.path.exists(pred_csv_path(tag, model_name, k_days, plan))

def load_pred_file(tag, model_name, k_days, plan):
    df = pd.read_csv(pred_csv_path(tag, model_name, k_days, plan))
    p = df["p_cal"].to_numpy(np.float32) if "p_cal" in df.columns else df["p_uncal"].to_numpy(np.float32)
    y = df["y_true"].to_numpy(np.float32); return y, p, df

# —— Dynamic batch sizing —— #
def _nearest_pow2(x, lo=TRAIN_BATCH_MIN, hi=BATCH_SIZE):
    x = max(lo, min(hi, int(x)))
    return 1 << int(np.log2(max(1, x)))

def choose_batches(n_train: int, target_min_steps: int = TARGET_MIN_STEPS,
                   train_fixed: int = BATCH_SIZE) -> Tuple[int,int,int,bool,int]:
    """
    Returns: BATCH_TRAIN, BATCH_VAL, BATCH_TEST, auto_flag, steps_if_fixed
    """
    steps_if_fixed = max(1, math.ceil(n_train / max(1, train_fixed)))
    if steps_if_fixed < target_min_steps:
        bs_needed = max(TRAIN_BATCH_MIN, n_train // max(1, target_min_steps))
        BATCH_TRAIN = _nearest_pow2(bs_needed, lo=TRAIN_BATCH_MIN, hi=train_fixed)  # ≤ fixed
        auto = True
    else:
        BATCH_TRAIN = train_fixed
        auto = False
    BATCH_VAL  = min(VAL_BATCH_MAX,  max(256, BATCH_TRAIN))
    BATCH_TEST = min(TEST_BATCH_MAX, max(512,  BATCH_TRAIN * 2))
    return BATCH_TRAIN, BATCH_VAL, BATCH_TEST, auto, steps_if_fixed

# =========================
# Runner: one "region tag" (can be combined region)
# =========================
def run_gru_for_one_region(df_reg: pd.DataFrame,
                           region: str,
                           lat_zone: str,
                           tag: str,
                           summary_rows: List[Dict[str, Any]],
                           PRED_STORE: Dict[Tuple[str,str,int,str], Dict[str, Any]]) -> None:
    """
    Run full GRU pipeline for a region tag: training, prediction, plots, per-region Excel.
    df_reg already contains only this region (or concatenated subregions).
    """
    base_fig_dir = os.path.join(FIG_DIR, safe_filename(tag)); ensure_dir(base_fig_dir)
    region_excel = os.path.join(OUT_ROOT, f"{safe_filename(tag)}_rnn.xlsx")
    region_sheets: Dict[str, pd.DataFrame] = {}
    region_summary_rows: List[Dict[str, Any]] = []

    print("\n" + "="*70)
    print(f"[REGION] {region} / {lat_zone} (tag={tag})")
    print("="*70)

    # Sort & ensure year
    df_reg = df_reg.copy()
    df_reg["datetime"] = pd.to_datetime(df_reg["datetime"])
    if "year" not in df_reg.columns:
        df_reg["year"] = df_reg["datetime"].dt.year
    df_reg = df_reg.sort_values(["lat_bin","lon_bin","datetime"]).reset_index(drop=True)

    # Feature prep
    phys_cols, prior_cols = choose_feature_columns(df_reg)
    all_feats_all = phys_cols + prior_cols
    for c in all_feats_all:
        df_reg[c] = pd.to_numeric(df_reg[c], errors='coerce')
    df_reg['Wildfire'] = pd.to_numeric(df_reg['Wildfire'], errors='coerce').fillna(0.0).clip(0,1).astype('float32')
    if 'rnn_weight' not in df_reg.columns:
        df_reg['rnn_weight'] = 1.0
    df_reg['rnn_weight'] = pd.to_numeric(df_reg['rnn_weight'], errors='coerce').fillna(1.0).clip(0,100).astype('float32')
    med = df_reg[all_feats_all].median(numeric_only=True)
    df_reg[all_feats_all] = df_reg[all_feats_all].fillna(med)

    model_flags = (['phys'] if RUN_MODE=='phys'
                   else ['prior'] if RUN_MODE=='prior'
                   else ['phys','prior'])

    def build_idx_for(k_days, seq_len=SEQ_LEN):
        return build_index_lists_for_split(df_reg, seq_len, k_days,
                                           TRAIN_MAX_YEAR, VAL_YEAR, TEST_MIN_YEAR,
                                           ENFORCE_GAP)

    for model_flag in model_flags:
        use_prior = (model_flag=='prior') and ("logit_cb" in prior_cols)
        feat_cols = phys_cols + (prior_cols if use_prior else [])
        model_name = "GRU_with_prior" if use_prior else "GRU_no_prior"
        input_shape = (SEQ_LEN, len(feat_cols))

        for k_days in PRED_K_LIST:

            # Build indices first (to assess window size / steps)
            idx_tr, idx_va, idx_te = build_idx_for(k_days, SEQ_LEN)
            print(f"[INDEX] tag={tag} model={model_name} k={k_days}  "
                  f"train={len(idx_tr)}  val={len(idx_va)}  test={len(idx_te)}  (SEQ_LEN={SEQ_LEN})")

            if RUN_PLAN=='smoke':
                idx_tr = maybe_subsample_indices(idx_tr, 60_000, "train")
                idx_va = maybe_subsample_indices(idx_va, 30_000, "val")
                idx_te = maybe_subsample_indices(idx_te,120_000, "test")
            if len(idx_tr)==0 or len(idx_te)==0:
                print("[SKIP] 样本不足。")
                continue

            # Dynamic batch sizing
            BATCH_TRAIN, BATCH_VAL, BATCH_TEST, auto_batch, steps_if_fixed = choose_batches(len(idx_tr))
            print(f"[BATCH] would_be_steps@fixed1024={steps_if_fixed}  "
                  f"-> train={BATCH_TRAIN} ({'auto' if auto_batch else 'fixed'}), "
                  f"val={BATCH_VAL}, test={BATCH_TEST}")

            have_pred_file = has_pred_file(tag, model_name, k_days, RUN_PLAN)

            # Resume logic
            if RESUME_MODE=="auto" and have_pred_file and not (AUTO_RERUN_IF_FEW_STEPS and auto_batch):
                print(f"[RESUME] use existing predictions: {tag}, {model_name}, k={k_days}")
                y, p, meta = load_pred_file(tag, model_name, k_days, RUN_PLAN)
                PRED_STORE[(region, lat_zone, k_days, model_name)] = dict(y=y, p=p, meta=meta)
                save_dir = os.path.join(base_fig_dir, f"k{k_days}")
                metr, df_th = evaluate_and_plot_all(f"{tag}_k{k_days}",
                                                    f"{model_name} ({region}_{lat_zone})",
                                                    y, p, save_dir)
                summary_rows.append({"region":region,"lat_zone":lat_zone,
                                     "model":model_name,"horizon":k_days,
                                     "Temp":np.nan, **metr})
                region_summary_rows.append({"region":region,"lat_zone":lat_zone,
                                            "model":model_name,"horizon":k_days,
                                            "Temp":np.nan, **metr})
                region_sheets[f"{tag}_{model_flag}_k{k_days}"] = df_th.copy()
                continue

            if have_pred_file and AUTO_RERUN_IF_FEW_STEPS and auto_batch:
                print(f"[RERUN] {tag}, {model_name}, k={k_days} "
                      f"existing CSV but steps={steps_if_fixed} (<{TARGET_MIN_STEPS}), retraining...")

            # Train
            gen_tr = KWindowSequence(df_reg, feat_cols, SEQ_LEN, k_days,
                                     idx_tr, batch_size=BATCH_TRAIN,
                                     shuffle=True,  name="train")
            gen_va = KWindowSequence(df_reg, feat_cols, SEQ_LEN, k_days,
                                     idx_va, batch_size=BATCH_VAL,
                                     shuffle=False, name="val")
            gen_te = KWindowSequence(df_reg, feat_cols, SEQ_LEN, k_days,
                                     idx_te, batch_size=BATCH_TEST,
                                     shuffle=False, name="test")
            print(f"[STEPS] steps/epoch={len(gen_tr)}  val_steps={len(gen_va)}")
            report_base_rate(gen_tr, "train")
            report_base_rate(gen_va, "val")

            loss_obj, metrics, optimizer = get_loss_and_metrics()
            model = (build_gru_with_prior(input_shape, feat_cols, prior_cols,
                                          loss_obj, metrics, optimizer)
                     if use_prior else
                     build_gru_model(input_shape, loss_obj, metrics, optimizer))

            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor=("val_loss" if FAST_MODE else "val_PR_AUC"),
                    mode=("min" if FAST_MODE else "max"),
                    patience=(2 if RUN_PLAN=='smoke' else PATIENCE),
                    restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor=("val_loss" if FAST_MODE else "val_PR_AUC"),
                    mode=("min" if FAST_MODE else "max"),
                    factor=0.5, patience=1, verbose=1)
            ]
            _ = model.fit(gen_tr, validation_data=gen_va,
                          steps_per_epoch=(1500 if FAST_MODE else None),
                          validation_steps=(200 if FAST_MODE else None),
                          epochs=(4 if RUN_PLAN=='smoke' else EPOCHS),
                          callbacks=callbacks, verbose=1)

            # Temperature scaling on val
            yv_all=[]; pv_all=[]
            for Xb,yb,wb in gen_va:
                pv_all.append(model.predict(Xb, batch_size=len(Xb), verbose=0).ravel().astype("float32"))
                yv_all.append(yb.astype("float32"))
            if len(yv_all)==0:
                T = 1.0
            else:
                yv_all = np.concatenate(yv_all)
                pv_all = np.concatenate(pv_all)
                T = fit_temperature(yv_all, pv_all)

            # Test predictions (save calibrated)
            yte_all=[]; pte_uncal=[]; pte_cal=[]
            for Xb,yb,wb in gen_te:
                pu = model.predict(Xb,batch_size=len(Xb),verbose=0).ravel().astype("float32")
                pc = apply_temperature(pu, T)
                yte_all.append(yb.astype("float32"))
                pte_uncal.append(pu)
                pte_cal.append(pc)
            yte_all = np.concatenate(yte_all)
            pte_uncal = np.concatenate(pte_uncal)
            pte_cal = np.concatenate(pte_cal)

            # Store calibrated probs
            PRED_STORE[(region, lat_zone, k_days, model_name)] = dict(y=yte_all, p=pte_cal, meta=None)

            # Plots/eval (historical consistency: plot with uncalibrated)
            save_dir = os.path.join(base_fig_dir, f"k{k_days}")
            metr, df_th = evaluate_and_plot_all(f"{tag}_k{k_days}",
                                                f"{model_name} ({region}_{lat_zone})",
                                                yte_all, pte_uncal, save_dir)

            summary_rows.append({"region":region,"lat_zone":lat_zone,
                                 "model":model_name,"horizon":k_days,
                                 "Temp":T, **metr})
            region_summary_rows.append({"region":region,"lat_zone":lat_zone,
                                        "model":model_name,"horizon":k_days,
                                        "Temp":T, **metr})
            region_sheets[f"{tag}_{model_flag}_k{k_days}"] = df_th.copy()

            # Save prediction CSV
            if SAVE_PREDICTIONS:
                df_pred = pd.DataFrame({
                    "y_true": yte_all.astype("float32"),
                    "p_uncal": pte_uncal,
                    "p_cal": pte_cal,
                    "model": model_name,
                    "k_days": k_days,
                    "region": region,
                    "lat_zone": lat_zone
                })
                out_path = pred_csv_path(tag, model_name, k_days, RUN_PLAN)
                df_pred.to_csv(out_path, index=False)
                print(f"[SAVE] predictions -> {out_path}")

            del gen_tr, gen_va, gen_te, model
            gc.collect()

    # Per-region Excel (threshold sweep + summary)
    if region_sheets:
        df_region_sum = pd.DataFrame(region_summary_rows)
        cols = ["region","lat_zone","model","horizon","Temp",
                "PR_AUC","NAP","Base","ROC_AUC","Brier","BSS"]
        for c in cols:
            if c not in df_region_sum.columns:
                df_region_sum[c] = np.nan
        region_sheets["summary"] = df_region_sum[cols].copy()
        write_region_excel(region_excel, region_sheets)
        print(f"[REGION] Excel -> {region_excel}")
    else:
        print(f"[REGION] no sheets for {region}_{lat_zone}, skip Excel.")

# =========================
# Main
# =========================
if __name__ == "__main__":
    print("[INFO] TensorFlow:", tf.__version__)
    ensure_dir(OUT_ROOT); ensure_dir(FIG_DIR)

    summary_rows: List[Dict[str, Any]] = []
    PRED_STORE: Dict[Tuple[str,str,int,str], Dict[str, Any]] = {}

    # ========== Mode 1: standard (each region×lat_zone) ==========
    if TRAIN_MODE == "standard":
        if REGION_OVERRIDE and LATZONE_OVERRIDE:
            EXP_REGS = [(REGION_OVERRIDE, LATZONE_OVERRIDE)]
            df_stats = pd.DataFrame([{
                'region': REGION_OVERRIDE,
                'lat_zone': LATZONE_OVERRIDE,
                'n_fire': np.nan,
                'n_total': np.nan,
                'fire_rate': np.nan
            }])
        else:
            EXP_REGS, df_stats = auto_list_all_regions(PARQUET_ROOT)
            print(f"[SELECT] 将跑全部 {len(EXP_REGS)} 个 region×lat_zone。")

        for ridx, (region, lat_zone) in enumerate(EXP_REGS, 1):
            if CLEAR_GPU_BETWEEN_JOBS:
                K.clear_session(); gc.collect()
            if ridx > 1 and WAIT_BETWEEN_REGIONS_SEC > 0:
                time.sleep(WAIT_BETWEEN_REGIONS_SEC)

            df_reg = load_region_latzone_df(PARQUET_ROOT, region, lat_zone)
            if df_reg.empty:
                continue

            tag = f"{region}_{lat_zone}".replace(" ","")
            run_gru_for_one_region(df_reg, region, lat_zone, tag,
                                   summary_rows, PRED_STORE)

    # ========== Mode 2: combined regions only ==========
    elif TRAIN_MODE == "combined":
        print("[MODE] TRAIN_MODE = 'combined' → 只训练 COMBINED_REGION_GROUPS 中的合并区域")
        for idx, (combo_name, pair_list) in enumerate(COMBINED_REGION_GROUPS.items(), 1):
            if CLEAR_GPU_BETWEEN_JOBS:
                K.clear_session(); gc.collect()
            if idx > 1 and WAIT_BETWEEN_REGIONS_SEC > 0:
                time.sleep(WAIT_BETWEEN_REGIONS_SEC)

            print(f"\n[COMBINED] {combo_name} 由以下子区域组成:")
            dfs = []
            for (reg0, lat0) in pair_list:
                print(f"  - {reg0} / {lat0}")
                df_sub = load_region_latzone_df(PARQUET_ROOT, reg0, lat0)
                if df_sub.empty:
                    print(f"    [WARN] {reg0}/{lat0} 无数据，跳过该子区域")
                    continue
                dfs.append(df_sub)
            if not dfs:
                print(f"[COMBINED] {combo_name} 没有可用子区域，跳过")
                continue

            df_combo = pd.concat(dfs, ignore_index=True)
            # Tag combined region (affects filenames only)
            df_combo["region"] = combo_name
            df_combo["lat_zone"] = COMBINED_LATZONE_TAG

            tag = combo_name
            run_gru_for_one_region(df_combo,
                                   region=combo_name,
                                   lat_zone=COMBINED_LATZONE_TAG,
                                   tag=tag,
                                   summary_rows=summary_rows,
                                   PRED_STORE=PRED_STORE)
    else:
        raise ValueError("TRAIN_MODE 必须是 'standard' 或 'combined'")

    # =========================
    # Combined-region summary (only meaningful for standard mode)
    # =========================
    if TRAIN_MODE == "standard" and COMBINED_REGION_GROUPS:
        combo_excel_sheets: Dict[str, pd.DataFrame] = {}
        for combo_name, pair_list in COMBINED_REGION_GROUPS.items():
            for k_days in PRED_K_LIST:
                for model_name in ["GRU_with_prior", "GRU_no_prior"]:
                    ys = []
                    ps = []
                    for (reg0, lat0) in pair_list:
                        key = (reg0, lat0, k_days, model_name)
                        d = PRED_STORE.get(key)
                        if d is None:
                            print(f"[COMBO] skip missing member: {reg0}, {lat0}, k={k_days}, model={model_name}")
                            continue
                        ys.append(d["y"])
                        ps.append(d["p"])
                    if not ys:
                        continue

                    y_all = np.concatenate(ys)
                    p_all = np.concatenate(ps)

                    combo_tag = f"{combo_name}_k{k_days}"
                    save_dir = os.path.join(FIG_DIR, safe_filename(combo_name), f"k{k_days}")
                    metr, df_th = evaluate_and_plot_all(
                        combo_tag,
                        f"{model_name} ({combo_name})",
                        y_all,
                        p_all,
                        save_dir
                    )

                    summary_rows.append({
                        "region": combo_name,
                        "lat_zone": COMBINED_LATZONE_TAG,
                        "model": model_name,
                        "horizon": k_days,
                        "Temp": np.nan,
                        **metr
                    })

                    sheet_key = f"{combo_name}_{model_name}_k{k_days}"
                    combo_excel_sheets[sheet_key] = df_th.copy()

                    PRED_STORE[(combo_name, COMBINED_LATZONE_TAG, k_days, model_name)] = dict(
                        y=y_all,
                        p=p_all,
                        meta=None
                    )

        if combo_excel_sheets:
            combo_excel_path = os.path.join(OUT_ROOT, "combined_regions_rnn.xlsx")
            write_region_excel(combo_excel_path, combo_excel_sheets)
            print(f"[COMBO] Excel -> {combo_excel_path}")

    # =========================
    # Prior-only / ΔValue / budget curves / lead-time payoff
    # =========================
    def compute_prior_only(region, lat_zone, k_days):
        df_reg = load_region_latzone_df(PARQUET_ROOT, region, lat_zone)
        if df_reg.empty or "logit_cb" not in df_reg.columns: return None
        idx_tr, idx_va, idx_te = build_index_lists_for_split(df_reg, SEQ_LEN, k_days, TRAIN_MAX_YEAR, VAL_YEAR, TEST_MIN_YEAR, ENFORCE_GAP)
        if len(idx_te)==0: return None
        groups = list(df_reg.groupby(["lat_bin","lon_bin"]).groups.items()); gid_to_rows=[rows for (_,rows) in groups]
        y_list=[]; p_list=[]
        for (gid,t) in idx_te:
            g=df_reg.iloc[gid_to_rows[gid]]
            logit=float(g["logit_cb"].iloc[t-1]); p=1/(1+np.exp(-logit))
            y=1.0 if g["Wildfire"].iloc[t+1:t+1+k_days].max()>0 else 0.0
            y_list.append(y); p_list.append(p)
        return np.array(y_list,np.float32), np.array(p_list,np.float32)

    # ΔValue (k=14), budget recommendations, lead-time payoff (Top-3%)
    delta_rows=[]
    budget_marks=[]
    lead_rows=[]

    done_regions = sorted({(r,l) for (r,l,_,_) in PRED_STORE.keys()})
    for (region, lat_zone) in done_regions:
        # Budget curve (k=14, with_prior)
        d = PRED_STORE.get((region, lat_zone, K_FOR_BUDGET, "GRU_with_prior"))
        if d is not None:
            hits, recs, precs = budget_hits_curve(d['y'], d['p'], BUDGET_GRID)
            idx_ok = np.where(precs >= FIXED_PRECISION_TARGET)[0]
            i_star = idx_ok[-1] if len(idx_ok)>0 else int(np.argmin(np.abs(BUDGET_GRID - TOP_K_PER_MILLE/1000.0)))
            budget_marks.append(dict(region=region, lat_zone=lat_zone, cov=float(BUDGET_GRID[i_star]), hits=int(hits[i_star])))
            out_png = os.path.join(FIG_DIR, f"budget_{safe_filename(region)}_{safe_filename(lat_zone)}_k{K_FOR_BUDGET}.png")
            plot_budget_curves({f"{region}_{lat_zone}":(hits,recs,precs)}, BUDGET_GRID,
                               f"Budget curve (k={K_FOR_BUDGET}) — {region}_{lat_zone}",
                               out_png,
                               mark_points=[dict(cov=BUDGET_GRID[i_star], hits=int(hits[i_star]), label="★ Rec.")])

        # ΔValue: prior / no_prior / with_prior at k=14
        prior_pair = compute_prior_only(region, lat_zone, 14)
        def topk(store_key):
            dd = PRED_STORE.get(store_key)
            return None if dd is None else (
                precision_at_topk(dd['y'], dd['p'], 0.01),
                precision_at_topk(dd['y'], dd['p'], 0.03),
                precision_at_topk(dd['y'], dd['p'], 0.05)
            )
        prior_t = (np.nan,np.nan,np.nan) if prior_pair is None else (
            precision_at_topk(prior_pair[0], prior_pair[1], 0.01),
            precision_at_topk(prior_pair[0], prior_pair[1], 0.03),
            precision_at_topk(prior_pair[0], prior_pair[1], 0.05)
        )
        no_t  = topk((region,lat_zone,14,"GRU_no_prior"))
        yes_t = topk((region,lat_zone,14,"GRU_with_prior"))
        if prior_t or no_t or yes_t:
            row=dict(region=region, lat_zone=lat_zone)
            labs=["top1","top3","top5"]
            for i,lab in enumerate(labs):
                pv=prior_t[i] if prior_t is not None else np.nan
                nv=no_t[i] if no_t is not None else np.nan
                yv=yes_t[i] if yes_t is not None else np.nan
                row[f"prior_{lab}"]=pv; row[f"no_prior_{lab}"]=nv; row[f"with_prior_{lab}"]=yv
                row[f"gain_yes_vs_no_{lab}"] = (yv-nv) if (not np.isnan(yv) and not np.isnan(nv)) else np.nan
                row[f"gain_yes_vs_prior_{lab}"] = (yv-pv) if (not np.isnan(yv) and not np.isnan(pv)) else np.nan
            delta_rows.append(row)

        # Lead-time payoff (Top-3% budget)
        vals={}
        for kk in [7,14,21]:
            dd=PRED_STORE.get((region,lat_zone,kk,"GRU_with_prior"))
            if dd is None: continue
            vals[kk]=precision_at_topk(dd['y'], dd['p'], 0.03)
        if len(vals)>=2:
            lead_rows.append(dict(region=region, lat_zone=lat_zone, **{f"k{kk}_top3":v for kk,v in vals.items()}))

    # Export ΔValue & lead-time payoff
    if len(delta_rows)>0:
        df_delta=pd.DataFrame(delta_rows)
        df_delta.to_csv(os.path.join(OUT_ROOT,"delta_value_topk_k14.csv"), index=False)
        print(f"[DELTA] -> {os.path.join(OUT_ROOT,'delta_value_topk_k14.csv')}")
    if len(lead_rows)>0:
        df_lead=pd.DataFrame(lead_rows)
        df_lead.to_csv(os.path.join(OUT_ROOT,"leadtime_payoff_top3.csv"), index=False)
        print(f"[LEAD]  -> {os.path.join(OUT_ROOT,'leadtime_payoff_top3.csv')}")

    # =========================
    # Comparison plots: k=7 vs 14; with vs without prior
    # =========================
    def paired_metric_across_regions(metric_func, keys_a, keys_b, title, tag):
        xs=[]; ys=[]; labels=[]
        for (region,lat_zone) in done_regions:
            ka = keys_a(region,lat_zone); kb = keys_b(region,lat_zone)
            if ka not in PRED_STORE or kb not in PRED_STORE: continue
            da,db = PRED_STORE[ka], PRED_STORE[kb]
            xs.append(metric_func(da['y'], da['p'])); ys.append(metric_func(db['y'], db['p']))
            labels.append(f"{region}_{lat_zone}")
        if len(xs)==0: return
        xs=np.array(xs); ys=np.array(ys)
        paired_scatter(xs,ys,labels,title,"A","B", os.path.join(CMP_DIR,f"paired_{tag}.png"))
        diff_bar(xs.mean(), ys.mean(), "A mean", "B mean", os.path.join(CMP_DIR,f"paired_mean_{tag}.png"))

    auc_metric = lambda y,p: roc_auc_score(y,p)

    # k=7 vs 14 (with_prior)
    paired_metric_across_regions(
        auc_metric,
        keys_a=lambda r,lz: (r,lz,7,"GRU_with_prior"),
        keys_b=lambda r,lz: (r,lz,14,"GRU_with_prior"),
        title="ROC-AUC: k=7 vs k=14 (with_prior)",
        tag="k7_vs_k14_withprior"
    )

    # with_prior vs no_prior (k=14)
    paired_metric_across_regions(
        auc_metric,
        keys_a=lambda r,lz: (r,lz,14,"GRU_no_prior"),
        keys_b=lambda r,lz: (r,lz,14,"GRU_with_prior"),
        title="ROC-AUC: no_prior vs with_prior (k=14)",
        tag="nprior_vs_yprior_k14"
    )

    # =========================
    # Global summary
    # =========================
    if len(summary_rows)>0:
        df_sum_all=pd.DataFrame(summary_rows)
        cols=["region","lat_zone","model","horizon","Temp","PR_AUC","NAP","Base","ROC_AUC","Brier","BSS"]
        for c in cols:
            if c not in df_sum_all.columns: df_sum_all[c]=np.nan
        out=os.path.join(OUT_ROOT,"rnn_results_summary_GLOBAL.xlsx")
        with pd.ExcelWriter(out, engine="openpyxl", mode="w") as xw:
            df_sum_all[cols].to_excel(xw, sheet_name="summary", index=False)
        print(f"[GLOBAL] summary -> {out}")
    else:
        print("[GLOBAL] 没有可写入的 summary。")

    print(f"[DONE] Figures -> {FIG_DIR} (comparisons in {CMP_DIR})")
    print(f"[DONE] Predictions -> {PRED_OUT_DIR} (SAVE_PREDICTIONS={SAVE_PREDICTIONS})")
