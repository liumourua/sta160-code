# -*- coding: utf-8 -*-
"""
Wildfire post-fire hazard analysis (resumable, run-all)
"""

import os
import re
import gc
import json
import time
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import matplotlib.pyplot as plt

# =========================
# Config
# =========================

PARQUET_ROOT = r'D:\桌面\大学\大四\第一学期\sta160\project\Wildfire_CLEANED_WITH_OOF_LITE_parquet_region_latzone'

OUT_ROOT = r'D:\桌面\大学\大四\第一学期\sta160\project\hazard2'
FIG_DIR = os.path.join(OUT_ROOT, "figs")
CSV_DIR = os.path.join(OUT_ROOT, "csv")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

# Run all regions; limit how many curves are shown on global overlays (display only; stats remain full)
RUN_ALL_REGIONS = True
TOP_K_FOR_FIGS = 10

MAX_WAIT_DAYS_FOR_HAZARD = 60    # x-axis horizon for hazard curves
MIN_COOLDOWN_DAYS = 3            # merge fires <3 days apart into one episode
MIN_INTERVALS_PER_REGION = 30    # min intervals to plot trends/curves

# Hazard bins (inclusive) used in summary
HAZARD_BIN_DEFS = [
    (1, 3),
    (4, 7),
    (8, 14),
    (15, 30),
    (31, 9999)
]

# Combined regions (use child waiting_intervals_* only; no recomputation from parquet)
ENABLE_COMBINED_REGIONS = True

# key = (combined region name, combined lat_zone name)
# value = list of (region, lat_zone) members — must match parquet values exactly
COMBINED_REGION_GROUPS: Dict[Tuple[str, str], List[Tuple[str, str]]] = {
    ("North East", "NorthBand"): [
        ("North East", "Far North"),
        ("North East", "North"),
    ],
    ("East", "NorthBand"): [
        ("East", "Far North"),
        ("East", "North"),
    ],
    ("Central West", "Mid+South"): [
        ("Central West", "Mid"),
        ("Central West", "South"),
    ],
}

# Resuming
RESUME_MODE = "auto"  # "auto"=skip if signature matches; "force"=recompute all
REBUILD_FIGS_FROM_CSV_IF_MISSING = True  # if skipped, rebuild figures from CSVs when absent

# =========================
# Utilities
# =========================

def safe_filename(s: str, max_len: int = 140) -> str:
    s = re.sub(r'[<>:"/\\|?*]+', '_', s)
    s = re.sub(r'\s+', '_', s)
    return s[:max_len].strip('._')

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def tag_of(region: str, lat_zone: str) -> str:
    return f"{region}_{lat_zone}".replace(" ", "_")

def paths_for(tag: str) -> Dict[str, str]:
    """Per-region output paths."""
    return dict(
        intervals_csv = os.path.join(CSV_DIR, f"waiting_intervals_{safe_filename(tag)}.csv"),
        trend_csv     = os.path.join(CSV_DIR, f"waiting_trend_{safe_filename(tag)}.csv"),
        hazard_csv    = os.path.join(CSV_DIR, f"hazard_curve_{safe_filename(tag)}.csv"),
        hist_png      = os.path.join(FIG_DIR, safe_filename(tag), f"waiting_hist_{safe_filename(tag)}.png"),
        trend_png     = os.path.join(FIG_DIR, safe_filename(tag), f"waiting_trend_{safe_filename(tag)}.png"),
        hazard_png    = os.path.join(FIG_DIR, safe_filename(tag), f"hazard_curve_{safe_filename(tag)}.png"),
        meta_json     = os.path.join(CSV_DIR, f"hazard_meta_{safe_filename(tag)}.json"),
        fig_dir       = os.path.join(FIG_DIR, safe_filename(tag)),
    )

def current_signature() -> Dict[str, Any]:
    """Parameter signature used for resuming."""
    return dict(
        min_cooldown_days = int(MIN_COOLDOWN_DAYS),
        max_wait_days     = int(MAX_WAIT_DAYS_FOR_HAZARD),
        bins              = list(map(list, HAZARD_BIN_DEFS)),
        dataset_root      = str(PARQUET_ROOT)
    )

def has_finished_and_match_signature(tag: str) -> bool:
    p = paths_for(tag)
    need = [p["intervals_csv"], p["hazard_csv"], p["meta_json"]]
    if not all(os.path.exists(x) for x in need):
        return False
    try:
        with open(p["meta_json"], "r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta.get("signature") == current_signature()
    except Exception:
        return False

# =========================
# Region listing & loading
# =========================

def list_all_regions(root_path: str) -> Tuple[List[Tuple[str,str]], pd.DataFrame]:
    dataset = ds.dataset(root_path, format="parquet", partitioning="hive")
    tbl = dataset.to_table(columns=["region", "lat_zone", "Wildfire"])
    df_idx = tbl.to_pandas()

    stats = (df_idx.groupby(["region", "lat_zone"])["Wildfire"]
             .agg(["sum", "count"])
             .rename(columns={"sum": "n_fire", "count": "n_total"}))
    stats["fire_rate"] = stats["n_fire"] / stats["n_total"]
    stats = stats.reset_index().sort_values("n_fire", ascending=False)
    pairs = [(r, z) for r, z in stats[["region", "lat_zone"]].itertuples(index=False)]
    return pairs, stats

def load_region_latzone_df(root_path: str, region: str, lat_zone: str) -> pd.DataFrame:
    dataset = ds.dataset(root_path, format="parquet", partitioning="hive")
    flt = (ds.field("region") == region) & (ds.field("lat_zone") == lat_zone)
    table = dataset.to_table(filter=flt)
    df = table.to_pandas()

    if df.empty:
        print(f"[LOAD] {region}/{lat_zone} 行数 0，跳过。")
        return df

    df["datetime"] = pd.to_datetime(df["datetime"])
    if "year" not in df.columns:
        df["year"] = df["datetime"].dt.year

    df = df.sort_values(["lat_bin", "lon_bin", "datetime"]).reset_index(drop=True)
    df["Wildfire"] = pd.to_numeric(df["Wildfire"], errors="coerce").fillna(0.0)
    df["Wildfire"] = df["Wildfire"].clip(0, 1).astype("int8")
    print(f"[LOAD] {region}/{lat_zone} rows={len(df)}, fires={int(df['Wildfire'].sum())}")
    return df

# =========================
# Compute waiting intervals (with cooldown merging)
# =========================

def compute_waiting_intervals_for_region(
    df_reg: pd.DataFrame,
    min_cooldown: int = 3
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for (latb, lonb), g in df_reg.groupby(["lat_bin", "lon_bin"]):
        g = g.sort_values("datetime")
        y = g["Wildfire"].to_numpy().astype(int)
        dates = g["datetime"].to_numpy()

        episodes: List[Tuple[np.datetime64, np.datetime64]] = []
        in_fire = False
        start = None

        n = len(y)
        for t in range(n):
            if (not in_fire) and (y[t] == 1):
                in_fire = True
                start = dates[t]

            is_end = in_fire and ((t == n - 1) or (y[t] == 1 and y[t+1] == 0))
            if is_end:
                end = dates[t]
                episodes.append((start, end))
                in_fire = False

        if len(episodes) <= 1:
            continue

        merged: List[List[np.datetime64]] = [list(episodes[0])]
        for s, e in episodes[1:]:
            prev_s, prev_e = merged[-1]
            gap_days = int((s - prev_e) / np.timedelta64(1, "D"))
            if gap_days < min_cooldown:
                if e > prev_e:
                    merged[-1][1] = e
            else:
                merged.append([s, e])

        if len(merged) <= 1:
            continue

        for i in range(len(merged) - 1):
            s1, e1 = merged[i]
            s2, e2 = merged[i + 1]
            gap_days = int((s2 - e1) / np.timedelta64(1, "D"))
            if gap_days <= 0:
                continue
            rows.append(dict(
                region=str(g["region"].iloc[0]),
                lat_zone=str(g["lat_zone"].iloc[0]),
                lat_bin=float(latb),
                lon_bin=float(lonb),
                start_year=int(pd.Timestamp(s1).year),
                gap_days=float(gap_days)
            ))

    return pd.DataFrame(rows)

# =========================
# Plotting & hazard
# =========================

def plot_waiting_hist(df_wait: pd.DataFrame, region: str, lat_zone: str,
                      max_wait: int, out_dir: str, out_path: str):
    if df_wait.empty:
        return
    ensure_dir(out_dir)
    gaps = df_wait["gap_days"].values
    plt.figure(figsize=(6, 4))
    plt.hist(np.clip(gaps, 1, max_wait),
             bins=np.arange(1, max_wait + 2),
             edgecolor="k", alpha=0.7)
    plt.xlabel("Days between consecutive fires")
    plt.ylabel("Count")
    plt.title(f"Inter-fire waiting times – {region} / {lat_zone}")
    plt.xlim(1, max_wait + 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[FIG] waiting_hist -> {out_path}")

def plot_waiting_trend(df_wait: pd.DataFrame, region: str, lat_zone: str,
                       min_intervals: int, out_dir: str,
                       csv_path: str, out_path: str):
    if df_wait.empty or len(df_wait) < min_intervals:
        return
    ensure_dir(out_dir)
    grp = df_wait.groupby("start_year")["gap_days"]
    df_year = grp.agg(
        median="median",
        p25=lambda x: np.percentile(x, 25),
        p75=lambda x: np.percentile(x, 75),
        n="count"
    ).reset_index().sort_values("start_year")
    if len(df_year) <= 1:
        return
    df_year.to_csv(csv_path, index=False)
    print(f"[CSV] waiting_trend -> {csv_path}")

    years = df_year["start_year"].values
    med = df_year["median"].values
    p25 = df_year["p25"].values
    p75 = df_year["p75"].values
    plt.figure(figsize=(6, 4))
    plt.plot(years, med, marker="o", label="median")
    plt.fill_between(years, p25, p75, alpha=0.2, label="IQR")
    plt.xlabel("Start year of interval")
    plt.ylabel("Days between fires")
    plt.title(f"Inter-fire waiting time trend – {region} / {lat_zone}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[FIG] waiting_trend -> {out_path}")

def compute_daily_hazard(df_wait: pd.DataFrame, max_wait: int) -> pd.DataFrame:
    gaps = df_wait["gap_days"].values.astype(int)
    gaps = gaps[gaps > 0]
    if len(gaps) == 0:
        return pd.DataFrame(columns=["day", "n_events", "n_at_risk", "hazard", "cum_prob"])
    total = len(gaps)
    days = np.arange(1, max_wait + 1)
    n_events = np.array([(gaps == d).sum() for d in days], dtype=float)
    n_at_risk = np.array([(gaps >= d).sum() for d in days], dtype=float)
    hazard = n_events / np.maximum(n_at_risk, 1.0)
    cum_prob = np.cumsum(n_events) / float(total)
    return pd.DataFrame({
        "day": days,
        "n_events": n_events,
        "n_at_risk": n_at_risk,
        "hazard": hazard,
        "cum_prob": cum_prob
    })

def plot_hazard_curve(df_h: pd.DataFrame, region: str, lat_zone: str,
                      out_dir: str, csv_path: str, out_path: str):
    if df_h.empty:
        return
    ensure_dir(out_dir)
    df_h.to_csv(csv_path, index=False)
    print(f"[CSV] hazard_curve -> {csv_path}")
    d = df_h["day"].values
    hz = df_h["hazard"].values
    cp = df_h["cum_prob"].values
    plt.figure(figsize=(6, 4))
    plt.plot(d, hz, label="daily hazard")
    plt.plot(d, cp, linestyle="--", label="P(T ≤ d)")
    plt.xlabel("Days since last fire")
    plt.ylabel("Probability")
    plt.title(f"Post-fire hazard – {region} / {lat_zone}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[FIG] hazard_curve -> {out_path}")

def compute_binned_hazard(gaps: np.ndarray) -> Dict[str, float]:
    gaps = gaps.astype(float)
    res = {}
    for lo, hi in HAZARD_BIN_DEFS:
        at_risk = (gaps >= lo).sum()
        in_bin = ((gaps >= lo) & (gaps <= hi)).sum()
        h = in_bin / max(at_risk, 1)
        label = f"{lo}-{hi if hi < 9999 else 'max'}"
        res[f"haz_{label}"] = float(h)
    return res

# =========================
# Global plots
# =========================

def plot_global_median_bar(df_summary: pd.DataFrame, out_dir: str):
    if df_summary.empty:
        return
    ensure_dir(out_dir)
    df = df_summary.sort_values("gap_median")
    labels = [f"{r}_{z}" for r, z in zip(df["region"], df["lat_zone"])]
    vals = df["gap_median"].values
    plt.figure(figsize=(8, 5))
    plt.barh(labels, vals)
    plt.xlabel("Median days between fires")
    plt.title("Median inter-fire waiting time by region")
    plt.gca().invert_yaxis()
    out_path = os.path.join(out_dir, "global_median_gap_bar.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[FIG] global median bar -> {out_path}")

def plot_global_hazard_overlay(hazard_curves: Dict[str, pd.DataFrame],
                               out_dir: str,
                               max_regions: int = 10):
    if not hazard_curves:
        return
    ensure_dir(out_dir)
    plt.figure(figsize=(7, 5))
    for i, (tag, df_h) in enumerate(hazard_curves.items()):
        if i >= max_regions:
            break
        if df_h is None or df_h.empty:
            continue
        plt.plot(df_h["day"], df_h["cum_prob"], label=tag)
    plt.xlabel("Days since last fire")
    plt.ylabel("P(T ≤ d)")
    plt.title("Post-fire cumulative probability by region")
    plt.legend(fontsize=8)
    out_path = os.path.join(out_dir, "global_hazard_cumprob_overlay.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[FIG] global hazard overlay -> {out_path}")

# =========================
# Rebuild figures from CSVs / derive summary from CSVs
# =========================

def replot_from_csv_if_needed(region: str, lat_zone: str, tag: str, p: Dict[str,str]):
    """Rebuild figures from CSVs if missing (no recomputation)."""
    if not REBUILD_FIGS_FROM_CSV_IF_MISSING:
        return
    ensure_dir(p["fig_dir"])

    # Histogram
    if (not os.path.exists(p["hist_png"])) and os.path.exists(p["intervals_csv"]):
        df_wait = pd.read_csv(p["intervals_csv"])
        plot_waiting_hist(df_wait, region, lat_zone, MAX_WAIT_DAYS_FOR_HAZARD, p["fig_dir"], p["hist_png"])

    # Trend
    if (not os.path.exists(p["trend_png"])) and os.path.exists(p["trend_csv"]):
        df_year = pd.read_csv(p["trend_csv"])
        if len(df_year) > 1:
            years = df_year["start_year"].values
            med = df_year["median"].values
            p25 = df_year["p25"].values
            p75 = df_year["p75"].values
            plt.figure(figsize=(6, 4))
            plt.plot(years, med, marker="o", label="median")
            plt.fill_between(years, p25, p75, alpha=0.2, label="IQR")
            plt.xlabel("Start year of interval")
            plt.ylabel("Days between fires")
            plt.title(f"Inter-fire waiting time trend – {region} / {lat_zone}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(p["trend_png"], dpi=150)
            plt.close()
            print(f"[FIG] waiting_trend (rebuild) -> {p['trend_png']}")

    # Hazard
    if (not os.path.exists(p["hazard_png"])) and os.path.exists(p["hazard_csv"]):
        df_h = pd.read_csv(p["hazard_csv"])
        if not df_h.empty:
            plot_hazard_curve(df_h, region, lat_zone, p["fig_dir"], p["hazard_csv"], p["hazard_png"])

def compute_summary_from_intervals_csv(region: str, lat_zone: str, intervals_csv: str) -> Dict[str, Any]:
    df_wait = pd.read_csv(intervals_csv)
    if df_wait.empty:
        return None
    gaps = df_wait["gap_days"].values
    row = dict(
        region=region, lat_zone=lat_zone,
        n_intervals=int(len(gaps)),
        gap_median=float(np.median(gaps)),
        gap_p25=float(np.percentile(gaps, 25)),
        gap_p75=float(np.percentile(gaps, 75)),
        P_le_7=float((gaps <= 7).mean()),
        P_le_14=float((gaps <= 14).mean()),
        P_le_30=float((gaps <= 30).mean()),
    )
    row.update(compute_binned_hazard(gaps))
    return row

# =========================
# Combined regions (merge child intervals CSVs)
# =========================

def run_combined_regions(summary_rows: List[Dict[str, Any]],
                         hazard_curves_store: Dict[str, pd.DataFrame]):
    if not ENABLE_COMBINED_REGIONS or not COMBINED_REGION_GROUPS:
        return

    print("\n[COMBINED] 开始计算合并区域 hazard…")
    for (combo_region, combo_lat_zone), members in COMBINED_REGION_GROUPS.items():
        combo_tag = tag_of(combo_region, combo_lat_zone)
        p_combo = paths_for(combo_tag)

        print("\n" + "-"*68)
        print(f"[COMBINED] {combo_region} / {combo_lat_zone}")
        print("-"*68)

        # Resume for combined regions too
        if RESUME_MODE == "auto" and has_finished_and_match_signature(combo_tag):
            print("[COMBINED][RESUME] 已有 CSV 且参数签名一致，跳过重算。")
            if os.path.exists(p_combo["hazard_csv"]):
                try:
                    hazard_curves_store[combo_tag] = pd.read_csv(p_combo["hazard_csv"])
                except Exception:
                    hazard_curves_store[combo_tag] = None
            if os.path.exists(p_combo["intervals_csv"]):
                row = compute_summary_from_intervals_csv(combo_region, combo_lat_zone, p_combo["intervals_csv"])
                if row is not None:
                    summary_rows.append(row)
            replot_from_csv_if_needed(combo_region, combo_lat_zone, combo_tag, p_combo)
            continue

        print(f"[COMBINED] 由以下子区域组成:")
        df_list = []
        for (reg0, lat0) in members:
            sub_tag = tag_of(reg0, lat0)
            p_sub = paths_for(sub_tag)
            print(f"  - {reg0} / {lat0}")
            if not os.path.exists(p_sub["intervals_csv"]):
                print(f"    [WARN] 子区域 {reg0}/{lat0} 没有 waiting_intervals CSV，跳过该子区域")
                continue
            try:
                df_sub = pd.read_csv(p_sub["intervals_csv"])
            except Exception as e:
                print(f"    [WARN] 读取 {p_sub['intervals_csv']} 失败：{e}")
                continue
            if df_sub.empty:
                print(f"    [WARN] 子区域 {reg0}/{lat0} intervals 为空，跳过该子区域")
                continue
            df_list.append(df_sub)

        if not df_list:
            print(f"[COMBINED] {combo_region}/{combo_lat_zone} 没有任何可用子区域，跳过。")
            continue

        df_combo = pd.concat(df_list, ignore_index=True)
        df_combo["region"] = combo_region
        df_combo["lat_zone"] = combo_lat_zone

        ensure_dir(p_combo["fig_dir"])
        df_combo.to_csv(p_combo["intervals_csv"], index=False)
        print(f"[COMBINED CSV] waiting_intervals -> {p_combo['intervals_csv']} (rows={len(df_combo)})")

        # Plots + hazard
        plot_waiting_hist(df_combo, combo_region, combo_lat_zone,
                          MAX_WAIT_DAYS_FOR_HAZARD, p_combo["fig_dir"], p_combo["hist_png"])
        plot_waiting_trend(df_combo, combo_region, combo_lat_zone,
                           MIN_INTERVALS_PER_REGION, p_combo["fig_dir"], p_combo["trend_csv"], p_combo["trend_png"])
        df_h_combo = compute_daily_hazard(df_combo, MAX_WAIT_DAYS_FOR_HAZARD)
        plot_hazard_curve(df_h_combo, combo_region, combo_lat_zone,
                          p_combo["fig_dir"], p_combo["hazard_csv"], p_combo["hazard_png"])
        hazard_curves_store[combo_tag] = df_h_combo.copy()

        row = compute_summary_from_intervals_csv(combo_region, combo_lat_zone, p_combo["intervals_csv"])
        if row is not None:
            summary_rows.append(row)

        meta = dict(
            signature=current_signature(),
            rows=len(df_combo),
            intervals=len(df_combo),
            combined_from=[{"region": r, "lat_zone": lz} for (r, lz) in members],
            generated_figs=[
                os.path.basename(p_combo["hist_png"]),
                os.path.basename(p_combo["trend_png"]),
                os.path.basename(p_combo["hazard_png"]),
            ],
        )
        with open(p_combo["meta_json"], "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[COMBINED META] -> {p_combo['meta_json']}")

# =========================
# Main
# =========================

if __name__ == "__main__":
    ensure_dir(OUT_ROOT); ensure_dir(FIG_DIR); ensure_dir(CSV_DIR)

    # 1) Enumerate all region × lat_zone (or top-K for display)
    ALL_REGS, df_stats = list_all_regions(PARQUET_ROOT)
    if RUN_ALL_REGIONS:
        EXP_REGS = ALL_REGS
        print(f"\n[SELECT] 将分析全部 {len(EXP_REGS)} 个 region×lat_zone。")
    else:
        EXP_REGS = ALL_REGS[:TOP_K_FOR_FIGS]
        print(f"\n[SELECT] 将分析前 {len(EXP_REGS)} 个最活跃地区（仅示例）。")

    summary_rows: List[Dict[str, Any]] = []
    hazard_curves_store: Dict[str, pd.DataFrame] = {}

    # 2) Per-region pipeline
    for (region, lat_zone) in EXP_REGS:
        gc.collect()
        tag = tag_of(region, lat_zone)
        p = paths_for(tag)

        print("\n" + "="*68)
        print(f"[REGION] {region} / {lat_zone}")
        print("="*68)

        # Resuming: skip compute if signature matches; still rebuild missing figs and include in global outputs
        if RESUME_MODE == "auto" and has_finished_and_match_signature(tag):
            print("[RESUME] CSV 已存在且参数签名一致，跳过重算。")
            # 1) Fill global overlay & summary from CSVs
            if os.path.exists(p["hazard_csv"]):
                try:
                    hazard_curves_store[tag] = pd.read_csv(p["hazard_csv"])
                except Exception:
                    hazard_curves_store[tag] = None
            if os.path.exists(p["intervals_csv"]):
                row = compute_summary_from_intervals_csv(region, lat_zone, p["intervals_csv"])
                if row is not None:
                    summary_rows.append(row)
            # 2) Rebuild figures if missing
            replot_from_csv_if_needed(region, lat_zone, tag, p)
            continue

        # Normal path: load data, compute intervals, plot, hazard
        t0 = time.time()
        df_reg = load_region_latzone_df(PARQUET_ROOT, region, lat_zone)
        if df_reg.empty:
            continue

        df_wait = compute_waiting_intervals_for_region(df_reg, MIN_COOLDOWN_DAYS)
        if df_wait.empty:
            print("[INFO] 该地区有效间隔为 0，跳过图形和 hazard。")
            continue

        # Save intervals for later reuse / rebuild
        ensure_dir(p["fig_dir"])
        df_wait.to_csv(p["intervals_csv"], index=False)
        print(f"[CSV] waiting_intervals -> {p['intervals_csv']}")

        # Per-region figures
        plot_waiting_hist(df_wait, region, lat_zone, MAX_WAIT_DAYS_FOR_HAZARD, p["fig_dir"], p["hist_png"])
        plot_waiting_trend(df_wait, region, lat_zone, MIN_INTERVALS_PER_REGION, p["fig_dir"], p["trend_csv"], p["trend_png"])

        # Hazard curve + figure
        df_h = compute_daily_hazard(df_wait, MAX_WAIT_DAYS_FOR_HAZARD)
        plot_hazard_curve(df_h, region, lat_zone, p["fig_dir"], p["hazard_csv"], p["hazard_png"])
        hazard_curves_store[tag] = df_h.copy()

        # Summary row
        gaps = df_wait["gap_days"].values
        row = dict(
            region=region, lat_zone=lat_zone,
            n_intervals=int(len(gaps)),
            gap_median=float(np.median(gaps)),
            gap_p25=float(np.percentile(gaps, 25)),
            gap_p75=float(np.percentile(gaps, 75)),
            P_le_7=float((gaps <= 7).mean()),
            P_le_14=float((gaps <= 14).mean()),
            P_le_30=float((gaps <= 30).mean()),
        )
        row.update(compute_binned_hazard(gaps))
        summary_rows.append(row)

        # Write meta (signature) for future resume
        meta = dict(signature=current_signature(),
                    rows=len(df_reg), intervals=len(df_wait),
                    generated_figs=[os.path.basename(p["hist_png"]),
                                    os.path.basename(p["trend_png"]),
                                    os.path.basename(p["hazard_png"])])
        with open(p["meta_json"], "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[META] -> {p['meta_json']}  |  elapsed {time.time()-t0:.1f}s")

    # 2b) Combined regions (from existing intervals CSVs)
    run_combined_regions(summary_rows, hazard_curves_store)

    # 3) Global summary + plots
    if len(summary_rows) > 0:
        df_summary = pd.DataFrame(summary_rows)
        summary_path = os.path.join(CSV_DIR, "hazard_summary_GLOBAL.csv")
        df_summary.to_csv(summary_path, index=False)
        print(f"\n[GLOBAL] hazard_summary_GLOBAL -> {summary_path}")

        plot_global_median_bar(df_summary, FIG_DIR)

        # Overlay: show up to TOP_K_FOR_FIGS curves for readability (store keeps as many as available)
        ordered_tags = list(hazard_curves_store.keys())
        sub = {k: hazard_curves_store[k] for k in ordered_tags[:TOP_K_FOR_FIGS]}
        plot_global_hazard_overlay(sub, FIG_DIR, max_regions=TOP_K_FOR_FIGS)
    else:
        print("[GLOBAL] 没有任何 region 产生有效的 waiting intervals。")

    print("\n[DONE] Hazard analysis finished.")
    print(f"       Figures in: {FIG_DIR}")
    print(f"       CSVs in   : {CSV_DIR}")
