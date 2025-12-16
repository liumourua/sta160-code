# -*- coding: utf-8 -*-
"""
Convert *_WITH_OOF_LITE.csv to a Parquet dataset partitioned by [region, lat_zone].
- Stream CSV in chunks (handles very large files).
- Downcast numerics (float32 / int16) to reduce size.
- Optionally keep only key columns (KEEP_COLS).
- Fill missing region/lat_zone with 'Unknown' to avoid partition errors.
"""

import os, sys
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds

# ============= Config =============
# Input CSV (the *_WITH_OOF_LITE.csv you exported earlier)
IN_CSV = "/Users/liuliuyiyu/Documents/senior/first quarter/sta160/Wildfire_CLEANED_WITH_OOF_LITE.csv"

# Output Parquet dataset root directory
OUT_DIR = os.path.splitext(IN_CSV)[0] + "_parquet_region_latzone"

# Rows per chunk (tune for your memory)
CHUNK_SIZE = 500_000

# Keep only key columns (True reduces size; False keeps all)
USE_KEEP_COLS = True

# Key columns (adjust as needed)
KEEP_COLS = [
    # Keys / metadata
    "latitude","longitude","datetime","year","month","lat_bin","lon_bin",
    "region","lat_zone","season","Wildfire",
    # CatBoost OOF & selected derived (commonly used by RNN)
    "p_cb_oof","logit_cb","rank_cb_day",
    "p_cb_lag1","p_cb_lag2","p_cb_lag3","p_cb_roll7_mean","p_cb_roll7_max",
    # Core met/fuel (today + lag1)
    "pr","vpd","tmmx_c","tmmn_c","rmax","rmin","sph","srad","vs","etr","pet","bi","fm100","fm1000","erc",
    "pr_lag1","vpd_lag1","tmmx_c_lag1","tmmn_c_lag1","rmax_lag1","rmin_lag1",
    "sph_lag1","srad_lag1","vs_lag1","etr_lag1","pet_lag1","bi_lag1","fm100_lag1","fm1000_lag1","erc_lag1",
    # Minimal rolling/agg/extremes
    "vpd_mean_7","vpd_mean_14","vpd_max_7",
    "tmmx_c_mean_7","tmmx_c_mean_14","tmmx_slope_14",
    "tmmn_c_mean_7","tmmn_c_mean_14",
    "pr_sum_7","pr_sum_14","dry_30",
    # Training weight (used at fit-time, not as model feature)
    "rnn_weight",
    # Optional: add one-hot columns (e.g., cb_TP/FN) if present
]

# ============= Utilities =============
def downcast_df(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns (float32/int16); ensure datetime and categoricals are sane."""
    # Parse datetime if needed
    if "datetime" in df.columns and not np.issubdtype(df["datetime"].dtype, np.datetime64):
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    # region/lat_zone/season → string with 'Unknown'
    for c in ["region","lat_zone","season"]:
        if c in df.columns:
            df[c] = df[c].astype(str).replace({"nan": "Unknown"}).fillna("Unknown")

    # Derive year/month if missing
    if "year" not in df.columns and "datetime" in df.columns:
        df["year"] = df["datetime"].dt.year
    if "month" not in df.columns and "datetime" in df.columns:
        df["month"] = df["datetime"].dt.month

    # Downcast numerics
    for c in df.select_dtypes(include=["float64","float32","float"]).columns:
        df[c] = df[c].astype("float32")
    for c in df.select_dtypes(include=["int64","int32","int"]).columns:
        if df[c].min() >= -32768 and df[c].max() <= 32767:
            df[c] = df[c].astype("int16")

    return df


def write_chunk_to_dataset(pdf: pd.DataFrame, root_path: str, schema):
    """Write one pandas chunk to a Parquet dataset partitioned by [region, lat_zone]."""
    # Optional column filtering
    if USE_KEEP_COLS:
        keep = [c for c in KEEP_COLS if c in pdf.columns]
        pdf = pdf[keep]

    # Downcast types
    pdf = downcast_df(pdf)

    # Ensure partition keys exist and are strings
    for c in ["region","lat_zone"]:
        if c not in pdf.columns:
            pdf[c] = "Unknown"
        pdf[c] = pdf[c].fillna("Unknown").astype(str)

    # To Arrow Table
    table = pa.Table.from_pandas(pdf, preserve_index=False, schema=schema, safe=False)

    # Append to partitioned dataset (may create multiple files per partition)
    pq.write_to_dataset(
        table,
        root_path=root_path,
        partition_cols=["region","lat_zone"],
        existing_data_behavior="overwrite_or_ignore",
        compression="zstd"
    )

# ============= Main =============
os.makedirs(OUT_DIR, exist_ok=True)

# Stream CSV by chunks
reader = pd.read_csv(IN_CSV, chunksize=CHUNK_SIZE)

# First chunk to build schema
first_chunk = next(reader)
if USE_KEEP_COLS:
    first_chunk = first_chunk[[c for c in KEEP_COLS if c in first_chunk.columns]]
first_chunk = downcast_df(first_chunk)
for c in ["region","lat_zone"]:
    if c not in first_chunk.columns:
        first_chunk[c] = "Unknown"
    first_chunk[c] = first_chunk[c].fillna("Unknown").astype(str)

schema = pa.Table.from_pandas(first_chunk, preserve_index=False).schema

# Write first chunk
write_chunk_to_dataset(first_chunk, OUT_DIR, schema)
print(f"[Write] first chunk -> {OUT_DIR}")

# Append remaining chunks
for i, chunk in enumerate(reader, start=1):
    write_chunk_to_dataset(chunk, OUT_DIR, schema)
    if i % 5 == 0:
        print(f"[Write] {i} chunks appended...")

print(f"[Done] Parquet dataset at: {OUT_DIR}")

# ============= Quick check =============
dataset = ds.dataset(OUT_DIR, format="parquet", partitioning="hive")

# Row counts per [region, lat_zone] (top 10)
by_part = (
    dataset.to_table(columns=["region","lat_zone"])
           .to_pandas()
           .value_counts(["region","lat_zone"])
           .reset_index(name="rows")
           .sort_values("rows", ascending=False)
)

print("\n[Sample partitions head]")
print(by_part.head(10))

# Example: read a single partition (East × South)
# import pyarrow.dataset as ds
# filt = (ds.field("region") == "East") & (ds.field("lat_zone") == "South")
# tbl = dataset.to_table(filter=filt)
# df_east_south = tbl.to_pandas()
# print(df_east_south.shape)
