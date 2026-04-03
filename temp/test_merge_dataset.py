#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd


def load_jsonl(path):
    df = pd.read_json(path, lines=True)
    df["timestamp"] = pd.to_numeric(df["timestamp"])
    return df.sort_values("timestamp")


def align_timestamp_domain(reference_df, candidate_df):
    if reference_df.empty or candidate_df.empty:
        return candidate_df

    ref_median = float(reference_df["timestamp"].median())
    cand_median = float(candidate_df["timestamp"].median())

    if not np.isfinite(ref_median) or not np.isfinite(cand_median) or cand_median == 0:
        return candidate_df

    scale_ratio = abs(ref_median) / max(abs(cand_median), 1.0)
    if scale_ratio > 1_000 or scale_ratio < 0.001:
        offset = int(ref_median - cand_median)
        adjusted = candidate_df.copy()
        adjusted["timestamp"] = adjusted["timestamp"] + offset
        return adjusted

    return candidate_df


def preprocess_and_merge(state_path, pmc_path, rapl_path, output_file="merged_dataset.csv"):
    print(f"[*] Loading State: {state_path}")
    state = load_jsonl(state_path)

    print(f"[*] Loading PMC: {pmc_path}")
    pmc = load_jsonl(pmc_path)

    print(f"[*] Loading RAPL: {rapl_path}")
    rapl = load_jsonl(rapl_path)

    pmc = align_timestamp_domain(state, pmc)
    rapl = align_timestamp_domain(state, rapl)

    rapl["dt_sec"] = rapl["timestamp"].diff() / 1e9
    rapl["dt_sec"] = rapl["dt_sec"].replace(0, np.nan)
    rapl["power_watts"] = (rapl["delta_uj"] / 1e6) / rapl["dt_sec"]
    rapl["power_watts"] = rapl["power_watts"].replace([np.inf, -np.inf], np.nan).fillna(0)

    if "ipc" not in pmc.columns:
        pmc["ipc"] = 0.0

    if "cache_misses" not in pmc.columns:
        pmc["cache_misses"] = 0

    print("[*] Merging State with PMCs (src_cpu -> cpu)...")
    merged = pd.merge_asof(
        state,
        pmc,
        on="timestamp",
        left_by="src_cpu",
        right_by="cpu",
        direction="backward",
    )

    print("[*] Merging with RAPL energy data...")
    merged = pd.merge_asof(
        merged,
        rapl,
        on="timestamp",
        direction="nearest",
    )

    merged["power_watts"] = pd.to_numeric(merged.get("power_watts", 0.0), errors="coerce")
    merged["power_watts"] = merged["power_watts"].replace([np.inf, -np.inf], np.nan).fillna(0)
    power_cap = merged["power_watts"].quantile(0.995)
    if np.isfinite(power_cap) and power_cap > 0:
        merged["power_watts"] = merged["power_watts"].clip(upper=power_cap)

    merged["ipc"] = pd.to_numeric(merged.get("ipc", 0.0), errors="coerce")
    merged["ipc"] = merged["ipc"].replace([np.inf, -np.inf], np.nan).fillna(0)

    merged["cache_misses"] = pd.to_numeric(merged.get("cache_misses", 0), errors="coerce")
    merged["cache_misses"] = merged["cache_misses"].replace([np.inf, -np.inf], np.nan).fillna(0)
    merged["cache_misses"] = merged["cache_misses"].clip(lower=0)

    merged.to_csv(output_file, index=False)
    print(f"[✓] Success! Processed {len(merged)} samples into {output_file}")
    return merged


def validate_output(df):
    required_columns = ["decision", "power_watts", "ipc", "cache_misses"]
    missing_columns = [c for c in required_columns if c not in df.columns]
    if missing_columns:
        raise AssertionError(f"Missing required columns: {missing_columns}")

    if not np.isfinite(df["power_watts"]).all():
        raise AssertionError("power_watts contains non-finite values")

    if not np.isfinite(df["ipc"]).all():
        raise AssertionError("ipc contains non-finite values")

    if not np.isfinite(df["cache_misses"]).all():
        raise AssertionError("cache_misses contains non-finite values")

    if (df["power_watts"] < 0).any():
        raise AssertionError("power_watts contains negative values")

    if (df["cache_misses"] < 0).any():
        raise AssertionError("cache_misses contains negative values")

    print("[✓] Validation passed")
    print(f"[*] power_watts range: {df['power_watts'].min():.3f} -> {df['power_watts'].max():.3f}")
    print(f"[*] ipc range: {df['ipc'].min():.3f} -> {df['ipc'].max():.3f}")
    print(f"[*] cache_misses range: {df['cache_misses'].min():.3f} -> {df['cache_misses'].max():.3f}")


def main():
    parser = argparse.ArgumentParser(description="Merge regression check with IPC validations")
    parser.add_argument("--state", required=True)
    parser.add_argument("--pmc", required=True)
    parser.add_argument("--rapl", required=True)
    parser.add_argument("--output", default="temp/merged_dataset_debug.csv")
    args = parser.parse_args()

    merged = preprocess_and_merge(args.state, args.pmc, args.rapl, args.output)
    validate_output(merged)


if __name__ == "__main__":
    main()