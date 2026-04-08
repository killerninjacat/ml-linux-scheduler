#!/usr/bin/env python3

import pandas as pd
import numpy as np
import argparse
import os
import sys

def load_jsonl(path):
    df = pd.read_json(path, lines=True)
    df["timestamp"] = pd.to_numeric(df["timestamp"])
    return df


def align_timestamp_domain(reference_df, candidate_df, name):
    """
    Align candidate timestamps to reference domain when scales differ (e.g.,
    monotonic ns vs epoch ns in legacy files).
    """
    if reference_df.empty or candidate_df.empty:
        return candidate_df

    ref_median = float(reference_df["timestamp"].median())
    cand_median = float(candidate_df["timestamp"].median())

    if not np.isfinite(ref_median) or not np.isfinite(cand_median) or cand_median == 0:
        return candidate_df

    scale_ratio = abs(ref_median) / max(abs(cand_median), 1.0)
    if scale_ratio > 1_000 or scale_ratio < 0.001:
        offset = int(ref_median - cand_median)
        print(f"[!] {name} timestamps appear in a different domain; applying offset {offset} ns")
        adjusted = candidate_df.copy()
        adjusted["timestamp"] = adjusted["timestamp"] + offset
        return adjusted

    return candidate_df

def main():
    parser = argparse.ArgumentParser(description="Clean and Prepare Scheduler Dataset")
    parser.add_argument("--state", required=True)
    parser.add_argument("--pmc", required=True)
    parser.add_argument("--rapl", required=True)
    parser.add_argument("--output", default="training_ready_dataset.csv")
    parser.add_argument(
        "--pmc-tolerance-ms",
        type=float,
        default=50.0,
        help="Maximum age gap (ms) when matching state rows to PMC rows",
    )
    parser.add_argument(
        "--rapl-tolerance-ms",
        type=float,
        default=100.0,
        help="Maximum absolute gap (ms) when matching merged rows to RAPL rows",
    )
    args = parser.parse_args()

    print("[*] Loading datasets...")
    state = load_jsonl(args.state).sort_values("timestamp")
    pmc = load_jsonl(args.pmc).sort_values("timestamp")
    rapl = load_jsonl(args.rapl).sort_values("timestamp")

    pmc = align_timestamp_domain(state, pmc, "PMC")
    rapl = align_timestamp_domain(state, rapl, "RAPL")

    # Compute power from RAPL-native cadence before timestamp alignment.
    rapl['dt_sec'] = rapl['timestamp'].diff() / 1e9
    rapl['dt_sec'] = rapl['dt_sec'].replace(0, np.nan)
    rapl['power_watts'] = (rapl['delta_uj'] / 1e6) / rapl['dt_sec']
    rapl['power_watts'] = rapl['power_watts'].replace([float('inf'), -float('inf')], np.nan)
    rapl['power_watts'] = rapl['power_watts'].clip(lower=0).fillna(0)

    if 'ipc' not in pmc.columns:
        print("[!] IPC field missing in PMC input. Creating fallback ipc=0.0")
        pmc['ipc'] = 0.0

    if 'instructions_retired' not in pmc.columns:
        pmc['instructions_retired'] = 0

    if 'cycles' not in pmc.columns:
        pmc['cycles'] = 0

    if 'cache_misses' not in pmc.columns:
        pmc['cache_misses'] = 0

    if 'cache_miss_available' not in pmc.columns:
        pmc['cache_miss_available'] = 0

    pmc_tolerance_ns = int(args.pmc_tolerance_ms * 1_000_000)
    rapl_tolerance_ns = int(args.rapl_tolerance_ms * 1_000_000)
    print(f"[*] PMC merge tolerance: {args.pmc_tolerance_ms:.1f} ms")
    print(f"[*] RAPL merge tolerance: {args.rapl_tolerance_ms:.1f} ms")

    # 1. Align PMCs to the Source CPU
    # We map the PMC activity of the 'src_cpu' to the decision being made
    print("[*] Merging State with PMCs (Mapping to src_cpu)...")
    merged = pd.merge_asof(
        state,
        pmc,
        on="timestamp",
        left_by="src_cpu",
        right_by="cpu",
        direction="backward",
        tolerance=pmc_tolerance_ns,
    )

    pmc_unmatched_ratio = merged['runtime_ns'].isna().mean() if 'runtime_ns' in merged.columns else 1.0
    print(f"[*] PMC unmatched rows after tolerance: {pmc_unmatched_ratio * 100:.2f}%")

    # 2. Merge with RAPL Energy
    print("[*] Merging with RAPL energy data...")
    merged = pd.merge_asof(
        merged,
        rapl,
        on="timestamp",
        direction="nearest",
        tolerance=rapl_tolerance_ns,
    )

    rapl_unmatched_ratio = merged['power_watts'].isna().mean() if 'power_watts' in merged.columns else 1.0
    print(f"[*] RAPL unmatched rows after tolerance: {rapl_unmatched_ratio * 100:.2f}%")

    # 3. FEATURE ENGINEERING: Power was computed in RAPL table prior to merge.
    print("[*] Using precomputed RAPL power values...")

    # Limit extreme outliers from timestamp alignment jitter
    power_cap = merged['power_watts'].quantile(0.995)
    if np.isfinite(power_cap) and power_cap > 0:
        merged['power_watts'] = merged['power_watts'].clip(upper=power_cap)
        print(f"[*] Applied power outlier cap at {power_cap:.3f} W (99.5th percentile)")

    print("[*] Cleaning IPC and throughput features...")
    merged['ipc'] = pd.to_numeric(merged.get('ipc', 0.0), errors='coerce')
    merged['ipc'] = merged['ipc'].replace([float('inf'), -float('inf')], np.nan)
    merged['ipc'] = merged['ipc'].clip(lower=0, upper=10).fillna(0)

    merged['cache_misses'] = pd.to_numeric(merged.get('cache_misses', 0), errors='coerce')
    merged['cache_misses'] = merged['cache_misses'].replace([float('inf'), -float('inf')], np.nan)
    merged['cache_misses'] = merged['cache_misses'].clip(lower=0).fillna(0)

    runtime_ms = pd.to_numeric(merged.get('runtime_ms', 0.0), errors='coerce').replace(0, np.nan)
    merged['instructions_per_ms'] = pd.to_numeric(merged.get('instructions_retired', 0), errors='coerce') / runtime_ms
    merged['cycles_per_ms'] = pd.to_numeric(merged.get('cycles', 0), errors='coerce') / runtime_ms
    merged['cache_misses_per_ms'] = merged['cache_misses'] / runtime_ms

    instructions = pd.to_numeric(merged.get('instructions_retired', 0), errors='coerce').replace(0, np.nan)
    merged['cache_mpki'] = (merged['cache_misses'] * 1000.0) / instructions

    merged['instructions_per_ms'] = merged['instructions_per_ms'].replace([float('inf'), -float('inf')], np.nan).fillna(0)
    merged['cycles_per_ms'] = merged['cycles_per_ms'].replace([float('inf'), -float('inf')], np.nan).fillna(0)
    merged['cache_misses_per_ms'] = merged['cache_misses_per_ms'].replace([float('inf'), -float('inf')], np.nan).fillna(0)
    merged['cache_mpki'] = merged['cache_mpki'].replace([float('inf'), -float('inf')], np.nan).fillna(0)

    # 4. CLEANUP: Remove "Leakage" and non-numeric columns
    # These columns either crash the NN or provide 'cheating' info the kernel won't have
    cols_to_drop = [
        'timestamp', 'pid_x', 'comm_x', 'pid_y', 'comm_y', 'cpu', 
        'package', 'energy_uj', 'total_uj', 'delta_uj', 'dt_sec',
        'instructions_retired', 'cycles', 'ipc_available', 'cache_miss_available'
    ]
    
    # Only drop if they exist (prevents errors on re-runs)
    existing_drops = [c for c in cols_to_drop if c in merged.columns]
    final_df = merged.drop(columns=existing_drops)

    # 5. Final Filtering
    # Remove any rows where decision might be null or metadata failed
    final_df = final_df.dropna()

    print(f"[*] Final rows: {len(final_df)}")
    print(f"[*] Power stats (W): min={final_df['power_watts'].min():.3f}, max={final_df['power_watts'].max():.3f}")
    if 'ipc' in final_df.columns:
        print(f"[*] IPC stats: min={final_df['ipc'].min():.3f}, max={final_df['ipc'].max():.3f}")
    if 'cache_misses' in final_df.columns:
        print(f"[*] Cache-miss stats: min={final_df['cache_misses'].min():.3f}, max={final_df['cache_misses'].max():.3f}")

    print(f"[*] Final feature set: {list(final_df.columns)}")
    final_df.to_csv(args.output, index=False)
    print(f"[✓] Training-ready dataset saved to {args.output}")

if __name__ == "__main__":
    main()