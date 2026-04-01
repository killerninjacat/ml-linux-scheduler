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

def main():
    parser = argparse.ArgumentParser(description="Clean and Prepare Scheduler Dataset")
    parser.add_argument("--state", required=True)
    parser.add_argument("--pmc", required=True)
    parser.add_argument("--rapl", required=True)
    parser.add_argument("--output", default="training_ready_dataset.csv")
    args = parser.parse_args()

    print("[*] Loading datasets...")
    state = load_jsonl(args.state).sort_values("timestamp")
    pmc = load_jsonl(args.pmc).sort_values("timestamp")
    rapl = load_jsonl(args.rapl).sort_values("timestamp")

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

    # 1. Align PMCs to the Source CPU
    # We map the PMC activity of the 'src_cpu' to the decision being made
    print("[*] Merging State with PMCs (Mapping to src_cpu)...")
    merged = pd.merge_asof(
        state,
        pmc,
        on="timestamp",
        left_by="src_cpu",
        right_by="cpu",
        direction="backward"
    )

    # 2. Merge with RAPL Energy
    print("[*] Merging with RAPL energy data...")
    merged = pd.merge_asof(
        merged,
        rapl,
        on="timestamp",
        direction="nearest"
    )

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

    runtime_ms = pd.to_numeric(merged.get('runtime_ms', 0.0), errors='coerce').replace(0, np.nan)
    merged['instructions_per_ms'] = pd.to_numeric(merged.get('instructions_retired', 0), errors='coerce') / runtime_ms
    merged['cycles_per_ms'] = pd.to_numeric(merged.get('cycles', 0), errors='coerce') / runtime_ms
    merged['instructions_per_ms'] = merged['instructions_per_ms'].replace([float('inf'), -float('inf')], np.nan).fillna(0)
    merged['cycles_per_ms'] = merged['cycles_per_ms'].replace([float('inf'), -float('inf')], np.nan).fillna(0)

    # 4. CLEANUP: Remove "Leakage" and non-numeric columns
    # These columns either crash the NN or provide 'cheating' info the kernel won't have
    cols_to_drop = [
        'timestamp', 'pid_x', 'comm_x', 'pid_y', 'comm_y', 'cpu', 
        'package', 'energy_uj', 'total_uj', 'delta_uj', 'dt_sec',
        'instructions_retired', 'cycles', 'ipc_available'
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

    print(f"[*] Final feature set: {list(final_df.columns)}")
    final_df.to_csv(args.output, index=False)
    print(f"[✓] Training-ready dataset saved to {args.output}")

if __name__ == "__main__":
    main()