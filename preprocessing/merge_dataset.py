#!/usr/bin/env python3

import pandas as pd
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

    # 3. FEATURE ENGINEERING: Calculate Power (Watts)
    # Energy_uj is cumulative; Watts = (delta_joules) / (delta_time)
    print("[*] Calculating Power consumption...")
    # Convert ns timestamp difference to seconds
    merged['dt_sec'] = merged.groupby('src_cpu')['timestamp'].diff() / 1e9
    # watts = (microjoules / 1,000,000) / seconds
    merged['power_watts'] = (merged['delta_uj'] / 1e6) / merged['dt_sec']
    
    # Handle the first row of each CPU group (NaN) and potential divide by zero
    merged['power_watts'] = merged['power_watts'].replace([float('inf'), -float('inf')], 0).fillna(0)

    # 4. CLEANUP: Remove "Leakage" and non-numeric columns
    # These columns either crash the NN or provide 'cheating' info the kernel won't have
    cols_to_drop = [
        'timestamp', 'pid_x', 'comm_x', 'pid_y', 'comm_y', 'cpu', 
        'package', 'energy_uj', 'total_uj', 'delta_uj', 'dt_sec'
    ]
    
    # Only drop if they exist (prevents errors on re-runs)
    existing_drops = [c for c in cols_to_drop if c in merged.columns]
    final_df = merged.drop(columns=existing_drops)

    # 5. Final Filtering
    # Remove any rows where decision might be null or metadata failed
    final_df = final_df.dropna()

    print(f"[*] Final feature set: {list(final_df.columns)}")
    final_df.to_csv(args.output, index=False)
    print(f"[✓] Training-ready dataset saved to {args.output}")

if __name__ == "__main__":
    main()