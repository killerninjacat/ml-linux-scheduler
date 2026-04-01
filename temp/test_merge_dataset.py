#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd


def load_jsonl(path):
    df = pd.read_json(path, lines=True)
    df["timestamp"] = pd.to_numeric(df["timestamp"])
    return df.sort_values("timestamp")


def preprocess_and_merge(state_path, pmc_path, rapl_path, output_file="merged_dataset.csv"):
    print(f"[*] Loading State: {state_path}")
    state = load_jsonl(state_path)

    print(f"[*] Loading PMC: {pmc_path}")
    pmc = load_jsonl(pmc_path)

    print(f"[*] Loading RAPL: {rapl_path}")
    rapl = load_jsonl(rapl_path)

    rapl["dt_sec"] = rapl["timestamp"].diff() / 1e9
    rapl["dt_sec"] = rapl["dt_sec"].replace(0, np.nan)
    rapl["power_watts"] = (rapl["delta_uj"] / 1e6) / rapl["dt_sec"]
    rapl["power_watts"] = rapl["power_watts"].replace([np.inf, -np.inf], np.nan).fillna(0)

    if "ipc" not in pmc.columns:
        pmc["ipc"] = 0.0

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

    merged.to_csv(output_file, index=False)
    print(f"[✓] Success! Processed {len(merged)} samples into {output_file}")
    return merged


def validate_output(df):
    required_columns = ["decision", "power_watts", "ipc"]
    missing_columns = [c for c in required_columns if c not in df.columns]
    if missing_columns:
        raise AssertionError(f"Missing required columns: {missing_columns}")

    if not np.isfinite(df["power_watts"]).all():
        raise AssertionError("power_watts contains non-finite values")

    if not np.isfinite(df["ipc"]).all():
        raise AssertionError("ipc contains non-finite values")

    if (df["power_watts"] < 0).any():
        raise AssertionError("power_watts contains negative values")

    print("[✓] Validation passed")
    print(f"[*] power_watts range: {df['power_watts'].min():.3f} -> {df['power_watts'].max():.3f}")
    print(f"[*] ipc range: {df['ipc'].min():.3f} -> {df['ipc'].max():.3f}")


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