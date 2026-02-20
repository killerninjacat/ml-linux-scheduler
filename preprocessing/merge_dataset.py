#!/usr/bin/env python3

import pandas as pd
import argparse
import os
import sys


def check_file(path):
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)


def load_jsonl(path):
    df = pd.read_json(path, lines=True)

    # Force timestamp to int64 (nanoseconds)
    df["timestamp"] = df["timestamp"].astype("int64")

    return df


def main():
    parser = argparse.ArgumentParser(description="Merge state, PMC and RAPL datasets")
    parser.add_argument("--state", required=True)
    parser.add_argument("--pmc", required=True)
    parser.add_argument("--rapl", required=True)
    parser.add_argument("--output", default="merged_dataset.csv")
    args = parser.parse_args()

    check_file(args.state)
    check_file(args.pmc)
    check_file(args.rapl)

    print("[*] Loading datasets...")

    state = load_jsonl(args.state)
    pmc = load_jsonl(args.pmc)
    rapl = load_jsonl(args.rapl)

    # Sort (required for merge_asof)
    state = state.sort_values("timestamp")
    pmc = pmc.sort_values("timestamp")
    rapl = rapl.sort_values("timestamp")

    print("[*] Merging state + PMC...")
    merged = pd.merge_asof(
        state,
        pmc,
        on="timestamp",
        direction="nearest"
    )

    print("[*] Merging with RAPL...")
    merged = pd.merge_asof(
        merged,
        rapl,
        on="timestamp",
        direction="nearest"
    )

    print(f"[*] Saving merged dataset to {args.output}")
    merged.to_csv(args.output, index=False)

    print("[✓] Merge complete")
    print(f"[*] Final dataset shape: {merged.shape}")


if __name__ == "__main__":
    main()
