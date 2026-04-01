import pandas as pd
import os

def preprocess_and_merge(state_path, pmc_path, rapl_path, output_file="merged_dataset.csv"):
    print(f"[*] Loading State: {state_path}")
    df_state = pd.read_json(state_path, lines=True)
    
    print(f"[*] Loading PMC: {pmc_path}")
    df_pmc = pd.read_json(pmc_path, lines=True)
    
    print(f"[*] Loading RAPL: {rapl_path}")
    df_rapl = pd.read_json(rapl_path, lines=True)

    # Convert timestamps to numeric and sort (Essential for merge_asof)
    for df in [df_state, df_pmc, df_rapl]:
        df['timestamp'] = pd.to_numeric(df['timestamp'])
        df.sort_values('timestamp', inplace=True)

    # 1. Merge State with PMC on Timestamp AND CPU
    print("[*] Merging State with PMCs (by CPU)...")
    merged = pd.merge_asof(
        df_state, 
        df_pmc, 
        on='timestamp', 
        by='cpu', 
        direction='backward'
    )

    # 2. Merge with RAPL
    # RAPL is usually package-wide, so we don't 'by' on CPU
    print("[*] Merging with RAPL energy data...")
    final_df = pd.merge_asof(
        merged, 
        df_rapl, 
        on='timestamp', 
        direction='nearest'
    )

    # 3. Handle specific PMC/RAPL metrics
    # If energy data has delta_uj, we can calculate power
    if 'delta_uj' in final_df.columns:
        # Time delta in seconds (assuming timestamp is ns)
        final_df['dt'] = final_df.groupby('cpu')['timestamp'].diff() / 1e9
        final_df['power_watts'] = (final_df['delta_uj'] / 1e6) / final_df.dt.replace(0, 1)

    final_df.to_csv(output_file, index=False)
    print(f"[✓] Success! Processed {len(final_df)} samples into {output_file}")

if __name__ == "__main__":
    # Update these paths to match your actual files exactly
    preprocess_and_merge(
        'data/raw/state_20260220_010514.jsonl',
        'data/raw/pmc_20260220_010514.jsonl', 
        'data/raw/rapl_20260220_010514.jsonl'
    )