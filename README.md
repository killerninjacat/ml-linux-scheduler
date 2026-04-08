## Environment

```bash
source venv/bin/activate
```

## Collect raw traces

```bash
sudo python3 collect_training_data.py --duration 60
```

Outputs are written to data/raw with session-specific names:
- state_<session>.jsonl
- pmc_<session>.jsonl
- rapl_<session>.jsonl
- scheduler_<session>.jsonl

## Merge and build training dataset

```bash
python3 preprocessing/merge_dataset.py \
  --state data/raw/state_YYYYMMDD_HHMMSS.jsonl \
  --pmc data/raw/pmc_YYYYMMDD_HHMMSS.jsonl \
  --rapl data/raw/rapl_YYYYMMDD_HHMMSS.jsonl \
  --pmc-tolerance-ms 50 \
  --rapl-tolerance-ms 100 \
  --output data/processed/final_dataset.csv
```

## Train model (single run)

Energy-aware loss is default in single run mode.

```bash
python3 train/train_and_validate_model.py \
  --data-path data/processed/final_dataset.csv \
  --model-dir models/final \
  --run-mode single \
  --loss-mode energy \
  --batch-size 256 \
  --patience 15 \
  --epochs 100
```

By default the trainer now performs validation-threshold optimization for better F1/accuracy trade-off. Use --no-optimize-threshold to force a fixed 0.5 decision cutoff.

### Baseline BCE-only run

```bash
python3 train/train_and_validate_model.py \
  --data-path data/processed/final_dataset.csv \
  --model-dir models/final_bce \
  --run-mode single \
  --loss-mode bce
```

## Run ablation suite

Runs three experiments on the same split/seed:
1. baseline_bce
2. bce_plus_power
3. bce_plus_power_ipc

```bash
python3 train/train_and_validate_model.py \
  --data-path data/processed/final_dataset.csv \
  --model-dir models/ablation \
  --run-mode ablation
```

## IPC and cache-miss collection requirements

- Root privileges (collectors are run with sudo)
- CPU and kernel support for hardware perf counters
- perf_event permissions must allow counter access (commonly perf_event_paranoid <= 1)

If hardware counters are unavailable, the PMC collector falls back to:
- ipc=0 with ipc_available=0
- cache_misses=0 with cache_miss_available=0

The merge script keeps the pipeline runnable with this fallback, but efficiency/cache-aware behavior will be limited.

## Recommended energy-loss tuning

Start conservatively to avoid always-migrate behavior:
- energy_alpha = 0.01
- energy_beta = 0.01

Then increase gradually while watching energy-case diagnostics from training output.

## Single-script execution

Use the full automation script:

```bash
./complete_script.sh [duration] [epochs] [run_graphs] [energy_alpha] [energy_beta] [power_threshold] [ipc_threshold] [run_sweep] [alpha_grid] [beta_grid] [pmc_tolerance_ms] [rapl_tolerance_ms] [batch_size] [patience] [hidden_dim] [dropout] [weight_decay] [no_optimize_threshold] [decision_threshold]
```

Example:

```bash
./complete_script.sh 60 100 0 0.01 0.01 35.0 0.9
```

Accuracy-focused example (tolerances + stronger trainer config + auto threshold):

```bash
./complete_script.sh 60 120 0 0.01 0.01 35.0 0.9 0 "0.005,0.01,0.02" "0.005,0.01,0.02" 50 100 256 15 32 0.1 1e-5 0
```

Sweep example (runs grid search and saves a ranked comparison table):

```bash
./complete_script.sh 60 100 0 0.01 0.01 35.0 0.9 1 "0.005,0.01,0.02" "0.005,0.01,0.02" 50 100 256 15 32 0.1 1e-5 0
```

Sweep outputs:
- results/alpha_beta_sweep.csv
- results/alpha_beta_sweep_ranked.csv
- results/sweeps/
- models/sweeps/

## Optional graph generation

```bash
sudo python3 graphs/review_1.py
```