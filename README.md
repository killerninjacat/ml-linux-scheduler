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
  --epochs 100
```

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

## IPC collection requirements

- Root privileges (collectors are run with sudo)
- CPU and kernel support for hardware perf counters
- perf_event permissions must allow counter access (commonly perf_event_paranoid <= 1)

If hardware counters are unavailable, the PMC collector falls back to ipc=0 and sets ipc_available=0. The merge script keeps the pipeline runnable with this fallback, but energy-aware IPC behavior will be limited.

## Optional graph generation

```bash
sudo python3 graphs/review_1.py
```