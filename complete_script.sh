#!/usr/bin/env bash
set -Eeuo pipefail

DURATION="${1:-60}"
EPOCHS="${2:-100}"
RUN_GRAPHS="${3:-0}"

log() { echo "[INFO] $*"; }
warn() { echo "[WARN] $*" >&2; }
fail() { echo "[ERROR] $*" >&2; exit 1; }

trap 'fail "Failed at line $LINENO"' ERR

REPO_DIR="/home/nithin/fyp-scheduler"
cd "$REPO_DIR"

command -v python3 >/dev/null 2>&1 || fail "python3 not found"
command -v sudo >/dev/null 2>&1 || fail "sudo not found"
command -v stress-ng >/dev/null 2>&1 || fail "stress-ng not found"

[[ -f "venv/bin/activate" ]] || fail "venv/bin/activate not found"
source venv/bin/activate

log "Checking Python dependencies in venv..."
python -c "import pandas, torch, sklearn, joblib, numpy; print('Python dependencies OK')"

log "Refreshing sudo credentials..."
sudo -v

if [[ -r /proc/sys/kernel/perf_event_paranoid ]]; then
  perf_paranoid="$(cat /proc/sys/kernel/perf_event_paranoid || true)"
  if [[ -n "${perf_paranoid}" ]] && (( perf_paranoid > 1 )); then
    warn "perf_event_paranoid is ${perf_paranoid}; trying to set to 1 for IPC counters"
    sudo sysctl -w kernel.perf_event_paranoid=1 >/dev/null || warn "Could not change perf_event_paranoid"
  fi
fi

log "Starting raw data collection (duration=${DURATION}s per workload)..."
sudo python3 collect_training_data.py --duration "$DURATION"

latest_or_fail() {
  local pattern="$1"
  local latest
  latest="$(ls -1t $pattern 2>/dev/null | head -n 1 || true)"
  [[ -n "$latest" ]] || fail "No files found for pattern: $pattern"
  echo "$latest"
}

STATE="$(latest_or_fail "data/raw/state_*.jsonl")"
PMC="$(latest_or_fail "data/raw/pmc_*.jsonl")"
RAPL="$(latest_or_fail "data/raw/rapl_*.jsonl")"

log "Using files:"
echo "  STATE=$STATE"
echo "  PMC=$PMC"
echo "  RAPL=$RAPL"

log "Quick IPC sanity check (first 2000 PMC rows)..."
python -c '
import json, sys
pmc = sys.argv[1]
total = avail = nonzero = 0
with open(pmc, "r") as f:
    for i, line in enumerate(f):
        if i >= 2000:
            break
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        total += 1
        avail += int(r.get("ipc_available", 0))
        nonzero += 1 if float(r.get("ipc", 0.0)) > 0 else 0
print(f"PMC samples checked: {total}")
print(f"ipc_available count: {avail}")
print(f"ipc > 0 count: {nonzero}")
if total == 0:
    print("WARNING: PMC file has no rows")
elif avail == 0:
    print("WARNING: IPC fallback mode detected (hardware counters unavailable)")
' "$PMC"

mkdir -p data/processed results temp models/final models/final_bce models/ablation

log "Merging dataset..."
python preprocessing/merge_dataset.py \
  --state "$STATE" \
  --pmc "$PMC" \
  --rapl "$RAPL" \
  --output data/processed/final_dataset.csv

log "Running merge validation..."
python temp/test_merge_dataset.py \
  --state "$STATE" \
  --pmc "$PMC" \
  --rapl "$RAPL" \
  --output temp/merged_dataset_debug.csv

log "Training energy-aware model..."
python train/train_and_validate_model.py \
  --data-path data/processed/final_dataset.csv \
  --model-dir models/final \
  --run-mode single \
  --loss-mode energy \
  --epochs "$EPOCHS" \
  --metrics-output results/single_energy_metrics.csv

log "Training BCE baseline model..."
python train/train_and_validate_model.py \
  --data-path data/processed/final_dataset.csv \
  --model-dir models/final_bce \
  --run-mode single \
  --loss-mode bce \
  --epochs "$EPOCHS" \
  --metrics-output results/single_bce_metrics.csv

log "Running ablation suite..."
python train/train_and_validate_model.py \
  --data-path data/processed/final_dataset.csv \
  --model-dir models/ablation \
  --run-mode ablation \
  --epochs "$EPOCHS" \
  --metrics-output results/ablation_results.csv

if [[ "$RUN_GRAPHS" == "1" ]]; then
  log "Generating graphs..."
  sudo python3 graphs/review_1.py
fi

log "Done. Key outputs:"
echo "  data/processed/final_dataset.csv"
echo "  results/single_energy_metrics.csv"
echo "  results/single_bce_metrics.csv"
echo "  results/ablation_results.csv"
echo "  models/final/"
echo "  models/final_bce/"
echo "  models/ablation/"