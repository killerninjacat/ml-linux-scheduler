#!/usr/bin/env bash
set -Eeuo pipefail

DURATION="${1:-60}"
EPOCHS="${2:-100}"
RUN_GRAPHS="${3:-0}"
ENERGY_ALPHA="${4:-0.01}"
ENERGY_BETA="${5:-0.01}"
POWER_THRESHOLD="${6:-35.0}"
IPC_THRESHOLD="${7:-0.9}"
RUN_SWEEP="${8:-0}"
ALPHA_GRID="${9:-0.005,0.01,0.02}"
BETA_GRID="${10:-0.005,0.01,0.02}"

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

log "Energy loss parameters: alpha=${ENERGY_ALPHA}, beta=${ENERGY_BETA}, power_th=${POWER_THRESHOLD}, ipc_th=${IPC_THRESHOLD}"
if [[ "$RUN_SWEEP" == "1" ]]; then
  log "Sweep mode enabled with alpha_grid=${ALPHA_GRID} and beta_grid=${BETA_GRID}"
fi

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

run_energy_training() {
  local alpha="$1"
  local beta="$2"
  local model_dir="$3"
  local metrics_file="$4"

  python train/train_and_validate_model.py \
    --data-path data/processed/final_dataset.csv \
    --model-dir "$model_dir" \
    --run-mode single \
    --loss-mode energy \
    --energy-alpha "$alpha" \
    --energy-beta "$beta" \
    --power-threshold "$POWER_THRESHOLD" \
    --ipc-threshold "$IPC_THRESHOLD" \
    --epochs "$EPOCHS" \
    --metrics-output "$metrics_file"
}

safe_tag() {
  echo "$1" | tr '.' 'p' | tr '-' 'm'
}

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
run_energy_training "$ENERGY_ALPHA" "$ENERGY_BETA" "models/final" "results/single_energy_metrics.csv"

if [[ "$RUN_SWEEP" == "1" ]]; then
  mkdir -p models/sweeps results/sweeps
  SWEEP_SUMMARY="results/alpha_beta_sweep.csv"
  SWEEP_RANKED="results/alpha_beta_sweep_ranked.csv"

  echo "alpha,beta,accuracy,precision,f1,latency_us,energy_case_count,energy_case_avg_migrate_prob,metrics_file,model_dir" > "$SWEEP_SUMMARY"

  IFS=',' read -r -a ALPHA_VALUES <<< "$ALPHA_GRID"
  IFS=',' read -r -a BETA_VALUES <<< "$BETA_GRID"

  for alpha in "${ALPHA_VALUES[@]}"; do
    alpha="${alpha// /}"
    [[ -n "$alpha" ]] || continue

    for beta in "${BETA_VALUES[@]}"; do
      beta="${beta// /}"
      [[ -n "$beta" ]] || continue

      alpha_tag="$(safe_tag "$alpha")"
      beta_tag="$(safe_tag "$beta")"
      model_dir="models/sweeps/a_${alpha_tag}_b_${beta_tag}"
      metrics_file="results/sweeps/metrics_a_${alpha_tag}_b_${beta_tag}.csv"

      log "Sweep run: alpha=${alpha}, beta=${beta}"
      run_energy_training "$alpha" "$beta" "$model_dir" "$metrics_file"

      python - "$metrics_file" "$alpha" "$beta" "$SWEEP_SUMMARY" "$model_dir" <<'PY'
import csv
import pandas as pd
import sys

metrics_file, alpha, beta, summary_file, model_dir = sys.argv[1:]
df = pd.read_csv(metrics_file)
if df.empty:
    raise SystemExit(f"Empty metrics file: {metrics_file}")

row = df.iloc[0]
with open(summary_file, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        alpha,
        beta,
        row.get("accuracy", 0.0),
        row.get("precision", 0.0),
        row.get("f1", 0.0),
        row.get("latency_us", 0.0),
        row.get("energy_case_count", 0),
        row.get("energy_case_avg_migrate_prob", 0.0),
        metrics_file,
        model_dir,
    ])
PY
    done
  done

  python - "$SWEEP_SUMMARY" "$SWEEP_RANKED" <<'PY'
import pandas as pd
import sys

summary_file, ranked_file = sys.argv[1:]
df = pd.read_csv(summary_file)
if df.empty:
    raise SystemExit("Sweep summary is empty")

ranked = df.sort_values(
    by=["f1", "accuracy", "energy_case_avg_migrate_prob"],
    ascending=[False, False, False],
)
ranked.to_csv(ranked_file, index=False)

print(f"[INFO] Sweep summary saved: {summary_file}")
print(f"[INFO] Sweep ranked results saved: {ranked_file}")
print("[INFO] Top 5 sweep configs:")
print(ranked.head(5).to_string(index=False))
PY
fi

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
  --energy-alpha "$ENERGY_ALPHA" \
  --energy-beta "$ENERGY_BETA" \
  --power-threshold "$POWER_THRESHOLD" \
  --ipc-threshold "$IPC_THRESHOLD" \
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
if [[ "$RUN_SWEEP" == "1" ]]; then
  echo "  results/alpha_beta_sweep.csv"
  echo "  results/alpha_beta_sweep_ranked.csv"
  echo "  results/sweeps/"
  echo "  models/sweeps/"
fi
echo "  models/final/"
echo "  models/final_bce/"
echo "  models/ablation/"
