#!/usr/bin/env python3

import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class SchedulerMLP(nn.Module):
    def __init__(self, input_size, hidden_dim=32, dropout=0.1):
        super(SchedulerMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.network(x)


class EnergyAwareLoss(nn.Module):
    def __init__(
        self,
        pos_weight,
        power_idx=None,
        ipc_idx=None,
        alpha=0.02,
        beta=0.02,
        power_threshold=35.0,
        ipc_threshold=0.9,
        enable_power_term=True,
        enable_ipc_term=True,
    ):
        super(EnergyAwareLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.power_idx = power_idx
        self.ipc_idx = ipc_idx
        self.alpha = alpha
        self.beta = beta
        self.power_threshold = power_threshold
        self.ipc_threshold = ipc_threshold
        self.enable_power_term = enable_power_term and power_idx is not None
        self.enable_ipc_term = enable_ipc_term and ipc_idx is not None

    def forward(self, logits, targets, raw_features):
        base_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        stay_prob = 1.0 - probs

        power_term = logits.new_tensor(0.0)
        ipc_term = logits.new_tensor(0.0)

        if self.enable_power_term:
            power = raw_features[:, self.power_idx:self.power_idx + 1]
            power_scale = max(self.power_threshold, 1e-6)
            power_excess = torch.relu((power - self.power_threshold) / power_scale)
            power_term = (stay_prob * power_excess).mean()

        if self.enable_ipc_term:
            ipc = raw_features[:, self.ipc_idx:self.ipc_idx + 1]
            ipc_scale = max(self.ipc_threshold, 1e-6)
            ipc_deficit = torch.relu((self.ipc_threshold - ipc) / ipc_scale)

            if self.enable_power_term:
                power = raw_features[:, self.power_idx:self.power_idx + 1]
                power_scale = max(self.power_threshold, 1e-6)
                power_excess = torch.relu((power - self.power_threshold) / power_scale)
                ipc_term = (stay_prob * power_excess * ipc_deficit).mean()
            else:
                ipc_term = (stay_prob * ipc_deficit).mean()

        total_loss = base_loss + (self.alpha * power_term) + (self.beta * ipc_term)

        components = {
            "bce": float(base_loss.detach().cpu().item()),
            "power": float(power_term.detach().cpu().item()),
            "ipc": float(ipc_term.detach().cpu().item()),
            "total": float(total_loss.detach().cpu().item()),
        }
        return total_loss, components


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def measure_latency(model, input_size):
    model.eval()
    dummy_input = torch.randn(1, input_size)
    latencies = []
    with torch.no_grad():
        for _ in range(100):
            start = time.perf_counter_ns()
            _ = model(dummy_input)
            end = time.perf_counter_ns()
            latencies.append(end - start)
    return np.mean(latencies) / 1000.0


def load_and_split_dataset(data_path, test_size=0.2, random_state=42):
    print(f"[*] Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    if "decision" not in df.columns:
        raise ValueError("Input dataset must contain a 'decision' column")

    feature_columns = [c for c in df.columns if c != "decision"]

    for col in feature_columns + ["decision"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["decision"])
    df[feature_columns] = df[feature_columns].fillna(0)

    X = df[feature_columns].values.astype(np.float32)
    y = df["decision"].clip(lower=0, upper=1).values.astype(np.float32).reshape(-1, 1)

    stratify_labels = y.ravel() if np.unique(y).size > 1 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels,
    )

    return {
        "feature_columns": feature_columns,
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
    }


def compute_energy_case_stats(X_raw, y_probs, feature_index, power_threshold, ipc_threshold):
    power_idx = feature_index.get("power_watts")
    ipc_idx = feature_index.get("ipc")

    if power_idx is None:
        return {
            "energy_case_count": 0,
            "energy_case_avg_migrate_prob": 0.0,
        }

    mask = X_raw[:, power_idx] > power_threshold
    if ipc_idx is not None:
        mask = np.logical_and(mask, X_raw[:, ipc_idx] < ipc_threshold)

    case_count = int(mask.sum())
    if case_count == 0:
        return {
            "energy_case_count": 0,
            "energy_case_avg_migrate_prob": 0.0,
        }

    return {
        "energy_case_count": case_count,
        "energy_case_avg_migrate_prob": float(y_probs[mask].mean()),
    }


def find_best_threshold(y_true, y_probs, step=0.01):
    if step <= 0 or step >= 1:
        step = 0.01

    thresholds = np.arange(step, 1.0, step)
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in thresholds:
        y_pred = (y_probs > threshold).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, best_f1


def run_experiment(
    data_bundle,
    args,
    experiment_name,
    loss_mode,
    enable_power_term,
    enable_ipc_term,
):
    set_seed(args.seed)

    feature_columns = data_bundle["feature_columns"]
    X_train_raw = data_bundle["X_train"]
    X_val_raw = data_bundle["X_val"]
    y_train = data_bundle["y_train"]
    y_val = data_bundle["y_val"]

    experiment_dir = args.model_dir if experiment_name == "single" else os.path.join(args.model_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    model_path = os.path.join(experiment_dir, "scheduler_model.pth")
    scaler_path = os.path.join(experiment_dir, "scaler.pkl")
    metadata_path = os.path.join(experiment_dir, "training_metadata.json")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)

    feature_index = {name: idx for idx, name in enumerate(feature_columns)}

    print(f"[*] Experiment: {experiment_name}")
    print(f"[*] Loss mode: {loss_mode}")
    print(f"[*] Feature count: {len(feature_columns)}")
    print(f"[*] Features: {feature_columns}")

    if "ipc" in feature_index:
        ipc_values = X_train_raw[:, feature_index["ipc"]]
        print(f"[*] IPC range (train): {ipc_values.min():.4f} -> {ipc_values.max():.4f}")
    else:
        print("[!] IPC feature is missing from dataset")

    if "cache_misses" in feature_index:
        cache_values = X_train_raw[:, feature_index["cache_misses"]]
        print(f"[*] Cache-misses range (train): {cache_values.min():.4f} -> {cache_values.max():.4f}")
    else:
        print("[!] Cache-misses feature is missing from dataset")

    input_size = X_train_scaled.shape[1]
    model = SchedulerMLP(
        input_size,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )

    if args.resume and os.path.exists(model_path):
        print("[*] Resuming from previous model state...")
        try:
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as exc:
            print(f"[!] Resume skipped due to architecture mismatch: {exc}")

    num_pos = float((y_train == 1).sum())
    num_neg = float((y_train == 0).sum())
    pos_weight = torch.tensor([num_neg / max(num_pos, 1.0)], dtype=torch.float32)

    power_idx = feature_index.get("power_watts")
    ipc_idx = feature_index.get("ipc")

    if loss_mode == "energy":
        if enable_power_term and power_idx is None:
            print("[!] power_watts not found; disabling power penalty term")
            enable_power_term = False
        if enable_ipc_term and ipc_idx is None:
            print("[!] ipc not found; disabling IPC penalty term")
            enable_ipc_term = False

        criterion = EnergyAwareLoss(
            pos_weight=pos_weight,
            power_idx=power_idx,
            ipc_idx=ipc_idx,
            alpha=args.energy_alpha,
            beta=args.energy_beta,
            power_threshold=args.power_threshold,
            ipc_threshold=args.ipc_threshold,
            enable_power_term=enable_power_term,
            enable_ipc_term=enable_ipc_term,
        )
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    X_train_tensor = torch.FloatTensor(X_train_scaled.copy())
    X_train_raw_tensor = torch.FloatTensor(X_train_raw.copy())
    y_train_tensor = torch.FloatTensor(y_train.copy())

    X_val_tensor = torch.FloatTensor(X_val_scaled.copy())
    X_val_raw_tensor = torch.FloatTensor(X_val_raw.copy())
    y_val_tensor = torch.FloatTensor(y_val.copy())

    train_dataset = TensorDataset(X_train_tensor, X_train_raw_tensor, y_train_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    best_val_loss = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    patience_counter = 0

    model.train()
    print(f"[*] Training for {args.epochs} epochs (class weight: {pos_weight.item():.2f}x)")
    print(
        f"[*] Training config: batch_size={args.batch_size}, "
        f"hidden_dim={args.hidden_dim}, dropout={args.dropout}, "
        f"weight_decay={args.weight_decay}, patience={args.patience}"
    )

    for epoch in range(args.epochs):
        model.train()
        epoch_samples = 0
        epoch_components = {
            "bce": 0.0,
            "power": 0.0,
            "ipc": 0.0,
            "total": 0.0,
        }

        for batch_scaled, batch_raw, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_scaled)

            if loss_mode == "energy":
                loss, components = criterion(logits, batch_y, batch_raw)
            else:
                loss = criterion(logits, batch_y)
                components = {
                    "bce": float(loss.item()),
                    "power": 0.0,
                    "ipc": 0.0,
                    "total": float(loss.item()),
                }

            loss.backward()
            optimizer.step()

            batch_size_actual = batch_y.shape[0]
            epoch_samples += batch_size_actual
            for key in epoch_components:
                epoch_components[key] += components[key] * batch_size_actual

        if epoch_samples > 0:
            for key in epoch_components:
                epoch_components[key] /= epoch_samples

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_tensor)
            if loss_mode == "energy":
                val_loss, _ = criterion(val_logits, y_val_tensor, X_val_raw_tensor)
            else:
                val_loss = criterion(val_logits, y_val_tensor)

            y_val_probs = torch.sigmoid(val_logits).numpy().reshape(-1)
            y_val_pred = (y_val_probs > 0.5).astype(int)
            y_val_true = y_val.reshape(-1).astype(int)
            val_f1 = f1_score(y_val_true, y_val_pred, zero_division=0)

        val_loss_value = float(val_loss.detach().cpu().item())

        if val_loss_value < (best_val_loss - args.min_delta):
            best_val_loss = val_loss_value
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        report_interval = max(1, args.epochs // 4)
        if (epoch + 1) % report_interval == 0 or epoch == 0:
            print(
                "    Epoch "
                f"[{epoch + 1}/{args.epochs}] "
                f"TrainTotal={epoch_components['total']:.4f} "
                f"TrainBCE={epoch_components['bce']:.4f} "
                f"TrainPower={epoch_components['power']:.4f} "
                f"TrainIPC={epoch_components['ipc']:.4f} "
                f"ValLoss={val_loss_value:.4f} "
                f"ValF1@0.5={val_f1:.4f}"
            )

        if args.patience > 0 and patience_counter >= args.patience:
            print(
                f"[*] Early stopping at epoch {epoch + 1} "
                f"(best val loss={best_val_loss:.4f})"
            )
            break

    model.load_state_dict(best_state)

    torch.save(model.state_dict(), model_path)
    joblib.dump({"scaler": scaler, "feature_columns": feature_columns}, scaler_path)

    with open(metadata_path, "w") as f:
        json.dump(
            {
                "experiment": experiment_name,
                "loss_mode": loss_mode,
                "enable_power_term": enable_power_term,
                "enable_ipc_term": enable_ipc_term,
                "energy_alpha": args.energy_alpha,
                "energy_beta": args.energy_beta,
                "power_threshold": args.power_threshold,
                "ipc_threshold": args.ipc_threshold,
                "batch_size": args.batch_size,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "weight_decay": args.weight_decay,
                "best_val_loss": best_val_loss,
                "feature_columns": feature_columns,
            },
            f,
            indent=2,
        )

    model.eval()
    with torch.no_grad():
        y_probs = torch.sigmoid(model(X_val_tensor)).numpy().reshape(-1)
        y_true = y_val.reshape(-1).astype(int)

        if args.decision_threshold is not None:
            decision_threshold = float(args.decision_threshold)
            threshold_source = "manual"
        elif args.no_optimize_threshold:
            decision_threshold = 0.5
            threshold_source = "fixed"
        else:
            decision_threshold, best_threshold_f1 = find_best_threshold(
                y_true,
                y_probs,
                step=args.threshold_step,
            )
            threshold_source = f"optimized_f1={best_threshold_f1:.4f}"

        y_pred = (y_probs > decision_threshold).astype(int)

        acc = accuracy_score(y_true, y_pred)
        pre = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        latency = measure_latency(model, input_size)
        energy_case_stats = compute_energy_case_stats(
            X_val_raw,
            y_probs,
            feature_index,
            args.power_threshold,
            args.ipc_threshold,
        )

    metrics = {
        "experiment": experiment_name,
        "loss_mode": loss_mode,
        "accuracy": float(acc),
        "precision": float(pre),
        "f1": float(f1),
        "latency_us": float(latency),
        "decision_threshold": float(decision_threshold),
        "energy_case_count": int(energy_case_stats["energy_case_count"]),
        "energy_case_avg_migrate_prob": float(energy_case_stats["energy_case_avg_migrate_prob"]),
    }

    print("\n" + "=" * 40)
    print(f"Experiment: {experiment_name}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"F1-score: {metrics['f1']:.4f}")
    print(f"Decision threshold: {metrics['decision_threshold']:.4f} ({threshold_source})")
    print(f"Inference latency: {metrics['latency_us']:.3f} us")
    print(
        "Energy-case diagnostics: "
        f"count={metrics['energy_case_count']}, "
        f"avg_migrate_prob={metrics['energy_case_avg_migrate_prob']:.4f}"
    )
    print("=" * 40)

    return metrics


def run_single_mode(data_bundle, args):
    loss_mode = args.loss_mode
    enable_power_term = not args.disable_power_term
    enable_ipc_term = not args.disable_ipc_term

    if loss_mode == "bce":
        enable_power_term = False
        enable_ipc_term = False

    metrics = run_experiment(
        data_bundle=data_bundle,
        args=args,
        experiment_name="single",
        loss_mode=loss_mode,
        enable_power_term=enable_power_term,
        enable_ipc_term=enable_ipc_term,
    )

    if args.metrics_output:
        pd.DataFrame([metrics]).to_csv(args.metrics_output, index=False)
        print(f"[✓] Metrics saved to {args.metrics_output}")


def run_ablation_mode(data_bundle, args):
    experiments = [
        {
            "name": "baseline_bce",
            "loss_mode": "bce",
            "enable_power_term": False,
            "enable_ipc_term": False,
        },
        {
            "name": "bce_plus_power",
            "loss_mode": "energy",
            "enable_power_term": True,
            "enable_ipc_term": False,
        },
        {
            "name": "bce_plus_power_ipc",
            "loss_mode": "energy",
            "enable_power_term": True,
            "enable_ipc_term": True,
        },
    ]

    results = []
    for experiment in experiments:
        metrics = run_experiment(
            data_bundle=data_bundle,
            args=args,
            experiment_name=experiment["name"],
            loss_mode=experiment["loss_mode"],
            enable_power_term=experiment["enable_power_term"],
            enable_ipc_term=experiment["enable_ipc_term"],
        )
        results.append(metrics)

    results_df = pd.DataFrame(results)
    os.makedirs(args.model_dir, exist_ok=True)
    ablation_output = args.metrics_output or os.path.join(args.model_dir, "ablation_results.csv")
    results_df.to_csv(ablation_output, index=False)
    print(f"[✓] Ablation results saved to {ablation_output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train and validate scheduler MLP")
    parser.add_argument(
        "--data-path",
        default="data/processed/final_dataset.csv",
        help="Path to merged CSV dataset",
    )
    parser.add_argument(
        "--model-dir",
        default="models/final",
        help="Directory to save model/scaler/metadata",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Resume from saved model weights")
    parser.add_argument(
        "--run-mode",
        choices=["single", "ablation"],
        default="single",
        help="Run one experiment or all ablations",
    )
    parser.add_argument(
        "--loss-mode",
        choices=["bce", "energy"],
        default="energy",
        help="Loss for single-mode run",
    )
    parser.add_argument("--energy-alpha", type=float, default=0.01, help="Weight for power penalty")
    parser.add_argument("--energy-beta", type=float, default=0.01, help="Weight for IPC penalty")
    parser.add_argument("--power-threshold", type=float, default=35.0, help="Power threshold in watts")
    parser.add_argument("--ipc-threshold", type=float, default=0.9, help="IPC low-efficiency threshold")
    parser.add_argument("--decision-threshold", type=float, default=None, help="Optional fixed probability threshold")
    parser.add_argument("--threshold-step", type=float, default=0.01, help="Step size for threshold search")
    parser.add_argument("--no-optimize-threshold", action="store_true", help="Disable validation threshold tuning")
    parser.add_argument("--disable-power-term", action="store_true")
    parser.add_argument("--disable-ipc-term", action="store_true")
    parser.add_argument("--metrics-output", default=None, help="Optional CSV path for metric output")
    return parser.parse_args()


def main():
    args = parse_args()
    data_bundle = load_and_split_dataset(
        data_path=args.data_path,
        test_size=args.test_size,
        random_state=args.seed,
    )

    if args.run_mode == "ablation":
        run_ablation_mode(data_bundle, args)
    else:
        run_single_mode(data_bundle, args)


if __name__ == "__main__":
    main()