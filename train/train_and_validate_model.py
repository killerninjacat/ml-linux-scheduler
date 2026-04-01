import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import joblib
import os
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score

# --- ARCHITECTURE ---
class SchedulerMLP(nn.Module):
    def __init__(self, input_size):
        super(SchedulerMLP, self).__init__()
        # Single hidden layer with 12 neurons for kernel-level efficiency
        self.network = nn.Sequential(
            nn.Linear(input_size, 12),
            nn.ReLU(),
            nn.Linear(12, 1) # Sigmoid handled by BCEWithLogitsLoss during training
        )

    def forward(self, x):
        return self.network(x)

# --- UTILS ---
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
    return np.mean(latencies) / 1000 # Convert ns to us

# --- MAIN PIPELINE ---
def run_pipeline(data_path, model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "scheduler_model.pth")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    # 1. Data Preparation
    print(f"[*] Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    X = df.drop(columns=['decision']).values
    y = df['decision'].values.reshape(-1, 1)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Scaler Persistence
    if os.path.exists(scaler_path):
        print("[*] Loading existing scaler...")
        scaler = joblib.load(scaler_path)
        X_train_scaled = scaler.transform(X_train)
    else:
        print("[*] Creating new scaler...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        joblib.dump(scaler, scaler_path)
    
    X_val_scaled = scaler.transform(X_val)
    input_size = X_train_scaled.shape[1]

    # 3. Model & Class Weight Calculation
    model = SchedulerMLP(input_size)
    if os.path.exists(model_path):
        print("[*] Resuming from previous session...")
        model.load_state_dict(torch.load(model_path))

    # Calculate weights to handle class imbalance (Stays vs Migrations)
    num_pos = (y_train == 1).sum()
    num_neg = (y_train == 0).sum()
    weight = torch.tensor([num_neg / num_pos]) if num_pos > 0 else torch.tensor([1.0])

    # 4. Training Loop
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    X_train_tensor = torch.FloatTensor(X_train_scaled.copy())
    y_train_tensor = torch.FloatTensor(y_train.copy())

    print(f"[*] Training for 100 epochs (Weighted: {weight.item():.2f}x)...")
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 25 == 0:
            print(f"    Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")

    # 5. Save State
    torch.save(model.state_dict(), model_path)
    print(f"[✓] Model saved to {model_path}")

    # 6. Validation Output
    model.eval()
    with torch.no_grad():
        X_val_tensor = torch.FloatTensor(X_val_scaled.copy())
        y_probs = torch.sigmoid(model(X_val_tensor)).numpy()
        y_pred = (y_probs > 0.5).astype(int)

        acc = accuracy_score(y_val, y_pred)
        pre = precision_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        latency = measure_latency(model, input_size)

        print("\n" + "="*30)
        print(f"Inference Latency: {latency:.3f} µs")
        print(f"Overall Accuracy: {acc:.4f}")
        print(f"Overall Precision: {pre:.4f}")
        print(f"Overall F1-Score: {f1:.4f}")
        print("="*30)

if __name__ == "__main__":
    run_pipeline('/home/anish/ml-linux-scheduler/data/processed/final_dataset_3.csv')