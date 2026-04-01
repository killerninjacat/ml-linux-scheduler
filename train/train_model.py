import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- ARCHITECTURE ---
class SchedulerMLP(nn.Module):
    def __init__(self, input_size):
        super(SchedulerMLP, self).__init__()
        # 1 Hidden Layer, 10-16 nodes to match Illinois research constraints
        self.network = nn.Sequential(
            nn.Linear(input_size, 12),
            nn.ReLU(),
            nn.Linear(12, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

# --- TRAINING & PERSISTENCE LOGIC ---
def run_training_session(data_path, model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "scheduler_model.pth")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    # 1. Load the new training data session
    print(f"[*] Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    X = df.drop(columns=['decision']).values
    y = df['decision'].values.reshape(-1, 1)

    # 2. Persistence Check: The Scaler
    # We must use the same scaler across sessions to keep data distribution consistent
    if os.path.exists(scaler_path):
        print("[*] Loading existing scaler...")
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X)
    else:
        print("[*] Creating new scaler...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, scaler_path)

    # 3. Persistence Check: The Model
    input_size = X_scaled.shape[1]

    print("input_size:", input_size)

    model = SchedulerMLP(input_size)
    
    if os.path.exists(model_path):
        print("[*] Loading existing weights from previous session...")
        model.load_state_dict(torch.load(model_path))
    else:
        print("[*] No previous model found. Starting fresh.")

    # 4. Training Setup
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # X_tensor = torch.from_numpy(X_scaled).float()
    # y_tensor = torch.from_numpy(y).float()
    
    X_tensor = torch.FloatTensor(X_scaled.copy())
    y_tensor = torch.FloatTensor(y.copy())


    # 5. Incremental Training Loop
    print("[*] Beginning training session...")
    model.train()
    for epoch in range(50): # Shorter sessions for incremental learning
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/50], Loss: {loss.item():.4f}")

    # 6. Save State for Next Session
    torch.save(model.state_dict(), model_path)
    print(f"[✓] Session complete. Weights saved to {model_path}")

if __name__ == "__main__":
    # You can point this to any new CSV you generate
    run_training_session('/home/anish/ml-linux-scheduler/data/processed/final_dataset_3.csv')