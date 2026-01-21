# ============================================================
# acoustic_only.py — RIGOROUS NEURAL BASELINE (PYTORCH)
# Matches Hyperparameters of Textual Stream (train.py)
# ============================================================

import os
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler

# --- Configuration and Directories ---
DATA_DIR = "processed_data"
MODEL_DIR = "trained_acoustic_baseline_model"
os.makedirs(MODEL_DIR, exist_ok=True)

# Hyperparameters (Matched to Textual Stream)
SEED = 42
EPOCHS = 20
BATCH_SIZE = 8
ACC_STEPS = 4
LR = 5e-4
PATIENCE = 7

# ============================================================
# DATASET
# ============================================================

class AcousticDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            "features": self.X[idx],
            "labels": self.y[idx]
        }

# ============================================================
# MODEL: Acoustic MLP
# ============================================================

class AcousticClassifier(nn.Module):
    def __init__(self, input_dim, num_labels, class_weights):
        super().__init__()
        # Architecture designed to be comparable to the classification head of the Mamba model
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32), # Equivalent to hidden // 4 in your Mamba head
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_labels)
        )
        
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=0.1
        )

    def forward(self, features, labels=None):
        logits = self.net(features)
        loss = self.criterion(logits, labels) if labels is not None else None
        return loss, logits

# ============================================================
# UTILITIES (EarlyStopper cloned from train.py)
# ============================================================

class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.005, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if self.mode == 'min': self.best_score = float('inf')
        elif self.mode == 'max': self.best_score = float('-inf')

    def __call__(self, current_score, model_state, model_path):
        if self.best_score is None:
            self.best_score = current_score
            self.save_checkpoint(model_state, model_path)
        elif (self.mode == 'min' and current_score > self.best_score - self.min_delta) or \
             (self.mode == 'max' and current_score < self.best_score + self.min_delta):
            self.counter += 1
            print(f'Patience: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (self.mode == 'min' and current_score < self.best_score) or \
               (self.mode == 'max' and current_score > self.best_score):
                self.best_score = current_score
                self.save_checkpoint(model_state, model_path)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model_state, model_path):
        print(f'✓ New best model saved (Val F1: {self.best_score:.3f})')
        torch.save(model_state, model_path)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels, losses = [], [], []
    for batch in loader:
        x = batch["features"].to(device)
        y = batch["labels"].to(device)
        loss, logits = model(x, y)
        losses.append(loss.item())
        preds.extend(torch.argmax(logits, 1).cpu().numpy())
        labels.extend(y.cpu().numpy())
    return np.mean(losses), accuracy_score(labels, preds), f1_score(labels, preds, average="weighted"), labels, preds

# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load and Standardize Data
    metadata = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))
    features_df = pd.read_csv(os.path.join(DATA_DIR, "acoustic_features.csv"))
    
    # Standardize ID column
    features_df['participant_id'] = features_df['file_id'].str.replace('_participant', '', regex=False)
    df = pd.merge(metadata, features_df, on="participant_id")
    df["label_id"] = df["label"].map({"Control": 0, "Dementia": 1})

    # Drop non-numeric/identifier columns
    drop_cols = ['participant_id', 'file_id', 'label', 'audio_path', 'label_id']
    X_raw = df.drop(columns=[c for c in drop_cols if c in df.columns]).values
    y_raw = df["label_id"].values

    # 2. Split (Exact Same split logic as train.py)
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.15, stratify=y_raw, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, stratify=y_train, random_state=SEED)

    # 3. Scale Features (MLP specific requirement)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # 4. Sampler and Dataloaders
    train_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    sample_weights = [train_weights[label] for label in y_train]
    sampler = WeightedRandomSampler(sample_weights, len(y_train), replacement=True)

    train_loader = DataLoader(AcousticDataset(X_train, y_train), batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(AcousticDataset(X_val, y_val), batch_size=BATCH_SIZE * 2)
    test_loader = DataLoader(AcousticDataset(X_test, y_test), batch_size=BATCH_SIZE * 2)

    # 5. Model initialization
    class_weights_tensor = torch.tensor(train_weights, device=device, dtype=torch.float)
    model = AcousticClassifier(input_dim=X_train.shape[1], num_labels=2, class_weights=class_weights_tensor).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.02)
    early_stopper = EarlyStopper(patience=PATIENCE, min_delta=0.005, mode='max')
    
    # 6. Training loop
    history = {'epoch': [], 'train_f1': [], 'val_f1': []}
    
    print(f"\nTraining Acoustic Stream on {device}...")
    for epoch in range(EPOCHS):
        model.train()
        train_preds, train_labels = [], []
        optimizer.zero_grad()

        for i, batch in enumerate(train_loader):
            x = batch["features"].to(device)
            y = batch["labels"].to(device)
            
            loss, logits = model(x, y)
            loss = loss / ACC_STEPS
            loss.backward()

            if (i + 1) % ACC_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            train_preds.extend(logits.argmax(1).cpu().numpy())
            train_labels.extend(y.cpu().numpy())

        # Validation
        _, _, v_f1, _, _ = evaluate(model, val_loader, device)
        t_f1 = f1_score(train_labels, train_preds, average='weighted')
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train F1: {t_f1:.3f} | Val F1: {v_f1:.3f}")
        
        if early_stopper(v_f1, model.state_dict(), os.path.join(MODEL_DIR, "best_acoustic_model.bin")):
            break

    # 7. Final Test
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_acoustic_model.bin")))
    _, test_acc, test_f1, test_labels, test_preds = evaluate(model, test_loader, device)

    print("\n" + "="*50)
    print("FINAL ACOUSTIC NEURAL RESULTS (Hyperparameter Matched)")
    print("="*50)
    print(f"F1 Score: {test_f1:.4f} | Accuracy: {test_acc:.4f}")
    print(classification_report(test_labels, test_preds, target_names=['Control', 'Dementia']))

if __name__ == "__main__":
    main()