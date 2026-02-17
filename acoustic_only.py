# ============================================================
# acoustic_only.py — STATISTICALLY CORRECT ACOUSTIC BASELINE
# Designed for ~70+ dimensional clinical acoustic features
# ============================================================

import os
import argparse
import random
import json
import joblib

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# Configuration
# ============================================================

DATA_PATH = "processed_data/master_fusion_data.csv"
MODEL_DIR = "trained_acoustic_baseline_model"
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Dataset
# ============================================================

class AcousticDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "features": self.X[idx],
            "labels": self.y[idx]
        }


# ============================================================
# Model
# ============================================================

class AcousticClassifier(nn.Module):
    def __init__(self, input_dim, num_labels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.35),

            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.25),

            nn.Linear(64, num_labels)
        )

        # No class weights here (sampler handles imbalance)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, x, labels=None):
        logits = self.net(x)
        loss = self.criterion(logits, labels) if labels is not None else None
        return loss, logits


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels, losses = [], [], []

    for batch in loader:
        x = batch["features"].to(device)
        y = batch["labels"].to(device)

        with autocast(enabled=(device.type == "cuda")):
            loss, logits = model(x, y)

        losses.append(loss.item())
        preds.extend(torch.argmax(logits, 1).cpu().numpy())
        labels.extend(y.cpu().numpy())

    return (
        np.mean(losses),
        accuracy_score(labels, preds),
        f1_score(labels, preds, average="weighted"),
        labels,
        preds
    )


# ============================================================
# Early Stopper
# ============================================================

class EarlyStopper:
    def __init__(self, patience=8, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -np.inf
        self.counter = 0

    def step(self, score, model, path):
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            torch.save(model.state_dict(), path)
            print(f"✓ New best model saved (Val F1: {score:.4f})")
        else:
            self.counter += 1
            print(f"Patience: {self.counter}/{self.patience}")

        return self.counter >= self.patience


# ============================================================
# Training Pipeline
# ============================================================

def main(args):

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    y = df["label"].map({"Control": 0, "Dementia": 1}).values
    X = df.drop(
        columns=["participant_id", "file_id", "label",
                 "transcription", "audio_path", "label_id"],
        errors="ignore"
    ).values

    print("Total samples:", X.shape[0])
    print("Feature dimension:", X.shape[1])

    # --------------------------------------------------------
    # Split FIRST (no leakage)
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=args.seed
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.10, stratify=y_train, random_state=args.seed
    )

    # --------------------------------------------------------
    # Standardize (FIT ONLY ON TRAIN)
    # --------------------------------------------------------

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    joblib.dump(scaler, os.path.join(MODEL_DIR, "acoustic_scaler.pkl"))

    # --------------------------------------------------------
    # Sampler (better than class weights alone)
    # --------------------------------------------------------

    class_weights = compute_class_weight(
        "balanced",
        classes=np.array([0, 1]),
        y=y_train
    )

    sample_weights = [class_weights[label] for label in y_train]
    sampler = WeightedRandomSampler(sample_weights, len(y_train), replacement=True)

    # --------------------------------------------------------
    # Dataloaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        AcousticDataset(X_train, y_train),
        batch_size=args.batch_size,
        sampler=sampler
    )

    val_loader = DataLoader(
        AcousticDataset(X_val, y_val),
        batch_size=args.batch_size * 2
    )

    test_loader = DataLoader(
        AcousticDataset(X_test, y_test),
        batch_size=args.batch_size * 2
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = AcousticClassifier(
        input_dim=X_train.shape[1],
        num_labels=2
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.02
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    scaler_amp = GradScaler(enabled=(device.type == "cuda"))
    early_stopper = EarlyStopper(patience=args.patience)

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\nStarting acoustic training...\n")

    for epoch in range(args.epochs):

        model.train()
        train_preds, train_labels = [], []
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for i, batch in enumerate(pbar):

            x = batch["features"].to(device)
            y_batch = batch["labels"].to(device)

            with autocast(enabled=(device.type == "cuda")):
                loss, logits = model(x, y_batch)
                loss = loss / args.acc_steps

            scaler_amp.scale(loss).backward()

            if (i + 1) % args.acc_steps == 0:
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler_amp.step(optimizer)
                scaler_amp.update()
                optimizer.zero_grad()

            train_preds.extend(torch.argmax(logits, 1).cpu().numpy())
            train_labels.extend(y_batch.cpu().numpy())

        scheduler.step()

        train_f1 = f1_score(train_labels, train_preds, average="weighted")
        _, _, val_f1, _, _ = evaluate(model, val_loader, device)

        print(f"\nEpoch {epoch+1} | Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")

        if early_stopper.step(
            val_f1,
            model,
            os.path.join(MODEL_DIR, "best_acoustic_model.bin")
        ):
            print(">> Early stopping triggered.")
            break

    # --------------------------------------------------------
    # Final Test
    # --------------------------------------------------------

    model.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, "best_acoustic_model.bin"),
        map_location=device
    ))

    _, test_acc, test_f1, test_labels, test_preds = evaluate(model, test_loader, device)

    print("\n==================================================")
    print("FINAL ACOUSTIC RESULTS (Leak-Free)")
    print("==================================================")
    print(f"F1 Score: {test_f1:.4f} | Accuracy: {test_acc:.4f}\n")

    print(classification_report(test_labels, test_preds,
                                target_names=["Control", "Dementia"]))

    print("\nConfusion Matrix:")
    print(confusion_matrix(test_labels, test_preds))


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--acc_steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)

