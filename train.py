# ============================================================
# train.py — OFFICIAL MAMBA (CUDA FUSED) IMPLEMENTATION (STABLE)
# Target: Achieving F1 > 0.75 using Aggressive Fine-Tuning
# ============================================================

import os
import json
import argparse
import random
from collections import Counter
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel


# --- Configuration and Directories ---
DATA_DIR = "processed_data"
MODEL_DIR = "trained_mamba_pretrained_model"
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# DATASET
# ============================================================

class MCIDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=384): # Default Max Len set
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.df.loc[idx, "transcription"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "labels": torch.tensor(self.df.loc[idx, "label_id"], dtype=torch.long),
        }


# ============================================================
# MODEL COMPONENTS
# ============================================================

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):
        w = torch.softmax(self.attn(x), dim=1)
        return (x * w).sum(dim=1)


class MambaClassifier(nn.Module):
    def __init__(self, base_model, num_labels, class_weights):
        super().__init__()
        self.base = base_model.to(torch.float32) # Ensure model runs in float32 initially
        hidden = base_model.config.d_model
        
        # --- Custom Classification Head ---
        self.pool = AttentionPooling(hidden) 
        self.fc = nn.Linear(hidden, hidden // 4)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden // 4, num_labels)
        
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=0.1
        )

    def forward(self, input_ids, labels=None):
        out = self.base.backbone(input_ids) 
        
        # Apply classification head
        pooled = self.pool(out)
        x = self.dropout(torch.nn.functional.gelu(self.fc(pooled)))
        logits = self.classifier(x)
        
        loss = self.criterion(logits, labels) if labels is not None else None
        return loss, logits


# ============================================================
# UTILITIES
# ============================================================

class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.005, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if self.mode == 'min':
            self.best_score = float('inf')
        elif self.mode == 'max':
            self.best_score = float('-inf')

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
        print(f'✓ New best model saved (Val F1: {self.best_score:.3f}')
        torch.save(model_state, model_path)


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels, losses = [], [], []

    for batch in loader:
        ids = batch["input_ids"].to(device)
        y = batch["labels"].to(device)

        with autocast(enabled=(device == "cuda")):
            loss, logits = model(ids, y)

        losses.append(loss.item())
        preds.extend(torch.argmax(logits, 1).cpu().numpy())
        labels.extend(y.cpu().numpy())

    return (
        np.mean(losses),
        accuracy_score(labels, preds),
        f1_score(labels, preds, average="weighted"),
        labels,
        preds,
    )


# ============================================================
# TRAINING PIPELINE
# ============================================================

def main(args):
    # -------------------- Setup --------------------
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # -------------------- Load data --------------------
    df = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))

    with open(os.path.join(DATA_DIR, "transcripts_cache.json")) as f:
        cache = json.load(f)

    df["transcription"] = df["audio_path"].map(cache["transcripts"])
    df = df.dropna(subset=["transcription"])
    df["label_id"] = df["label"].map({"Control": 0, "Dementia": 1})

    # Splits
    train_df, test_df = train_test_split(
        df, test_size=0.15, stratify=df["label_id"], random_state=args.seed
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.10, stratify=train_df["label_id"], random_state=args.seed
    )

    # -------------------- Tokenizer --------------------
    TOKENIZER_ID = "EleutherAI/gpt-neox-20b"
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    tokenizer.pad_token = tokenizer.eos_token

    train_ds = MCIDataset(train_df, tokenizer, max_len=args.max_len) 
    val_ds = MCIDataset(val_df, tokenizer, max_len=args.max_len)
    test_ds = MCIDataset(test_df, tokenizer, max_len=args.max_len)

    # Weighted Sampling (CRITICAL)
    train_weights = compute_class_weight(
        "balanced", classes=np.array([0, 1]), y=train_df["label_id"].values
    )
    sample_weights = [train_weights[label] for label in train_df["label_id"]]
    sampler = WeightedRandomSampler(sample_weights, len(train_df), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2)

    # -------------------- Model --------------------
    class_weights = torch.tensor(train_weights, device=device, dtype=torch.float)
    
    print("Loading Mamba CUDA-fused backbone...")
    base_model = MambaLMHeadModel.from_pretrained(args.mamba_model)
    
    model = MambaClassifier(base_model, num_labels=2, class_weights=class_weights).to(device)

    # --- AGGRESSIVE FINE-TUNING STRATEGY (Reduced Unfreeze: 5 Layers) ---
    FREEZE_COUNT = base_model.config.n_layer - args.unfreeze_layers # 24 - 5 = 19 layers frozen
    
    # Freeze embedding layer
    for p in model.base.backbone.embedding.parameters(): 
        p.requires_grad = False
    
    # Freeze first FREEZE_COUNT Mamba blocks
    for i in range(FREEZE_COUNT):
        for p in model.base.backbone.layers[i].parameters():
            p.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    # Define optimizer parameters for differential learning rates
    optimizer_params = [
        {"params": filter(lambda p: p.requires_grad, model.base.backbone.parameters()), "lr": args.lr * 0.1}, # Lower LR for backbone
        {"params": model.fc.parameters(), "lr": args.lr},
        {"params": model.classifier.parameters(), "lr": args.lr},
    ]

    # --- Optimizer ---
    opt = torch.optim.AdamW(optimizer_params, weight_decay=0.02) # Increased WD for stability
    scaler = GradScaler(enabled=(device == "cuda"))
    early_stopper = EarlyStopper(patience=args.patience, min_delta=0.005, mode='max')
    
    # --- History Initialization ---
    history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'train_f1': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    
    print(f"\n{'='*70}")
    print(f"MAMBA CUDA-FUSED AGGRESSIVE FINE-TUNING")
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
    print(f"Learning Rate: {args.lr} | Unfrozen Layers: {args.unfreeze_layers}")
    print(f"{'='*70}")

    # -------------------- Training loop --------------------
    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        train_preds, train_labels, train_losses = [], [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for i, batch in enumerate(pbar):
            ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with autocast(enabled=(device == 'cuda')): 
                loss, logits = model(ids, labels)
                loss = loss / args.acc_steps

            scaler.scale(loss).backward()
            train_losses.append(loss.item() * args.acc_steps)
            train_preds.extend(logits.argmax(1).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            # --- Gradient Accumulation and Step (CRITICAL) ---
            if (i + 1) % args.acc_steps == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # CRITICAL: Gradient Clipping
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                
            pbar.set_postfix({'loss': f'{np.mean(train_losses):.4f}'})

        # Calculate Train Metrics
        train_loss = float(np.mean(train_losses))
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='weighted')

        # Validation
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, device)
        
        # --- Record History ---
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1} Summary:")
        print(f"TRAIN Loss: {train_loss:.4f} | Acc: {train_acc:.3f} | F1: {train_f1:.3f}")
        print(f"VAL   Loss: {val_loss:.4f} | Acc: {val_acc:.3f} | F1: {val_f1:.3f}")
        print(f"{'='*70}")

        if early_stopper(val_f1, model.state_dict(), os.path.join(MODEL_DIR, "best_pytorch_model.bin")):
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break

    # --- Final Test ---
    try:
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_pytorch_model.bin"), map_location=device))
    except FileNotFoundError:
        print("Warning: Best model not found. Using last epoch's state.")

    test_loss, test_acc, test_f1, test_labels, test_preds = evaluate(model, test_loader, device)

    print("\nFINAL TEST RESULTS")
    print(f"Loss {test_loss:.4f} | Acc {test_acc:.3f} | F1 {test_f1:.3f}")
    print("\nClassification report (Test):")
    print(classification_report(test_labels, test_preds, target_names=['Control', 'Dementia']))
    
    cm = confusion_matrix(test_labels, test_preds)
    print("\nConfusion Matrix:")
    print(f"                  Predicted")
    print(f"              Control  Dementia")
    print(f"Actual Control    {cm[0][0]:3d}      {cm[0][1]:3d}")
    print(f"       Dementia   {cm[1][0]:3d}      {cm[1][1]:3d}")
    
    tokenizer.save_pretrained(MODEL_DIR)
    
    # --- Final History Save ---
    history_path = os.path.join(MODEL_DIR, 'training_history.json')
    with open(history_path, 'w') as f:
        # Convert NumPy float types to standard Python floats for JSON serialization
        serializable_history = {k: [float(x) for x in v] for k, v in history.items()}
        json.dump(serializable_history, f, indent=4)
    print(f"\nTraining history successfully saved to {history_path}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Decreased unfreeze layers and increased patience
    parser.add_argument("--epochs", type=int, default=20) 
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--acc_steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience.") # Increased patience
    parser.add_argument("--mamba_model", type=str, default="state-spaces/mamba-130m")
    parser.add_argument("--unfreeze_layers", type=int, default=5, help="Number of Mamba layers to unfreeze.") # Decreased unfreeze
    parser.add_argument("--max_len", type=int, default=512, help="Maximum sequence length.") # Increased max_len
    args = parser.parse_args()

    main(args)