import os
import json
import random
import argparse
from collections import Counter
import warnings
import time

warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, LongformerConfig

# --- CONFIG ---
MODEL_ID = "allenai/longformer-base-4096"
MAX_LEN = 512 # REDUCED for critical VRAM optimization

# --- MODEL (Simplified Hugging Face Wrapper) ---

class LongformerClassifier(nn.Module):
    def __init__(self, config, num_labels, class_weights):
        super().__init__()
        # Load Longformer with a sequence classification head
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config, 
            num_labels=num_labels, 
            ignore_mismatched_sizes=True,
            # CRITICAL FIX: Force loading the safetensors variant to bypass PyTorch security vulnerability
            use_safetensors=True,
            variant="safetensors"
        )
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        return loss, logits

# --- DATA ---

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        # Ensure attention_mask is present, crucial for Longformer
        if 'attention_mask' not in item:
             item['attention_mask'] = torch.ones_like(item['input_ids'])
        return item
    
    def __len__(self):
        return len(self.labels)

# --- UTILITIES (Including Early Stopping) ---

class EarlyStopper:
    def __init__(self, patience=7, min_delta=0.001, mode='max'):
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
            print(f'EarlyStopper counter: {self.counter} out of {self.patience}')
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
        print(f"✓ Saved new best model (val F1 = {self.best_score:.3f})")
        torch.save(model_state, model_path)


def load_cached_data(path, cache="processed_data/transcripts_cache.json"):
    """Load only cached transcripts - assumes cache has been generated."""
    df = pd.read_csv(path)
    if not os.path.exists(cache):
        raise FileNotFoundError(f"Cache not found: {cache}. Run train.py with Whisper once to generate cache.")
    
    with open(cache) as f:
        cache = json.load(f)
    df['transcription'] = df['audio_path'].map(cache['transcripts'])
    df = df[df['transcription'].str.strip() != '']
    return df

def evaluate(model, loader, device):
    model.eval()
    losses, preds, labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            y = batch['labels'].to(device)
            
            with autocast(enabled=(device == "cuda")):
                loss, logits = model(ids, mask, y)

            losses.append(loss.item())
            preds.extend(logits.argmax(1).cpu().numpy())
            labels.extend(y.cpu().numpy())

    return (
        float(np.mean(losses)), 
        accuracy_score(labels, preds), 
        f1_score(labels, preds, average="weighted"), 
        labels, 
        preds
    )

# --- MAIN TRAINING PIPELINE ---

def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUTPUT_DIR = 'trained_longformer_baseline'
    BEST_PATH = os.path.join(OUTPUT_DIR, 'best_pytorch_model.bin')
    HISTORY_PATH = os.path.join(OUTPUT_DIR, 'training_history.json') # Added history path
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\n--- LONGFORMER BASELINE TRAINING ---")
    print(f"Device: {device}")
    
    # Load and Split Data
    df = load_cached_data("processed_data/metadata.csv")
    df["label_id"] = (df["label"] == "Dementia").astype(int)

    train_df, test_df = train_test_split(df, test_size=0.15, stratify=df["label_id"], random_state=args.seed)
    train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df["label_id"], random_state=args.seed)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    encode = lambda x: tokenizer(x.tolist(), truncation=True, padding='max_length', max_length=MAX_LEN)

    train_ds = SimpleDataset(encode(train_df.transcription), train_df.label_id.tolist())
    val_ds = SimpleDataset(encode(val_df.transcription), val_df.label_id.tolist())
    test_ds = SimpleDataset(encode(test_df.transcription), test_df.label_id.tolist())

    # Weighted Sampling (for Imbalance)
    sampler = WeightedRandomSampler(
        [1 / Counter(train_df.label_id)[i] for i in train_df.label_id], len(train_df)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2)

    # Class Weights (for Loss Function)
    weights = torch.tensor(
        compute_class_weight("balanced", classes=np.array([0, 1]), y=train_df.label_id.values),
        device=device, dtype=torch.float,
    )
    
    # Model Initialization (Longformer)
    model = LongformerClassifier(MODEL_ID, num_labels=2, class_weights=weights).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = GradScaler(enabled=(device == "cuda"))

    best_val_f1 = -1.0
    early_stopper = EarlyStopper(patience=args.patience, min_delta=0.005, mode='max') # Changed min_delta for stability
    
    # --- History Initialization ---
    history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'train_f1': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    
    # -------- Training Loop --------
    for epoch in range(args.epochs):
        model.train()
        train_preds, train_labels, train_losses = [], [], []

        optimizer.zero_grad()
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)

            with autocast(enabled=(device == "cuda")):
                loss, logits = model(ids, mask, y)
                loss = loss / args.acc_steps

            scaler.scale(loss).backward()
            train_losses.append(loss.item() * args.acc_steps)
            train_preds.extend(logits.argmax(1).cpu().numpy())
            train_labels.extend(y.cpu().numpy())

            if (i + 1) % args.acc_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        
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

        # Early Stopping Logic
        if early_stopper(val_f1, model.state_dict(), BEST_PATH):
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break

    # -------- Final Test --------
    try:
        model.load_state_dict(torch.load(BEST_PATH, map_location=device))
    except FileNotFoundError:
        print("Warning: Best model not saved (ran too few epochs or early stopped). Using last epoch.")

    test_loss, test_acc, test_f1, test_labels, test_preds = evaluate(model, test_loader, device)

    print("\nFINAL LONGFORMER TEST RESULTS")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.3f}")
    print(f"Test F1 Score: {test_f1:.3f}")
    print("\nClassification report (Test):")
    print(classification_report(test_labels, test_preds, target_names=['Control', 'Dementia']))
    
    cm = confusion_matrix(test_labels, test_preds)
    print("\nConfusion Matrix:")
    print(f"                  Predicted")
    print(f"              Control  Dementia")
    print(f"Actual Control    {cm[0][0]:3d}      {cm[0][1]:3d}")
    print(f"       Dementia   {cm[1][0]:3d}      {cm[1][1]:3d}")
    
    tokenizer.save_pretrained(OUTPUT_DIR)

    # --- Final History Save ---
    with open(HISTORY_PATH, 'w') as f:
        # Convert NumPy float types to standard Python floats for JSON serialization
        serializable_history = {k: [float(x) for x in v] for k, v in history.items()}
        json.dump(serializable_history, f, indent=4)
    print(f"\nTraining history successfully saved to {HISTORY_PATH}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # FIX: Increase epochs and patience to improve stability and performance
    p.add_argument("--epochs", type=int, default=20) 
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--acc_steps", "--accumulation_steps", dest='acc_steps', type=int, default=4)
    p.add_argument("--lr", "--learning_rate", dest='lr', type=float, default=5e-5) # Lower LR for Transformer baseline
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=7, help="Early stopping patience based on validation F1 score.") # Increased patience
    p.add_argument("--mamba_model", type=str, default='state-spaces/mamba-130m', help="Placeholder: Not used, kept for reference")
    args = p.parse_args()
    main(args)