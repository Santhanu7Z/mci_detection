# ===============================
# train.py — PERFORMANCE-OPTIMIZED (TARGET ≥70% F1)
# Partial Unfreezing + Attention Pooling + Stable Training
# ===============================

import os
import json
import argparse
import warnings
from collections import Counter

warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

# ------------------------------------------------------------
# MAMBA MODEL DEFINITIONS (STABLE)
# ------------------------------------------------------------

class InferredMambaConfig:
    def __init__(self, vocab_size, hidden_size, num_hidden_layers,
                 d_inner, dt_rank, state_size, conv_kernel, num_labels=2):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.d_inner = d_inner
        self.dt_rank = dt_rank
        self.state_size = state_size
        self.conv_kernel = conv_kernel
        self.num_labels = num_labels


class MambaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_proj = nn.Linear(config.hidden_size, config.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            config.d_inner,
            config.d_inner,
            kernel_size=config.conv_kernel,
            groups=config.d_inner,
            padding=config.conv_kernel - 1,
        )
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + config.state_size * 2, bias=False)
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner)
        self.A_log = nn.Parameter(torch.zeros(config.d_inner, config.state_size))
        self.out_proj = nn.Linear(config.d_inner, config.hidden_size, bias=False)

    def forward(self, x):
        b, l, _ = x.shape
        xr = self.in_proj(x)
        x, res = xr.chunk(2, dim=-1)
        x = self.conv1d(x.transpose(1, 2))[:, :, :l].transpose(1, 2)
        x = torch.nn.functional.silu(x)
        x_dbl = self.x_proj(x)
        dt, B, C = x_dbl.split([
            self.dt_proj.in_features,
            self.A_log.shape[1],
            self.A_log.shape[1]], dim=-1)
        delta = torch.nn.functional.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log.float())
        with torch.no_grad():
            y = self.scan(x, delta, A, B, C)
        return self.out_proj(y * torch.nn.functional.silu(res))

    def scan(self, u, delta, A, B, C):
        b, l, d = u.shape
        n = A.shape[1]
        h = torch.zeros(b, d, n, device=u.device, dtype=u.dtype)
        ys = []
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaBu = (delta * u).unsqueeze(-1) * B.unsqueeze(2)
        for i in range(l):
            h = deltaA[:, i] * h + deltaBu[:, i]
            ys.append((h * C[:, i].unsqueeze(1)).sum(dim=-1))
        return torch.stack(ys, dim=1)


class MambaResidualBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = nn.LayerNorm(config.hidden_size)
        self.mixer = MambaBlock(config)

    def forward(self, x):
        return x + self.mixer(self.norm(x))


class MambaBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            MambaResidualBlock(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):
        w = torch.softmax(self.attn(x), dim=1)
        return (x * w).sum(dim=1)


class MambaClassifier(nn.Module):
    def __init__(self, config, class_weights):
        super().__init__()
        self.backbone = MambaBackbone(config)
        self.pool = AttentionPooling(config.hidden_size)
        self.fc = nn.Linear(config.hidden_size, config.hidden_size // 4)
        self.classifier = nn.Linear(config.hidden_size // 4, config.num_labels)
        self.dropout = nn.Dropout(0.3)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    def forward(self, input_ids, labels=None):
        x = self.backbone(input_ids)
        pooled = self.pool(x)
        x = self.dropout(torch.nn.functional.gelu(self.fc(pooled)))
        logits = self.classifier(x)
        loss = self.criterion(logits, labels) if labels is not None else None
        return loss, logits


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# ------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------

def infer_config(sd):
    emb = sd['backbone.embedding.weight']
    vocab, hidden = emb.shape
    layers = 0
    while f'backbone.layers.{layers}.mixer.in_proj.weight' in sd:
        layers += 1
    in_proj = sd['backbone.layers.0.mixer.in_proj.weight']
    d_inner = in_proj.shape[0] // 2
    dt_rank = sd['backbone.layers.0.mixer.dt_proj.weight'].shape[1]
    total = sd['backbone.layers.0.mixer.x_proj.weight'].shape[0]
    state = (total - dt_rank) // 2
    kernel = sd['backbone.layers.0.mixer.conv1d.weight'].shape[-1]
    return InferredMambaConfig(vocab, hidden, layers, d_inner, dt_rank, state, kernel)


def evaluate(model, loader, device):
    model.eval()
    losses, preds, labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch['input_ids'].to(device)
            y = batch['labels'].to(device)
            with autocast():
                loss, logits = model(ids, y)
            losses.append(loss.item())
            preds.extend(logits.argmax(1).cpu().numpy())
            labels.extend(y.cpu().numpy())
    return float(np.mean(losses)), accuracy_score(labels, preds), f1_score(labels, preds, average='weighted'), labels, preds


# ------------------------------------------------------------
# TRAINING PIPELINE
# ------------------------------------------------------------

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*70}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*70}")
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Accumulation steps: {args.acc_steps}")
    print(f"Learning rate: {args.lr}")
    print(f"Max sequence length: 384")
    print(f"{'='*70}\n")

    # FIXED: Use correct directory name to match predict.py
    OUTPUT_DIR = 'trained_model_mamba_pretrained'
    BEST_PATH = os.path.join(OUTPUT_DIR, 'best_pytorch_model.bin')
    HISTORY_PATH = os.path.join(OUTPUT_DIR, 'training_history.json')
    CONFIG_PATH = os.path.join(OUTPUT_DIR, 'config.json')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -------- Load data --------
    print("Loading data and transcripts...")
    df = pd.read_csv('processed_data/metadata.csv')
    with open('processed_data/transcripts_cache.json') as f:
        cache = json.load(f)
    df['transcription'] = df['audio_path'].map(cache['transcripts'])
    df = df[df['transcription'].str.strip() != '']
    df['label_id'] = (df['label'] == 'Dementia').astype(int)

    print(f"\nClass distribution:")
    print(df['label'].value_counts())

    train_df, test_df = train_test_split(df, test_size=0.15, stratify=df['label_id'], random_state=args.seed)
    train_df, val_df = train_test_split(train_df, test_size=0.10, stratify=train_df['label_id'], random_state=args.seed)

    print(f"\nData splits:")
    print(f"  Train: {len(train_df)}")
    print(f"  Val:   {len(val_df)}")
    print(f"  Test:  {len(test_df)}\n")

    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    tokenizer.pad_token = tokenizer.eos_token

    MAX_LEN = 384
    encode = lambda x: tokenizer(x.tolist(), truncation=True, padding='max_length', max_length=MAX_LEN)

    print("Tokenizing datasets...")
    train_ds = SimpleDataset(encode(train_df.transcription), train_df.label_id.tolist())
    val_ds = SimpleDataset(encode(val_df.transcription), val_df.label_id.tolist())
    test_ds = SimpleDataset(encode(test_df.transcription), test_df.label_id.tolist())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2)

    class_weights = torch.tensor(
        compute_class_weight('balanced', classes=np.array([0, 1]), y=train_df.label_id.values),
        device=device, dtype=torch.float
    )
    print(f"Class weights: Control={class_weights[0]:.4f}, Dementia={class_weights[1]:.4f}\n")

    # -------- Model --------
    print(f"Loading pretrained Mamba weights from {args.mamba_model}...")
    sd = torch.load(hf_hub_download(args.mamba_model, 'pytorch_model.bin'), map_location='cpu')
    config = infer_config(sd)

    # Save config for predict.py
    config_dict = {
        'vocab_size': config.vocab_size,
        'hidden_size': config.hidden_size,
        'num_hidden_layers': config.num_hidden_layers,
        'd_inner': config.d_inner,
        'dt_rank': config.dt_rank,
        'state_size': config.state_size,
        'conv_kernel': config.conv_kernel,
        'num_labels': config.num_labels,
        'max_length': MAX_LEN,
        'model_type': 'MambaClassifier'
    }
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config_dict, f, indent=2)

    model = MambaClassifier(config, class_weights).to(device)
    model.load_state_dict({k: v for k, v in sd.items() if k.startswith('backbone.')}, strict=False)

    # Freeze all, unfreeze last 2 layers
    print("Freezing backbone (except last 2 layers)...")
    for p in model.backbone.parameters():
        p.requires_grad = False
    for layer in model.backbone.layers[-2:]:
        for p in layer.parameters():
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)\n")

    optimizer = torch.optim.AdamW([
        {'params': model.backbone.layers[-2:].parameters(), 'lr': 1e-4},
        {'params': model.fc.parameters(), 'lr': args.lr},
        {'params': model.classifier.parameters(), 'lr': args.lr},
    ], weight_decay=0.01)

    scaler = GradScaler(enabled=(device == 'cuda'))

    # History tracking
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'train_f1': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': []
    }

    best_val_f1 = -1.0
    patience = 5
    patience_counter = 0

    # -------- Training --------
    print("Starting training...\n")
    
    for epoch in range(args.epochs):
        model.train()
        train_preds, train_labels, train_losses = [], [], []
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for i, batch in enumerate(pbar):
            ids = batch['input_ids'].to(device)
            y = batch['labels'].to(device)

            with autocast():
                loss, logits = model(ids, y)
                loss = loss / args.acc_steps

            scaler.scale(loss).backward()

            if (i + 1) % args.acc_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_losses.append(loss.item() * args.acc_steps)
            train_preds.extend(logits.argmax(1).detach().cpu().numpy())
            train_labels.extend(y.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item() * args.acc_steps:.4f}'})

        train_loss = float(np.mean(train_losses))
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='weighted')

        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, device)

        # Save to history
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        history['val_f1'].append(val_f1)

        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  TRAIN - Loss: {train_loss:.4f} | Acc: {train_acc:.3f} | F1: {train_f1:.3f}")
        print(f"  VAL   - Loss: {val_loss:.4f} | Acc: {val_acc:.3f} | F1: {val_f1:.3f}")

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), BEST_PATH)
            print(f"  ✓ New best model saved (Val F1: {val_f1:.3f})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{patience}")
            
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
            break
        
        print()

    # Save training history
    with open(HISTORY_PATH, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {HISTORY_PATH}")

    # -------- Final Test --------
    print(f"\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(BEST_PATH, map_location=device))
    test_loss, test_acc, test_f1, test_labels, test_preds = evaluate(model, test_loader, device)

    print(f"\n{'='*70}")
    print(f"FINAL TEST RESULTS")
    print(f"{'='*70}")
    print(f"Loss: {test_loss:.4f} | Accuracy: {test_acc:.3f} | F1 Score: {test_f1:.3f}")
    print(f"{'='*70}\n")

    print("Classification Report (Test Set):")
    print(classification_report(test_labels, test_preds, target_names=['Control', 'Dementia']))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(test_labels, test_preds)
    print(f"              Predicted")
    print(f"            Control  Dementia")
    print(f"Control       {cm[0][0]:3d}      {cm[0][1]:3d}")
    print(f"Dementia      {cm[1][0]:3d}      {cm[1][1]:3d}\n")

    # Save final results
    results = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'test_f1': float(test_f1),
        'best_val_f1': float(best_val_f1),
        'confusion_matrix': cm.tolist()
    }
    
    with open(os.path.join(OUTPUT_DIR, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"All files saved to {OUTPUT_DIR}/")
    print(f"  ✓ best_pytorch_model.bin")
    print(f"  ✓ training_history.json")
    print(f"  ✓ config.json")
    print(f"  ✓ test_results.json")
    print(f"  ✓ tokenizer files\n")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--acc_steps', type=int, default=4)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--mamba_model', type=str, default='state-spaces/mamba-130m')
    args = p.parse_args()
    main(args)