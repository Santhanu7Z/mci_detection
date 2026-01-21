# ============================================================
# mamba_attention_fusion.py — MAMBA + SELF-ATTENTION FUSION
# Phase 2: Advanced Multimodal MCI Detection
# ============================================================

import os
import json
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler

from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

# --- Directories ---
DATA_DIR = "processed_data"
MODEL_DIR = "trained_mamba_attention_fusion"
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# DATASET
# ============================================================

class MCIFusionDataset(Dataset):
    def __init__(self, df, tokenizer, acoustic_features, max_len=512):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.acoustic_features = acoustic_features 

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Textual Input
        enc = self.tokenizer(
            str(self.df.loc[idx, "transcription"]),
            truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "acoustic_features": torch.tensor(self.acoustic_features[idx], dtype=torch.float32),
            "labels": torch.tensor(self.df.loc[idx, "label_id"], dtype=torch.long)
        }

# ============================================================
# ARCHITECTURE: Attention-Based Fusion
# ============================================================

class AttentionFusionBlock(nn.Module):
    """
    Learns to weigh the importance of Text vs Audio dynamically.
    Instead of simple concatenation, this uses Multi-Head Attention
    to find cross-modal correlations.
    """
    def __init__(self, embed_dim=256, num_heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1, batch_first=True)
        self.layernorm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, x):
        # x shape: [Batch, 2, embed_dim] -> (Token 0: Text, Token 1: Audio)
        attn_out, weights = self.mha(x, x, x)
        x = self.layernorm(x + attn_out)
        x = self.layernorm(x + self.ffn(x))
        return x, weights

class MambaAttentionFusion(nn.Module):
    def __init__(self, mamba_model, acoustic_dim, num_labels, class_weights):
        super().__init__()
        self.mamba = mamba_model.backbone
        text_dim = mamba_model.config.d_model # 768
        fusion_dim = 256 # Shared space dimension
        
        # Projections to shared space
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.audio_proj = nn.Sequential(
            nn.Linear(acoustic_dim, 128),
            nn.GELU(),
            nn.Linear(128, fusion_dim)
        )
        
        # Attention Fusion Module
        self.fusion_block = AttentionFusionBlock(embed_dim=fusion_dim)
        
        # Final Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim * 2, 128), # Concatenating the 2 attended tokens
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_labels)
        )
        
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    def forward(self, input_ids, acoustic_features, labels=None):
        # 1. Extract Features from Towers
        text_out = self.mamba(input_ids).mean(dim=1) # [B, 768]
        audio_out = self.audio_proj(acoustic_features) # [B, 256]
        text_out = self.text_proj(text_out) # [B, 256]
        
        # 2. Prepare Sequence for Attention: [Batch, Tokens=2, Dim=256]
        combined = torch.stack([text_out, audio_out], dim=1)
        
        # 3. Apply Self-Attention Fusion
        fused_seq, attn_weights = self.fusion_block(combined)
        
        # 4. Flatten Attended Tokens and Classify
        fused_flat = fused_seq.view(fused_seq.size(0), -1) # [B, 512]
        logits = self.classifier(fused_flat)
        
        loss = self.criterion(logits, labels) if labels is not None else None
        return loss, logits, attn_weights

# ============================================================
# EVALUATION & TRAINING
# ============================================================

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels, losses = [], [], []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        acoustics = batch["acoustic_features"].to(device)
        y = batch["labels"].to(device)
        
        with autocast(enabled=(device.type == "cuda")):
            loss, logits, _ = model(ids, acoustics, y)
            
        losses.append(loss.item())
        preds.extend(torch.argmax(logits, 1).cpu().numpy())
        labels.extend(y.cpu().numpy())
    return np.mean(losses), accuracy_score(labels, preds), f1_score(labels, preds, average="weighted")

def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on Device: {device}")

    # 1. Load Data
    meta_df = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))
    acoustic_df = pd.read_csv(os.path.join(DATA_DIR, "acoustic_features.csv"))
    
    with open(os.path.join(DATA_DIR, "transcripts_cache.json")) as f:
        cache = json.load(f)
    
    meta_df["transcription"] = meta_df["audio_path"].map(cache["transcripts"])
    meta_df = meta_df.dropna(subset=["transcription"])
    
    # Merge on participant_id
    acoustic_df['participant_id'] = acoustic_df['file_id'].str.replace('_participant', '', regex=False)
    df = pd.merge(meta_df, acoustic_df, on="participant_id", how="inner")
    df["label_id"] = df["label"].map({"Control": 0, "Dementia": 1})
    
    # Define feature set (Exclude IDs and Duration)
    feat_cols = [c for c in acoustic_df.columns if c not in ['participant_id', 'file_id', 'duration']]
    acoustic_data = df[feat_cols].values

    # 2. Splits & Scaling
    indices = np.arange(len(df))
    tr_idx, te_idx = train_test_split(indices, test_size=0.15, stratify=df["label_id"], random_state=args.seed)
    tr_idx, va_idx = train_test_split(tr_idx, test_size=0.10, stratify=df.iloc[tr_idx]["label_id"], random_state=args.seed)

    scaler = StandardScaler()
    acoustic_train = scaler.fit_transform(acoustic_data[tr_idx])
    acoustic_val = scaler.transform(acoustic_data[va_idx])
    acoustic_test = scaler.transform(acoustic_data[te_idx])

    # 3. Setup Dataloaders
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token = tokenizer.eos_token
    
    train_ds = MCIFusionDataset(df.iloc[tr_idx], tokenizer, acoustic_train, max_len=args.max_len)
    val_ds = MCIFusionDataset(df.iloc[va_idx], tokenizer, acoustic_val, max_len=args.max_len)
    test_ds = MCIFusionDataset(df.iloc[te_idx], tokenizer, acoustic_test, max_len=args.max_len)

    train_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=df.iloc[tr_idx]["label_id"].values)
    sampler = WeightedRandomSampler([train_weights[l] for l in df.iloc[tr_idx]["label_id"]], len(tr_idx))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2)

    # 4. Initialize model
    print(f"Loading Mamba Backbone: {args.mamba_model}")
    base_mamba = MambaLMHeadModel.from_pretrained(args.mamba_model)
    model = MambaAttentionFusion(base_mamba, acoustic_dim=len(feat_cols), num_labels=2, 
                                 class_weights=torch.FloatTensor(train_weights).to(device)).to(device)

    # Differential Learning Rates (Backbone trains 10x slower)
    optimizer = torch.optim.AdamW([
        {"params": model.mamba.parameters(), "lr": args.lr * 0.1},
        {"params": model.text_proj.parameters(), "lr": args.lr},
        {"params": model.audio_proj.parameters(), "lr": args.lr},
        {"params": model.fusion_block.parameters(), "lr": args.lr},
        {"params": model.classifier.parameters(), "lr": args.lr}
    ], weight_decay=0.02)
    
    scaler_gpu = GradScaler(enabled=(device.type == "cuda"))

    # 5. Training Loop
    best_f1 = 0
    print(f"Starting Multimodal Attention Fusion Training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            ids, acoustics, labels = batch["input_ids"].to(device), batch["acoustic_features"].to(device), batch["labels"].to(device)
            optimizer.zero_grad()
            
            with autocast(enabled=(device.type == "cuda")):
                loss, _, _ = model(ids, acoustics, labels)
                
            scaler_gpu.scale(loss).backward()
            scaler_gpu.step(optimizer)
            scaler_gpu.update()
            train_losses.append(loss.item())
        
        v_loss, v_acc, v_f1 = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1} | Val Acc: {v_acc:.4f} | Val F1: {v_f1:.4f}")
        
        if v_f1 > best_f1:
            best_f1 = v_f1
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_attention_fusion.bin"))
            print(f"✓ Saved Best Model (F1: {best_f1:.4f})")

    # 6. Final Test Set Results
    print("\nTraining Complete. Running Final Test...")
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_attention_fusion.bin")))
    _, test_acc, test_f1 = evaluate(model, test_loader, device)
    
    print("\n" + "="*40 + "\nFINAL ATTENTION FUSION RESULTS\n" + "="*40)
    print(f"Final Accuracy: {test_acc:.4f}")
    print(f"Final F1 Score: {test_f1:.4f}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--mamba_model", type=str, default="state-spaces/mamba-130m")
    main(parser.parse_args())