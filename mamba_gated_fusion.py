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

# Environmental Guardrail
os.environ["MAMBA_NO_TRITON"] = "1"

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
# ARCHITECTURE: Gated Multimodal Fusion (Statistically Robust)
# ============================================================

class GatedFusion(nn.Module):
    """
    Replaces Multi-Head Attention with a Gating Mechanism.
    Learns exactly how much to trust the Audio vs the Text.
    """
    def __init__(self, dim=256):
        super().__init__()
        # The gate learns a value between 0 and 1 for the audio stream
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

    def forward(self, text_emb, audio_emb):
        # Calculate how much to trust audio based on both modalities
        combined = torch.cat([text_emb, audio_emb], dim=-1)
        audio_weight = self.gate(combined) # [B, 1]
        
        # Apply the gate to audio, keep text as the primary anchor
        fused = text_emb + (audio_weight * audio_emb)
        return fused, audio_weight

class MambaGatedFusion(nn.Module):
    def __init__(self, mamba_model, acoustic_dim, num_labels, freeze_backbone=True):
        super().__init__()
        self.mamba = mamba_model.backbone
        
        # 100% Freeze Backbone to prevent Representation Drift (N=544)
        if freeze_backbone:
            for param in self.mamba.parameters():
                param.requires_grad = False
        
        text_dim = mamba_model.config.d_model # 768
        fusion_dim = 256 

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU()
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(acoustic_dim, 128),
            nn.GELU(),
            nn.Linear(128, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU()
        )

        self.fusion_layer = GatedFusion(dim=fusion_dim)

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_labels)
        )

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, input_ids, acoustic_features, labels=None):
        # 1. Feature Extraction (No-grad for efficiency)
        with torch.no_grad():
            text_out = self.mamba(input_ids).mean(dim=1) 
        
        text_emb = self.text_proj(text_out)
        audio_emb = self.audio_proj(acoustic_features)

        # 2. Gated Fusion (Acoustic gate)
        fused_features, gate_weights = self.fusion_layer(text_emb, audio_emb)

        # 3. Final Classification
        logits = self.classifier(fused_features)

        loss = self.criterion(logits, labels) if labels is not None else None
        return loss, logits, gate_weights

# ============================================================
# EVALUATION & MAIN
# ============================================================

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels, losses = [], [], []
    for batch in loader:
        ids, acoustics, y = batch["input_ids"].to(device), batch["acoustic_features"].to(device), batch["labels"].to(device)
        with autocast(enabled=(device.type == "cuda")):
            loss, logits, _ = model(ids, acoustics, y)
        losses.append(loss.item())
        preds.extend(torch.argmax(logits, 1).cpu().numpy())
        labels.extend(y.cpu().numpy())
    return np.mean(losses), accuracy_score(labels, preds), f1_score(labels, preds, average="weighted")

def main(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Data
    meta_df = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))
    acoustic_df = pd.read_csv(os.path.join(DATA_DIR, "egemaps_features.csv"))
    with open(os.path.join(DATA_DIR, "transcripts_cache.json")) as f:
        cache = json.load(f)
    meta_df["transcription"] = meta_df["audio_path"].map(cache["transcripts"])
    meta_df = meta_df.dropna(subset=["transcription"])
    acoustic_df['participant_id'] = acoustic_df['file_id'].str.replace('_participant', '', regex=False)
    df = pd.merge(meta_df, acoustic_df, on="participant_id", how="inner")
    df["label_id"] = df["label"].map({"Control": 0, "Dementia": 1})
    feat_cols = [c for c in acoustic_df.columns if c not in ['participant_id', 'file_id', 'duration']]
    acoustic_data = df[feat_cols].values

    # Splits & Scaling
    indices = np.arange(len(df))
    tr_idx, te_idx = train_test_split(indices, test_size=0.15, stratify=df["label_id"], random_state=args.seed)
    tr_idx, va_idx = train_test_split(tr_idx, test_size=0.10, stratify=df.iloc[tr_idx]["label_id"], random_state=args.seed)

    scaler = StandardScaler()
    acoustic_train = scaler.fit_transform(acoustic_data[tr_idx])
    acoustic_val = scaler.transform(acoustic_data[va_idx])
    acoustic_test = scaler.transform(acoustic_data[te_idx])

    # Dataloaders
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token = tokenizer.eos_token
    train_ds = MCIFusionDataset(df.iloc[tr_idx], tokenizer, acoustic_train)
    val_ds = MCIFusionDataset(df.iloc[va_idx], tokenizer, acoustic_val)
    test_ds = MCIFusionDataset(df.iloc[te_idx], tokenizer, acoustic_test)
    train_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=df.iloc[tr_idx]["label_id"].values)
    sampler = WeightedRandomSampler([train_weights[l] for l in df.iloc[tr_idx]["label_id"]], len(tr_idx))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2)

    # Model
    base_mamba = MambaLMHeadModel.from_pretrained(args.mamba_model)
    model = MambaGatedFusion(base_mamba, acoustic_dim=len(feat_cols), num_labels=2).to(device)

    # Optimizer (Only trainable head params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.02)
    
    # SWITCHED: Standard CosineAnnealing (No restarts) for stability on tiny data
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler_gpu = GradScaler(enabled=(device.type == "cuda"))
    best_f1 = 0
    patience_counter = 0

    print(f"\n🚀 Training Statistically Stable Gated Fusion on {device}...")

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        optimizer.zero_grad()
        for i, batch in enumerate(pbar):
            ids, acoustics, labels = batch["input_ids"].to(device), batch["acoustic_features"].to(device), batch["labels"].to(device)
            with autocast(enabled=(device.type == "cuda")):
                loss, _, _ = model(ids, acoustics, labels)
                loss = loss / args.acc_steps
            scaler_gpu.scale(loss).backward()
            if (i + 1) % args.acc_steps == 0:
                scaler_gpu.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler_gpu.step(optimizer)
                scaler_gpu.update()
                optimizer.zero_grad()
        
        scheduler.step()
        _, _, v_f1 = evaluate(model, val_loader, device)
        print(f" -> Val F1: {v_f1:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if v_f1 > best_f1:
            best_f1 = v_f1
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_attention_fusion.bin"))
            print(f" ✓ Saved Best Model (F1: {best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

    print("\n--- Final Test Evaluation ---")
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_attention_fusion.bin")))
    _, test_acc, test_f1 = evaluate(model, test_loader, device)
    print(f"Final Accuracy: {test_acc:.4f} | Final F1: {test_f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--acc_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4) # Balanced LR
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mamba_model", type=str, default="state-spaces/mamba-130m")
    main(parser.parse_args())
