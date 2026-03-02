import os
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import copy

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from scipy.stats import ttest_rel, wilcoxon

from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

# --- Configuration ---
os.environ["MAMBA_NO_TRITON"] = "1"
DATA_DIR = "processed_data"
NUM_FOLDS = 5
SEED = 42

# ============================================================
# DATASET & GATED FUSION ARCHITECTURE
# ============================================================

class MCIFusionDataset(Dataset):
    def __init__(self, df, tokenizer, acoustic_features, max_len=512):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.acoustic_features = acoustic_features

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        enc = self.tokenizer(str(self.df.loc[idx, "transcription"]), truncation=True, 
                             padding="max_length", max_length=self.max_len, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "acoustic_features": torch.tensor(self.acoustic_features[idx], dtype=torch.float32),
            "labels": torch.tensor(self.df.loc[idx, "label_id"], dtype=torch.long)
        }

class GatedFusion(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, 1), nn.Sigmoid())
    def forward(self, text_emb, audio_emb):
        combined = torch.cat([text_emb, audio_emb], dim=-1)
        audio_weight = self.gate(combined)
        return text_emb + (audio_weight * audio_emb), audio_weight

class MambaGatedFusion(nn.Module):
    def __init__(self, mamba_model, acoustic_dim, num_labels, mode="fusion"):
        super().__init__()
        self.mode = mode
        self.mamba = mamba_model.backbone 
        
        text_dim, fusion_dim = mamba_model.config.d_model, 256
        self.text_proj = nn.Sequential(nn.Linear(text_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU())
        self.audio_proj = nn.Sequential(nn.Linear(acoustic_dim, 128), nn.GELU(), nn.Linear(128, fusion_dim), 
                                        nn.LayerNorm(fusion_dim), nn.GELU())
        self.fusion_layer = GatedFusion(dim=fusion_dim)
        self.classifier = nn.Sequential(nn.Linear(fusion_dim, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, num_labels))
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, input_ids, acoustic_features, labels=None):
        with torch.no_grad():
            text_out = self.mamba(input_ids).mean(dim=1)
        
        text_emb = self.text_proj(text_out)
        
        if self.mode == "text_only":
            fused_features = text_emb
        else:
            audio_emb = self.audio_proj(acoustic_features)
            fused_features, _ = self.fusion_layer(text_emb, audio_emb)

        logits = self.classifier(fused_features)
        loss = self.criterion(logits, labels) if labels is not None else None
        return loss, logits

# ============================================================
# TRAIN / EVAL LOOP
# ============================================================

def run_fold(fold_idx, train_idx, val_idx, df, mode, acoustic_data, feat_cols, device, base_mamba, tokenizer):
    # 1. Reset Seeds for Fold Independency
    random.seed(SEED + fold_idx)
    np.random.seed(SEED + fold_idx)
    torch.manual_seed(SEED + fold_idx)
    torch.cuda.manual_seed_all(SEED + fold_idx)

    # 2. Slice Data
    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    
    scaler = StandardScaler()
    train_acoustic = scaler.fit_transform(acoustic_data[train_idx])
    val_acoustic = scaler.transform(acoustic_data[val_idx])

    train_ds = MCIFusionDataset(train_df, tokenizer, train_acoustic)
    val_ds = MCIFusionDataset(val_df, tokenizer, val_acoustic)
    
    y_train = train_df["label_id"].values
    weights = compute_class_weight("balanced", classes=np.array([0,1]), y=y_train)
    sampler = WeightedRandomSampler([weights[l] for l in y_train], len(y_train))
    
    train_loader = DataLoader(train_ds, batch_size=8, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=16)

    model = MambaGatedFusion(base_mamba, len(feat_cols), 2, mode=mode).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    
    best_f1 = 0
    best_model_state = None
    
    for epoch in range(15):
        model.train()
        for batch in train_loader:
            ids, acc, labels = batch["input_ids"].to(device), batch["acoustic_features"].to(device), batch["labels"].to(device)
            optimizer.zero_grad()
            loss, _ = model(ids, acc, labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                ids, acc, labels = batch["input_ids"].to(device), batch["acoustic_features"].to(device), batch["labels"].to(device)
                _, logits = model(ids, acc)
                all_preds.extend(torch.argmax(logits, 1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        f1 = f1_score(all_labels, all_preds, average="weighted")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_state = copy.deepcopy(model.state_dict())

    if best_model_state:
        model.load_state_dict(best_model_state)
    
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            ids, acc, labels = batch["input_ids"].to(device), batch["acoustic_features"].to(device), batch["labels"].to(device)
            _, logits = model(ids, acc)
            all_preds.extend(torch.argmax(logits, 1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return f1_score(all_labels, all_preds, average="weighted")

# ============================================================
# MAIN BENCHMARK
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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

    print("Loading Mamba Backbone once...")
    base_mamba = MambaLMHeadModel.from_pretrained("state-spaces/mamba-130m")
    for param in base_mamba.parameters():
        param.requires_grad = False

    # Efficiency: Load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token = tokenizer.eos_token

    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    results = {"text_only": [], "fusion": []}

    for mode in ["text_only", "fusion"]:
        print(f"\n--- Benchmarking Mode: {mode.upper()} ---")
        for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["label_id"])):
            print(f"Fold {fold+1}/{NUM_FOLDS}...")
            f1 = run_fold(fold, train_idx, val_idx, df, mode, acoustic_data, feat_cols, device, base_mamba, tokenizer)
            results[mode].append(f1)
            print(f"Fold {fold+1} F1: {f1:.4f}")

    print("\n" + "="*50)
    print("FINAL 5-FOLD CROSS-VALIDATION RESULTS")
    print("="*50)
    
    summary_stats = {}
    for mode, scores in results.items():
        mean_val = np.mean(scores)
        std_val = np.std(scores)
        # 95% Confidence Interval Calculation
        ci_95 = 1.96 * (std_val / np.sqrt(NUM_FOLDS))
        summary_stats[mode] = mean_val
        print(f"{mode.upper()}: Mean F1 = {mean_val:.4f} (±{std_val:.4f}) | 95% CI: [{mean_val-ci_95:.4f}, {mean_val+ci_95:.4f}]")
    
    # Statistical Significance Calculations
    diffs = np.array(results["fusion"]) - np.array(results["text_only"])
    improvement = np.mean(diffs)
    
    # Cohen's d (Effect Size)
    cohens_d = np.mean(diffs) / np.std(diffs) if np.std(diffs) != 0 else 0
    
    # Paired T-test & Wilcoxon Signed-Rank
    t_stat, p_val_t = ttest_rel(results["fusion"], results["text_only"])
    _, p_val_w = wilcoxon(results["fusion"], results["text_only"])

    print("\n--- Comparative Analysis ---")
    print(f"Net Mean Improvement: {improvement:+.4f}")
    print(f"Cohen's d (Effect Size): {cohens_d:.4f}")
    print(f"Paired T-test p-value: {p_val_t:.4f}")
    print(f"Wilcoxon signed-rank p-value: {p_val_w:.4f}")
    
    if p_val_t < 0.05:
        print("\nResult: STATISTICAL SIGNIFICANCE ACHIEVED (p < 0.05)")
    else:
        print("\nResult: NO STATISTICAL SIGNIFICANCE (Benefit may be noise)")
    print("="*50)

if __name__ == "__main__":
    main()
