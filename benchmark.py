#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Mamba-Fusion Benchmark v3.4 - JCSSE PRODUCTION ENGINE
- Cohort: Unified Cleaned Pitt, ADReSS, and TAUKADIAL (N=802).
- Rigor: Stratified Group 5-Fold CV (Participant-Level Separation).
- Modalities: Linguistic, Acoustic, Gated Fusion, Attention Fusion.
- Fix: Robust audio_path-based feature alignment to handle multiple recordings.
"""

import os
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

# --- Environmental Guardrails ---
os.environ["MAMBA_NO_TRITON"] = "1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ============================================================
# DATASET ARCHITECTURE
# ============================================================

class UnifiedMCIFusionDataset(Dataset):
    def __init__(self, df, tokenizer, acoustic_features, max_len=512):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.acoustic_features = acoustic_features
        self.max_len = max_len

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        text = str(self.df.loc[idx, "text"])
        enc = self.tokenizer(
            text, truncation=True, padding="max_length", 
            max_length=self.max_len, return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "acoustic_features": torch.tensor(self.acoustic_features[idx], dtype=torch.float32),
            "labels": torch.tensor(self.df.loc[idx, "label_id"], dtype=torch.long),
            "participant_id": self.df.loc[idx, "participant_id"],
            "audio_path": self.df.loc[idx, "audio_path"],
            "dataset": self.df.loc[idx, "dataset"]
        }

# ============================================================
# FUSION MODEL (Support for 4 Modalities)
# ============================================================

class MambaFusionEngine(nn.Module):
    def __init__(self, mamba_backbone, acoustic_dim, mode="fusion_attn", num_labels=2):
        super().__init__()
        self.mode = mode
        self.mamba = mamba_backbone.backbone 
        text_dim = mamba_backbone.config.d_model # 768
        self.fusion_dim = 256
        
        # Projections
        self.text_proj = nn.Linear(text_dim, self.fusion_dim)
        self.audio_proj = nn.Sequential(
            nn.Linear(acoustic_dim, 128),
            nn.GELU(),
            nn.Linear(128, self.fusion_dim)
        )
        
        # 1. Gated Fusion
        self.gate = nn.Sequential(
            nn.Linear(self.fusion_dim * 2, self.fusion_dim),
            nn.Sigmoid()
        )
        
        # 2. Attention Fusion
        self.attention = nn.MultiheadAttention(self.fusion_dim, num_heads=4, batch_first=True)
        
        clf_input_dim = self.fusion_dim * 2 if mode == "fusion_attn" else self.fusion_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(clf_input_dim, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_labels)
        )

    def forward(self, input_ids, acoustic_features):
        text_out = self.mamba(input_ids).mean(dim=1) 
        text_emb = self.text_proj(text_out)
        audio_emb = self.audio_proj(acoustic_features)
        
        attn_weights = None
        
        if self.mode == "linguistic":
            fused = text_emb
        elif self.mode == "acoustic":
            fused = audio_emb
        elif self.mode == "fusion_gated":
            g = self.gate(torch.cat([text_emb, audio_emb], dim=-1))
            fused = g * text_emb + (1 - g) * audio_emb
        else: # fusion_attn
            combined = torch.stack([text_emb, audio_emb], dim=1)
            attn_out, attn_weights = self.attention(combined, combined, combined)
            fused = attn_out.view(attn_out.size(0), -1) 
            
        logits = self.classifier(fused)
        return logits, attn_weights

# ============================================================
# TRAIN / EVAL LOGIC
# ============================================================

def train_fold(fold_idx, train_idx, val_idx, df, acoustic_data, acoustic_dim, mode):
    set_seed(SEED + fold_idx)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token = tokenizer.eos_token
    
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
    
    # Stratified Scaling (Strictly on Groups)
    scaler = StandardScaler()
    train_acc = scaler.fit_transform(acoustic_data[train_idx])
    val_acc = scaler.transform(acoustic_data[val_idx])
    
    train_ds = UnifiedMCIFusionDataset(train_df, tokenizer, train_acc)
    val_ds = UnifiedMCIFusionDataset(val_df, tokenizer, val_acc)
    
    y_train = train_df["label_id"].values
    class_counts = np.bincount(y_train)
    weights = 1. / class_counts
    samples_weight = torch.from_numpy(np.array([weights[t] for t in y_train])).double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    
    train_loader = DataLoader(train_ds, batch_size=8, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=16)
    
    mamba_base = MambaLMHeadModel.from_pretrained("state-spaces/mamba-130m")
    model = MambaFusionEngine(mamba_base, acoustic_dim, mode=mode).to(DEVICE)
    
    optimizer = torch.optim.AdamW([
        {'params': model.mamba.parameters(), 'lr': 2e-5},
        {'params': [p for n, p in model.named_parameters() if 'mamba' not in n], 'lr': 1e-4}
    ], weight_decay=0.01)
    
    criterion = nn.CrossEntropyLoss()
    
    best_f1 = 0
    best_metrics = {}
    fold_xai = []

    for epoch in range(12):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            logits, _ = model(batch["input_ids"].to(DEVICE), batch["acoustic_features"].to(DEVICE))
            loss = criterion(logits, batch["labels"].to(DEVICE))
            loss.backward()
            optimizer.step()
            
        model.eval()
        all_preds, all_labels, all_datasets = [], [], []
        temp_xai = []
        with torch.no_grad():
            for batch in val_loader:
                logits, attn_weights = model(batch["input_ids"].to(DEVICE), batch["acoustic_features"].to(DEVICE))
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch["labels"].numpy())
                all_datasets.extend(batch["dataset"])
                
                if attn_weights is not None:
                    for i, p_id in enumerate(batch["participant_id"]):
                        temp_xai.append({
                            "id": p_id, "mode": mode, "audio_path": batch["audio_path"][i],
                            "text_weight": float(attn_weights[i, 0, 0]),
                            "audio_weight": float(attn_weights[i, 0, 1]),
                            "correct": bool(preds[i] == batch["labels"][i])
                        })
        
        prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
        acc = accuracy_score(all_labels, all_preds)
        
        if f1 > best_f1: 
            best_f1 = f1
            ds_metrics = {}
            for ds in ["pitt", "adress", "taukadial"]:
                mask = [d == ds for d in all_datasets]
                if any(mask):
                    ds_metrics[ds] = f1_score(np.array(all_labels)[mask], np.array(all_preds)[mask], average='weighted')
            
            best_metrics = {"f1": f1, "acc": acc, "ds_breakdown": ds_metrics}
            fold_xai = temp_xai

    return best_metrics, fold_xai

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("\n--- Listening Between the Lines: Unified Fusion Benchmark v3.4 ---")
    
    meta_path = "processed_data/master_metadata_cleaned.csv"
    feat_path = "processed_data/master_acoustic_features.csv"
    cache_path = "processed_data/cleaned_transcripts.json"
    
    if not all(os.path.exists(p) for p in [meta_path, feat_path, cache_path]):
        print("❌ Error: Assets missing.")
        return

    meta_df = pd.read_csv(meta_path)
    feat_df = pd.read_csv(feat_path)
    with open(cache_path, 'r') as f: transcripts = json.load(f)['transcripts']
    
    meta_df['text'] = meta_df['audio_path'].map(transcripts)
    df = meta_df.dropna(subset=['text']).reset_index(drop=True)
    df['label_id'] = df['label'].map({"Control": 0, "Dementia": 1})
    
    # SCIENTIFIC FIX: Robust alignment using audio_path as the primary key
    # This prevents misalignment if a participant has multiple recordings
    print("Aligning acoustic features via audio_path...")
    
    # Filter only relevant columns from features to prevent naming collisions
    ignore_meta = ['participant_id', 'label', 'dataset', 'split', 'age', 'gender', 'mmse']
    feat_cols = [c for c in feat_df.columns if c not in ignore_meta and c != 'audio_path']
    
    # We attempt to merge on audio_path if available in feat_df, fallback to participant_id
    if 'audio_path' in feat_df.columns:
        df = pd.merge(df, feat_df[['audio_path'] + feat_cols], on='audio_path', how='inner')
    else:
        # Warning: Using participant_id merge on longitudinal data is risky
        print("⚠️ Warning: audio_path not found in features. Falling back to participant_id.")
        df = pd.merge(df, feat_df[['participant_id'] + feat_cols], on='participant_id', how='inner')
    
    acoustic_data = df[feat_cols].values
    
    print(f"Cleaned Cohort: {len(df)} subjects | Participant IDs: {df['participant_id'].nunique()}")

    # ABLATION SUITE
    modes = ["linguistic", "acoustic", "fusion_gated", "fusion_attn"]
    final_report_data = []
    master_xai = []

    for mode in modes:
        print(f"\n🚀 Evaluating Modality: {mode.upper()}")
        # Participant-Level Seperation
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(sgkf.split(df, df['label_id'], groups=df['participant_id'])):
            metrics, xai_data = train_fold(fold, train_idx, val_idx, df, acoustic_data, len(feat_cols), mode)
            fold_scores.append(metrics)
            master_xai.extend(xai_data)
            print(f"   Fold {fold+1} F1: {metrics['f1']:.4f}")
            
        avg_f1 = np.mean([m['f1'] for m in fold_scores])
        ds_summary = {ds: np.mean([m['ds_breakdown'].get(ds, 0) for m in fold_scores]) for ds in ["pitt", "adress", "taukadial"]}

        final_report_data.append({
            "Modality": mode, "F1_Mean": avg_f1,
            "Pitt_F1": ds_summary["pitt"], "ADReSS_F1": ds_summary["adress"], "TAUK_F1": ds_summary["taukadial"]
        })

    # FINAL CONSOLIDATED REPORT
    report_df = pd.DataFrame(final_report_data)
    print("\n" + "="*85)
    print("                      FINAL ABLATION PERFORMANCE MATRIX (GROUP CV)")
    print("="*85)
    print(report_df.to_string(index=False))
    print("="*85)
    
    if master_xai:
        pd.DataFrame(master_xai).to_csv("modality_attribution_results.csv", index=False)
        print("\n✓ Robust XAI attribution weights saved successfully.")

if __name__ == "__main__":
    main()