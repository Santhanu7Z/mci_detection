#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Mamba-Fusion Benchmark v5.2 - JCSSE FINAL PRODUCTION (ULTRA-STABLE)
- Fix: Robust Alignment (Join on participant_id if audio_path is missing in features).
- Fix: JSON Serialization (Casts numpy.float64 to Python float to prevent TypeError).
- Fix: Robust Checkpoint Resumption (Verifies all targets before skipping).
- Fix: STRICT Backbone Isolation (copy.deepcopy inside fold loop).
- Fix: Explicit error logging in statistical reporting (no silent skips).
- Feature: Cross-Corpus Generalization + Multi-Domain In-Distribution (Unified Pool).
"""

import os
import json
import random
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from scipy.stats import ttest_rel

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
# FUSION MODEL
# ============================================================

class MambaFusionEngine(nn.Module):
    def __init__(self, mamba_backbone, acoustic_dim, mode="fusion_attn", num_labels=2):
        super().__init__()
        self.mode = mode
        self.mamba = mamba_backbone.backbone 
        self.fusion_dim = 256
        
        # Projections to global representation space
        self.text_proj = nn.Linear(mamba_backbone.config.d_model, self.fusion_dim)
        self.audio_proj = nn.Sequential(
            nn.Linear(acoustic_dim, 128), nn.GELU(), nn.Linear(128, self.fusion_dim)
        )
        
        # Gating
        self.gate_layer = nn.Sequential(
            nn.Linear(self.fusion_dim * 2, self.fusion_dim),
            nn.GELU(),
            nn.Linear(self.fusion_dim, 1),
            nn.Sigmoid()
        )
        
        # Classifier dim scaling
        if mode == "fusion_attn": clf_input_dim = self.fusion_dim * 3
        else: clf_input_dim = self.fusion_dim

        self.attention = nn.MultiheadAttention(self.fusion_dim, num_heads=4, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(clf_input_dim, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, num_labels)
        )

    def forward(self, input_ids, acoustic_features):
        weights = None 
        text_raw = self.mamba(input_ids)
        text_out = text_raw.max(dim=1).values # Global representation extraction
        
        text_emb = self.text_proj(text_out)
        audio_emb = self.audio_proj(acoustic_features)
        
        if self.mode == "linguistic": 
            fused = text_emb
        elif self.mode == "acoustic": 
            fused = audio_emb
        elif self.mode == "interaction_only": 
            fused = text_emb * audio_emb
        elif self.mode == "fusion_gated":
            g = self.gate_layer(torch.cat([text_emb, audio_emb], dim=-1))
            fused = g * text_emb + (1 - g) * audio_emb
            weights = g.detach().cpu() 
        else: # fusion_attn
            combined = torch.stack([text_emb, audio_emb], dim=1)
            attn_out, attn_w = self.attention(combined, combined, combined)
            if attn_w.dim() == 4: attn_w = attn_w.mean(dim=1)
            weights = attn_w.detach().cpu()
            interaction = text_emb * audio_emb
            fused = torch.cat([attn_out.flatten(start_dim=1), interaction], dim=-1)
            
        return self.classifier(fused), weights

# ============================================================
# EXPERIMENT CORE
# ============================================================

def evaluate_on_dataset(model, df, acoustic_data, tokenizer):
    model.eval()
    ds = UnifiedMCIFusionDataset(df, tokenizer, acoustic_data)
    loader = DataLoader(ds, batch_size=16)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits, _ = model(batch["input_ids"].to(DEVICE), batch["acoustic_features"].to(DEVICE))
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(batch["labels"].numpy())
    
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, labels=[0, 1], zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return {
        "weighted_f1": weighted_f1, "macro_f1": macro_f1,
        "dementia_recall": rec[1], "dementia_precision": prec[1]
    }

def run_experiment(df, acoustic_data, feat_cols, mode, mamba_backbone, sources, tokenizer, output_file):
    datasets = df['dataset'].unique()
    num_expected_targets = len(datasets)
    
    # LOAD EXISTING DATA FOR RESUMPTION
    processed_combos = {} # (mode, source) -> count of targets found
    if os.path.exists(output_file):
        try:
            existing_df = pd.read_csv(output_file)
            for _, r in existing_df.iterrows():
                key = (str(r['mode']), str(r['source']))
                processed_combos[key] = processed_combos.get(key, 0) + 1
        except Exception as e:
            print(f"⚠️ Could not read checkpoint: {e}. Starting fresh where needed.")

    for source_name in tqdm(sources, desc=f"Evaluating {mode}"):
        # SKIP LOGIC: Verify all targets exist
        if processed_combos.get((mode, source_name), 0) >= num_expected_targets:
            print(f"⏩ Skipping {mode} - {source_name} (Complete)")
            continue

        is_pool = (source_name == "UNIFIED_POOL")
        if is_pool:
            source_df, source_acoustic = df.copy(), acoustic_data
        else:
            mask = df['dataset'] == source_name
            source_df, source_acoustic = df[mask].reset_index(drop=True), acoustic_data[mask]
        
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        fold_gen_metrics = {d: {k: [] for k in ["weighted_f1", "macro_f1", "dementia_recall"]} for d in datasets}

        for fold, (train_idx, val_idx) in enumerate(sgkf.split(source_df, source_df['label_id'], groups=source_df['participant_id'])):
            train_df, val_df = source_df.iloc[train_idx], source_df.iloc[val_idx]
            scaler = StandardScaler()
            train_acc = scaler.fit_transform(source_acoustic[train_idx])
            val_acc = scaler.transform(source_acoustic[val_idx])
            
            # STRICT BACKBONE ISOLATION
            # Ensuring no cross-fold contamination
            model = MambaFusionEngine(copy.deepcopy(mamba_backbone), len(feat_cols), mode=mode).to(DEVICE)
            best_composite_score, best_model_wts = -1.0, copy.deepcopy(model.state_dict())
            
            optimizer = torch.optim.AdamW([
                {'params': model.mamba.parameters(), 'lr': 2e-5},
                {'params': [p for n, p in model.named_parameters() if 'mamba' not in n], 'lr': 1e-4}
            ], weight_decay=0.01)
            criterion = nn.CrossEntropyLoss()
            
            y_train = train_df["label_id"].values
            class_counts = np.bincount(y_train, minlength=2)
            weights = 1. / np.clip(class_counts, a_min=1, a_max=None)
            sample_weights = torch.from_numpy(np.array([weights[t] for t in y_train])).double()
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
            
            train_loader = DataLoader(UnifiedMCIFusionDataset(train_df, tokenizer, train_acc), 
                                      batch_size=8, sampler=sampler)

            for epoch in range(12):
                model.train()
                for b in train_loader:
                    optimizer.zero_grad()
                    logits, _ = model(b["input_ids"].to(DEVICE), b["acoustic_features"].to(DEVICE))
                    loss = criterion(logits, b["labels"].to(DEVICE))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                v_m = evaluate_on_dataset(model, val_df, val_acc, tokenizer)
                # ALIGNED SELECTION: Balanced focus on minority (Macro) and general (Weighted) F1
                composite_score = 0.5 * v_m["macro_f1"] + 0.5 * v_m["weighted_f1"]
                if composite_score > best_composite_score:
                    best_composite_score = composite_score
                    best_model_wts = copy.deepcopy(model.state_dict())

            model.load_state_dict(best_model_wts)
            for target_ds in datasets:
                t_mask = df['dataset'] == target_ds
                t_acc = scaler.transform(acoustic_data[t_mask])
                t_metrics = evaluate_on_dataset(model, df[t_mask].reset_index(drop=True), t_acc, tokenizer)
                for k in fold_gen_metrics[target_ds]: fold_gen_metrics[target_ds][k].append(t_metrics[k])
            
            print(f"  Fold {fold+1} Source-Val Composite: {best_composite_score:.4f}")

        source_summary = []
        for target_ds, d in fold_gen_metrics.items():
            entry = {
                "source": source_name, "target": target_ds, "mode": mode,
                "f1_weighted_mean": np.mean(d["weighted_f1"]), 
                "f1_weighted_std": np.std(d["weighted_f1"], ddof=1),
                "f1_macro_mean": np.mean(d["macro_f1"]),
                "recall_dementia_mean": np.mean(d["dementia_recall"]),
                # Safe JSON storage: Casting numpy.float64 to Python float
                "raw_weighted_f1": json.dumps([float(x) for x in d["weighted_f1"]]),
                "raw_macro_f1": json.dumps([float(x) for x in d["macro_f1"]]) 
            }
            source_summary.append(entry)
        
        # Incremental Save to Disk
        pd.DataFrame(source_summary).to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
        print(f"✅ Flushed {mode}-{source_name} to results.")

def main():
    print("\n--- Mamba-Fusion Unified Benchmark v5.2 (Production) ---")
    set_seed(SEED)
    out_file = "cross_corpus_generalization_results.csv"
    
    # 1. Asset Loading
    meta_path = "processed_data/master_metadata_cleaned.csv"
    feat_path = "processed_data/master_acoustic_features.csv"
    cache_path = "processed_data/cleaned_transcripts.json"
    
    if not all(os.path.exists(p) for p in [meta_path, feat_path, cache_path]):
        print("❌ Error: Research assets missing.")
        return

    meta_df = pd.read_csv(meta_path)
    feat_df = pd.read_csv(feat_path)
    with open(cache_path, 'r') as f: transcripts = json.load(f)['transcripts']
    
    meta_df['text'] = meta_df['audio_path'].map(transcripts); df = meta_df.dropna(subset=['text']).reset_index(drop=True)
    df = df[df['label'].isin(['Control', 'Dementia'])].reset_index(drop=True); df['label_id'] = df['label'].map({"Control": 0, "Dementia": 1})
    
    # ROBUST ALIGNMENT: Join on participant_id if audio_path is missing in features
    join_col = 'audio_path' if 'audio_path' in feat_df.columns else 'participant_id'
    print(f"Alignment Strategy: Using '{join_col}'")
    
    feat_cols = [c for c in feat_df.columns if c not in ['participant_id', 'label', 'dataset', 'split', 'age', 'gender', 'mmse', 'audio_path']]
    df = pd.merge(df, feat_df[[join_col] + feat_cols], on=join_col, how='inner'); acoustic_data = df[feat_cols].values
    
    # 2. Global Resources
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token = tokenizer.eos_token
    mamba_backbone = MambaLMHeadModel.from_pretrained("state-spaces/mamba-130m")
    
    sources = list(df['dataset'].unique()) + ["UNIFIED_POOL"]
    modes = ["linguistic", "acoustic", "interaction_only", "fusion_gated", "fusion_attn"]
    
    for mode in modes:
        run_experiment(df, acoustic_data, feat_cols, mode, mamba_backbone, sources, tokenizer, out_file)

    # 3. FINAL STATISTICAL REPORTING (Including resumed data)
    if os.path.exists(out_file):
        try:
            report_df = pd.read_csv(out_file)
            print("\n" + "="*80 + "\nUNIFIED POOL: STATISTICAL RIGOR (WEIGHTED F1)\n" + "="*80)
            pool_mask = report_df['source'] == "UNIFIED_POOL"
            for target in df['dataset'].unique():
                t_m = (report_df['target'] == target) & pool_mask
                try:
                    ling = np.array(json.loads(report_df[t_m & (report_df['mode'] == 'linguistic')]['raw_weighted_f1'].iloc[0]))
                    attn = np.array(json.loads(report_df[t_m & (report_df['mode'] == 'fusion_attn')]['raw_weighted_f1'].iloc[0]))
                    diff = attn - ling
                    std = diff.std(ddof=1)
                    cohen_d = diff.mean() / std if std > 1e-8 else 0
                    _, p = ttest_rel(attn, ling)
                    print(f"TARGET {target.upper():10} | Δ F1: {diff.mean():+.3f} | d: {cohen_d:.2f} | p: {p:.4f}")
                except Exception as e:
                    print(f"  ⚠️ Could not compute stats for {target}: {e}")
            
            print("\n" + "="*80 + "\nFINAL MATRIX (MACRO F1 | DEMENTIA RECALL)\n" + "="*80)
            report_df['Sum'] = report_df.apply(lambda x: f"F1:{x['f1_macro_mean']:.3f}|R:{x['recall_dementia_mean']:.3f}", axis=1)
            print(report_df.pivot_table(index=['mode', 'source'], columns='target', values='Sum', aggfunc='first'))
        except Exception as e: print(f"❌ Reporting error: {e}")

    print(f"\n✅ SUBMISSION READY: v5.2 hardened logic applied to {out_file}")

if __name__ == "__main__":
    main()