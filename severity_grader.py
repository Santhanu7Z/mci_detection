#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Severity Grader v1.0 — MMSE-Aligned Ordinal Multi-Class Classification
Extends binary dementia detection to 4-class severity grading:
  0 = Healthy      (MMSE 27–30 or Control label)
  1 = Mild         (MMSE 21–26)
  2 = Moderate     (MMSE 11–20)
  3 = Severe       (MMSE  0–10)

MMSE band boundaries follow standard clinical convention:
  Folstein et al. (1975) / APA DSM-5 thresholds.

Architecture: TrimodalFusionEngine backbone (frozen Mamba + eGeMAPS + deep audio)
              with an ordinal output head using Conditional Ordinal Regression
              for Neural Networks (CORN) loss from the coral-pytorch library.

If MMSE values are missing (e.g. Pitt corpus partial metadata), the script
falls back to two-level imputation:
  1. Dataset-level MMSE medians from literature (Pitt, ADReSS, TAUKADIAL).
  2. Label-level medians: Control → 28, Dementia → 15.

Outputs:
  severity_model/best_severity_model.bin
  severity_results.json        — per-class metrics
  severity_confusion.csv       — confusion matrix
  severity_calibration.png     — ordinal calibration plot
"""

import os
import json
import copy
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler

from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    mean_absolute_error,
)
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

os.environ["MAMBA_NO_TRITON"] = "1"

# ── MMSE band boundaries ──────────────────────────────────────────────────────
MMSE_BANDS = {          # right-exclusive upper bounds
    0: (27, 31),        # Healthy
    1: (21, 27),        # Mild
    2: (11, 21),        # Moderate
    3: ( 0, 11),        # Severe
}
CLASS_NAMES = ["Healthy", "Mild", "Moderate", "Severe"]
N_ORDINAL   = len(CLASS_NAMES)            # 4

# Literature-derived dataset-level MMSE medians for imputation
# Sources:
#   Pitt (DementiaBank): Becker et al. 1994 — Control≈29, Dementia≈17
#   ADReSS 2020: Luz et al. 2020          — Control≈29, Dementia≈18
#   TAUKADIAL: Luz et al. 2024            — NC≈26, MCI/Dem≈18
DS_MMSE_MEDIANS = {
    ("pitt",      "Control"):  29,
    ("pitt",      "Dementia"): 17,
    ("adress",    "Control"):  29,
    ("adress",    "Dementia"): 18,
    ("taukadial", "Control"):  26,
    ("taukadial", "Dementia"): 18,
}
LABEL_FALLBACK = {"Control": 28, "Dementia": 15}

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR       = "processed_data"
DEEP_AUDIO_DIR = os.path.join(DATA_DIR, "deep_audio")
MODEL_DIR      = "severity_model"
os.makedirs(MODEL_DIR, exist_ok=True)

MAMBA_ID     = "state-spaces/mamba-130m"
TOKENIZER_ID = "EleutherAI/gpt-neox-20b"


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── MMSE → severity class ─────────────────────────────────────────────────────

def mmse_to_severity(mmse_val: float) -> int:
    """Map a numeric MMSE score to a 0-indexed severity class."""
    if mmse_val is None or (isinstance(mmse_val, float) and np.isnan(mmse_val)):
        return -1   # sentinel for missing
    mmse_val = float(mmse_val)
    for cls, (lo, hi) in MMSE_BANDS.items():
        if lo <= mmse_val < hi:
            return cls
    return 3        # MMSE < 0 treated as Severe

def impute_mmse(df: pd.DataFrame) -> pd.Series:
    df['mmse'] = pd.to_numeric(df['mmse'], errors='coerce')
    """
    Three-level MMSE imputation strategy:
    1. Use existing MMSE if present.
    2. Dataset × label median from literature.
    3. Global label fallback.
    Returns a Series of MMSE floats.
    """
    mmse = df["mmse"].copy()
    missing_mask = mmse.isna()

    for idx in df.index[missing_mask]:
        ds    = str(df.loc[idx, "dataset"]).lower()
        label = df.loc[idx, "label"]
        key   = (ds, label)
        if key in DS_MMSE_MEDIANS:
            mmse.at[idx] = DS_MMSE_MEDIANS[key]
        else:
            mmse.at[idx] = LABEL_FALLBACK.get(label, 20)

    return mmse


# ── CORN loss (Ordinal) ───────────────────────────────────────────────────────
#
# CORN (Conditional Ordinal Regression for Neural Networks) —
# Shi et al. NeurIPS 2021, https://arxiv.org/abs/2111.08851
#
# The model outputs K-1 binary classifiers, one per ordinal boundary.
# rank[k] = P(y > k | y > k-1) (conditional probability).
# CORN trains each rank independently with BCE, then derives P(y=k) from
# the product of conditional probabilities.
# This avoids the monotonicity issues of plain multi-class softmax for ordinal
# labels (no rank crossings at inference time).

class CORNLoss(nn.Module):
    """
    CORN loss for K ordinal classes (K-1 binary boundaries).

    Model must output logits of shape [B, K-1].
    Labels must be in {0, 1, …, K-1}.
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.K  = num_classes     # 4
        self.Km1 = num_classes - 1

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        logits : [B, K-1]  — raw scores for each boundary
        labels : [B]       — integer class indices 0 … K-1
        """
        # Binary targets for each boundary rank k: 1 if y >= k+1
        # Rank 0: y >= 1, Rank 1: y >= 2, …, Rank K-2: y >= K-1
        sets = []
        for k in range(self.Km1):
            label_mask = (labels > k).float()
            # Condition: only samples with label >= k contribute to rank k
            # (samples with y < k are excluded from the conditional set)
            in_set = (labels >= k).float()
            sets.append((label_mask, in_set))

        loss = torch.tensor(0.0, device=logits.device)
        n_active = 0

        for k, (targets, in_set) in enumerate(sets):
            if in_set.sum() == 0:
                continue
            # Select samples in conditional set
            idx    = in_set.bool()
            logit_k = logits[idx, k]
            target_k = targets[idx]
            loss    += F.binary_cross_entropy_with_logits(logit_k, target_k)
            n_active += 1

        return loss / max(n_active, 1)


def corn_predict(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert CORN boundary logits [B, K-1] to class predictions [B].
    P(y > k) = prod_{j=0}^{k} sigmoid(logit_j)
    """
    probs_gt = torch.sigmoid(logits)          # [B, K-1]  P(y > k | y >= k)

    # Cumulative product to get marginal P(y > k)
    cum = torch.cumprod(probs_gt, dim=1)      # [B, K-1]
    # P(y = 0) = 1 - P(y > 0)
    # P(y = k) = P(y > k-1) - P(y > k)   for k = 1 … K-2
    # P(y = K-1) = P(y > K-2)
    K   = logits.shape[1] + 1
    B   = logits.shape[0]

    p_class = torch.zeros(B, K, device=logits.device)
    p_class[:, 0] = 1.0 - cum[:, 0]
    for k in range(1, K - 1):
        p_class[:, k] = cum[:, k-1] - cum[:, k]
    p_class[:, K-1] = cum[:, K-2]

    # Clamp for numerical safety
    p_class = torch.clamp(p_class, min=1e-6)
    return torch.argmax(p_class, dim=1), p_class


# ── Dataset ───────────────────────────────────────────────────────────────────

class SeverityDataset(Dataset):
    def __init__(self, df, tokenizer, egemap_arr, deep_audio_arr, max_len=512):
        self.df         = df.reset_index(drop=True)
        self.tokenizer  = tokenizer
        self.egemap     = egemap_arr
        self.deep_audio = deep_audio_arr
        self.max_len    = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.df.loc[idx, "text"]),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids":  enc["input_ids"].squeeze(0),
            "egemap":     torch.tensor(self.egemap[idx],     dtype=torch.float32),
            "deep_audio": torch.tensor(self.deep_audio[idx], dtype=torch.float32),
            "labels":     torch.tensor(int(self.df.loc[idx, "severity"]), dtype=torch.long),
        }


# ── Model ─────────────────────────────────────────────────────────────────────

class SeverityFusionModel(nn.Module):
    """
    Trimodal fusion backbone + CORN ordinal output head.

    Backbone architecture mirrors TrimodalFusionEngine from lodo_benchmark.py
    but the classification head is replaced with K-1 binary boundary logits.
    """

    def __init__(self,
                 mamba_model,
                 egemap_dim:     int,
                 deep_audio_dim: int,
                 n_classes:      int = N_ORDINAL,
                 freeze_backbone: bool = True):
        super().__init__()

        self.mamba = mamba_model.backbone
        text_dim   = mamba_model.config.d_model   # 768
        fdim       = 256
        self.fdim  = fdim
        self.K     = n_classes

        if freeze_backbone:
            for p in self.mamba.parameters():
                p.requires_grad = False

        # Projections
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, fdim), nn.LayerNorm(fdim), nn.GELU()
        )
        self.egemap_proj = nn.Sequential(
            nn.Linear(egemap_dim, 128), nn.GELU(),
            nn.Linear(128, fdim), nn.LayerNorm(fdim), nn.GELU()
        )
        self.deep_proj = nn.Sequential(
            nn.Linear(deep_audio_dim, 512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, fdim), nn.LayerNorm(fdim), nn.GELU()
        )

        # 3-way cross-modal attention
        self.fusion_attn = nn.MultiheadAttention(fdim, num_heads=4,
                                                  dropout=0.25, batch_first=True)
        self.fusion_norm = nn.LayerNorm(fdim)
        self.ffn = nn.Sequential(
            nn.Linear(fdim, fdim * 2), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(fdim * 2, fdim)
        )
        self.ffn_norm = nn.LayerNorm(fdim)

        # CORN head: K-1 binary boundary classifiers
        # Shared trunk → K-1 outputs
        self.ordinal_trunk = nn.Sequential(
            nn.Linear(fdim * 3, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64), nn.GELU(),
        )
        # K-1 parallel boundary heads (shared parameters)
        self.boundary_head = nn.Linear(64, n_classes - 1)

        self.corn_loss = CORNLoss(n_classes)

    def _encode_text(self, input_ids):
        backbone_grad = any(p.requires_grad for p in self.mamba.parameters())
        with torch.set_grad_enabled(backbone_grad):
            hidden = self.mamba(input_ids)
        return hidden.mean(dim=1)

    def forward(self, input_ids, egemap, deep_audio, labels=None):
        t = self.text_proj(self._encode_text(input_ids))   # [B, 256]
        a = self.egemap_proj(egemap)                        # [B, 256]
        d = self.deep_proj(deep_audio)                      # [B, 256]

        tokens   = torch.stack([t, a, d], dim=1)           # [B, 3, 256]
        attn_out, attn_w = self.fusion_attn(tokens, tokens, tokens)
        tokens   = self.fusion_norm(tokens + attn_out)
        tokens   = self.ffn_norm(tokens + self.ffn(tokens))

        fused    = tokens.view(tokens.size(0), -1)         # [B, 768]
        trunk    = self.ordinal_trunk(fused)               # [B, 64]
        logits   = self.boundary_head(trunk)               # [B, K-1]

        loss = self.corn_loss(logits, labels) if labels is not None else None
        return loss, logits, attn_w


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_severity(model, loader, device) -> dict:
    model.eval()
    preds, labels, probs_all, losses = [], [], [], []

    for batch in loader:
        ids = batch["input_ids"].to(device)
        eg  = batch["egemap"].to(device)
        da  = batch["deep_audio"].to(device)
        y   = batch["labels"].to(device)

        with autocast(enabled=(device.type == "cuda")):
            loss, logits, _ = model(ids, eg, da, y)

        losses.append(loss.item())
        p_cls, p_soft = corn_predict(logits)
        preds.extend(p_cls.cpu().numpy())
        labels.extend(y.cpu().numpy())
        probs_all.extend(p_soft.cpu().numpy())

    labels_np = np.array(labels)
    preds_np  = np.array(preds)

    return {
        "loss":        np.mean(losses),
        "accuracy":    accuracy_score(labels_np, preds_np),
        "f1_weighted": f1_score(labels_np, preds_np, average="weighted",
                                zero_division=0, labels=list(range(N_ORDINAL))),
        "f1_macro":    f1_score(labels_np, preds_np, average="macro",
                                zero_division=0, labels=list(range(N_ORDINAL))),
        "cohen_kappa_linear":   cohen_kappa_score(labels_np, preds_np,
                                                  weights="linear"),
        "cohen_kappa_quadratic":cohen_kappa_score(labels_np, preds_np,
                                                  weights="quadratic"),
        "mae":         mean_absolute_error(labels_np, preds_np),
        "labels":      labels_np,
        "preds":       preds_np,
        "probs":       np.array(probs_all),
    }


# ── Calibration plot ──────────────────────────────────────────────────────────

def plot_calibration(results: dict, fold: int, output_dir: str):
    """Confidence calibration: mean predicted prob vs actual class frequency."""
    probs  = results["probs"]      # [N, K]
    labels = results["labels"]     # [N]
    n_bins = 8

    fig, axes = plt.subplots(1, N_ORDINAL,
                             figsize=(4 * N_ORDINAL, 4), sharey=True)
    fig.suptitle(f"Ordinal Calibration — Fold {fold}", fontsize=12)

    for k, ax in enumerate(axes):
        p_k   = probs[:, k]
        is_k  = (labels == k).astype(float)

        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_acc   = []
        bin_conf  = []
        bin_count = []

        for i in range(n_bins):
            mask = (p_k >= bin_edges[i]) & (p_k < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_acc.append(is_k[mask].mean())
                bin_conf.append(p_k[mask].mean())
                bin_count.append(mask.sum())

        if not bin_conf:
            continue

        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
        ax.scatter(bin_conf, bin_acc, s=[10 * c for c in bin_count],
                   alpha=0.75, color="#4C72B0")
        ax.plot(bin_conf, bin_acc, "-o", color="#4C72B0")
        ax.set_title(CLASS_NAMES[k])
        ax.set_xlabel("Mean predicted P")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    axes[0].set_ylabel("Fraction true positives")
    plt.tight_layout()
    out = os.path.join(output_dir, f"severity_calibration_fold{fold}.png")
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def plot_confusion(cm: np.ndarray, output_dir: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(N_ORDINAL)); ax.set_yticks(range(N_ORDINAL))
    ax.set_xticklabels(CLASS_NAMES, rotation=20, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Severity Confusion Matrix")
    for i in range(N_ORDINAL):
        for j in range(N_ORDINAL):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    out = os.path.join(output_dir, "severity_confusion.png")
    plt.savefig(out, dpi=200)
    plt.close()
    return out


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    print("\n─── Loading assets ─────────────────────────────────────────────")

    meta = pd.read_csv(os.path.join(DATA_DIR, "master_metadata_cleaned.csv"))

    with open(os.path.join(DATA_DIR, "cleaned_transcripts.json")) as f:
        transcripts = json.load(f)["transcripts"]

    meta["text"] = meta["audio_path"].map(transcripts)
    meta = meta.dropna(subset=["text"])
    meta = meta[meta["label"].isin(["Control", "Dementia"])].reset_index(drop=True)

    # MMSE imputation
    if "mmse" not in meta.columns:
        meta["mmse"] = np.nan
    meta["mmse_filled"] = impute_mmse(meta)
    meta["severity"]    = meta["mmse_filled"].apply(mmse_to_severity)
    meta = meta[meta["severity"] >= 0].reset_index(drop=True)

    print("\nMMSE imputation complete:")
    total        = len(meta)
    had_mmse     = meta["mmse"].notna().sum()
    was_imputed  = total - had_mmse
    print(f"  Samples with real MMSE : {had_mmse} ({100*had_mmse/total:.1f}%)")
    print(f"  Imputed from literature: {was_imputed} ({100*was_imputed/total:.1f}%)")
    print("\nSeverity class distribution:")
    for k, name in enumerate(CLASS_NAMES):
        n = (meta["severity"] == k).sum()
        print(f"  {k} – {name:<10}: {n:>4} samples")

    # eGeMAPS
    eg_df    = pd.read_csv(os.path.join(DATA_DIR, "master_acoustic_features.csv"))
    eg_cols  = [c for c in eg_df.columns if c not in
                ["participant_id","audio_path","label","dataset",
                 "split","age","gender","mmse"]]
    join_col = "audio_path" if "audio_path" in eg_df.columns else "participant_id"
    meta     = pd.merge(meta, eg_df[[join_col] + eg_cols], on=join_col, how="inner")

    # Deep audio
    ids_path = os.path.join(DEEP_AUDIO_DIR, "participant_ids.npy")
    if not os.path.exists(ids_path):
        raise FileNotFoundError(
            "Deep audio embeddings not found. Run: python deep_audio_extractor.py"
        )
    da_ids  = np.load(ids_path, allow_pickle=True)
    w2v     = np.load(os.path.join(DEEP_AUDIO_DIR, "wav2vec2_embeddings.npy"))
    hub     = np.load(os.path.join(DEEP_AUDIO_DIR, "hubert_embeddings.npy"))
    da_map  = {pid: i for i, pid in enumerate(da_ids)}

    da_idx  = meta["participant_id"].map(da_map)
    valid   = da_idx.notna()
    meta    = meta[valid].reset_index(drop=True)
    da_idx  = da_idx[valid].astype(int).values

    deep_all  = np.hstack([w2v[da_idx], hub[da_idx]])
    egemap_all = meta[eg_cols].values

    print(f"\nAligned N = {len(meta)}  |  eGeMAPS dim = {egemap_all.shape[1]}"
          f"  |  Deep dim = {deep_all.shape[1]}")
    return meta, egemap_all, deep_all, eg_cols


# ── Training loop ─────────────────────────────────────────────────────────────

def run_fold(train_df, tr_eg, tr_da,
             val_df,   va_eg, va_da,
             tokenizer, device, args, seed: int) -> nn.Module:

    set_seed(seed)
    base  = MambaLMHeadModel.from_pretrained(MAMBA_ID)
    model = SeverityFusionModel(
        copy.deepcopy(base),
        egemap_dim=tr_eg.shape[1],
        deep_audio_dim=tr_da.shape[1],
        n_classes=N_ORDINAL,
        freeze_backbone=False,
    ).to(device)

    # Weighted sampler for class imbalance
    y_tr   = train_df["severity"].values
    cw     = compute_class_weight("balanced",
                                  classes=np.arange(N_ORDINAL), y=y_tr)
    sw     = torch.tensor([float(cw[y]) for y in y_tr], dtype=torch.double)
    sampler= WeightedRandomSampler(sw, len(sw), replacement=True)

    train_ds = SeverityDataset(train_df, tokenizer, tr_eg, tr_da)
    val_ds   = SeverityDataset(val_df,   tokenizer, va_eg, va_da)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size,
                          sampler=sampler, num_workers=0, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch_size * 2, num_workers=0)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                             T_max=args.epochs)
    amp_scaler= GradScaler(enabled=(device.type == "cuda"))

    best_kappa, best_state = -1.0, copy.deepcopy(model.state_dict())
    patience = 0

    print(f"    Training {len(train_df)} samples, validating {len(val_df)} …")

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        for i, batch in enumerate(train_ld):
            ids = batch["input_ids"].to(device)
            eg  = batch["egemap"].to(device)
            da  = batch["deep_audio"].to(device)
            y   = batch["labels"].to(device)

            with autocast(enabled=(device.type == "cuda")):
                loss, _, _ = model(ids, eg, da, y)
                loss       = loss / args.acc_steps

            amp_scaler.scale(loss).backward()
            if (i + 1) % args.acc_steps == 0:
                amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                amp_scaler.step(optimizer)
                amp_scaler.update()
                optimizer.zero_grad()

        scheduler.step()

        res = evaluate_severity(model, val_ld, device)
        kappa = res["cohen_kappa_quadratic"]

        if kappa > best_kappa:
            best_kappa = kappa
            best_state = copy.deepcopy(model.state_dict())
            patience   = 0
            marker     = " ✓"
        else:
            patience += 1
            marker   = ""

        if (epoch + 1) % 5 == 0 or marker:
            print(f"    Epoch {epoch+1:>3}/{args.epochs}  "
                  f"κ²={kappa:.4f}  F1-W={res['f1_weighted']:.4f}"
                  f"  MAE={res['mae']:.3f}{marker}")

        if patience >= args.patience:
            print(f"    Early stop at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    return model


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    import builtins
    import functools
    builtins.print = functools.partial(builtins.print, flush=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    meta, egemap_all, deep_all, eg_cols = load_data()

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # ── Stratified group k-fold ───────────────────────────────────────────────
    sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True,
                                 random_state=args.seed)
    groups = meta["participant_id"].values
    labels = meta["severity"].values

    fold_results = []

    for fold, (tr_idx, te_idx) in enumerate(
            sgkf.split(meta, labels, groups=groups)):
        print(f"\n{'='*68}")
        print(f"  FOLD {fold+1}/{args.folds}")
        print(f"{'='*68}")

        tr_df = meta.iloc[tr_idx].reset_index(drop=True)
        te_df = meta.iloc[te_idx].reset_index(drop=True)

        # Internal 10% val from train for early stopping
        tr_df2, va_df = train_test_split(
            tr_df, test_size=0.10,
            stratify=tr_df["severity"],
            random_state=args.seed
        )
        tr_idx2 = tr_df2.index.values
        va_idx2 = va_df.index.values

        eg_sc = StandardScaler()
        da_sc = StandardScaler()

        tr_eg = eg_sc.fit_transform(egemap_all[tr_idx][tr_idx2])
        va_eg = eg_sc.transform(egemap_all[tr_idx][va_idx2])
        te_eg = eg_sc.transform(egemap_all[te_idx])

        tr_da = da_sc.fit_transform(deep_all[tr_idx][tr_idx2])
        va_da = da_sc.transform(deep_all[tr_idx][va_idx2])
        te_da = da_sc.transform(deep_all[te_idx])

        tr_df2 = tr_df2.reset_index(drop=True)
        va_df  = va_df.reset_index(drop=True)

        model = run_fold(tr_df2, tr_eg, tr_da,
                         va_df,  va_eg, va_da,
                         tokenizer, device, args, args.seed + fold)

        # Test evaluation
        te_ds = SeverityDataset(te_df, tokenizer, te_eg, te_da)
        te_ld = DataLoader(te_ds, batch_size=args.batch_size * 2, num_workers=0)
        res   = evaluate_severity(model, te_ld, device)

        fold_results.append(res)

        print(f"\n  ── Fold {fold+1} Test Results:")
        print(f"     Accuracy          : {res['accuracy']:.4f}")
        print(f"     F1 Weighted       : {res['f1_weighted']:.4f}")
        print(f"     F1 Macro          : {res['f1_macro']:.4f}")
        print(f"     Cohen κ (linear)  : {res['cohen_kappa_linear']:.4f}")
        print(f"     Cohen κ (quadratic): {res['cohen_kappa_quadratic']:.4f}")
        print(f"     MAE               : {res['mae']:.4f}")
        print()
        print(classification_report(res["labels"], res["preds"],
                                    target_names=CLASS_NAMES,
                                    zero_division=0,
                                    labels=list(range(N_ORDINAL))))

        # Save calibration plot for this fold
        cal_path = plot_calibration(res, fold + 1, MODEL_DIR)
        print(f"  Calibration plot → {cal_path}")

        # Save best model per fold
        ckpt = os.path.join(MODEL_DIR, f"severity_fold{fold+1}.bin")
        torch.save(model.state_dict(), ckpt)

        del model
        torch.cuda.empty_cache()

    # ── Aggregate across folds ────────────────────────────────────────────────
    def agg(key):
        return float(np.mean([r[key] for r in fold_results]))
    def std(key):
        vals = [r[key] for r in fold_results]
        return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    print(f"\n{'='*68}")
    print(f"  AGGREGATED RESULTS ({args.folds}-fold Stratified Group CV)")
    print(f"{'='*68}")
    metrics = {
        "accuracy":              (agg("accuracy"),             std("accuracy")),
        "f1_weighted":           (agg("f1_weighted"),          std("f1_weighted")),
        "f1_macro":              (agg("f1_macro"),             std("f1_macro")),
        "cohen_kappa_linear":    (agg("cohen_kappa_linear"),   std("cohen_kappa_linear")),
        "cohen_kappa_quadratic": (agg("cohen_kappa_quadratic"),std("cohen_kappa_quadratic")),
        "mae":                   (agg("mae"),                  std("mae")),
    }
    for name, (mean_v, std_v) in metrics.items():
        print(f"  {name:<28}: {mean_v:.4f} ± {std_v:.4f}")

    # Pooled confusion matrix
    all_labels = np.concatenate([r["labels"] for r in fold_results])
    all_preds  = np.concatenate([r["preds"]  for r in fold_results])
    cm         = confusion_matrix(all_labels, all_preds,
                                  labels=list(range(N_ORDINAL)))

    print("\n  Pooled Confusion Matrix:")
    print("  " + "  ".join(f"{n:>9}" for n in CLASS_NAMES))
    for i, row in enumerate(cm):
        cells = "  ".join(f"{v:>9}" for v in row)
        print(f"  {CLASS_NAMES[i]:<9} {cells}")

    cm_path = plot_confusion(cm, MODEL_DIR)
    print(f"\n  Confusion matrix plot → {cm_path}")

    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        os.path.join(MODEL_DIR, "severity_confusion.csv")
    )

    # Save summary JSON
    summary = {k: {"mean": round(v[0], 4), "std": round(v[1], 4)}
               for k, v in metrics.items()}
    summary["class_names"]  = CLASS_NAMES
    summary["mmse_bands"]   = {CLASS_NAMES[k]: list(v)
                                for k, v in MMSE_BANDS.items()}
    summary["n_folds"]      = args.folds
    summary["n_samples"]    = len(meta)
    summary["severity_dist"]= {CLASS_NAMES[k]: int((all_labels == k).sum())
                                for k in range(N_ORDINAL)}
    with open(os.path.join(MODEL_DIR, "severity_results.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n✅ Results saved → {MODEL_DIR}/severity_results.json")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MMSE-Aligned Ordinal Severity Grader")
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--acc_steps",  type=int,   default=4)
    parser.add_argument("--lr",         type=float, default=5e-5)
    parser.add_argument("--patience",   type=int,   default=12)
    parser.add_argument("--folds",      type=int,   default=5)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()
    main(args)
