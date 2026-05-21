#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Leave-One-Dataset-Out (LODO) Benchmark v4.1
============================================
Includes Unified Pool (all-datasets 5-fold CV) as upper-bound reference,
matching benchmark.py's proven training strategy.

Critical fixes applied vs v3.0
────────────────────────────────
[F1] MASKED MEAN POOLING (not max)
     hidden * attention_mask → mean over non-padding positions only.
     Max pooling is brittle to noisy/padded tokens.

[F2] STABLE PAIRWISE INTERACTION (not triple product)
     interaction_only / fusion modes use:
       feat = [t, a, d, t*a, t*d, a*d]  →  Linear(6×256, 256)
     Triple product t*a*d has exploding/vanishing gradient when any
     factor is near zero (grad w.r.t. t = a*d, which collapses).

[F3] CONSISTENT LOSS OBJECTIVE
     Dropped WeightedRandomSampler. Using CrossEntropyLoss(weight=)
     instead, which keeps the gradient distribution and probability
     calibration consistent (sampler + unweighted loss is incoherent).
     Weights are clipped to [0.5, 5.0] to prevent explosions on small splits.

[F4] HONEST AUC FALLBACK & BRIER SCORE
     Returns np.nan (not 0.5) when AUC is undefined. Added Brier Score
     to track true probability calibration.

[F5] STRATIFIED VALIDATION SPLIT
     Uses StratifiedShuffleSplit instead of np.random.choice so that
     small (100-sample) training sets get a val set with both classes.

[F6] ATTENTION WEIGHT CAVEAT
     Attention-based modality weights are logged as attn_proxy_* to
     signal they are not causal importance scores (attention ≠ importance).

[F7] AMP RESTORED + STABLE SOFTMAX
     torch.cuda.amp autocast + GradScaler reintroduced for GPU efficiency.
     Logits are cast to float32 before softmax for stability.

[F8] CHECKPOINT PER RUN (DISABLED)
     Checkpoint saving has been commented out to prevent exceeding 
     the 100GB Slurm user quota limit during the extensive LODO sweeps.

[F9] STRUCTURED LOGGING
     Per-epoch CSV log written to logs/training_log.csv.

[F10] SHM FIX & BOUNDED RESIDUALS
     num_workers=0 to fix Shared Memory crashes. fusion_gated uses a 
     Sigmoid-bounded learned alpha. fusion_attn flattens to prevent double-smoothing.

[F11] ABLATION SUPPORT (v4.1 addition)
     --feature_csv  selects any eGeMAPS CSV in processed_data/ without
                    editing the source file.
     --run_tag      appends a suffix to all four output files so baseline
                    and ablation runs never collide and resumption works
                    correctly for each independently.
"""

import os, json, copy, random, argparse, csv, time
from datetime import datetime
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_recall_fscore_support,
    roc_auc_score, matthews_corrcoef,
    brier_score_loss
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

os.environ["MAMBA_NO_TRITON"] = "1"

# ── Hyperparameters (aligned with benchmark.py) ───────────────────────────────
HPARAMS = dict(
    epochs      = 20,
    min_epochs  = 8,
    patience    = 5,
    batch_size  = 8,
    lr_backbone = 2e-5,
    lr_head     = 1e-4,
    wd_backbone = 0.01,
    wd_head     = 0.05,
    max_len     = 512,
    n_seeds     = 3,
    base_seed   = 42,
    val_frac    = 0.15,
    pool_folds  = 5,
)

MODES = [
    "linguistic",
    "acoustic_egemaps",
    "acoustic_w2v",
    "acoustic_hubert",
    "interaction_only",
    "fusion_gated",
    "fusion_attn",
]

LODO_COMBOS = [
    # Multi-domain
    {"train": ["pitt", "adress"],      "test": "taukadial", "tag": "PA->T",  "pool": False},
    {"train": ["pitt", "taukadial"],   "test": "adress",    "tag": "PT->A",  "pool": False},
    {"train": ["adress", "taukadial"], "test": "pitt",      "tag": "AT->P",  "pool": False},
    # Single-domain
    {"train": ["pitt"],      "test": "adress",    "tag": "P->A",   "pool": False},
    {"train": ["pitt"],      "test": "taukadial", "tag": "P->T",   "pool": False},
    {"train": ["adress"],    "test": "pitt",      "tag": "A->P",   "pool": False},
    {"train": ["adress"],    "test": "taukadial", "tag": "A->T",   "pool": False},
    {"train": ["taukadial"], "test": "pitt",      "tag": "T->P",   "pool": False},
    {"train": ["taukadial"], "test": "adress",    "tag": "T->A",   "pool": False},
    # Unified Pool (all datasets, 5-fold CV)
    {"train": ["pitt", "adress", "taukadial"], "test": "all_cv", "tag": "POOL", "pool": True},
]

DATA_DIR     = "processed_data"
DEEP_DIR     = os.path.join(DATA_DIR, "deep_audio")
CKPT_DIR     = "checkpoints"
LOG_DIR      = "logs"
MAMBA_ID     = "state-spaces/mamba-130m"
TOKENIZER_ID = "EleutherAI/gpt-neox-20b"

# These four are reassigned in __main__ when --run_tag is supplied.
RESULTS_FILE = "lodo_results.csv"
SUMMARY_FILE = "lodo_summary.csv"
REPORT_FILE  = "lodo_report.txt"
TRAIN_LOG    = os.path.join(LOG_DIR, "training_log.csv")

METRIC_COLS = [
    "accuracy", "f1_weighted", "f1_macro", "auc_roc", "brier_score",
    "mcc", "sensitivity", "specificity", "precision_dem", "precision_ctrl",
]

_TEXT_MODES = {"linguistic", "interaction_only", "fusion_gated", "fusion_attn"}
_EG_MODES   = {"acoustic_egemaps", "interaction_only", "fusion_gated", "fusion_attn"}
_W2V_MODES  = {"acoustic_w2v"}
_HUB_MODES  = {"acoustic_hubert"}
_DEEP_MODES = {"interaction_only", "fusion_gated", "fusion_attn"}
_PAIRWISE   = {"interaction_only", "fusion_gated", "fusion_attn"}


# ════════════════════════════════════════════════════════════════════════════════
# SETUP
# ════════════════════════════════════════════════════════════════════════════════

def setup_dirs():
    for d in (CKPT_DIR, LOG_DIR):
        os.makedirs(d, exist_ok=True)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ════════════════════════════════════════════════════════════════════════════════
# LOGGING  [F9]
# ════════════════════════════════════════════════════════════════════════════════

_log_writer = None
_log_file   = None

def init_log():
    global _log_writer, _log_file
    _log_file   = open(TRAIN_LOG, "a", newline="")
    _log_writer = csv.writer(_log_file)
    if os.path.getsize(TRAIN_LOG) == 0:
        _log_writer.writerow([
            "timestamp", "mode", "tag", "seed", "fold",
            "epoch", "composite", "f1_weighted", "f1_macro",
            "sensitivity", "specificity", "loss"
        ])

def log_epoch(mode, tag, seed, fold, epoch, m: dict):
    if _log_writer is None:
        return
    _log_writer.writerow([
        datetime.now().isoformat(), mode, tag, seed, fold, epoch,
        f"{m.get('composite', 0):.4f}",
        f"{m.get('f1_weighted', 0):.4f}",
        f"{m.get('f1_macro', 0):.4f}",
        f"{m.get('sensitivity', 0):.4f}",
        f"{m.get('specificity', 0):.4f}",
        f"{m.get('loss', 0):.4f}",
    ])
    _log_file.flush()

def close_log():
    if _log_file:
        _log_file.close()


# ════════════════════════════════════════════════════════════════════════════════
# DATASET
# ════════════════════════════════════════════════════════════════════════════════

class TrimodalDataset(Dataset):
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
            truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "egemap":         torch.tensor(self.egemap[idx],     dtype=torch.float32),
            "deep_audio":     torch.tensor(self.deep_audio[idx], dtype=torch.float32),
            "labels":         torch.tensor(int(self.df.loc[idx, "label_id"]), dtype=torch.long),
        }


# ════════════════════════════════════════════════════════════════════════════════
# PAIRWISE INTERACTION MODULE  [F2]
# ════════════════════════════════════════════════════════════════════════════════

class PairwiseInteraction(nn.Module):
    """
    Stable multi-modal interaction replacing triple product.
    Input : three 256-d embeddings (t, a, d)
    Output: 256-d fused feature via concat([t,a,d,t*a,t*d,a*d]) → Linear → GELU
    """
    def __init__(self, fdim: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(fdim * 6, fdim),
            nn.LayerNorm(fdim),
            nn.GELU(),
        )

    def forward(self, t: torch.Tensor, a: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([t, a, d, t * a, t * d, a * d], dim=-1))


# ════════════════════════════════════════════════════════════════════════════════
# MODEL
# ════════════════════════════════════════════════════════════════════════════════

class MultiModeFusionEngine(nn.Module):
    def __init__(self, mode: str, mamba_model=None,
                 egemap_dim: int = 88, deep_audio_dim: int = 1536,
                 num_labels: int = 2, class_weights=None):
        super().__init__()
        assert mode in MODES, f"Unknown mode: {mode}"
        self.mode = mode
        fdim      = 256
        text_dim  = 768

        if mode in _TEXT_MODES:
            assert mamba_model is not None
            self.mamba = mamba_model.backbone
            self.text_proj = nn.Linear(text_dim, fdim)

        if mode in _EG_MODES:
            self.egemap_proj = nn.Sequential(
                nn.Linear(egemap_dim, 128), nn.GELU(), nn.Linear(128, fdim)
            )

        if mode in _W2V_MODES:
            self.w2v_proj = nn.Sequential(
                nn.Linear(768, 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, fdim)
            )

        if mode in _HUB_MODES:
            self.hubert_proj = nn.Sequential(
                nn.Linear(768, 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, fdim)
            )

        if mode in _DEEP_MODES:
            self.deep_proj = nn.Sequential(
                nn.Linear(deep_audio_dim, 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, fdim)
            )

        if mode in _PAIRWISE:
            self.pairwise = PairwiseInteraction(fdim)

        if mode == "fusion_gated":
            self.gate_layer = nn.Linear(fdim * 3, 3)
            self.pw_alpha = nn.Parameter(torch.tensor(-2.2))

        if mode == "fusion_attn":
            self.attention = nn.MultiheadAttention(fdim, num_heads=4, batch_first=True)
            clf_in = fdim * 4
        else:
            clf_in = fdim

        self.classifier = nn.Sequential(
            nn.Linear(clf_in, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, num_labels)
        )

        w = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        self.criterion = nn.CrossEntropyLoss(weight=w)

    def _masked_mean(self, hidden: torch.Tensor,
                     attention_mask: torch.Tensor) -> torch.Tensor:
        mask     = attention_mask.unsqueeze(-1).float()
        summed   = (hidden * mask).sum(dim=1)
        n_tokens = mask.sum(dim=1).clamp(min=1e-6)
        return summed / n_tokens

    def _encode_text(self, input_ids, attention_mask):
        hidden = self.mamba(input_ids)
        return self._masked_mean(hidden, attention_mask)

    def forward(self, input_ids, attention_mask, egemap, deep_audio, labels=None):
        m          = self.mode
        attn_proxy = None

        if m == "linguistic":
            t    = self.text_proj(self._encode_text(input_ids, attention_mask))
            feat = t

        elif m == "acoustic_egemaps":
            feat = self.egemap_proj(egemap)

        elif m == "acoustic_w2v":
            feat = self.w2v_proj(deep_audio[:, :768])

        elif m == "acoustic_hubert":
            feat = self.hubert_proj(deep_audio[:, 768:])

        elif m == "interaction_only":
            t    = self.text_proj(self._encode_text(input_ids, attention_mask))
            a    = self.egemap_proj(egemap)
            d    = self.deep_proj(deep_audio)
            feat = self.pairwise(t, a, d)

        elif m == "fusion_gated":
            t      = self.text_proj(self._encode_text(input_ids, attention_mask))
            a      = self.egemap_proj(egemap)
            d      = self.deep_proj(deep_audio)
            pw     = self.pairwise(t, a, d)
            g      = torch.softmax(self.gate_layer(torch.cat([t, a, d], dim=-1)), dim=-1)
            alpha  = torch.sigmoid(self.pw_alpha)
            feat   = g[:, 0:1] * t + g[:, 1:2] * a + g[:, 2:3] * d + pw * alpha
            attn_proxy = g.detach()

        elif m == "fusion_attn":
            t      = self.text_proj(self._encode_text(input_ids, attention_mask))
            a      = self.egemap_proj(egemap)
            d      = self.deep_proj(deep_audio)
            pw     = self.pairwise(t, a, d)
            tokens = torch.stack([t, a, d], dim=1)
            out, w = self.attention(tokens, tokens, tokens)
            attn_proxy = w.detach().mean(dim=1) if w is not None else None
            feat   = torch.cat([out.flatten(start_dim=1), pw], dim=-1)

        logits = self.classifier(feat)
        loss   = self.criterion(logits, labels) if labels is not None else None
        return loss, logits, attn_proxy


# ════════════════════════════════════════════════════════════════════════════════
# METRICS  [F4]
# ════════════════════════════════════════════════════════════════════════════════

def compute_metrics(labels, preds, probs_dem=None) -> dict:
    labels = np.asarray(labels)
    preds  = np.asarray(preds)
    prec, rec, _, _ = precision_recall_fscore_support(
        labels, preds, average=None, labels=[0, 1], zero_division=0
    )
    auc = np.nan
    brier = np.nan
    if probs_dem is not None and len(np.unique(labels)) > 1:
        try:
            auc   = float(roc_auc_score(labels, probs_dem))
            brier = float(brier_score_loss(labels, probs_dem))
        except ValueError:
            auc   = np.nan
            brier = np.nan

    return {
        "accuracy"      : float(accuracy_score(labels, preds)),
        "f1_weighted"   : float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "f1_macro"      : float(f1_score(labels, preds, average="macro",    zero_division=0)),
        "auc_roc"       : auc,
        "brier_score"   : brier,
        "mcc"           : float(matthews_corrcoef(labels, preds)),
        "sensitivity"   : float(rec[1]),
        "specificity"   : float(rec[0]),
        "precision_dem" : float(prec[1]),
        "precision_ctrl": float(prec[0]),
    }


# ════════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ════════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    all_preds, all_labels, all_probs_dem = [], [], []
    all_proxy = []
    losses    = []

    for batch in loader:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        eg   = batch["egemap"].to(device)
        da   = batch["deep_audio"].to(device)
        y    = batch["labels"].to(device)

        with autocast(enabled=(device.type == "cuda")):
            loss, logits, proxy = model(ids, mask, eg, da, y)

        probs = torch.softmax(logits.float(), dim=-1)
        losses.append(loss.item())
        all_preds.extend(torch.argmax(logits, 1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        all_probs_dem.extend(probs[:, 1].cpu().numpy())
        if proxy is not None:
            all_proxy.append(proxy.cpu().numpy())

    m = compute_metrics(all_labels, all_preds, probs_dem=np.array(all_probs_dem))
    m["loss"]      = float(np.mean(losses))
    m["composite"] = 0.5 * m["f1_macro"] + 0.5 * m["f1_weighted"]

    if all_proxy:
        w_mean = np.concatenate(all_proxy, axis=0).mean(axis=0)
        m["attn_proxy_text"]   = float(w_mean[0])
        m["attn_proxy_egemap"] = float(w_mean[1])
        m["attn_proxy_deep"]   = float(w_mean[2])
    else:
        m["attn_proxy_text"] = m["attn_proxy_egemap"] = m["attn_proxy_deep"] = None

    return m


# ════════════════════════════════════════════════════════════════════════════════
# TRAINING
# ════════════════════════════════════════════════════════════════════════════════

def train_model(
    mode: str,
    train_df, tr_eg, tr_da,
    tokenizer, device, hp: dict, seed: int,
    mamba_backbone=None,
    ckpt_key: str = "",
    fold: int = 0,
    tag: str = "",
) -> nn.Module:
    set_seed(seed)
    n_train = len(train_df)

    n_val = max(20, int(hp["val_frac"] * n_train))
    n_val = min(n_val, n_train // 3)
    try:
        sss     = StratifiedShuffleSplit(n_splits=1, test_size=n_val, random_state=seed)
        tr_pos, val_pos = next(sss.split(np.zeros(n_train), train_df["label_id"].values))
    except ValueError:
        val_pos = np.random.choice(n_train, n_val, replace=False)
        tr_pos  = np.setdiff1d(np.arange(n_train), val_pos)

    tr_df_i = train_df.iloc[tr_pos].reset_index(drop=True)
    va_df_i = train_df.iloc[val_pos].reset_index(drop=True)
    tr_eg_i = tr_eg[tr_pos];  va_eg_i = tr_eg[val_pos]
    tr_da_i = tr_da[tr_pos];  va_da_i = tr_da[val_pos]

    y_tr    = tr_df_i["label_id"].values
    cw      = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr)
    class_w = np.clip(cw, 0.5, 5.0).tolist()

    model = MultiModeFusionEngine(
        mode=mode,
        mamba_model=copy.deepcopy(mamba_backbone) if mamba_backbone else None,
        egemap_dim=tr_eg.shape[1],
        deep_audio_dim=tr_da.shape[1],
        class_weights=class_w,
    ).to(device)

    if model.criterion.weight is not None:
        model.criterion.weight = model.criterion.weight.to(device)

    if mode in _TEXT_MODES:
        param_groups = [
            {"params": model.mamba.parameters(),
             "lr": hp["lr_backbone"], "weight_decay": hp["wd_backbone"]},
            {"params": [p for n, p in model.named_parameters() if "mamba" not in n],
             "lr": hp["lr_head"],     "weight_decay": hp["wd_head"]},
        ]
    else:
        param_groups = [
            {"params": model.parameters(),
             "lr": hp["lr_head"], "weight_decay": hp["wd_head"]},
        ]

    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp["epochs"])
    scaler    = GradScaler(enabled=(device.type == "cuda"))

    tr_ds = TrimodalDataset(tr_df_i, tokenizer, tr_eg_i, tr_da_i, hp["max_len"])
    va_ds = TrimodalDataset(va_df_i, tokenizer, va_eg_i, va_da_i, hp["max_len"])
    tr_ld = DataLoader(tr_ds, batch_size=hp["batch_size"],
                       shuffle=True, num_workers=0, pin_memory=True)
    va_ld = DataLoader(va_ds, batch_size=hp["batch_size"] * 2)

    best_composite, best_state, patience_ctr = -1.0, copy.deepcopy(model.state_dict()), 0

    for epoch in range(hp["epochs"]):
        model.train()
        for batch in tr_ld:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            eg   = batch["egemap"].to(device)
            da   = batch["deep_audio"].to(device)
            y    = batch["labels"].to(device)
            optimizer.zero_grad()
            with autocast(enabled=(device.type == "cuda")):
                loss, _, _ = model(ids, mask, eg, da, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()
        val_m     = evaluate(model, va_ld, device)
        composite = val_m["composite"]

        log_epoch(mode, tag, seed, fold, epoch + 1, val_m)

        improved = composite > best_composite
        if improved:
            best_composite = composite
            best_state     = copy.deepcopy(model.state_dict())
            patience_ctr   = 0
        elif epoch >= hp["min_epochs"]:
            patience_ctr += 1
            if patience_ctr >= hp["patience"]:
                print(f"    Early stop at epoch {epoch+1} "
                      f"(best composite={best_composite:.4f})")
                break

        if (epoch + 1) % 5 == 0 or improved:
            print(f"    Ep {epoch+1:>2}/{hp['epochs']}  "
                  f"comp={composite:.4f}  "
                  f"f1w={val_m['f1_weighted']:.4f}  "
                  f"f1m={val_m['f1_macro']:.4f}  "
                  f"sens={val_m['sensitivity']:.4f}  "
                  f"spec={val_m['specificity']:.4f}"
                  + (" ✓" if improved else ""))

    model.load_state_dict(best_state)
    return model


# ════════════════════════════════════════════════════════════════════════════════
# UNIFIED POOL RUNNER
# ════════════════════════════════════════════════════════════════════════════════

def run_pool(
    mode: str, meta, egemap_all, deep_all,
    tokenizer, device, hp: dict, seed: int,
    mamba_backbone=None,
) -> dict:
    set_seed(seed)
    sgkf    = StratifiedGroupKFold(n_splits=hp["pool_folds"], shuffle=True, random_state=seed)
    groups  = meta["participant_id"].values
    labels  = meta["label_id"].values
    datasets = meta["dataset"].unique()

    fold_metrics = {ds: [] for ds in datasets}

    for fold, (tr_idx, te_idx) in enumerate(sgkf.split(meta, labels, groups=groups)):
        print(f"\n  ── Pool fold {fold+1}/{hp['pool_folds']} ─────────────────────────────")
        tr_df = meta.iloc[tr_idx].reset_index(drop=True)
        te_df = meta.iloc[te_idx].reset_index(drop=True)

        eg_sc = StandardScaler()
        da_sc = StandardScaler()
        tr_eg = eg_sc.fit_transform(egemap_all[tr_idx])
        te_eg = eg_sc.transform(egemap_all[te_idx])
        tr_da = da_sc.fit_transform(deep_all[tr_idx])
        te_da = da_sc.transform(deep_all[te_idx])

        model = train_model(
            mode, tr_df, tr_eg, tr_da,
            tokenizer, device, hp, seed,
            mamba_backbone=mamba_backbone,
            ckpt_key=f"pool_{mode}_s{seed}",
            fold=fold + 1,
            tag="POOL",
        )

        for ds in datasets:
            ds_mask  = te_df["dataset"] == ds
            if ds_mask.sum() == 0:
                continue
            ds_te_df = te_df[ds_mask].reset_index(drop=True)
            ds_te_eg = te_eg[ds_mask.values]
            ds_te_da = te_da[ds_mask.values]
            ds_ld    = DataLoader(
                TrimodalDataset(ds_te_df, tokenizer, ds_te_eg, ds_te_da, hp["max_len"]),
                batch_size=hp["batch_size"] * 2
            )
            m = evaluate(model, ds_ld, device)
            fold_metrics[ds].append(m)

        del model
        torch.cuda.empty_cache()

    result_rows = []
    for ds, folds in fold_metrics.items():
        if not folds:
            continue
        row = {"mode": mode, "tag": "POOL", "seed": seed,
               "train_datasets": "all", "test_dataset": ds,
               "n_train": "CV", "n_test": "CV",
               "attn_proxy_text": None, "attn_proxy_egemap": None,
               "attn_proxy_deep": None}
        for k in METRIC_COLS:
            vals = [f[k] for f in folds if not np.isnan(f.get(k, np.nan))]
            row[k] = float(np.mean(vals)) if vals else np.nan
        for pw in ("attn_proxy_text", "attn_proxy_egemap", "attn_proxy_deep"):
            vals = [f[pw] for f in folds if f.get(pw) is not None]
            row[pw] = float(np.mean(vals)) if vals else None
        result_rows.append(row)

    return result_rows


# ════════════════════════════════════════════════════════════════════════════════
# DATA LOADING  [F11: feature_csv parameter]
# ════════════════════════════════════════════════════════════════════════════════

def load_all_data(feature_csv="master_acoustic_features.csv"):
    """
    Load and align all modalities.

    Parameters
    ----------
    feature_csv : str
        Filename of the eGeMAPS feature CSV, relative to DATA_DIR.
        Defaults to 'master_acoustic_features.csv' (baseline).
        Pass 'master_acoustic_features_robust.csv' (or any other name)
        for ablation runs without editing any other code.
    """
    print("\n─── Loading data ────────────────────────────────────────────────")
    print(f"    eGeMAPS CSV : {feature_csv}")

    meta = pd.read_csv(os.path.join(DATA_DIR, "master_metadata_cleaned.csv"))

    with open(os.path.join(DATA_DIR, "cleaned_transcripts.json")) as f:
        transcripts = json.load(f)["transcripts"]

    meta["text"] = meta["audio_path"].map(transcripts)
    meta = (meta.dropna(subset=["text"])
                .query("label in ['Control','Dementia']")
                .reset_index(drop=True))
    meta["label_id"] = meta["label"].map({"Control": 0, "Dementia": 1})

    # eGeMAPS — robust join (benchmark.py pattern)
    eg_df   = pd.read_csv(os.path.join(DATA_DIR, feature_csv))
    eg_cols = [c for c in eg_df.columns
               if c not in ("participant_id", "audio_path", "label",
                            "dataset", "split", "age", "gender", "mmse")]
    jcol    = "audio_path" if "audio_path" in eg_df.columns else "participant_id"
    meta    = pd.merge(meta, eg_df[[jcol] + eg_cols],
                       on=jcol, how="inner").reset_index(drop=True)

    # Deep audio
    ids_path = os.path.join(DEEP_DIR, "participant_ids.npy")
    if not os.path.exists(ids_path):
        raise FileNotFoundError(
            "Deep-audio embeddings not found.\nRun: python deep_audio_extractor.py"
        )
    da_ids = np.load(ids_path, allow_pickle=True)
    w2v    = np.load(os.path.join(DEEP_DIR, "wav2vec2_embeddings.npy"))
    hub    = np.load(os.path.join(DEEP_DIR, "hubert_embeddings.npy"))
    da_map = {pid: i for i, pid in enumerate(da_ids)}

    da_idx = meta["participant_id"].map(da_map)
    valid  = da_idx.notna()
    meta   = meta[valid].reset_index(drop=True)
    da_idx = da_idx[valid].astype(int).values

    egemap_all = meta[eg_cols].values.astype(np.float32)
    deep_all   = np.hstack([w2v[da_idx], hub[da_idx]]).astype(np.float32)

    print(f"    Aligned N      : {len(meta)}")
    print(f"    eGeMAPS dim    : {egemap_all.shape[1]}")
    print(f"    Deep-audio dim : {deep_all.shape[1]}")
    print(meta.groupby(["dataset", "label"]).size().to_string())
    return meta, egemap_all, deep_all, eg_cols


# ════════════════════════════════════════════════════════════════════════════════
# REPORTING
# ════════════════════════════════════════════════════════════════════════════════

HDR = ["Accuracy", "F1-W   ", "F1-M   ", "AUC-ROC", "Brier  ",
       "MCC    ", "Sensitiv", "Specific", "Prec-D ", "Prec-C "]

def _hdr_str():
    return "  ".join(f"{h:>8}" for h in HDR)

def _fmt_row(r, prefix=""):
    m_str = "  ".join(
        f"{r[f'{k}_mean']:>8.4f}" if not np.isnan(r.get(f"{k}_mean", np.nan))
        else f"{'NaN':>8}"
        for k in METRIC_COLS
    )
    s_str = "  ".join(
        f"±{r[f'{k}_std']:.3f}  " if not np.isnan(r.get(f"{k}_std", np.nan))
        else f"  {'—':>5}  "
        for k in METRIC_COLS
    )
    return m_str, s_str

def generate_report(summary_df: pd.DataFrame):
    sep   = "=" * 118
    lines = [
        sep,
        "  LODO BENCHMARK v4.1  —  9 LODO Splits + Unified Pool × 7 Architectures × 3 Seeds",
        "  Fixes: masked-mean pooling | pairwise interaction | class-weighted CE |",
        "          nan AUC | stratified val split | AMP | checkpointing | epoch logging",
        "  ⚠ attn_proxy_* columns are correlation proxies, NOT causal importance scores",
        sep,
    ]

    multi    = [c for c in LODO_COMBOS if not c["pool"] and len(c["train"]) > 1]
    single   = [c for c in LODO_COMBOS if not c["pool"] and len(c["train"]) == 1]
    pool_c   = [c for c in LODO_COMBOS if c["pool"]]

    def _block(combos, label):
        lines.append(f"\n  ── {label} " + "─" * 60)
        for combo in combos:
            tag     = combo["tag"]
            is_pool = combo["pool"]
            sub     = summary_df[summary_df["tag"] == tag]

            if is_pool:
                target_datasets = sub["test_dataset"].dropna().unique() if "test_dataset" in sub.columns else []
                for tds in sorted(target_datasets):
                    lines += [
                        "",
                        f"  [POOL → {tds.upper()}]  5-fold CV across all datasets",
                        "  " + "─" * 113,
                        f"  {'Architecture':<22}  {_hdr_str()}",
                        "  " + "─" * 113,
                    ]
                    sub_t = sub[sub["test_dataset"] == tds]
                    for mode in MODES:
                        row = sub_t[sub_t["mode"] == mode]
                        if row.empty:
                            lines.append(f"  {mode:<22}  (not run)")
                            continue
                        r = row.iloc[0]
                        m_str, s_str = _fmt_row(r)
                        lines.append(f"  {mode:<22}  {m_str}")
                        lines.append(f"  {'(±std)':<22}  {s_str}")
                    lines.append("  " + "─" * 113)
            else:
                lines += [
                    "",
                    f"  [{tag}]  Train: {combo['train']}  →  Test: {combo['test']}",
                    "  " + "─" * 113,
                    f"  {'Architecture':<22}  {_hdr_str()}",
                    "  " + "─" * 113,
                ]
                for mode in MODES:
                    row = sub[sub["mode"] == mode]
                    if row.empty:
                        lines.append(f"  {mode:<22}  (not run)")
                        continue
                    r = row.iloc[0]
                    m_str, s_str = _fmt_row(r)
                    lines.append(f"  {mode:<22}  {m_str}")
                    lines.append(f"  {'(±std)':<22}  {s_str}")
                lines.append("  " + "─" * 113)

    _block(multi,  "MULTI-DOMAIN (2→1)")
    _block(single, "SINGLE-DOMAIN (1→1)")
    _block(pool_c, "UNIFIED POOL (5-fold CV — upper bound reference)")

    lines += ["", sep, "  BEST ARCHITECTURE PER SPLIT  (by mean AUC-ROC)", "─" * 85]
    for combo in LODO_COMBOS:
        tag = combo["tag"]
        sub = summary_df[summary_df["tag"] == tag]
        if sub.empty:
            continue
        if combo["pool"]:
            for tds in sub["test_dataset"].dropna().unique():
                sub_t = sub[sub["test_dataset"] == tds]
                if sub_t.empty:
                    continue
                aucs = sub_t["auc_roc_mean"].replace(np.nan, -1)
                if aucs.max() < 0:
                    continue
                best = sub_t.loc[aucs.idxmax()]
                lines.append(
                    f"  POOL→{tds:<10}  →  {best['mode']:<22}  "
                    f"AUC={best['auc_roc_mean']:.4f}  "
                    f"F1-W={best['f1_weighted_mean']:.4f}  "
                    f"MCC={best['mcc_mean']:.4f}"
                )
        else:
            aucs = sub["auc_roc_mean"].replace(np.nan, -1)
            if aucs.max() < 0:
                continue
            best = sub.loc[aucs.idxmax()]
            lines.append(
                f"  {tag:<6}  →  {best['mode']:<22}  "
                f"AUC={best['auc_roc_mean']:.4f}  "
                f"F1-W={best['f1_weighted_mean']:.4f}  "
                f"Sens={best['sensitivity_mean']:.4f}  "
                f"MCC={best['mcc_mean']:.4f}"
            )

    lodo_sub = summary_df[summary_df["tag"] != "POOL"]
    pool_sub = summary_df[summary_df["tag"] == "POOL"]
    if not lodo_sub.empty and not pool_sub.empty:
        lines += ["", sep,
                  "  LODO GAP vs UNIFIED POOL  (mean AUC-ROC — lower = harder transfer)",
                  "─" * 85]
        for mode in MODES:
            lodo_auc = lodo_sub[lodo_sub["mode"] == mode]["auc_roc_mean"].mean()
            pool_auc = pool_sub[pool_sub["mode"] == mode]["auc_roc_mean"].mean()
            if np.isnan(lodo_auc) or np.isnan(pool_auc):
                continue
            lines.append(
                f"  {mode:<22}  LODO={lodo_auc:.4f}  Pool={pool_auc:.4f}  "
                f"gap={pool_auc - lodo_auc:+.4f}"
            )

    lines += ["", sep,
              "  GLOBAL RANKING  (mean AUC-ROC — LODO splits only)", "─" * 70]
    if not lodo_sub.empty:
        ranked = (lodo_sub.groupby("mode")["auc_roc_mean"]
                  .mean().sort_values(ascending=False))
        for rank, (mode, auc) in enumerate(ranked.items(), 1):
            f1w = lodo_sub.groupby("mode")["f1_weighted_mean"].mean().get(mode, np.nan)
            mcc = lodo_sub.groupby("mode")["mcc_mean"].mean().get(mode, np.nan)
            lines.append(
                f"  {rank}. {mode:<22}  "
                f"AUC={auc:.4f}  F1-W={f1w:.4f}  MCC={mcc:.4f}"
            )

    lines += [
        "", sep,
        "  NOTE ON MODALITY PROXY WEIGHTS",
        "  ─────────────────────────────────────────────────────────────────────",
        "  attn_proxy_* values reflect average attention gate weights.",
        "  For fusion_gated: these are explicit softmax scores — more reliable.",
        "  For fusion_attn: these are averaged attention matrix rows — weaker signal.",
        "  Neither implies causal modality importance. See Jain & Wallace (2019).",
        sep,
    ]

    report = "\n".join(lines)
    print("\n" + report)
    with open(REPORT_FILE, "w") as fh:
        fh.write(report)
    print(f"\n✅ Report saved → {REPORT_FILE}")


# ════════════════════════════════════════════════════════════════════════════════
# MAIN LOOP  [F11: feature_csv forwarded from __main__]
# ════════════════════════════════════════════════════════════════════════════════

def run_lodo(hp: dict, selected_modes: list,
             feature_csv: str = "master_acoustic_features.csv"):
    setup_dirs()
    init_log()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device      : {device}")
    print(f"AMP         : {'enabled' if device.type == 'cuda' else 'disabled'}")
    print(f"Modes       : {selected_modes}")
    print(f"Feature CSV : {feature_csv}")
    print(f"Results     : {RESULTS_FILE}")
    print(f"LR backbone : {hp['lr_backbone']}  |  LR head : {hp['lr_head']}")
    print(f"Epochs      : {hp['epochs']} "
          f"(min={hp['min_epochs']}, patience={hp['patience']})")
    print(f"Pooling     : masked mean over non-padding tokens")
    print(f"Interaction : pairwise [t,a,d,t*a,t*d,a*d] → Linear")
    print(f"Loss        : CrossEntropyLoss(weight=class_weights)  [no sampler]")
    print(f"Val split   : stratified, frac={hp['val_frac']}, min=20")

    meta, egemap_all, deep_all, eg_cols = load_all_data(feature_csv=feature_csv)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    tokenizer.pad_token = tokenizer.eos_token

    mamba_backbone = None
    if any(m in _TEXT_MODES for m in selected_modes):
        print(f"\nLoading Mamba backbone ({MAMBA_ID}) …")
        mamba_backbone = MambaLMHeadModel.from_pretrained(MAMBA_ID)

    # Resume support
    all_rows: list = []
    done_keys: set = set()
    if os.path.exists(RESULTS_FILE):
        prev = pd.read_csv(RESULTS_FILE)
        done_keys = set(zip(prev["mode"], prev["tag"],
                            prev.get("seed", pd.Series([0]*len(prev))).astype(int),
                            prev.get("test_dataset", pd.Series([""]*len(prev)))))
        all_rows  = prev.to_dict("records")
        print(f"Resumed: {len(done_keys)} rows already complete.")

    def _is_done(mode, tag, seed, test_dataset=""):
        return (mode, tag, seed, test_dataset) in done_keys

    def _save_row(row):
        all_rows.append(row)
        done_keys.add((row["mode"], row["tag"], row["seed"],
                       row.get("test_dataset", "")))
        pd.DataFrame(all_rows).to_csv(RESULTS_FILE, index=False)

    # ── 9 LODO splits ─────────────────────────────────────────────────────────
    for combo in [c for c in LODO_COMBOS if not c["pool"]]:
        tag         = combo["tag"]
        train_names = combo["train"]
        test_name   = combo["test"]

        tr_mask  = meta["dataset"].isin(train_names)
        te_mask  = meta["dataset"] == test_name
        tr_idx   = np.where(tr_mask)[0]
        te_idx   = np.where(te_mask)[0]
        train_df = meta[tr_mask].reset_index(drop=True)
        test_df  = meta[te_mask].reset_index(drop=True)

        print(f"\n{'='*72}")
        print(f"  [{tag}]  Train={train_names}  Test={test_name}")
        print(f"  Train n={tr_mask.sum()}  {train_df.groupby('label').size().to_dict()}")
        print(f"  Test  n={te_mask.sum()}  {test_df.groupby('label').size().to_dict()}")
        print(f"{'='*72}")

        for mode in selected_modes:
            for seed_off in range(hp["n_seeds"]):
                seed = hp["base_seed"] + seed_off
                if _is_done(mode, tag, seed, test_name):
                    print(f"  ⏩ {mode:<22} | {tag} | seed={seed}")
                    continue

                print(f"\n  ── {mode:<22} | {tag} | seed={seed} ──────────────")

                eg_sc = StandardScaler()
                da_sc = StandardScaler()
                tr_eg = eg_sc.fit_transform(egemap_all[tr_idx])
                te_eg = eg_sc.transform(egemap_all[te_idx])
                tr_da = da_sc.fit_transform(deep_all[tr_idx])
                te_da = da_sc.transform(deep_all[te_idx])

                set_seed(seed)
                model = train_model(
                    mode, train_df, tr_eg, tr_da,
                    tokenizer, device, hp, seed,
                    mamba_backbone=(mamba_backbone if mode in _TEXT_MODES else None),
                    ckpt_key=f"{mode}_{tag}_s{seed}",
                    fold=0, tag=tag,
                )

                te_ds = TrimodalDataset(test_df, tokenizer, te_eg, te_da, hp["max_len"])
                te_ld = DataLoader(te_ds, batch_size=hp["batch_size"] * 2)
                m     = evaluate(model, te_ld, device)

                row = {
                    "mode": mode, "tag": tag, "seed": seed,
                    "train_datasets": str(train_names),
                    "test_dataset":   test_name,
                    "n_train": len(train_df), "n_test": len(test_df),
                    "attn_proxy_text":   m["attn_proxy_text"],
                    "attn_proxy_egemap": m["attn_proxy_egemap"],
                    "attn_proxy_deep":   m["attn_proxy_deep"],
                }
                row.update({k: m[k] for k in METRIC_COLS})
                _save_row(row)

                print(
                    f"\n  ✓ TEST  Acc={m['accuracy']:.4f}  AUC={m['auc_roc']:.4f}  "
                    f"MCC={m['mcc']:.4f}  F1-W={m['f1_weighted']:.4f}  "
                    f"Sens={m['sensitivity']:.4f}  Spec={m['specificity']:.4f}"
                )

                del model
                torch.cuda.empty_cache()

    # ── Unified Pool ───────────────────────────────────────────────────────────
    all_ds = meta["dataset"].unique().tolist()

    print(f"\n{'='*72}")
    print(f"  [POOL]  5-fold CV on ALL datasets  (upper-bound reference)")
    print(f"  N total = {len(meta)}  {meta.groupby('label').size().to_dict()}")
    print(f"{'='*72}")

    for mode in selected_modes:
        for seed_off in range(hp["n_seeds"]):
            seed = hp["base_seed"] + seed_off
            pool_done = all(_is_done(mode, "POOL", seed, ds) for ds in all_ds)
            if pool_done:
                print(f"  ⏩ Pool | {mode:<22} | seed={seed}")
                continue

            print(f"\n  ── Pool | {mode:<22} | seed={seed} ──────────────────")
            result_rows = run_pool(
                mode, meta, egemap_all, deep_all,
                tokenizer, device, hp, seed,
                mamba_backbone=(mamba_backbone if mode in _TEXT_MODES else None),
            )
            for r in result_rows:
                r.update({"mode": mode, "tag": "POOL", "seed": seed,
                           "train_datasets": "all"})
                _save_row(r)

    # ── Aggregate ──────────────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_rows)
    agg = []
    for (mode, tag, test_ds), grp in results_df.groupby(
            ["mode", "tag", "test_dataset"], dropna=False):
        combo = next((c for c in LODO_COMBOS if c["tag"] == tag), None)
        row   = {
            "mode": mode, "tag": tag,
            "test_dataset": test_ds,
            "train": str(combo["train"]) if combo else "all",
            "n_seeds": len(grp),
        }
        for k in METRIC_COLS:
            vals = grp[k].dropna()
            row[f"{k}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{k}_std"]  = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        for pw in ("attn_proxy_text", "attn_proxy_egemap", "attn_proxy_deep"):
            vals = grp[pw].dropna() if pw in grp.columns else pd.Series(dtype=float)
            row[f"{pw}_mean"] = float(vals.mean()) if len(vals) else None
        agg.append(row)

    summary_df = pd.DataFrame(agg)
    summary_df.to_csv(SUMMARY_FILE, index=False)
    print(f"\n✅ Per-seed results → {RESULTS_FILE}")
    print(f"✅ Summary          → {SUMMARY_FILE}")
    close_log()
    generate_report(summary_df)


# ════════════════════════════════════════════════════════════════════════════════
# ENTRY POINT  [F11: --feature_csv and --run_tag added]
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="LODO Benchmark v4.1 — all fixes applied + Unified Pool"
    )
    p.add_argument("--epochs",      type=int,   default=HPARAMS["epochs"])
    p.add_argument("--min_epochs",  type=int,   default=HPARAMS["min_epochs"])
    p.add_argument("--patience",    type=int,   default=HPARAMS["patience"])
    p.add_argument("--batch_size",  type=int,   default=HPARAMS["batch_size"])
    p.add_argument("--lr_backbone", type=float, default=HPARAMS["lr_backbone"])
    p.add_argument("--lr_head",     type=float, default=HPARAMS["lr_head"])
    p.add_argument("--wd_backbone", type=float, default=HPARAMS["wd_backbone"])
    p.add_argument("--wd_head",     type=float, default=HPARAMS["wd_head"])
    p.add_argument("--max_len",     type=int,   default=HPARAMS["max_len"])
    p.add_argument("--n_seeds",     type=int,   default=HPARAMS["n_seeds"])
    p.add_argument("--base_seed",   type=int,   default=HPARAMS["base_seed"])
    p.add_argument("--val_frac",    type=float, default=HPARAMS["val_frac"])
    p.add_argument("--pool_folds",  type=int,   default=HPARAMS["pool_folds"])
    p.add_argument("--modes", nargs="+", default=MODES,
                   help=f"Subset of modes. Choices: {MODES}")
    # ── Ablation support ──────────────────────────────────────────────────────
    p.add_argument(
        "--feature_csv",
        type=str,
        default="master_acoustic_features.csv",
        help=(
            "eGeMAPS feature CSV filename, relative to processed_data/. "
            "Defaults to 'master_acoustic_features.csv' (baseline). "
            "Example: --feature_csv master_acoustic_features_robust.csv"
        ),
    )
    p.add_argument(
        "--run_tag",
        type=str,
        default="",
        help=(
            "Short label appended to all four output files so baseline and "
            "ablation runs never collide and resumption works independently. "
            "Example: --run_tag ablation  →  lodo_results_ablation.csv"
        ),
    )
    # ─────────────────────────────────────────────────────────────────────────
    args = p.parse_args()

    # ── Dynamically reroute output filenames when a run_tag is given  [F11] ──
    if args.run_tag:
        import sys
        suffix = f"_{args.run_tag}"
        _mod = sys.modules[__name__]
        _mod.RESULTS_FILE = f"lodo_results{suffix}.csv"
        _mod.SUMMARY_FILE = f"lodo_summary{suffix}.csv"
        _mod.REPORT_FILE  = f"lodo_report{suffix}.txt"
        _mod.TRAIN_LOG    = os.path.join(LOG_DIR, f"training_log{suffix}.csv")
        print(f"[config] run_tag       : '{args.run_tag}'")
        print(f"[config] Results file  : {_mod.RESULTS_FILE}")
        print(f"[config] Summary file  : {_mod.SUMMARY_FILE}")

    print(f"[config] feature_csv   : {args.feature_csv}")

    hp = {k: getattr(args, k) for k in HPARAMS}
    selected = [m for m in args.modes if m in MODES]
    if not selected:
        raise ValueError(f"No valid modes. Choose from {MODES}")

    run_lodo(hp, selected, feature_csv=args.feature_csv)
