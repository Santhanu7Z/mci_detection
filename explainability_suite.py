"""
explainability_suite.py — v6.0
Fixes applied:
  [F1] Mean-embedding replacement (not zero — zero is OOD in embedding space)
  [F2] Monte Carlo masking (5 random-subset samples per word — approximates
       marginal attribution, reduces interaction bias)
  [F3] Length-normalised importance (mean pooling dilutes long transcripts)
  [F4] Wilcoxon signed-rank test (non-parametric, robust to skew)
  [F5] Strict token containment in offset mapping (no double-masking)
  [F6] Confidence-calibrated importance: Δp / (p*(1-p))

Kept from v5:
  Embedding-level masking, offset-mapping, BH-FDR, paired Cohen's d,
  per-dataset breakdown, CI everywhere, explicit modality caveat.
Label encoding: Control=0, Dementia=1
"""

import os
import re
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, mannwhitneyu, wilcoxon
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from tqdm import tqdm

try:
    from statsmodels.stats.multitest import multipletests
    STATSMODELS_OK = True
except ImportError:
    STATSMODELS_OK = False
    print("⚠️  statsmodels missing. pip install statsmodels")

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_OK = True
except Exception:
    SPACY_OK = False
    print("⚠️  spaCy missing. pip install spacy && python -m spacy download en_core_web_sm")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(0)
torch.manual_seed(42)

_OrigCudaDevice = torch.cuda.device
class _EmbedOverride(nn.Module):
    """Thin nn.Module wrapper that returns pre-computed embeddings,
    ignoring the input_ids. Used to inject modified embeddings into
    Mamba's MixerModel without changing its forward signature."""
    def __init__(self, precomputed):
        super().__init__()
        self._precomputed = precomputed          # [1, T, D] tensor

    def forward(self, _ids):
        return self._precomputed

MC_SAMPLES = 5   # Monte Carlo samples per word for F2


# ============================================================
# STATISTICS HELPERS
# ============================================================

def cohen_d_paired(a, b):
    diff = np.asarray(a) - np.asarray(b)
    sd   = diff.std(ddof=1)
    return diff.mean() / sd if sd > 1e-9 else 0.0


def cohen_d_independent(a, b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na-1)*a.std(ddof=1)**2 + (nb-1)*b.std(ddof=1)**2)
                     / (na + nb - 2))
    return (a.mean() - b.mean()) / pooled if pooled > 1e-9 else 0.0


def mean_ci95(arr):
    from scipy.stats import t as t_dist
    n, m = len(arr), arr.mean()
    se   = arr.std(ddof=1) / np.sqrt(n)
    half = t_dist.ppf(0.975, df=n-1) * se
    return m, m - half, m + half


def gate_entropy(g):
    g = np.clip(np.asarray(g), 1e-6, 1-1e-6)
    return -(g * np.log2(g) + (1-g) * np.log2(1-g))


# ============================================================
# CHECKPOINT DETECTION
# ============================================================

def detect_classifier_input_dim(sd):
    for k, v in sd.items():
        if "classifier.0.weight" in k:
            return v.shape[1]
    return 256


def detect_arch_flags(sd):
    keys       = list(sd.keys())
    is_old_seq = any("text_proj.0.weight" in k for k in keys)
    uses_fusion= any("fusion_layer.gate"  in k for k in keys)
    clf_dim    = detect_classifier_input_dim(sd)
    has_attn   = any("attention.in_proj_weight" in k for k in keys)
    tp1        = sd.get("text_proj.1.weight")
    tp1_type   = ("gelu" if tp1 is None else
                  "linear" if tp1.dim()==2 else "norm")
    ap3        = sd.get("audio_proj.3.weight")
    ap3_type   = ("gelu" if ap3 is None else
                  "linear" if ap3.dim()==2 else "norm")
    return is_old_seq, uses_fusion, clf_dim, has_attn, tp1_type, ap3_type


# ============================================================
# MODEL
# ============================================================

class MambaFusionEngineDynamic(nn.Module):
    def __init__(self, backbone, acoustic_dim,
                 is_old_seq, uses_fusion, clf_dim,
                 has_attn, tp1_type, ap3_type):
        super().__init__()
        self.mamba   = backbone.backbone
        self.fdim    = 256
        self.clf_dim = clf_dim

        slot1 = (nn.Linear(self.fdim, self.fdim) if tp1_type=="linear"
                 else nn.LayerNorm(self.fdim) if tp1_type=="norm"
                 else nn.GELU())
        self.text_proj = (nn.Sequential(
                              nn.Linear(backbone.config.d_model, self.fdim), slot1)
                          if is_old_seq else
                          nn.Linear(backbone.config.d_model, self.fdim))

        slot3 = (nn.Linear(self.fdim, self.fdim) if ap3_type=="linear"
                 else nn.LayerNorm(self.fdim) if ap3_type=="norm"
                 else nn.GELU())
        self.audio_proj = (nn.Sequential(
                               nn.Linear(acoustic_dim,128), nn.GELU(),
                               nn.Linear(128,self.fdim), slot3)
                           if is_old_seq else
                           nn.Sequential(nn.Linear(acoustic_dim,128), nn.GELU(),
                                         nn.Linear(128,self.fdim)))

        if uses_fusion:
            self.fusion_layer = nn.ModuleDict({"gate": nn.Sequential(
                nn.Linear(self.fdim*2,self.fdim), nn.GELU(),
                nn.Linear(self.fdim,1), nn.Sigmoid())})

        if has_attn:
            self.attention = nn.MultiheadAttention(
                self.fdim, num_heads=4, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(clf_dim,128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128,2))

    def forward(self, input_ids, acoustic_features, input_embeds=None):
        if input_embeds is not None:
            orig_emb = self.mamba.embedding
            self.mamba.embedding = _EmbedOverride(input_embeds)
            text_raw = self.mamba(input_ids)
            self.mamba.embedding = orig_emb
        else:
            text_raw = self.mamba(input_ids)

        text_out  = text_raw.mean(dim=1)           # mean pooling
        text_emb  = self.text_proj(text_out)
        audio_emb = self.audio_proj(acoustic_features)

        if self.clf_dim == 256:
            if hasattr(self, "fusion_layer"):
                g       = self.fusion_layer["gate"](
                              torch.cat([text_emb, audio_emb], dim=-1))
                fused   = g * text_emb + (1-g) * audio_emb
                weights = g.detach().cpu()
            else:
                fused, weights = text_emb * audio_emb, None
        else:
            combined         = torch.stack([text_emb, audio_emb], dim=1)
            attn_out, attn_w = self.attention(combined, combined, combined)
            if attn_w.dim()==4:
                attn_w = attn_w.mean(dim=1)
            fused   = torch.cat(
                [attn_out.flatten(start_dim=1), text_emb*audio_emb], dim=-1)
            weights = attn_w.detach().cpu()

        return self.classifier(fused), weights

    def get_input_embeddings(self):
        return self.mamba.embedding


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(checkpoint_path, acoustic_dim):
    print("🔍 Inspecting checkpoint...")
    state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    flags = detect_arch_flags(state)
    print(f"  {dict(zip(['old_seq','fusion','clf_dim','attn','tp1','ap3'], flags))}")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token = tokenizer.eos_token
    backbone = MambaLMHeadModel.from_pretrained("state-spaces/mamba-130m")
    model    = MambaFusionEngineDynamic(backbone, acoustic_dim, *flags).to(DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()
    return tokenizer, model


# ============================================================
# HELPERS — shared encoding + word grouping
# ============================================================

def encode_with_offsets(tokenizer, text):
    """Return (input_ids_tensor, offset_list, n_tokens)."""
    enc = tokenizer(
        text, return_tensors="pt", truncation=True,
        max_length=512, return_offsets_mapping=True)
    input_ids = enc["input_ids"].to(DEVICE)
    offsets   = enc["offset_mapping"][0].tolist()   # [(start,end), ...]
    return input_ids, offsets


def word_spans(text):
    """Return list of (char_start, char_end, word_lower) for non-space runs."""
    return [(m.start(), m.end(), m.group().lower())
            for m in re.finditer(r"\S+", text)]


def tokens_for_word(offsets, w_start, w_end):
    """
    FIX F5: strict containment — token must be fully inside the word span.
    Avoids double-masking punctuation/contraction edge cases.
    """
    return [i for i, (ts, te) in enumerate(offsets)
            if te > ts and ts >= w_start and te <= w_end]


# ============================================================
# FIX F1 — MEAN-EMBEDDING BASELINE
# Replace masked positions with the sequence mean embedding,
# not the zero vector. The mean is in-distribution (unlike 0)
# and neutral with respect to any particular token's identity.
# ============================================================

def mean_embed_baseline(base_embeds):
    """Shape [1, T, D] → mean over T, broadcast back to [1, 1, D]."""
    return base_embeds.mean(dim=1, keepdim=True)


# ============================================================
# FIX F2 — MONTE CARLO MASKED WORD IMPORTANCE
# For each word w, approximate marginal attribution:
#   E_S[ f(x) - f(x_{S ∪ w masked}) ]
# where S is a random subset of OTHER words also masked.
# This reduces interaction bias at low cost (MC_SAMPLES iters).
#
# FIX F3 — LENGTH NORMALISATION
# Multiply raw Δp by sequence_length so importance is not
# diluted by mean pooling over longer transcripts.
#
# FIX F6 — CONFIDENCE CALIBRATION
# Divide by p*(1-p) so importance is comparable across samples
# with different base probabilities.
# ============================================================

def get_word_importance(model, tokenizer, text, audio_feat):
    """
    Returns (words, importance_scores).
    Scores are: (mean Δp over MC_SAMPLES) * seq_len / (p*(1-p))
    """
    input_ids, offsets = encode_with_offsets(tokenizer, text)
    n_tokens           = input_ids.shape[1]
    audio_t            = torch.from_numpy(
        np.array([audio_feat], dtype=np.float32)).to(DEVICE)

    emb_table = model.get_input_embeddings()
    with torch.no_grad():
        base_embeds = emb_table(input_ids)              # [1, T, D]
        logits, _   = model(input_ids, audio_t)
        base_prob   = torch.softmax(logits, dim=1)[0,1].item()

    baseline    = mean_embed_baseline(base_embeds)      # FIX F1
    spans       = word_spans(text)
    all_indices = list(range(len(spans)))

    # FIX F6: calibration denominator (clip for numerical safety)
    calib = max(base_prob * (1 - base_prob), 1e-4)

    words, scores = [], []

    for w_idx, (w_start, w_end, word) in enumerate(spans):
        if len(word) <= 2:
            continue

        tok_indices = tokens_for_word(offsets, w_start, w_end)  # FIX F5
        if not tok_indices:
            continue

        other_indices = [j for j in all_indices if j != w_idx]

        delta_list = []
        for _ in range(MC_SAMPLES):                     # FIX F2
            # Random subset of OTHER words to also mask (~20% of remaining)
            n_extra = max(0, int(0.2 * len(other_indices)))
            extra   = np.random.choice(other_indices, size=n_extra,
                                       replace=False).tolist() if n_extra else []

            masked_embeds = base_embeds.clone()

            # Mask word of interest
            masked_embeds[0, tok_indices, :] = baseline  # FIX F1

            # Mask random context words
            for oi in extra:
                ots, ote, _ = spans[oi]
                extra_toks  = tokens_for_word(offsets, ots, ote)
                if extra_toks:
                    masked_embeds[0, extra_toks, :] = baseline

            with torch.no_grad():
                logits, _ = model(input_ids, audio_t,
                                  input_embeds=masked_embeds)
                m_prob    = torch.softmax(logits, dim=1)[0,1].item()

            delta_list.append(base_prob - m_prob)

        raw_imp   = np.mean(delta_list)
        norm_imp  = raw_imp * n_tokens / calib           # FIX F3 + F6
        words.append(word)
        scores.append(norm_imp)

    return words, scores


# ============================================================
# FEATURE-CATEGORY PERTURBATION  (mean-embed baseline, FIX F1+F5)
# Caveat: text-marginal importance, audio held fixed.
# ============================================================

DISFLUENCY_RE     = re.compile(r"\b(uh+|um+|er+|ah+|mm+|hmm+)\b", re.I)
DISCOURSE_MARKERS = {"well","so","and","but","because","then","now","like",
                     "okay","right","you","know","mean"}

def get_feature_perturbation(model, tokenizer, text, audio_feat):
    input_ids, offsets = encode_with_offsets(tokenizer, text)
    audio_t            = torch.from_numpy(
        np.array([audio_feat], dtype=np.float32)).to(DEVICE)

    emb_table = model.get_input_embeddings()
    with torch.no_grad():
        base_embeds = emb_table(input_ids)
        logits, _   = model(input_ids, audio_t)
        base_prob   = torch.softmax(logits, dim=1)[0,1].item()

    baseline = mean_embed_baseline(base_embeds)          # FIX F1
    calib    = max(base_prob * (1 - base_prob), 1e-4)    # FIX F6

    categories = ["disfluency", "discourse"]
    if SPACY_OK:
        categories += ["pronoun", "verb", "noun"]

    results = {}
    for cat in categories:
        masked_embeds = base_embeds.clone()

        if cat == "disfluency":
            for m in re.finditer(DISFLUENCY_RE, text):
                for i, (ts, te) in enumerate(offsets):
                    # FIX F5: strict containment
                    if te > ts and ts >= m.start() and te <= m.end():
                        masked_embeds[0, i, :] = baseline

        elif cat == "discourse":
            for m in re.finditer(r"\S+", text):
                if m.group().lower() in DISCOURSE_MARKERS:
                    for i, (ts, te) in enumerate(offsets):
                        if te > ts and ts >= m.start() and te <= m.end():
                            masked_embeds[0, i, :] = baseline

        elif SPACY_OK:
            POS = {"pronoun":"PRON", "verb":"VERB", "noun":"NOUN"}
            doc = nlp(text)
            for tok in doc:
                if tok.pos_ == POS.get(cat,""):
                    ts_word = tok.idx
                    te_word = tok.idx + len(tok.text)
                    for i, (ts, te) in enumerate(offsets):
                        if te > ts and ts >= ts_word and te <= te_word:
                            masked_embeds[0, i, :] = baseline

        with torch.no_grad():
            logits, _ = model(input_ids, audio_t,
                              input_embeds=masked_embeds)
            m_prob    = torch.softmax(logits, dim=1)[0,1].item()

        # FIX F3 + F6: length-scale and calibrate
        raw_imp = base_prob - m_prob
        results[cat] = raw_imp * input_ids.shape[1] / calib

    return results


# ============================================================
# LINGUISTIC FEATURE EXTRACTION
# ============================================================

def extract_linguistic_features(text):
    feats      = {}
    raw_tokens = text.split()
    n          = max(len(raw_tokens), 1)
    feats["disfluency_rate"] = len(DISFLUENCY_RE.findall(text)) / n
    reps = sum(1 for a, b in zip(raw_tokens, raw_tokens[1:])
               if a.lower() == b.lower())
    feats["repetition_rate"] = reps / n
    if SPACY_OK:
        doc   = nlp(text)
        toks  = [t for t in doc if not t.is_space]
        nt    = max(len(toks), 1)
        lems  = [t.lemma_.lower() for t in toks if t.is_alpha]
        feats["ttr"]             = len(set(lems)) / max(len(lems), 1)
        feats["pronoun_ratio"]   = sum(1 for t in toks if t.pos_=="PRON") / nt
        feats["verb_ratio"]      = sum(1 for t in toks if t.pos_=="VERB") / nt
        feats["content_density"] = sum(1 for t in toks
                                       if t.pos_ in ("NOUN","VERB","ADJ","ADV")) / nt
        sents = list(doc.sents)
        feats["mean_sent_len"]   = np.mean([len(s) for s in sents]) if sents else 0.
        feats["unique_nouns"]    = len({t.lemma_.lower() for t in toks if t.pos_=="NOUN"})
    else:
        words = [w.lower() for w in raw_tokens if w.isalpha()]
        feats["ttr"] = len(set(words)) / max(len(words), 1)
        PRON = {"i","he","she","it","they","we","you","this","that",
                "these","those","them","him","her","us"}
        feats["pronoun_ratio"] = sum(1 for w in words if w in PRON) / max(len(words),1)
    feats["total_words"] = n
    return feats


# ============================================================
# MAIN XAI PIPELINE
# ============================================================

def run_xai(df, acoustic_data, tokenizer, model, n_samples=200):

    print(f"\n{'='*65}")
    print(f"XAI v6.0  —  n={n_samples}  MC_SAMPLES={MC_SAMPLES}")
    print(f"{'='*65}")
    print("F1: Mean-embed baseline (not zero)")
    print("F2: Monte Carlo masking per word (interaction bias ↓)")
    print("F3: Length-normalised importance")
    print("F4: Wilcoxon signed-rank (non-parametric)")
    print("F5: Strict token containment in offset mapping")
    print("F6: Confidence-calibrated Δp / p(1-p)")

    os.makedirs("xai_results", exist_ok=True)

    # ── Stratified sample: equal Control / Dementia, shuffled ────────────────
    ctrl_idx = df[df["label"] == "Control"].index.tolist()
    dem_idx  = df[df["label"] == "Dementia"].index.tolist()
    np.random.seed(42)
    np.random.shuffle(ctrl_idx)
    np.random.shuffle(dem_idx)
    n_each   = min(n_samples // 2, len(ctrl_idx), len(dem_idx))
    sel_idx  = sorted(ctrl_idx[:n_each] + dem_idx[:n_each])
    df_sel   = df.loc[sel_idx].reset_index(drop=True)
    ac_sel   = acoustic_data[sel_idx]
    print(f"\nStratified sample: {n_each} Control + {n_each} Dementia "
          f"= {len(df_sel)} total")
    print(df_sel.groupby(["dataset","label"]).size().to_string())

    text_weights, audio_weights, gate_vals = [], [], []
    word_records, ling_records, feat_pert  = [], [], []
    gate_mode = False

    for i in tqdm(range(len(df_sel))):
        row   = df_sel.iloc[i]
        text  = str(row["text"])
        audio = ac_sel[i]
        label = row.get("label", None)
        dset  = row.get("dataset", "unknown")

        inputs    = tokenizer(text, return_tensors="pt",
                              truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(DEVICE)
        audio_t   = torch.from_numpy(
            np.array([audio], dtype=np.float32)).to(DEVICE)

        with torch.no_grad():
            _, weights = model(input_ids, audio_t)

        if weights is not None:
            w = weights.squeeze()
            if w.dim() == 0:
                gate_mode = True
                gv = w.item()
                gate_vals.append(gv)
                text_weights.append(gv)
                audio_weights.append(1.0 - gv)
            else:
                attn = w.mean(dim=0).numpy()
                attn /= attn.sum() + 1e-9
                text_weights.append(float(attn[0]))
                audio_weights.append(float(attn[1]))

        words, imps = get_word_importance(model, tokenizer, text, audio)
        for word, imp in zip(words, imps):
            word_records.append({"word":word,"importance":imp,
                                 "label":label,"dataset":dset})

        fp = get_feature_perturbation(model, tokenizer, text, audio)
        fp.update({"label":label,"dataset":dset,"sample_idx":i})
        feat_pert.append(fp)

        lf = extract_linguistic_features(text)
        lf.update({"label":label,"dataset":dset,"sample_idx":i})
        ling_records.append(lf)

    # =================================================================
    # A — MODALITY CONTRIBUTION
    # =================================================================
    if text_weights:
        t_arr = np.array(text_weights)
        a_arr = np.array(audio_weights)
        t_stat, p_val = ttest_rel(t_arr, a_arr)
        d_paired      = cohen_d_paired(t_arr, a_arr)
        tm, tlo, thi  = mean_ci95(t_arr)
        am, alo, ahi  = mean_ci95(a_arr)
        H             = gate_entropy(gate_vals) if gate_mode else None

        print(f"\n{'='*65}")
        print("A — MODALITY CONTRIBUTION")
        print(f"{'='*65}")
        print(f"Text  : {tm:.4f}  95%CI [{tlo:.4f}, {thi:.4f}]")
        print(f"Audio : {am:.4f}  95%CI [{alo:.4f}, {ahi:.4f}]")
        print(f"Ratio : {tm/am:.2f}x text-dominant")
        print(f"Paired t   : t={t_stat:.3f}, p={p_val:.2e}")
        print(f"Paired d   : {d_paired:.3f}")
        if H is not None:
            pct_low = (H < 0.5).mean() * 100
            print(f"Gate entropy : {H.mean():.4f} ± {H.std():.4f}")
            print(f"  {pct_low:.1f}% of samples have H<0.5")
            print("  → Modality dominance consistent with linguistic "
                  "signal primacy in this task.")

        mod_df = pd.DataFrame({
            "text_weight":  t_arr, "audio_weight": a_arr,
            "gate_entropy": H if H is not None else np.zeros(len(t_arr)),
            "label":   df_sel.iloc[:len(t_arr)]["label"].values,
            "dataset": df_sel.iloc[:len(t_arr)]["dataset"].values,
        })
        mod_df.to_csv("xai_results/modality_weights.csv", index=False)

        datasets = mod_df["dataset"].unique()
        fig, axes = plt.subplots(1, 1+len(datasets),
                                 figsize=(5*(1+len(datasets)),4), sharey=True)
        axes = np.atleast_1d(axes)

        def _bar(ax, tw, aw, title):
            m = [tw.mean(), aw.mean()]; e = [tw.std(), aw.std()]
            bars = ax.bar(["Text","Audio"], m, yerr=e,
                          color=["#4C72B0","#DD8452"], capsize=6, width=0.5)
            ax.set_ylim(0,1); ax.set_title(title, fontsize=9)
            for b,v in zip(bars,m):
                ax.text(b.get_x()+b.get_width()/2, v+0.02,
                        f"{v:.3f}", ha="center", fontsize=9)

        _bar(axes[0], t_arr, a_arr, f"Overall (n={len(t_arr)})")
        for ax, ds in zip(axes[1:], datasets):
            sub = mod_df[mod_df["dataset"]==ds]
            _bar(ax, sub["text_weight"].values,
                 sub["audio_weight"].values, f"{ds}\n(n={len(sub)})")
        axes[0].set_ylabel("Mean Fusion Weight")
        fig.suptitle(f"Modality Contribution — Paired d={d_paired:.2f}, p={p_val:.2e}",
                     y=1.03)
        plt.tight_layout()
        plt.savefig("xai_results/modality_weights.png", dpi=300,
                    bbox_inches="tight")
        plt.close()
        print("Saved → xai_results/modality_weights.png")

    # =================================================================
    # B — LINGUISTIC FEATURES  (BH-FDR + independent Cohen's d)
    # =================================================================
    ling_df = pd.DataFrame(ling_records)
    ling_df.to_csv("xai_results/linguistic_features.csv", index=False)
    feat_cols_ling = [c for c in ling_df.columns
                      if c not in ("label","dataset","sample_idx")]
    ctrl = ling_df[ling_df["label"]=="Control"]
    dem  = ling_df[ling_df["label"]=="Dementia"]

    raw_stats = []
    print(f"  Control samples : {len(ctrl)}  Dementia samples : {len(dem)}")
    for feat in feat_cols_ling:
        cv = ctrl[feat].dropna().values
        dv = dem[feat].dropna().values
        if len(cv) < 2 or len(dv) < 2:
            continue
        _, p  = mannwhitneyu(cv, dv, alternative="two-sided")
        d_ind = cohen_d_independent(dv, cv)
        raw_stats.append({"feature":feat,
                           "ctrl_mean":cv.mean(),"dem_mean":dv.mean(),
                           "p_raw":p,"cohen_d":d_ind})

    stat_df = pd.DataFrame(raw_stats)
    if stat_df.empty:
        print("⚠️  Not enough samples per group for linguistic stats — skipping.")
        stat_df["p_adj"] = pd.Series(dtype=float)
        stat_df["sig"]   = pd.Series(dtype=str)
    else:
        if STATSMODELS_OK:
            _, p_adj, _, _ = multipletests(stat_df["p_raw"].values, method="fdr_bh")
            stat_df["p_adj"] = p_adj
        else:
            stat_df["p_adj"] = stat_df["p_raw"]
        stat_df["sig"] = stat_df["p_adj"].apply(
            lambda p: "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns")
    stat_df.to_csv("xai_results/linguistic_stats.csv", index=False)

    print(f"\n{'='*65}")
    print("B — LINGUISTIC FEATURES  (BH-FDR corrected)")
    print(f"{'='*65}")
    if stat_df.empty:
        print("⚠️  No features computed — check spaCy installation.")
    else:
        print(f"{'Feature':<22} {'Ctrl':>7} {'Dem':>7} "
              f"{'p_raw':>9} {'p_adj':>9} {'d':>6}  sig")
        print("-"*65)
        for _, row in stat_df.sort_values("p_adj").iterrows():
            print(f"{row.feature:<22} {row.ctrl_mean:>7.4f} {row.dem_mean:>7.4f} "
                  f"{row.p_raw:>9.4f} {row.p_adj:>9.4f} "
                  f"{row.cohen_d:>6.3f}  {row.sig}")
        n_sig = (stat_df["sig"] != "ns").sum()
        print(f"\n  {n_sig}/{len(stat_df)} features significant after BH-FDR.")
        if n_sig == 0:
            print("  (Raw p-values saved — differences may be real but underpowered.)")

    sig_df = stat_df[stat_df["sig"]!="ns"].copy()
    if not sig_df.empty:
        sig_df = sig_df.sort_values("p_adj")
        sig_df["delta"] = sig_df["dem_mean"] - sig_df["ctrl_mean"]
        fig, ax = plt.subplots(figsize=(10, max(4, len(sig_df)*0.55)))
        colors  = ["#d62728" if d>0 else "#1f77b4" for d in sig_df["delta"]]
        ax.barh(sig_df["feature"], sig_df["delta"], color=colors)
        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.invert_yaxis()
        for j, (_, row) in enumerate(sig_df.iterrows()):
            ax.text(row.delta+0.001*np.sign(row.delta), j,
                    f"{row.sig} d={row.cohen_d:.2f}", va="center", fontsize=8)
        ax.set_xlabel("Δ Mean (Dementia−Control) — red=higher in Dementia")
        ax.set_title("Significant Linguistic Features "
                     "(Mann-Whitney U + BH-FDR, independent Cohen's d)")
        plt.tight_layout()
        plt.savefig("xai_results/linguistic_features.png", dpi=300)
        plt.close()
        print("Saved → xai_results/linguistic_features.png")
    else:
        print("⚠️  No features survive FDR correction.")

    print(f"\n{'='*65}")
    print("B2 — PER-DATASET LINGUISTIC BREAKDOWN")
    print(f"{'='*65}")
    for ds in ling_df["dataset"].unique():
        sub = ling_df[ling_df["dataset"]==ds]
        print(f"\n  {ds}  Control={(sub['label']=='Control').sum()}  "
              f"Dementia={(sub['label']=='Dementia').sum()}")
        for feat in feat_cols_ling[:6]:
            cm = sub[sub["label"]=="Control"][feat].mean()
            dm = sub[sub["label"]=="Dementia"][feat].mean()
            print(f"    {feat:<22}  ctrl={cm:.4f}  dem={dm:.4f}")

    # =================================================================
    # C — FEATURE-CATEGORY PERTURBATION
    # FIX F4: Wilcoxon signed-rank test vs H0: median=0
    # =================================================================
    fp_df = pd.DataFrame(feat_pert)
    fp_df.to_csv("xai_results/feature_perturbation.csv", index=False)
    cat_cols = [c for c in fp_df.columns
                if c not in ("label","dataset","sample_idx")]

    print(f"\n{'='*65}")
    print("C — FEATURE-CATEGORY PERTURBATION")
    print("⚠️  Text-marginal importance, audio held FIXED.")
    print("    Scores: Δp * seq_len / p(1-p)  [length + confidence calibrated]")
    print(f"{'='*65}")

    cat_stats = []
    for cat in cat_cols:
        vals = fp_df[cat].dropna().values
        if len(vals) < 5:
            continue
        m, lo, hi = mean_ci95(vals)
        # FIX F4: Wilcoxon signed-rank vs 0 — guard against near-constant data
        nonzero = vals[vals != 0.0]
        if len(nonzero) < 5:
            # Not enough variation for Wilcoxon — use one-sample t-test as fallback
            from scipy.stats import ttest_1samp
            stat_w, p_w = ttest_1samp(vals, popmean=0)
            test_name = "t"
        else:
            try:
                stat_w, p_w = wilcoxon(vals, alternative="two-sided")
                test_name = "W"
            except ValueError:
                from scipy.stats import ttest_1samp
                stat_w, p_w = ttest_1samp(vals, popmean=0)
                test_name = "t"
        cat_stats.append({"category":cat,"mean":m,
                          "ci_lo":lo,"ci_hi":hi,
                          "stat":stat_w,"p":p_w,"test":test_name})
        print(f"  {cat:<14}  score={m:+.4f}  95%CI [{lo:+.4f},{hi:+.4f}]  "
              f"{test_name}={stat_w:.2f}  p={p_w:.3e}")

    cat_df = pd.DataFrame(cat_stats).sort_values("mean", ascending=False)
    cat_df.to_csv("xai_results/feature_perturbation_stats.csv", index=False)

    if not cat_df.empty:
        colors = ["#d62728" if m>=0 else "#1f77b4" for m in cat_df["mean"]]
        err    = [(r["mean"]-r["ci_lo"], r["ci_hi"]-r["mean"])
                  for _, r in cat_df.iterrows()]
        fig, ax = plt.subplots(figsize=(8,4))
        ax.barh(cat_df["category"], cat_df["mean"],
                color=colors, xerr=np.array(err).T, capsize=5)
        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.invert_yaxis()
        ax.set_xlabel(
            "Calibrated Importance = Δp × seq_len / p(1−p)\n"
            "Mean-embed baseline · Audio fixed · Wilcoxon vs 0\n"
            "red=Dementia-indicative  blue=Control-indicative")
        ax.set_title("Feature-Category Perturbation (v6.0)")
        plt.tight_layout()
        plt.savefig("xai_results/feature_perturbation.png", dpi=300)
        plt.close()
        print("Saved → xai_results/feature_perturbation.png")

    # =================================================================
    # D — WORD IMPORTANCE  (all fixes applied)
    # =================================================================
    word_df = pd.DataFrame(word_records)

    if word_df.empty:
        print("\n⚠️  No word records.")
    else:
        word_df.to_csv("xai_results/word_importance_raw.csv", index=False)

        for min_freq in (5, 3, 2, 1):
            ws = (word_df.groupby("word")
                  .agg(mean_importance=("importance","mean"),
                       freq=("importance","size"))
                  .query(f"freq >= {min_freq}").copy())
            if len(ws) >= 10:
                break

        ws["norm_importance"] = ws["mean_importance"] / np.log1p(ws["freq"])
        top20 = ws.sort_values("norm_importance", ascending=False).head(20)
        top20.to_csv("xai_results/word_importance_top.csv")

        colors = ["#d62728" if v>=0 else "#1f77b4"
                  for v in top20["norm_importance"]]
        fig, ax = plt.subplots(figsize=(11, max(6, len(top20)*0.5)))
        ax.barh(top20.index, top20["norm_importance"], color=colors)
        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.invert_yaxis()
        ax.set_xlabel(
            "Normalised calibrated importance = mean_score / log(freq+1)\n"
            "MC masking (F2) · mean-embed baseline (F1) · "
            "length+conf calibrated (F3,F6)\n"
            "red=Dementia-indicative  blue=Control-indicative")
        ax.set_title("Top Words — MC Embedding-Masked Perturbation (v6.0)")
        plt.tight_layout()
        plt.savefig("xai_results/words.png", dpi=300)
        plt.close()

        for ds in word_df["dataset"].unique():
            sub = word_df[word_df["dataset"]==ds]
            dsw = (sub.groupby("word")
                   .agg(mean_importance=("importance","mean"),
                        freq=("importance","size"))
                   .query("freq >= 1")          # relaxed: per-dataset counts are small
                   .sort_values("mean_importance", ascending=False)
                   .head(15))
            if not dsw.empty:
                dsw.to_csv(f"xai_results/words_{ds}.csv")
                print(f"Saved → xai_results/words_{ds}.csv  ({len(dsw)} words)")

        print(f"\nWord plot: {len(top20)} words (min_freq={min_freq})")
        print("Saved → xai_results/words.png + per-dataset CSVs")

    print(f"\n{'='*65}")
    print("✅  XAI v6.0 Complete")
    print(f"{'='*65}")
    for f in sorted(os.listdir("xai_results")):
        kb = os.path.getsize(f"xai_results/{f}") // 1024
        print(f"  {f:<48} {kb:>4} KB")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    os.makedirs("xai_results", exist_ok=True)

    df      = pd.read_csv("processed_data/master_metadata_cleaned.csv")
    feat_df = pd.read_csv("processed_data/master_acoustic_features.csv")

    with open("processed_data/cleaned_transcripts.json") as f:
        transcripts = json.load(f)["transcripts"]

    df["text"] = df["audio_path"].map(transcripts)
    df = df.dropna(subset=["text"])
    df = df[df["label"].isin(["Control","Dementia"])].reset_index(drop=True)

    join_col  = "audio_path" if "audio_path" in feat_df.columns else "participant_id"
    feat_cols = [c for c in feat_df.columns if c not in [
        "participant_id","audio_path","label","dataset",
        "split","age","gender","mmse"]]
    df = pd.merge(df, feat_df[[join_col]+feat_cols],
                  on=join_col, how="inner").reset_index(drop=True)
    acoustic_data = df[feat_cols].values

    print(f"Dataset breakdown:\n{df.groupby(['dataset','label']).size()}\n")

    # Shuffle and balance so XAI samples see both classes
    ctrl_df = df[df["label"] == "Control"].sample(frac=1, random_state=42)
    dem_df  = df[df["label"] == "Dementia"].sample(frac=1, random_state=42)
    n_each  = min(len(ctrl_df), len(dem_df))
    df = pd.concat([ctrl_df.iloc[:n_each], dem_df.iloc[:n_each]]
                   ).sample(frac=1, random_state=42).reset_index(drop=True)
    acoustic_data = df[[c for c in df.columns if c in feat_cols]].values

    print(f"Balanced dataset: {len(df)} samples "
          f"({df['label'].value_counts().to_dict()})\n")

    tokenizer, model = load_model(
        "trained_mamba_attention_fusion/best_attention_fusion.bin",
        len(feat_cols))

    run_xai(df, acoustic_data, tokenizer, model, n_samples=200)