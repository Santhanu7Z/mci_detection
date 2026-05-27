"""
explainability_suite.py — v6.3
Root cause fix:
  mamba_ssm's Triton layer_norm (used in every Block) calls
  torch.cuda.device(x.device.index) — when x is on CPU, index=None,
  so exchangeDevice(-1) fires.  MAMBA_NO_TRITON only disables the
  SSM scan kernel, NOT the layer norm.

  Fix: cpu_safe_model() walks the loaded model and:
    1. Sets Block.fused_add_norm = False  → bypasses Triton fused path
    2. Replaces every Triton RMSNorm with a pure-PyTorch equivalent
       that has the same interface (accepts residual= kwarg)
    3. Does the same for MixerModel.norm_f (the final layer norm)
  No venv patching, no env vars required.

All v6.2 fixes retained:
  Hook-based embedding injection, F1-F6, returns word_df.
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
    print("statsmodels missing — pip install statsmodels")

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_OK = True
except Exception:
    SPACY_OK = False
    print("spaCy missing — pip install spacy && python -m spacy download en_core_web_sm")

DEVICE = torch.device("cpu")
torch.manual_seed(42)
MC_SAMPLES = 1


# ============================================================
# CPU SAFETY FIX  [v6.3]
# Replaces all Triton RMSNorm instances with a pure-PyTorch
# equivalent and disables fused_add_norm on all Block modules.
# ============================================================

class _PyTorchRMSNorm(nn.Module):
    """
    Pure-PyTorch RMSNorm that matches the Triton RMSNorm interface
    used by mamba_ssm (accepts residual= and prenorm= kwargs).
    Weights are copied from the Triton version so numerics are identical.
    """
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x, residual=None, prenorm=False, residual_in_fp32=False):
        if residual is not None:
            x = (x + residual.to(x.dtype))
        fp32 = x.to(torch.float32)
        normed = fp32 * torch.rsqrt(fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        out = (self.weight * normed).to(x.dtype)
        return (out, x) if prenorm else out


def cpu_safe_model(model: nn.Module) -> nn.Module:
    """
    Make a mamba_ssm model safe for CPU inference by removing ALL CUDA-only
    components.  Three independent layers need patching:

    Layer 1 — Triton layer norm (block.py)
      Each Block uses fused_add_norm + Triton RMSNorm.
      Fix: fused_add_norm=False + replace RMSNorm with _PyTorchRMSNorm.

    Layer 2 — causal_conv1d CUDA extension (mamba_simple.py)
      The fast path calls causal_conv1d_cuda.causal_conv1d_fwd().
      Fix: set module-level causal_conv1d_fn=None → uses self.conv1d (nn.Conv1d).

    Layer 3 — selective_scan CUDA extension (selective_scan_interface.py)
      The slow path still calls selective_scan_fn (CUDA).
      Fix: replace with selective_scan_ref (pure PyTorch reference impl).

    Also sets use_fast_path=False on every Mamba instance so mamba_inner_fn
    (which combines layers 2+3 in one fused CUDA kernel) is never called.

    All patches are module-level variable replacements — no venv files touched.
    Numerics are identical to the trained model.
    """
    # ── Layer 1: Triton layer norm ────────────────────────────────────────────
    from mamba_ssm.modules.block import Block as MambaBlock
    try:
        from mamba_ssm.ops.triton.layer_norm import RMSNorm as TritonRMSNorm
    except ImportError:
        TritonRMSNorm = None

    norm_replaced = 0

    def _swap_norm(attr, parent):
        nonlocal norm_replaced
        norm = getattr(parent, attr, None)
        if norm is None:
            return
        if TritonRMSNorm is not None and isinstance(norm, TritonRMSNorm):
            new_norm = _PyTorchRMSNorm(norm.weight.shape[0], eps=norm.eps)
            new_norm.weight.data.copy_(norm.weight.data)
            setattr(parent, attr, new_norm)
            norm_replaced += 1

    for mod in model.modules():
        if isinstance(mod, MambaBlock):
            mod.fused_add_norm = False
            _swap_norm("norm", mod)

    backbone = getattr(model, "mamba", None) or getattr(model, "backbone", None)
    if backbone is not None:
        if hasattr(backbone, "fused_add_norm"):
            backbone.fused_add_norm = False
        _swap_norm("norm_f", backbone)

    print(f"  cpu_safe_model: layer norm  — replaced {norm_replaced} Triton RMSNorm(s)")

    # ── Layer 2 + 3: SSM CUDA extensions ─────────────────────────────────────
    try:
        import mamba_ssm.modules.mamba_simple as _ms

        # Disable fused CUDA kernel (mamba_inner_fn = conv + scan in one shot)
        _ms.mamba_inner_fn         = None
        _ms.mamba_inner_ref        = None   # may not exist; safe to set
        _ms.causal_conv1d_fn       = None   # → uses self.conv1d (nn.Conv1d)
        _ms.causal_conv1d_update   = None

        # Replace CUDA selective scan with pure-PyTorch reference
        try:
            from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
            _ms.selective_scan_fn = selective_scan_ref
            print("  cpu_safe_model: selective_scan — using PyTorch reference impl")
        except (ImportError, AttributeError):
            # Fallback: import from alternate location or skip
            print("  cpu_safe_model: selective_scan_ref not found — trying alternate")
            try:
                import mamba_ssm.ops.selective_scan_interface as _ssi
                # Some versions export it directly on the module
                ref = getattr(_ssi, "selective_scan_ref", None)
                if ref is not None:
                    _ms.selective_scan_fn = ref
                    print("  cpu_safe_model: selective_scan — found via module attr")
                else:
                    print("  WARNING: selective_scan_ref unavailable — CPU may still fail")
            except Exception as e:
                print(f"  WARNING: could not patch selective_scan_fn: {e}")

        # Force slow path on every Mamba SSM instance
        try:
            from mamba_ssm.modules.mamba_simple import Mamba
            mamba_count = sum(1 for m in model.modules() if isinstance(m, Mamba))
            for m in model.modules():
                if isinstance(m, Mamba):
                    m.use_fast_path = False
            print(f"  cpu_safe_model: use_fast_path  — disabled on {mamba_count} Mamba modules")
        except ImportError:
            print("  WARNING: could not import Mamba class to disable fast path")

    except ImportError:
        print("  cpu_safe_model: mamba_simple not found — skipping SSM patch")

    return model.cpu().eval()


# ============================================================
# EMBEDDING INJECTION VIA FORWARD HOOK
# ============================================================

def _run_with_embeds(model, input_ids, acoustic_features, modified_embeds):
    hook_handle = None
    def _embed_hook(module, inp, out):
        return modified_embeds
    try:
        hook_handle = model.get_input_embeddings().register_forward_hook(_embed_hook)
        with torch.no_grad():
            logits, weights = model(input_ids, acoustic_features)
    finally:
        if hook_handle is not None:
            hook_handle.remove()
    return logits, weights


# ============================================================
# STATISTICS HELPERS
# ============================================================

def cohen_d_paired(a, b):
    diff = np.asarray(a) - np.asarray(b)
    sd   = diff.std(ddof=1)
    return diff.mean() / sd if sd > 1e-9 else 0.0

def cohen_d_independent(a, b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na-1)*a.std(ddof=1)**2 + (nb-1)*b.std(ddof=1)**2) / (na+nb-2))
    return (a.mean()-b.mean()) / pooled if pooled > 1e-9 else 0.0

def mean_ci95(arr):
    from scipy.stats import t as t_dist
    n, m = len(arr), arr.mean()
    se   = arr.std(ddof=1) / np.sqrt(n)
    half = t_dist.ppf(0.975, df=n-1) * se
    return m, m-half, m+half

def gate_entropy(g):
    g = np.clip(np.asarray(g), 1e-6, 1-1e-6)
    return -(g*np.log2(g) + (1-g)*np.log2(1-g))


# ============================================================
# CHECKPOINT DETECTION
# ============================================================

def detect_classifier_input_dim(sd):
    for k, v in sd.items():
        if "classifier.0.weight" in k:
            return v.shape[1]
    return 256

def detect_arch_flags(sd):
    keys        = list(sd.keys())
    is_old_seq  = any("text_proj.0.weight"      in k for k in keys)
    uses_fusion = any("fusion_layer.gate"        in k for k in keys)
    clf_dim     = detect_classifier_input_dim(sd)
    has_attn    = any("attention.in_proj_weight" in k for k in keys)
    tp1         = sd.get("text_proj.1.weight")
    tp1_type    = ("gelu" if tp1 is None else "linear" if tp1.dim()==2 else "norm")
    ap3         = sd.get("audio_proj.3.weight")
    ap3_type    = ("gelu" if ap3 is None else "linear" if ap3.dim()==2 else "norm")
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
                 else nn.LayerNorm(self.fdim) if tp1_type=="norm" else nn.GELU())
        self.text_proj = (nn.Sequential(nn.Linear(backbone.config.d_model, self.fdim), slot1)
                          if is_old_seq else nn.Linear(backbone.config.d_model, self.fdim))

        slot3 = (nn.Linear(self.fdim, self.fdim) if ap3_type=="linear"
                 else nn.LayerNorm(self.fdim) if ap3_type=="norm" else nn.GELU())
        self.audio_proj = (nn.Sequential(nn.Linear(acoustic_dim,128), nn.GELU(),
                                         nn.Linear(128,self.fdim), slot3)
                           if is_old_seq else
                           nn.Sequential(nn.Linear(acoustic_dim,128), nn.GELU(),
                                         nn.Linear(128,self.fdim)))

        if uses_fusion:
            self.fusion_layer = nn.ModuleDict({"gate": nn.Sequential(
                nn.Linear(self.fdim*2,self.fdim), nn.GELU(),
                nn.Linear(self.fdim,1), nn.Sigmoid())})
        if has_attn:
            self.attention = nn.MultiheadAttention(self.fdim, num_heads=4, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(clf_dim,128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128,2))

    def forward(self, input_ids, acoustic_features):
        text_raw  = self.mamba(input_ids)
        text_out  = text_raw.mean(dim=1)
        text_emb  = self.text_proj(text_out)
        audio_emb = self.audio_proj(acoustic_features)

        if self.clf_dim == 256:
            if hasattr(self, "fusion_layer"):
                g       = self.fusion_layer["gate"](torch.cat([text_emb, audio_emb], dim=-1))
                fused   = g * text_emb + (1-g) * audio_emb
                weights = g.detach().cpu()
            else:
                fused, weights = text_emb * audio_emb, None
        else:
            combined         = torch.stack([text_emb, audio_emb], dim=1)
            attn_out, attn_w = self.attention(combined, combined, combined)
            if attn_w.dim() == 4: attn_w = attn_w.mean(dim=1)
            fused   = torch.cat([attn_out.flatten(start_dim=1), text_emb*audio_emb], dim=-1)
            weights = attn_w.detach().cpu()

        return self.classifier(fused), weights

    def get_input_embeddings(self):
        return self.mamba.embedding


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(checkpoint_path, acoustic_dim):
    print("Inspecting checkpoint...")
    state     = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    flags     = detect_arch_flags(state)
    print(f"  flags: {dict(zip(['old_seq','fusion','clf_dim','attn','tp1','ap3'], flags))}")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token = tokenizer.eos_token
    backbone  = MambaLMHeadModel.from_pretrained("state-spaces/mamba-130m")
    model     = MambaFusionEngineDynamic(backbone, acoustic_dim, *flags)
    model.load_state_dict(state, strict=True)
    model     = cpu_safe_model(model)   # remove all Triton layer norm calls
    print(f"  Model device: {next(model.parameters()).device}")
    return tokenizer, model


# ============================================================
# HELPERS
# ============================================================

def encode_with_offsets(tokenizer, text):
    enc       = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=512, return_offsets_mapping=True)
    input_ids = enc["input_ids"].to(DEVICE)
    offsets   = enc["offset_mapping"][0].tolist()
    return input_ids, offsets

def word_spans(text):
    return [(m.start(), m.end(), m.group().lower())
            for m in re.finditer(r"\S+", text)]

def tokens_for_word(offsets, w_start, w_end):
    return [i for i,(ts,te) in enumerate(offsets)
            if te > ts and ts >= w_start and te <= w_end]

def mean_embed_baseline(base_embeds):
    return base_embeds.mean(dim=1, keepdim=True)


# ============================================================
# WORD IMPORTANCE
# ============================================================

def get_word_importance(model, tokenizer, text, audio_feat):
    input_ids, offsets = encode_with_offsets(tokenizer, text)
    n_tokens           = input_ids.shape[1]
    audio_t = torch.from_numpy(np.array([audio_feat], dtype=np.float32)).to(DEVICE)

    emb_table = model.get_input_embeddings()
    with torch.no_grad():
        base_embeds = emb_table(input_ids)
        logits, _   = model(input_ids, audio_t)
        base_prob   = torch.softmax(logits, dim=1)[0,1].item()

    baseline    = mean_embed_baseline(base_embeds)
    spans       = word_spans(text)
    all_indices = list(range(len(spans)))
    calib       = max(base_prob*(1-base_prob), 1e-4)
    words, scores = [], []

    for w_idx, (w_start, w_end, word) in enumerate(spans):
        if len(word) <= 2: continue
        tok_indices = tokens_for_word(offsets, w_start, w_end)
        if not tok_indices: continue
        other_indices = [j for j in all_indices if j != w_idx]
        delta_list    = []
        for _ in range(MC_SAMPLES):
            n_extra = max(0, int(0.2*len(other_indices)))
            extra   = (np.random.choice(other_indices, size=n_extra, replace=False).tolist()
                       if n_extra else [])
            masked  = base_embeds.clone()
            masked[0, tok_indices, :] = baseline
            for oi in extra:
                ots, ote, _ = spans[oi]
                for ti in tokens_for_word(offsets, ots, ote):
                    masked[0, ti, :] = baseline
            log2, _ = _run_with_embeds(model, input_ids, audio_t, masked)
            delta_list.append(base_prob - torch.softmax(log2, dim=1)[0,1].item())
        words.append(word)
        scores.append(np.mean(delta_list) * n_tokens / calib)

    return words, scores


# ============================================================
# FEATURE-CATEGORY PERTURBATION
# ============================================================

DISFLUENCY_RE     = re.compile(r"\b(uh+|um+|er+|ah+|mm+|hmm+)\b", re.I)
DISCOURSE_MARKERS = {"well","so","and","but","because","then","now","like",
                     "okay","right","you","know","mean"}

def get_feature_perturbation(model, tokenizer, text, audio_feat):
    input_ids, offsets = encode_with_offsets(tokenizer, text)
    audio_t = torch.from_numpy(np.array([audio_feat], dtype=np.float32)).to(DEVICE)
    emb_table = model.get_input_embeddings()
    with torch.no_grad():
        base_embeds = emb_table(input_ids)
        logits, _   = model(input_ids, audio_t)
        base_prob   = torch.softmax(logits, dim=1)[0,1].item()

    baseline = mean_embed_baseline(base_embeds)
    calib    = max(base_prob*(1-base_prob), 1e-4)
    categories = ["disfluency","discourse"] + (["pronoun","verb","noun"] if SPACY_OK else [])

    results = {}
    for cat in categories:
        masked = base_embeds.clone()
        if cat == "disfluency":
            for m in re.finditer(DISFLUENCY_RE, text):
                for i,(ts,te) in enumerate(offsets):
                    if te>ts and ts>=m.start() and te<=m.end():
                        masked[0,i,:] = baseline
        elif cat == "discourse":
            for m in re.finditer(r"\S+", text):
                if m.group().lower() in DISCOURSE_MARKERS:
                    for i,(ts,te) in enumerate(offsets):
                        if te>ts and ts>=m.start() and te<=m.end():
                            masked[0,i,:] = baseline
        elif SPACY_OK:
            POS = {"pronoun":"PRON","verb":"VERB","noun":"NOUN"}
            doc = nlp(text)
            for tok in doc:
                if tok.pos_ == POS.get(cat,""):
                    ts_w,te_w = tok.idx, tok.idx+len(tok.text)
                    for i,(ts,te) in enumerate(offsets):
                        if te>ts and ts>=ts_w and te<=te_w:
                            masked[0,i,:] = baseline
        log2, _ = _run_with_embeds(model, input_ids, audio_t, masked)
        m_prob  = torch.softmax(log2, dim=1)[0,1].item()
        results[cat] = (base_prob-m_prob) * input_ids.shape[1] / calib

    return results


# ============================================================
# LINGUISTIC FEATURE EXTRACTION
# ============================================================

def extract_linguistic_features(text):
    feats      = {}
    raw_tokens = text.split()
    n          = max(len(raw_tokens), 1)
    feats["disfluency_rate"] = len(DISFLUENCY_RE.findall(text)) / n
    reps = sum(1 for a,b in zip(raw_tokens, raw_tokens[1:]) if a.lower()==b.lower())
    feats["repetition_rate"] = reps / n
    if SPACY_OK:
        doc  = nlp(text)
        toks = [t for t in doc if not t.is_space]
        nt   = max(len(toks),1)
        lems = [t.lemma_.lower() for t in toks if t.is_alpha]
        feats["ttr"]             = len(set(lems))/max(len(lems),1)
        feats["pronoun_ratio"]   = sum(1 for t in toks if t.pos_=="PRON")/nt
        feats["verb_ratio"]      = sum(1 for t in toks if t.pos_=="VERB")/nt
        feats["content_density"] = sum(1 for t in toks if t.pos_ in ("NOUN","VERB","ADJ","ADV"))/nt
        sents = list(doc.sents)
        feats["mean_sent_len"]   = np.mean([len(s) for s in sents]) if sents else 0.
        feats["unique_nouns"]    = len({t.lemma_.lower() for t in toks if t.pos_=="NOUN"})
    else:
        words = [w.lower() for w in raw_tokens if w.isalpha()]
        feats["ttr"] = len(set(words))/max(len(words),1)
        PRON = {"i","he","she","it","they","we","you","this","that",
                "these","those","them","him","her","us"}
        feats["pronoun_ratio"] = sum(1 for w in words if w in PRON)/max(len(words),1)
    feats["total_words"] = n
    return feats


# ============================================================
# MAIN XAI PIPELINE
# ============================================================

def run_xai(df, acoustic_data, tokenizer, model, n_samples=200, output_dir="xai_results"):

    # Ensure all Triton norms are replaced and model is on CPU
    model = cpu_safe_model(model)

    print(f"\n{'='*65}")
    print(f"XAI v6.3  —  n={n_samples}  MC={MC_SAMPLES}  device=cpu")
    print(f"Triton RMSNorm replaced with PyTorch equivalent")
    print(f"{'='*65}")

    os.makedirs(output_dir, exist_ok=True)

    ctrl_idx = df[df["label"]=="Control"].index.tolist()
    dem_idx  = df[df["label"]=="Dementia"].index.tolist()
    np.random.seed(42)
    np.random.shuffle(ctrl_idx); np.random.shuffle(dem_idx)
    n_each   = min(n_samples//2, len(ctrl_idx), len(dem_idx))
    sel_idx  = sorted(ctrl_idx[:n_each] + dem_idx[:n_each])
    df_sel   = df.loc[sel_idx].reset_index(drop=True)
    ac_sel   = acoustic_data[sel_idx]
    print(f"Stratified sample: {n_each}x2 = {len(df_sel)} total")
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

        inputs    = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(DEVICE)
        audio_t   = torch.from_numpy(np.array([audio], dtype=np.float32)).to(DEVICE)

        with torch.no_grad():
            _, weights = model(input_ids, audio_t)

        if weights is not None:
            w = weights.squeeze()
            if w.dim() == 0:
                gate_mode = True; gv = w.item()
                gate_vals.append(gv); text_weights.append(gv); audio_weights.append(1.0-gv)
            else:
                attn = w.mean(dim=0).numpy(); attn /= attn.sum()+1e-9
                text_weights.append(float(attn[0])); audio_weights.append(float(attn[1]))

        words, imps = get_word_importance(model, tokenizer, text, audio)
        for word,imp in zip(words,imps):
            word_records.append({"word":word,"importance":imp,"label":label,"dataset":dset})

        fp = get_feature_perturbation(model, tokenizer, text, audio)
        fp.update({"label":label,"dataset":dset,"sample_idx":i})
        feat_pert.append(fp)

        lf = extract_linguistic_features(text)
        lf.update({"label":label,"dataset":dset,"sample_idx":i})
        ling_records.append(lf)

    # ── A: MODALITY CONTRIBUTION ──────────────────────────────────────────────
    if text_weights:
        t_arr = np.array(text_weights); a_arr = np.array(audio_weights)
        t_stat, p_val = ttest_rel(t_arr, a_arr)
        d_paired = cohen_d_paired(t_arr, a_arr)
        tm,tlo,thi = mean_ci95(t_arr); am,alo,ahi = mean_ci95(a_arr)
        H = gate_entropy(gate_vals) if gate_mode else None

        print(f"\n{'='*65}\nA — MODALITY CONTRIBUTION\n{'='*65}")
        print(f"Text  : {tm:.4f}  95%CI [{tlo:.4f}, {thi:.4f}]")
        print(f"Audio : {am:.4f}  95%CI [{alo:.4f}, {ahi:.4f}]")
        print(f"Ratio : {tm/am:.2f}x  |  t={t_stat:.3f}, p={p_val:.2e}  |  d={d_paired:.3f}")
        if H is not None:
            print(f"Gate entropy: {H.mean():.4f} +/- {H.std():.4f}  "
                  f"({(H<0.5).mean()*100:.1f}% H<0.5)")

        mod_df = pd.DataFrame({
            "text_weight":t_arr,"audio_weight":a_arr,
            "gate_entropy":H if H is not None else np.zeros(len(t_arr)),
            "label":df_sel.iloc[:len(t_arr)]["label"].values,
            "dataset":df_sel.iloc[:len(t_arr)]["dataset"].values})
        mod_df.to_csv(f"{output_dir}/modality_weights.csv", index=False)

        datasets = mod_df["dataset"].unique()
        fig, axes = plt.subplots(1, 1+len(datasets), figsize=(5*(1+len(datasets)),4), sharey=True)
        axes = np.atleast_1d(axes)
        def _bar(ax,tw,aw,title):
            m=[tw.mean(),aw.mean()]; e=[tw.std(),aw.std()]
            bars=ax.bar(["Text","Audio"],m,yerr=e,color=["#4C72B0","#DD8452"],capsize=6,width=0.5)
            ax.set_ylim(0,1); ax.set_title(title,fontsize=9)
            for b,v in zip(bars,m): ax.text(b.get_x()+b.get_width()/2,v+0.02,f"{v:.3f}",ha="center",fontsize=9)
        _bar(axes[0],t_arr,a_arr,f"Overall (n={len(t_arr)})")
        for ax,ds in zip(axes[1:],datasets):
            sub=mod_df[mod_df["dataset"]==ds]
            _bar(ax,sub["text_weight"].values,sub["audio_weight"].values,f"{ds}\n(n={len(sub)})")
        axes[0].set_ylabel("Mean Fusion Weight")
        fig.suptitle(f"Modality Contribution — d={d_paired:.2f}, p={p_val:.2e}",y=1.03)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/modality_weights.png",dpi=300,bbox_inches="tight"); plt.close()
        print(f"Saved -> {output_dir}/modality_weights.png")

    # ── B: LINGUISTIC FEATURES ────────────────────────────────────────────────
    ling_df = pd.DataFrame(ling_records)
    ling_df.to_csv(f"{output_dir}/linguistic_features.csv", index=False)
    feat_cols_ling = [c for c in ling_df.columns if c not in ("label","dataset","sample_idx")]
    ctrl = ling_df[ling_df["label"]=="Control"]; dem = ling_df[ling_df["label"]=="Dementia"]

    raw_stats = []
    print(f"\n  Control: {len(ctrl)}  Dementia: {len(dem)}")
    for feat in feat_cols_ling:
        cv=ctrl[feat].dropna().values; dv=dem[feat].dropna().values
        if len(cv)<2 or len(dv)<2: continue
        _,p=mannwhitneyu(cv,dv,alternative="two-sided")
        raw_stats.append({"feature":feat,"ctrl_mean":cv.mean(),"dem_mean":dv.mean(),
                           "p_raw":p,"cohen_d":cohen_d_independent(dv,cv)})

    stat_df = pd.DataFrame(raw_stats)
    if not stat_df.empty:
        if STATSMODELS_OK:
            _,p_adj,_,_=multipletests(stat_df["p_raw"].values,method="fdr_bh")
            stat_df["p_adj"]=p_adj
        else:
            stat_df["p_adj"]=stat_df["p_raw"]
        stat_df["sig"]=stat_df["p_adj"].apply(lambda p:"***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns")
    stat_df.to_csv(f"{output_dir}/linguistic_stats.csv", index=False)

    print(f"\n{'='*65}\nB — LINGUISTIC FEATURES (BH-FDR)\n{'='*65}")
    if not stat_df.empty:
        for _,row in stat_df.sort_values("p_adj").iterrows():
            print(f"  {row.feature:<22} ctrl={row.ctrl_mean:>7.4f}  dem={row.dem_mean:>7.4f}  "
                  f"p_adj={row.p_adj:>8.4f}  d={row.cohen_d:>6.3f}  {row.sig}")
        print(f"\n  {(stat_df['sig']!='ns').sum()}/{len(stat_df)} significant after BH-FDR")

    for ds in ling_df["dataset"].unique():
        sub=ling_df[ling_df["dataset"]==ds]
        print(f"\n  [{ds}] Control={( sub['label']=='Control').sum()}  Dementia={(sub['label']=='Dementia').sum()}")
        for feat in feat_cols_ling[:6]:
            print(f"    {feat:<22} ctrl={sub[sub['label']=='Control'][feat].mean():.4f}  "
                  f"dem={sub[sub['label']=='Dementia'][feat].mean():.4f}")

    # ── C: FEATURE-CATEGORY PERTURBATION ─────────────────────────────────────
    fp_df    = pd.DataFrame(feat_pert)
    fp_df.to_csv(f"{output_dir}/feature_perturbation.csv", index=False)
    cat_cols = [c for c in fp_df.columns if c not in ("label","dataset","sample_idx")]

    print(f"\n{'='*65}\nC — FEATURE-CATEGORY PERTURBATION\n{'='*65}")
    cat_stats = []
    for cat in cat_cols:
        vals = fp_df[cat].dropna().values
        if len(vals)<5: continue
        m,lo,hi = mean_ci95(vals)
        nonzero = vals[vals!=0.0]
        if len(nonzero)<5:
            from scipy.stats import ttest_1samp
            sw,pw=ttest_1samp(vals,popmean=0); tn="t"
        else:
            try: sw,pw=wilcoxon(vals,alternative="two-sided"); tn="W"
            except ValueError:
                from scipy.stats import ttest_1samp
                sw,pw=ttest_1samp(vals,popmean=0); tn="t"
        cat_stats.append({"category":cat,"mean":m,"ci_lo":lo,"ci_hi":hi,"stat":sw,"p":pw,"test":tn})
        print(f"  {cat:<14}  {m:+.4f}  [{lo:+.4f},{hi:+.4f}]  {tn}={sw:.2f}  p={pw:.3e}")

    cat_df = pd.DataFrame(cat_stats).sort_values("mean", ascending=False)
    cat_df.to_csv(f"{output_dir}/feature_perturbation_stats.csv", index=False)
    if not cat_df.empty:
        colors=["#d62728" if m>=0 else "#1f77b4" for m in cat_df["mean"]]
        err=[(r["mean"]-r["ci_lo"],r["ci_hi"]-r["mean"]) for _,r in cat_df.iterrows()]
        fig,ax=plt.subplots(figsize=(8,4))
        ax.barh(cat_df["category"],cat_df["mean"],color=colors,xerr=np.array(err).T,capsize=5)
        ax.axvline(0,color="black",lw=0.8,ls="--"); ax.invert_yaxis()
        ax.set_xlabel("Calibrated Importance = Dp x seq_len / p(1-p)")
        ax.set_title("Feature-Category Perturbation (v6.3)")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/feature_perturbation.png",dpi=300); plt.close()
        print(f"Saved -> {output_dir}/feature_perturbation.png")

    # ── D: WORD IMPORTANCE ────────────────────────────────────────────────────
    word_df = pd.DataFrame(word_records)
    if word_df.empty:
        print("\n  No word records.")
    else:
        word_df.to_csv(f"{output_dir}/word_importance_raw.csv", index=False)
        for min_freq in (5,3,2,1):
            ws=(word_df.groupby("word")
                .agg(mean_importance=("importance","mean"),freq=("importance","size"))
                .query(f"freq>={min_freq}").copy())
            if len(ws)>=10: break
        ws["norm_importance"]=ws["mean_importance"]/np.log1p(ws["freq"])
        top20=ws.sort_values("norm_importance",ascending=False).head(20)
        top20.to_csv(f"{output_dir}/word_importance_top.csv")
        colors=["#d62728" if v>=0 else "#1f77b4" for v in top20["norm_importance"]]
        fig,ax=plt.subplots(figsize=(11,max(6,len(top20)*0.5)))
        ax.barh(top20.index,top20["norm_importance"],color=colors)
        ax.axvline(0,color="black",lw=0.8,ls="--"); ax.invert_yaxis()
        ax.set_xlabel("Norm. calibrated importance\nred=Dementia-indicative  blue=Control-indicative")
        ax.set_title("Top Words — MC Embedding-Masked Perturbation (v6.3)")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/words.png",dpi=300); plt.close()
        for ds in word_df["dataset"].unique():
            sub=word_df[word_df["dataset"]==ds]
            dsw=(sub.groupby("word").agg(mean_importance=("importance","mean"),freq=("importance","size"))
                 .query("freq>=1").sort_values("mean_importance",ascending=False).head(15))
            if not dsw.empty:
                dsw.to_csv(f"{output_dir}/words_{ds}.csv")
        print(f"Saved -> {output_dir}/words.png  ({len(top20)} words, min_freq={min_freq})")

    print(f"\n{'='*65}\nXAI v6.3 Complete\n{'='*65}")
    for f in sorted(os.listdir(output_dir)):
        kb=os.path.getsize(f"{output_dir}/{f}")//1024
        print(f"  {f:<48} {kb:>4} KB")

    return word_df if not word_df.empty else pd.DataFrame()


# ============================================================
# MAIN (standalone)
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
    feat_cols = [c for c in feat_df.columns if c not in
                 ["participant_id","audio_path","label","dataset","split","age","gender","mmse"]]
    df = pd.merge(df, feat_df[[join_col]+feat_cols], on=join_col, how="inner").reset_index(drop=True)
    acoustic_data = df[feat_cols].values
    ctrl_df = df[df["label"]=="Control"].sample(frac=1, random_state=42)
    dem_df  = df[df["label"]=="Dementia"].sample(frac=1, random_state=42)
    n_each  = min(len(ctrl_df), len(dem_df))
    df = pd.concat([ctrl_df.iloc[:n_each],dem_df.iloc[:n_each]]).sample(frac=1,random_state=42).reset_index(drop=True)
    acoustic_data = df[[c for c in df.columns if c in feat_cols]].values
    print(f"Balanced: {len(df)} samples ({df['label'].value_counts().to_dict()})")
    tokenizer, model = load_model(
        "trained_mamba_attention_fusion/best_attention_fusion.bin", len(feat_cols))
    run_xai(df, acoustic_data, tokenizer, model, n_samples=200)
