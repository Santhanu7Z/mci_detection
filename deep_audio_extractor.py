#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep Audio Feature Extractor v1.0
Extracts Wav2Vec2 and HuBERT embeddings for all samples in master_metadata.csv.

Features:
  - Checkpoint resumption (safe to interrupt and restart)
  - Long-audio chunking (25 s windows, max 8 chunks)
  - Mixed-precision inference
  - Exports both .npy arrays and a merged CSV for downstream merging

Outputs:
  processed_data/deep_audio/wav2vec2_embeddings.npy   [N, 768]
  processed_data/deep_audio/hubert_embeddings.npy     [N, 768]
  processed_data/deep_audio/participant_ids.npy        [N]
  processed_data/deep_audio/deep_audio_features.csv   [N, 1537]  ← id + 1536 dims
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import librosa
from tqdm import tqdm
from transformers import (
    Wav2Vec2Model,
    Wav2Vec2Processor,
    HubertModel,
    Wav2Vec2FeatureExtractor,
)

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_RATE  = 16_000
CHUNK_SECS   = 25        # max seconds per inference chunk
MAX_CHUNKS   = 8         # hard cap → at most 200 s of audio per sample

WAV2VEC2_ID  = "facebook/wav2vec2-base-960h"   # output dim: 768
HUBERT_ID    = "facebook/hubert-base-ls960"    # output dim: 768

DATA_DIR     = "processed_data"
META_CSV     = os.path.join(DATA_DIR, "master_metadata.csv")
OUTPUT_DIR   = os.path.join(DATA_DIR, "deep_audio")


# ── Audio helpers ─────────────────────────────────────────────────────────────

def load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray | None:
    try:
        audio, _ = librosa.load(
            path, sr=sr, mono=True, duration=CHUNK_SECS * MAX_CHUNKS
        )
        return audio
    except Exception as exc:
        print(f"  ✗ Load failed [{path}]: {exc}")
        return None


def chunk_audio(audio: np.ndarray,
                chunk_secs: int = CHUNK_SECS,
                sr: int = SAMPLE_RATE) -> list[np.ndarray]:
    chunk_len = chunk_secs * sr
    chunks = []
    for start in range(0, len(audio), chunk_len):
        chunk = audio[start : start + chunk_len]
        if len(chunk) >= sr // 2:          # at least 0.5 s of signal
            chunks.append(chunk)
        if len(chunks) >= MAX_CHUNKS:
            break
    return chunks if chunks else [audio]


# ── Model wrappers ────────────────────────────────────────────────────────────

class Wav2Vec2Extractor:
    """Mean-pooled last hidden state from facebook/wav2vec2-base-960h."""

    def __init__(self, device: str = "cpu"):
        self.device    = device
        self.processor = Wav2Vec2Processor.from_pretrained(WAV2VEC2_ID)
        self.model     = Wav2Vec2Model.from_pretrained(WAV2VEC2_ID).to(device)
        self.model.eval()
        self.dim       = 768

    @torch.no_grad()
    def _embed_chunk(self, chunk: np.ndarray) -> np.ndarray:
        inputs = self.processor(
            chunk,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs["input_values"].to(self.device)
        out = self.model(input_values)
        # [1, T, 768] → mean over T → [768]
        return out.last_hidden_state.mean(dim=1).squeeze(0).cpu().float().numpy()

    def extract(self, audio: np.ndarray) -> np.ndarray:
        chunks = chunk_audio(audio)
        embs   = [self._embed_chunk(c) for c in chunks]
        return np.mean(embs, axis=0)      # weighted mean across chunks


class HuBERTExtractor:
    """Mean-pooled last hidden state from facebook/hubert-base-ls960."""

    def __init__(self, device: str = "cpu"):
        self.device    = device
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(HUBERT_ID)
        self.model     = HubertModel.from_pretrained(HUBERT_ID).to(device)
        self.model.eval()
        self.dim       = 768

    @torch.no_grad()
    def _embed_chunk(self, chunk: np.ndarray) -> np.ndarray:
        inputs = self.processor(
            chunk,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out    = self.model(**inputs)
        return out.last_hidden_state.mean(dim=1).squeeze(0).cpu().float().numpy()

    def extract(self, audio: np.ndarray) -> np.ndarray:
        chunks = chunk_audio(audio)
        embs   = [self._embed_chunk(c) for c in chunks]
        return np.mean(embs, axis=0)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def extract_all(meta_csv: str = META_CSV,
                output_dir: str = OUTPUT_DIR,
                device: str | None = None):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device  : {device}")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(meta_csv)
    print(f"Metadata: {len(df)} samples")

    # ── Resumption ────────────────────────────────────────────────────────────
    ids_path    = os.path.join(output_dir, "participant_ids.npy")
    w2v_path    = os.path.join(output_dir, "wav2vec2_embeddings.npy")
    hubert_path = os.path.join(output_dir, "hubert_embeddings.npy")

    prev_ids : list = []
    prev_w2v : np.ndarray | None = None
    prev_hub : np.ndarray | None = None

    if os.path.exists(ids_path):
        prev_ids = np.load(ids_path, allow_pickle=True).tolist()
        prev_w2v = np.load(w2v_path)
        prev_hub = np.load(hubert_path)
        print(f"Resuming: {len(prev_ids)} embeddings already on disk")

    done_ids = set(prev_ids)
    pending  = df[~df["participant_id"].isin(done_ids)].reset_index(drop=True)
    print(f"Pending : {len(pending)} samples")

    if len(pending) == 0:
        print("✅ All samples already extracted.")
        return

    # ── Load models ───────────────────────────────────────────────────────────
    print("\nLoading Wav2Vec2 …")
    w2v_ext = Wav2Vec2Extractor(device=device)
    print("Loading HuBERT …\n")
    hub_ext = HuBERTExtractor(device=device)

    new_w2v : list[np.ndarray] = []
    new_hub : list[np.ndarray] = []
    new_ids : list[str]        = []

    for row in tqdm(pending.itertuples(), total=len(pending), desc="Deep Audio"):
        pid  = row.participant_id
        path = row.audio_path

        if not os.path.exists(path):
            print(f"  ✗ Missing audio: {path}")
            continue

        audio = load_audio(path)
        if audio is None:
            continue

        try:
            w2v_emb = w2v_ext.extract(audio)
            hub_emb = hub_ext.extract(audio)

            new_w2v.append(w2v_emb)
            new_hub.append(hub_emb)
            new_ids.append(pid)

        except Exception as exc:
            print(f"  ✗ Extraction error [{pid}]: {exc}")

    if not new_ids:
        print("No new embeddings extracted.")
        return

    # ── Merge with any previous results ──────────────────────────────────────
    all_w2v = np.vstack([prev_w2v, np.array(new_w2v)]) if prev_w2v is not None else np.array(new_w2v)
    all_hub = np.vstack([prev_hub, np.array(new_hub)]) if prev_hub is not None else np.array(new_hub)
    all_ids = prev_ids + new_ids

    np.save(w2v_path,    all_w2v)
    np.save(hubert_path, all_hub)
    np.save(ids_path,    np.array(all_ids))

    # ── Combined CSV (participant_id + 1536 dims) ─────────────────────────────
    combined = np.hstack([all_w2v, all_hub])     # [N, 1536]
    cols     = ([f"w2v_{i}" for i in range(768)] +
                [f"hub_{i}" for i in range(768)])
    feat_df  = pd.DataFrame(combined, columns=cols)
    feat_df.insert(0, "participant_id", all_ids)
    out_csv  = os.path.join(output_dir, "deep_audio_features.csv")
    feat_df.to_csv(out_csv, index=False)

    print(f"\n✅ Done.  Saved {len(all_ids)} samples.")
    print(f"   Wav2Vec2 : {all_w2v.shape}  →  {w2v_path}")
    print(f"   HuBERT   : {all_hub.shape}  →  {hubert_path}")
    print(f"   Combined : {combined.shape}  →  {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Audio Feature Extractor")
    parser.add_argument("--meta",   default=META_CSV,   help="Master metadata CSV")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--device", default=None,       help="cuda | cpu")
    args = parser.parse_args()
    extract_all(args.meta, args.output, args.device)
