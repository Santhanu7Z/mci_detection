#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clinical-Grade Whisper Pipeline (v5.2) - Optimized Research Implementation
Optimized for MCI Detection (Whisper + Mamba Fusion).

Refinements:
- Removed redundant 'best_of' parameter (incompatible with temperature=0).
- Optimized torch settings for faster inference (enabled cudnn benchmark).
- Maintained word-level timestamps and bilingual TAUKADIAL logic.
"""

import os
import json
import argparse
import pandas as pd
import whisper
import torch
import numpy as np
import wave
import contextlib
from tqdm import tqdm
from pathlib import Path

def get_duration(file_path):
    """Get audio duration using standard wave library."""
    try:
        with contextlib.closing(wave.open(file_path, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            return round(frames / float(rate), 3)
    except Exception:
        return 0.0

def set_reproducibility(seed=42):
    """Sets seeds for reproducibility and optimizes for inference speed."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Optimized for inference: benchmark=True finds fastest algorithms
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    np.random.seed(seed)

def run_transcription(master_metadata_csv, output_cache_json, whisper_model_size):
    set_reproducibility(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Data
    if not os.path.exists(master_metadata_csv):
        print(f"Error: {master_metadata_csv} not found.")
        return
    
    df = pd.read_csv(master_metadata_csv)
    
    # 2. Resumption Logic
    master_cache = {"transcripts": {}}
    if os.path.exists(output_cache_json):
        try:
            with open(output_cache_json, 'r') as f:
                master_cache = json.load(f)
            if "transcripts" not in master_cache:
                master_cache = {"transcripts": master_cache}
            print(f"✓ Resuming from {len(master_cache['transcripts'])} entries.")
        except Exception as e:
            print(f"Warning: Starting fresh cache ({e}).")

    # 3. Load Whisper
    print(f"Loading Whisper {whisper_model_size} on {device}...")
    model = whisper.load_model(whisper_model_size, device=device)

    # 4. Filter Pending
    completed = set(master_cache['transcripts'].keys())
    pending_rows = [row for row in df.itertuples() if row.audio_path not in completed]
    
    if not pending_rows:
        print("All samples processed.")
        return

    print(f"Transcribing {len(pending_rows)} samples with Optimized Robust Decoding...")

    # 5. Transcription Loop
    for idx, row in enumerate(tqdm(pending_rows, desc="ASR Phase")):
        audio_path = row.audio_path
        dataset = row.dataset
        
        # TAUKADIAL: Bilingual Mandarin/English. language=None enables auto-detect.
        if dataset == 'taukadial':
            lang = None 
            prompt = "Clinical interview. Mandarin Chinese and English code-switching. Patient may stutter, use fillers like uh, um, 呃, 那个. Verbatim."
        else:
            lang = 'en'
            prompt = "Clinical interview. Patient may stutter, use fillers like uh, um, ah. Verbatim transcription."

        try:
            if not os.path.exists(audio_path):
                continue

            duration = get_duration(audio_path)

            # RESEARCH SOTA PARAMETERS (v5.2)
            # beam_size=5 provides stable search. best_of is removed for temp=0 efficiency.
            result = model.transcribe(
                audio_path,
                language=lang,
                task="transcribe",
                temperature=0,
                beam_size=5,
                initial_prompt=prompt,
                condition_on_previous_text=False,
                word_timestamps=True, 
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                fp16=(device == "cuda")
            )
            
            # 6. Structured Storage
            master_cache['transcripts'][audio_path] = {
                "text": result['text'].strip(),
                "duration": duration,
                "dataset": dataset,
                "detected_language": result.get('language', lang),
                "segments": [
                    {
                        "start": round(s['start'], 2),
                        "end": round(s['end'], 2),
                        "words": [
                            {"word": w['word'].strip(), "start": round(w['start'], 2), "end": round(w['end'], 2)}
                            for w in s.get('words', [])
                        ]
                    } for s in result['segments']
                ]
            }
            
        except Exception as e:
            print(f"\nError processing {audio_path}: {e}")
            master_cache['transcripts'][audio_path] = {"text": "", "error": str(e)}

        # Checkpointing (Every 10)
        if (idx + 1) % 10 == 0:
            with open(output_cache_json, 'w') as f:
                json.dump(master_cache, f, indent=2)

    # 7. Final Save
    with open(output_cache_json, 'w') as f:
        json.dump(master_cache, f, indent=2)
    
    print(f"\n✅ Pipeline Complete. Saved to: {output_cache_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', type=str, default='processed_data/master_metadata.csv')
    parser.add_argument('--output', type=str, default='processed_data/transcripts_cache.json')
    parser.add_argument('--model', type=str, default='large-v3')
    args = parser.parse_args()
    run_transcription(args.metadata, args.output, args.model)