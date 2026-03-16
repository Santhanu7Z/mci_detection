#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Preprocessing v4.2 - CLUSTER OPTIMIZED (Bulletproof Edition)
- Participant-Only Audio Extraction (Segmentation).
- Fix: Flexible TAUKADIAL Parser (handles both ',' and ';' delimiters).
- Fix: Enhanced TAUKADIAL Metadata (captures Age, Sex/Gender, and MMSE).
- Fix: TAUKADIAL Training Label Recovery (loads both train/test ground truth).
- standardizes to 16kHz Mono (Whisper Optimized).
"""

import argparse
import os
import re
import json
from pathlib import Path
import pandas as pd
from pydub import AudioSegment
from tqdm import tqdm

# ============================================================
# AUDIO UTILITIES
# ============================================================

def standardize_audio(audio_segment):
    """Ensures 16kHz, Mono, 16-bit depth for Whisper optimization."""
    if audio_segment.channels > 1:
        audio_segment = audio_segment.set_channels(1)
    if audio_segment.frame_rate != 16000:
        audio_segment = audio_segment.set_frame_rate(16000)
    return audio_segment

def save_processed_audio(audio_segment, output_path):
    """Standardizes and saves audio; returns False if audio is invalid."""
    audio_segment = standardize_audio(audio_segment)
    if len(audio_segment) < 500:
        return False
    audio_segment.export(output_path, format="wav")
    return True

# ============================================================
# CHAT SEGMENTATION ENGINE
# ============================================================

def extract_par_segments_from_cha(cha_path):
    """Parses CHAT files to find all timestamps belonging to the Participant (*PAR)."""
    segments = []
    par_pattern = re.compile(r"^\*PAR[^\:]*\s*:\s*(.*)", flags=re.IGNORECASE)
    ts_pattern = re.compile(r"(\d+)_(\d+)")

    try:
        with open(cha_path, "r", encoding="utf-8-sig", errors="ignore") as f:
            for line in f:
                if par_pattern.match(line.strip()):
                    timestamps = ts_pattern.findall(line)
                    for start, end in timestamps:
                        s, e = int(start), int(end)
                        if e > s:
                            segments.append((s, e))
    except Exception as e:
        print(f"Error parsing CHAT {cha_path}: {e}")
    
    if not segments: return []
    
    segments.sort()
    merged = [segments[0]]
    for curr in segments[1:]:
        prev = merged[-1]
        if curr[0] <= prev[1] + 100:
            merged[-1] = (prev[0], max(prev[1], curr[1]))
        else:
            merged.append(curr)
    return merged

def segment_audio(audio_path, cha_path):
    """Cuts audio to include ONLY participant segments."""
    segments = extract_par_segments_from_cha(cha_path)
    if not segments: return None

    try:
        ext = Path(audio_path).suffix.lower()
        if ext == '.wav':
            full_audio = AudioSegment.from_wav(audio_path)
        elif ext == '.mp3':
            full_audio = AudioSegment.from_mp3(audio_path)
        else:
            full_audio = AudioSegment.from_file(audio_path)

        combined = AudioSegment.empty()
        padding = 100 
        
        for i, (start, end) in enumerate(segments):
            s_pad = max(0, start - padding)
            e_pad = min(len(full_audio), end + padding)
            combined += full_audio[s_pad:e_pad]
            if i < len(segments) - 1:
                combined += AudioSegment.silent(duration=100)
            
        return combined
    except Exception as e:
        print(f"Segmentation failed for {audio_path}: {e}")
        return None

# ============================================================
# DATASET PROCESSORS
# ============================================================

def clean_df_headers(df):
    """Aggressively strips whitespace and invisible chars from headers."""
    df.columns = [str(c).strip() for c in df.columns]
    return df

def process_pitt(output_dir):
    print("\nProcessing Pitt Corpus...")
    records = []
    data_dir = Path("data/Pitt")
    out_audio = Path(output_dir) / "pitt_segmented"
    out_audio.mkdir(parents=True, exist_ok=True)

    for group in ["Control", "Dementia"]:
        group_dir = data_dir / group
        if not group_dir.exists(): continue
        
        for cha_path in tqdm(sorted(group_dir.glob("*.cha")), desc=group):
            audio_file = None
            for ext in [".wav", ".mp3"]:
                candidate = cha_path.with_suffix(ext)
                if candidate.exists():
                    audio_file = candidate
                    break
            
            if not audio_file: continue
            
            segmented = segment_audio(audio_file, cha_path)
            if segmented:
                out_path = out_audio / f"{cha_path.stem}.wav"
                if save_processed_audio(segmented, out_path):
                    records.append({
                        "participant_id": cha_path.stem,
                        "label": group,
                        "split": "cv",
                        "audio_path": os.path.relpath(out_path, start=Path.cwd()),
                        "dataset": "pitt",
                        "age": None, "gender": None, "mmse": None
                    })
    return records

def process_adress(output_dir):
    print("\nProcessing ADReSS...")
    records = []
    data_root = Path("data/ADReSS")
    out_audio = Path(output_dir) / "adress_segmented"
    out_audio.mkdir(parents=True, exist_ok=True)

    # 1. TRAIN SET
    train_root = data_root / "ADReSS-IS2020-train" / "ADReSS-IS2020-data" / "train"
    if train_root.exists():
        for label, folder in [("Control", "cc"), ("Dementia", "cd")]:
            meta_file = train_root / f"{folder}_meta_data.txt"
            if not meta_file.exists(): continue
            
            meta_df = pd.read_csv(meta_file, sep=';')
            meta_df = clean_df_headers(meta_df)
            id_col = next((c for c in meta_df.columns if c.upper() == "ID"), meta_df.columns[0])
            
            for _, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc=f"train-{folder}"):
                pid = str(row[id_col]).strip()
                wav_path = train_root / "Full_wave_enhanced_audio" / folder / f"{pid}.wav"
                cha_path = train_root / "transcription" / folder / f"{pid}.cha"
                
                if not wav_path.exists() or not cha_path.exists(): continue
                
                segmented = segment_audio(wav_path, cha_path)
                if segmented and save_processed_audio(segmented, out_audio / f"{pid}.wav"):
                    records.append({
                        "participant_id": pid, "label": label, "split": "train",
                        "audio_path": os.path.relpath(out_audio / f"{pid}.wav", Path.cwd()),
                        "dataset": "adress",
                        "age": row.get('age'), "gender": row.get('gender'), "mmse": row.get('mmse')
                    })

    # 2. TEST SET
    test_root = data_root / "ADReSS-IS2020-test" / "ADReSS-IS2020-data" / "test"
    if test_root.exists():
        meta_file = test_root / "test_results.txt"
        if not meta_file.exists(): meta_file = test_root / "meta_data.txt"
        
        if meta_file.exists():
            test_df = pd.read_csv(meta_file, sep=';')
            test_df = clean_df_headers(test_df)
            
            id_col = next((c for c in test_df.columns if c.upper() == "ID"), test_df.columns[0])
            dx_candidates = ["dx", "label", "prediction", "class", "diagnosis"]
            dx_col = next((c for c in test_df.columns if c.lower() in dx_candidates), None)
            
            if dx_col is None and len(test_df.columns) > 3:
                dx_col = test_df.columns[3]
            
            for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="test"):
                pid = str(row[id_col]).strip()
                dx_val = str(row[dx_col]).strip() if dx_col else "1"
                label = "Control" if dx_val == '0' else "Dementia"
                
                wav_path = test_root / "Full_wave_enhanced_audio" / f"{pid}.wav"
                cha_path = test_root / "transcription" / f"{pid}.cha"
                
                if not wav_path.exists() or not cha_path.exists(): continue
                
                segmented = segment_audio(wav_path, cha_path)
                if segmented and save_processed_audio(segmented, out_audio / f"{pid}.wav"):
                    records.append({
                        "participant_id": pid, "label": label, "split": "test",
                        "audio_path": os.path.relpath(out_audio / f"{pid}.wav", Path.cwd()),
                        "dataset": "adress",
                        "age": row.get('age'), "gender": row.get('gender'), "mmse": row.get('mmse')
                    })
    return records

def process_taukadial(output_dir):
    print("\nProcessing TAUKADIAL...")
    records = []
    data_dir = Path("data/TAUKADIAL")
    out_audio = Path(output_dir) / "taukadial_segmented"
    out_audio.mkdir(parents=True, exist_ok=True)
    
    gt_candidates = [
        data_dir / "testgroundtruth.csv",
        data_dir / "TAUKADIAL-24-train" / "TAUKADIAL-24" / "train" / "groundtruth.csv"
    ]
    
    labels = {}
    clinical_meta = {}
    
    for gt_path in gt_candidates:
        if gt_path.exists():
            try:
                # Use engine='python' and sep=None to handle both ',' and ';'
                gt_df = pd.read_csv(gt_path, sep=None, engine='python')
                gt_df = clean_df_headers(gt_df)
                
                name_col = next((c for c in gt_df.columns if 'name' in c.lower()), gt_df.columns[0])
                dx_col = next((c for c in gt_df.columns if 'dx' in c.lower()), gt_df.columns[-1])
                mmse_col = next((c for c in gt_df.columns if 'mmse' in c.lower()), None)
                age_col = next((c for c in gt_df.columns if 'age' in c.lower()), None)
                sex_col = next((c for c in gt_df.columns if c.lower() in ['sex', 'gender']), None)
                
                for row in gt_df.itertuples(index=False):
                    row_dict = row._asdict()
                    fname = str(row_dict.get(name_col))
                    match = re.search(r"taukdial-(\d+)", fname)
                    pid_key = match.group(1) if match else fname
                    
                    dx = str(row_dict.get(dx_col, '')).strip()
                    labels[pid_key] = 'Control' if dx == 'NC' else 'Dementia'
                    
                    clinical_meta[pid_key] = {
                        'mmse': row_dict.get(mmse_col),
                        'age': row_dict.get(age_col),
                        'gender': row_dict.get(sex_col)
                    }
                        
            except Exception as e:
                print(f"Warning: Failed to parse ground truth {gt_path}: {e}")

    for split in ["train", "test"]:
        split_dir = data_dir / f"TAUKADIAL-24-{split}" / "TAUKADIAL-24" / split
        if not split_dir.exists(): continue
        
        p_groups = {}
        for wav in split_dir.glob("*.wav"):
            pid_match = re.search(r"taukdial-(\d+)", wav.stem)
            pid = pid_match.group(1) if pid_match else wav.stem
            if pid not in p_groups: p_groups[pid] = []
            p_groups[pid].append(wav)
            
        for pid, files in tqdm(p_groups.items(), desc=f"TAUKADIAL {split}"):
            combined = AudioSegment.empty()
            for f in sorted(files):
                try: combined += AudioSegment.from_wav(f)
                except: continue
            
            if len(combined) == 0: continue
            out_path = out_audio / f"taukdial-{pid}.wav"
            if save_processed_audio(combined, out_path):
                label = labels.get(pid, "Unknown")
                meta = clinical_meta.get(pid, {})
                records.append({
                    "participant_id": f"taukdial-{pid}", 
                    "label": label, 
                    "split": split,
                    "audio_path": os.path.relpath(out_path, start=Path.cwd()),
                    "dataset": "taukadial", 
                    "age": meta.get('age'), 
                    "gender": meta.get('gender'), 
                    "mmse": meta.get('mmse')
                })
    return records

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="processed_data")
    parser.add_argument("--datasets", nargs="+", default=["pitt", "adress", "taukadial"])
    args = parser.parse_args()

    all_records = []
    if "pitt" in args.datasets: all_records.extend(process_pitt(args.output_dir))
    if "adress" in args.datasets: all_records.extend(process_adress(args.output_dir))
    if "taukadial" in args.datasets: all_records.extend(process_taukadial(args.output_dir))
    
    if not all_records:
        print("Warning: No records processed.")
        return

    master_df = pd.DataFrame(all_records)
    out_csv = Path(args.output_dir) / "master_metadata.csv"
    master_df.to_csv(out_csv, index=False)
    
    print(f"\nPipeline Complete. {len(master_df)} files processed.")
    print(master_df.groupby(["dataset", "label"]).size())

if __name__ == "__main__":
    main()