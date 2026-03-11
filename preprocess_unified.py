#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified preprocessing for Pitt, ADReSS, and TAUKADIAL.
Final TAUKADIAL Label Fix: Handles mixed delimiters (comma/semicolon) and varying column counts.
"""

import argparse
from pathlib import Path
import re
import os
import pandas as pd
from pydub import AudioSegment
from tqdm import tqdm

# ============================================================
# COMMON AUDIO UTILITIES
# ============================================================

def standardize_audio(audio_segment):
    """Standardize to 16kHz, Mono WAV."""
    if audio_segment.channels > 1:
        audio_segment = audio_segment.set_channels(1)
    if audio_segment.frame_rate != 16000:
        audio_segment = audio_segment.set_frame_rate(16000)
    return audio_segment

def save_processed_audio(audio_segment, output_path):
    """Standardizes and saves audio; returns False if audio is < 500ms."""
    audio_segment = standardize_audio(audio_segment)
    if len(audio_segment) < 500:
        return False
    audio_segment.export(output_path, format='wav')
    return True

# ============================================================
# DATASET PROCESSORS
# ============================================================

def preprocess_dementiabank(output_base_dir):
    print("\n" + "="*60 + "\nPREPROCESSING: DementiaBank (Pitt Corpus)\n" + "="*60)
    DATA_DIR = Path("data/Pitt")
    OUTPUT_DIR = Path(output_base_dir) / "dementiabank"
    AUDIO_OUT = OUTPUT_DIR / "audio"
    AUDIO_OUT.mkdir(parents=True, exist_ok=True)
    
    PAR_PATTERN = re.compile(r'^\*PAR[^\:]*\s*:\s*(.*)', flags=re.IGNORECASE)
    TS_PATTERN = re.compile(r'(\d+)_(\d+)')
    records = []

    for group in ['Control', 'Dementia']:
        group_folder = DATA_DIR / group
        if not group_folder.exists(): continue
        
        cha_files = sorted(list(group_folder.glob('*.cha')))
        for cha_path in tqdm(cha_files, desc=f"Processing {group}"):
            audio_file = None
            for ext in ['.mp3', '.wav', '.WAV', '.MP3']:
                if (cha_path.with_suffix(ext)).exists():
                    audio_file = cha_path.with_suffix(ext)
                    break
            
            if not audio_file: continue

            try:
                with open(cha_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
                    segments = []
                    for line in f:
                        match = PAR_PATTERN.match(line.strip())
                        if match:
                            text_ts = match.group(1).strip()
                            ts = TS_PATTERN.findall(text_ts)
                            if ts:
                                start, end = map(int, ts[-1])
                                segments.append((start, end))
                
                if not segments: continue
                full_audio = AudioSegment.from_file(audio_file)
                p_audio = AudioSegment.empty()
                for s, e in segments: p_audio += full_audio[s:e]
                
                out_path = AUDIO_OUT / f"{cha_path.stem}_participant.wav"
                if save_processed_audio(p_audio, out_path):
                    records.append({
                        'participant_id': cha_path.stem, 'label': group, 'split': 'train',
                        'audio_path': os.path.relpath(out_path, start=Path.cwd()), 'dataset': 'pitt'
                    })
            except: continue

    return pd.DataFrame(records)

def preprocess_adress(output_base_dir):
    print("\n" + "="*60 + "\nPREPROCESSING: ADReSS 2020\n" + "="*60)
    DATA_DIR = Path("data/ADReSS")
    OUTPUT_DIR = Path(output_base_dir) / "adress"
    AUDIO_OUT = OUTPUT_DIR / "audio"
    AUDIO_OUT.mkdir(parents=True, exist_ok=True)
    records = []
    
    # 1. Train
    train_dir = DATA_DIR / "ADReSS-IS2020-train" / "ADReSS-IS2020-data" / "train"
    for label_type, folder in [('Control', 'cc'), ('Dementia', 'cd')]:
        meta_file = train_dir / f"{folder}_meta_data.txt"
        audio_dir = train_dir / "Full_wave_enhanced_audio" / folder
        if meta_file.exists() and audio_dir.exists():
            meta = pd.read_csv(meta_file, sep=';', skipinitialspace=True, header=None)
            if "ID" in str(meta.iloc[0, 0]): meta = meta.iloc[1:].reset_index(drop=True)
            for _, row in tqdm(meta.iterrows(), desc=f"ADReSS Train ({label_type})", total=len(meta)):
                p_id = str(row[0]).strip()
                wav_path = audio_dir / f"{p_id}.wav"
                if wav_path.exists():
                    try:
                        audio = AudioSegment.from_wav(wav_path)
                        out_path = AUDIO_OUT / f"{p_id}.wav"
                        if save_processed_audio(audio, out_path):
                            records.append({
                                'participant_id': p_id, 'label': label_type, 'split': 'train',
                                'audio_path': os.path.relpath(out_path, start=Path.cwd()), 'dataset': 'adress',
                                'age': str(row[1]).strip(), 'gender': 'Male' if str(row[2]).strip() == '0' else 'Female'
                            })
                    except: continue

    # 2. Test
    test_dir = DATA_DIR / "ADReSS-IS2020-test" / "ADReSS-IS2020-data" / "test"
    test_results = test_dir / "test_results.txt"
    test_audio = test_dir / "Full_wave_enhanced_audio"
    if test_results.exists() and test_audio.exists():
        test_df = pd.read_csv(test_results, sep=';', skipinitialspace=True, header=None)
        if "ID" in str(test_df.iloc[0, 0]): test_df = test_df.iloc[1:].reset_index(drop=True)
        for _, row in tqdm(test_df.iterrows(), desc="ADReSS Test", total=len(test_df)):
            p_id = str(row[0]).strip()
            wav_path = test_audio / f"{p_id}.wav"
            if wav_path.exists():
                try:
                    audio = AudioSegment.from_wav(wav_path)
                    out_path = AUDIO_OUT / f"{p_id}.wav"
                    if save_processed_audio(audio, out_path):
                        label_code = str(row[3]).strip()
                        records.append({
                            'participant_id': p_id, 'label': 'Control' if label_code == '0' else 'Dementia',
                            'split': 'test', 'audio_path': os.path.relpath(out_path, start=Path.cwd()), 
                            'dataset': 'adress', 'age': str(row[1]).strip(), 
                            'gender': 'Male' if str(row[2]).strip() == '0' else 'Female', 'mmse': str(row[4]).strip()
                        })
                except: continue

    return pd.DataFrame(records)

def preprocess_taukadial(output_base_dir):
    print("\n" + "="*60 + "\nPREPROCESSING: TAUKADIAL\n" + "="*60)
    DATA_DIR = Path("data/TAUKADIAL")
    OUTPUT_DIR = Path(output_base_dir) / "taukadial"
    AUDIO_OUT = OUTPUT_DIR / "audio"
    AUDIO_OUT.mkdir(parents=True, exist_ok=True)
    
    # 1. Aggressive Label Collection with Dynamic Delimiter and Column Support
    labels = {}
    gt_files = [
        DATA_DIR / "TAUKADIAL-24-train/TAUKADIAL-24/train/groundtruth.csv",
        DATA_DIR / "testgroundtruth.csv"
    ]
    
    for gt_path in gt_files:
        if not gt_path.exists(): continue
        try:
            # Use sep=None to auto-detect semicolon vs comma
            gt_df = pd.read_csv(gt_path, sep=None, engine='python', skipinitialspace=True)
            
            # Identify column indices by name (agnostic to column count/order)
            cols = [c.strip().lower() for c in gt_df.columns]
            name_idx = next((i for i, c in enumerate(cols) if 'name' in c), 0)
            dx_idx = next((i for i, c in enumerate(cols) if 'dx' in c), -1)
            mmse_idx = next((i for i, c in enumerate(cols) if 'mmse' in c), -1)
            
            count = 0
            for _, row in gt_df.iterrows():
                filename = str(row.iloc[name_idx]).strip()
                dx = str(row.iloc[dx_idx]).strip() if dx_idx != -1 else "Unknown"
                mmse = str(row.iloc[mmse_idx]).strip() if mmse_idx != -1 else None
                
                # Extract pid: e.g., taukdial-001 from taukdial-001-1.wav
                pid_match = re.search(r'(taukdial-\d+)', filename)
                if pid_match:
                    pid = pid_match.group(1)
                    # NC/NC-Control -> Control, MCI/Dementia -> Dementia
                    clean_label = 'Control' if dx in ['NC', '0', 'HC', 'Control'] else 'Dementia'
                    labels[pid] = {'label': clean_label, 'mmse': mmse}
                    count += 1
            print(f"Loaded {count} unique labels from {gt_path.name}")
        except Exception as e:
            print(f"Warning: Failed to parse {gt_path}: {e}")

    # 2. Process Files
    records = []
    for split in ['train', 'test']:
        split_dir = DATA_DIR / f"TAUKADIAL-24-{split}" / "TAUKADIAL-24" / split
        if not split_dir.exists(): continue
        
        # Group chunks
        p_groups = {}
        for wav in split_dir.glob("*.wav"):
            pid_match = re.search(r'(taukdial-\d+)', wav.stem)
            pid = pid_match.group(1) if pid_match else wav.stem
            if pid not in p_groups: p_groups[pid] = []
            p_groups[pid].append(wav)
            
        for pid, files in tqdm(p_groups.items(), desc=f"TAUKADIAL {split}"):
            meta = labels.get(pid, {'label': 'Unknown', 'mmse': None})
            
            combined = AudioSegment.empty()
            for f in sorted(files):
                try: combined += AudioSegment.from_wav(f)
                except: continue
            
            out_path = AUDIO_OUT / f"{pid}.wav"
            if save_processed_audio(combined, out_path):
                records.append({
                    'participant_id': pid, 'label': meta['label'], 'split': split,
                    'audio_path': os.path.relpath(out_path, start=Path.cwd()), 
                    'dataset': 'taukadial', 'mmse': meta['mmse']
                })

    return pd.DataFrame(records)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['pitt', 'adress', 'taukadial'])
    parser.add_argument('--output_dir', type=str, default='processed_data')
    args = parser.parse_args()

    results = []
    if 'pitt' in args.datasets: results.append(preprocess_dementiabank(args.output_dir))
    if 'adress' in args.datasets: results.append(preprocess_adress(args.output_dir))
    if 'taukadial' in args.datasets: results.append(preprocess_taukadial(args.output_dir))

    if results:
        master_df = pd.concat(results, ignore_index=True)
        cols = ['participant_id', 'label', 'split', 'audio_path', 'dataset', 'age', 'gender', 'mmse']
        for col in cols:
            if col not in master_df.columns: master_df[col] = None
        
        master_df = master_df[cols]
        master_df.to_csv(Path(args.output_dir) / "master_metadata.csv", index=False)
        print("\n" + "="*60 + "\nUNIFIED PIPELINE COMPLETE\n" + "="*60)
        print(master_df.groupby(['dataset', 'label']).size())

if __name__ == "__main__":
    main()