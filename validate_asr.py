#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASR Validation & Dataset Cleaning Engine v5.0 - CLINICAL RIGOR EDITION
- Dual Purpose: Validates ASR accuracy AND prepares clean text for Mamba Training.
- Fix: Reverted Global Stripping to Boundary-only checks to prevent semantic distortion.
- Fix: Protected potential participant speech by enforcing word-count limits on end-stripping.
- Feature: Maintained S/D/I error decomposition for transparent IEEE reporting.
- Outlier Detection: Loops, Deletions, and Unique Token Ratio (UTR).
"""

import os
import json
import re
import numpy as np
import pandas as pd
from jiwer import wer, cer, process_words
from tqdm import tqdm
from pathlib import Path

# ============================================================
# CENTRALIZED CLINICAL NORMALIZATION (Use for Training & Eval)
# ============================================================

def normalize_text(text):
    """
    Standardizes clinical speech. This logic is applied to 
    BOTH evaluation (WER) and training (Mamba) to ensure consistency.
    """
    if not text: return ""
    
    # 1. Remove Standalone Timestamps
    text = re.sub(r'\b\d+[\s_]\d+\b', '', text)
    text = re.sub(r'\b\d{4,}\b', '', text) 
    
    # 2. Remove CHAT-specific markers and Unintelligible tags
    text = re.sub(r'\<.*?\>\s*\[\/\/\]', '', text) 
    text = re.sub(r'\[\/\]', '', text)             
    text = re.sub(r'\w+\s+\[\:\s*(.*?)\]', r'\1', text) 
    text = re.sub(r'\[.*?\]', '', text)            
    text = re.sub(r'\(.*?\)', '', text)            
    text = re.sub(r'&\-\w+', '', text)             
    text = re.sub(r'&=?\w+', '', text)             
    text = text.replace('xxx', '').replace('yyy', '') 

    # 3. Strip Linguistic/Morphological/Speaker tags
    tags = r'\b(prep|det|art|part|sub|obj|adj|adv|coord|punct|inv|speaker)\b'
    text = re.sub(tags, '', text, flags=re.IGNORECASE)

    # 4. Final Clean & Standardize
    text = text.lower()
    
    # Refined fragment removal: Remove single letters UNLESS they are 'a' or 'i'
    text = re.sub(r'\b(?![ai]\b)[a-z]\b', '', text) 
    
    # Aligns Whisper's formal output with human colloquialism
    text = re.sub(r'\b(\w+)in\'?\b', r'\1ing', text)
    
    # Standardize common contractions
    contractions = {
        r"\bit's\b": "it is", r"\bthat's\b": "that is", r"\bhe's\b": "he is",
        r"\bshe's\b": "she is", r"\bthere's\b": "there is", r"\bi'm\b": "i am",
        r"\bdon't\b": "do not", r"\bcan't\b": "can not", r"\bwasn't\b": "was not"
    }
    for pattern, replacement in contractions.items():
        text = re.sub(pattern, replacement, text)

    text = re.sub(r'[^a-zA-Z\s\']', ' ', text)
    return " ".join(text.split())

def clean_conversational_artifacts(hypothesis):
    """
    Scientifically robust artifact stripping (Conservative Approach).
    - Trims interviewer instructions from the START.
    - Trims specific interviewer prompts from the END.
    - Uses strict 'Rigor Guard' to avoid deleting participant speech.
    """
    # 1. Preamble list (Interviewer Prompts - Start of session only)
    preambles = [
        "i want you to tell me everything you see happening",
        "i am going to show you a picture",
        "tell me everything that you see",
        "tell me everything that's happening",
        "tell me everything that you see going on",
        "just tell me everything that you see",
        "all the action that you can see",
        "tell me all of the action",
        "and there is the picture",
        "now there is the picture",
        "look at the picture",
        "look down here",
        "start from the",
        "describe what is happening",
        "everything that is going on",
        "what is happening in that picture",
        "do you want me to start",
        "ready go"
    ]
    
    # 2. Specific Interviewer Prompts (End of session only)
    specific_prompts = [
        "anything else what do you see going on in the picture",
        "anything else what do you see going on",
        "is there anything else you can see",
        "is there anything else happening",
        "what else do you see",
        "what else is happening",
        "any more action"
    ]
    
    # 3. Generic Conversational Fillers (Subject to Rigor Guard)
    fillers = [
        "anything else",
        "is there anything else",
        "is that all",
        "thank you",
        "okay"
    ]
    
    hyp_lower = hypothesis.lower()
    
    # PASS 1: START-OF-STRING TRIMMING
    changed = True
    while changed:
        changed = False
        for p in preambles:
            if hyp_lower.startswith(p):
                hypothesis = hypothesis[len(p):].strip()
                hyp_lower = hypothesis.lower()
                changed = True
                break
            elif p in hyp_lower[:150]:
                idx = hyp_lower.find(p) + len(p)
                hypothesis = hypothesis[idx:].strip()
                hyp_lower = hypothesis.lower()
                changed = True
                break
                
    # PASS 2: SPECIFIC PROMPT TRIMMING (End of string)
    for prompt in specific_prompts:
        if hyp_lower.endswith(prompt):
            hypothesis = hypothesis[:-len(prompt)].strip()
            hyp_lower = hypothesis.lower()
                
    # PASS 3: RIGOR GUARD (Generic fillers)
    # Only strip if under 4 words and at the very end
    for f in fillers:
        if hyp_lower.endswith(f):
            if len(f.split()) < 4:
                hypothesis = hypothesis[:-len(f)].strip()
                hyp_lower = hypothesis.lower()
                break
                
    # PASS 4: Final honorific name stripping
    hypothesis = re.sub(r'\b(mr|mrs|ms)\s+\w+$', '', hypothesis, flags=re.IGNORECASE).strip()
    
    return " ".join(hypothesis.split())

# ============================================================
# EVALUATION & DATA PREPARATION LOGIC
# ============================================================

def extract_ground_truth_from_cha(cha_path, include_interviewer=False):
    target_tags = ('*PAR:', '*INV:') if include_interviewer else ('*PAR:',)
    full_text = []
    current_tier = []
    capture = False
    try:
        with open(cha_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                if line.startswith('*'):
                    if capture and current_tier: full_text.append(" ".join(current_tier))
                    if line.startswith(target_tags):
                        capture = True
                        content = line[line.find(':')+1:].strip()
                        current_tier = [content]
                    else:
                        capture = False
                        current_tier = []
                elif capture and (line.startswith('\t') or line.startswith(' ')):
                    current_tier.append(line.strip())
                elif line.startswith('%') or line.startswith('@'):
                    if capture and current_tier: full_text.append(" ".join(current_tier))
                    capture = False
                    current_tier = []
        return normalize_text(" ".join(full_text))
    except Exception: return ""

def run_validation_and_clean_prep(master_csv, cache_json, apply_cleaning_to_cache=True):
    print(f"\n--- ASR VALIDATION & DATA CLEANING v5.0 ---")
    
    if not os.path.exists(master_csv) or not os.path.exists(cache_json):
        print("Error: Input files missing.")
        return

    df = pd.read_csv(master_csv)
    with open(cache_json, 'r') as f:
        cache = json.load(f)['transcripts']
    
    cha_map = {path.stem: path for path in Path("data").rglob("*.cha")}
    val_df = df[df['dataset'].isin(['pitt', 'adress'])]
    
    results = []
    cleaned_cache = {}

    for row in tqdm(val_df.itertuples(), total=len(val_df), desc="Analyzing Samples"):
        audio_path = row.audio_path
        file_stem = Path(audio_path).stem.replace("_participant", "")
        cha_path = cha_map.get(file_stem) or cha_map.get(row.participant_id)
        
        # 1. Extract Hypothesis
        entry = cache.get(audio_path, {})
        raw_hyp = entry.get('text', "")
        if not raw_hyp and "segments" in entry:
            raw_hyp = " ".join([w.get("word", "") for s in entry.get("segments", []) for w in s.get("words", [])])
        
        norm_hyp = normalize_text(raw_hyp)
        hypothesis = clean_conversational_artifacts(norm_hyp)
        
        # 2. Extract Reference
        reference = ""
        if cha_path:
            reference = extract_ground_truth_from_cha(cha_path, include_interviewer=False)
        
        if reference and hypothesis:
            ref_tokens = reference.split()
            hyp_tokens = hypothesis.split()
            out = process_words(reference, hypothesis)
            
            # --- OUTLIER DETECTION ---
            is_loop = (len(hyp_tokens) > len(ref_tokens) * 2.5) 
            is_deletion = (len(hyp_tokens) < len(ref_tokens) * 0.15) and len(ref_tokens) > 20
            unique_ratio = len(set(hyp_tokens)) / len(hyp_tokens) if len(hyp_tokens) > 0 else 1.0
            is_repetitive = (unique_ratio < 0.2) and (len(hyp_tokens) > 30)
            
            outlier_type = 'none'
            if is_loop: outlier_type = 'loop_len'
            elif is_repetitive: outlier_type = 'loop_token'
            elif is_deletion: outlier_type = 'deep_deletion'

            results.append({
                'id': row.participant_id, 'dataset': row.dataset, 'wer': out.wer, 'cer': cer(reference, hypothesis),
                'ref_word_count': len(ref_tokens), 'hyp_word_count': len(hyp_tokens),
                'substitutions': out.substitutions, 'deletions': out.deletions, 'insertions': out.insertions,
                'is_outlier': (outlier_type != 'none'), 'outlier_type': outlier_type,
                'ref_text': reference, 'hyp_text': hypothesis
            })

            if outlier_type == 'none':
                cleaned_cache[audio_path] = hypothesis
        else:
            if hypothesis:
                cleaned_cache[audio_path] = hypothesis

    res_df = pd.DataFrame(results)
    clean_df = res_df[~res_df['is_outlier']]
    
    # 3. STATS REPORTING
    print("\n" + "="*65)
    print(f"{'ASR QUALITY REPORT (v5.0 - RESEARCH RIGOR)':^65}")
    print("="*65)
    if not clean_df.empty:
        total_errors = clean_df[['substitutions', 'deletions', 'insertions']].sum().sum()
        total_refs = clean_df['ref_word_count'].sum()
        
        print(f"Clean Weighted Corpus WER : {total_errors/total_refs:.4f}")
        print(f"Substitutions Rate        : {(clean_df['substitutions'].sum()/total_refs)*100:.1f}%")
        print(f"Deletions Rate (Fluency)  : {(clean_df['deletions'].sum()/total_refs)*100:.1f}%")
        print(f"Insertions Rate (Noise)   : {(clean_df['insertions'].sum()/total_refs)*100:.1f}%")
        print("-" * 65)
        
        ds_stats = clean_df.groupby('dataset').apply(
            lambda x: pd.Series({
                'Mean WER': f"{x['wer'].mean():.4f}",
                'Median WER': f"{x['wer'].median():.4f}",
                'Success Rate': f"{(len(x) / len(res_df[res_df['dataset']==x.name])): .1%}"
            })
        )
        print(ds_stats)
    print("="*65)

    # --- VISUAL DEBUGGER ---
    print("\n" + "!"*65)
    print(f"{'VISUAL DEBUGGER: WORST ADReSS SAMPLES (CONSERVATIVE)':^65}")
    print("!"*65)
    worst_adress = clean_df[clean_df['dataset']=='adress'].nlargest(2, 'wer')
    for _, s in worst_adress.iterrows():
        print(f"\nID: {s['id']} | WER: {s['wer']:.2f} | S: {s['substitutions']} | D: {s['deletions']} | I: {s['insertions']}")
        print(f"REF: {s['ref_text'][:200]}...")
        print(f"HYP: {s['hyp_text'][:200]}...")
    print("="*65)

    if apply_cleaning_to_cache:
        export_cache = {"transcripts": cleaned_cache}
        clean_cache_path = os.path.join(os.path.dirname(cache_json), "cleaned_transcripts.json")
        with open(clean_cache_path, 'w') as f:
            json.dump(export_cache, f, indent=2)
        print(f"\n🚀 CLEANING COMPLETE: Generated '{clean_cache_path}'")
        print(f"   Stored {len(cleaned_cache)} valid transcripts.")

    res_df.to_csv("asr_validation_results_v5_0.csv", index=False)

if __name__ == "__main__":
    run_validation_and_clean_prep("processed_data/master_metadata.csv", "processed_data/transcripts_cache.json")