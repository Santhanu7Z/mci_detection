import pandas as pd
import json
import os
import re
from pathlib import Path
from tqdm import tqdm
from collections import Counter

# Attempt to import jiwer for WER calculation
try:
    from jiwer import wer, process_words
except ImportError:
    print("⚠️ 'jiwer' not found. Run: pip install jiwer")

# ============================================================
# CLINICAL NORMALIZATION UTILITIES
# ============================================================

def normalize_clinical_text(text, is_ref=False):
    """
    Standardized normalization for Clinical Speech.
    - Standardizes contractions.
    - Removes CHAT markers (Reference only).
    - Removes non-diagnostic fillers from BOTH streams to avoid substitution bias.
    """
    if not text: return ""
    
    # 1. Handle CHAT specific markers in Reference
    if is_ref:
        # Remove unintelligible markers (xxx, yyy, etc)
        text = re.sub(r'\b(xxx|yyy|www|v|u)\b', '', text)
        # Remove retracing/repetitions markers
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\<.*?\>', '', text)
        # Remove pause markers and duration markers like (1.2) or (.)
        text = re.sub(r'\(\d*\.?\d*\)', '', text)
        text = re.sub(r'\(\.+\)', '', text)

    text = text.lower()

    # 2. Standardize common contractions (Reduces substitution noise)
    contractions = {
        r"\bit's\b": "it is", r"\bthat's\b": "that is", 
        r"\bhe's\b": "he is", r"\bshe's\b": "she is",
        r"\bi'm\b": "i am", r"\bdon't\b": "do not",
        r"\bcan't\b": "can not", r"\bthere's\b": "there is",
        r"\bwon't\b": "will not", r"\bi've\b": "i have",
        r"\byou're\b": "you are", r"\bthey're\b": "they are"
    }
    for pattern, replacement in contractions.items():
        text = re.sub(pattern, replacement, text)

    # 3. Remove non-diagnostic fillers from BOTH (Scientific Alignment)
    # Note: 'right' removed to prevent spatial lexical loss in Cookie Theft task
    fillers = [r'\buh\b', r'\bum\b', r'\ber\b', r'\bah\b', r'\bmhm\b', r'\buh\-huh\b', r'\b\w\-\b']
    for f in fillers:
        text = re.sub(f, '', text)

    # 4. Strip all non-alpha
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # 5. Final cleaning of fragments (preserving 'a' and 'i')
    text = " ".join([w for w in text.split() if len(w) > 1 or w in ['a', 'i']])
    return text.strip()

def strip_interviewer_scaffolding(text):
    """
    Upgraded Regex-based global stripping. 
    Removes interviewer prompts anywhere in the hypothesis sequence.
    """
    scaffolds = [
        r"i am going to show you a picture", r"tell me everything you see",
        r"describe what is happening", r"tell me what is going on",
        r"tell me everything you see going on", r"anything else", 
        r"is there anything else", r"thank you", r"okay", r"all right", 
        r"the scene is", r"look at the picture", r"start from the",
        r"good morning", r"good afternoon", r"hi there"
    ]
    cleaned = text.lower()
    for s in scaffolds:
        # Using word boundaries to prevent accidental partial word deletion
        cleaned = re.sub(rf'\b{s}\b', '', cleaned)
    
    return " ".join(cleaned.split())

# ============================================================
# ASR QUALITY GUARD: HALLUCINATION DETECTOR
# ============================================================

def detect_hallucination_loop(text):
    """Identifies ASR failure modes by looking for repeating N-grams."""
    words = text.split()
    if len(words) < 10: return False
    
    # Check for 3-word loops (Trigram repetition)
    trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
    if trigrams:
        most_common = Counter(trigrams).most_common(1)[0]
        if most_common[1] > 3: 
            return True
            
    # Check for extreme word-repetition ratio
    if len(words) > 30:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.22: 
            return True
            
    return False

# ============================================================
# DATASET ACCESSORS
# ============================================================

def locate_ground_truth(p_id, dataset):
    """Dataset-aware lookup to prevent ID collisions."""
    if not hasattr(locate_ground_truth, "cache"):
        locate_ground_truth.cache = {}
    
    cache_key = f"{dataset}_{p_id}"
    if cache_key in locate_ground_truth.cache:
        return locate_ground_truth.cache[cache_key]

    search_root = Path("data/Pitt") if dataset == "pitt" else Path("data/ADReSS")
    matches = list(search_root.rglob(f"{p_id}.cha"))
    result = matches[0] if matches else None
    
    locate_ground_truth.cache[cache_key] = result
    return result

def extract_reference_from_cha(cha_path):
    """Robust multi-utterance extraction of Participant text."""
    full_text = []
    try:
        with open(cha_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
            content = f.read()
            par_utterances = re.findall(r'\*PAR:\s+(.*?)(?=\n\*|\n%|\Z)', content, re.DOTALL)
            for utt in par_utterances:
                clean_utt = re.sub(r'\d+_\d+', '', utt) 
                full_text.append(clean_utt)
        return " ".join(full_text)
    except:
        return ""

# ============================================================
# ALIGNMENT & PERFORMANCE CHECKER
# ============================================================

def check_alignment():
    print("\n" + "="*60)
    print("      SCIENTIFIC PIPELINE VALIDATION (PHASE 2 - V3.6)")
    print("      SCIENTIFIC RIGOR & TAUKADIAL INTEGRATION")
    print("="*60)
    
    meta = pd.read_csv("processed_data/master_metadata.csv")
    with open("processed_data/transcripts_cache.json", 'r') as f:
        cache = json.load(f)['transcripts']
    
    detailed_results = []
    cleaned_cache = {}
    
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Evaluating"):
        audio_path = row['audio_path']
        if audio_path not in cache: continue
        
        # Determine Dataset Context
        dataset = row['dataset']
        p_id = row['participant_id']
        raw_hyp_text = cache[audio_path].get('text', '')

        # CASE 1: TAUKADIAL (No ground truth, but needs normalization for Mamba)
        if dataset == 'taukadial':
            hyp_clin = normalize_clinical_text(strip_interviewer_scaffolding(raw_hyp_text), is_ref=False)
            if hyp_clin:
                cleaned_cache[audio_path] = hyp_clin
            continue

        # CASE 2: PITT/ADRESS (Validation possible)
        gt_path = locate_ground_truth(p_id, dataset)
        if not gt_path: continue 
        
        raw_ref_text = extract_reference_from_cha(gt_path)
        
        # 1. Standard Norm
        ref_norm = normalize_clinical_text(raw_ref_text, is_ref=True)
        hyp_norm = normalize_clinical_text(raw_hyp_text, is_ref=False)
        
        # 2. Denoised (Applied ONLY to hypothesis per scientific review)
        hyp_clin = normalize_clinical_text(strip_interviewer_scaffolding(raw_hyp_text), is_ref=False)
        
        if ref_norm and hyp_norm:
            try:
                # Outlier detection
                is_loop = detect_hallucination_loop(hyp_norm)
                ref_len, hyp_len = len(ref_norm.split()), len(hyp_norm.split())
                is_deletion = (hyp_len < ref_len * 0.15) and ref_len > 20
                is_outlier = is_loop or (hyp_len > ref_len * 2.8) or is_deletion or (hyp_len < 3 and ref_len > 10)
                
                raw_res = process_words(ref_norm, hyp_norm)
                clin_res = process_words(ref_norm, hyp_clin)

                detailed_results.append({
                    'id': p_id,
                    'dataset': dataset,
                    'raw_wer': raw_res.wer,
                    'clin_wer': clin_res.wer,
                    'is_hallucination': is_outlier,
                    'ref_text': ref_norm,
                    'hyp_text': hyp_clin
                })
                
                # Add to cleaned training cache if technically sound
                if not is_outlier:
                    cleaned_cache[audio_path] = hyp_clin

            except Exception:
                pass

    if detailed_results:
        res_df = pd.DataFrame(detailed_results)
        
        print(f"\n📊 SCIENTIFIC PERFORMANCE SUMMARY:")
        for ds in res_df['dataset'].unique():
            ds_df = res_df[res_df['dataset'] == ds]
            avg_raw = ds_df['raw_wer'].mean()
            valid_df = ds_df[~ds_df['is_hallucination']]
            avg_clin = valid_df['clin_wer'].mean()
            
            print(f"   • {ds:10}: Raw WER: {avg_raw:.4f} | Filtered Clin WER: {avg_clin:.4f}")
            print(f"                Technical failures purged: {ds_df['is_hallucination'].sum()}")

        # Update Master Metadata Cleaned
        valid_ids = set(res_df[~res_df['is_hallucination']]['id'])
        valid_ids = valid_ids.union(set(meta[meta['dataset'] == 'taukadial']['participant_id']))
        cleaned_meta = meta[meta['participant_id'].isin(valid_ids)]
        cleaned_meta.to_csv("processed_data/master_metadata_cleaned.csv", index=False)
        
        # Save cleaned cache for training (Now includes Pitt, ADReSS, and TAUKADIAL)
        with open("processed_data/cleaned_transcripts.json", 'w') as f:
            json.dump({"transcripts": cleaned_cache}, f, indent=2)
            
        print(f"\n✅ DATA GUARD: Created 'master_metadata_cleaned.csv' (N={len(cleaned_meta)})")
        print(f"✅ UNIFIED CACHE: Created 'cleaned_transcripts.json' (N={len(cleaned_cache)})")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    check_alignment()