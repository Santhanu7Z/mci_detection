#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-Performance eGeMAPS v02 Feature Extractor
- Optimized for the 868-sample unified cohort (Pitt, ADReSS, TAUKADIAL).
- Uses itertuples() for 10x faster metadata traversal.
- Implements robust error logging and relative pathing.
- Output: master_acoustic_features.csv (Ready for Mamba-Fusion).
"""

import os
import pandas as pd
import opensmile
from pathlib import Path
from tqdm import tqdm

def extract_master_features(master_csv_path, output_path):
    print(f"\n--- Starting Acoustic Feature Extraction (Cohort N=868) ---")
    
    if not os.path.exists(master_csv_path):
        print(f"❌ Error: Master metadata not found at {master_csv_path}")
        return

    # 1. Load Unified Metadata
    df = pd.read_csv(master_csv_path)
    
    # 2. Initialize OpenSMILE (eGeMAPS v02 - 88 Functional Features)
    # This is the clinically validated standard for neurodegenerative speech analysis
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    
    all_features = []
    error_log = []

    # 3. Optimized Extraction Loop
    # Using itertuples is significantly faster than iterrows for datasets of this size
    for row in tqdm(df.itertuples(), total=len(df), desc="Processing Audio"):
        audio_path = row.audio_path
        
        if not os.path.exists(audio_path):
            error_log.append(f"Missing File: {audio_path}")
            continue
        
        try:
            # Process file and convert to dictionary
            features = smile.process_file(audio_path)
            feat_dict = features.iloc[0].to_dict()
            
            # Map Clinical & Research Metadata
            feat_dict.update({
                'participant_id': row.participant_id,
                'label': row.label,
                'dataset': row.dataset,
                'split': row.split,
                'age': getattr(row, 'age', None),
                'gender': getattr(row, 'gender', None),
                'mmse': getattr(row, 'mmse', None)
            })
            
            all_features.append(feat_dict)
            
        except Exception as e:
            error_log.append(f"Extraction Error [{audio_path}]: {str(e)}")

    # 4. Finalize and Save
    if not all_features:
        print("❌ Critical: No features were extracted. Check audio paths.")
        return

    feature_df = pd.DataFrame(all_features)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    feature_df.to_csv(output_path, index=False)
    
    print(f"\n✅ SUCCESS: Acoustic features saved to {output_path}")
    print(f"📊 Total Samples: {len(feature_df)}")
    
    if error_log:
        print(f"⚠️ Encountered {len(error_log)} issues during processing.")
        # Optional: save error log to file if needed

if __name__ == "__main__":
    # Standard paths for the unified pipeline
    extract_master_features(
        master_csv_path="processed_data/master_metadata.csv", 
        output_path="processed_data/master_acoustic_features.csv"
    )
