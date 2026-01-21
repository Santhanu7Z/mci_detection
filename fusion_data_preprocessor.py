import pandas as pd
import os

# Paths
METADATA_PATH = "processed_data/metadata.csv"
ACOUSTIC_PATH = "processed_data/acoustic_features.csv"

def prepare_fusion_data():
    if not os.path.exists(METADATA_PATH) or not os.path.exists(ACOUSTIC_PATH):
        print("Error: Required CSV files missing.")
        return

    # 1. Load data
    metadata_df = pd.read_csv(METADATA_PATH)
    acoustic_df = pd.read_csv(ACOUSTIC_PATH)

    print(f"Metadata records: {len(metadata_df)}")
    print(f"Acoustic records: {len(acoustic_df)}")

    # 2. Standardize ID columns to fix KeyError
    # If acoustic data uses 'file_id', we transform it to match metadata's 'participant_id'
    if 'file_id' in acoustic_df.columns:
        # Remove '_participant' and '.wav' if present to get the raw ID
        acoustic_df['participant_id'] = acoustic_df['file_id'].str.replace('_participant', '', regex=False).str.replace('.wav', '', regex=False)
        print("✓ Standardized 'file_id' to 'participant_id' in acoustic data.")

    # 3. Merge on participant_id
    # We use an inner join to ensure only records present in both sets remain
    merged_df = pd.merge(metadata_df, acoustic_df, on="participant_id", how="inner")

    print(f"Successfully merged records: {len(merged_df)}")
    
    # 4. Diagnostics
    dropped = len(metadata_df) - len(merged_df)
    if dropped > 0:
        print(f"Note: {dropped} records from metadata were not found in acoustic features (likely too short/silent).")

    # 5. Save the master dataset for Phase 2
    # Ensure the directory exists
    os.makedirs(os.path.dirname("processed_data/"), exist_ok=True)
    merged_df.to_csv("processed_data/master_fusion_data.csv", index=False)
    print("Master dataset saved to: processed_data/master_fusion_data.csv")
    print("\nPreview of merged data:")
    print(merged_df[['participant_id', 'label', 'f0_mean', 'jitter']].head())

if __name__ == "__main__":
    prepare_fusion_data()