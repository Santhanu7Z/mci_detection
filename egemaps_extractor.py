# ============================================================
# egemaps_extractor.py
# Extract eGeMAPSv02 features using openSMILE
# Outputs fixed-size clinical acoustic vectors
# ============================================================

import os
import pandas as pd
import opensmile
from tqdm import tqdm

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

AUDIO_DIR = "processed_data/audio"
OUTPUT_CSV = "processed_data/egemaps_features.csv"

# ------------------------------------------------------------
# Initialize openSMILE
# ------------------------------------------------------------

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

# ------------------------------------------------------------
# Extraction
# ------------------------------------------------------------

rows = []

audio_files = [
    f for f in os.listdir(AUDIO_DIR)
    if f.endswith(".wav")
]

print(f"Found {len(audio_files)} audio files.")

for file in tqdm(audio_files):

    path = os.path.join(AUDIO_DIR, file)

    try:
        features = smile.process_file(path)

        # features is DataFrame with 1 row
        features = features.reset_index(drop=True)

        row = features.iloc[0].to_dict()
        row["file_id"] = file.replace(".wav", "")

        rows.append(row)

    except Exception as e:
        print(f"Error processing {file}: {e}")

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

df = pd.DataFrame(rows)

df.to_csv(OUTPUT_CSV, index=False)

print("\nExtraction complete.")
print(f"Saved to {OUTPUT_CSV}")
print("Feature dimension:", df.shape[1] - 1)  # minus file_id
