import os
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler

from checkpoint_utils import migrate_legacy_checkpoint

# ============================================================
# CONFIG
# ============================================================

CHECKPOINT_DIR = "trained_mamba_attention_fusion"
OUTPUT_DIR = "trained_mamba_attention_fusion_migrated"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD FEATURE INFO
# ============================================================

feat_df = pd.read_csv(
    "processed_data/master_acoustic_features.csv"
)

feature_cols = [
    c for c in feat_df.columns
    if c not in [
        "participant_id",
        "audio_path",
        "label",
        "dataset",
        "split",
        "age",
        "gender",
        "mmse"
    ]
]

# ============================================================
# REBUILD SCALER
# ============================================================

scaler = StandardScaler()

scaler.fit(
    feat_df[feature_cols].values
)

# ============================================================
# MIGRATE ALL CHECKPOINTS
# ============================================================

checkpoint_files = [
    f for f in os.listdir(CHECKPOINT_DIR)
    if f.endswith(".bin")
]

print("=" * 60)
print("CHECKPOINT MIGRATION")
print("=" * 60)

for ckpt in checkpoint_files:

    old_path = os.path.join(
        CHECKPOINT_DIR,
        ckpt
    )

    new_path = os.path.join(
        OUTPUT_DIR,
        ckpt
    )

    # --------------------------------------------
    # INFER TEST DATASET NAME
    # --------------------------------------------

    dataset_name = "unknown"

    name_lower = ckpt.lower()

    for ds in ["pitt", "adress", "taukadial"]:

        if ds in name_lower:
            dataset_name = ds
            break

    print(f"\nMigrating: {ckpt}")
    print(f"Held-out dataset: {dataset_name}")

    # --------------------------------------------
    # MIGRATE
    # --------------------------------------------

    migrate_legacy_checkpoint(
        legacy_checkpoint_path=old_path,

        output_path=new_path,

        feature_ordering=feature_cols,

        scaler=scaler,

        config={
            "test_dataset": dataset_name,
            "architecture": "mamba_attention_fusion_legacy",
            "seed": 42
        }
    )

print("\n✅ ALL CHECKPOINTS MIGRATED")
