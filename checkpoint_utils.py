# checkpoint_utils.py

import os
import torch
import sklearn
import transformers
import platform
import sys
from datetime import datetime

SCHEMA_VERSION = 2

def build_environment_metadata():
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "sklearn_version": sklearn.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "timestamp": datetime.utcnow().isoformat()
    }

def create_experiment_bundle(
    model,
    scaler,
    feature_ordering,
    config,
    checkpoint_path,
    tokenizer_name="EleutherAI/gpt-neox-20b",
    backbone_name="state-spaces/mamba-130m"
):
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "experiment_config": {
            **config,
            "tokenizer_name": tokenizer_name,
            "backbone_name": backbone_name
        },
        "preprocessing_metadata": {
            "feature_ordering": feature_ordering,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "scaler_var": scaler.var_.tolist(),
        },
        "environment_metadata": build_environment_metadata(),
        "model_state_dict": model.state_dict()
    }

    torch.save(bundle, checkpoint_path)
    print(f"✅ Saved artifact bundle → {checkpoint_path}")

def migrate_legacy_checkpoint(
    legacy_checkpoint_path,
    output_path,
    feature_ordering,
    scaler,
    config
):
    """
    Converts old state_dict-only checkpoints into modern experiment bundles.
    """
    print(f"🔄 Migrating legacy checkpoint: {legacy_checkpoint_path}")

    state_dict = torch.load(legacy_checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        print("⚠️ Already modern format.")
        return

    bundle = {
        "schema_version": SCHEMA_VERSION,
        "experiment_config": {
            **config,
            "tokenizer_name": "EleutherAI/gpt-neox-20b",
            "backbone_name": "state-spaces/mamba-130m",
            "migrated_from_legacy": True
        },
        "preprocessing_metadata": {
            "feature_ordering": feature_ordering,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "scaler_var": scaler.var_.tolist(),
        },
        "environment_metadata": build_environment_metadata(),
        "model_state_dict": state_dict
    }

    torch.save(bundle, output_path)
    print(f"✅ Migrated checkpoint saved → {output_path}")
