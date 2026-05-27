"""
run_lodo_xai.py — v4.3
Fix: uses cpu_safe_model() from explainability_suite which replaces all
Triton RMSNorm instances with pure-PyTorch equivalents and disables
fused_add_norm on every Block.  No env vars needed.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import random
from scipy.stats import spearmanr
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from sklearn.preprocessing import StandardScaler

from explainability_suite import (
    MambaFusionEngineDynamic, run_xai, detect_arch_flags, cpu_safe_model
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# HELPERS
# ============================================================

def reconstruct_training_scaler(prep_meta):
    scaler = StandardScaler()
    scaler.mean_          = np.array(prep_meta["scaler_mean"])
    scaler.scale_         = np.array(prep_meta["scaler_scale"])
    scaler.var_           = np.array(prep_meta["scaler_var"])
    scaler.n_features_in_ = len(prep_meta["scaler_mean"])
    return scaler


def analyze_cross_seed_stability(all_word_runs, output_path):
    if len(all_word_runs) < 2:
        print("Not enough seed runs for stability analysis.")
        return

    pivoted = []
    for i, df_run in enumerate(all_word_runs):
        if "word" not in df_run.columns or "importance" not in df_run.columns:
            continue
        agg = df_run.groupby("word")["importance"].mean().rename(f"run_{i}")
        pivoted.append(agg)

    if not pivoted:
        return

    merged = pd.concat(pivoted, axis=1).fillna(0.0)
    cols   = merged.columns
    corrs  = [spearmanr(merged[cols[i]], merged[cols[j]]).correlation
              for i in range(len(cols)) for j in range(i+1, len(cols))]

    rho = float(np.mean(corrs)) if corrs else 1.0
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("Explanation Stability Analysis\n")
        f.write("===============================\n")
        f.write(f"Seed runs analysed : {len(all_word_runs)}\n")
        f.write(f"Mean Spearman rho  : {rho:.4f}\n")
    print(f"Stability report -> {output_path}  (rho={rho:.4f})")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print("EXPLAINABILITY ENGINE — ARTIFACT REGISTRY CV PIPELINE")
    print("Fix v4.3: Triton RMSNorm replaced via cpu_safe_model()")
    print("=" * 65)

    checkpoint_dir = "trained_mamba_attention_fusion_migrated"
    if not os.path.exists(checkpoint_dir):
        print(f"Directory not found: {checkpoint_dir}")
        return

    # Load base data
    df_master = pd.read_csv("processed_data/master_metadata_cleaned.csv")
    with open("processed_data/cleaned_transcripts.json") as f:
        transcripts = json.load(f)["transcripts"]
    df_master["text"] = df_master["audio_path"].map(transcripts)
    df_master = df_master.dropna(subset=["text"])
    df_master = df_master[df_master["label"].isin(["Control","Dementia"])].reset_index(drop=True)

    df_acoustic_raw = pd.read_csv("processed_data/master_acoustic_features.csv")
    join_col = "audio_path" if "audio_path" in df_acoustic_raw.columns else "participant_id"

    global_word_runs = []

    for ckpt_file in ["best_attention_fusion.bin"]:
        ckpt_path = os.path.join(checkpoint_dir, ckpt_file)
        print(f"\nLoading: {ckpt_file}")

        try:
            bundle = torch.load(ckpt_path, map_location="cpu", weights_only=False)

            if not isinstance(bundle, dict):
                print("Invalid checkpoint structure — skipping."); continue
            if bundle.get("schema_version", 0) < 2:
                print("Legacy checkpoint — run migration first."); continue

            missing = [k for k in ["experiment_config","preprocessing_metadata","model_state_dict"]
                       if k not in bundle]
            if missing:
                print(f"Missing keys: {missing}"); continue

            config     = bundle["experiment_config"]
            prep_meta  = bundle["preprocessing_metadata"]
            state_dict = bundle["model_state_dict"]
            print(f"Schema v{bundle['schema_version']} verified.")

            # Determine held-out domain
            target_domain = config.get("test_dataset", "unknown_domain")
            if target_domain.lower() in ["unknown","unknown_domain","global_pool","none"]:
                print("Generic domain — defaulting to TAUKADIAL.")
                target_domain = "taukadial"
            print(f"Held-out domain: {target_domain}")

            # Feature order + scaler
            feature_order = prep_meta["feature_ordering"]
            scaler        = reconstruct_training_scaler(prep_meta)
            print(f"Restored {len(feature_order)} acoustic features")

            # Filter test set
            df_test = df_master[
                df_master["dataset"].str.lower() == target_domain.lower()
            ].reset_index(drop=True)
            if df_test.empty:
                print("Empty test set — skipping."); continue

            df_test       = pd.merge(df_test,
                                     df_acoustic_raw[[join_col] + feature_order],
                                     on=join_col, how="inner")
            acoustic_data = scaler.transform(df_test[feature_order].values)

            # Build model — load to CPU then make Triton-safe
            tokenizer = AutoTokenizer.from_pretrained(
                config.get("tokenizer_name", "EleutherAI/gpt-neox-20b"))
            tokenizer.pad_token = tokenizer.eos_token

            backbone = MambaLMHeadModel.from_pretrained(
                config.get("backbone_name", "state-spaces/mamba-130m"))
            flags = detect_arch_flags(state_dict)

            model = MambaFusionEngineDynamic(backbone, len(feature_order), *flags)
            model.load_state_dict(state_dict, strict=True)
            model = cpu_safe_model(model)   # ← replaces Triton norms, moves to CPU
            print(f"Model device: {next(model.parameters()).device}")

            # Run XAI
            output_dir = os.path.join("xai_results", target_domain)
            os.makedirs(output_dir, exist_ok=True)

            word_df = run_xai(
                df=df_test,
                acoustic_data=acoustic_data,
                tokenizer=tokenizer,
                model=model,
                output_dir=output_dir,
                n_samples=200,
            )

            if word_df is not None and not word_df.empty:
                global_word_runs.append(word_df)

        except Exception as e:
            import traceback
            print(f"Failed on {ckpt_file}: {e}")
            traceback.print_exc()

    print("\n" + "=" * 65)
    print("CROSS-SEED STABILITY ANALYSIS")
    print("=" * 65)
    analyze_cross_seed_stability(global_word_runs,
                                 "xai_results/systemwide_stability_report.txt")


if __name__ == "__main__":
    main()
