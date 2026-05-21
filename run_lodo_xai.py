"""
run_lodo_xai.py — Production Grade v4.0
Strict artifact validation, reproducible environments, and schema-aware execution.
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

from explainability_suite import MambaFusionEngineDynamic, run_xai, detect_arch_flags

# ============================================================
# DETERMINISTIC EXECUTION
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def reconstruct_training_scaler(prep_meta):
    scaler = StandardScaler()
    scaler.mean_ = np.array(prep_meta["scaler_mean"])
    scaler.scale_ = np.array(prep_meta["scaler_scale"])
    scaler.var_ = np.array(prep_meta["scaler_var"])
    scaler.n_features_in_ = len(prep_meta["scaler_mean"])
    return scaler

def analyze_cross_seed_stability(all_word_runs, output_path):
    if len(all_word_runs) < 2:
        print("⚠️ Not enough completed seed runs to perform cross-seed stability calculations.")
        return
    
    pivoted_runs = []
    for i, df_run in enumerate(all_word_runs):
        if "word" not in df_run.columns or "importance" not in df_run.columns:
            continue
        agg = df_run.groupby("word")["importance"].mean().rename(f"run_{i}")
        pivoted_runs.append(agg)
        
    if not pivoted_runs:
        return
        
    merged_runs = pd.concat(pivoted_runs, axis=1).fillna(0.0)
    correlations = []
    cols = merged_runs.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            rho, _ = spearmanr(merged_runs[cols[i]], merged_runs[cols[j]])
            correlations.append(rho)
            
    stability_score = np.mean(correlations) if correlations else 1.0
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f"Systematic Explanation Stability Analysis\n")
        f.write(f"=========================================\n")
        f.write(f"Number of evaluated model instances: {len(all_word_runs)}\n")
        f.write(f"Mean Inter-Run Spearman Rank Correlation (ρ): {stability_score:.4f}\n")
    print(f"📊 Stability analysis metrics saved -> {output_path} (ρ = {stability_score:.4f})")

# ============================================================
# CHECKPOINT LOADING & EXECUTION
# ============================================================
def main():
    print("=============================================================")
    print("💎 EXPLAINABILITY ENGINE: ARTIFACT REGISTRY CV PIPELINE")
    print("=============================================================")

    checkpoint_dir = "trained_mamba_attention_fusion_migrated"
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Target directory '{checkpoint_dir}' does not exist.")
        return

    checkpoints = ["best_attention_fusion.bin"]
    if not checkpoints:
        print("❌ No checkpoint binaries located.")
        return

    # Load Base Data
    df_master = pd.read_csv("processed_data/master_metadata_cleaned.csv")
    with open("processed_data/cleaned_transcripts.json") as f:
        transcripts = json.load(f)["transcripts"]
    df_master["text"] = df_master["audio_path"].map(transcripts)
    df_master = df_master.dropna(subset=["text"])
    df_master = df_master[df_master["label"].isin(["Control", "Dementia"])].reset_index(drop=True)
    
    df_acoustic_raw = pd.read_csv("processed_data/master_acoustic_features.csv")
    join_col = "audio_path" if "audio_path" in df_acoustic_raw.columns else "participant_id"

    global_word_importance_runs = []

    for ckpt_file in checkpoints:
        ckpt_path = os.path.join(checkpoint_dir, ckpt_file)
        print(f"\n🔒 Loading Experiment Artifact: {ckpt_file}")

        try:
            experiment_bundle = torch.load(ckpt_path, map_location="cpu", weights_only=False)

            # --- SCHEMA VALIDATION ---
            if not isinstance(experiment_bundle, dict):
                print("⚠️ Invalid checkpoint structure.")
                continue

            schema_version = experiment_bundle.get("schema_version", 0)
            if schema_version < 2:
                print("⚠️ Legacy checkpoint detected. Run migration utility first.")
                continue

            required_keys = ["experiment_config", "preprocessing_metadata", "model_state_dict"]
            missing = [k for k in required_keys if k not in experiment_bundle]
            if missing:
                print(f"⚠️ Missing keys: {missing}")
                continue

            config = experiment_bundle["experiment_config"]
            prep_meta = experiment_bundle["preprocessing_metadata"]
            state_dict = experiment_bundle["model_state_dict"]

            print(f"✅ Schema v{schema_version} artifact verified.")

            # --- DOMAIN EXTRACTION ---
            target_domain = config.get("test_dataset", "unknown_domain")
            
            # If the checkpoint didn't specify a test domain, force it to test on TAUKADIAL
            if target_domain.lower() in ["unknown", "unknown_domain", "global_pool", "none"]:
                print("⚠️ Generic domain detected. Forcing zero-shot evaluation on TAUKADIAL.")
                target_domain = "taukadial"
                
            print(f"🎯 Held-out domain: {target_domain}")

            # --- FEATURE ORDER RESTORATION ---
            feature_order = prep_meta["feature_ordering"]
            scaler = reconstruct_training_scaler(prep_meta)
            print(f"📊 Restored {len(feature_order)} acoustic features")

            # --- FILTER TEST SET ---
            df_test = df_master[df_master["dataset"].str.lower() == target_domain.lower()].reset_index(drop=True)
            if df_test.empty:
                print("⚠️ Empty held-out test set.")
                continue

            df_test = pd.merge(df_test, df_acoustic_raw[[join_col] + feature_order], on=join_col, how="inner")
            acoustic_data = scaler.transform(df_test[feature_order].values)

            # --- LOAD MODEL ---
            tokenizer = AutoTokenizer.from_pretrained(config.get("tokenizer_name", "EleutherAI/gpt-neox-20b"))
            tokenizer.pad_token = tokenizer.eos_token
            backbone = MambaLMHeadModel.from_pretrained(config.get("backbone_name", "state-spaces/mamba-130m"))
            flags = detect_arch_flags(state_dict)
            
            model = MambaFusionEngineDynamic(backbone, len(feature_order), *flags).to(DEVICE)
            model.load_state_dict(state_dict, strict=True)
            model.eval()

            # --- EXECUTE XAI ---
            output_dir = os.path.join("xai_results", target_domain)
            os.makedirs(output_dir, exist_ok=True)

            word_importance_df = run_xai(
                df=df_test,
                acoustic_data=acoustic_data,
                tokenizer=tokenizer,
                model=model,
                output_dir=output_dir,
                n_samples=8,
                )

            if word_importance_df is not None and not word_importance_df.empty:
                global_word_importance_runs.append(word_importance_df)

        except Exception as e:
            print(f"❌ Failed on {ckpt_file}")
            print(f"❌ Error: {str(e)}")

    print("\n" + "=" * 80)
    print("📈 AGGREGATION PARADIGM: EXPLANATION CONSISTENCY ANALYSIS")
    print("=" * 80)
    analyze_cross_seed_stability(global_word_importance_runs, "xai_results/systemwide_stability_report.txt")

if __name__ == "__main__":
    main()
