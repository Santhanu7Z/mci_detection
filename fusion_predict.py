import os
import json
import argparse
import numpy as np
import pandas as pd
import librosa
import whisper
import torch
import torch.nn as nn
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from datetime import datetime

# --- CONFIGURATION ---
DATA_DIR = "processed_data"
MODEL_DIR = "trained_mamba_attention_fusion"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_attention_fusion.bin")
ACOUSTIC_CSV_PATH = os.path.join(DATA_DIR, "acoustic_features.csv")
TOKENIZER_ID = "EleutherAI/gpt-neox-20b"
PREDICTION_LOG = "prediction_results.json"

# ============================================================
# 1. MODEL ARCHITECTURE (Must Match mamba_attention_fusion.py)
# ============================================================

class AttentionFusionBlock(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1, batch_first=True)
        self.layernorm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, x):
        attn_out, weights = self.mha(x, x, x)
        x = self.layernorm(x + attn_out)
        x = self.layernorm(x + self.ffn(x))
        return x, weights

class MambaAttentionFusion(nn.Module):
    def __init__(self, mamba_model, acoustic_dim, num_labels):
        super().__init__()
        self.mamba = mamba_model.backbone
        text_dim = mamba_model.config.d_model # 768
        fusion_dim = 256
        
        # Projections
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.audio_proj = nn.Sequential(
            nn.Linear(acoustic_dim, 128),
            nn.GELU(),
            nn.Linear(128, fusion_dim)
        )
        
        # Attention Fusion
        self.fusion_block = AttentionFusionBlock(embed_dim=fusion_dim)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim * 2, 128),
            nn.GELU(),
            nn.Linear(128, num_labels)
        )

    def forward(self, input_ids, acoustic_features):
        text_out = self.mamba(input_ids).mean(dim=1)
        text_proj = self.text_proj(text_out)
        
        audio_out = self.audio_proj(acoustic_features)
        
        # Stack for attention [Batch, 2, 256]
        combined = torch.stack([text_proj, audio_out], dim=1)
        fused_seq, attn_weights = self.fusion_block(combined)
        
        fused_flat = fused_seq.view(fused_seq.size(0), -1)
        logits = self.classifier(fused_flat)
        return logits, attn_weights

# ============================================================
# 2. FEATURE EXTRACTION (On-the-Fly) - FIXED VERSION
# ============================================================

def extract_live_acoustic_features(audio_path):
    """
    Extracts the exact same features used in training from a new audio file.
    MUST MATCH: duration, f0_mean, f0_std, jitter, shimmer, hnr, 
                spectral_centroid, spectral_bandwidth, spectral_rolloff, zcr, 
                mfcc_1...mfcc_13
    Total: 23 features
    """
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        
        # --- 1. Duration ---
        duration = librosa.get_duration(y=y, sr=sr)
        
        # --- 2. Voice Quality (F0, Jitter, Shimmer, HNR) ---
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7'), 
            hop_length=512
        )
        f0_clean = f0[~np.isnan(f0)]
        
        if len(f0_clean) < 2:
            f0_mean, f0_std, jitter, shimmer, hnr = 0.0, 0.0, 0.0, 0.0, 0.0
        else:
            f0_mean = np.mean(f0_clean)
            f0_std = np.std(f0_clean)
            
            # Jitter
            periods = 1.0 / f0_clean
            jitter = np.mean(np.abs(np.diff(periods))) / np.mean(periods)
            
            # Shimmer
            peaks, _ = find_peaks(y, distance=int(sr//f0_mean) if f0_mean > 0 else 100)
            amplitudes = np.abs(y[peaks])
            shimmer = np.mean(np.abs(np.diff(amplitudes))) / np.mean(amplitudes) if len(amplitudes) > 1 else 0.0
            
            # HNR
            autocorr = librosa.autocorrelate(y)
            max_autocorr = np.max(autocorr[sr//400:sr//50]) if len(autocorr) > sr//50 else 0
            hnr = 10 * np.log10(max_autocorr / (np.abs(np.max(autocorr) - max_autocorr) + 1e-6)) if max_autocorr > 0 else 0.0

        # --- 3. Spectral Features ---
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))  # FIXED: Added
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))      # FIXED: Added
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # --- 4. MFCCs ---
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)

        # --- 5. Pack into list in EXACT training order ---
        features = [
            duration, f0_mean, f0_std, jitter, shimmer, hnr,
            centroid, bandwidth, rolloff, zcr
        ]
        features.extend(mfccs_mean.tolist())
        
        return np.array(features).reshape(1, -1)
    
    except Exception as e:
        print(f"Error extracting audio features: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_fitted_scaler():
    """Reconstructs the StandardScaler from the training CSV."""
    if not os.path.exists(ACOUSTIC_CSV_PATH):
        raise FileNotFoundError(f"Acoustic features CSV not found at {ACOUSTIC_CSV_PATH}")
    
    df = pd.read_csv(ACOUSTIC_CSV_PATH)
    # FIXED: Include duration in scaling (match training)
    feat_cols = [c for c in df.columns if c not in ['participant_id', 'file_id']]
    data = df[feat_cols].values
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler, len(feat_cols)

# ============================================================
# 3. EXPLAINABILITY FUNCTIONS
# ============================================================

def analyze_acoustic_biomarkers(raw_features):
    """
    Provides clinical interpretation of acoustic features.
    """
    feature_names = [
        'duration', 'f0_mean', 'f0_std', 'jitter', 'shimmer', 'hnr',
        'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'zcr'
    ] + [f'mfcc_{i+1}' for i in range(13)]
    
    analysis = []
    features_flat = raw_features.flatten()
    
    # Clinical significance thresholds (based on research)
    if features_flat[3] > 0.01:  # Jitter
        analysis.append("⚠ Elevated jitter detected (vocal instability)")
    if features_flat[4] > 0.05:  # Shimmer
        analysis.append("⚠ Elevated shimmer detected (amplitude variability)")
    if features_flat[5] < 10:    # HNR
        analysis.append("⚠ Low harmonics-to-noise ratio (voice quality degradation)")
    if features_flat[2] > 50:    # F0 std
        analysis.append("⚠ High pitch variability detected")
    
    if not analysis:
        analysis.append("✓ All acoustic biomarkers within normal range")
    
    return analysis

def generate_explanation(transcription, attention_weights, acoustic_analysis):
    """
    Generates a detailed clinical explanation.
    """
    text_w = attention_weights[0, 0, 0].item()
    audio_w = attention_weights[0, 0, 1].item()
    
    explanation = {
        "linguistic_analysis": {
            "weight": f"{text_w*100:.1f}%",
            "transcript_sample": transcription[:200] + "..." if len(transcription) > 200 else transcription,
            "indicators": []
        },
        "acoustic_analysis": {
            "weight": f"{audio_w*100:.1f}%",
            "biomarkers": acoustic_analysis
        },
        "dominant_modality": "Linguistic (Text)" if text_w > audio_w else "Acoustic (Voice)"
    }
    
    # Simple linguistic indicators
    transcript_lower = transcription.lower()
    if any(word in transcript_lower for word in ['um', 'uh', 'er', 'ah']):
        explanation["linguistic_analysis"]["indicators"].append("Disfluencies detected")
    if len(transcription.split()) < 20:
        explanation["linguistic_analysis"]["indicators"].append("Short response length")
    
    return explanation

# ============================================================
# 4. PREDICTION PIPELINE
# ============================================================

def predict(audio_path, whisper_size, threshold, save_log, device):
    print(f"\n{'='*60}")
    print(f"MULTIMODAL DEMENTIA DETECTION SYSTEM")
    print(f"{'='*60}")
    print(f"Processing: {os.path.basename(audio_path)}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # --- Pre-flight Checks ---
    if not os.path.exists(audio_path):
        print(f"❌ Error: Audio file not found at {audio_path}")
        return
    
    if not os.path.exists(BEST_MODEL_PATH):
        print(f"❌ Error: Model not found at {BEST_MODEL_PATH}")
        print("Please train the fusion model first using mamba_attention_fusion.py")
        return
    
    # --- A. Acoustic Stream ---
    print("🔊 STEP 1: Extracting Acoustic Biomarkers...")
    raw_features = extract_live_acoustic_features(audio_path)
    if raw_features is None:
        print("❌ Failed to extract acoustic features")
        return
    
    scaler, expected_dim = get_fitted_scaler()
    if raw_features.shape[1] != expected_dim:
        print(f"❌ Feature dimension mismatch! Expected {expected_dim}, got {raw_features.shape[1]}")
        return
    
    scaled_features = scaler.transform(raw_features)
    acoustic_tensor = torch.tensor(scaled_features, dtype=torch.float32).to(device)
    print(f"   ✓ Extracted {acoustic_tensor.shape[1]} features")
    
    acoustic_analysis = analyze_acoustic_biomarkers(raw_features)

    # --- B. Text Stream ---
    print(f"\n📝 STEP 2: Transcribing with Whisper ({whisper_size})...")
    w_model = whisper.load_model(whisper_size, device=device)
    result = w_model.transcribe(
        audio_path, 
        fp16=(device.type == 'cuda'),
        temperature=0.0,
        condition_on_previous_text=False,
        best_of=5,
        beam_size=5
    )
    transcription = result['text'].strip()
    print(f"   ✓ Transcript: \"{transcription[:100]}{'...' if len(transcription) > 100 else ''}\"")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(
        transcription, 
        return_tensors="pt", 
        truncation=True, 
        padding="max_length", 
        max_length=512
    ).to(device)

    # --- C. Mamba-Fusion Inference ---
    print(f"\n🧠 STEP 3: Running Multimodal Fusion Inference...")
    base_mamba = MambaLMHeadModel.from_pretrained("state-spaces/mamba-130m")
    model = MambaAttentionFusion(
        base_mamba, 
        acoustic_dim=acoustic_tensor.shape[1], 
        num_labels=2
    ).to(device)
    
    # Load Weights
    state_dict = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            logits, attn_weights = model(inputs['input_ids'], acoustic_tensor)
            probs = torch.softmax(logits, dim=-1)

    # --- D. Results ---
    p_control = probs[0][0].item()
    p_dementia = probs[0][1].item()
    pred = "Dementia" if p_dementia > threshold else "Control"

    print(f"\n{'='*60}")
    print(f"🎯 FINAL DIAGNOSIS: {pred.upper()}")
    print(f"{'='*60}")
    print(f"Confidence Scores:")
    print(f"   • Dementia:  {p_dementia:.4f} ({p_dementia*100:.1f}%)")
    print(f"   • Control:   {p_control:.4f} ({p_control*100:.1f}%)")
    print(f"   • Threshold: {threshold}")
    
    # --- E. Explainability ---
    explanation = generate_explanation(transcription, attn_weights, acoustic_analysis)
    
    print(f"\n{'='*60}")
    print(f"📊 EXPLAINABILITY REPORT")
    print(f"{'='*60}")
    
    print(f"\n🔍 Modality Contribution:")
    print(f"   • Linguistic Features: {explanation['linguistic_analysis']['weight']}")
    print(f"   • Acoustic Features:   {explanation['acoustic_analysis']['weight']}")
    print(f"   • Dominant Signal:     {explanation['dominant_modality']}")
    
    print(f"\n🎤 Acoustic Biomarker Analysis:")
    for indicator in explanation['acoustic_analysis']['biomarkers']:
        print(f"   {indicator}")
    
    print(f"\n💬 Linguistic Analysis:")
    if explanation['linguistic_analysis']['indicators']:
        for indicator in explanation['linguistic_analysis']['indicators']:
            print(f"   • {indicator}")
    else:
        print(f"   • No significant linguistic markers detected")
    
    print(f"\n📄 Full Transcript:")
    print(f"   \"{transcription}\"")
    print(f"{'='*60}\n")
    
    # --- F. Save Results ---
    if save_log:
        result_entry = {
            "timestamp": datetime.now().isoformat(),
            "file": os.path.basename(audio_path),
            "prediction": pred,
            "confidence": {
                "dementia": round(p_dementia, 4),
                "control": round(p_control, 4)
            },
            "threshold": threshold,
            "modality_weights": {
                "text": round(explanation['linguistic_analysis']['weight'].rstrip('%'), 2),
                "audio": round(explanation['acoustic_analysis']['weight'].rstrip('%'), 2)
            },
            "transcript": transcription,
            "acoustic_findings": explanation['acoustic_analysis']['biomarkers'],
            "linguistic_findings": explanation['linguistic_analysis']['indicators']
        }
        
        # Append to log file
        log_data = []
        if os.path.exists(PREDICTION_LOG):
            with open(PREDICTION_LOG, 'r') as f:
                log_data = json.load(f)
        
        log_data.append(result_entry)
        
        with open(PREDICTION_LOG, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"✓ Results saved to {PREDICTION_LOG}\n")

# ============================================================
# 5. BATCH PREDICTION (BONUS)
# ============================================================

def batch_predict(audio_dir, whisper_size, threshold, device):
    """Process multiple audio files in a directory."""
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith(('.wav', '.mp3'))]
    
    if not audio_files:
        print(f"No audio files found in {audio_dir}")
        return
    
    print(f"\n🔄 Batch Processing: {len(audio_files)} files\n")
    
    for audio_file in audio_files:
        audio_path = os.path.join(audio_dir, audio_file)
        predict(audio_path, whisper_size, threshold, save_log=True, device=device)
        print("\n" + "-"*60 + "\n")

# ============================================================
# 6. MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multimodal Dementia Detection using Mamba-Fusion"
    )
    parser.add_argument(
        "audio_path", 
        type=str, 
        help="Path to WAV file or directory for batch processing"
    )
    parser.add_argument(
        "--whisper", 
        type=str, 
        default="tiny.en", 
        help="Whisper model size (tiny.en, base.en, small.en, medium.en)"
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.5, 
        help="Classification threshold (default: 0.5)"
    )
    parser.add_argument(
        "--batch", 
        action="store_true", 
        help="Enable batch mode for directory processing"
    )
    parser.add_argument(
        "--no-log", 
        action="store_true", 
        help="Disable saving results to JSON log"
    )
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.batch:
        batch_predict(args.audio_path, args.whisper, args.threshold, device)
    else:
        predict(args.audio_path, args.whisper, args.threshold, save_log=not args.no_log, device=device)