import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from scipy.signal import find_peaks
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================
AUDIO_DIR = "processed_data/audio"
OUTPUT_CSV = "processed_data/acoustic_features.csv"

def extract_voice_quality(y, sr):
    """
    Extracts Jitter, Shimmer, and HNR (Harmonics-to-Noise Ratio).
    These are key biomarkers for neurological vocal stability.
    """
    # Get fundamental frequency (F0) using Yin algorithm
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    
    # Filter out NaNs for calculation
    f0_clean = f0[~np.isnan(f0)]
    
    if len(f0_clean) < 2:
        return 0, 0, 0, 0 # Return zeros if audio is too short or silent

    # 1. Fundamental Frequency Stats
    f0_mean = np.mean(f0_clean)
    f0_std = np.std(f0_clean)

    # 2. Jitter (Frequency instability)
    # Simple Local Jitter: Average absolute difference between consecutive periods
    periods = 1.0 / f0_clean
    jitter = np.mean(np.abs(np.diff(periods))) / np.mean(periods)

    # 3. Shimmer (Amplitude instability)
    # Extract peaks (amplitudes)
    peaks, _ = find_peaks(y, distance=sr//f0_mean if f0_mean > 0 else 100)
    amplitudes = np.abs(y[peaks])
    if len(amplitudes) > 1:
        shimmer = np.mean(np.abs(np.diff(amplitudes))) / np.mean(amplitudes)
    else:
        shimmer = 0

    # 4. HNR (Harmonics-to-Noise Ratio) - Simple Autocorrelation method
    autocorr = librosa.autocorrelate(y)
    max_autocorr = np.max(autocorr[sr//400:sr//50]) # Look in human voice range
    hnr = 10 * np.log10(max_autocorr / (np.abs(np.max(autocorr) - max_autocorr) + 1e-6))

    return f0_mean, f0_std, jitter, shimmer, hnr

def extract_features(file_path):
    """
    Extracts a total of 30+ core features (Temporal, Prosodic, and Spectral).
    """
    try:
        y, sr = librosa.load(file_path, sr=16000)
        if len(y) == 0: return None
        
        # --- 1. Temporal Features ---
        duration = librosa.get_duration(y=y, sr=sr)
        
        # --- 2. Voice Quality & Prosody ---
        f0_mean, f0_std, jitter, shimmer, hnr = extract_voice_quality(y, sr)
        
        # --- 3. Spectral Features ---
        # MFCCs (13 coefficients - standard for clinical NLP)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        
        # Spectral Centroid, Bandwidth, and Rolloff
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # --- 4. Consolidate into Dictionary ---
        feature_dict = {
            "file_id": os.path.basename(file_path).replace(".wav", ""),
            "duration": duration,
            "f0_mean": f0_mean,
            "f0_std": f0_std,
            "jitter": jitter,
            "shimmer": shimmer,
            "hnr": hnr,
            "spectral_centroid": centroid,
            "spectral_bandwidth": bandwidth,
            "spectral_rolloff": rolloff,
            "zcr": zcr
        }
        
        # Add MFCCs individually
        for i, m in enumerate(mfccs_mean):
            feature_dict[f"mfcc_{i+1}"] = m
            
        return feature_dict

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    if not os.path.exists(AUDIO_DIR):
        print(f"Error: Audio directory {AUDIO_DIR} not found.")
    else:
        files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")]
        print(f"Found {len(files)} audio files. Starting extraction...")
        
        all_features = []
        for f in tqdm(files):
            full_path = os.path.join(AUDIO_DIR, f)
            feats = extract_features(full_path)
            if feats:
                all_features.append(feats)
        
        # Save to CSV
        df = pd.DataFrame(all_features)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSuccess! Features saved to {OUTPUT_CSV}")
        print(df.head())