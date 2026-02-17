import os
import numpy as np
import pandas as pd
import librosa
from scipy.signal import find_peaks
from tqdm import tqdm

AUDIO_DIR = "processed_data/audio"
OUTPUT_CSV = "processed_data/acoustic_features.csv"


# ============================================================
# Voice Quality (Jitter, Shimmer, HNR)
# ============================================================

def extract_voice_quality(y, sr):
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7')
    )

    f0_clean = f0[~np.isnan(f0)]

    if len(f0_clean) < 2:
        return [0]*8

    f0_mean = np.mean(f0_clean)
    f0_std = np.std(f0_clean)
    f0_min = np.min(f0_clean)
    f0_max = np.max(f0_clean)
    f0_range = f0_max - f0_min
    f0_median = np.median(f0_clean)

    periods = 1.0 / f0_clean
    jitter = np.mean(np.abs(np.diff(periods))) / np.mean(periods)

    peaks, _ = find_peaks(y, distance=int(sr / f0_mean) if f0_mean > 0 else 100)
    amplitudes = np.abs(y[peaks])
    shimmer = (
        np.mean(np.abs(np.diff(amplitudes))) / np.mean(amplitudes)
        if len(amplitudes) > 1 else 0
    )

    autocorr = librosa.autocorrelate(y)
    max_autocorr = np.max(autocorr[sr//400:sr//50])
    hnr = 10 * np.log10(max_autocorr / (np.abs(np.max(autocorr) - max_autocorr) + 1e-6))

    return [
        f0_mean, f0_std, f0_min, f0_max,
        f0_range, f0_median,
        jitter, shimmer, hnr
    ]


# ============================================================
# Main Feature Extraction
# ============================================================

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)

        if len(y) < sr:
            return None

        duration = librosa.get_duration(y=y, sr=sr)

        # --- Voice quality ---
        (
            f0_mean, f0_std, f0_min, f0_max,
            f0_range, f0_median,
            jitter, shimmer, hnr
        ) = extract_voice_quality(y, sr)

        # --- Energy ---
        rms = librosa.feature.rms(y=y)[0]
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        rms_cv = rms_std / (rms_mean + 1e-6)

        # --- Spectral features ---
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
        flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        zcr_mean = np.mean(librosa.feature.zero_crossing_rate(y))
        zcr_std = np.std(librosa.feature.zero_crossing_rate(y))

        # --- MFCC ---
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        delta_mean = np.mean(delta, axis=1)
        delta2_mean = np.mean(delta2, axis=1)

        features = {
            "file_id": os.path.basename(file_path).replace(".wav", ""),
            "duration": duration,
            "f0_mean": f0_mean,
            "f0_std": f0_std,
            "f0_min": f0_min,
            "f0_max": f0_max,
            "f0_range": f0_range,
            "f0_median": f0_median,
            "jitter": jitter,
            "shimmer": shimmer,
            "hnr": hnr,
            "rms_mean": rms_mean,
            "rms_std": rms_std,
            "rms_cv": rms_cv,
            "spectral_centroid": centroid,
            "spectral_bandwidth": bandwidth,
            "spectral_rolloff": rolloff,
            "spectral_contrast": contrast,
            "spectral_flatness": flatness,
            "zcr_mean": zcr_mean,
            "zcr_std": zcr_std
        }

        for i in range(13):
            features[f"mfcc_mean_{i+1}"] = mfcc_mean[i]
            features[f"mfcc_std_{i+1}"] = mfcc_std[i]
            features[f"delta_mfcc_{i+1}"] = delta_mean[i]
            features[f"delta2_mfcc_{i+1}"] = delta2_mean[i]

        return features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")]
    print(f"Found {len(files)} files.")

    all_features = []

    for f in tqdm(files):
        path = os.path.join(AUDIO_DIR, f)
        feats = extract_features(path)
        if feats:
            all_features.append(feats)

    df = pd.DataFrame(all_features)
    df.to_csv(OUTPUT_CSV, index=False)

    print("Saved:", OUTPUT_CSV)
    print(df.head())

