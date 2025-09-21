# -*- coding: utf-8 -*-
"""preprocess.py (Final working version for DementiaBank .cha files)
"""
from pathlib import Path
import re
import sys
import traceback
import pandas as pd
from pydub import AudioSegment
from tqdm import tqdm

# --- Paths ---
CWD = Path.cwd()
DATA_DIR = CWD / "data" / "Pitt"
OUTPUT_DIR = CWD / "processed_data"
AUDIO_OUTPUT_DIR = OUTPUT_DIR / "audio"
METADATA_PATH = OUTPUT_DIR / "metadata.csv"
SKIP_LOG_PATH = OUTPUT_DIR / "skip_log.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Regex ---
# Matches any line starting with *PAR and optional extra tier info
PAR_TIER_PATTERN = re.compile(r'^\*PAR[^\:]*\s*:\s*(.*)', flags=re.IGNORECASE)
# Matches timestamps in the form digits_digits
TIMESTAMP_PATTERN = re.compile(r'(\d+)_(\d+)')

def parse_cha_file(cha_path: Path):
    """
    Parse a .cha file and return a list of (participant_text, start_ms, end_ms) segments.
    Extracts the last timestamp on the line and text before it.
    """
    segments = []
    try:
        with open(cha_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
            for line in f:
                line = line.strip()
                par_match = PAR_TIER_PATTERN.match(line)
                if par_match:
                    text_with_ts = par_match.group(1).strip()
                    ts_matches = TIMESTAMP_PATTERN.findall(text_with_ts)
                    if ts_matches:
                        try:
                            start_ms, end_ms = map(int, ts_matches[-1])  # use the last timestamp
                            # Remove the last timestamp from text
                            text_part = text_with_ts[:text_with_ts.rfind(f"{start_ms}_{end_ms}")].strip()
                            if text_part:
                                segments.append((text_part, start_ms, end_ms))
                        except (ValueError, IndexError):
                            continue
    except Exception as exc:
        print(f"Error reading/parsing {cha_path}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return []
    return segments

def find_audio_file_for(cha_path: Path):
    """Try to find corresponding audio file (.wav or .mp3)."""
    for ext in ('.wav', '.mp3', '.WAV', '.MP3'):
        candidate = cha_path.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None

def process_group(group_name: str, cha_folder: Path, metadata_accum: list, skip_accum: list):
    """Process all .cha files in a group folder."""
    if not cha_folder.is_dir():
        print(f"Warning: group folder not found: {cha_folder}")
        return

    cha_files = sorted(list(cha_folder.glob('*.cha')))
    if not cha_files:
        print(f"Warning: no .cha files found in {cha_folder}")
        return

    for cha_path in tqdm(cha_files, desc=f"Processing {group_name}", unit="file"):
        stem = cha_path.stem
        try:
            audio_file = find_audio_file_for(cha_path)
            if audio_file is None:
                skip_accum.append((group_name, stem, 'AUDIO_MISSING', 'No audio file found (.wav/.mp3)'))
                continue

            segments = parse_cha_file(cha_path)
            if not segments:
                skip_accum.append((group_name, stem, 'NO_PAR_SEGMENTS', 'No participant segments with timestamps'))
                continue

            try:
                full_audio = AudioSegment.from_file(audio_file)
            except Exception as e:
                skip_accum.append((group_name, stem, 'AUDIO_READ_ERROR', f'Could not read audio: {e}'))
                continue

            participant_audio = AudioSegment.empty()
            for text, start_ms, end_ms in segments:
                if start_ms < end_ms:
                    participant_audio += full_audio[start_ms:end_ms]

            if len(participant_audio) > 500:  # Only save if audio is longer than 0.5s
                out_path = AUDIO_OUTPUT_DIR / f"{stem}_participant.wav"
                participant_audio.export(out_path, format='wav')
                metadata_accum.append({
                    'participant_id': stem,
                    'label': group_name,
                    'audio_path': str(out_path),
                })
            else:
                skip_accum.append((group_name, stem, 'NO_AUDIO_EXTRACTED', 'No valid audio after extracting segments'))

        except Exception as ex:
            skip_accum.append((group_name, stem, 'UNEXPECTED_ERROR', f"{ex}"))

def save_csvs(metadata, skips):
    """Save metadata and skip logs to CSV."""
    if metadata:
        pd.DataFrame(metadata).to_csv(METADATA_PATH, index=False)
    elif METADATA_PATH.exists():
        METADATA_PATH.unlink()

    if skips:
        pd.DataFrame(skips, columns=['group', 'file_stem', 'reason_type', 'reason_detail']).to_csv(SKIP_LOG_PATH, index=False)
    elif SKIP_LOG_PATH.exists():
        SKIP_LOG_PATH.unlink()

def print_summary(metadata, skips):
    """Print preprocessing summary."""
    processed_count = len(metadata)
    skipped_count = len(skips)
    print("\n" + "="*32)
    print("PREPROCESSING SUMMARY")
    print("="*32)
    print(f"Processed (audio produced): {processed_count}")
    print(f"Skipped: {skipped_count}")

    if skipped_count:
        reason_counts = pd.Series([r[2] for r in skips]).value_counts()
        print("\nSkip reasons:")
        for reason, cnt in reason_counts.items():
            print(f"  - {reason}: {cnt}")

    if processed_count:
        label_counts = pd.Series([m['label'] for m in metadata]).value_counts()
        print("\nLabel distribution:")
        print(label_counts.to_string())
        print(f"\nMetadata saved to: {METADATA_PATH}")
        print(f"Processed audio saved in: {AUDIO_OUTPUT_DIR}")
    else:
        print("\nNo processed audio was created.")

    if skipped_count:
        print(f"\nSkip log saved to: {SKIP_LOG_PATH}")
    print("="*32 + "\n")

def main():
    metadata_accum = []
    skip_accum = []
    groups = ['Control', 'Dementia']

    for g in groups:
        group_folder = DATA_DIR / g
        process_group(g, group_folder, metadata_accum, skip_accum)

    save_csvs(metadata_accum, skip_accum)
    print_summary(metadata_accum, skip_accum)

if __name__ == "__main__":
    main()
