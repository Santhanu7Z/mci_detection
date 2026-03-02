import os
import json
import pandas as pd
import whisper
import torch
from tqdm import tqdm

DATA_DIR = "processed_data"
CACHE_PATH = os.path.join(DATA_DIR, "transcripts_cache.json")

# Load metadata
df = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))

WHISPER_MODEL = "large"   # Change to "medium" if memory issues

print("=" * 60)
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available — aborting.")

print("CUDA device name:", torch.cuda.get_device_name(0))
print("=" * 60)

# -------- LOAD MODEL PROPERLY --------
print("Loading Whisper model:", WHISPER_MODEL)

model = whisper.load_model(WHISPER_MODEL)   # Load normally
model = model.to("cuda")                    # FORCE GPU transfer

print("Model device:", next(model.parameters()).device)

print("=" * 60)
print(f"Transcribing {len(df)} audio files...")
print("=" * 60)

transcripts = {}

# Resume from existing cache if exists
if os.path.exists(CACHE_PATH):
    print("Loading existing cache...")
    with open(CACHE_PATH, "r") as f:
        existing_cache = json.load(f)
        transcripts = existing_cache.get("transcripts", {})
    print(f"Resuming from {len(transcripts)} existing transcripts.")

# -------- TRANSCRIPTION LOOP --------
for idx, row in tqdm(df.iterrows(), total=len(df)):

    audio_path = row["audio_path"]

    if audio_path in transcripts:
        continue  # skip already done

    if not os.path.exists(audio_path):
        print(f"Missing file: {audio_path}")
        transcripts[audio_path] = ""
        continue

    try:
        result = model.transcribe(
            audio_path,
            fp16=True,                     # GPU half precision
            temperature=0.0,
            beam_size=5,
            condition_on_previous_text=False,
            language="en",
            task="transcribe"
        )

        transcripts[audio_path] = result["text"].strip()

    except Exception as e:
        print(f"\nError transcribing {audio_path}: {e}")
        transcripts[audio_path] = ""

    # ---- SAVE CHECKPOINT EVERY 10 FILES ----
    if idx % 10 == 0:
        with open(CACHE_PATH, "w") as f:
            json.dump({
                "model": WHISPER_MODEL,
                "transcripts": transcripts
            }, f, indent=2)

# -------- FINAL SAVE --------
with open(CACHE_PATH, "w") as f:
    json.dump({
        "model": WHISPER_MODEL,
        "transcripts": transcripts
    }, f, indent=2)

print("=" * 60)
print("Transcription complete.")
print("Total completed:", len([t for t in transcripts.values() if t]))
print("Total empty:", len([t for t in transcripts.values() if not t]))
print("=" * 60)

