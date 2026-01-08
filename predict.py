# ============================================================
# predict.py — MAMBA INFERENCE SCRIPT
# Loads the best_pytorch_model.bin and runs prediction on the test set.
# ============================================================

import os
import json
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

# --- ADDED: Whisper import for single audio file mode ---
import whisper 


# --- Configuration and Directories ---
DATA_DIR = "processed_data"
MODEL_DIR = "trained_mamba_pretrained_model"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_pytorch_model.bin")
TOKENIZER_ID = "EleutherAI/gpt-neox-20b"


# ============================================================
# DATASET (Copied from train.py)
# ============================================================

class MCIDataset(Dataset):
    """Dataset class matching the structure and encoding in train.py."""
    def __init__(self, df, tokenizer, max_len=512): # Use 512 max_len as per final stable run
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.df.loc[idx, "transcription"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        # Prediction needs input_ids and labels (for evaluation)
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "labels": torch.tensor(self.df.loc[idx, "label_id"], dtype=torch.long),
        }


# ============================================================
# MODEL COMPONENTS (Copied from train.py)
# ============================================================

class AttentionPooling(nn.Module):
    """Attention pooling layer used in the classifier head."""
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):
        w = torch.softmax(self.attn(x), dim=1)
        return (x * w).sum(dim=1)


class MambaClassifier(nn.Module):
    """Mamba backbone plus custom classification head."""
    def __init__(self, base_model, num_labels):
        super().__init__()
        self.base = base_model.to(torch.float32) 
        hidden = base_model.config.d_model
        
        self.pool = AttentionPooling(hidden) 
        self.fc = nn.Linear(hidden, hidden // 4)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden // 4, num_labels)
        
        # Define criterion placeholder, used only for loss calculation in evaluate_inference
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, labels=None):
        out = self.base.backbone(input_ids)
        
        pooled = self.pool(out)
        x = self.dropout(torch.nn.functional.gelu(self.fc(pooled)))
        logits = self.classifier(x)
        
        loss = self.criterion(logits, labels) if labels is not None else None
        return loss, logits 


# ============================================================
# PREDICTION AND EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_inference(model, loader, device):
    """Runs inference on the test data and computes final metrics."""
    model.eval()
    preds, labels, losses = [], [], []

    for batch in tqdm(loader, desc="Running Inference"):
        ids = batch["input_ids"].to(device)
        y = batch["labels"].to(device)

        with autocast(enabled=(device == "cuda")):
            # Loss is calculated internally by the model to report test loss
            loss, logits = model(ids, y) 

        losses.append(loss.item())
        preds.extend(torch.argmax(logits, 1).cpu().numpy())
        labels.extend(y.cpu().numpy())

    return (
        np.mean(losses),
        accuracy_score(labels, preds),
        f1_score(labels, preds, average="weighted"),
        labels,
        preds,
    )


def predict_single_audio(model, tokenizer, audio_path, whisper_model_size, device):
    """Runs prediction on a single audio file, including Whisper transcription."""
    
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        return

    print(f"Loading Whisper model ({whisper_model_size}) and transcribing...")
    
    # Load Whisper to CPU first for safety, then move to device
    whisper_model = whisper.load_model(whisper_model_size, device="cpu")
    whisper_model.to(device)
    
    # Use the same robust transcription settings as the training data cache
    result = whisper_model.transcribe(
        audio_path, 
        fp16=(device == 'cuda'),
        temperature=0.0, 
        condition_on_previous_text=False, 
        best_of=5,
        beam_size=5,
        patience=1.0,
        no_speech_threshold=0.6,
        logprob_threshold=-1.0
    )
    transcription = result['text']
    
    print(f"\n[Transcription: {transcription}]")
    
    # Tokenize the single transcription
    inputs = tokenizer(
        transcription, 
        return_tensors="pt", 
        truncation=True, 
        padding='max_length', 
        max_length=512
    ).to(device)
    
    model.eval()
    with torch.no_grad():
        with autocast(enabled=(device == "cuda")):
            _, logits = model(inputs['input_ids'])

        probabilities = torch.softmax(logits, dim=-1)
        dementia_confidence = probabilities[0][1].item()
        predicted_class_id = torch.argmax(probabilities).item()
        
    prediction = "Dementia" if predicted_class_id == 1 else "Control"
    
    print("\n============================================================")
    print("               MAMBA SINGLE PREDICTION RESULT               ")
    print("============================================================")
    print(f"Input File: {os.path.basename(audio_path)}")
    print(f"Predicted Label: {prediction}")
    print(f"Confidence (Control): {probabilities[0][0].item():.4f}")
    print(f"Confidence (Dementia): {dementia_confidence:.4f}")
    print("============================================================")
    return


# ============================================================
# MAIN PREDICTION PIPELINE
# ============================================================

def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if not os.path.exists(BEST_MODEL_PATH):
        print(f"Error: Model weights not found at {BEST_MODEL_PATH}. Run train.py first.")
        return

    # -------------------- Model Loading (Required for both modes) --------------------
    print("Instantiating Mamba architecture...")
    
    base_model = MambaLMHeadModel.from_pretrained(args.mamba_model)
    model = MambaClassifier(base_model, num_labels=2).to(device)
    
    # Load the best trained weights
    print(f"Loading weights from {BEST_MODEL_PATH}...")
    
    # --- FIX: Use strict=False to ignore criterion.weight in the checkpoint ---
    model.load_state_dict(
        torch.load(BEST_MODEL_PATH, map_location=device),
        strict=False 
    )
    
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # -------------------- MODE SWITCH: Single File vs. Batch Evaluation --------------------
    if args.single_audio_path:
        # --- MODE 1: Single Audio File Prediction ---
        predict_single_audio(model, tokenizer, args.single_audio_path, args.whisper_model, device)
        return
    
    # -------------------- MODE 2: Full Test Set Evaluation (Original Logic) --------------------

    # Load and Split Data 
    df = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))

    with open(os.path.join(DATA_DIR, "transcripts_cache.json")) as f:
        cache = json.load(f)

    df["transcription"] = df["audio_path"].map(cache["transcripts"])
    df = df.dropna(subset=["transcription"])
    df["label_id"] = df["label"].map({"Control": 0, "Dementia": 1})

    # Recreate the exact split used in training to ensure test set integrity
    train_df, test_df = train_test_split(
        df, test_size=0.15, stratify=df["label_id"], random_state=args.seed
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.10, stratify=train_df["label_id"], random_state=args.seed
    )

    # Tokenizer and DataLoader 
    test_ds = MCIDataset(test_df, tokenizer, max_len=args.max_len)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size) 

    # Run Inference 
    test_loss, test_acc, test_f1, test_labels, test_preds = evaluate_inference(
        model, test_loader, device
    )

    # Report Results 
    print("\n============================================================")
    print("               MAMBA FINAL PREDICTION RESULTS               ")
    print("============================================================")
    print(f"Model: Mamba (Textual Baseline)")
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.3f} | Test F1 Score: {test_f1:.3f}")
    
    print("\nClassification Report:")
    print(
        classification_report(
            test_labels, test_preds, target_names=['Control', 'Dementia']
        )
    )
    
    cm = confusion_matrix(test_labels, test_preds)
    print("Confusion Matrix:")
    print(f"                  Predicted")
    print(f"              Control  Dementia")
    print(f"Actual Control    {cm[0][0]:3d}      {cm[0][1]:3d}")
    print(f"       Dementia   {cm[1][0]:3d}      {cm[1][1]:3d}")
    print("============================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16) 
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mamba_model", type=str, default="state-spaces/mamba-130m")
    parser.add_argument("--max_len", type=int, default=512, help="Maximum sequence length.")
    parser.add_argument("--whisper_model", type=str, default="tiny.en", help="Whisper model size for single audio mode.")
    parser.add_argument("--single_audio_path", type=str, default=None, help="Path to a single audio file for prediction.")
    args = parser.parse_args()

    main(args)