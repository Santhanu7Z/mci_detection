import os
import math
import json
import random
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import whisper
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

# ------------------------------------------------------------
# MAMBA WEIGHT-COMPATIBLE, SERVER-SAFE IMPLEMENTATION
# ------------------------------------------------------------

class InferredMambaConfig:
    def __init__(self, vocab_size, hidden_size, num_hidden_layers, d_inner, dt_rank, state_size, conv_kernel, num_labels=2):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.d_inner = d_inner
        self.dt_rank = dt_rank
        self.state_size = state_size
        self.conv_kernel = conv_kernel
        self.num_labels = num_labels

class MambaBlock(nn.Module):
    def __init__(self, config: InferredMambaConfig):
        super().__init__()
        self.config = config
        self.in_proj = nn.Linear(config.hidden_size, config.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=config.d_inner,
            out_channels=config.d_inner,
            kernel_size=config.conv_kernel,
            bias=True,
            groups=config.d_inner,
            padding=config.conv_kernel - 1,
        )
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + config.state_size * 2, bias=False)
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.zeros(config.d_inner, config.state_size))
        self.D = nn.Parameter(torch.ones(config.d_inner))
        self.out_proj = nn.Linear(config.d_inner, config.hidden_size, bias=False)

    def forward(self, x):
        b, l, d = x.shape
        xr = self.in_proj(x)
        x, res = xr.split([self.config.d_inner, self.config.d_inner], dim=-1)
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :l]
        x = x.transpose(1, 2)
        x = torch.nn.functional.silu(x)
        x_dbl = self.x_proj(x)
        dt, B, C = x_dbl.split([self.config.dt_rank, self.config.state_size, self.config.state_size], dim=-1)
        delta = torch.nn.functional.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log.float())
        y = self._selective_scan(x, delta, A, B, C)
        y = y * torch.nn.functional.silu(res)
        return self.out_proj(y)

    def _selective_scan(self, u, delta, A, B, C):
        b, l, d_in = u.shape
        n = A.shape[1]
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB_u = (delta * u).unsqueeze(-1) * B.unsqueeze(2)
        h = torch.zeros(b, d_in, n, device=u.device, dtype=u.dtype)
        ys = []
        for i in range(l):
            h = deltaA[:, i] * h + deltaB_u[:, i]
            y = (h * C[:, i].unsqueeze(1)).sum(dim=-1)
            ys.append(y)
        return torch.stack(ys, dim=1)

class MambaResidualBlock(nn.Module):
    def __init__(self, config: InferredMambaConfig):
        super().__init__()
        self.mixer = MambaBlock(config)
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(self, x):
        return x + self.mixer(self.norm(x))

class MambaBackbone(nn.Module):
    def __init__(self, config: InferredMambaConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([MambaResidualBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm_f = nn.LayerNorm(config.hidden_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.norm_f(x)

class MambaForClassification(nn.Module):
    def __init__(self, config: InferredMambaConfig):
        super().__init__()
        self.backbone = MambaBackbone(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # Using CrossEntropyLoss here to allow validation loss calculation
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, labels=None):
        x = self.backbone(input_ids)
        # Get the feature vector for the last token (classification head input)
        # x[:, -1, :] effectively acts as a sequence-level representation (like the CLS token in BERT)
        logits = self.classifier(x[:, -1, :])
        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)
        return type("Output", (), {"loss": loss, "logits": logits})

# ------------------------------------------------------------
# Utilities - CONFIG INFERRAL, DATA LOADING, AND EARLY STOPPER
# ------------------------------------------------------------

class EarlyStopper:
    def __init__(self, patience=3, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if self.mode == 'min':
            self.best_score = float('inf')
        elif self.mode == 'max':
            self.best_score = float('-inf')

    def __call__(self, current_score, model_state, model_path):
        """
        Check if training should stop.
        Saves the best model based on the metric.
        """
        if self.best_score is None:
            self.best_score = current_score
            self.save_checkpoint(model_state, model_path)
        elif (self.mode == 'min' and current_score > self.best_score + self.min_delta) or \
             (self.mode == 'max' and current_score < self.best_score - self.min_delta):
            self.counter += 1
            print(f'EarlyStopper counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (self.mode == 'min' and current_score < self.best_score) or \
               (self.mode == 'max' and current_score > self.best_score):
                self.best_score = current_score
                self.save_checkpoint(model_state, model_path)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model_state, model_path):
        """Saves model when the metric improves."""
        print(f"Validation score improved ({self.best_score:.4f}). Saving model to {model_path}...")
        torch.save(model_state, model_path)


def infer_config_from_state_dict(sd: dict) -> InferredMambaConfig:
    emb_w = sd.get('backbone.embedding.weight', None)
    if emb_w is None:
        raise ValueError("Checkpoint missing 'backbone.embedding.weight'.")
    vocab_size, hidden_size = emb_w.shape

    layer_idx = 0
    while f'backbone.layers.{layer_idx}.mixer.in_proj.weight' in sd:
        layer_idx += 1
    num_hidden_layers = layer_idx

    in_proj = sd['backbone.layers.0.mixer.in_proj.weight']
    d_inner = in_proj.shape[0] // 2

    dt_proj_w = sd['backbone.layers.0.mixer.dt_proj.weight']
    dt_rank = dt_proj_w.shape[1]

    x_proj_w = sd['backbone.layers.0.mixer.x_proj.weight']
    total_out = x_proj_w.shape[0]
    state_size = (total_out - dt_rank) // 2

    conv_w = sd['backbone.layers.0.mixer.conv1d.weight']
    conv_kernel = conv_w.shape[-1]

    return InferredMambaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        d_inner=d_inner,
        dt_rank=dt_rank,
        state_size=state_size,
        conv_kernel=conv_kernel,
        num_labels=2,
    )


def load_and_transcribe_data(metadata_path, whisper_model_size, device):
    print("Loading metadata...")
    df = pd.read_csv(metadata_path)
    
    print(f"Loading Whisper model: {whisper_model_size}...")
    
    # Always load to CPU first for safety on shared memory systems
    whisper_model = whisper.load_model(whisper_model_size, device="cpu")
    
    # Move Whisper to the actual compute device after loading
    whisper_model.to(device)
    
    # --- ADDED VERBATIM TRANSCRIPTION PROMPT FOR LINGUISTIC FEATURES ---
    # This prompt forces Whisper to keep fillers (um, uh) and disfluencies, 
    # which are critical features of cognitive impairment.
    VERBATIM_PROMPT = "The following is a verbatim transcript containing stutters, fillers like um and uh, false starts, and repetitions."
        
    transcriptions = []
    print("Transcribing audio files with anti-hallucination and **VERBATIM** settings...")
    
    for audio_path in tqdm(df['audio_path'], desc="Transcribing"):
        if not os.path.exists(audio_path):
            transcriptions.append("")
            continue
        
        # Apply robust constraints, using fp16 only if the device is CUDA
        result = whisper_model.transcribe(
            audio_path, 
            fp16=(device == 'cuda'),
            temperature=0.0, 
            condition_on_previous_text=False,
            initial_prompt=VERBATIM_PROMPT, # <- NEW: Forces Whisper to keep disfluencies
            best_of=5,
            beam_size=5,
            patience=1.0,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0
        )
        
        transcriptions.append(result['text'])
        
    df['transcription'] = transcriptions
    df = df[df['transcription'].str.strip() != ''].reset_index(drop=True)

    # Note: We skip the caching fix for now, but this is where it would be implemented.
    
    return df

class MCIDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.encodings['input_ids'])

# ------------------------------------------------------------
# Evaluation Function
# ------------------------------------------------------------

def evaluate(model, data_loader, device):
    """Calculates loss, accuracy, and F1 score on a given dataset."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # The model is now guaranteed to return a loss if labels are provided
            out = model(input_ids=input_ids, labels=labels)
            
            total_loss += out.loss.item()
            preds = torch.argmax(out.logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / max(1, len(data_loader))
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted') # Corrected: f1 calculation bug fix
    
    return avg_loss, acc, f1

# ------------------------------------------------------------
# Main training
# ------------------------------------------------------------

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    PROCESSED_DATA_PATH = "processed_data/metadata.csv"
    MODEL_OUTPUT_DIR = "trained_model_mamba_pretrained"
    TOKENIZER_ID = "EleutherAI/gpt-neox-20b"
    BEST_MODEL_PATH = os.path.join(MODEL_OUTPUT_DIR, "best_pytorch_model.bin")

    # --- Initial Device Check ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")

    # Data loading uses the determined device. Transcription now uses VERBATIM prompt.
    df = load_and_transcribe_data(PROCESSED_DATA_PATH, args.whisper_model, device)
    
    df['label_id'] = df['label'].apply(lambda x: 1 if x == 'Dementia' else 0)
    
    # Split: Train (80%) -> Remaining (20%)
    train_df, remaining_df = train_test_split(df, test_size=0.2, random_state=args.seed, stratify=df['label_id'])
    
    # Split: Remaining (10% of total) -> Validation (50% of remaining), Test (50% of remaining)
    val_df, test_df = train_test_split(remaining_df, test_size=0.5, random_state=args.seed, stratify=remaining_df['label_id'])

    print(f"Data Split - Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
    
    print(f"Loading tokenizer: {TOKENIZER_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Tokenizing datasets...")
    max_length = 2048
    train_encodings = tokenizer(train_df['transcription'].tolist(), truncation=True, padding=True, max_length=max_length)
    val_encodings = tokenizer(val_df['transcription'].tolist(), truncation=True, padding=True, max_length=max_length)
    test_encodings = tokenizer(test_df['transcription'].tolist(), truncation=True, padding=True, max_length=max_length)

    train_dataset = MCIDataset(train_encodings, train_df['label_id'].tolist())
    val_dataset = MCIDataset(val_encodings, val_df['label_id'].tolist())
    test_dataset = MCIDataset(test_encodings, test_df['label_id'].tolist())

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    print(f"Downloading pre-trained weights from {args.mamba_model}...")
    model_weights_path = hf_hub_download(repo_id=args.mamba_model, filename="pytorch_model.bin")
    pretrained_weights = torch.load(model_weights_path, map_location="cpu")

    config = infer_config_from_state_dict(pretrained_weights)
    print(f"Inferred config -> hidden_size: {config.hidden_size}, layers: {config.num_hidden_layers}, d_inner: {config.d_inner}, dt_rank: {config.dt_rank}, state_size: {config.state_size}, conv_kernel: {config.conv_kernel}")

    print("Initializing Mamba-like architecture (server-safe)...")
    model = MambaForClassification(config)

    missing, unexpected = model.load_state_dict(pretrained_weights, strict=False)
    print(f"Weights loaded with {len(missing)} missing and {len(unexpected)} unexpected keys.")

    # Move Mamba model to device
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Initialize Early Stopper
    # patience=3, mode='min' for validation loss
    early_stopper = EarlyStopper(patience=3, min_delta=0, mode='min') 

    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'val_f1': []}

    print("Starting training...")
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True) # Ensure output dir exists before training

    for epoch in range(args.epochs):
        # --- Training Step ---
        model.train()
        total_train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} (Train)")
        
        for batch in pbar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        avg_train_loss = total_train_loss / max(1, len(train_loader))
        history['train_loss'].append(avg_train_loss)
        print(f"\nEpoch {epoch+1} Average Training Loss: {avg_train_loss:.4f}")

        # --- Validation Step ---
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, device)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        print(f"Epoch {epoch+1} - Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, F1 Score: {val_f1:.4f}")

        # --- Early Stopping Check ---
        if early_stopper(val_loss, model.state_dict(), BEST_MODEL_PATH):
            print(f"Early stopping triggered at epoch {epoch+1} (patience={early_stopper.patience})")
            break

    # --- Final Evaluation and Save ---
    
    # Load the best model weights found during training
    if os.path.exists(BEST_MODEL_PATH):
        print(f"Loading best model from {BEST_MODEL_PATH} for final evaluation.")
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
        
        # Run final evaluation on the separate test set
        test_loss, test_acc, test_f1 = evaluate(model, test_loader, device)
        print("----------------------------------------------------------------------")
        print(f"FINAL TEST RESULTS (Best Model):")
        print(f"Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, F1 Score: {test_f1:.4f}")
        print("----------------------------------------------------------------------")

        # Save tokenizer and history for the best model
        tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
        
    else:
        # If early stopping was never triggered (e.g., training finished all epochs), 
        # save the last epoch's model and evaluate it on the test set.
        print(f"Training complete. Saving final model weights to {MODEL_OUTPUT_DIR}")
        torch.save(model.state_dict(), os.path.join(MODEL_OUTPUT_DIR, "pytorch_model.bin"))
        tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
        test_loss, test_acc, test_f1 = evaluate(model, test_loader, device)
        print("----------------------------------------------------------------------")
        print(f"FINAL TEST RESULTS (Last Epoch Model):")
        print(f"Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, F1 Score: {test_f1:.4f}")
        print("----------------------------------------------------------------------")

    history_path = os.path.join(MODEL_OUTPUT_DIR, 'training_history.json')
    with open(history_path, 'w') as f:
        # Convert NumPy types in history to standard Python types for JSON serialization
        json.dump({k: [float(x) for x in v] for k, v in history.items()}, f, indent=4)
    print(f"Training history saved to {history_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a GLIBC-safe Mamba-like classifier for MCI detection.")
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=4, help='Training and evaluation batch size.') 
    parser.add_argument('--whisper_model', type=str, default='medium.en', help='Whisper model size.')
    parser.add_argument('--mamba_model', type=str, default='state-spaces/mamba-130m', help='HF repo id for weights.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    args = parser.parse_args()
    main(args)