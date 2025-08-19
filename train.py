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

    def forward(self, input_ids, labels=None):
        x = self.backbone(input_ids)
        logits = self.classifier(x[:, -1, :])
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return type("Output", (), {"loss": loss, "logits": logits})

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def infer_config_from_state_dict(sd: dict) -> InferredMambaConfig:
    emb_w = sd.get('backbone.embedding.weight', None)
    if emb_w is None:
        raise ValueError("Checkpoint missing 'backbone.embedding.weight'.")
    vocab_size, hidden_size = emb_w.shape

    in_proj = sd['backbone.layers.0.mixer.in_proj.weight']
    d_inner = in_proj.shape[0] // 2

    dt_proj_w = sd['backbone.layers.0.mixer.dt_proj.weight']
    dt_rank = dt_proj_w.shape[1]

    x_proj_w = sd['backbone.layers.0.mixer.x_proj.weight']
    total_out = x_proj_w.shape[0]
    dt_rank_inferred = dt_proj_w.shape[1]
    state_size = (total_out - dt_rank_inferred) // 2

    conv_w = sd['backbone.layers.0.mixer.conv1d.weight']
    conv_kernel = conv_w.shape[-1]

    layer_idx = 0
    while f'backbone.layers.{layer_idx}.mixer.in_proj.weight' in sd:
        layer_idx += 1
    num_hidden_layers = layer_idx

    return InferredMambaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        d_inner=d_inner,
        dt_rank=dt_rank_inferred,
        state_size=state_size,
        conv_kernel=conv_kernel,
        num_labels=2,
    )

# ------------------------------------------------------------
# Data loading & tokenization
# ------------------------------------------------------------

def load_and_transcribe_data(metadata_path, whisper_model_size):
    print("Loading metadata...")
    df = pd.read_csv(metadata_path)
    print(f"Loading Whisper model: {whisper_model_size}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    whisper_model = whisper.load_model(whisper_model_size, device=device)
    transcriptions = []
    print("Transcribing audio files...")
    for audio_path in tqdm(df['audio_path'], desc="Transcribing"):
        if not os.path.exists(audio_path):
            transcriptions.append("")
            continue
        result = whisper_model.transcribe(audio_path, fp16=torch.cuda.is_available())
        transcriptions.append(result['text'])
    df['transcription'] = transcriptions
    df = df[df['transcription'].str.strip() != ''].reset_index(drop=True)
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
# Main training
# ------------------------------------------------------------

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    PROCESSED_DATA_PATH = "processed_data/metadata.csv"
    MODEL_OUTPUT_DIR = "trained_model_mamba_pretrained"
    TOKENIZER_ID = "EleutherAI/gpt-neox-20b"

    df = load_and_transcribe_data(PROCESSED_DATA_PATH, args.whisper_model)
    df['label_id'] = df['label'].apply(lambda x: 1 if x == 'Dementia' else 0)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=args.seed, stratify=df['label_id'])

    print(f"Loading tokenizer: {TOKENIZER_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Tokenizing datasets...")
    max_length = 2048
    train_encodings = tokenizer(train_df['transcription'].tolist(), truncation=True, padding=True, max_length=max_length)
    test_encodings = tokenizer(test_df['transcription'].tolist(), truncation=True, padding=True, max_length=max_length)

    train_dataset = MCIDataset(train_encodings, train_df['label_id'].tolist())
    test_dataset = MCIDataset(test_encodings, test_df['label_id'].tolist())

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Downloading pre-trained weights from {args.mamba_model}...")
    model_weights_path = hf_hub_download(repo_id=args.mamba_model, filename="pytorch_model.bin")
    pretrained_weights = torch.load(model_weights_path, map_location="cpu")

    config = infer_config_from_state_dict(pretrained_weights)
    print(f"Inferred config -> hidden_size: {config.hidden_size}, layers: {config.num_hidden_layers}, d_inner: {config.d_inner}, dt_rank: {config.dt_rank}, state_size: {config.state_size}, conv_kernel: {config.conv_kernel}")

    print("Initializing Mamba-like architecture (server-safe)...")
    model = MambaForClassification(config)

    missing, unexpected = model.load_state_dict(pretrained_weights, strict=False)
    print(f"Weights loaded with {len(missing)} missing and {len(unexpected)} unexpected keys.")
    if missing:
        print(f"Missing keys (expected at least classifier.*): {missing[:8]}{' ...' if len(missing)>8 else ''}")
    if unexpected:
        print(f"Unexpected keys (often lm_head.*): {unexpected[:8]}{' ...' if len(unexpected)>8 else ''}")

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    history = {'train_loss': [], 'val_accuracy': [], 'val_f1': []}

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        avg_loss = total_loss / max(1, len(train_loader))
        history['train_loss'].append(avg_loss)
        print(f"Epoch {epoch+1} Average Training Loss: {avg_loss:.4f}")

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                out = model(input_ids=input_ids)
                preds = torch.argmax(out.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        history['val_accuracy'].append(acc)
        history['val_f1'].append(f1)
        print(f"Epoch {epoch+1} - Validation Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

    MODEL_OUTPUT_DIR = "trained_model_mamba_pretrained"
    print(f"Training complete. Saving model to {MODEL_OUTPUT_DIR}")
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_OUTPUT_DIR, "pytorch_model.bin"))
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

    history_path = os.path.join(MODEL_OUTPUT_DIR, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to {history_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a GLIBC-safe Mamba-like classifier for MCI detection.")
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=1, help='Training and evaluation batch size.')
    parser.add_argument('--whisper_model', type=str, default='tiny.en', help='Whisper model size.')
    parser.add_argument('--mamba_model', type=str, default='state-spaces/mamba-130m', help='HF repo id for weights.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    args = parser.parse_args()
    main(args)
