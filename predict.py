import torch
import torch.nn as nn
import whisper
from transformers import AutoTokenizer
import argparse
import os
import math
import json
import random

# --- SERVER-COMPATIBLE MAMBA IMPLEMENTATION ---
# This section is an EXACT copy of the final, working train.py script's
# model definitions to ensure perfect architectural alignment.

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

# --- Configuration ---
MODEL_PATH = "trained_model_mamba_pretrained"
TOKENIZER_ID = "EleutherAI/gpt-neox-20b" # Use the original tokenizer for consistency

def predict_mci(audio_path, whisper_model_size):
    if not os.path.exists(audio_path):
        return "Error: Audio file not found.", 0.0
    if not os.path.exists(os.path.join(MODEL_PATH, "pytorch_model.bin")):
        return "Error: Trained model not found. Please run train.py first.", 0.0

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading Whisper model ({whisper_model_size}) and transcribing...")
    whisper_model = whisper.load_model(whisper_model_size, device=device)
    result = whisper_model.transcribe(audio_path, fp16=torch.cuda.is_available())
    transcription = result['text']
    print(f"\nTranscription: {transcription}\n")

    print("Loading fine-tuned Mamba model...")
    # Load the tokenizer from the original source to ensure consistency
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Instantiate the config with the known, correct values from our training run
    config = InferredMambaConfig(
        vocab_size=50280, # True vocab size from the pre-trained model
        hidden_size=768,
        num_hidden_layers=24,
        d_inner=1536,
        dt_rank=48,
        state_size=16,
        conv_kernel=4,
        num_labels=2
    )
    
    # Instantiate the model with the correct architecture
    model = MambaForClassification(config)
    
    # Load the fine-tuned weights, using your map_location suggestion for robustness
    state_dict = torch.load(os.path.join(MODEL_PATH, "pytorch_model.bin"), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print("Making prediction...")
    inputs = tokenizer(transcription, return_tensors="pt", truncation=True, padding=True, max_length=2048).to(device)

    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'])
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        
    dementia_confidence = probabilities[0][1].item()
    predicted_class_id = torch.argmax(probabilities).item()
    
    prediction = "Dementia" if predicted_class_id == 1 else "Control"

    return prediction, dementia_confidence

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict MCI from an audio file.")
    parser.add_argument('audio_file', type=str, help="Path to the audio file to be analyzed.")
    parser.add_argument('--whisper_model', type=str, default='tiny.en', help='Whisper model size to use for transcription.')
    args = parser.parse_args()

    predicted_label, confidence = predict_mci(args.audio_file, args.whisper_model)
    
    if "Error" not in predicted_label:
        print(f"\n--- Prediction Result ---")
        print(f"Predicted Label: {predicted_label}")
        print(f"Confidence (Dementia): {confidence:.4f}")
    else:
        print(f"\n--- Error ---")
        print(predicted_label)
