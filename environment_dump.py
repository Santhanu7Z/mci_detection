# environment_dump.py

import torch
import transformers
import sklearn
import platform
import sys

print("=" * 60)
print("ENVIRONMENT DIAGNOSTICS")
print("=" * 60)
print(f"PYTHON:       {sys.version}")
print(f"PLATFORM:     {platform.platform()}")
print("-" * 60)
print(f"TORCH:        {torch.__version__}")
print(f"CUDA:         {torch.version.cuda if torch.cuda.is_available() else 'None'}")
print("-" * 60)
print(f"TRANSFORMERS: {transformers.__version__}")
print(f"SKLEARN:      {sklearn.__version__}")
print("=" * 60)
