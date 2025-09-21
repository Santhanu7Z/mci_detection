#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
from pathlib import Path
import re

# Path to your data folder
DATA_DIR = Path.cwd() / "data" / "Pitt"

# Simple regex to find any "digits_digits" pattern (timestamps)
TIMESTAMP_PATTERN = re.compile(r'(\d+)_(\d+)')

# Function to debug one file
def debug_cha_file(cha_path):
    print(f"\n=== Debugging file: {cha_path} ===\n")
    with open(cha_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
        for i, line in enumerate(f):
            line_strip = line.strip()
            if "*PAR" in line_strip:  # Simple filter for lines containing *PAR
                ts_matches = TIMESTAMP_PATTERN.findall(line_strip)
                print(f"[Line {i}] {line_strip}")
                if ts_matches:
                    print(f"   >> Timestamps found: {ts_matches}")
                else:
                    print(f"   >> No timestamps found")
    print("\n============================\n")


# Loop over some files (adjust to one or a few for inspection)
example_files = list((DATA_DIR / "Control").glob("*.cha"))[:3]  # first 3 files
for fpath in example_files:
    debug_cha_file(fpath)

