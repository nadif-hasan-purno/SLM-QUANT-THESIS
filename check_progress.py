"""
check_progress.py
Shows exactly how much of the benchmark run survived, grouped by model/quant.
Run this from your project folder:
    python check_progress.py
"""

import pandas as pd
from config import RAW_RESULTS_PATH, MODELS, REPETITIONS
import pandas as pd

try:
    df = pd.read_csv(RAW_RESULTS_PATH)
except FileNotFoundError:
    print(f"No results file found at {RAW_RESULTS_PATH} -- nothing was saved yet.")
    raise SystemExit

print(f"Total rows saved: {len(df)}\n")

print("Progress by model x quantization:")
print(df.groupby(["model", "quantization"]).size())

print("\nStatus breakdown:")
print(df["status"].value_counts())

# How much is left to do, per the current config.py
total_prompts = None
try:
    from config import CLEAN_DATASET_PATH
    dataset = pd.read_csv(CLEAN_DATASET_PATH)
    total_prompts = len(dataset)
except Exception:
    pass

if total_prompts:
    total_planned = sum(len(q) for q in MODELS.values()) * total_prompts * REPETITIONS
    print(f"\nTotal planned rows (per current config.py): {total_planned}")
    print(f"Remaining: {total_planned - len(df)}")
