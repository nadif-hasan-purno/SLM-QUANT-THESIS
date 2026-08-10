"""
resume_benchmark.py
Resumes benchmarking ONLY for rows in raw_results.csv where status != 'ok'.
Leaves successful runs untouched and saves progress incrementally.
"""

import os
import pandas as pd
from benchmark import run_single_prompt, model_is_available
from config import (
    CLEAN_DATASET_PATH, RAW_RESULTS_PATH, MODELS
)

def main():
    if not os.path.exists(RAW_RESULTS_PATH):
        print(f"Error: Could not find {RAW_RESULTS_PATH}")
        return

    raw_df = pd.read_csv(RAW_RESULTS_PATH)
    clean_df = pd.read_csv(CLEAN_DATASET_PATH)

    # Map prompt_id -> actual prompt text
    prompt_map = dict(zip(clean_df['id'], clean_df['prompt']))

    failed_mask = raw_df['status'] != 'ok'
    failed_count = failed_mask.sum()

    print(f"Total rows in raw_results.csv: {len(raw_df)}")
    print(f"Already completed ('ok'): {len(raw_df) - failed_count}")
    print(f"Rows remaining to process: {failed_count}\n")

    if failed_count == 0:
        print("All 4,320 rows are already status='ok'! Nothing to do.")
        return

    # Create model x quant -> tag mapping
    tag_map = {}
    for m_name, quants in MODELS.items():
        for q_name, tag in quants.items():
            tag_map[(m_name, q_name)] = tag

    processed = 0
    for idx, row in raw_df[failed_mask].iterrows():
        model_name = row['model']
        quant = row['quantization']
        prompt_id = row['prompt_id']
        tag = tag_map.get((model_name, quant))

        prompt_text = prompt_map.get(prompt_id)
        if not tag or not prompt_text:
            continue

        if not model_is_available(tag):
            print(f"[{processed+1}/{failed_count}] Skipping {model_name} [{quant}] ({prompt_id}): Tag '{tag}' NOT PULLED in Ollama.")
            raw_df.at[idx, 'status'] = "unavailable"
            processed += 1
            continue

        print(f"[{processed+1}/{failed_count}] Processing {model_name} [{quant}] -> Prompt: {prompt_id}...")
        try:
            metrics = run_single_prompt(tag, prompt_text)
            raw_df.at[idx, 'status'] = 'ok'
            for k, v in metrics.items():
                raw_df.at[idx, k] = v
        except Exception as e:
            print(f"   -> Failed again: {e}")
            raw_df.at[idx, 'status'] = f"error: {e}"

        processed += 1

        # Save progress every 10 rows so you never lose work if interrupted
        if processed % 10 == 0 or processed == failed_count:
            raw_df.to_csv(RAW_RESULTS_PATH, index=False)

    print(f"\nDone! Updated raw results saved to {RAW_RESULTS_PATH}")

if __name__ == "__main__":
    main()