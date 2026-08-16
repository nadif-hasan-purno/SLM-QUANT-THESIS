"""
prepare_data.py
1. Validates and cleans BNEdgeBench-400 before it's used for benchmarking.
2. Cleans, dedupes, and splits the Bangla instruction dataset for LoRA
   training -- and enforces the rule that no LoRA training example is a
   near-duplicate of a benchmark prompt (so we're not "training on the test set").

Run:
    python prepare_data.py
"""

import json
import random

import pandas as pd

from config import (
    RAW_DATASET_PATH, CLEAN_DATASET_PATH,
    DATASET_COLUMNS, VALID_LANGUAGES, VALID_TASKS, VALID_DIFFICULTIES,
    BANGLA_INSTRUCTION_RAW_PATH, BANGLA_TRAIN_PATH, BANGLA_VALIDATION_PATH,
    BANGLA_TEST_PATH, TRAIN_RATIO, VALIDATION_RATIO, TEST_RATIO, SPLIT_SEED,
)


def load_dataset(path):
    df = pd.read_csv(path)
    missing = set(DATASET_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def validate_dataset(df):
    """Return a list of human-readable warnings (doesn't raise)."""
    issues = []

    bad_lang = df[~df["language"].isin(VALID_LANGUAGES)]
    if len(bad_lang):
        issues.append(f"{len(bad_lang)} rows with invalid 'language' values")

    bad_task = df[~df["task"].isin(VALID_TASKS)]
    if len(bad_task):
        issues.append(f"{len(bad_task)} rows with invalid 'task' values")

    bad_difficulty = df[~df["difficulty"].isin(VALID_DIFFICULTIES)]
    if len(bad_difficulty):
        issues.append(f"{len(bad_difficulty)} rows with invalid 'difficulty' values")

    empty_prompt = df[df["prompt"].isna() | (df["prompt"].astype(str).str.strip() == "")]
    if len(empty_prompt):
        issues.append(f"{len(empty_prompt)} rows with empty 'prompt'")

    empty_ref = df[df["reference_answer"].isna() | (df["reference_answer"].astype(str).str.strip() == "")]
    if len(empty_ref):
        issues.append(f"{len(empty_ref)} rows with empty 'reference_answer'")

    dup_ids = df[df["id"].duplicated()]
    if len(dup_ids):
        issues.append(f"{len(dup_ids)} duplicate 'id' values")

    return issues


def clean_dataset(df):
    df = df.dropna(subset=["prompt", "reference_answer"]).copy()
    for col in ["prompt", "reference_answer"]:
        df[col] = df[col].astype(str).str.strip()
    df = df.drop_duplicates(subset=["prompt"])
    df = df.drop_duplicates(subset=["id"])
    return df.reset_index(drop=True)


def summarize(df):
    print("\n--- Dataset summary (after cleaning) ---")
    print("Total prompts:", len(df))
    print("\nBy language:\n", df["language"].value_counts())
    print("\nBy task:\n", df["task"].value_counts())
    print("\nBy difficulty:\n", df["difficulty"].value_counts())
    print("\nBy language x task:\n", df.groupby(["language", "task"]).size())


def load_instruction_data(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    required_keys = {"instruction", "input", "output"}
    for i, record in enumerate(data):
        missing = required_keys - set(record.keys())
        if missing:
            raise ValueError(f"Record {i} is missing keys: {missing}")
    return data


def clean_instruction_data(data):
    """Strip whitespace, drop empty/near-empty records, and dedupe."""
    cleaned = []
    seen = set()
    for record in data:
        instruction = str(record["instruction"]).strip()
        input_text = str(record["input"]).strip()
        output = str(record["output"]).strip()

        if not instruction or not output:
            continue

        key = (instruction, input_text, output)
        if key in seen:
            continue
        seen.add(key)

        cleaned.append({"instruction": instruction, "input": input_text, "output": output})
    return cleaned


def check_overlap_with_benchmark(instruction_records, benchmark_df):
    """Warn (don't crash) if any LoRA training text closely matches a
    benchmark prompt or reference answer -- this would leak the eval set
    into training. Uses exact substring/equality checks, which is a coarse
    but fast check; review manually if this flags a nontrivial count."""
    benchmark_prompts = set(benchmark_df["prompt"].astype(str).str.strip())
    benchmark_refs = set(benchmark_df["reference_answer"].astype(str).str.strip())

    overlapping = []
    for record in instruction_records:
        combined_input = (record["instruction"] + " " + record["input"]).strip()
        if combined_input in benchmark_prompts or record["output"] in benchmark_refs:
            overlapping.append(record)

    return overlapping


def split_instruction_data(data, train_ratio, val_ratio, test_ratio, seed):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO must sum to 1.0"

    rng = random.Random(seed)
    shuffled = data[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    return train, val, test


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def prepare_instruction_dataset(benchmark_df_clean):
    if not BANGLA_INSTRUCTION_RAW_PATH.exists():
        print(f"\n(No Bangla instruction file found at {BANGLA_INSTRUCTION_RAW_PATH} "
              f"-- skipping LoRA data prep. This is fine if you haven't reached that "
              f"stage yet.)")
        return

    print("\n--- Preparing Bangla instruction dataset (for LoRA) ---")
    raw = load_instruction_data(BANGLA_INSTRUCTION_RAW_PATH)
    print(f"Loaded {len(raw)} raw instruction records")

    cleaned = clean_instruction_data(raw)
    print(f"After cleaning/deduping: {len(cleaned)} records "
          f"({len(raw) - len(cleaned)} removed)")

    overlapping = check_overlap_with_benchmark(cleaned, benchmark_df_clean)
    if overlapping:
        print(f"WARNING: {len(overlapping)} instruction records exactly match a "
              f"benchmark prompt/answer -- removing them to avoid train/test leakage.")
        overlap_keys = {(r["instruction"], r["input"], r["output"]) for r in overlapping}
        cleaned = [r for r in cleaned
                   if (r["instruction"], r["input"], r["output"]) not in overlap_keys]
    else:
        print("No overlap found between instruction data and benchmark prompts.")

    train, val, test = split_instruction_data(
        cleaned, TRAIN_RATIO, VALIDATION_RATIO, TEST_RATIO, SPLIT_SEED
    )
    write_jsonl(train, BANGLA_TRAIN_PATH)
    write_jsonl(val, BANGLA_VALIDATION_PATH)
    write_jsonl(test, BANGLA_TEST_PATH)

    print(f"Split: {len(train)} train / {len(val)} validation / {len(test)} test")
    print(f"Saved -> {BANGLA_TRAIN_PATH}")
    print(f"Saved -> {BANGLA_VALIDATION_PATH}")
    print(f"Saved -> {BANGLA_TEST_PATH}")


def main():
    df = load_dataset(RAW_DATASET_PATH)

    issues = validate_dataset(df)
    if issues:
        print("Validation warnings:")
        for i in issues:
            print(" -", i)
    else:
        print("No validation issues found.")

    df_clean = clean_dataset(df)
    summarize(df_clean)

    df_clean.to_csv(CLEAN_DATASET_PATH, index=False)
    print(f"\nSaved cleaned dataset -> {CLEAN_DATASET_PATH}")

    prepare_instruction_dataset(df_clean)


if __name__ == "__main__":
    main()
