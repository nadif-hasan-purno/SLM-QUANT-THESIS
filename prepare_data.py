"""
prepare_data.py
Validates and cleans BNEdgeBench-400 before it's used for benchmarking.

Run:
    python prepare_data.py
"""

import pandas as pd

from config import (
    RAW_DATASET_PATH, CLEAN_DATASET_PATH,
    DATASET_COLUMNS, VALID_LANGUAGES, VALID_TASKS, VALID_DIFFICULTIES,
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


if __name__ == "__main__":
    main()
