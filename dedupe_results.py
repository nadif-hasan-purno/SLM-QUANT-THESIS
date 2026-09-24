import pandas as pd
from pathlib import Path
from config import RAW_RESULTS_PATH

raw_path = Path(RAW_RESULTS_PATH)

if not raw_path.exists():
    print("ERROR: raw_results.csv not found")
    exit()

df = pd.read_csv(raw_path)

print("Original rows:", len(df))

columns = ["model", "quantization", "prompt_id", "repetition"]

duplicates = df.duplicated(subset=columns).sum()

df = df.drop_duplicates(
    subset=columns,
    keep="first"
)

df.to_csv(raw_path, index=False)

print("Duplicates removed:", duplicates)
print("Final rows:", len(df))
print("Saved:", raw_path)

print()
print("Rows by model and quantization:")
print(
    df.groupby(["model", "quantization"])
      .size()
      .to_string()
)
