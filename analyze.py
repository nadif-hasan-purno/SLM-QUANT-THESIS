import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from config import (
    EVALUATED_RESULTS_PATH,
    RESULTS_DIR,
    FIGURES_DIR,
)

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("LOADING EVALUATED RESULTS")
print("=" * 60)

df = pd.read_csv(EVALUATED_RESULTS_PATH)

print(f"Total rows: {len(df)}")
print(f"Models: {df['model'].unique().tolist()}")
print(f"Quantizations: {df['quantization'].unique().tolist()}")

# Convert important columns to numeric
numeric_columns = [
    "latency_seconds",
    "memory_gb",
    "cpu_percent",
    "input_tokens",
    "output_tokens",
    "tokens_per_second",
    "token_f1",
    "exact_match",
    "bleu",
    "chrf",
    "rouge1",
    "rouge2",
    "rougeL",
    "syntax_ok",
    "appropriate_response_flag",
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


# ============================================================
# BASIC DATA CHECK
# ============================================================

print("\n" + "=" * 60)
print("ROWS BY MODEL AND QUANTIZATION")
print("=" * 60)

counts = (
    df.groupby(["model", "quantization"])
      .size()
      .reset_index(name="rows")
)

print(counts.to_string(index=False))


# ============================================================
# FIGURE 1 — AVERAGE LATENCY
# ============================================================

print("\nCreating Figure 1...")

latency = (
    df.groupby(["model", "quantization"])["latency_seconds"]
      .mean()
      .reset_index()
)

plt.figure(figsize=(12, 6))

labels = (
    latency["model"] + "\n" +
    latency["quantization"]
)

plt.bar(labels, latency["latency_seconds"])

plt.xlabel("Model / Quantization")
plt.ylabel("Average Latency (seconds)")
plt.title("Average Inference Latency")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(FIGURES_DIR / "1_average_latency.png", dpi=300)
plt.close()

print("Saved: 1_average_latency.png")


# ============================================================
# FIGURE 2 — MEMORY USAGE
# ============================================================

print("Creating Figure 2...")

memory = (
    df.groupby(["model", "quantization"])["memory_gb"]
      .mean()
      .reset_index()
)

plt.figure(figsize=(12, 6))

labels = (
    memory["model"] + "\n" +
    memory["quantization"]
)

plt.bar(labels, memory["memory_gb"])

plt.xlabel("Model / Quantization")
plt.ylabel("Average Memory (GB)")
plt.title("Average Memory Usage")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(FIGURES_DIR / "2_average_memory.png", dpi=300)
plt.close()

print("Saved: 2_average_memory.png")


# ============================================================
# FIGURE 3 — TOKENS PER SECOND
# ============================================================

print("Creating Figure 3...")

speed = (
    df.groupby(["model", "quantization"])["tokens_per_second"]
      .mean()
      .reset_index()
)

plt.figure(figsize=(12, 6))

labels = (
    speed["model"] + "\n" +
    speed["quantization"]
)

plt.bar(labels, speed["tokens_per_second"])

plt.xlabel("Model / Quantization")
plt.ylabel("Tokens per Second")
plt.title("Average Inference Speed")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(FIGURES_DIR / "3_tokens_per_second.png", dpi=300)
plt.close()

print("Saved: 3_tokens_per_second.png")


# ============================================================
# FIGURE 4 — TOKEN F1
# ============================================================

print("Creating Figure 4...")

f1 = (
    df.groupby(["model", "quantization"])["token_f1"]
      .mean()
      .reset_index()
)

plt.figure(figsize=(12, 6))

labels = (
    f1["model"] + "\n" +
    f1["quantization"]
)

plt.bar(labels, f1["token_f1"])

plt.xlabel("Model / Quantization")
plt.ylabel("Average Token F1")
plt.title("Average Token F1 Score")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(FIGURES_DIR / "4_token_f1.png", dpi=300)
plt.close()

print("Saved: 4_token_f1.png")


# ============================================================
# TASK-LEVEL TOKEN F1
# ============================================================

print("\nCreating task-level summary...")

task_f1 = (
    df.groupby(["model", "quantization", "task"])["token_f1"]
      .mean()
      .reset_index()
)

task_f1.to_csv(
    RESULTS_DIR / "task_token_f1_summary.csv",
    index=False
)

print("Saved: task_token_f1_summary.csv")


# ============================================================
# LANGUAGE-LEVEL TOKEN F1
# ============================================================

language_f1 = (
    df.groupby(["model", "quantization", "language"])["token_f1"]
      .mean()
      .reset_index()
)

language_f1.to_csv(
    RESULTS_DIR / "language_token_f1_summary.csv",
    index=False
)

print("Saved: language_token_f1_summary.csv")


# ============================================================
# TINYLLAMA BASE 4-BIT VS LORA 4-BIT
# ============================================================

print("\n" + "=" * 60)
print("TINYLLAMA BASE 4-BIT VS LORA 4-BIT")
print("=" * 60)

# IMPORTANT:
# Only compare TinyLlama 4-bit with LoRA 4-bit.
# This prevents fp16 and 8-bit rows from being mixed into
# the comparison.

base = df[
    (df["model"] == "tinyllama") &
    (df["quantization"] == "4bit")
].copy()

lora = df[
    (df["model"] == "tinyllama_lora") &
    (df["quantization"] == "4bit")
].copy()

print(f"TinyLlama 4-bit rows: {len(base)}")
print(f"LoRA 4-bit rows: {len(lora)}")


# ------------------------------------------------------------
# Overall comparison
# ------------------------------------------------------------

base_latency = base["latency_seconds"].mean()
lora_latency = lora["latency_seconds"].mean()

base_memory = base["memory_gb"].mean()
lora_memory = lora["memory_gb"].mean()

base_speed = base["tokens_per_second"].mean()
lora_speed = lora["tokens_per_second"].mean()

base_f1 = base["token_f1"].mean()
lora_f1 = lora["token_f1"].mean()

comparison = pd.DataFrame({
    "metric": [
        "Average Latency (seconds)",
        "Average Memory (GB)",
        "Tokens per Second",
        "Token F1"
    ],
    "TinyLlama_4bit": [
        base_latency,
        base_memory,
        base_speed,
        base_f1
    ],
    "TinyLlama_LoRA_4bit": [
        lora_latency,
        lora_memory,
        lora_speed,
        lora_f1
    ]
})

comparison.to_csv(
    RESULTS_DIR / "tinyllama_base_vs_lora_summary.csv",
    index=False
)

print("\nOverall comparison:")
print(comparison.to_string(index=False))


# ============================================================
# FIGURE 5 — BASE VS LORA
# ============================================================

print("\nCreating Base vs LoRA figure...")

metrics = [
    "Latency",
    "Memory",
    "Tokens/sec",
    "Token F1"
]

base_values = [
    base_latency,
    base_memory,
    base_speed,
    base_f1
]

lora_values = [
    lora_latency,
    lora_memory,
    lora_speed,
    lora_f1
]

# Normalize each metric for visual comparison.
# This avoids latency/memory dominating the graph.

import numpy as np

base_array = np.array(base_values, dtype=float)
lora_array = np.array(lora_values, dtype=float)

max_values = np.maximum(base_array, lora_array)

base_normalized = base_array / max_values
lora_normalized = lora_array / max_values

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    x - width / 2,
    base_normalized,
    width,
    label="TinyLlama 4-bit"
)

plt.bar(
    x + width / 2,
    lora_normalized,
    width,
    label="TinyLlama LoRA 4-bit"
)

plt.xticks(x, metrics)
plt.ylabel("Normalized Value")
plt.title("TinyLlama 4-bit vs LoRA 4-bit")
plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "5_tinyllama_before_after_lora.png",
    dpi=300
)

plt.close()

print("Saved: 5_tinyllama_before_after_lora.png")


# ============================================================
# OVERALL SUMMARY
# ============================================================

overall_summary = (
    df.groupby(["model", "quantization"])
      .agg(
          samples=("prompt_id", "count"),
          avg_latency_seconds=("latency_seconds", "mean"),
          avg_memory_gb=("memory_gb", "mean"),
          avg_cpu_percent=("cpu_percent", "mean"),
          avg_input_tokens=("input_tokens", "mean"),
          avg_output_tokens=("output_tokens", "mean"),
          avg_tokens_per_second=("tokens_per_second", "mean"),
          avg_token_f1=("token_f1", "mean"),
          avg_exact_match=("exact_match", "mean"),
          avg_bleu=("bleu", "mean"),
          avg_chrf=("chrf", "mean"),
          avg_rouge1=("rouge1", "mean"),
          avg_rouge2=("rouge2", "mean"),
          avg_rougeL=("rougeL", "mean"),
          syntax_success_rate=("syntax_ok", "mean"),
          appropriate_response_rate=(
              "appropriate_response_flag",
              "mean"
          ),
      )
      .reset_index()
)

overall_summary.to_csv(
    RESULTS_DIR / "overall_summary.csv",
    index=False
)

print("\nSaved: overall_summary.csv")


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)

print(f"Evaluated rows: {len(df)}")
print(f"Figures directory: {FIGURES_DIR}")
print(f"Results directory: {RESULTS_DIR}")

print("\nGenerated figures:")

for file in sorted(FIGURES_DIR.glob("*.png")):
    print(" -", file.name)

