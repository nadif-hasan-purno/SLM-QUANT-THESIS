"""
benchmark.py
Runs every prompt in BNEdgeBench-400 through each model x quantization
combination using Ollama (local inference on Mac mini M4), recording
efficiency metrics (latency, memory, CPU, tokens/sec).

Key features for long runs:
  - Writes each result to disk immediately (not just at the end), so a
    crash/interrupt doesn't lose progress.
  - RESUMES automatically: re-running the script skips (model, quant,
    prompt_id, repetition) combos that already SUCCEEDED (status == "ok").
    Rows that failed with "unavailable" or "error: ..." are NOT counted as
    done -- they get retried, since the underlying cause (model not pulled
    yet, Ollama server down, etc.) may since have been fixed.
  - Prints running progress + ETA so you can tell it's not frozen.
  - --limit N restricts to the first N prompts, for a quick smoke test.
  - --models tinyllama_lora restricts to specific models (comma-separated).
  - --quant 4bit restricts to specific quant levels (comma-separated).
  - Handles both dict-style and typed-object-style responses from the
    `ollama` Python package, since this varies by installed version.

Requires:
    pip install ollama psutil pandas
    Ollama app running locally, with model tags already pulled/created

Run:
    python benchmark.py                                    # full sweep, resumes if re-run
    python benchmark.py --limit 5 --models tinyllama_lora --quant 4bit   # quick scoped test
    python benchmark.py --models tinyllama_lora --quant 4bit             # full scoped run
"""

import argparse
import csv
import os
import time

import ollama
import pandas as pd
import psutil

from config import (
    CLEAN_DATASET_PATH, RAW_RESULTS_PATH, MODELS,
    GENERATION_OPTIONS, REPETITIONS,
)

FIELDNAMES = [
    "model", "quantization", "prompt_id", "language", "task", "repetition",
    "status", "response", "cold_start_latency", "warm_inference_latency",
    "latency_seconds", "memory_gb", "cpu_percent", "input_tokens",
    "output_tokens", "tokens_per_second",
]


def _get(obj, key, default=None):
    """Read a field from an Ollama response whether it's a dict (older
    ollama-python) or a typed object with attributes (newer ollama-python)."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def model_is_available(tag):
    """Check whether a model tag has already been pulled/created in Ollama."""
    try:
        listing = ollama.list()
        models = _get(listing, "models", [])
        local_tags = [_get(m, "model") or _get(m, "name") for m in models]
        return tag in local_tags
    except Exception as e:
        print(f"   (could not check local model list: {e})")
        return False


def run_single_prompt(tag, prompt):
    """Send one prompt to Ollama and measure latency/memory/CPU/tokens."""
    process = psutil.Process()
    cpu_before = psutil.cpu_percent(interval=None)
    mem_before = process.memory_info().rss / (1024 ** 3)  # GB

    start = time.time()
    response = ollama.generate(model=tag, prompt=prompt, options=GENERATION_OPTIONS)
    elapsed = time.time() - start

    cpu_after = psutil.cpu_percent(interval=None)
    mem_after = process.memory_info().rss / (1024 ** 3)

    load_ns = _get(response, "load_duration", 0) or 0
    eval_ns = _get(response, "eval_duration", 1) or 1
    output_tokens = _get(response, "eval_count", 0) or 0
    input_tokens = _get(response, "prompt_eval_count", 0) or 0

    tokens_per_second = output_tokens / (eval_ns / 1e9) if eval_ns else 0.0

    return {
        "response": _get(response, "response", "") or "",
        "cold_start_latency": load_ns / 1e9,
        "warm_inference_latency": eval_ns / 1e9,
        "latency_seconds": elapsed,
        "memory_gb": max(mem_after, mem_before),
        "cpu_percent": max(cpu_after, cpu_before),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokens_per_second": tokens_per_second,
    }


def load_completed_keys(path):
    """Read any existing raw_results.csv and return the set of
    (model, quantization, prompt_id, repetition) combos that already
    SUCCEEDED (status == 'ok'). Rows that failed are NOT counted as done."""
    done = set()
    if not os.path.exists(path):
        return done
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("status") != "ok":
                    continue
                key = (row["model"], row["quantization"], row["prompt_id"], row["repetition"])
                done.add(key)
    except Exception as e:
        print(f"Could not read existing results for resume ({e}); starting fresh.")
    return done


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None,
                    help="Only use the first N prompts (per model/quant) -- quick test")
    p.add_argument("--models", type=str, default=None,
                    help="Comma-separated subset of MODELS keys, e.g. tinyllama_lora")
    p.add_argument("--quant", type=str, default=None,
                    help="Comma-separated subset of quant levels, e.g. 4bit,8bit")
    return p.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(CLEAN_DATASET_PATH)
    if args.limit:
        df = df.head(args.limit)

    models_to_run = MODELS
    if args.models:
        wanted = set(args.models.split(","))
        models_to_run = {k: v for k, v in MODELS.items() if k in wanted}
        unknown = wanted - set(MODELS.keys())
        if unknown:
            print(f"WARNING: --models included unknown key(s) not in config.py MODELS: {unknown}")

    quants_to_run = args.quant.split(",") if args.quant else None

    print(f"\nRunning models: {list(models_to_run.keys())}"
          f"{f' (quant filter: {quants_to_run})' if quants_to_run else ''}")

    try:
        listing = ollama.list()
        local_tags = [_get(m, "model") or _get(m, "name") for m in _get(listing, "models", [])]
        print("\nLocally available Ollama models:")
        for t in local_tags:
            print("  -", t)
    except Exception as e:
        print(f"Could not list local Ollama models ({e}). Is `ollama serve` running?")

    done_keys = load_completed_keys(RAW_RESULTS_PATH)
    if done_keys:
        print(f"\nFound {len(done_keys)} already-completed rows in "
              f"{RAW_RESULTS_PATH} -- these will be skipped (resume mode).")

    combos = [(m, q, t) for m, qt in models_to_run.items() for q, t in qt.items()
              if quants_to_run is None or q in quants_to_run]

    if not combos:
        print("\nNo model/quant combinations matched your filters. Nothing to do.")
        print(f"  models_to_run keys: {list(models_to_run.keys())}")
        print(f"  quants_to_run: {quants_to_run}")
        return

    total_planned = len(combos) * len(df) * REPETITIONS
    completed_count = 0
    run_start = time.time()

    file_exists = os.path.exists(RAW_RESULTS_PATH)
    csv_file = open(RAW_RESULTS_PATH, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
    if not file_exists:
        writer.writeheader()

    for model_name, quant, tag in combos:
        available = model_is_available(tag)
        print(f"\n== {model_name} [{quant}] -> {tag} "
              f"({'available' if available else 'NOT PULLED, skipping'}) ==")

        for _, row in df.iterrows():
            for rep in range(REPETITIONS):
                key = (model_name, quant, row["id"], str(rep))
                if key in done_keys:
                    completed_count += 1
                    continue

                base_info = {
                    "model": model_name, "quantization": quant,
                    "prompt_id": row["id"], "language": row["language"],
                    "task": row["task"], "repetition": rep,
                }

                if not available:
                    result = {**base_info, "status": "unavailable"}
                else:
                    try:
                        metrics = run_single_prompt(tag, row["prompt"])
                        result = {**base_info, "status": "ok", **metrics}
                    except Exception as e:
                        result = {**base_info, "status": f"error: {e}"}

                writer.writerow(result)
                csv_file.flush()
                completed_count += 1

                if completed_count % 10 == 0 or completed_count == total_planned:
                    elapsed = time.time() - run_start
                    rate = completed_count / elapsed if elapsed > 0 else 0
                    remaining = total_planned - completed_count
                    eta_min = (remaining / rate / 60) if rate > 0 else float("inf")
                    print(f"  progress: {completed_count}/{total_planned} "
                          f"({completed_count/total_planned*100:.1f}%) "
                          f"-- ETA ~{eta_min:.1f} min", end="\r")

    csv_file.close()
    print(f"\n\nSaved raw benchmark results -> {RAW_RESULTS_PATH}")


if __name__ == "__main__":
    main()
