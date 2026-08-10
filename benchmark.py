"""
benchmark.py
Runs every prompt in BNEdgeBench-400 through each model x quantization
combination using Ollama (local inference on Mac mini M4), recording
efficiency metrics (latency, memory, CPU, tokens/sec).

Requires:
    pip install ollama psutil pandas
    Ollama app running locally, with model tags already pulled
    (see config.py -> MODELS for the exact tags to `ollama pull`)

Run:
    python benchmark.py
"""

import time

import ollama
import pandas as pd
import psutil

from config import (
    CLEAN_DATASET_PATH, RAW_RESULTS_PATH, MODELS,
    GENERATION_OPTIONS, REPETITIONS,
)


def _get(obj, key, default=None):
    """Read a field from an Ollama response whether it's a dict (older
    ollama-python) or a typed object with attributes (newer ollama-python)."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def model_is_available(tag):
    """Check whether a model tag has already been pulled in Ollama."""
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

    load_ns = _get(response, "load_duration", 0) or 0        # ~ cold-start / model-load time
    eval_ns = _get(response, "eval_duration", 1) or 1         # generation-only time
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


def main():
    df = pd.read_csv(CLEAN_DATASET_PATH)
    results = []

    # Debug aid: show exactly what Ollama thinks is installed, so any
    # mismatch with the tags in config.py -> MODELS is easy to spot.
    try:
        listing = ollama.list()
        local_tags = [_get(m, "model") or _get(m, "name") for m in _get(listing, "models", [])]
        print("Locally available Ollama models:")
        for t in local_tags:
            print("  -", t)
    except Exception as e:
        print(f"Could not list local Ollama models ({e}). Is `ollama serve` running?")

    for model_name, quant_tags in MODELS.items():
        for quant, tag in quant_tags.items():
            available = model_is_available(tag)
            print(f"\n== {model_name} [{quant}] -> {tag} "
                  f"({'available' if available else 'NOT PULLED, skipping'}) ==")

            for _, row in df.iterrows():
                for rep in range(REPETITIONS):
                    base_info = {
                        "model": model_name, "quantization": quant,
                        "prompt_id": row["id"], "language": row["language"],
                        "task": row["task"], "repetition": rep,
                    }

                    if not available:
                        results.append({**base_info, "status": "unavailable"})
                        continue

                    try:
                        metrics = run_single_prompt(tag, row["prompt"])
                        results.append({**base_info, "status": "ok", **metrics})
                    except Exception as e:
                        results.append({**base_info, "status": f"error: {e}"})

    out_df = pd.DataFrame(results)
    out_df.to_csv(RAW_RESULTS_PATH, index=False)
    print(f"\nSaved raw benchmark results -> {RAW_RESULTS_PATH}")


if __name__ == "__main__":
    main()
