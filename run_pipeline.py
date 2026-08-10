"""
run_pipeline.py
Runs the full BNEdgeBench-400 benchmarking pipeline end-to-end:

    1. (optional) pulls every Ollama tag listed in config.MODELS
    2. prepare_data.py  -- clean + validate the dataset
    3. benchmark.py     -- run every prompt through every model x quantization
    4. evaluate.py      -- score the responses

Run:
    python run_pipeline.py            # pulls missing models, then runs everything
    python run_pipeline.py --no-pull  # skip pulling, just run the 3 steps
    python run_pipeline.py --pull-only  # only pull models, don't run anything
"""

import subprocess
import sys
import time

from config import MODELS


def pull_models():
    """Pull every Ollama tag in config.MODELS that isn't already downloaded."""
    print("\n=== Checking / pulling Ollama models ===")
    for model_name, quant_tags in MODELS.items():
        for quant, tag in quant_tags.items():
            print(f"\n-> ollama pull {tag}  ({model_name} / {quant})")
            result = subprocess.run(["ollama", "pull", tag])
            if result.returncode != 0:
                print(f"   WARNING: failed to pull {tag} -- benchmark.py will "
                      f"mark this combination as 'unavailable' and continue.")


def run_step(description, script):
    print(f"\n=== {description} ({script}) ===")
    start = time.time()
    result = subprocess.run([sys.executable, script])
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"\n'{script}' failed (exit code {result.returncode}). Stopping pipeline.")
        sys.exit(result.returncode)
    print(f"--- {script} finished in {elapsed:.1f}s ---")


def main():
    args = sys.argv[1:]
    do_pull = "--no-pull" not in args
    pull_only = "--pull-only" in args

    if do_pull:
        pull_models()

    if pull_only:
        print("\n--pull-only set, skipping the rest of the pipeline.")
        return

    run_step("Step 1/3: Cleaning and validating dataset", "prepare_data.py")
    run_step("Step 2/3: Running benchmark", "benchmark.py")
    run_step("Step 3/3: Evaluating responses", "evaluate.py")

    print("\nAll done. Check the results/ folder for:")
    print("  - raw_results.csv       (benchmark.py output)")
    print("  - evaluated_results.csv (evaluate.py output)")


if __name__ == "__main__":
    main()
