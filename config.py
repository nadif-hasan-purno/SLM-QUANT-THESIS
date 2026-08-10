"""
config.py
Central configuration for the SLM Bangla-English benchmarking pipeline.

Target environment : Mac mini M4 (Apple Silicon, 16GB unified memory)
Inference backend   : Ollama (handles FP16 / 8-bit / 4-bit GGUF quantization locally)
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasets"
RESULTS_DIR = BASE_DIR / "results"

DATASET_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Your uploaded CSV (rename/copy it to this path, or edit this line)
# Updated dataset: 480 rows, 6 balanced categories (incl. Hallucination and Safety),
# Coding now 40/40 English-Bangla, plus a "difficulty" column.
RAW_DATASET_PATH = DATASET_DIR / "BNEdgeBench-400_updated.csv"
# RAW_DATASET_PATH = DATASET_DIR / "BNEdgeBench-400_mini.csv"
CLEAN_DATASET_PATH = DATASET_DIR / "BNEdgeBench-400_clean.csv"

RAW_RESULTS_PATH = RESULTS_DIR / "raw_results.csv"
EVALUATED_RESULTS_PATH = RESULTS_DIR / "evaluated_results.csv"

# ---------------------------------------------------------------------------
# Models & quantization
# ---------------------------------------------------------------------------
# Maps a friendly model name -> Ollama tag for each quantization level.
# Pull each tag once before benchmarking, e.g.:
#   ollama pull tinyllama:1.1b-chat-v1-fp16
#   ollama pull tinyllama:1.1b-chat-v1-q8_0
#   ollama pull tinyllama:1.1b-chat-v1-q4_0
#   (repeat for phi and gemma tags below)
#
# These tags were verified against the Ollama library (ollama.com/library).
# Approx download sizes: TinyLlama 2.2/1.2/0.64GB, Phi-2 5.6/3.0/1.6GB,
# Gemma 2B 4.5/2.7/~1.5GB (fp16/8bit/4bit).
MODELS = {
    "tinyllama": {
        "fp16": "tinyllama:1.1b-chat-v1-fp16",
        "8bit": "tinyllama:1.1b-chat-v1-q8_0",
        "4bit": "tinyllama:1.1b-chat-v1-q4_0",
    },
    "phi2": {
        "fp16": "phi:2.7b-chat-v2-fp16",
        "8bit": "phi:2.7b-chat-v2-q8_0",
        "4bit": "phi:2.7b-chat-v2-q4_0",
    },
    "gemma2b": {
        "fp16": "gemma:2b-instruct-fp16",
        "8bit": "gemma:2b-instruct-q8_0",
        "4bit": "gemma:2b-instruct-q4_0",
    },
}
# If a tag isn't pulled on your machine, benchmark.py records
# status="unavailable" for that combination instead of crashing.

# ---------------------------------------------------------------------------
# Generation settings
# ---------------------------------------------------------------------------
GENERATION_OPTIONS = {
    "temperature": 0.2,
    "num_predict": 256,   # max new tokens per response
    "top_p": 0.9,
}

REPETITIONS = 1   # runs per prompt; raise to 3 for more stable timing averages

# ---------------------------------------------------------------------------
# Dataset schema (matches BNEdgeBench-400_updated.csv)
# ---------------------------------------------------------------------------
DATASET_COLUMNS = ["id", "language", "task", "domain", "difficulty", "prompt", "reference_answer", "source"]
VALID_LANGUAGES = {"English", "Bangla"}
VALID_TASKS = {
    "Question Answering", "Translation", "Summarization",
    "Coding", "Reasoning", "Hallucination and Safety",
}
VALID_DIFFICULTIES = {"Easy", "Medium", "Hard"}
