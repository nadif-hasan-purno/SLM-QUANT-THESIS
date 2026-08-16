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
(BASE_DIR / "models").mkdir(exist_ok=True)

# Benchmark dataset (BNEdgeBench-400, updated: 480 rows, 6 balanced categories,
# includes "difficulty" column)
RAW_DATASET_PATH = DATASET_DIR / "BNEdgeBench-400_updated.csv"
CLEAN_DATASET_PATH = DATASET_DIR / "BNEdgeBench-400_clean.csv"

RAW_RESULTS_PATH = RESULTS_DIR / "raw_results.csv"
EVALUATED_RESULTS_PATH = RESULTS_DIR / "evaluated_results.csv"
COMPARISON_RESULTS_PATH = RESULTS_DIR / "comparison_results.csv"
TRAINING_HISTORY_PATH = RESULTS_DIR / "training_history.csv"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Bangla instruction dataset (for LoRA fine-tuning)
# ---------------------------------------------------------------------------
BANGLA_INSTRUCTION_RAW_PATH = DATASET_DIR / "bangla-instruction-dataset-1000.json"

BANGLA_TRAIN_PATH = DATASET_DIR / "bangla_train.jsonl"
BANGLA_VALIDATION_PATH = DATASET_DIR / "bangla_validation.jsonl"
BANGLA_TEST_PATH = DATASET_DIR / "bangla_test.jsonl"

TRAIN_RATIO = 0.8
VALIDATION_RATIO = 0.1
TEST_RATIO = 0.1
SPLIT_SEED = 42

# ---------------------------------------------------------------------------
# LoRA fine-tuning settings (train_lora.py / merge_and_convert.py)
# ---------------------------------------------------------------------------
LORA_BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

LORA_ADAPTER_DIR = BASE_DIR / "models" / "tinyllama_lora_adapter"
LORA_MERGED_MODEL_DIR = BASE_DIR / "models" / "tinyllama_lora_merged"

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-4
EPOCHS = 3
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
MAX_SEQ_LENGTH = 512

LORA_MODEL_NAME = "tinyllama_lora"
LORA_OLLAMA_TAG = "tinyllama_lora:4bit"

# ---------------------------------------------------------------------------
# Models & quantization (verified real Ollama tags)
# ---------------------------------------------------------------------------
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
    # Added after LoRA fine-tuning + merge_and_convert.py + `ollama create`.
    # Only 4bit exists for this one -- you built a single quantized GGUF from
    # the merged model, not separate fp16/8bit/4bit versions of it.
    "tinyllama_lora": {
        "4bit": "tinyllama_lora:4bit",
    },
}
# If a tag isn't pulled/created on your machine, benchmark.py records
# status="unavailable" for that combination instead of crashing.

# ---------------------------------------------------------------------------
# Generation settings
# ---------------------------------------------------------------------------
GENERATION_OPTIONS = {
    "temperature": 0.2,
    "num_predict": 256,
    "top_p": 0.9,
}

REPETITIONS = 1

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
