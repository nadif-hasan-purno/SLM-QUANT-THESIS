"""
train_lora.py
Fine-tunes TinyLlama on the Bangla instruction dataset using LoRA.

This step uses Hugging Face Transformers + PEFT + PyTorch directly (NOT
Ollama) -- Ollama can run models, but it can't train/fine-tune them.
On Mac mini M4, PyTorch will use the Metal (MPS) backend automatically.

Requires:
    pip install torch transformers peft accelerate datasets

Run:
    python train_lora.py
"""

import time

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling,
)

from config import (
    LORA_BASE_MODEL, LORA_ADAPTER_DIR, BANGLA_TRAIN_PATH, BANGLA_VALIDATION_PATH,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LEARNING_RATE, EPOCHS, BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS, MAX_SEQ_LENGTH, TRAINING_HISTORY_PATH,
)

PROMPT_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)


def get_device():
    if torch.backends.mps.is_available():
        return "mps"   # Apple Silicon GPU
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def format_example(example):
    text = PROMPT_TEMPLATE.format(
        instruction=example["instruction"],
        input=example["input"],
        output=example["output"],
    )
    return {"text": text}


def tokenize_function(tokenizer):
    def _tokenize(example):
        tokens = tokenizer(
            example["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens
    return _tokenize


def main():
    device = get_device()
    print(f"Using device: {device}")

    print(f"\nLoading base model: {LORA_BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(LORA_BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        LORA_BASE_MODEL,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    )
    model.to(device)

    print("\nApplying LoRA config:")
    print(f"  r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "v_proj"],  # standard for Llama-family attention
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("\nLoading Bangla train/validation data")
    dataset = load_dataset(
        "json",
        data_files={"train": str(BANGLA_TRAIN_PATH), "validation": str(BANGLA_VALIDATION_PATH)},
    )
    dataset = dataset.map(format_example)
    dataset = dataset.map(tokenize_function(tokenizer), remove_columns=dataset["train"].column_names)

    print(f"Train examples: {len(dataset['train'])}, Validation examples: {len(dataset['validation'])}")

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(LORA_ADAPTER_DIR / "checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        report_to=[],           # disable wandb/etc
        use_mps_device=(device == "mps"),
        fp16=(device == "cuda"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collator,
    )

    print("\n--- Starting training ---")
    start = time.time()
    train_result = trainer.train()
    elapsed = time.time() - start
    print(f"--- Training finished in {elapsed / 60:.1f} min ---")

    eval_result = trainer.evaluate()
    print(f"Final validation loss: {eval_result.get('eval_loss')}")

    print(f"\nSaving LoRA adapter -> {LORA_ADAPTER_DIR}")
    model.save_pretrained(str(LORA_ADAPTER_DIR))
    tokenizer.save_pretrained(str(LORA_ADAPTER_DIR))

    with open(TRAINING_HISTORY_PATH, "w", encoding="utf-8") as f:
        f.write("metric,value\n")
        f.write(f"train_runtime_minutes,{elapsed / 60:.2f}\n")
        f.write(f"final_eval_loss,{eval_result.get('eval_loss')}\n")
        for log in trainer.state.log_history:
            if "loss" in log:
                f.write(f"train_loss_step_{log.get('step')},{log['loss']}\n")
    print(f"Saved training summary -> {TRAINING_HISTORY_PATH}")

    print("\nNext step: run merge_and_convert.py to merge this adapter into the "
          "base model and prepare it for benchmarking via Ollama.")


if __name__ == "__main__":
    main()
