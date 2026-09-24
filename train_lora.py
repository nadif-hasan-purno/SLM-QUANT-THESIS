"""
train_lora.py
Fine-tunes TinyLlama on the Bangla instruction dataset using LoRA.

Fixes applied (cumulative, vs. the original version):
  1. Loss is only computed on the RESPONSE tokens (+ the EOS token that ends
     it) -- the instruction/input portion is masked out with -100, so the
     model is never trained to predict the prompt itself. The original
     version trained on the full sequence including the (highly repeated)
     instruction text, which is the most likely cause of the infinite-loop
     repetition seen in early testing.
  2. Padding is fully masked out too (-100), via a custom data collator that
     pads dynamically per-batch instead of a fixed max_length with
     real-looking pad-token labels.
  3. Uses TinyLlama-Chat's OWN official Zephyr-style chat template
     (<|system|>/<|user|>/<|assistant|>, </s> turn markers) instead of a
     generic Alpaca format -- the base model already understands this
     structure, so the LoRA adapter only has to adapt WHAT it says, not
     learn an entirely new format from 731 examples.
  4. Removed `use_mps_device` from TrainingArguments -- newer `transformers`
     versions auto-detect Apple Silicon's MPS backend and no longer accept
     this argument.

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
    Trainer, TrainingArguments,
)

from config import (
    LORA_BASE_MODEL, LORA_ADAPTER_DIR, BANGLA_TRAIN_PATH, BANGLA_VALIDATION_PATH,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LEARNING_RATE, EPOCHS, BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS, MAX_SEQ_LENGTH, TRAINING_HISTORY_PATH,
)

# NOTE: this must exactly match the TEMPLATE in your Ollama Modelfile
# (Modelfile.tinyllama_lora) and merge_and_convert.py's MODELFILE_CONTENT,
# or the adapted model will be prompted differently at inference time than
# it was trained on. This is TinyLlama-Chat's own official template.
SYSTEM_MESSAGE = "You are a helpful assistant that answers clearly in the language of the question."
PROMPT_TEMPLATE = (
    "<|system|>\n" + SYSTEM_MESSAGE + "</s>\n"
    "<|user|>\n{instruction}\n{input}</s>\n"
    "<|assistant|>\n"
)


def get_device():
    if torch.backends.mps.is_available():
        return "mps"   # Apple Silicon GPU
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_and_tokenize(tokenizer):
    """Returns a function that, given one raw {instruction, input, output}
    example, produces input_ids/labels with the prompt portion masked out
    of the loss and an explicit EOS token ending the response.

    Reserves a minimum token budget for the response: a few examples have
    very long instruction+input text (one is nearly 4,000 characters), and
    naive right-truncation of the full [prompt][response] sequence cuts the
    RESPONSE first since it's appended at the end -- for a long enough
    prompt, truncation can remove the entire response, leaving every label
    masked to -100 for that example. With eval batch_size=1, a single such
    example produces a loss over zero valid tokens (0/0 = nan), which then
    poisons the whole eval_loss average. Reserving MIN_RESPONSE_TOKENS and
    truncating the PROMPT instead (keeping its END, so the "<|assistant|>"
    cue right before generation is never lost) guarantees this can't happen.
    """
    MIN_RESPONSE_TOKENS = 32
    max_prompt_tokens = MAX_SEQ_LENGTH - MIN_RESPONSE_TOKENS

    def _process(example):
        prompt_part = PROMPT_TEMPLATE.format(
            instruction=example["instruction"], input=example["input"]
        )
        prompt_ids = tokenizer(prompt_part, add_special_tokens=False)["input_ids"]
        if len(prompt_ids) > max_prompt_tokens:
            # Keep the END of the prompt (preserves "<|assistant|>\n" right
            # before generation) rather than the default right-truncation,
            # which would drop the crucial suffix instead.
            prompt_ids = prompt_ids[-max_prompt_tokens:]

        remaining_budget = MAX_SEQ_LENGTH - len(prompt_ids)
        response_text = example["output"] + tokenizer.eos_token
        response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]
        response_ids = response_ids[:remaining_budget]  # only bites for unusually long responses

        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids  # -100 = ignored by cross-entropy loss

        return {"input_ids": input_ids, "labels": labels}

    return _process


def make_collate_fn(tokenizer):
    """Dynamically pads each batch to its own longest example (not a fixed
    512 for every batch -- faster) and masks the padding out of the loss
    with -100, so padding never gets trained as a real prediction target."""

    pad_id = tokenizer.pad_token_id

    def collate(features):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []

        for f in features:
            ids = f["input_ids"]
            lbl = f["labels"]
            pad_len = max_len - len(ids)

            input_ids.append(ids + [pad_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
            labels.append(lbl + [-100] * pad_len)  # padding excluded from loss

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


def main():
    device = get_device()
    print(f"Using device: {device}")

    print(f"\nLoading base model: {LORA_BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(LORA_BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Safe even though pad_token == eos_token, because the collator above
    # explicitly masks every padded position out of the loss.

    model = AutoModelForCausalLM.from_pretrained(
        LORA_BASE_MODEL,
        dtype=torch.float16 if device != "cpu" else torch.float32,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)

    print("\nApplying LoRA config:")
    print(f"  r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
    # Widened from just q_proj/v_proj to all attention + MLP linear layers.
    # q_proj/v_proj alone gave only 0.1% trainable params -- likely too
    # little capacity for a 1.1B model to reliably learn 16 diverse
    # instruction categories from 731 examples. This covers all 7 linear
    # layers per transformer block instead of 2, for ~5.6x more capacity.
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("\nLoading Bangla train/validation data")
    dataset = load_dataset(
        "json",
        data_files={"train": str(BANGLA_TRAIN_PATH), "validation": str(BANGLA_VALIDATION_PATH)},
    )
    process_fn = build_and_tokenize(tokenizer)
    dataset = dataset.map(process_fn, remove_columns=dataset["train"].column_names)

    # Safety net: drop any example where every label is still -100 (should
    # be effectively impossible after the fix above, but this guarantees
    # the nan eval_loss bug can never silently come back).
    def has_valid_target(example):
        return any(l != -100 for l in example["labels"])
    before_train, before_val = len(dataset["train"]), len(dataset["validation"])
    dataset = dataset.filter(has_valid_target)
    dropped_train = before_train - len(dataset["train"])
    dropped_val = before_val - len(dataset["validation"])
    if dropped_train or dropped_val:
        print(f"WARNING: dropped {dropped_train} train / {dropped_val} validation examples "
              f"with no valid response tokens after truncation.")

    print(f"Train examples: {len(dataset['train'])}, Validation examples: {len(dataset['validation'])}")

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
        report_to=[],
        fp16=(device == "cuda"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=make_collate_fn(tokenizer),
    )

    print("\n--- Starting training ---")
    start = time.time()
    trainer.train()
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
          "base model, then re-convert to GGUF and re-create the Ollama model.")


if __name__ == "__main__":
    main()
