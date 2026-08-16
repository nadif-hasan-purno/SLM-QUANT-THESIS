"""
merge_and_convert.py
Bridges the gap between train_lora.py (produces a PEFT adapter) and
benchmark.py (only talks to Ollama, which needs a GGUF model).

What this script does automatically:
    1. Loads the base TinyLlama model + your trained LoRA adapter.
    2. Merges them into a single standalone model (no more adapter, just
       one set of weights) and saves it in Hugging Face format.

What you do manually after that (one-time, needs llama.cpp):
    3. Convert the merged HF model to GGUF using llama.cpp's converter.
    4. Quantize the GGUF to 4-bit (matching your other models' quant level).
    5. `ollama create` a new model tag from it.
    This script prints the exact commands for steps 3-5 -- they can't be
    run from here because they need llama.cpp cloned locally.

Requires:
    pip install torch transformers peft

Run:
    python merge_and_convert.py
"""

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import LORA_BASE_MODEL, LORA_ADAPTER_DIR, LORA_MERGED_MODEL_DIR, LORA_MODEL_NAME


def main():
    print(f"Loading base model: {LORA_BASE_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(LORA_BASE_MODEL, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(LORA_BASE_MODEL)

    print(f"Loading LoRA adapter from: {LORA_ADAPTER_DIR}")
    model = PeftModel.from_pretrained(base_model, str(LORA_ADAPTER_DIR))

    print("Merging adapter into base weights...")
    merged_model = model.merge_and_unload()

    LORA_MERGED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(LORA_MERGED_MODEL_DIR))
    tokenizer.save_pretrained(str(LORA_MERGED_MODEL_DIR))
    print(f"\nSaved merged model -> {LORA_MERGED_MODEL_DIR}")

    modelfile_path = LORA_MERGED_MODEL_DIR.parent / "Modelfile.tinyllama_lora"
    with open(modelfile_path, "w") as f:
        f.write(f"FROM ./{LORA_MODEL_NAME}.gguf\n")
        f.write('TEMPLATE """{{ .Prompt }}"""\n')
    print(f"Wrote a starter Modelfile -> {modelfile_path}")

    print("\n" + "=" * 70)
    print("REMAINING MANUAL STEPS (one-time, needs llama.cpp cloned locally)")
    print("=" * 70)
    print(f"""
# 1. Clone llama.cpp if you haven't already:
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
pip install -r requirements.txt

# 2. Convert the merged HF model to GGUF (fp16 first):
python convert_hf_to_gguf.py {LORA_MERGED_MODEL_DIR} \\
    --outfile {LORA_MODEL_NAME}.fp16.gguf --outtype f16

# 3. Quantize to 4-bit (to match your other models' 4-bit comparisons):
./llama-quantize {LORA_MODEL_NAME}.fp16.gguf {LORA_MODEL_NAME}.gguf Q4_0

# 4. Move the quantized file next to the Modelfile this script generated:
mv {LORA_MODEL_NAME}.gguf {LORA_MERGED_MODEL_DIR.parent}/

# 5. Create the Ollama model:
cd {LORA_MERGED_MODEL_DIR.parent}
ollama create {LORA_MODEL_NAME}:4bit -f Modelfile.tinyllama_lora

# 6. Verify it works:
ollama run {LORA_MODEL_NAME}:4bit "বাংলাদেশের রাজধানী কী?"

# 7. Add it to config.py -> MODELS so benchmark.py picks it up, e.g.:
MODELS["tinyllama_lora"] = {{"4bit": "{LORA_MODEL_NAME}:4bit"}}
""")


if __name__ == "__main__":
    main()
