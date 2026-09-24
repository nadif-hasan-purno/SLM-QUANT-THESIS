import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Starting...", flush=True)
model_path = "models/tinyllama_lora_merged" 

print("Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

# Loading natively as FP16 (no dtype forcing)
print("Loading model on CPU...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map="cpu"
)
print("Model loaded successfully!", flush=True)

prompt = """### Instruction:
বাংলাদেশের রাজধানী কী?

### Input:

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

print("Generating... (please wait up to 30 seconds)", flush=True)
outputs = model.generate(
    **inputs, 
    max_new_tokens=30, 
    pad_token_id=tokenizer.eos_token_id,
    do_sample=False
)

print("\n--- FINAL OUTPUT ---")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
