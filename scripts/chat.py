"""
chat.py — Interactive chat with your fine-tuned model
======================================================
Loads the base SmolLM2-135M model and your LoRA adapter,
then lets you ask questions interactively.

Run:
    python scripts/chat.py

Make sure you have trained the model first:
    python scripts/train.py
"""

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ─────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────
BASE_MODEL  = "HuggingFaceTB/SmolLM2-135M-Instruct"
ADAPTER_DIR = "outputs/lora-adapter"

# Generation settings — experiment with these!
MAX_NEW_TOKENS    = 300   # max length of the model's reply
TEMPERATURE       = 0.7   # 0 = deterministic, 1 = very creative
TOP_P             = 0.9   # nucleus sampling: only top 90% of probability mass
REPETITION_PENALTY = 1.15 # penalise repeated phrases (1.0 = off, 1.3 = strong)

# ─────────────────────────────────────────────────────────────────
# Check adapter exists
# ─────────────────────────────────────────────────────────────────
if not Path(ADAPTER_DIR).exists():
    print("ERROR: No trained adapter found.")
    print(f"Expected adapter at: {ADAPTER_DIR}")
    print("\nPlease run training first:")
    print("    python scripts/train.py")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────
print("\nLoading tokenizer and model...")
print("(Base model will download on first run — ~270 MB)\n")

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
tokenizer.pad_token = tokenizer.eos_token

# Load base model then attach LoRA adapter on top
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()  # switch to inference mode (disables dropout)

print("=" * 55)
print("Model ready! Ask anything from your training domain.")
print("Type 'quit' or press Ctrl+C to exit.")
print("=" * 55 + "\n")


# ─────────────────────────────────────────────────────────────────
# Chat loop
# ─────────────────────────────────────────────────────────────────
def generate_response(question: str) -> str:
    """
    Format the question as a prompt, generate a response,
    and return only the new tokens (not the prompt itself).
    """
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Slice off the prompt tokens; decode only the new ones
    prompt_length = inputs["input_ids"].shape[1]
    new_token_ids = output_ids[0][prompt_length:]
    response = tokenizer.decode(new_token_ids, skip_special_tokens=True)
    return response.strip()


try:
    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q", "bye"):
            print("Goodbye!")
            break

        print("Bot: ", end="", flush=True)
        response = generate_response(user_input)
        print(response)
        print()

except KeyboardInterrupt:
    print("\n\nExiting. Goodbye!")
