"""
train.py — Fine-tune SmolLM2-135M with LoRA (CPU-friendly)
============================================================
What this script does, step by step:
  1. Loads your JSONL training data from data/train.jsonl
  2. Downloads SmolLM2-135M from HuggingFace (~270 MB, first run only)
  3. Applies LoRA — adds tiny trainable adapters without changing the base model
  4. Trains for a few epochs on your data
  5. Saves only the LoRA adapter weights to outputs/ (small, not the full model)

Run:
    python scripts/train.py
"""

import json
from pathlib import Path

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

# ─────────────────────────────────────────────────────────────────
# Config — tweak these to experiment
# ─────────────────────────────────────────────────────────────────
MODEL_NAME   = "HuggingFaceTB/SmolLM2-135M-Instruct"  # ~270 MB download
DATA_PATH    = "data/train.jsonl"
OUTPUT_DIR   = "outputs/lora-adapter"

MAX_LENGTH   = 256   # max tokens per training example (lower = less RAM)
EPOCHS       = 3     # more epochs = more learning, but risks overfitting
BATCH_SIZE   = 2     # keep at 2 for 16 GB RAM
LR           = 2e-4  # learning rate; 1e-4 to 3e-4 works well for LoRA

# LoRA-specific settings
LORA_RANK    = 8     # higher rank = more capacity to learn (try 4, 8, or 16)
LORA_ALPHA   = 32    # scaling factor; usually set to 4 × rank
LORA_DROPOUT = 0.1   # regularisation; reduces overfitting

# ─────────────────────────────────────────────────────────────────
# 1. Load dataset
# ─────────────────────────────────────────────────────────────────
print("=" * 55)
print("Step 1/5 — Loading dataset")
print("=" * 55)

data_file = Path(DATA_PATH)
if not data_file.exists():
    raise FileNotFoundError(
        f"Training data not found at {DATA_PATH}.\n"
        "Make sure data/train.jsonl exists with 'input' and 'output' fields."
    )

records = []
with open(data_file, encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        assert "input" in record and "output" in record, (
            f"Line {i} must have 'input' and 'output' keys"
        )
        records.append(record)

print(f"Loaded {len(records)} training examples.\n")

# ─────────────────────────────────────────────────────────────────
# 2. Load tokenizer and format examples
# ─────────────────────────────────────────────────────────────────
print("=" * 55)
print("Step 2/5 — Loading tokenizer and formatting data")
print("=" * 55)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # SmolLM2 has no pad token by default


def format_example(record: dict) -> str:
    """
    Format one training example into a prompt-response string.
    The model learns to produce the Answer given the Question.
    """
    return f"Question: {record['input']}\nAnswer: {record['output']}"


texts = [format_example(r) for r in records]
dataset = Dataset.from_dict({"text": texts})

# Preview one example so you can see what the model trains on
print("Example training text:")
print("-" * 40)
print(texts[0])
print("-" * 40 + "\n")


def tokenize(batch):
    """Convert text to token IDs the model can process."""
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )


tokenized_dataset = dataset.map(
    tokenize,
    batched=True,
    remove_columns=["text"],
    desc="Tokenising",
)

# ─────────────────────────────────────────────────────────────────
# 3. Load base model
# ─────────────────────────────────────────────────────────────────
print("=" * 55)
print("Step 3/5 — Loading base model (first run = ~270 MB download)")
print("=" * 55)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
total_params = sum(p.numel() for p in model.parameters())
print(f"Base model loaded. Total parameters: {total_params:,}\n")

# ─────────────────────────────────────────────────────────────────
# 4. Apply LoRA
# ─────────────────────────────────────────────────────────────────
print("=" * 55)
print("Step 4/5 — Applying LoRA adapters")
print("=" * 55)
print(
    "LoRA inserts small trainable matrices into the attention layers.\n"
    "The original model weights are FROZEN — only the adapters train.\n"
    "This is why fine-tuning can happen on a laptop in minutes.\n"
)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "v_proj"],  # attention query and value layers
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print()

# ─────────────────────────────────────────────────────────────────
# 5. Train
# ─────────────────────────────────────────────────────────────────
print("=" * 55)
print("Step 5/5 — Training")
print("=" * 55)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LR,
    logging_steps=5,           # print loss every 5 steps
    save_strategy="epoch",     # save a checkpoint after each epoch
    fp16=False,                # CPU doesn't support fp16
    no_cuda=True,              # force CPU usage
    report_to="none",          # disable Weights & Biases logging
    warmup_ratio=0.1,          # gently ramp up learning rate at the start
    weight_decay=0.01,         # mild L2 regularisation
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # we're doing causal LM, not masked LM
    ),
)

trainer.train()

# ─────────────────────────────────────────────────────────────────
# 6. Save
# ─────────────────────────────────────────────────────────────────
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n" + "=" * 55)
print("Training complete!")
print("=" * 55)
print(f"LoRA adapter saved to: {OUTPUT_DIR}")
print("\nNote: Only the small adapter files are saved (~1–10 MB).")
print("The base model is downloaded fresh when you run chat.py.\n")
print("Next step — run the chat script:")
print("    python scripts/chat.py")
