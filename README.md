🚀 LLM Fine-Tuning with LoRA

Fine-tune a real Large Language Model (LLM) on a custom dataset using LoRA (Low-Rank Adaptation) — fully runnable on a normal laptop without requiring expensive GPUs.

📌 Project Overview

This project demonstrates how to fine-tune SmolLM2-135M-Instruct, a 135M parameter language model, using LoRA (Parameter-Efficient Fine-Tuning).

Instead of updating all 135M parameters, LoRA trains only small adapter layers, making the process:

⚡ Faster
💾 Memory efficient
💻 Laptop-friendly
✨ Key Features
✅ No GPU required
✅ Runs on CPU (normal laptop)
✅ Fast fine-tuning with LoRA
✅ Lightweight and efficient
✅ Easy to extend with more data
✅ Beginner-friendly ML pipeline
🛠️ Technologies Used
Tool	Purpose
🐍 Python 3.9	Programming Language
🔥 PyTorch	Deep Learning Framework
🤗 Hugging Face Transformers	Load & run LLMs
⚡ PEFT	LoRA fine-tuning
📊 Datasets	Dataset handling
💻 VS Code	Development environment
📁 Project Structure
llm-finetuning-project/
│
├── data/
│   └── train.jsonl              # ML Q&A dataset
│
├── scripts/
│   ├── train.py                 # LoRA training script
│   └── chat.py                  # Chat with fine-tuned model
│
├── outputs/                     # Saved LoRA adapters
├── requirements.txt
└── README.md
📊 Training Results

Training loss recorded during fine-tuning:

Step	Loss
1	2.568
2	2.701
3	2.388
Final	2.085