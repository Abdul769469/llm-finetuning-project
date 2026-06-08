🚀 LLM Fine-Tuning with LoRA

Fine-tune a real Large Language Model (LLM) on a custom dataset using LoRA (Low-Rank Adaptation) — all on a normal laptop without requiring expensive GPUs.

📌 Project Overview

This project demonstrates how to fine-tune SmolLM2-135M-Instruct, a 135-million-parameter language model, using LoRA, a parameter-efficient fine-tuning technique.

Instead of updating all 135 million parameters, LoRA trains only a small set of adapter layers, reducing memory usage and training time dramatically.

✨ Key Benefits

✅ No GPU required
✅ Runs on a standard laptop
✅ Fast training times
✅ Minimal memory consumption
✅ Easy to understand and extend

🛠️ Technologies Used
Tool	Purpose
🐍 Python 3.9	Programming Language
🔥 PyTorch	Deep Learning Framework
🤗 Hugging Face Transformers	Loading & Running LLMs
⚡ PEFT	LoRA Fine-Tuning
📊 Datasets	Dataset Management
💻 VS Code	Development Environment
📂 Project Structure
llm-finetuning-project/
│
├── data/
│   └── train.jsonl          # Machine Learning Q&A dataset
│
├── scripts/
│   ├── train.py             # LoRA training script
│   └── chat.py              # Chat with fine-tuned model
│
├── outputs/                 # Saved LoRA adapters
│
└── requirements.txt         # Project dependencies
⚙️ Installation & Usage
1️⃣ Clone the Repository
git clone https://github.com/Abdul769469/llm-finetuning-project.git

cd llm-finetuning-project
2️⃣ Create a Virtual Environment
python -m venv venv
Windows
venv\Scripts\activate
Linux / Mac
source venv/bin/activate
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Train the Model
python scripts/train.py
5️⃣ Chat with the Fine-Tuned Model
python scripts/chat.py
🧠 Understanding LoRA
Traditional Fine-Tuning
Model Parameters (135M)
        ↓
Update ALL Parameters ❌
        ↓
Requires GPU & High Memory
LoRA Fine-Tuning
Base Model (135M)  → Frozen ❄️
        +
LoRA Adapters       → Trainable ✅
Why LoRA?
Feature	Traditional Fine-Tuning	LoRA
Train All Parameters	✅	❌
Memory Efficient	❌	✅
GPU Required	Usually	Often No
Faster Training	❌	✅
Easy Deployment	❌	✅

🎯 Only around 0.5% of model parameters are trained while maintaining strong performance.

📊 Training Results
Epoch	Training Loss
1	2.758
2	2.451
3	2.280
📈 Progress
Epoch 1  ████████████████  2.758
Epoch 2  ██████████████    2.451
Epoch 3  ████████████      2.280

✅ Loss decreases over time, indicating that the model is learning effectively.

💬 Example Conversation
User
What is overfitting?
Model
Overfitting occurs when a machine learning model memorizes training data too closely and performs poorly on unseen data.
User
What is gradient descent?
Model
Gradient descent is an optimization algorithm that updates model parameters step by step to minimize prediction errors.
🎓 Learning Outcomes

Through this project, I gained hands-on experience with:

✅ LoRA (Low-Rank Adaptation)
✅ Parameter-Efficient Fine-Tuning (PEFT)
✅ Hugging Face Transformers
✅ Preparing datasets for LLM training
✅ Running language model training on CPU
✅ Model inference and evaluation
✅ Git & GitHub project management
