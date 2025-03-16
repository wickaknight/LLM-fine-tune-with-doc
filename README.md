Fine-Tuning LLMs with Low-Resource Hardware

This project demonstrates how to fine-tune large language models (LLMs) on consumer-grade hardware with limited VRAM. Using Parameter-Efficient Fine-Tuning (PEFT) with Low-Rank Adaptation (LoRA), we can fine-tune powerful models like Llama-3 on GPUs with as little as 8GB VRAM.

📑 Table of Contents
Overview
Installation
Dataset
Model Architecture
Training Process
Evaluation
Usage
Project Structure
Results
Contributing
License
🔍 Overview
This project demonstrates fine-tuning the Llama-3.2-1B-Instruct model for question answering tasks on limited hardware resources. By using:

8-bit quantization
LoRA adapters
Memory-efficient optimizers
Gradient checkpointing
We achieve successful fine-tuning on a consumer GPU with only 8GB VRAM.

🚀 Installation
Requirements
📊 Dataset
The project uses a dataset of context-question-answer triplets for fine-tuning:

The dataset is loaded from train.jsonl and automatically split into training and evaluation sets.

🧠 Model Architecture
We use the Llama-3.2-1B-Instruct model with LoRA adapters:

Base model: Meta's Llama-3.2-1B-Instruct
Adaptation method: Low-Rank Adaptation (LoRA)
LoRA configuration:
Rank (r): 8
Alpha: 32
Target modules: Query, Key, Value, and Output projections in attention layers
Dropout: 0.1
⚙️ Training Process
The training script (src/train.py) implements several optimizations:

Memory tracking to monitor GPU usage during training
8-bit quantization to reduce memory footprint
Gradient checkpointing to save memory during backpropagation
Memory-efficient optimizer (paged_adamw_8bit)
Processing in smaller batches to avoid memory spikes
Proper exception handling for graceful error recovery
Training parameters:

Learning rate: 2e-4
Epochs: 1
Batch size: 1 with gradient accumulation of 8 steps
Max sequence length: 128 tokens
FP16 mixed precision training
📈 Evaluation
Several evaluation scripts are provided to assess model performance:

evaluate_model.py: Calculates ROUGE scores and training loss
src/simple_eval.py: Basic evaluation with word overlap metrics
test_model_inference.py: Interactive testing interface
src/model_insights.py: Comprehensive model analysis

📁 Project Structure
low-resource-llm-finetuning/
├── data/
│   └── train.jsonl          # Training data
├── src/
│   ├── train.py             # Main training script
│   ├── evaluate_model.py    # Evaluation script
│   ├── test_model_inference.py # Interactive testing
│   ├── model_insights.py    # Analysis script
│   ├── simple_eval.py       # Simplified evaluation
│   └── test_train.py        # Unit tests
├── lora_fine_tuned_model/   # Output directory for model
├── logs/                    # Training logs
├── analysis_plots/          # Generated analysis charts
├── requirements.txt         # Dependencies
└── README.md                # This file   

📊 Results
Our fine-tuned model demonstrates improved performance on the question answering task:

Training Loss: Decreased from ~3.1 to ~2.5 during training
Word Overlap: Typically 0.60-0.75 with reference answers
Qualitative Improvements: Better contextual understanding and answer formulation
🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
