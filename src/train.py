import os
import torch
import gc
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset

# Set memory limits and enable garbage collection
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
torch.cuda.empty_cache()
gc.collect()

# Class for measuring GPU memory usage
class MemoryTracker:
    def __init__(self, device=0, log_interval=5):
        self.device = device
        self.log_interval = log_interval
        self.step_counter = 0
        
    def log_memory(self, step_name=""):
        if self.step_counter % self.log_interval == 0:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
                max_allocated = torch.cuda.max_memory_allocated(self.device) / (1024**3)
                reserved = torch.cuda.memory_reserved(self.device) / (1024**3)
                print(f"[{step_name}] GPU Memory: Current={allocated:.2f}GB, Peak={max_allocated:.2f}GB, Reserved={reserved:.2f}GB")
        self.step_counter += 1

# Initialize memory tracker
memory_tracker = MemoryTracker()

# ----- 1. Model and Tokenizer -----
model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

memory_tracker.log_memory("After tokenizer")

# Load base model with memory optimizations
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="auto",
    load_in_8bit=True,  # 8-bit quantization
    max_memory={0: "6GB", "cpu": "24GB"},  # Limit GPU memory usage to 6GB
)

memory_tracker.log_memory("After model loading")

# Prepare the model for k-bit training
model = prepare_model_for_kbit_training(model)

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.config.use_cache = False  # Disable KV cache

memory_tracker.log_memory("After gradient checkpointing setup")

# Configure LoRA adapters
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
)

# Add LoRA adapters to the model
model = get_peft_model(model, peft_config)

# Show trainable parameters info
model.print_trainable_parameters()

memory_tracker.log_memory("After LoRA setup")

# ----- 2. Load the Dataset -----
dataset = load_dataset("json", data_files={"train": "data/train.jsonl"}, split="train")
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

print("Train dataset columns:", train_dataset.column_names)
memory_tracker.log_memory("After dataset loading")

# ----- 3. Preprocessing and Tokenization -----
def preprocess_function(examples):
    texts = [
        f"Context: {c}\nQuestion: {q}\nAnswer: {a}"
        for c, q, a in zip(examples["context"], examples["question"], examples["answer"])
    ]
    # Using a smaller max_length reduces memory usage
    tokenized_inputs = tokenizer(texts, truncation=True, max_length=128, padding="max_length")
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs

# Process in smaller batches to reduce memory spikes
train_dataset = train_dataset.map(
    preprocess_function, 
    batched=True, 
    batch_size=4,  # Process in smaller batches
    remove_columns=train_dataset.column_names
)

eval_dataset = eval_dataset.map(
    preprocess_function, 
    batched=True,
    batch_size=4,
    remove_columns=eval_dataset.column_names
)

memory_tracker.log_memory("After preprocessing")

# ----- 4. Data Collator -----
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ----- 5. Training Arguments -----
training_args = TrainingArguments(
    output_dir="./lora_fine_tuned_model",
    overwrite_output_dir=True,  # Overwrite the content of the output directory
    evaluation_strategy="steps",
    eval_steps=20,
    logging_dir="./logs",
    logging_steps=10,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    max_grad_norm=0.3,
    num_train_epochs=1,  # Reduce for testing
    learning_rate=2e-4,
    weight_decay=0.01,
    fp16=True,  # Mixed precision training
    save_steps=20,
    save_total_limit=2,  # Only keep the 2 most recent checkpoints
    optim="paged_adamw_8bit",
    report_to="none",
    push_to_hub=False
)

memory_tracker.log_memory("After training args setup")

# Custom callback to monitor memory - FIXED VERSION
class MemoryMonitorCallback(TrainerCallback):
    def __init__(self, memory_tracker):
        self.memory_tracker = memory_tracker
        
    def on_train_begin(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        self.memory_tracker.log_memory("Training begin")
        return control
        
    def on_step_begin(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        self.memory_tracker.log_memory("Training step")
        return control
        
    def on_evaluate(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        self.memory_tracker.log_memory("Evaluation")
        return control
        
    def on_train_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        self.memory_tracker.log_memory("Training end")
        return control

# ----- 6. Initialize Trainer -----
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# Add memory monitor callback
trainer.add_callback(MemoryMonitorCallback(memory_tracker))

memory_tracker.log_memory("Before training")

# ----- 7. Start Fine Tuning -----
try:
    print("Starting training...")
    trainer.train()
    print("Training completed successfully!")
except Exception as e:
    print(f"Error during training: {e}")
    # Free memory even if training fails
    del model
    torch.cuda.empty_cache()
    gc.collect()
    raise

# ----- 8. Save the Fine-Tuned Model -----
trainer.save_model("./lora_fine_tuned_model")
tokenizer.save_pretrained("./lora_fine_tuned_model")

# Clean up
del model
torch.cuda.empty_cache()
gc.collect()

print("Fine tuning complete. The model is saved in './lora_fine_tuned_model'")