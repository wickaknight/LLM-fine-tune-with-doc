import os
import re
import torch
import gc
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
from datasets import load_dataset

# Initialize argument parser
parser = argparse.ArgumentParser(description="Generate insights for fine-tuned model")
parser.add_argument("--analysis", type=str, default="all", 
                    choices=["training_loss", "parameter_stats", "token_analysis", 
                             "prediction_samples", "all"],
                    help="Type of analysis to run")
parser.add_argument("--samples", type=int, default=3, help="Number of samples for prediction")
args = parser.parse_args()

# Model and data paths
MODEL_PATH = "./lora_fine_tuned_model"
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
LOG_DIR = "./logs"
DATA_PATH = "data/train.jsonl"

# ===== Utility Functions =====
def clean_memory():
    """Clean up CUDA memory"""
    torch.cuda.empty_cache()
    gc.collect()

def create_plots_dir():
    """Create directory for plots if it doesn't exist"""
    os.makedirs("./analysis_plots", exist_ok=True)

# ===== Training Loss Analysis =====
def analyze_training_loss():
    """Analyze training loss from log files"""
    print("\n===== TRAINING LOSS ANALYSIS =====")
    
    # Look for training logs
    log_files = list(Path(LOG_DIR).glob("**/events.out.tfevents.*"))
    
    if not log_files:
        # If TensorBoard logs aren't available, look for any other logs
        print("No TensorBoard logs found. Looking for alternative logs...")
        
        # Try to find training loss in output.txt or similar files
        output_files = list(Path(".").glob("**/output*.txt")) + list(Path(".").glob("**/*.log"))
        
        if output_files:
            for file in output_files:
                print(f"Analyzing {file}...")
                losses = []
                steps = []
                
                with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        # Look for lines containing loss information
                        match = re.search(r'(loss|Loss)[:\s=]+([0-9.]+)', line)
                        step_match = re.search(r'(step|Step)[:\s=]+([0-9]+)', line)
                        
                        if match:
                            try:
                                loss = float(match.group(2))
                                step = int(step_match.group(2)) if step_match else len(losses) + 1
                                losses.append(loss)
                                steps.append(step)
                            except (ValueError, IndexError):
                                pass
                
                if losses:
                    print(f"Found {len(losses)} loss values")
                    plt.figure(figsize=(10, 6))
                    plt.plot(steps, losses)
                    plt.title('Training Loss')
                    plt.xlabel('Step')
                    plt.ylabel('Loss')
                    plt.grid(True)
                    
                    # Save the plot
                    create_plots_dir()
                    plt.savefig('./analysis_plots/training_loss.png')
                    plt.close()
                    
                    print(f"Initial loss: {losses[0]:.4f}")
                    print(f"Final loss: {losses[-1]:.4f}")
                    print(f"Change: {(losses[0] - losses[-1]) / losses[0] * 100:.2f}%")
                    return
        
        print("No training logs found. Manual loss calculation required.")
        print("To track training loss in future runs, add 'logging_dir=\"./logs\"' to your TrainingArguments.")

# ===== Parameter Statistics =====
def analyze_parameters():
    """Analyze model parameters and LoRA adapters"""
    print("\n===== PARAMETER STATISTICS =====")
    
    try:
        # Load adapter config
        adapter_config_path = os.path.join(MODEL_PATH, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            
            print("LoRA Configuration:")
            print(f"  Rank (r): {adapter_config.get('r', 'N/A')}")
            print(f"  Alpha: {adapter_config.get('lora_alpha', 'N/A')}")
            print(f"  Dropout: {adapter_config.get('lora_dropout', 'N/A')}")
            print(f"  Target modules: {adapter_config.get('target_modules', 'N/A')}")
        
        # Load base model and adapter to analyze parameter statistics
        clean_memory()
        print("Loading model to analyze parameters...")
        
        # Load only the adapter without base model for parameter stats
        adapter_files = [f for f in os.listdir(MODEL_PATH) if f.endswith('.bin')]
        
        if adapter_files:
            # Sort to get the most recent checkpoint
            adapter_files.sort()
            adapter_path = os.path.join(MODEL_PATH, adapter_files[-1])
            
            # Load adapter state dict to analyze
            adapter_state = torch.load(adapter_path, map_location='cpu')
            
            # Analyze adapter parameters
            lora_a_params = [k for k in adapter_state.keys() if 'lora_A' in k]
            lora_b_params = [k for k in adapter_state.keys() if 'lora_B' in k]
            
            print(f"\nAdapter file analyzed: {os.path.basename(adapter_path)}")
            print(f"LoRA A matrices: {len(lora_a_params)}")
            print(f"LoRA B matrices: {len(lora_b_params)}")
            
            # Calculate parameter norms to see which adaptations were strongest
            if lora_a_params:
                norms = {}
                for key in lora_a_params[:5]:  # Just check first 5 to avoid memory issues
                    base_key = key.split('lora_A')[0]
                    if base_key + 'lora_B' in adapter_state:
                        # Simplified norm calculation
                        a_norm = torch.norm(adapter_state[key]).item()
                        b_norm = torch.norm(adapter_state[base_key + 'lora_B']).item()
                        norms[base_key] = a_norm * b_norm
                
                # Report top modules by adaptation strength
                print("\nTop modules by adaptation strength:")
                for i, (k, v) in enumerate(sorted(norms.items(), key=lambda x: x[1], reverse=True)[:3]):
                    print(f"  {i+1}. {k}: {v:.4f}")
        
        else:
            print("No adapter weights found for analysis.")
                
    except Exception as e:
        print(f"Error analyzing parameters: {str(e)}")
        import traceback
        traceback.print_exc()

# ===== Token-Level Analysis =====
def analyze_token_loss():
    """Analyze which tokens/words the model struggles with most"""
    print("\n===== TOKEN-LEVEL ANALYSIS =====")
    
    try:
        # Load a few samples from the dataset
        dataset = load_dataset("json", data_files={"train": DATA_PATH}, split="train")
        clean_memory()
        
        # Load tokenizer and model
        print("Loading model for token analysis...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        
        # Load base model with low precision to save memory
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Add adapter
        model = PeftModel.from_pretrained(base_model, MODEL_PATH)
        model.eval()
        
        # Free up memory
        del base_model
        clean_memory()
        
        # Take just first example to avoid memory issues
        example = dataset[0]
        
        # Format input
        input_text = f"Context: {example['context']}\nQuestion: {example['question']}\nAnswer: {example['answer']}"
        
        # Tokenize with labels for loss calculation
        inputs = tokenizer(input_text, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        inputs['labels'] = inputs['input_ids'].clone()
        
        # Get token-level losses
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Calculate loss per token
        loss = outputs.loss
        
        # Get logits and calculate per-token loss
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs['labels'][..., 1:].contiguous()
        
        # Get softmax
        probs = torch.nn.functional.softmax(shift_logits, dim=-1)
        
        # Get probability of correct token
        label_probs = torch.gather(probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
        
        # Convert to log probabilities and negation gives us per-token loss
        per_token_loss = -torch.log(label_probs)
        
        # Detach and move to CPU
        per_token_loss = per_token_loss[0].detach().cpu().numpy()
        
        # Get tokens
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].tolist())
        tokens = tokens[1:]  # Shift tokens to align with per-token loss
        
        # Get highest loss tokens
        token_losses = [(token, loss_val) for token, loss_val in zip(tokens, per_token_loss)]
        highest_loss_tokens = sorted(token_losses, key=lambda x: x[1], reverse=True)[:10]
        
        print(f"\nOverall loss: {loss.item():.4f}")
        print("\nTokens with highest loss:")
        for token, loss_val in highest_loss_tokens:
            print(f"  '{token}': {loss_val:.4f}")
        
        # Clean up
        del model, inputs, outputs, logits, probs
        clean_memory()
        
    except Exception as e:
        print(f"Error in token analysis: {str(e)}")
        import traceback
        traceback.print_exc()

# ===== Prediction Analysis =====
def analyze_predictions(num_samples=3):
    """Analyze model predictions with focus on quality"""
    print("\n===== PREDICTION ANALYSIS =====")
    
    try:
        # Load a few samples from dataset
        dataset = load_dataset("json", data_files={"train": DATA_PATH}, split="train")
        
        # Limit to requested number of samples
        indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        samples = [dataset[int(i)] for i in indices]
        
        clean_memory()
        
        # Load model
        print("Loading model for prediction analysis...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, MODEL_PATH)
        model.eval()
        
        # Free base model memory
        del base_model
        clean_memory()
        
        results = []
        
        # Process each example
        for i, example in enumerate(samples):
            print(f"\nAnalyzing sample {i+1}/{len(samples)}...")
            
            # Clean memory for each example
            clean_memory()
            
            # Format input
            question = example['question']
            context = example['context']
            reference = example['answer']
            
            # Create input
            input_text = f"Context: {context}\nQuestion: {question}\nAnswer:"
            
            # Generate prediction
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            
            # Generate with and without sampling to compare
            with torch.no_grad():
                # Greedy decoding
                greedy_output = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False
                )
                
                # Sampling with temperature
                sample_output = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
            
            # Decode outputs
            greedy_text = tokenizer.decode(greedy_output[0], skip_special_tokens=True)
            sample_text = tokenizer.decode(sample_output[0], skip_special_tokens=True)
            
            # Extract answers
            try:
                greedy_answer = greedy_text.split("Answer:")[1].strip()
            except:
                greedy_answer = greedy_text
                
            try:
                sample_answer = sample_text.split("Answer:")[1].strip()
            except:
                sample_answer = sample_text
            
            # Calculate simple word overlap
            ref_words = set(reference.lower().split())
            greedy_words = set(greedy_answer.lower().split())
            sample_words = set(sample_answer.lower().split())
            
            greedy_overlap = len(ref_words.intersection(greedy_words)) / len(ref_words) if ref_words else 0
            sample_overlap = len(ref_words.intersection(sample_words)) / len(ref_words) if ref_words else 0
            
            # Store results
            results.append({
                'question': question,
                'greedy_answer': greedy_answer,
                'sampled_answer': sample_answer,
                'reference': reference,
                'greedy_overlap': greedy_overlap,
                'sample_overlap': sample_overlap
            })
            
        # Print analysis
        print("\nPrediction Analysis Summary:")
        avg_greedy_overlap = np.mean([r['greedy_overlap'] for r in results])
        avg_sample_overlap = np.mean([r['sample_overlap'] for r in results])
        
        print(f"Average word overlap (greedy): {avg_greedy_overlap:.2f}")
        print(f"Average word overlap (sampling): {avg_sample_overlap:.2f}")
        
        # Print examples
        for i, result in enumerate(results):
            print(f"\nExample {i+1}:")
            print(f"Question: {result['question']}")
            print(f"Reference: {result['reference']}")
            print(f"Greedy prediction: {result['greedy_answer']}")
            print(f"Sampled prediction: {result['sampled_answer']}")
            print(f"Word overlap (greedy): {result['greedy_overlap']:.2f}")
            print(f"Word overlap (sampling): {result['sample_overlap']:.2f}")
        
        # Clean up
        del model
        clean_memory()
        
    except Exception as e:
        print(f"Error in prediction analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if args.analysis == "training_loss" or args.analysis == "all":
        analyze_training_loss()
        
    if args.analysis == "parameter_stats" or args.analysis == "all":
        analyze_parameters()
        
    if args.analysis == "token_analysis" or args.analysis == "all":
        analyze_token_loss()
        
    if args.analysis == "prediction_samples" or args.analysis == "all":
        analyze_predictions(args.samples)
        
    print("\nAnalysis complete!")