from transformers import AutoTokenizer, AutoModelForCausalLM

# Use Meta's official Llama-2-7B-Chat model.
# Ensure you have been granted access at https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
model_name = "meta-llama/Llama-2-7b-chat-hf"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",    # Automatically assigns layers to available devices.
        torch_dtype="auto"    # Adjusts dtype for performance (e.g., FP16)
    )
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading model:", e)
