import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel, PeftConfig

# Memory management
torch.cuda.empty_cache()

# Path to your fine-tuned model
model_path = "./lora_fine_tuned_model"
base_model_name = "meta-llama/Llama-3.2-1B-Instruct"

# Load the tokenizer from your saved model
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("Loading model with adapters...")
# First load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    trust_remote_code=True,
    device_map="auto",
    load_in_8bit=True,  # Use 8-bit to save memory
    max_memory={0: "6GB", "cpu": "24GB"}
)

# Then load the PEFT adapter
model = PeftModel.from_pretrained(base_model, model_path)

# Optional: merge adapter weights with base model for better performance
# model = model.merge_and_unload()  # Uncomment if you want to merge weights

# Put model in evaluation mode
model.eval()

# Enable a more efficient decoding strategy
model.config.use_cache = True

# Set generation parameters
generation_config = {
    "max_new_tokens": 100,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.2,
    "do_sample": True
}

# Create pipeline for easier generation
qa_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    **generation_config
)

# Test samples with contexts
test_samples = [
    {
        "context": "number of short stories in the Irish language. He spent most of his time in travelling and lived comfortably and quietly outside the spotlight. Glossary ledge (n)- a narrow shelf that juts out from a vertical surface shrilly (adv.)- producing a high-pitched and piercing voice or sound herring (n)- a long silver fish that swims in large groups in the sea devour (v)- to eat something eagerly and in large amounts, so that nothing is left cackle (n)- a sharp, broken noise or cry of a hen, goose or seagull mackerel (n)- a sea fish with a strong taste, often used as food gnaw (v)- to bite or chew something repeatedly trot (v)- to run at a moderate pace with short steps precipice (n)- a very steep side of a cliff or a mountain preening (v)- cleaning feathers with beak whet (v)- to sharpen plaintively (adv.)- sadly, calling in a sad way swoop (v)- to move very quickly and easily through the air beckoning (v)- making a gesture with the hand or head to encourage someone to approach or follow.",
        "question": "What happened to the young seagull when it landed on the green sea?"
    },
    {
        "context": "mother had come around calling to him shrilly, scolding him, threatening to let him starve on his ledge, unless he flew away. But for the life of him, he could not move. That was twenty-four hours ago. Since then, nobody had come near him. The day before, all day long, he had watched his parents flying about with his brothers and sister, perfecting them in the art of flight, teaching them how to skim the waves and how to dive for fish. He had, in fact, seen his older brother catch his first herring and devour it, standing on a rock, while his parents circled around raising a proud cackle. And all the morning, the whole family had walked about on the big plateau midway down the opposite cliff, laughing at his cowardice. The sun was now ascending the sky, blazing warmly on his ledge that faced the south. He felt the heat because he had not eaten since the previous nightfall. Then, he had found a dried piece of mackerel's tail at the far end of his ledge. Now, there was not a single scrap of food left. He had searched every inch, rooting among the rough, dirt-caked straw",
        "question": "How did the bird try to reach its parents without having to fly?"
    }
]

# Test the model with the samples
print("\n===== TESTING THE FINE-TUNED MODEL =====")
for i, sample in enumerate(test_samples):
    print(f"\nTest {i+1}:")
    print(f"Context: {sample['context'][:100]}...")  # Show just the beginning
    print(f"Question: {sample['question']}")
    
    # Format input like during training
    input_text = f"Context: {sample['context']}\nQuestion: {sample['question']}\nAnswer:"
    
    # Generate answer
    result = qa_pipeline(input_text, max_length=len(tokenizer.encode(input_text)) + 100)
    
    # Extract the answer part
    generated_text = result[0]['generated_text']
    answer = generated_text.split("Answer:")[1].strip()
    
    print(f"Generated Answer: {answer}")

# Interactive mode
def interactive_testing():
    print("\n===== INTERACTIVE TESTING MODE =====")
    print("Enter context, question, or type 'exit' to quit.")
    
    while True:
        context = input("\nContext (or 'exit' to quit): ")
        if context.lower() == 'exit':
            break
            
        question = input("Question: ")
        
        # Format input like during training
        input_text = f"Context: {context}\nQuestion: {question}\nAnswer:"
        
        # Generate answer
        result = qa_pipeline(input_text, max_length=len(tokenizer.encode(input_text)) + 100)
        
        # Extract the answer part
        generated_text = result[0]['generated_text']
        answer = generated_text.split("Answer:")[1].strip()
        
        print(f"Generated Answer: {answer}")

# Run interactive testing
if __name__ == "__main__":
    interactive_testing()