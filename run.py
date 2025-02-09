import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import time

# System prompt to instruct the LLM.
SYSTEM_PROMPT = "You are a helpful assistant."

def main():
    # Set up logging to include timestamps.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Model identifier on Hugging Face Hub.
    model_name = "issai/LLama-3.1-KazLLM-1.0-8B"

    # Determine the device: use Apple Silicon's MPS if available; otherwise, fallback to CPU.
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Load the tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Ensure that the pad token is set; if not, set it to the eos token.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Choose the data type: use half precision if on MPS to save memory.
    torch_dtype = torch.float16 if device == "mps" else torch.float32

    # Load the model with automatic device mapping.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch_dtype
    )
    # Do not call model.to(device) as device_map already places modules appropriately.

    # Optionally compile the model for improved performance (requires PyTorch 2.0+).
    # Skip compiling if using MPS, as torch.compile with the 'inductor' backend does not support MPS.
    if hasattr(torch, "compile"):
        if device != "mps":
            logging.info("Compiling the model for optimized performance...")
            model = torch.compile(model)
        else:
            logging.info("Skipping torch.compile on MPS device as it's not supported.")

    print("Model and tokenizer loaded successfully.\n")

    # Interactive loop for generating text.
    while True:
        user_input = input("User:\n")
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Exited.")
            break

        # Combine the system prompt with the user input.
        full_prompt = f"{SYSTEM_PROMPT}\nUser: {user_input}\nAssistant:"
        logging.info("Starting generation...")

        # Tokenize the prompt and ensure the attention mask is returned.
        inputs = tokenizer(full_prompt, return_tensors="pt", padding=True).to(device)

        # Start timing the generation.
        start_time = time.time()
        with torch.no_grad():
            generated_ids = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        end_time = time.time()
        elapsed_time = end_time - start_time

        logging.info(f"Generation completed in {elapsed_time:.2f} seconds.")

        # Decode the generated sequence.
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print("\nGenerated Text:\n", generated_text, "\n")

main()