import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# You can change the system prompt here
SYSTEM_PROMPT = "You are a helpful assistant. Answer user queries."

def nucleus_sampling(logits, top_p=0.9, temperature=1.0):
    """
    Applies nucleus (top-p) sampling on the logits.
    Returns a single token id sampled from the filtered distribution.
    """
    # Scale logits by temperature
    logits = logits / temperature

    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    
    # Compute softmax probabilities on the sorted logits
    probabilities = torch.softmax(sorted_logits, dim=-1)
    
    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(probabilities, dim=-1)
    
    # Identify tokens to remove: those where the cumulative probability exceeds top_p.
    sorted_indices_to_remove = cumulative_probs > top_p
    
    # Ensure that at least one token remains by shifting the mask right
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # Set logits for filtered-out tokens to -infinity
    sorted_logits[sorted_indices_to_remove] = -float('Inf')
    
    # Recompute probabilities on the filtered logits
    filtered_probs = torch.softmax(sorted_logits, dim=-1)
    
    # Sample one token from the filtered distribution
    next_token = torch.multinomial(filtered_probs, num_samples=1)
    # Map the sampled index back to the original vocabulary index
    next_token_id = sorted_indices[next_token]
    return next_token_id

def generate_token_by_token(model, tokenizer, input_ids, max_new_tokens=50, temperature=0.7, top_p=0.9):
    """
    Generates text token-by-token using the model.
    Logs each generated token.
    
    Args:
        model: The language model.
        tokenizer: The tokenizer.
        input_ids: Tensor with shape (1, seq_length) for the prompt.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability threshold.
    
    Returns:
        generated_ids: Tensor with shape (1, original_length + new_tokens).
    """
    # Copy the input_ids to start our generated sequence.
    generated_ids = input_ids.clone()
    
    # Run the initial forward pass with caching enabled.
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
    logits = outputs.logits
    past_key_values = outputs.past_key_values

    # Loop to generate new tokens one by one.
    for i in range(max_new_tokens):
        # Get logits for the last token in the current sequence.
        next_token_logits = logits[0, -1, :]

        # Sample the next token id using nucleus sampling.
        next_token_id = nucleus_sampling(next_token_logits, top_p=top_p, temperature=temperature)
        
        # Decode the token for logging.
        token_str = tokenizer.decode(next_token_id).strip()
        logging.info(f"Step {i+1}: Generated token id {next_token_id.item()} | Token: '{token_str}'")
        
        # Append the generated token to the sequence.
        # Unsqueeze once so that next_token_id has shape (1, 1)
        next_token_id = next_token_id.unsqueeze(0)
        generated_ids = torch.cat((generated_ids, next_token_id), dim=1)
        
        # If the EOS token is generated, stop early.
        if next_token_id.item() == tokenizer.eos_token_id:
            logging.info("EOS token generated, stopping generation.")
            break

        # Feed only the new token into the model along with the cached past key values.
        with torch.no_grad():
            outputs = model(next_token_id, use_cache=True, past_key_values=past_key_values)
        logits = outputs.logits
        past_key_values = outputs.past_key_values

    return generated_ids

def main():
    # Set up logging to include timestamps.
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    # Model identifier on Hugging Face Hub.
    model_name = "issai/LLama-3.1-KazLLM-1.0-8B"

    # Determine the device: use Apple Silicon's MPS if available; otherwise, fallback to CPU.
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Load the tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Choose the data type: use half precision if on MPS to save memory.
    torch_dtype = torch.float16 if device == "mps" else torch.float32

    # Load the model with automatic device mapping.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch_dtype
    )
    # Do not call model.to(device) as device_map already places modules appropriately.

    # Optionally compile the model for improved performance (requires PyTorch 2.0+)
    # Skip compiling if running on MPS since torch.compile with the 'inductor' backend doesn't support MPS.
    if hasattr(torch, "compile"):
        if device != "mps":
            logging.info("Compiling the model for optimized performance...")
            model = torch.compile(model)
        else:
            logging.info("Skipping torch.compile on MPS device as it's not supported.")

    print("Model and tokenizer loaded successfully.\n")

    while True:
        user_input = input("User:\n")
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Exited.")
            break

        full_prompt = f"{SYSTEM_PROMPT}\nUser: {user_input}\nAssistant:"
        logging.info("Starting token-by-token generation with system prompt...")
        
        inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            generated_ids = generate_token_by_token(
                model,
                tokenizer,
                inputs["input_ids"],
                max_new_tokens=50,
                temperature=0.7,
                top_p=0.9
            )

        # Decode the full generated sequence.
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        logging.info("Token-by-token generation completed.")
        print("\nGenerated Text:\n", generated_text, "\n")


main()