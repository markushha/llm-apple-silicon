import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

SYSTEM_PROMPT = "You are a helpful assistant. Answer user queries."

def nucleus_sampling(logits, top_p=0.9, temperature=1.0):
    logits = logits / temperature

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    
    probabilities = torch.softmax(sorted_logits, dim=-1)
    
    cumulative_probs = torch.cumsum(probabilities, dim=-1)
    
    sorted_indices_to_remove = cumulative_probs > top_p
    
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    sorted_logits[sorted_indices_to_remove] = -float('Inf')
    
    filtered_probs = torch.softmax(sorted_logits, dim=-1)
    
    next_token = torch.multinomial(filtered_probs, num_samples=1)

    next_token_id = sorted_indices[next_token]
    return next_token_id

def generate_token_by_token(model, tokenizer, input_ids, max_new_tokens=50, temperature=0.7, top_p=0.9):
    generated_ids = input_ids.clone()
    
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
    logits = outputs.logits
    past_key_values = outputs.past_key_values

    for i in range(max_new_tokens):
        next_token_logits = logits[0, -1, :]

        next_token_id = nucleus_sampling(next_token_logits, top_p=top_p, temperature=temperature)
        
        token_str = tokenizer.decode(next_token_id).strip()
        logging.info(f"Step {i+1}: Generated token id {next_token_id.item()} | Token: '{token_str}'")
        
        next_token_id = next_token_id.unsqueeze(0)
        generated_ids = torch.cat((generated_ids, next_token_id), dim=1)
        
        if next_token_id.item() == tokenizer.eos_token_id:
            logging.info("EOS token generated, stopping generation.")
            break

        with torch.no_grad():
            outputs = model(next_token_id, use_cache=True, past_key_values=past_key_values)
        logits = outputs.logits
        past_key_values = outputs.past_key_values

    return generated_ids

def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    model_name = "issai/LLama-3.1-KazLLM-1.0-8B"

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    torch_dtype = torch.float16 if device == "mps" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch_dtype
    )

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

        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        logging.info("Token-by-token generation completed.")
        print("\nGenerated Text:\n", generated_text, "\n")

main()