import torch
import tiktoken
from train_gpt import GPT, DEVICE, BLOCK_SIZE, VOCAB_SIZE, N_EMBD, N_HEAD, N_LAYER

def generate_response(model, prompt_text):
    """Generates a response from the model given a text prompt."""
    enc = tiktoken.get_encoding("gpt2")
    input_ids = torch.tensor(enc.encode(prompt_text), dtype=torch.long, device=DEVICE).unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        for _ in range(150): # Max new tokens
            # Crop context to the last BLOCK_SIZE tokens
            idx_cond = input_ids[:, -BLOCK_SIZE:]
            
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] # Focus only on the last time step
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append sampled token to the running sequence
            input_ids = torch.cat((input_ids, idx_next), dim=1)
            
            # Stop if the end-of-text token is generated
            if idx_next.item() == 50256:
                break
    
    # Decode the generated IDs back to text
    return enc.decode(input_ids[0].tolist())

def main():
    """Main function to load the model and start the interactive chat."""
    print("Loading Firmware-GPT model...")
    # Re-initialize the model with the same architecture
    model = GPT().to(DEVICE)
    try:
        model.load_state_dict(torch.load('model.pth', map_location=DEVICE))
    except FileNotFoundError:
        print("ERROR: model.pth not found! Please train the model first using train_gpt.py")
        return
        
    print("\nFirmware AI Assistant is ready!")
    print("Type your question and press Enter. Type 'quit' to exit.")

    while True:
        prompt = input("\nUser: ")
        if prompt.lower() == 'quit':
            break
        
        # Format the prompt for the SFT-trained model
        formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
        
        full_response = generate_response(model, formatted_prompt)
        
        # Clean up the output to only show the new text
        response_only = full_response[len(formatted_prompt):]
        
        print(f"\nAI: {response_only}")

if __name__ == '__main__':
    main()
