import torch
import tiktoken
from train_gpt import GPT, DEVICE, N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE

# 1. Load Model
model = GPT().to(DEVICE)
model.load_state_dict(torch.load('model.pth', map_location=DEVICE))
model.eval()

# 2. Setup Tokenizer
enc = tiktoken.get_encoding("gpt2")

# 3. Chat Loop
print("FIRMWARE BOT READY! (Type 'quit' to exit)")
while True:
    prompt = input("\nUser: ")
    if prompt.lower() == 'quit': break
    
    # Add Chat formatting if you did SFT
    # prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
    
    input_ids = torch.tensor(enc.encode(prompt), dtype=torch.long, device=DEVICE).unsqueeze(0)
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=200)[0].tolist()
    
    response = enc.decode(output_ids)
    
    # Clean up output (remove prompt echo)
    # response = response[len(prompt):] 
    
    print(f"Bot: {response}")
