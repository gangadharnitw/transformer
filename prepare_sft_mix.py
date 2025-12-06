import os
import re
import tiktoken
import numpy as np
from datasets import load_dataset

# --- CONFIGURATION ---
# Point this to your firmware directory to extract some C code for the mix.
SOURCE_DIR = r"C:\Work\Firmware\Src" 
# ---

def get_firmware_qa(limit_per_file=3):
    """Extracts simple function signature -> body pairs from local C files."""
    qa_pairs = []
    for root, _, files in os.walk(SOURCE_DIR):
        for file_name in files:
            if file_name.endswith('.c'):
                file_path = os.path.join(root, file_name)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        # Simple regex to find functions. May not be perfect.
                        # Matches: "void func_name(...) { body }"
                        functions = re.findall(r"((\w+\s+[\w\*]+\s*\(.*?\))\s*\{([^}]*)\})", content)
                        for _, signature, body in functions[:limit_per_file]:
                            entry = f"### Instruction:\nWrite the C implementation for the function: {signature}\n\n### Response:\n{{{body}}}\n<|endoftext|>\n"
                            qa_pairs.append(entry)
                except: continue
    return qa_pairs

def prepare_mixed_sft_dataset():
    """Downloads an open-source SFT dataset and mixes it with local firmware data."""
    # 1. Get open-source C-language instruction data
    print("Downloading CodeAlpaca from Hugging Face...")
    try:
        dataset = load_dataset("HuggingFaceH4/CodeAlpaca_20K", split="train")
    except Exception as e:
        print(f"Failed to download dataset. Check internet connection. Error: {e}")
        return

    alpaca_c_examples = []
    for item in dataset:
        instruction, output = item.get('instruction', ''), item.get('output', '')
        # Simple filter for C/C++ examples
        if "C" in instruction or "#include" in output:
            alpaca_c_examples.append(f"### Instruction:\n{instruction}\n\n### Response:\n{output}\n<|endoftext|>\n")

    # 2. Get private firmware Q&A data
    print("Scanning local firmware for self-supervised examples...")
    firmware_qa_examples = get_firmware_qa()
    
    # 3. Create the mix (e.g., 2000 open-source examples + all found firmware examples)
    mixed_data = alpaca_c_examples[:2000] + firmware_qa_examples
    
    if not mixed_data:
        print("No data was prepared! Check your paths and filters.")
        return
        
    print(f"Mixing Data: {min(2000, len(alpaca_c_examples))} Public Alpaca examples + {len(firmware_qa_examples)} Private Firmware examples.")
    
    # 4. Tokenize and save the final dataset
    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode("".join(mixed_data), disallowed_special=())
    
    np.array(ids, dtype=np.uint16).tofile('sft_train.bin')
    print("="*50)
    print("Phase 2 SFT Data Ready!")
    print(f"Created 'sft_train.bin' ({len(ids):,} tokens)")
    print("="*50)

if __name__ == '__main__':
    prepare_mixed_sft_dataset()
