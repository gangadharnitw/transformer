import os
import re
import tiktoken
import numpy as np

# --- CONFIGURATION ---
SOURCE_DIR = r"C:\Work\Firmware\Src"

def extract_functions_and_comments(code):
    # Regex: Matches "// Comment" followed immediately by "void function(...)"
    pattern = r"((?://.*?\n)+)\s*([\w\s\*]+\s+(\w+)\s*\(.*?\))"
    return re.findall(pattern, code)

def create_sft_dataset():
    qa_pairs = []
    print(f"Scanning {SOURCE_DIR} for SFT data...")

    for root, dirs, files in os.walk(SOURCE_DIR):
        for file in files:
            if file.endswith('.c'):
                try:
                    with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        matches = extract_functions_and_comments(content)
                        
                        for comment, signature, func_name in matches:
                            clean_comment = comment.replace('//', '').replace('\n', ' ').strip()
                            
                            # The "Chat" Format
                            entry = f"### Instruction:\nExplain {func_name}.\n\n"
                            entry += f"### Response:\n{clean_comment}\n\n"
                            entry += f"Signature: {signature}\n"
                            entry += "<|endoftext|>\n" 
                            qa_pairs.append(entry)
                except: pass

    full_text = "".join(qa_pairs)
    print(f"Generated {len(qa_pairs)} Q&A pairs ({len(full_text)} chars).")
    
    # Tokenize and Save
    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode(full_text)
    
    # Save as a separate binary for Fine-Tuning
    np.array(ids, dtype=np.uint16).tofile('sft_train.bin')
    print("Saved 'sft_train.bin'")

if __name__ == '__main__':
    create_sft_dataset()
