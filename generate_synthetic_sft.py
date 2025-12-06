import os
import requests
import re
import tiktoken
import numpy as np

# --- CONFIGURATION ---
# Only scan CRITICAL folders for SFT (don't waste time on boilerplate)
SOURCE_DIR = r"C:\Work\Firmware\Src\Drivers" 
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5-coder:7b" # Ensure you have this pulled in Ollama

def extract_functions(code):
    # Simple regex to grab "void func() { ... }" blocks
    # Note: Captures ~50 lines max to fit in context
    pattern = r"((?:\w+\s+)+\w+\s*\(.*?\)\s*\{[\s\S]{0,2000}\n\})"
    return re.findall(pattern, code)

def ask_ollama(code_snippet):
    prompt = f"""
    Act as a Senior Firmware Engineer. 
    Analyze this C function from our codebase and describe WHAT it does and WHY, in 1 short sentence.
    Do not describe syntax. Focus on the hardware/logic intent.
    
    Code:
    {code_snippet[:1000]} ... (truncated)

    Explanation:
    """
    
    data = { "model": MODEL, "prompt": prompt, "stream": False }
    try:
        resp = requests.post(OLLAMA_URL, json=data).json()
        return resp['response'].strip()
    except: return None

def create_synthetic_dataset():
    qa_pairs = []
    print(f"Scanning {SOURCE_DIR} for functions...")
    
    files_processed = 0
    functions_labeled = 0

    for root, dirs, files in os.walk(SOURCE_DIR):
        for file in files:
            if file.endswith('.c'):
                path = os.path.join(root, file)
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    functions = extract_functions(content)
                    
                    # Limit: Only label top 5 functions per file to save time
                    for func_code in functions[:5]:
                        explanation = ask_ollama(func_code)
                        
                        if explanation:
                            # Format as Chat
                            entry = f"### Instruction:\n{explanation}\n\n"
                            entry += f"### Response:\n{func_code}\n"
                            entry += "<|endoftext|>\n"
                            qa_pairs.append(entry)
                            functions_labeled += 1
                            print(f"Labeled: {explanation[:40]}...")
                
                files_processed += 1
                if files_processed % 10 == 0:
                    print(f"--- Processed {files_processed} files ---")

    # Save and Tokenize
    print(f"Total SFT Examples Generated: {len(qa_pairs)}")
    full_text = "".join(qa_pairs)
    
    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode(full_text)
    np.array(ids, dtype=np.uint16).tofile('sft_train.bin')
    print("Saved 'sft_train.bin'. Ready for Phase 2 training!")

if __name__ == '__main__':
    create_synthetic_dataset()
