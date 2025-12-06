import os
import tiktoken
import numpy as np

# --- CONFIGURATION ---
# CHANGE THIS to your actual firmware folder path
SOURCE_DIR = r"C:\Work\Firmware\Src"  
EXTENSIONS = {'.c', '.h', '.cpp', '.hpp', '.ld', '.s'} 

def process_codebase():
    data = []
    file_count = 0
    print(f"Scanning {SOURCE_DIR}...")
    
    for root, dirs, files in os.walk(SOURCE_DIR):
        for file in files:
            if any(file.endswith(ext) for ext in EXTENSIONS):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        # Add headers so the model learns file boundaries
                        data.append(f"\n/* --- FILE: {file} --- */\n") 
                        data.append(content)
                        file_count += 1
                except Exception as e:
                    print(f"Skipping {file}: {e}")
                    
    print(f"Found {file_count} files.")
    full_text = "".join(data)
    print(f"Total characters: {len(full_text):,}")

    print("Tokenizing... (This may take a minute)")
    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode(full_text)
    print(f"Total tokens: {len(ids):,}")

    # Split 90% Train / 10% Validation
    n = int(0.9 * len(ids))
    train_ids = np.array(ids[:n], dtype=np.uint16)
    val_ids = np.array(ids[n:], dtype=np.uint16)
    
    print("Saving binaries...")
    train_ids.tofile('train.bin')
    val_ids.tofile('val.bin')
    print("Done! Created 'train.bin' and 'val.bin'")

if __name__ == '__main__':
    process_codebase()
