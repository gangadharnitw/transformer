import os
import tiktoken
import numpy as np

# --- CONFIGURATION ---
# IMPORTANT: Update this path to the root directory of your firmware source code.
SOURCE_DIR = r"C:\Work\Firmware\Src"  
# ---

def process_codebase():
    """Scans a directory, tokenizes all code files, and saves to a binary for training."""
    data_content = []
    print(f"Scanning directory: {SOURCE_DIR}...")
    
    for root, _, files in os.walk(SOURCE_DIR):
        for file_name in files:
            if file_name.endswith(('.c', '.h', '.cpp', '.hpp', '.s', '.ld')):
                file_path = os.path.join(root, file_name)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        # Add a special header so the model learns file boundaries
                        data_content.append(f"\n/* --- FILE: {file_name} --- */\n")
                        data_content.append(f.read())
                except Exception as e:
                    print(f"Skipping {file_name} due to error: {e}")
    
    if not data_content:
        print("No files found! Check your SOURCE_DIR and file extensions.")
        return

    print(f"Found and read {len(data_content)//2} files.")
    full_text = "".join(data_content)
    
    print("Tokenizing text... (This can take a few minutes for large codebases)")
    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode(full_text, disallowed_special=())
    
    # Split data into 90% for training, 10% for validation
    split_index = int(0.9 * len(ids))
    train_ids = np.array(ids[:split_index], dtype=np.uint16)
    val_ids = np.array(ids[split_index:], dtype=np.uint16)
    
    # Save to binary files
    train_ids.tofile('train.bin')
    val_ids.tofile('val.bin')
    
    print("="*50)
    print("Phase 1 Data Ready!")
    print(f"Created 'train.bin' ({len(train_ids):,} tokens)")
    print(f"Created 'val.bin' ({len(val_ids):,} tokens)")
    print("="*50)

if __name__ == '__main__':
    process_codebase()
