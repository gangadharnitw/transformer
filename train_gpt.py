import os
import time
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

# ==========================================
#           CONFIGURATION SECTION
# ==========================================

# --- 1. RUN MODE ---
# Set to 'pretrain' for Phase 1 (Raw Code)
# Set to 'sft' for Phase 2 (Chat/Q&A)
RUN_MODE = 'pretrain'  

# --- 2. HARDWARE CONFIG ---
# UNCOMMENT the block that matches your machine

# [OPTION A] EPYC SERVER (64 Cores, 32GB RAM)
BATCH_SIZE = 32
BLOCK_SIZE = 1024
GRAD_ACCUM_STEPS = 1
NUM_THREADS = 32  
DEVICE = 'cpu'

# [OPTION B] LAPTOP (Ryzen 7840HS, 32GB RAM)
# BATCH_SIZE = 4
# BLOCK_SIZE = 512
# GRAD_ACCUM_STEPS = 8
# NUM_THREADS = 8
# DEVICE = 'cpu'

# --- 3. HYPERPARAMETERS ---
if RUN_MODE == 'pretrain':
    MAX_ITERS = 20000
    LEARNING_RATE = 3e-4
    DATA_FILE = 'train.bin'
    SAVE_EVERY = 500
    DROPOUT = 0.1
    print(f"--- STARTING PRE-TRAINING (Target Loss < 1.5) ---")
    
elif RUN_MODE == 'sft':
    MAX_ITERS = 1000
    LEARNING_RATE = 1e-5  # 30x slower for fine-tuning
    DATA_FILE = 'sft_train.bin'
    SAVE_EVERY = 100
    DROPOUT = 0.1
    print(f"--- STARTING SFT FINE-TUNING (Target Loss ~0.5) ---")

# Model Architecture (GPT-2 Small)
N_EMBD = 768
N_HEAD = 12
N_LAYER = 12
EVAL_INTERVAL = 100

# ==========================================
#           SYSTEM OPTIMIZATION
# ==========================================
if DEVICE == 'cpu':
    torch.set_num_threads(NUM_THREADS)
    torch.set_num_interop_threads(2)
    os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
    print(f"CPU Optimization: Using {NUM_THREADS} threads.")

# ==========================================
#              DATA LOADER
# ==========================================
def get_batch(split):
    # Select file based on Split and Mode
    if split == 'train':
        filename = DATA_FILE
    else:
        filename = 'val.bin' # Always validate on raw code to check regression
        
    if not os.path.exists(filename):
        print(f"ERROR: {filename} not found! Did you run the prepare script?")
        exit()
        
    data = np.memmap(filename, dtype=np.uint16, mode='r')
    
    # Random offsets
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    
    # Cast to int64 for PyTorch
    x = torch.stack([torch.from_numpy((data[i:i+BLOCK_SIZE]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+BLOCK_SIZE+1]).astype(np.int64)) for i in ix])
    
    return x.to(DEVICE), y.to(DEVICE)

# ==========================================
#           TRANSFORMER MODEL
# ==========================================
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        # Scaled Dot-Product Attention
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT),
        )
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(50257, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block(N_EMBD, N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, 50257)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

# ==========================================
#           TRAINING ENGINE
# ==========================================
def train():
    model = GPT().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Always try to load existing weights
    if os.path.exists("model.pth"):
        print(">> Loading existing 'model.pth'...")
        model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
        print(">> Weights Loaded! Continuing training...")
    else:
        print(">> No previous model found. Starting from scratch.")

    print(f"Model Size: {sum(p.numel() for p in model.parameters())/1e6:.2f} Million Parameters")
    
    model.train()
    start_time = time.time()

    for iter in range(MAX_ITERS):
        
        # --- EVALUATION LOGGING ---
        if iter % EVAL_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                losses = torch.zeros(5)
                for k in range(5):
                    X, Y = get_batch('train')
                    _, loss = model(X, Y)
                    losses[k] = loss.item()
                
                dt = time.time() - start_time
                print(f"Step {iter} | Loss: {losses.mean():.4f} | Time: {dt:.2f}s")
                start_time = time.time() # Reset timer
            
            # Save Checkpoint
            if iter > 0 and iter % SAVE_EVERY == 0:
                torch.save(model.state_dict(), "model.pth")
                print(f">> Saved model.pth at step {iter}")
                
            model.train()

        # --- GRADIENT ACCUMULATION STEP ---
        optimizer.zero_grad(set_to_none=True)
        for _ in range(GRAD_ACCUM_STEPS):
            X, Y = get_batch('train')
            _, loss = model(X, Y)
            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    # Final Save
    torch.save(model.state_dict(), "model.pth")
    print("TRAINING COMPLETE. Model saved.")

if __name__ == "__main__":
    train()
