import os
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

# ==========================================
#           MASTER CONFIGURATION
# ==========================================

# --- 1. RUN MODE ---
RUN_MODE = 'pretrain'  # Options: 'pretrain' or 'sft'

# --- 2. HARDWARE PROFILE ---
DEVICE_TYPE = 'epyc'   # Options: 'epyc' or 'laptop'

# --- 3. HYPERPARAMETERS (Derived from selections) ---
if DEVICE_TYPE == 'epyc':
    BATCH_SIZE, BLOCK_SIZE, THREADS = 32, 1024, 32
    GRAD_ACCUM_STEPS = 1
else: # Laptop config
    BATCH_SIZE, BLOCK_SIZE, THREADS = 4, 512, 8
    GRAD_ACCUM_STEPS = 8

if RUN_MODE == 'pretrain':
    LR, MAX_ITERS, DATA_FILE = 3e-4, 20000, 'train.bin'
else: # SFT config
    LR, MAX_ITERS, DATA_FILE = 1e-5, 1000, 'sft_train.bin'

# --- 4. MODEL ARCHITECTURE (GPT-2 Small) ---
N_EMBD, N_HEAD, N_LAYER, DROPOUT = 768, 12, 12, 0.1
VOCAB_SIZE = 50257

# --- 5. CONSTANTS ---
DEVICE = 'cpu'
EVAL_INTERVAL = 100
SAVE_EVERY = 500

# ==========================================
#           SYSTEM OPTIMIZATION
# ==========================================
torch.set_num_threads(THREADS)
os.environ["OMP_NUM_THREADS"] = str(THREADS)

# ==========================================
#              DATA LOADER
# ==========================================
def get_batch(split='train'):
    filename = VAL_FILE if split == 'val' else DATA_FILE
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} not found! Run the appropriate 'prepare' script.")
    
    data = np.memmap(filename, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy(data[i:i+BLOCK_SIZE].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+BLOCK_SIZE+1].astype(np.int64)) for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# ==========================================
#           TRANSFORMER MODEL
# ==========================================
# Using the more stable and standard nn.TransformerEncoderLayer
class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, N_EMBD)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=N_EMBD, nhead=N_HEAD, dim_feedforward=4*N_EMBD,
            dropout=DROPOUT, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYER)
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=DEVICE))
        x = tok_emb + pos_emb
        
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(DEVICE)
        x = self.transformer_encoder(x, mask=causal_mask)
        
        logits = self.lm_head(self.ln_f(x))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss

# ==========================================
#           TRAINING ENGINE
# ==========================================
def train():
    model = GPT().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    if os.path.exists("model.pth"):
        print(">> Loading existing 'model.pth' checkpoint...")
        model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
        print(">> Weights loaded! Resuming training.")
    else:
        print(">> No checkpoint found. Starting from scratch.")

    print(f"Starting {RUN_MODE.upper()} on {DEVICE_TYPE.upper()}...")
    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M | Threads: {THREADS}")

    for i in range(MAX_ITERS):
        t0 = time.time()
        
        # Eval and Save Logic
        if i % EVAL_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                val_loss = get_batch('val')[1] # Simple single batch validation
                print(f"Step {i} | Val Loss: {val_loss.item():.4f}")
            if i > 0 and i % SAVE_EVERY == 0:
                torch.save(model.state_dict(), "model.pth")
                print(f">> Checkpoint saved at step {i}")
            model.train()

        # Training Step with Gradient Accumulation
        optimizer.zero_grad()
        for _ in range(GRAD_ACCUM_STEPS):
            X, Y = get_batch()
            _, loss = model(X, Y)
            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if i % 10 == 0:
            dt = (time.time() - t0) * 1000
            print(f"Iter {i}: Loss {loss.item() * GRAD_ACCUM_STEPS:.4f}, Time: {dt:.2f}ms")

    torch.save(model.state_dict(), "model.pth")
    print("Final model saved. Training complete.")

if __name__ == '__main__':
    VAL_FILE = 'val.bin' # Define global for get_batch
    train()
