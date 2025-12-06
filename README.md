# GPT

Train a private LLM on your company's C codebase.

## Instructions for EPYC Server (Windows/Linux)

1. **Install Requirements:**
   `pip install -r requirements.txt`

2. **Ingest Codebase:**
   - Edit `prepare_codebase.py` -> Set `SOURCE_DIR = "path/to/firmware"`
   - Run `python prepare_codebase.py`
   - Creates `train.bin` (Binary dataset).

3. **Train Model:**
   - Edit `train_gpt.py` -> Ensure "EPYC SERVER" config is uncommented.
   - Run `python train_gpt.py`
   - Wait for Loss to drop below 1.5 (approx 2-3 days).

4. **Chat:**
   - Run `python play.py`



# Firmware-GPT

Train a private, secure AI assistant on your company's proprietary C codebase.

## Features
- **Pre-training:** Learns your coding style, proprietary APIs, and hardware registers from scratch.
- **SFT (Supervised Fine-Tuning):** Teaches the model to act as a "Consultant" that can explain logic and answer questions.
- **100% Local:** Runs entirely on your hardware (Ryzen Laptop or EPYC Server). No data leaves the building.

## Workflow Overview

| Stage | Goal | Data | Time (Est.) |
| :--- | :--- | :--- | :--- |
| **1. Ingest** | Convert 1000s of C files to binary | `prepare_codebase.py` | 5 mins |
| **2. Pre-train** | Teach model to *write* your code | `train_gpt.py` (Stage 1) | 2-5 Days |
| **3. Label (SFT)** | Create Q&A pairs using local AI | `generate_synthetic_sft.py` | 1-2 Hours |
| **4. Fine-tune** | Teach model to *explain* code | `train_gpt.py` (Stage 2) | 2 Hours |

---

## Step-by-Step Guide

### Phase 1: The "Brain" (Pre-training)
1.  **Configure:** Edit `prepare_codebase.py` and set `SOURCE_DIR` to your firmware src folder.
2.  **Run:** `python prepare_codebase.py`. This creates `train.bin` (the raw knowledge).
3.  **Train:** 
    - Open `train_gpt.py`.
    - Ensure `RUN_MODE = 'pretrain'`.
    - Run `python train_gpt.py`.
    - **Stop** when Loss < 1.5 (approx 20k-50k steps).

### Phase 2: The "Teacher" (SFT - Option B)
*Goal: Create a smart Q&A dataset so the bot understands natural language.*

1.  **Install Ollama:** Download and install [Ollama](https://ollama.com/).
2.  **Pull a Model:** Run `ollama run qwen2.5-coder:7b` (or `llama3`) in your terminal.
3.  **Generate Data:**
    - Edit `generate_synthetic_sft.py` -> Set `SOURCE_DIR` to your *critical* folders (e.g., drivers, core logic).
    - Run `python generate_synthetic_sft.py`.
    - This uses the local Ollama model to read your C functions and write English explanations for them.
    - Output: `sft_train.bin`.

4.  **Fine-Tune:**
    - Open `train_gpt.py`.
    - Set `RUN_MODE = 'sft'`.
    - Set `LEARNING_RATE = 1e-5` (Low and slow).
    - Run `python train_gpt.py`.
    - **Stop** after ~1000 steps.

### Phase 3: Chat
Run `python play.py` to talk to your new Firmware Assistant.

