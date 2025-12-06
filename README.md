# Firmware-GPT

This repository contains a complete pipeline to train a private, local AI assistant (a "Baby-GPT") on your company's proprietary C codebase. The model learns your coding style, APIs, and logic, then is fine-tuned to act as a helpful chatbot.

The entire process runs 100% offline on your own hardware, ensuring no proprietary code ever leaves your control.

## Workflow Overview

The process is divided into two main phases, followed by a final chat interface.

| Phase | Script | Goal | Data Source | Time Estimate |
| :--- | :--- | :--- | :--- | :--- |
| **1. Pre-training** | `prepare_codebase.py`<br>`train_gpt.py` | Teach the model the "language" of your C code. | Your entire firmware codebase (`.c`, `.h` files). | 2-10 days |
| **2. Fine-tuning** | `prepare_sft_mix.py`<br>`train_gpt.py` | Teach the model how to *answer questions* about code. | A mix of open-source Q&A and your own code. | ~4 hours |
| **3. Inference** | `chat.py` | Interact with your trained AI assistant. | User prompts. | Real-time |

---

## Step-by-Step Guide

### Step 0: Setup
1.  Clone this repository.
2.  Install the required Python libraries:
    ```
    pip install torch numpy tiktoken datasets
    ```

### Step 1: Pre-training (The "Brain")
*This is the longest but most important phase. The model learns your code.*

1.  **Configure:** Edit `prepare_codebase.py` and set the `SOURCE_DIR` variable to the root directory of your firmware source code.
2.  **Ingest Data:** Run the script. This will scan all your code files and create a single binary file (`train.bin`) for efficient training.
    ```
    python prepare_codebase.py
    ```
3.  **Train:**
    - Open `train_gpt.py`.
    - At the top, set `RUN_MODE = 'pretrain'`.
    - Select your hardware profile (`DEVICE_TYPE = 'epyc'` or `'laptop'`).
    - Run the script: `python train_gpt.py`
    - Let it run for several days until the loss drops below **1.5**. The script will save `model.pth` automatically.

### Step 2: Fine-Tuning (The "Teacher")
*Now we teach the code-literate model how to be a helpful chatbot.*

1.  **Prepare SFT Data:**
    - Edit `prepare_sft_mix.py` and confirm `SOURCE_DIR` is set.
    - Run the script. It will download an open-source dataset (`CodeAlpaca`) and mix it with snippets from your own code to create `sft_train.bin`.
    ```
    python prepare_sft_mix.py
    ```
2.  **Fine-Tune:**
    - Open `train_gpt.py`.
    - Change the configuration to `RUN_MODE = 'sft'`. The script will automatically use a lower learning rate and the new `sft_train.bin` file.
    - Run the training script again: `python train_gpt.py`
    - This will be very fast, likely finishing in a few hours. Let it run for its full 1000 steps.

### Step 3: Chat with Your AI
1.  Once fine-tuning is complete, run the interactive chat client:
    ```
    python chat.py
    ```
2.  Ask it questions about your code! For example:
    - *Explain the function `init_wifi`.*
    - *Write a C function to reverse a linked list.*

