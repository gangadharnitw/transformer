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
