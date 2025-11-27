# Chess Review MVP: React UI + FastAPI backend


## Backend (FastAPI)
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

```bash
#Engine start, not always needed
curl "http://localhost:8000/eval?depth=14"

```

### LLM coach (local, offline)
1. Install the extra dependency already listed in `backend/requirements.txt` (`llama-cpp-python`).
2. Download any quantized GGUF that fits on your GPU into `backend/models/`. Example:
   ```bash
   mkdir -p backend/models
   curl -L -o backend/models/phi-3.5-mini-instruct-q4.gguf \
     https://huggingface.co/microsoft/Phi-3.5-mini-instruct-gguf/resolve/main/Phi-3.5-mini-instruct-q4.gguf?download=1
   ```
3. Point the backend at the file with a `.env` inside `backend/`:
   ```
   LLM_MODEL_PATH=backend/models/phi-3.5-mini-instruct-q4.gguf
   # optional tweaks
   # LLM_CTX_TOKENS=4096
   # LLM_MAX_OUTPUT_TOKENS=512
   # LLM_EAGER_START=1
   ```
4. Hit `GET /coach/status?start=1` once to warm the model. The frontend now exposes an “Ask Coach” panel that sends the current board state to `/coach/analyze` for a beginner-friendly explanation.

#### Recommended model for RTX 3080 owners
- The card’s 10 GB of VRAM can comfortably drive **Meta Llama 3.1 8B Instruct Q4\_K\_M (≈6.5 GB)**, which is noticeably stronger than tiny Phi models while still running entirely offline.
- Download steps (requires a free Hugging Face account and acceptance of Meta’s license):
  ```bash
  pip install -U "huggingface_hub[cli]"
  huggingface-cli login  # paste your token once
  mkdir -p backend/models
  huggingface-cli download \
    meta-llama/Llama-3.1-8B-Instruct-GGUF \
    Llama-3.1-8B-Instruct.Q4_K_M.gguf \
    --local-dir backend/models \
    --local-dir-use-symlinks False
  ```
- Point `LLM_MODEL_PATH` at `backend/models/Llama-3.1-8B-Instruct.Q4_K_M.gguf` and restart the backend. First load takes ~30 s on a 3080; future requests reuse the weights instantly.



## Frontend (React + Vite)
```bash
cd frontend
npm install
cp .env.example .env  # optional; defaults to http://localhost:8000
npm run dev
```

