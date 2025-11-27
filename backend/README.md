# Chess State Server (FastAPI)

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Enable the local LLM coach

1. Pick a quantized `*.gguf` model that fits in GPU/CPU RAM. A good starter that runs on a 3080 is the Microsoft Phi-3.5 Mini Instruct Q4 build.
2. Download it into `backend/models` (create the folder if needed):
   ```bash
   mkdir -p backend/models
   curl -L -o backend/models/phi-3.5-mini-instruct-q4.gguf \
     https://huggingface.co/microsoft/Phi-3.5-mini-instruct-gguf/resolve/main/Phi-3.5-mini-instruct-q4.gguf?download=1
   ```
3. Create a `.env` file inside `backend/` to point the server at the file:
   ```bash
   cat <<'EOF' > backend/.env
   LLM_MODEL_PATH=backend/models/phi-3.5-mini-instruct-q4.gguf
   LLM_CTX_TOKENS=4096
   LLM_MAX_OUTPUT_TOKENS=512
   # Optional: set to 1 if you want the model to load on startup
   # LLM_EAGER_START=1
   EOF
   ```
4. Restart the FastAPI server. Check `GET /coach/status?start=1` to verify the model is loaded.

> Any GGUF that works with `llama-cpp-python` can be used—just update `LLM_MODEL_PATH`.


```bash
#Windows
npm install -D @vitejs/plugin-react

npm install

npm run dev

```

```bash
#Engine start
curl "http://localhost:8000/eval?depth=14"

```



## Endpoints
- `GET /state` → current FEN, legal moves, history
- `POST /new_game`
- `POST /move` JSON: `{ "uci": "e2e4" }` or `{ "san": "e4" }` (optional `"promotion": "q"`)
- `POST /undo`
- `POST /redo`
- `POST /set_fen` (plain text body containing FEN string)
- `GET /engine/status`, `GET /engine/diag`, `GET /eval?depth=12`
- `GET /coach/status?start=1`
- `POST /coach/analyze` JSON: `{ "question": "Explain weaknesses for White" }`

CORS is open for local dev. Tighten before production.
