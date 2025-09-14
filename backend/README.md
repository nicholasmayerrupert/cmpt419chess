# Chess State Server (FastAPI)

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```


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

CORS is open for local dev. Tighten before production.
