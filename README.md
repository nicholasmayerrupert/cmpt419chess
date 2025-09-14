# Chess Review MVP: React UI + FastAPI backend

This starter lets you move pieces on a React board and have the Python backend own the game state.

## Backend (FastAPI)
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
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



## Frontend (React + Vite)
```bash
cd frontend
npm install
cp .env.example .env  # optional; defaults to http://localhost:8000
npm run dev
```

