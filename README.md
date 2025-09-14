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



## Frontend (React + Vite)
```bash
cd frontend
npm install
cp .env.example .env  # optional; defaults to http://localhost:8000
npm run dev
```

