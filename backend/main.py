# main.py
import os
import sys
import time
import uuid
import asyncio
import logging
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import chess
import chess.engine

# ----------------------------- Windows event loop (for subprocess on Windows) -----------------------------
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# ----------------------------- Env & logging -----------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Reduce python-chess engine noise like "Unexpected engine output"
logging.getLogger("chess.engine").setLevel(logging.ERROR)

# ----------------------------- Config -----------------------------
APP_TITLE = "Chess Review Backend"
APP_DESC = "Board state, undo/redo, engine evaluation (Stockfish), ply-safe moves."

DEFAULT_STOCKFISH_PATH = os.path.join(os.path.dirname(__file__), "engine", "stockfish.exe")
STOCKFISH_PATH = os.getenv("STOCKFISH_PATH", DEFAULT_STOCKFISH_PATH)

ENGINE_THREADS = int(os.getenv("ENGINE_THREADS", "8"))
ENGINE_HASH_MB = int(os.getenv("ENGINE_HASH", "256"))

# ----------------------------- App & CORS -----------------------------
app = FastAPI(title=APP_TITLE, description=APP_DESC)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # dev-friendly
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------- Global board state -----------------------------
board = chess.Board()
redo_stack: List[chess.Move] = []

# Serialize game mutations/reads
BOARD_LOCK = asyncio.Lock()

# Idempotency for /move
PROCESSED_REQ_IDS: Dict[str, float] = {}
PROCESSED_REQ_IDS_LIMIT = 200


def _gc_req_ids():
    if len(PROCESSED_REQ_IDS) > PROCESSED_REQ_IDS_LIMIT:
        # drop oldest half
        for k in list(sorted(PROCESSED_REQ_IDS.keys(), key=lambda k: PROCESSED_REQ_IDS[k]))[: PROCESSED_REQ_IDS_LIMIT // 2]:
            PROCESSED_REQ_IDS.pop(k, None)


def _current_ply() -> int:
    return len(board.move_stack)


def _history_san() -> List[str]:
    """
    Build SAN history by replaying moves on a temp board.
    IMPORTANT: san() must be called on the position BEFORE the move.
    """
    tmp = chess.Board()  # standard initial position (we reset in /new_game)
    out: List[str] = []
    for mv in board.move_stack:
        try:
            out.append(tmp.san(mv))
            tmp.push(mv)
        except Exception:
            # Failsafe: append UCI, still try to push to keep sequence aligned
            out.append(mv.uci())
            try:
                tmp.push(mv)
            except Exception:
                break
    return out


def _legal_moves_uci() -> List[str]:
    return [m.uci() for m in board.legal_moves]


def _state_json(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    d = {
        "fen": board.fen(),
        "turn": "w" if board.turn else "b",
        "move_number": board.fullmove_number,
        "is_game_over": board.is_game_over(),
        "result": board.result() if board.is_game_over() else None,
        "history_san": _history_san(),
        "legal_moves": _legal_moves_uci(),
        "can_undo": len(board.move_stack) > 0,
        "can_redo": len(redo_stack) > 0,
    }
    if extra:
        d.update(extra)
    d.update(engine.snapshot_status_fields())
    return d


# ----------------------------- Engine manager -----------------------------
class EngineManager:
    def __init__(self, path: str, threads: int = 8, hash_mb: int = 256):
        self.path = os.path.abspath(path)
        self.threads = threads
        self.hash_mb = hash_mb
        self.engine: Optional[chess.engine.SimpleEngine] = None
        self.engine_error: str = ""
        self.engine_type: str = "stockfish"
        self.lock = asyncio.Lock()  # serialize analyse() calls

    def exists(self) -> bool:
        return os.path.exists(self.path)

    def is_ready(self) -> bool:
        return self.engine is not None

    async def stop(self) -> None:
        """Gracefully stop and clear any engine process."""
        if self.engine is not None:
            try:
                await asyncio.to_thread(self.engine.quit)
            except Exception:
                try:
                    await asyncio.to_thread(self.engine.close)
                except Exception:
                    pass
        self.engine = None

    async def ensure_started(self) -> None:
        """
        Start the engine if needed. If already started, ping it; if ping fails,
        fully restart. On success, clears self.engine_error; on failure, sets it.
        """
        # If we think it's running, verify with a ping
        if self.engine is not None:
            try:
                await asyncio.to_thread(self.engine.ping)
                return
            except Exception:
                await self.stop()  # drop and try fresh start below

        if not self.exists():
            self.engine_error = f"Engine not found at path: {self.path}"
            return

        # Start the engine in a worker thread (reliable on Windows / Py3.12)
        try:
            self.engine = await asyncio.to_thread(chess.engine.SimpleEngine.popen_uci, self.path)
        except Exception as e:
            self.engine = None
            self.engine_error = f"popen_uci failed: {type(e).__name__}: {e}"
            return

        # Configure (best-effort)
        try:
            self.engine.configure({"Threads": self.threads, "Hash": self.hash_mb})
        except Exception as e:
            # Not fatal; keep a note
            self.engine_error = f"configure warning: {type(e).__name__}: {e}"

        # Ping/handshake
        try:
            await asyncio.to_thread(self.engine.ping)
        except Exception as e:
            await self.stop()
            self.engine_error = f"ping failed: {type(e).__name__}: {e}"
            return

        # Light probe to ensure analyse/GO works
        try:
            start_b = chess.Board(chess.STARTING_FEN)
            await asyncio.to_thread(self.engine.analyse, start_b, chess.engine.Limit(depth=1))
        except Exception as e:
            await self.stop()
            self.engine_error = f"probe failed: {type(e).__name__}: {e}"
            return

        # If we reach here, engine is up; clear any previous error note.
        self.engine_error = ""

    async def analyse_fen(self, fen: str, depth: int = 12) -> Dict[str, Any]:
        """Analyse a FEN. Returns normalized JSON (score/bestmove/pv/depth/time)."""
        await self.ensure_started()
        if self.engine is None:
            raise RuntimeError(self.engine_error or "Engine unavailable")

        b = chess.Board(fen)
        limit = chess.engine.Limit(depth=depth)

        async with self.lock:
            t0 = time.time()
            result = await asyncio.to_thread(self.engine.analyse, b, limit, multipv=1)
            elapsed = int((time.time() - t0) * 1000)

        # python-chess returns InfoDict or List[InfoDict] when multipv is set
        info = result[0] if isinstance(result, list) and result else (result or {})

        # Score (White POV)
        score_json: Dict[str, Any] = {}
        score_obj = info.get("score")
        if score_obj is not None:
            pov = score_obj.pov(chess.WHITE)
            if pov.is_mate():
                score_json["mate"] = pov.mate()
            else:
                score_json["cp"] = pov.score()

        # PV / best move
        pv_moves: List[chess.Move] = list(info.get("pv") or [])
        bestmove_uci = pv_moves[0].uci() if pv_moves else None
        bestmove_san = chess.Board(fen).san(pv_moves[0]) if pv_moves else None

        pv_uci = [m.uci() for m in pv_moves]
        pv_san: List[str] = []
        if pv_moves:
            tmp = chess.Board(fen)
            for mv in pv_moves:
                try:
                    pv_san.append(tmp.san(mv))
                    tmp.push(mv)
                except Exception:
                    break

        return {
            "fen": fen,
            "turn": "w" if b.turn else "b",
            "engine_type": self.engine_type,
            "score": score_json,
            "bestmove": {"uci": bestmove_uci, "san": bestmove_san} if bestmove_uci else None,
            "pv": {"uci": pv_uci, "san": pv_san},
            "depth": depth,
            "elapsed_ms": elapsed,
        }

    def snapshot_status_fields(self) -> Dict[str, Any]:
        return {
            "engine_ready": self.is_ready(),
            "engine_error": self.engine_error,
            "engine_type": self.engine_type,
        }


engine = EngineManager(STOCKFISH_PATH, ENGINE_THREADS, ENGINE_HASH_MB)

# ----------------------------- Lifespan -----------------------------
@app.on_event("startup")
async def _on_startup():
    # Eager-start the engine so it's ready immediately.
    await engine.ensure_started()


@app.on_event("shutdown")
async def _on_shutdown():
    await engine.stop()

# ----------------------------- Models -----------------------------
class MoveReq(BaseModel):
    uci: str = Field(..., description="UCI move, e.g., e2e4 or e7e8q")
    ply: Optional[int] = Field(None, description="Client's expected ply (len(move_stack))")
    req_id: Optional[str] = Field(None, description="Idempotency key (unique per attempt)")

# ----------------------------- Routes: State & controls -----------------------------
@app.get("/state")
async def get_state():
    async with BOARD_LOCK:
        return JSONResponse(_state_json())


@app.post("/new_game")
async def new_game():
    async with BOARD_LOCK:
        board.reset()
        redo_stack.clear()
        return JSONResponse(_state_json())


@app.post("/undo")
async def undo():
    async with BOARD_LOCK:
        if len(board.move_stack) > 0:
            mv = board.pop()
            redo_stack.append(mv)
        return JSONResponse(_state_json())


@app.post("/redo")
async def redo():
    async with BOARD_LOCK:
        if len(redo_stack) > 0:
            mv = redo_stack.pop()
            board.push(mv)
        return JSONResponse(_state_json())

# ----------------------------- Routes: Engine status / eval -----------------------------
@app.get("/engine/status")
async def engine_status(start: int = Query(0, description="If 1, ensure engine is started")):
    if start:
        await engine.ensure_started()
    status = {
        "ready": engine.is_ready(),
        "error": engine.engine_error,
        "path": engine.path,
        "exists": engine.exists(),
        "threads": ENGINE_THREADS,
        "hash": ENGINE_HASH_MB,
    }
    return JSONResponse(status)


@app.get("/engine/diag")
async def engine_diag():
    await engine.ensure_started()
    if not engine.is_ready():
        return JSONResponse({"ok": False, "error": engine.engine_error})
    return JSONResponse(
        {"ok": True, "type": engine.engine_type, "path": engine.path, "threads": ENGINE_THREADS, "hash": ENGINE_HASH_MB}
    )


@app.get("/eval")
async def eval_position(depth: int = Query(12, ge=4, le=40)):
    # Snapshot FEN under lock
    async with BOARD_LOCK:
        fen = board.fen()

    # Try, restart, and retry once on known engine failures (incl. NotImplementedError)
    for attempt in (1, 2):
        try:
            data = await engine.analyse_fen(fen, depth=depth)
            return JSONResponse(data)
        except (chess.engine.EngineTerminatedError, RuntimeError, NotImplementedError) as e:
            await engine.stop()
            await asyncio.sleep(0.05)
            if attempt == 1:
                continue
            return JSONResponse({"detail": f"Engine unavailable: {e}"}, status_code=503)
        except Exception as e:
            return JSONResponse({"detail": f"Engine error: {type(e).__name__}: {e}"}, status_code=503)

# ----------------------------- Route: Move (ply-checked, idempotent) -----------------------------
@app.post("/move")
async def move(req: MoveReq):
    """
    Apply a move if it matches the client's view of the position (ply check).
    Reject stale or duplicate requests explicitly.
    """
    rid = req.req_id or str(uuid.uuid4())
    now = time.time()

    # Fast duplicate check
    if rid in PROCESSED_REQ_IDS:
        async with BOARD_LOCK:
            return JSONResponse(_state_json())

    async with BOARD_LOCK:
        # dup check under lock
        if rid in PROCESSED_REQ_IDS:
            return JSONResponse(_state_json())

        fen_before = board.fen()
        ply_now = _current_ply()

        if req.ply is not None and req.ply != ply_now:
            return JSONResponse(
                {"detail": f"Stale move: server ply={ply_now}, client ply={req.ply}"},
                status_code=409,
            )

        u = (req.uci or "").strip().lower()
        if not (len(u) == 4 or (len(u) == 5 and u[-1] in "qrbn")):
            return JSONResponse({"detail": "Invalid UCI format"}, status_code=400)

        try:
            mv = chess.Move.from_uci(u)
        except Exception:
            return JSONResponse({"detail": "Invalid UCI move"}, status_code=400)

        if mv not in board.legal_moves:
            sample = [m.uci() for _, m in zip(range(12), board.legal_moves)]
            return JSONResponse(
                {
                    "detail": "Illegal move in current position.",
                    "server_ply": ply_now,
                    "fen_before": fen_before,
                    "attempt": u,
                    "legal_sample": sample,
                },
                status_code=400,
            )

        board.push(mv)
        redo_stack.clear()

        PROCESSED_REQ_IDS[rid] = now
        _gc_req_ids()

        return JSONResponse(_state_json())

# ----------------------------- Root -----------------------------
@app.get("/")
async def root():
    return JSONResponse({"ok": True, "engine": engine.snapshot_status_fields()})

# ----------------------------- Uvicorn entry -----------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", reload=False, host="127.0.0.1", port=8000)
