import os
import sys
import time
import uuid
import asyncio
import logging
import threading
from typing import Optional, Dict, Any, List, AsyncGenerator, Tuple

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

import chess
import chess.engine
try:
    from llama_cpp import Llama
except Exception:  # ImportError or GPU init issues
    Llama = None  # type: ignore
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

# windows event loop (for subprocess on windows)
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# env & logging
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# reduce python-chess engine noise like "unexpected engine output"
logging.getLogger("chess.engine").setLevel(logging.ERROR)
COACH_LOGGER = logging.getLogger("coach")
COACH_LOGGER.setLevel(logging.INFO)
if not COACH_LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s coach: %(message)s")
    handler.setFormatter(formatter)
    COACH_LOGGER.addHandler(handler)

# config
APP_TITLE = "Chess Review Backend"
APP_DESC = "Board state, undo/redo, engine evaluation (Stockfish), ply-safe moves."

DEFAULT_STOCKFISH_PATH = os.path.join(os.path.dirname(__file__), "engine", "stockfish.exe")
STOCKFISH_PATH = os.getenv("STOCKFISH_PATH", DEFAULT_STOCKFISH_PATH)

ENGINE_THREADS = int(os.getenv("ENGINE_THREADS", "8"))
ENGINE_HASH_MB = int(os.getenv("ENGINE_HASH", "256"))

DEFAULT_LLM_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "Mistral-Nemo-12B-Instruct.Q4_K_M.gguf")
#DEFAULT_LLM_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf")
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", DEFAULT_LLM_MODEL_PATH)
LLM_CTX_TOKENS = int(os.getenv("LLM_CTX_TOKENS", "4096"))
LLM_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "512"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.6"))
LLM_EAGER_START = os.getenv("LLM_EAGER_START", "0") == "1"
COACH_PROVIDER = os.getenv("COACH_PROVIDER", "local").strip().lower()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENINGS_TSV_PATH = os.path.join(os.path.dirname(__file__), "data", "chess_openings.tsv")

# app & cors
app = FastAPI(title=APP_TITLE, description=APP_DESC)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # dev-friendly
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# global board state
board = chess.Board()
redo_stack: List[chess.Move] = []
OPENINGS_BY_EPD: Dict[str, Dict[str, str]] = {}

# serialize game mutations/reads
BOARD_LOCK = asyncio.Lock()

# idempotency for /move
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
            # failsafe: append uci, still try to push to keep sequence aligned
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


def _load_openings_database() -> None:
    """
    Load the lichess chess openings TSV if available.
    """
    if not os.path.exists(OPENINGS_TSV_PATH):
        logging.info("Opening data not found at %s", OPENINGS_TSV_PATH)
        return

    loaded = 0
    try:
        with open(OPENINGS_TSV_PATH, "rb") as fh:
            raw = fh.read()

        text = None
        for enc in ("utf-8-sig", "utf-16", "utf-16-le", "utf-16-be"):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue

        if text is None:
            raise UnicodeDecodeError("openings", b"", 0, 1, "Unsupported encoding")

        lines = text.splitlines()
        if not lines:
            logging.warning("Opening data file is empty: %s", OPENINGS_TSV_PATH)
            return

        for line in lines[1:]:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 5:
                continue
            eco, name, pgn, uci_line, epd = parts
            OPENINGS_BY_EPD[epd] = {"eco": eco, "name": name, "pgn": pgn, "uci": uci_line}
            loaded += 1
        logging.info("Loaded %d chess openings from %s", loaded, OPENINGS_TSV_PATH)
    except Exception:
        logging.exception("Failed to load opening data from %s", OPENINGS_TSV_PATH)


def _detect_opening_from_moves(moves: List[chess.Move]) -> Optional[Dict[str, str]]:
    """
    Walk the moves from the initial position and return the last matching opening entry.
    """
    if not OPENINGS_BY_EPD or not moves:
        return None

    tmp = chess.Board()
    last_match: Optional[Dict[str, str]] = None
    for mv in moves:
        try:
            tmp.push(mv)
        except Exception:
            break
        info = OPENINGS_BY_EPD.get(tmp.epd())
        if info:
            last_match = info
    return last_match


_load_openings_database()


# engine manager
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
        # if we think it's running, verify with a ping
        if self.engine is not None:
            try:
                await asyncio.to_thread(self.engine.ping)
                return
            except Exception:
                await self.stop()  # drop and try fresh start below

        if not self.exists():
            self.engine_error = f"Engine not found at path: {self.path}"
            return

        # start the engine in a worker thread (reliable on windows / py3.12)
        try:
            self.engine = await asyncio.to_thread(chess.engine.SimpleEngine.popen_uci, self.path)
        except Exception as e:
            self.engine = None
            self.engine_error = f"popen_uci failed: {type(e).__name__}: {e}"
            return

        # configure (best-effort)
        try:
            self.engine.configure({"Threads": self.threads, "Hash": self.hash_mb})
        except Exception as e:
            # not fatal; keep a note
            self.engine_error = f"configure warning: {type(e).__name__}: {e}"

        # ping/handshake
        try:
            await asyncio.to_thread(self.engine.ping)
        except Exception as e:
            await self.stop()
            self.engine_error = f"ping failed: {type(e).__name__}: {e}"
            return

        # light probe to ensure analyse/go works
        try:
            start_b = chess.Board(chess.STARTING_FEN)
            await asyncio.to_thread(self.engine.analyse, start_b, chess.engine.Limit(depth=1))
        except Exception as e:
            await self.stop()
            self.engine_error = f"probe failed: {type(e).__name__}: {e}"
            return

        # if we reach here, engine is up; clear any previous error note.
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

        # python-chess returns infodict or list[infodict] when multipv is set
        info = result[0] if isinstance(result, list) and result else (result or {})

        # score (white pov)
        score_json: Dict[str, Any] = {}
        score_obj = info.get("score")
        if score_obj is not None:
            pov = score_obj.pov(chess.WHITE)
            if pov.is_mate():
                score_json["mate"] = pov.mate()
            else:
                score_json["cp"] = pov.score()

        # pv / best move
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


def _normalize_chat_history(history: Optional[List[Dict[str, str]]]) -> List[Dict[str, str]]:
    """
    llama-cpp and OpenAI both prefer alternating user/assistant roles after any optional system prompt.
    Filter malformed entries, enforce alternation starting with a user turn, and drop a trailing user entry.
    """
    if not history:
        return []

    normalized: List[Dict[str, str]] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        role_raw = item.get("role")
        content_raw = item.get("content")
        if not isinstance(role_raw, str) or not isinstance(content_raw, str):
            continue
        role = role_raw.strip().lower()
        content = content_raw.strip()
        if role not in {"user", "assistant"} or not content:
            continue
        if not normalized:
            if role != "user":
                continue
        else:
            if normalized[-1]["role"] == role:
                continue
        normalized.append({"role": role, "content": content})

    if normalized and normalized[-1]["role"] == "user":
        normalized.pop()

    return normalized


def _summarize_engine_eval(engine_eval: Optional[Dict[str, Any]]) -> str:
    """
    Turn the latest Stockfish evaluation into a short human-readable summary.
    """
    if not engine_eval:
        return "Engine evaluation unavailable."

    error = engine_eval.get("error")
    if error:
        return f"Engine evaluation unavailable ({error})."

    score = engine_eval.get("score") or {}
    mate = score.get("mate")
    cp = score.get("cp")

    score_text: str
    if mate is not None:
        try:
            mate_val = int(mate)
        except (TypeError, ValueError):
            mate_val = 0
        pov = "White" if mate_val > 0 else "Black"
        score_text = f"mate in {abs(mate_val)} for {pov}"
    elif cp is not None:
        try:
            cp_val = float(cp)
        except (TypeError, ValueError):
            cp_val = 0.0
        pov = "White" if cp_val >= 0 else "Black"
        score_text = f"{abs(cp_val) / 100:.2f} pawns for {pov}"
    else:
        score_text = "no numeric score"

    bestmove = engine_eval.get("bestmove") or {}
    bestmove_text = bestmove.get("san") or bestmove.get("uci") or "n/a"

    pv_san = (engine_eval.get("pv") or {}).get("san") or []
    pv_text = " ".join(pv_san[:6]) if pv_san else "n/a"

    depth = engine_eval.get("depth")
    if isinstance(depth, int):
        depth_text = f"depth {depth}"
    else:
        depth_text = "unknown depth"

    return f"Score: {score_text}. Preferred move: {bestmove_text}. Search {depth_text}. PV: {pv_text}."


def _summarize_opening(opening_info: Optional[Dict[str, str]]) -> str:
    if not opening_info:
        return "Opening unknown or not listed in the reference."
    name = opening_info.get("name", "Unknown opening")
    line = opening_info.get("pgn") or "N/A"
    return f"{name}. Reference line: {line}."


def _build_coach_prompts(
    fen: str,
    history_san: List[str],
    side_to_move: str,
    question: Optional[str],
    chat_history: Optional[List[Dict[str, str]]],
    engine_eval: Optional[Dict[str, Any]],
    opening_info: Optional[Dict[str, str]],
) -> Tuple[str, str, List[Dict[str, str]]]:
    history_tail = history_san[-24:]
    base_question = (question or "").strip()[:400]
    if not base_question:
        base_question = (
            "Give a short, beginner-friendly explanation of the plans, "
            "threats, and simple tactics that both sides should look for."
        )
    board_ascii = str(chess.Board(fen))
    history_text = ", ".join(history_tail) if history_tail else "No moves played yet."

    engine_summary = _summarize_engine_eval(engine_eval)
    opening_summary = _summarize_opening(opening_info)

    system_prompt = (
        "You are 'Coach Bot', a friendly chess tutor. "
        "Explain moves clearly, highlight the most urgent threats, describe why key squares matter, "
        "and suggest candidate moves for the side to move. Avoid overwhelming jargon. "
        "Answer briefly, summaring board state, best moves, and so on. "
    )

    user_prompt = (
        f"Current FEN: {fen}\n"
        f"Side to move: {side_to_move}\n"
        f"Move list (SAN): {history_text}\n"
        f"ASCII board:\n{board_ascii}\n\n"
        f"Opening insight: {opening_summary}\n"
        f"Stockfish evaluation summary: {engine_summary}\n"
        f"Instruction: {base_question}\n"
    )

    sanitized_history = _normalize_chat_history(chat_history)
    return system_prompt, user_prompt, sanitized_history


class ChessCoachLLM:
    """
    Manages a local GGUF model through llama-cpp-python.
    Keeps loading lightweight and serializes inference requests.
    """

    def __init__(self, path: str, ctx_tokens: int, max_output_tokens: int, temperature: float = 0.6):
        self.path = os.path.abspath(path)
        self.ctx_tokens = ctx_tokens
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.model: Optional["Llama"] = None
        self.model_error: str = ""
        self.model_label = os.path.basename(self.path)
        self.lock = asyncio.Lock()  # serialize generate() calls

    def exists(self) -> bool:
        return os.path.exists(self.path)

    def is_ready(self) -> bool:
        return self.model is not None

    async def ensure_loaded(self) -> None:
        if self.model is not None:
            return
        if Llama is None:
            self.model_error = "llama-cpp-python is not installed"
            return
        if not self.exists():
            self.model_error = f"Model not found at path: {self.path}"
            return

        try:
            # load in a worker thread because llama-cpp does blocking init.
            self.model = await asyncio.to_thread(
                Llama,
                model_path=self.path,
                n_ctx=self.ctx_tokens,
                n_gpu_layers=-1,  # auto-detect; falls back to CPU if needed
                verbose=False,
            )
            self.model_label = getattr(self.model, "model_path", self.model_label)
            self.model_error = ""
        except Exception as e:
            self.model = None
            self.model_error = f"llama init failed: {type(e).__name__}: {e}"

    async def stream_analysis(
        self,
        fen: str,
        history_san: List[str],
        side_to_move: str,
        question: Optional[str],
        chat_history: Optional[List[Dict[str, str]]] = None,
        max_tokens: Optional[int] = None,
        engine_eval: Optional[Dict[str, Any]] = None,
        opening_info: Optional[Dict[str, str]] = None,
    ) -> AsyncGenerator[str, None]:
        
        await self.ensure_loaded()
        if self.model is None:
            yield f"Error: {self.model_error or 'LLM unavailable'}"
            return

        system_prompt, user_prompt, sanitized_history = _build_coach_prompts(
            fen=fen,
            history_san=history_san,
            side_to_move=side_to_move,
            question=question,
            chat_history=chat_history,
            engine_eval=engine_eval,
            opening_info=opening_info,
        )

        tokens = max_tokens or self.max_output_tokens

        async with self.lock:
            try:
                # we request a stream from llama-cpp
                # although llama-cpp-python releases gil, we wrap in to_thread 
                # to ensure the initial call doesn't block the loop.
                stream = await asyncio.to_thread(
                    self.model.create_chat_completion,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        *sanitized_history,
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=tokens,
                    stream=True,
                )

                for chunk in stream:
                    delta = chunk['choices'][0]['delta']
                    content = delta.get('content', '')
                    if content:
                        yield content
                        # yield to event loop to allow other tasks (ping/eval) to progress
                        await asyncio.sleep(0)

            except Exception as e:
                yield f"\n[Analysis failed: {type(e).__name__}: {e}]"


class OpenAIChatCoach:
    """
    Uses the OpenAI Chat Completions API instead of a local llama.cpp model.
    """

    def __init__(
        self,
        model_name: str,
        max_output_tokens: int,
        temperature: float = 0.6,
        api_key: Optional[str] = None,
    ):
        self.model_label = model_name
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client: Optional["OpenAI"] = None
        self.model_error: str = ""
        self.path = f"openai::{model_name}"
        self.lock = asyncio.Lock()

    def exists(self) -> bool:
        return bool(self.api_key)

    def is_ready(self) -> bool:
        return self.client is not None

    async def ensure_loaded(self) -> None:
        if self.client is not None:
            return
        if OpenAI is None:
            self.model_error = "openai package is not installed"
            return
        if not self.api_key:
            self.model_error = "OPENAI_API_KEY is not set"
            return
        try:
            self.client = OpenAI(api_key=self.api_key)
            self.model_error = ""
        except Exception as e:
            self.client = None
            self.model_error = f"OpenAI init failed: {type(e).__name__}: {e}"

    def _use_responses_api(self) -> bool:
        """
        gpt-4.1 and o-series models require the Responses API.
        """
        name = self.model_label.lower()
        return name.startswith(("o1-mini-2", "gpt-4.1")) 

    @staticmethod
    def _build_responses_input(
        system_prompt: str,
        sanitized_history: List[Dict[str, str]],
        user_prompt: str,
    ) -> List[Dict[str, Any]]:
        """
        Convert the conversation into the shape expected by the Responses API.
        """

        def _block(role: str, content: str) -> Dict[str, Any]:
            ctype = "output_text" if role == "assistant" else "input_text"
            return {"role": role, "content": [{"type": ctype, "text": content}]}

        items: List[Dict[str, Any]] = [_block("system", system_prompt)]
        for item in sanitized_history:
            items.append(_block(item["role"], item["content"]))
        items.append(_block("user", user_prompt))
        return items

    async def stream_analysis(
        self,
        fen: str,
        history_san: List[str],
        side_to_move: str,
        question: Optional[str],
        chat_history: Optional[List[Dict[str, str]]] = None,
        max_tokens: Optional[int] = None,
        engine_eval: Optional[Dict[str, Any]] = None,
        opening_info: Optional[Dict[str, str]] = None,
    ) -> AsyncGenerator[str, None]:

        await self.ensure_loaded()
        if self.client is None:
            yield f"Error: {self.model_error or 'LLM unavailable'}"
            return

        system_prompt, user_prompt, sanitized_history = _build_coach_prompts(
            fen=fen,
            history_san=history_san,
            side_to_move=side_to_move,
            question=question,
            chat_history=chat_history,
            engine_eval=engine_eval,
            opening_info=opening_info,
        )
        tokens = max_tokens or self.max_output_tokens
        messages = [
            {"role": "system", "content": system_prompt},
            *sanitized_history,
            {"role": "user", "content": user_prompt},
        ]

        async with self.lock:
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[str] = asyncio.Queue()
            finished = asyncio.Event()
            use_responses = self._use_responses_api()
            COACH_LOGGER.info(
                "OpenAI coach request model=%s responses_api=%s tokens=%s history_turns=%s",
                self.model_label,
                use_responses,
                tokens,
                len(sanitized_history),
            )

            def _enqueue(text: str) -> None:
                fut = asyncio.run_coroutine_threadsafe(queue.put(text), loop)
                fut.result()

            def _mark_finished() -> None:
                loop.call_soon_threadsafe(finished.set)

            def _worker() -> None:
                try:
                    if use_responses:
                        responses_input = self._build_responses_input(
                            system_prompt, sanitized_history, user_prompt
                        )
                        resp_kwargs: Dict[str, Any] = {
                            "model": self.model_label,
                            "max_output_tokens": tokens,
                            "input": responses_input,
                        }
                        # some of the cutting-edge models (gpt-5/o-series) reject temperature;
                        # omit it entirely in that case.
                        if not self.model_label.lower().startswith("gpt-5"):
                            resp_kwargs["temperature"] = self.temperature
                        resp = self.client.responses.create(**resp_kwargs)
                        COACH_LOGGER.info("DEBUG RESP DIR: %s", dir(resp))
                        COACH_LOGGER.info("Responses API call complete: %s", getattr(resp, "id", "n/a"))
                        chunks: List[str] = []
                        extra = getattr(resp, "output_text", None)
                        if isinstance(extra, list):
                            chunks.extend(extra)
                        elif isinstance(extra, str):
                            chunks.append(extra)
                        output = getattr(resp, "output", None)
                        if output:
                            for item in output:
                                contents = getattr(item, "content", None)
                                if not contents:
                                    continue
                                for content in contents:
                                    ctype = getattr(content, "type", "")
                                    if ctype not in {"output_text", "text", "summary_text"}:
                                        continue
                                    text_obj = getattr(content, "text", None)
                                    text_val: Optional[str] = None
                                    if isinstance(text_obj, str):
                                        text_val = text_obj
                                    elif hasattr(text_obj, "value"):
                                        text_val = getattr(text_obj, "value", None)
                                    elif isinstance(text_obj, dict):
                                        text_val = text_obj.get("value") or text_obj.get("text")
                                    if not text_val and hasattr(content, "summary"):
                                        summary_obj = getattr(content, "summary")
                                        if isinstance(summary_obj, str):
                                            text_val = summary_obj
                                        elif hasattr(summary_obj, "value"):
                                            text_val = getattr(summary_obj, "value", None)
                                        elif isinstance(summary_obj, dict):
                                            text_val = summary_obj.get("value") or summary_obj.get("text")
                                    if text_val:
                                        chunks.append(text_val)
                        if not chunks:
                            msg = "[No output returned by OpenAI]"
                            chunks.append(msg)
                            COACH_LOGGER.warning("%s Raw response: %s", msg, resp)
                        for idx, piece in enumerate(chunks):
                            COACH_LOGGER.info("Enqueue chunk %s len=%s repr=%r", idx, len(piece), piece[:80])
                            _enqueue(piece)
                    else:
                        # check if we should disable streaming for stability with reasoning models
                        is_reasoning = self.model_label.lower().startswith(("o1", "gpt-5"))
                        
                        req_kwargs = {
                            "model": self.model_label,
                            "messages": messages,
                        }

                        if is_reasoning:
                            # reasoning models often fail to stream or have high latency. 
                            # we force stream=false to ensure we get a response.
                            req_kwargs["stream"] = False
                            req_kwargs["max_completion_tokens"] = tokens
                        else:
                            req_kwargs["stream"] = True
                            req_kwargs["max_tokens"] = tokens
                            req_kwargs["temperature"] = self.temperature

                        # make the api call
                        response = self.client.chat.completions.create(**req_kwargs)

                        if is_reasoning:
                            # non-streaming response handling
                            # the response is a single object, not an iterator
                            msg = response.choices[0].message
                            content = msg.content
                            # check for reasoning_content if available (sometimes hidden in extra_fields)
                            # but usually just 'content' is what we want for the ui.
                            if content:
                                _enqueue(content)
                            else:
                                COACH_LOGGER.warning("Reasoning model returned no content: %s", response)
                        else:
                            # standard streaming for other models (gpt-4o-mini, etc)
                            for chunk in response:
                                try:
                                    choice = chunk.choices[0]
                                    delta = getattr(choice, "delta", None)
                                except (AttributeError, IndexError):
                                    continue
                                
                                if not delta:
                                    continue

                                # try to grab content or reasoning
                                content = getattr(delta, "content", None)
                                reasoning = getattr(delta, "reasoning_content", None)

                                if reasoning:
                                    _enqueue(reasoning)
                                if content:
                                    _enqueue(content)
                                
                except Exception as e:
                    COACH_LOGGER.exception("OpenAI coach call failed")
                    _enqueue(f"\n[Analysis failed: {type(e).__name__}: {e}]")
                finally:
                    _mark_finished()

            threading.Thread(target=_worker, daemon=True).start()

            while True:
                if finished.is_set() and queue.empty():
                    break
                try:
                    chunk = await asyncio.wait_for(queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
                COACH_LOGGER.debug("Yield chunk len=%s", len(chunk))
                yield chunk

            await finished.wait()


if COACH_PROVIDER not in {"local", "openai"}:
    COACH_PROVIDER = "local"

if COACH_PROVIDER == "openai":
    coach = OpenAIChatCoach(
        model_name=OPENAI_MODEL,
        max_output_tokens=LLM_MAX_OUTPUT_TOKENS,
        temperature=LLM_TEMPERATURE,
        api_key=OPENAI_API_KEY,
    )
else:
    coach = ChessCoachLLM(LLM_MODEL_PATH, LLM_CTX_TOKENS, LLM_MAX_OUTPUT_TOKENS, LLM_TEMPERATURE)

# lifespan
@app.on_event("startup")
async def _on_startup():
    # eager-start the engine so it's ready immediately.
    await engine.ensure_started()
    if LLM_EAGER_START:
        await coach.ensure_loaded()


@app.on_event("shutdown")
async def _on_shutdown():
    await engine.stop()

# models
class MoveReq(BaseModel):
    uci: str = Field(..., description="UCI move, e.g., e2e4 or e7e8q")
    ply: Optional[int] = Field(None, description="Client's expected ply (len(move_stack))")
    req_id: Optional[str] = Field(None, description="Idempotency key (unique per attempt)")


class CoachReq(BaseModel):
    question: Optional[str] = Field(None, description="Optional user question for the LLM coach", max_length=400)
    history: Optional[List[Dict[str, str]]] = Field(None, description="Previous chat history")
    max_tokens: Optional[int] = Field(None, ge=128, le=1024, description="Override response token budget")

# routes: state & controls
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

# routes: engine status / eval
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
    # snapshot fen under lock
    async with BOARD_LOCK:
        fen = board.fen()

    # try, restart, and retry once on known engine failures (incl. notimplementederror)
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

# routes: llm coach
@app.get("/coach/status")
async def coach_status(start: int = Query(0, description="If 1, try to load the LLM coach")):
    if start:
        await coach.ensure_loaded()
    status = {
        "provider": COACH_PROVIDER,
        "ready": coach.is_ready(),
        "error": coach.model_error,
        "path": coach.path,
        "exists": coach.exists(),
        "model": coach.model_label,
        "ctx_tokens": LLM_CTX_TOKENS,
        "max_output_tokens": LLM_MAX_OUTPUT_TOKENS,
    }
    return JSONResponse(status)


@app.post("/coach/analyze")
async def coach_analyze(req: CoachReq):
    async with BOARD_LOCK:
        fen = board.fen()
        history = _history_san()
        stm = "White" if board.turn else "Black"
        moves_snapshot = list(board.move_stack)

    opening_info = _detect_opening_from_moves(moves_snapshot)

    engine_eval: Optional[Dict[str, Any]] = None
    try:
        engine_eval = await engine.analyse_fen(fen, depth=12)
    except Exception as e:
        engine_eval = {"error": f"{type(e).__name__}: {e}"}

    # we return a streamingresponse that iterates over the generator
    return StreamingResponse(
        coach.stream_analysis(
            fen=fen,
            history_san=history,
            side_to_move=stm,
            question=req.question,
            chat_history=req.history,
            max_tokens=req.max_tokens,
            engine_eval=engine_eval,
            opening_info=opening_info,
        ),
        media_type="text/plain"
    )

# route: move (ply-checked, idempotent)
@app.post("/move")
async def move(req: MoveReq):
    """
    Apply a move if it matches the client's view of the position (ply check).
    Reject stale or duplicate requests explicitly.
    """
    rid = req.req_id or str(uuid.uuid4())
    now = time.time()

    # fast duplicate check
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

# root
@app.get("/")
async def root():
    return JSONResponse({"ok": True, "engine": engine.snapshot_status_fields()})

# uvicorn entry
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", reload=False, host="127.0.0.1", port=8000)