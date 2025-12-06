# Chess State Server

FastAPI service that tracks a single chess board, runs Stockfish for evaluations, and streams coaching advice from either a local llama.cpp model or the OpenAI API (switch via `.env`). The backend exposes a minimal REST surface—state, move control, engine eval, and coach streaming—used by the frontend to review games. All configuration (model paths, API keys, ports) lives in environment variables; train scripts for custom LoRA coaches and dataset helpers are kept alongside the server.
