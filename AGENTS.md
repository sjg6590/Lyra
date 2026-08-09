# AGENTS.md

## Cursor Cloud specific instructions

Lyra is a single Python FastAPI service (an ambient "Jarvis-style" assistant). There is one backend process; the browser UI and CLI are thin clients. See `architecture_guide.md` for the full design.

### Environment
- Python dependencies are installed into a virtualenv at `.venv/` (kept out of git). The update script (re)creates it and installs `requirements.txt` plus `pytest` (needed by the test suite) and `beautifulsoup4` (used only by the optional web-search HTML fallback in `lyra/server/search.py`). Neither `pytest` nor `beautifulsoup4` is listed in `requirements.txt`.
- Always run tooling via the venv, e.g. `.venv/bin/python`, `.venv/bin/pytest`, `.venv/bin/uvicorn`.

### Run
- Start Qdrant first when you want durable episodic memory: `docker compose up -d`
- Dev server (hot reload): `.venv/bin/python -m uvicorn lyra.server.app:app --host 0.0.0.0 --port 8000 --reload`
- Command Deck UI: open `http://localhost:8000`.
- CLI client (server must be running): `.venv/bin/python -m lyra.client.cli_streamer`

### Test / lint
- Tests: `.venv/bin/python -m pytest tests/`
- No linter/formatter is configured in the repo (no ruff/flake8/black config). Use `.venv/bin/python -m compileall lyra tests` for a basic syntax check.

### Non-obvious gotchas
- The voice profile is read from `user_voice_profile.json` at the current working directory (relative path in `lyra/server/app.py`). Start the server from the repo root.
- Tap-to-talk uses local Ollama (`qwen3.5:4b-mlx` by default) when reachable; otherwise it falls back to the heuristic synthesizer.
- Latency knobs live in `config.json` (`num_ctx: 2048`, `num_predict: 96`, `keep_alive: -1`). The UI streams via `/api/tap_to_talk/stream`.
- Episodic RAG uses Qdrant + EmbeddingGemma/BM25. If Qdrant is down, the server falls back to an in-process `:memory:` store (non-durable).
- Tap-to-talk (`POST /api/tap_to_talk`, SSE `/api/tap_to_talk/stream`, and the UI "Send"/orb) works with a typed query — no microphone needed. Mic capture, browser ASR (Web Speech API), and TTS only work in a real browser with mic permission and are not available headless.
- Web search is off by default (`config.json` `web_search_enabled: false`) and is now enforced by the agent; outbound DuckDuckGo requests may be blocked in the sandbox.
- Ambient transcripts are only stored when server-side VAD flags the chunk as speech; sending flat/low-energy audio over `/ws/ambient` yields `is_speech: false` and nothing is stored.
