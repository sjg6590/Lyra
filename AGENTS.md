# AGENTS.md

## Cursor Cloud specific instructions

Lyra is a single Python FastAPI service (an ambient "Jarvis-style" assistant). There is one backend process; the browser UI and CLI are thin clients. See `architecture_guide.md` for the full design.

### Environment
- Python dependencies are installed into a virtualenv at `.venv/` (kept out of git). The update script (re)creates it and installs `requirements.txt` plus `pytest` (needed by the test suite) and `beautifulsoup4` (used only by the optional web-search HTML fallback in `lyra/server/search.py`). Neither `pytest` nor `beautifulsoup4` is listed in `requirements.txt`.
- Always run tooling via the venv, e.g. `.venv/bin/python`, `.venv/bin/pytest`, `.venv/bin/uvicorn`.

### Run
- Dev server (hot reload): `.venv/bin/python -m uvicorn lyra.server.app:app --host 0.0.0.0 --port 8000 --reload`
- Command Deck UI: open `http://localhost:8000`.
- CLI client (server must be running): `.venv/bin/python -m lyra.client.cli_streamer`

### Test / lint
- Tests: `.venv/bin/python -m pytest tests/` (4 tests for the speaker-ID engine).
- No linter/formatter is configured in the repo (no ruff/flake8/black config). Use `.venv/bin/python -m compileall lyra tests` for a basic syntax check.

### Non-obvious gotchas
- The voice profile is read from `user_voice_profile.json` at the current working directory (relative path in `lyra/server/app.py`), so `/api/status` reports `enrolled: true, user: Shaun` when started from the repo root. Start the server from the repo root.
- The agent (`lyra/server/agent.py`) is rule/template-based — there is NO real LLM. Responses are heuristic, so "Understood. I have logged your context..." is expected default output.
- Tap-to-talk (`POST /api/tap_to_talk`, and the UI "Send"/orb) works purely over HTTP with a typed query — no microphone needed. Mic capture, browser ASR (Web Speech API), and TTS only work in a real browser with mic permission and are not available headless.
- Web search is off by default (`config.json` `web_search_enabled: false`) and outbound DuckDuckGo requests may be blocked in the sandbox; the agent tolerates 0 search results.
- Ambient transcripts are only stored when server-side VAD flags the chunk as speech; sending flat/low-energy audio over `/ws/ambient` yields `is_speech: false` and nothing is stored.
