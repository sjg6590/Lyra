# Lyra Architecture & Technical Guide
## Ambient-Aware Personal Assistant ("Jarvis" Paradigm)

---

## 1. System Topology Overview

Lyra is designed around a **Thin Client / Heavy Server** split to solve thermal, battery, and compute bottlenecks on mobile devices:

```
┌──────────────────────────────────────────────────────────┐
│ Client (Browser Command Deck and/or ambient_capture)     │
│  - Browser: mic via WebAudio (UI + enrollment)           │
│  - Native: mic + system/loopback mix (calls via BlackHole)│
│  - Streams PCM over WebSocket to server                  │
│  - Tap / Spacebar trigger for Jarvis answers             │
└───────────────────────────┬──────────────────────────────┘
                            │ Continuous Audio Stream
                            ▼
┌──────────────────────────────────────────────────────────┐
│ Server (Processing Hub)                                  │
│  - Low-Power VAD + utterance buffering                   │
│  - faster-whisper ASR (+ optional Ollama cleanup)        │
│  - Target Speaker Extraction (ECAPA Me / Not Me)         │
│  - Rolling Sliding Window Buffer (Last 30 mins)          │
│  - Episodic Memory Vector Store (Qdrant + EmbeddingGemma)│
│  - Agent Execution Engine + Live Web Search Tool        │
│  - Text-to-Speech (TTS) Streaming Generator              │
└──────────────────────────────────────────────────────────┘
```

Episodic RAG uses **Qdrant** hybrid search: dense vectors from FastEmbed `google/embeddinggemma-300m` plus sparse BM25 (`Qdrant/bm25`), fused with Reciprocal Rank Fusion (RRF).

---

## 2. Bandwidth & Network Calculations

* **Audio Codec:** Opus at 16 kHz Mono, 24 kbps.
* **Network Throughput:** ~3 KB/sec = ~10.8 MB per hour of active ambient speech.
* **VPN Transport:** WireGuard or Tailscale overlay network ensuring zero port forwarding exposure and persistent connection switching across Wi-Fi and Cellular networks.

---

## 3. Target Speaker Diarization Pipeline ("Me" vs "Not Me")

1. **Enrollment Phase:**
   - User reads a predetermined ~60s phonetically varied script in a **natural conversational voice** (Command Deck).
   - Browser ASR (when available) tracks word coverage against the expected prompt; the server rejects takes that are too short (< 45s) or below the coverage floor when a transcript is provided.
   - Lyra embeds speech-active 1.0s windows with a pretrained **WeSpeaker ECAPA-TDNN** ONNX model (`wespeaker-ecapa512`, 192-D, via `speakeronnx` / ONNX Runtime — no Torch).
   - Builds a global mean centroid plus a **farthest-point** diverse prototype set (capped at 12) to avoid overfitting to reading cadence.
   - Persists `user_voice_profile.json` with `model_id`, `feature_dim`, `prototype_strategy`, prompt metadata, and prototype vectors. Re-enroll after matcher upgrades; legacy handcrafted 32-D profiles are rejected.
2. **Identification Phase:**
   - Incoming speech frames update a **2.0s** ring buffer (non-speech freezes scoring; ~0.5s warm-up before External is allowed).
   - Score = `0.65 * cosine(global) + 0.35 * max(prototype cosines)`, EMA-smoothed.
   - Hysteresis: enter `User [Me]` at **≥ 0.28**; leave User only when **≤ 0.18**.

First server boot downloads the ECAPA ONNX weights into the HuggingFace cache (network required once).

---

## 4. Context-Aware Tap-to-Talk Execution Loop

```
User Taps Earbud / Button
       │
       ▼
Sends `TRIGGER_EVENT` to Server
       │
       ▼
Assembles Prompt Context:
  ├── Rolling Context Window (Last 15 minutes of ambient transcripts)
  ├── Episodic Memory Retrieval (Top K vector matches from past days)
  └── User Query + Live Web Search Snippets (DuckDuckGo Search Tool)
       │
       ▼
Agent Synthesizes Concise Response
       │
       ▼
Streams Audio Response via Fast Local TTS (Sub-800ms Latency)
```

---

## 5. Software Stack & File Map

| Path | Purpose |
| --- | --- |
| [lyra/server/app.py](lyra/server/app.py) | FastAPI WebSockets server & REST API |
| [lyra/server/vad.py](lyra/server/vad.py) | Low-power Voice Activity Detector |
| [lyra/server/utterance_buffer.py](lyra/server/utterance_buffer.py) | VAD utterance accumulation for server ASR |
| [lyra/server/asr.py](lyra/server/asr.py) | faster-whisper ambient speech-to-text |
| [lyra/server/transcript_cleanup.py](lyra/server/transcript_cleanup.py) | Ollama / heuristic ASR cleanup |
| [lyra/server/speaker_id.py](lyra/server/speaker_id.py) | Target Speaker Extractor & Voice Biometrics (ECAPA) |
| [lyra/server/speaker_embedder.py](lyra/server/speaker_embedder.py) | WeSpeaker ECAPA ONNX embedding backend |
| [lyra/server/enrollment_prompt.py](lyra/server/enrollment_prompt.py) | Predetermined enrollment script + coverage |
| [lyra/server/rolling_memory.py](lyra/server/rolling_memory.py) | Rolling sliding buffer & Qdrant episodic RAG |
| [lyra/server/qdrant_memory.py](lyra/server/qdrant_memory.py) | Qdrant store + EmbeddingGemma/BM25 FastEmbed hybrid |
| [lyra/server/agent.py](lyra/server/agent.py) | Jarvis LLM core & tool execution |
| [lyra/server/ollama_client.py](lyra/server/ollama_client.py) | Local Ollama `/api/chat` client (`qwen3.5:4b-mlx`) |
| [lyra/server/search.py](lyra/server/search.py) | Live web search engine tool |
| [lyra/server/tts.py](lyra/server/tts.py) | Text-to-Speech synthesizer |
| [lyra/client/ambient_capture.py](lyra/client/ambient_capture.py) | Native mic + system/loopback capture client |
| [static/index.html](static/index.html) | Jarvis Command Deck Control UI |
| [lyra/client/cli_streamer.py](lyra/client/cli_streamer.py) | Headless CLI / Terminal client |
| [scripts/setup_ollama_mac.sh](scripts/setup_ollama_mac.sh) | Mac Ollama install + model pull |

---

## 6. Local LLM (Ollama + Qwen3.5 MLX)

Tap-to-talk responses are generated by a local Ollama model configured in `config.json`:

- **Model:** `qwen3.5:4b-mlx` (Apple Silicon MLX build, ~4.0GB, multimodal Text/Image; lighter/faster than 9B)
- **Latency defaults for M3 / 18GB:** `num_ctx: 2048`, `num_predict: 96`, `temperature: 0.4`, `think: false`, `keep_alive: -1`
- **Streaming:** Command Deck uses `POST /api/tap_to_talk/stream` (SSE) with live token paint + sentence-level TTS; non-streaming `POST /api/tap_to_talk` remains for CLI
- **Warm-up:** on server startup Lyra pings Ollama and EmbeddingGemma so the first user tap is not a cold load
- **Web search:** only runs when `agent.web_search_enabled` is true or the client sets `force_search`
- **Fallback:** if Ollama is unreachable, the agent uses the heuristic synthesizer
- **Optional quality:** set `agent.ollama.model` / `LYRA_OLLAMA_MODEL` to `qwen3.5:9b-mlx` if you prefer quality over speed

Setup on Mac:

```bash
./scripts/setup_ollama_mac.sh
# or: ollama pull qwen3.5:4b-mlx
```

Vision inputs are supported by the model but are not wired into the Command Deck yet — the agent path remains text + ambient transcript + RAG + search.

Cold first request after reboot can still exceed a few seconds while MLX weights load; subsequent warm taps target sub-3s spoken replies with streaming first tokens earlier.

---

## 7. How to Run Lyra Locally

1. **Start Qdrant (episodic vector DB):**
   ```bash
   docker compose up -d
   ```
   Qdrant listens on `http://localhost:6333` (configured in `config.json` → `memory.qdrant`).
2. **Install Python dependencies** (includes `qdrant-client` and FastEmbed from git for EmbeddingGemma; first EmbeddingGemma run downloads ~1.2 GB ONNX weights):
   ```bash
   pip install -r requirements.txt
   ```
   Note: PyPI `fastembed==0.8.0` does not yet include `google/embeddinggemma-300m`; `requirements.txt` installs FastEmbed from GitHub main where EmbeddingGemma is registered.
3. **Install & pull the LLM (Mac):**
   ```bash
   ./scripts/setup_ollama_mac.sh
   ```
4. **Start the Lyra Server:**
   ```bash
   python3 -m uvicorn lyra.server.app:app --host 0.0.0.0 --port 8000
   ```
5. **Open the Command Deck UI:**
   Navigate to `http://localhost:8000` in your web browser.
6. **Start Ambient Listening:**
   Click **"Start Continuous Ambient Listening"**.
7. **Enroll Your Voice:**
   Click **"Start Enrollment (60s)"** and read the on-screen script aloud to train your ECAPA target-speaker profile (re-enroll if you still have a legacy handcrafted profile).
8. **Tap to Talk:**
   Press the **Spacebar** or click **"TAP TO TALK"** to query Lyra with full ambient context!
