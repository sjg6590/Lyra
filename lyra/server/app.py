import base64
import asyncio
import json
import os
import threading
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from lyra.server.agent import LyraAgentEngine, format_sse
from lyra.server.ollama_client import OllamaClient
from lyra.server.rolling_memory import RollingMemoryEngine
from lyra.server.search import WebSearchEngine
from lyra.server.speaker_id import TargetSpeakerExtractor
from lyra.server.vad import VoiceActivityDetector


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    threading.Thread(target=_warmup_models, name="lyra-warmup", daemon=True).start()
    yield


app = FastAPI(title="Lyra Ambient Assistant API", version="1.0.0", lifespan=_lifespan)
# Load configuration
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.json")
config = {
    "sample_rate": 16000,
    "vad_threshold": 0.015,
    "similarity_threshold": 0.70,
    "agent": {
        "name": "Lyra",
        "persona": "Jarvis-style intelligent, ambient personal assistant. Concise, sharp, proactive, and contextually aware.",
        "web_search_enabled": False,
        "max_search_results": 4,
        "ollama": {
            "enabled": True,
            "host": "http://127.0.0.1:11434",
            "model": "qwen3.5:4b-mlx",
            "think": False,
            "num_ctx": 2048,
            "num_predict": 96,
            "temperature": 0.4,
            "keep_alive": -1,
            "timeout_seconds": 90,
        },
    },
}
memory_cfg = {
    "rolling_buffer_max_minutes": 30,
    "max_episodic_entries": 1000,
    "context_window_turns": 8,
}
qdrant_cfg = {
    "url": "http://localhost:6333",
    "collection": "lyra_episodic",
    "embedding_model": "google/embeddinggemma-300m",
    "sparse_model": "Qdrant/bm25",
    "vector_size": 768,
}

if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r") as f:
            cfg_data = json.load(f)
            config["sample_rate"] = cfg_data.get("audio", {}).get("sample_rate", 16000)
            config["vad_threshold"] = cfg_data.get("audio", {}).get("vad_energy_threshold", 0.015)
            config["similarity_threshold"] = cfg_data.get("audio", {}).get("speaker_similarity_threshold", 0.70)
            memory_cfg.update(cfg_data.get("memory", {}))
            nested_qdrant = memory_cfg.pop("qdrant", None) or cfg_data.get("memory", {}).get("qdrant")
            if nested_qdrant:
                qdrant_cfg.update(nested_qdrant)
            if isinstance(cfg_data.get("agent"), dict):
                config["agent"] = {**config["agent"], **cfg_data["agent"]}
                nested_ollama = cfg_data["agent"].get("ollama")
                if isinstance(nested_ollama, dict):
                    config["agent"]["ollama"] = {
                        **config["agent"].get("ollama", {}),
                        **nested_ollama,
                    }
    except Exception as e:
        print(f"[Server] Error reading config: {e}")

agent_cfg = config["agent"]
ollama_cfg = agent_cfg.get("ollama") if isinstance(agent_cfg.get("ollama"), dict) else {}
ollama_client = OllamaClient.from_config(ollama_cfg)

# Instantiate Core Engines
vad_detector = VoiceActivityDetector(sample_rate=config["sample_rate"], energy_threshold=config["vad_threshold"])
speaker_extractor = TargetSpeakerExtractor(profile_path="user_voice_profile.json", similarity_threshold=config["similarity_threshold"])
memory_engine = RollingMemoryEngine(
    max_buffer_minutes=int(memory_cfg.get("rolling_buffer_max_minutes", 30)),
    max_episodic_entries=int(memory_cfg.get("max_episodic_entries", 1000)),
    qdrant_url=str(qdrant_cfg.get("url", "http://localhost:6333")),
    qdrant_collection=str(qdrant_cfg.get("collection", "lyra_episodic")),
    embedding_model=str(qdrant_cfg.get("embedding_model", "google/embeddinggemma-300m")),
    sparse_model=str(qdrant_cfg.get("sparse_model", "Qdrant/bm25")),
    vector_size=int(qdrant_cfg.get("vector_size", 768)),
)
search_engine = WebSearchEngine(max_results=int(agent_cfg.get("max_search_results", 4)))
agent_engine = LyraAgentEngine(
    name=agent_cfg.get("name", "Lyra"),
    persona=agent_cfg.get("persona"),
    search_engine=search_engine,
    ollama_client=ollama_client,
    web_search_enabled=bool(agent_cfg.get("web_search_enabled", False)),
    context_window_turns=int(memory_cfg.get("context_window_turns", 8)),
)

if ollama_client is None:
    print("[Server] Ollama disabled in config; using heuristic response synthesizer.")
elif ollama_client.is_reachable():
    model_ok = ollama_client.has_model()
    print(
        f"[Server] Ollama reachable at {ollama_client.host}; "
        f"model '{ollama_client.model}' {'found' if model_ok else 'NOT FOUND — run: ollama pull ' + ollama_client.model}."
    )
else:
    print(
        f"[Server] Ollama not reachable at {ollama_client.host}; "
        "tap-to-talk will fall back to heuristic responses until Ollama is running."
    )


def _warmup_models() -> None:
    """Load Ollama MLX weights + EmbeddingGemma so the first tap is warm."""
    print("[Server] Warming models in background...")
    try:
        if ollama_client is not None and ollama_client.is_reachable():
            ollama_client.warm()
            print(f"[Server] Ollama model '{ollama_client.model}' warmed (keep_alive={ollama_client.keep_alive}).")
        else:
            print("[Server] Skipping Ollama warm-up (unreachable or disabled).")
    except Exception as e:
        print(f"[Server] Ollama warm-up failed: {e}")

    try:
        store = getattr(memory_engine, "episodic_store", None)
        embedder = getattr(store, "embedder", None) if store is not None else None
        if embedder is not None and hasattr(embedder, "embed_query"):
            embedder.embed_query("warmup")
            print("[Server] Embedding models warmed.")
        else:
            print("[Server] Skipping embedder warm-up (no embedder available).")
    except Exception as e:
        print(f"[Server] Embedder warm-up failed: {e}")


class TapToTalkRequest(BaseModel):
    query: str
    force_search: bool = False


class VoiceEnrollRequest(BaseModel):
    user_name: str = "User"
    audio_base64: str


# Mount static files
STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def get_dashboard():
    """Serves the Lyra Command Deck UI."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Lyra Ambient Personal Assistant</h1><p>Dashboard initializing...</p>")


@app.get("/api/status")
def get_system_status():
    """Returns server and engine status."""
    backend = memory_engine.backend_status()
    return {
        "status": "online",
        "timestamp": time.time(),
        "enrolled": speaker_extractor.enrolled_profile is not None,
        "enrolled_user": speaker_extractor.enrolled_metadata.get("user_name", "None"),
        "rolling_memory_entries": len(memory_engine.rolling_buffer),
        "episodic_memory_entries": memory_engine.episodic_count(),
        "episodic_backend": backend.get("episodic_backend", "qdrant"),
        "qdrant": backend.get("qdrant", {}),
        "config": config,
    }


@app.post("/api/tap_to_talk")
def tap_to_talk_handler(req: TapToTalkRequest):
    """
    HTTP trigger endpoint for user Tap-to-Talk query with ambient context synthesis.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    result = agent_engine.process_tap_to_talk(
        query=req.query,
        memory_engine=memory_engine,
        force_search=req.force_search,
    )
    return result


@app.post("/api/tap_to_talk/stream")
async def tap_to_talk_stream_handler(req: TapToTalkRequest):
    """SSE stream of tap-to-talk: status → token* → done|error.

    Uses an async generator that pulls sync Ollama/agent events via to_thread
    so each SSE frame is flushed to the client immediately (avoids buffering the
    whole reply until generation finishes).
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    stream_iter = agent_engine.process_tap_to_talk_stream(
        query=req.query,
        memory_engine=memory_engine,
        force_search=req.force_search,
    )
    sentinel = object()

    async def event_generator():
        while True:
            event = await asyncio.to_thread(next, stream_iter, sentinel)
            if event is sentinel:
                break
            yield format_sse(event)
            # Yield control so uvicorn can flush the chunk to the socket.
            await asyncio.sleep(0)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/enroll_voice")
def enroll_voice_profile(req: VoiceEnrollRequest):
    """
    Enrolls user voice baseline profile from raw float audio array or base64 WAV.
    """
    try:
        audio_bytes = base64.b64decode(req.audio_base64)
        audio_array = np.frombuffer(audio_bytes, dtype=np.float32)

        if len(audio_array) == 0:
            # Fallback mock array if testing with raw audio
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        success = speaker_extractor.enroll_user(audio_array, user_name=req.user_name, sample_rate=config["sample_rate"])
        return {
            "success": success,
            "message": f"Successfully enrolled voice profile for '{req.user_name}'",
            "enrolled": True,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enrollment failed: {e!s}")


@app.get("/api/memory")
def get_memory_log():
    """Returns stored rolling & episodic conversation memories."""
    return {
        "rolling_buffer": list(memory_engine.rolling_buffer),
        "episodic_memory": memory_engine.episodic_memory,
    }


@app.delete("/api/memory")
def clear_memory_log():
    """Clears stored memories."""
    memory_engine.clear_memory()
    return {"message": "Memory successfully cleared."}


@app.websocket("/ws/ambient")
async def ambient_audio_stream(websocket: WebSocket):
    """
    WebSocket endpoint for continuous background ambient audio streaming from mobile/browser client.
    Performs VAD, Target Speaker Extraction, ASR transcript append, and live UI updates.
    """
    await websocket.accept()
    speaker_extractor.clear_stream_history()
    print("[WebSocket] Client connected for continuous ambient streaming.")

    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            msg_type = payload.get("type", "audio_chunk")

            if msg_type == "audio_chunk":
                raw_audio = payload.get("audio", [])
                text_transcript = payload.get("transcript", "")
                sample_rate = payload.get("sample_rate", config["sample_rate"])

                if isinstance(raw_audio, list):
                    audio_array = np.array(raw_audio, dtype=np.float32)
                else:
                    audio_array = np.zeros(1024, dtype=np.float32)

                vad_result = vad_detector.is_speech(audio_array)
                speaker_info = speaker_extractor.identify_speaker(audio_array, sample_rate=sample_rate)

                transcript_entry = None
                if vad_result["is_speech"] and text_transcript.strip():
                    speaker_tag = speaker_info["speaker_id"]
                    is_user = speaker_info["is_user"]

                    transcript_entry = memory_engine.add_transcript(
                        speaker=speaker_tag,
                        text=text_transcript,
                        confidence=speaker_info["confidence"],
                        is_user=is_user,
                    )

                await websocket.send_json({
                    "type": "stream_update",
                    "vad": vad_result,
                    "speaker": speaker_info,
                    "transcript_entry": transcript_entry,
                    "rolling_count": len(memory_engine.rolling_buffer),
                })

            elif msg_type == "tap_to_talk":
                query = payload.get("query", "")
                force_search = bool(payload.get("force_search", False))
                for event in agent_engine.process_tap_to_talk_stream(
                    query=query,
                    memory_engine=memory_engine,
                    force_search=force_search,
                ):
                    await websocket.send_json({"type": "tap_stream", "event": event})
                    if event.get("event") in ("done", "error"):
                        break

    except WebSocketDisconnect:
        speaker_extractor.clear_stream_history()
        print("[WebSocket] Client disconnected.")
    except Exception as e:
        speaker_extractor.clear_stream_history()
        print(f"[WebSocket] Stream error: {e}")
