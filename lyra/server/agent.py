import re
import time
from typing import Any

from lyra.server.ollama_client import OllamaClient
from lyra.server.rolling_memory import RollingMemoryEngine
from lyra.server.search import WebSearchEngine
from lyra.server.tts import TextToSpeechEngine


class LyraAgentEngine:
    """
    Jarvis-style Personal Assistant Core Agent Engine.
    Handles context synthesis, proactive RAG retrieval, tool execution (Web Search),
    and fast response generation via local Ollama (with heuristic fallback).
    """

    def __init__(
        self,
        name: str = "Lyra",
        persona: str | None = None,
        search_engine: WebSearchEngine | None = None,
        ollama_client: OllamaClient | None = None,
        ollama_config: dict[str, Any] | None = None,
    ):
        self.name = name
        self.persona = persona or (
            "Jarvis-style intelligent, ambient personal assistant. "
            "Concise, sharp, proactive, and contextually aware."
        )
        self.search_engine = search_engine or WebSearchEngine()
        self.tts_engine = TextToSpeechEngine()
        if ollama_client is not None:
            self.ollama = ollama_client
        else:
            self.ollama = OllamaClient.from_config(ollama_config)

    def process_tap_to_talk(self, query: str, memory_engine: RollingMemoryEngine, force_search: bool = False) -> dict[str, Any]:
        """
        Executes the Tap-to-Talk context injection pipeline when triggered by earbud button or user prompt.
        """
        start_time = time.time()
        thoughts: list[str] = []

        # 1. Fetch Rolling Context Buffer
        recent_transcript_str = memory_engine.format_context_for_prompt(max_entries=15)
        thoughts.append("Fetched recent ambient rolling context window (last 15 mins).")

        # 2. Episodic RAG Memory Search
        memory_results = memory_engine.search_memory(query, top_k=3)
        if memory_results:
            thoughts.append(f"Retrieved {len(memory_results)} relevant past conversation records from Episodic RAG Memory.")
        else:
            thoughts.append("No historical episodic memory records matched the query.")

        # 3. Determine if Web Search is needed
        needs_search = force_search or self._check_needs_search(query, recent_transcript_str)
        search_results: list[dict[str, str]] = []

        if needs_search:
            search_query = self._extract_search_keywords(query, recent_transcript_str)
            thoughts.append(f"Triggered Live Web Search for: '{search_query}'")
            search_results = self.search_engine.search(search_query)
            thoughts.append(f"Retrieved {len(search_results)} live web search snippets.")

        # 4. Generate Response Answer
        response_text, used_ollama = self._synthesize_response(
            query=query,
            recent_transcript=recent_transcript_str,
            memory_results=memory_results,
            search_results=search_results,
            thoughts=thoughts,
        )

        # 5. Synthesize TTS
        tts_payload = self.tts_engine.synthesize(response_text)

        elapsed_ms = int((time.time() - start_time) * 1000)
        backend = "ollama" if used_ollama else "heuristic"
        thoughts.append(f"Response synthesized via {backend} in {elapsed_ms}ms.")

        return {
            "query": query,
            "response": response_text,
            "thoughts": thoughts,
            "ambient_context_used": recent_transcript_str,
            "memory_results": memory_results,
            "search_results": search_results,
            "tts": tts_payload,
            "latency_ms": elapsed_ms,
            "llm_backend": backend,
        }

    def _check_needs_search(self, query: str, recent_context: str) -> bool:
        """Determines if query warrants external web search."""
        query_lower = query.lower()
        search_triggers = [
            "search", "what is", "who is", "latest", "news", "weather", "price",
            "stock", "restaurant", "meaning", "definition", "how to", "score",
            "company", "location", "address", "phone number", "specs", "current"
        ]

        if any(trigger in query_lower for trigger in search_triggers):
            return True

        # Check if query references "that company", "that restaurant", "what did they say"
        if re.search(r'\b(that|the|what|who|which)\b', query_lower) and len(query.split()) > 2:
            return True

        return False

    def _extract_search_keywords(self, query: str, recent_context: str) -> str:
        """
        Derives an optimal web search query by combining user query with ambient context entities.
        """
        # If query specifically asks "What was that company John mentioned?", extract entities from context
        clean_query = re.sub(r'^(lyra|hey lyra|jarvis|hey jarvis|please|can you|what is|search for)\b', '', query, flags=re.IGNORECASE).strip()

        # Find recent capitalized terms / entities in context
        entities = re.findall(r'\b[A-Z][a-zA-Z0-9]+\b', recent_context)
        filtered_entities = [e for e in entities if e not in ["User", "Me", "External", "Speaker", "Lyra", "Jarvis"]]

        if filtered_entities and ("that" in query.lower() or "they" in query.lower()):
            search_query = f"{clean_query} {' '.join(filtered_entities[-2:])}"
        else:
            search_query = clean_query if clean_query else query

        return search_query.strip()

    def build_system_prompt(
        self,
        recent_transcript: str,
        memory_results: list[dict[str, Any]],
        search_results: list[dict[str, str]],
    ) -> str:
        """Assemble the system prompt for Ollama from persona + ambient/RAG/search context."""
        memory_block = "(No episodic memory matches)"
        if memory_results:
            lines = []
            for item in memory_results:
                lines.append(
                    f"- [{item.get('readable_time', '?')}] {item.get('speaker', '?')}: "
                    f"\"{item.get('text', '')}\""
                )
            memory_block = "\n".join(lines)

        search_block = self.search_engine.format_search_for_prompt(search_results)

        return (
            f"You are {self.name}, {self.persona}\n\n"
            "Respond as a spoken voice assistant: clear, helpful, and concise "
            "(usually 1–3 short sentences). Do not narrate your reasoning. "
            "Use the ambient transcript, episodic memory, and web search evidence when relevant. "
            "If evidence is missing, say so briefly instead of inventing facts.\n\n"
            f"## Recent ambient transcript\n{recent_transcript}\n\n"
            f"## Episodic memory matches\n{memory_block}\n\n"
            f"## Live web search\n{search_block}"
        )

    def _synthesize_response(
        self,
        query: str,
        recent_transcript: str,
        memory_results: list[dict[str, Any]],
        search_results: list[dict[str, str]],
        thoughts: list[str] | None = None,
    ) -> tuple[str, bool]:
        """
        Synthesizes a Jarvis-style voice response via Ollama, with heuristic fallback.
        Returns (response_text, used_ollama).
        """
        thoughts = thoughts if thoughts is not None else []

        if self.ollama is not None:
            try:
                system_prompt = self.build_system_prompt(
                    recent_transcript=recent_transcript,
                    memory_results=memory_results,
                    search_results=search_results,
                )
                thoughts.append(f"Calling Ollama model '{self.ollama.model}'.")
                text = self.ollama.chat(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query},
                    ]
                )
                return text, True
            except Exception as e:
                thoughts.append(f"Ollama unavailable ({e}); falling back to heuristic synthesizer.")

        return self._heuristic_synthesize(query, recent_transcript, memory_results, search_results), False

    def _heuristic_synthesize(
        self,
        query: str,
        recent_transcript: str,
        memory_results: list[dict[str, Any]],
        search_results: list[dict[str, str]],
    ) -> str:
        """Rule/keyword synthesizer used when Ollama is disabled or unreachable."""
        q_lower = query.lower()

        # Contextual question referencing recent speech
        if "restaurant" in q_lower or "company" in q_lower or "person" in q_lower or "mentioned" in q_lower or "say" in q_lower:
            if memory_results:
                top_mem = memory_results[0]
                return f"Based on your recent conversation at {top_mem['readable_time']}, {top_mem['speaker']} mentioned: \"{top_mem['text']}\"."

            if search_results:
                top_s = search_results[0]
                return f"According to live search, {top_s['title']} is described as: {top_s['snippet']}"

            if recent_transcript and "(No recent ambient" not in recent_transcript:
                return f"Scanning your recent audio log: The latest context captured is: {recent_transcript.splitlines()[-1]}"

        # Search-based response
        if search_results:
            top_s = search_results[0]
            answer = f"{top_s['title']}: {top_s['snippet']}"
            if len(search_results) > 1:
                answer += f" Additionally, {search_results[1]['title']} notes {search_results[1]['snippet']}"
            return answer[:350]

        # Memory lookup response
        if memory_results:
            top = memory_results[0]
            return f"I recalled from your memory log at {top['readable_time']} ({top['speaker']}): \"{top['text']}\"."

        # General conversational response
        return "Understood. I have logged your context. How else can I assist with this conversation?"
