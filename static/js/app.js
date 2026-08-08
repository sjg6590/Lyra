/**
 * Lyra Ambient Personal Assistant Client Application
 * Continuous Ambient Audio Streaming, Target Speaker Extraction, Tap-to-Talk Jarvis Trigger, and Speech Synthesis.
 */

let ws = null;
let audioContext = null;
let mediaStream = null;
let scriptProcessor = null;
let speechRecognizer = null;

let isStreaming = false;
let isEnrolling = false;
let lastTranscriptChunk = "";

// Canvas Visualizer
let visualizerCanvas = null;
let visualizerCtx = null;

document.addEventListener("DOMContentLoaded", () => {
    initUI();
    initWebSocket();
    initVisualizer();
    checkServerStatus();
    setupKeyListeners();
});

function initUI() {
    document.getElementById("btn-toggle-mic").addEventListener("click", toggleAmbientStream);
    document.getElementById("btn-tap-to-talk").addEventListener("click", triggerTapToTalk);
    document.getElementById("btn-send-query").addEventListener("click", () => {
        const query = document.getElementById("input-explicit-query").value;
        if (query) triggerTapToTalkWithQuery(query);
    });
    document.getElementById("input-explicit-query").addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            const query = e.target.value;
            if (query) triggerTapToTalkWithQuery(query);
        }
    });

    document.getElementById("btn-enroll-voice").addEventListener("click", startVoiceEnrollment);
    document.getElementById("btn-clear-memory").addEventListener("click", clearMemory);
    document.getElementById("btn-search-memory").addEventListener("click", searchEpisodicMemory);
}

function setupKeyListeners() {
    window.addEventListener("keydown", (e) => {
        // Spacebar trigger when not focused on an input field
        if (e.code === "Space" && document.activeElement.tagName !== "INPUT" && document.activeElement.tagName !== "TEXTAREA") {
            e.preventDefault();
            triggerTapToTalk();
        }
    });
}

function initWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/ambient`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log("[Lyra WS] Connected to server ambient audio stream.");
        document.getElementById("server-status-text").innerText = "Connected";
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.type === "stream_update") {
            updateStreamMetrics(data);
        } else if (data.type === "tap_response") {
            renderAgentResponse(data.result);
        }
    };

    ws.onclose = () => {
        console.log("[Lyra WS] Disconnected. Reconnecting in 3s...");
        document.getElementById("server-status-text").innerText = "Reconnecting...";
        setTimeout(initWebSocket, 3000);
    };

    ws.onerror = (err) => {
        console.error("[Lyra WS] Error:", err);
    };
}

async function checkServerStatus() {
    try {
        const resp = await fetch("/api/status");
        const data = await resp.json();
        
        if (data.enrolled) {
            document.getElementById("enrolled-user-text").innerText = data.enrolled_user;
            document.getElementById("dot-enrolled").className = "status-dot green";
        } else {
            document.getElementById("enrolled-user-text").innerText = "Not Enrolled";
            document.getElementById("dot-enrolled").className = "status-dot yellow";
        }

        document.getElementById("memory-count-text").innerText = `${data.rolling_memory_entries} items`;
    } catch (e) {
        console.warn("[Lyra] Status check error:", e);
    }
}

async function toggleAmbientStream() {
    if (isStreaming) {
        stopAmbientStream();
    } else {
        await startAmbientStream();
    }
}

async function startAmbientStream() {
    try {
        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });

        const source = audioContext.createMediaStreamSource(mediaStream);
        scriptProcessor = audioContext.createScriptProcessor(2048, 1, 1);

        source.connect(scriptProcessor);
        scriptProcessor.connect(audioContext.destination);

        scriptProcessor.onaudioprocess = (e) => {
            if (!isStreaming) return;
            const pcmData = e.inputBuffer.getChannelData(0);
            
            // Draw visualizer frame
            drawVisualizerFrame(pcmData);

            // Send audio chunk frame to WebSocket server
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: "audio_chunk",
                    audio: Array.from(pcmData),
                    transcript: lastTranscriptChunk,
                    sample_rate: audioContext.sampleRate
                }));
            }
        };

        // Initialize Web Speech API for real-time local ASR stream
        initSpeechRecognition();

        isStreaming = true;
        document.getElementById("btn-toggle-mic").className = "btn-primary stop";
        document.getElementById("mic-btn-text").innerText = "Stop Ambient Listening";
        document.getElementById("live-indicator").innerText = "LISTENING 24/7";
        document.getElementById("live-indicator").className = "live-badge active";

    } catch (err) {
        alert("Microphone access failed: " + err.message);
    }
}

function stopAmbientStream() {
    isStreaming = false;
    if (scriptProcessor) scriptProcessor.disconnect();
    if (mediaStream) mediaStream.getTracks().forEach(track => track.stop());
    if (audioContext) audioContext.close();
    if (speechRecognizer) speechRecognizer.stop();

    document.getElementById("btn-toggle-mic").className = "btn-primary start";
    document.getElementById("mic-btn-text").innerText = "Start Continuous Ambient Listening";
    document.getElementById("live-indicator").innerText = "STANDBY";
    document.getElementById("live-indicator").className = "live-badge";
    document.getElementById("val-vad-state").innerText = "SILENT";
    document.getElementById("val-vad-state").className = "inactive";
}

function initSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        console.warn("[Lyra] SpeechRecognition API not supported in browser. Relying on server VAD.");
        return;
    }

    speechRecognizer = new SpeechRecognition();
    speechRecognizer.continuous = true;
    speechRecognizer.interimResults = true;
    speechRecognizer.lang = "en-US";

    speechRecognizer.onresult = (event) => {
        let interimText = "";
        for (let i = event.resultIndex; i < event.results.length; ++i) {
            if (event.results[i].isFinal) {
                lastTranscriptChunk = event.results[i][0].transcript;
            } else {
                interimText += event.results[i][0].transcript;
            }
        }
        if (interimText) lastTranscriptChunk = interimText;
    };

    speechRecognizer.onerror = (err) => {
        console.warn("[SpeechRec] Error:", err.error);
    };

    speechRecognizer.onend = () => {
        if (isStreaming) speechRecognizer.start();
    };

    speechRecognizer.start();
}

function updateStreamMetrics(data) {
    const vad = data.vad;
    const speaker = data.speaker;
    const entry = data.transcript_entry;

    // VAD UI
    const vadElem = document.getElementById("val-vad-state");
    if (vad.is_speech) {
        vadElem.innerText = "SPEECH DETECTED";
        vadElem.className = "active";
    } else {
        vadElem.innerText = "SILENT";
        vadElem.className = "inactive";
    }

    document.getElementById("val-rms").innerText = vad.rms.toFixed(4);
    document.getElementById("val-vad-conf").innerText = `${(vad.confidence * 100).toFixed(0)}%`;

    // Speaker Identification UI
    const speakerBadge = document.getElementById("speaker-badge");
    const speakerName = document.getElementById("speaker-name-text");
    const simBar = document.getElementById("similarity-bar");
    const simVal = document.getElementById("val-similarity");

    if (speaker.is_user) {
        speakerBadge.className = "speaker-badge user";
        speakerName.innerText = `${speaker.speaker_id}`;
    } else {
        speakerBadge.className = "speaker-badge external";
        speakerName.innerText = `${speaker.speaker_id}`;
    }

    const simScore = speaker.similarity_score || 0;
    simVal.innerText = simScore.toFixed(2);
    simBar.style.width = `${Math.max(0, Math.min(100, (simScore * 100)))}%`;

    // New Transcript Entry Append
    if (entry) {
        appendTranscriptItem(entry);
        document.getElementById("memory-count-text").innerText = `${data.rolling_count} items`;
    }
}

function appendTranscriptItem(entry) {
    const feed = document.getElementById("transcript-feed");
    const emptyMsg = feed.querySelector(".empty-feed-msg");
    if (emptyMsg) emptyMsg.remove();

    const item = document.createElement("div");
    item.className = `transcript-item ${entry.is_user ? 'user' : 'external'}`;
    item.innerHTML = `
        <div class="item-header">
            <span class="${entry.is_user ? 'speaker-tag-user' : 'speaker-tag-ext'}">${entry.speaker}</span>
            <span>${entry.readable_time}</span>
        </div>
        <div class="item-body">"${entry.text}"</div>
    `;

    feed.appendChild(item);
    feed.scrollTop = feed.scrollHeight;
}

// Visualizer
function initVisualizer() {
    visualizerCanvas = document.getElementById("audio-visualizer");
    visualizerCtx = visualizerCanvas.getContext("2d");
}

function drawVisualizerFrame(pcmData) {
    if (!visualizerCtx) return;
    const width = visualizerCanvas.width;
    const height = visualizerCanvas.height;

    visualizerCtx.fillStyle = "#050811";
    visualizerCtx.fillRect(0, 0, width, height);

    visualizerCtx.lineWidth = 2;
    visualizerCtx.strokeStyle = "#38bdf8";
    visualizerCtx.beginPath();

    const sliceWidth = width / pcmData.length;
    let x = 0;

    for (let i = 0; i < pcmData.length; i += 4) {
        const v = pcmData[i];
        const y = (v + 1) * (height / 2);

        if (i === 0) visualizerCtx.moveTo(x, y);
        else visualizerCtx.lineTo(x, y);

        x += sliceWidth * 4;
    }

    visualizerCtx.lineTo(width, height / 2);
    visualizerCtx.stroke();
}

// Voice Enrollment
async function startVoiceEnrollment() {
    const userName = document.getElementById("input-enroll-name").value || "User";
    const statusElem = document.getElementById("enroll-status");

    try {
        statusElem.innerText = "Recording 10s voice sample... Please speak naturally!";
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const context = new AudioContext({ sampleRate: 16000 });
        const source = context.createMediaStreamSource(stream);
        const processor = context.createScriptProcessor(4096, 1, 1);

        let pcmBuffer = [];

        processor.onaudioprocess = (e) => {
            const input = e.inputBuffer.getChannelData(0);
            pcmBuffer.push(...input);
        };

        source.connect(processor);
        processor.connect(context.destination);

        setTimeout(async () => {
            processor.disconnect();
            stream.getTracks().forEach(t => t.stop());

            const floatArray = new Float32Array(pcmBuffer);
            const uint8Array = new Uint8Array(floatArray.buffer);
            let binary = "";
            for (let i = 0; i < uint8Array.byteLength; i++) {
                binary += String.fromCharCode(uint8Array[i]);
            }
            const base64Audio = btoa(binary);

            statusElem.innerText = "Sending voice embedding to server...";

            const resp = await fetch("/api/enroll_voice", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ user_name: userName, audio_base64: base64Audio })
            });

            const resData = await resp.json();
            if (resData.success) {
                statusElem.innerText = `✅ Voice profile enrolled for ${userName}!`;
                checkServerStatus();
            } else {
                statusElem.innerText = "❌ Enrollment failed.";
            }
        }, 8000);

    } catch (err) {
        statusElem.innerText = "Error: " + err.message;
    }
}

// Tap to Talk Execution
async function triggerTapToTalk() {
    const queryInput = document.getElementById("input-explicit-query");
    const query = queryInput.value || "What was that company mentioned in our recent conversation?";
    await triggerTapToTalkWithQuery(query);
}

async function triggerTapToTalkWithQuery(query) {
    const orb = document.getElementById("btn-tap-to-talk");
    orb.classList.add("listening");
    
    document.getElementById("response-output").innerText = "Thinking & Synthesizing...";
    
    const thoughtList = document.getElementById("thought-list");
    thoughtList.innerHTML = "<li>⚡ Triggered Tap-to-Talk workflow...</li>";

    try {
        const resp = await fetch("/api/tap_to_talk", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: query })
        });

        const data = await resp.json();
        renderAgentResponse(data);
    } catch (e) {
        document.getElementById("response-output").innerText = "Error invoking Lyra agent: " + e.message;
    } finally {
        orb.classList.remove("listening");
    }
}

function renderAgentResponse(data) {
    document.getElementById("response-output").innerText = data.response;

    // Render Thoughts
    const thoughtList = document.getElementById("thought-list");
    thoughtList.innerHTML = "";
    if (data.thoughts) {
        data.thoughts.forEach(t => {
            const li = document.createElement("li");
            li.innerText = `▸ ${t}`;
            thoughtList.appendChild(li);
        });
    }

    // Render Search Snippets if any
    const snippetsBox = document.getElementById("search-snippets-box");
    const container = document.getElementById("snippets-container");
    container.innerHTML = "";

    if (data.search_results && data.search_results.length > 0) {
        snippetsBox.classList.remove("hidden");
        data.search_results.forEach(s => {
            const d = document.createElement("div");
            d.className = "rag-item";
            d.innerHTML = `<strong>${s.title}</strong>: ${s.snippet}`;
            container.appendChild(d);
        });
    } else {
        snippetsBox.classList.add("hidden");
    }

    // Text-to-Speech Playback
    if (data.tts && 'speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(data.tts.text);
        utterance.rate = data.tts.rate || 1.05;
        window.speechSynthesis.speak(utterance);
    }
}

async function clearMemory() {
    await fetch("/api/memory", { method: "DELETE" });
    document.getElementById("transcript-feed").innerHTML = '<div class="empty-feed-msg">Memory cleared.</div>';
    document.getElementById("memory-count-text").innerText = "0 items";
}

async function searchEpisodicMemory() {
    const q = document.getElementById("input-memory-query").value;
    if (!q) return;

    const resp = await fetch("/api/memory");
    const data = await resp.json();

    const container = document.getElementById("rag-results-container");
    container.innerHTML = "";

    const matched = data.episodic_memory.filter(m => m.text.toLowerCase().includes(q.toLowerCase()));
    if (matched.length === 0) {
        container.innerHTML = '<div class="box-desc">No memory entries matched your search query.</div>';
        return;
    }

    matched.forEach(m => {
        const div = document.createElement("div");
        div.className = "rag-item";
        div.innerHTML = `<strong>[${m.readable_time}] ${m.speaker}:</strong> "${m.text}"`;
        container.appendChild(div);
    });
}
