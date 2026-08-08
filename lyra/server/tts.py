from typing import Any


class TextToSpeechEngine:
    """
    Text-to-Speech Output Synthesizer for Lyra.
    Formats audio synthesis specifications to stream responses back to the user's earbuds or client app.
    """

    def __init__(self, voice_name: str = "en-US-Jarvis", speech_rate: float = 1.05):
        self.voice_name = voice_name
        self.speech_rate = speech_rate

    def synthesize(self, text: str) -> dict[str, Any]:
        """
        Generates TTS synthesis configuration payload to be rendered on client or stream back audio.
        """
        clean_text = text.replace("*", "").strip()

        return {
            "text": clean_text,
            "voice": self.voice_name,
            "rate": self.speech_rate,
            "pitch": 1.0,
            "format": "speech_synthesis_v1"
        }
