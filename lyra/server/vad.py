import numpy as np


class VoiceActivityDetector:
    """
    Real-time, low-power Voice Activity Detector (VAD)
    Filters ambient white noise, room reverberation, and silence.
    """

    def __init__(self, sample_rate: int = 16000, energy_threshold: float = 0.015):
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold

    def is_speech(self, audio_data: np.ndarray) -> dict:
        """
        Analyzes audio slice and returns speech classification & metrics.
        """
        if len(audio_data) == 0:
            return {"is_speech": False, "rms": 0.0, "zcr": 0.0, "confidence": 0.0}

        # Ensure float32 array
        samples = audio_data.astype(np.float32)
        if np.max(np.abs(samples)) > 1.0:
            samples = samples / 32768.0

        # Calculate Root Mean Square (RMS) energy
        rms = float(np.sqrt(np.mean(samples ** 2)))

        # Calculate Zero Crossing Rate (ZCR)
        zero_crossings = np.nonzero(np.diff(samples > 0))[0]
        zcr = float(len(zero_crossings) / len(samples))

        # Calculate Spectral Centroid for speech frequency detection (300Hz - 3400Hz)
        is_speech = False
        confidence = 0.0

        if rms > self.energy_threshold:
            # Check frequency content via FFT
            fft_vals = np.abs(np.fft.rfft(samples))
            freqs = np.fft.rfftfreq(len(samples), 1.0 / self.sample_rate)

            # Energy in speech band (300Hz - 3400Hz)
            speech_band_mask = (freqs >= 300) & (freqs <= 3400)
            speech_band_energy = np.sum(fft_vals[speech_band_mask])
            total_energy = np.sum(fft_vals) + 1e-9
            speech_ratio = float(speech_band_energy / total_energy)

            if speech_ratio > 0.35 and zcr < 0.40:
                is_speech = True
                confidence = min(1.0, (rms / self.energy_threshold) * speech_ratio)

        return {
            "is_speech": is_speech,
            "rms": round(rms, 5),
            "zcr": round(zcr, 4),
            "confidence": round(confidence, 3)
        }
