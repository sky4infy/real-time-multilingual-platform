import whisper
import numpy as np
import torch
import time

# ─── CONFIGURATION ───────────────────────────────────────────
# Model sizes:  tiny | base | small | medium
# CPU recommendation: base (fast) or small (more accurate, ~3-5s on CPU)
WHISPER_MODEL   = "base"
SAMPLE_RATE     = 16000

# Languages we support — used for display only
# Whisper auto-detects, but we can force if needed
SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "bn": "Bengali",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "pa": "Punjabi",
    "ur": "Urdu",
    "ja": "Japanese",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "zh": "Chinese",
    "ar": "Arabic",
}


class STT:
    """
    Speech-to-Text using OpenAI Whisper.
    Takes numpy float32 audio arrays (from VAD)
    and returns transcribed text + detected language.
    """

    def __init__(self, model_size: str = WHISPER_MODEL):
        print(f"[STT] Loading Whisper model: {model_size} ...")
        self.model      = whisper.load_model(model_size)
        self.model_size = model_size
        print(f"[STT] Whisper '{model_size}' loaded successfully.")

        #  improvement, eleminate hallucinations
    def detect_language_constrained(self, audio: np.ndarray, allowed_languages: list) -> str:
        """
        Detect language but only choose from allowed_languages.
        Prevents Whisper from hallucinating random languages
        (Spanish, Turkish, Greek) on short/noisy audio clips.
        """
        audio_padded = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio_padded).to(self.model.device)
        _, probs = self.model.detect_language(mel)

        # filter to only allowed languages, pick highest probability
        filtered = {lang: probs.get(lang, 0.0) for lang in allowed_languages}
        best_lang = max(filtered, key=filtered.get)
        return best_lang

    def transcribe(self, audio: np.ndarray, language: str = None) -> dict:
        """
        Transcribe audio to text.

        Args:
            audio:    float32 numpy array at 16kHz (from VAD)
            language: optional ISO code to force language (e.g. 'hi')
                      if None, Whisper auto-detects

        Returns dict:
            {
                "text":     "transcribed text",
                "language": "hi",
                "lang_name": "Hindi",
                "duration": 2.3,
                "latency":  1.8
            }
        """
        if audio is None or len(audio) == 0:
            return {"text": "", "language": "unknown", "lang_name": "Unknown",
                    "duration": 0, "latency": 0}

        duration = len(audio) / SAMPLE_RATE

        # Whisper needs float32 audio normalised between -1 and 1
        audio = audio.astype(np.float32)
        if audio.max() > 1.0:
            audio = audio / 32768.0  # normalise if 16-bit int was passed

        # build options
        options = {
            "fp16": False,          # CPU doesn't support fp16
            "task": "transcribe",   # transcribe (keep language) not translate
        }
        if language:
            options["language"] = language

        t_start = time.time()
        result  = self.model.transcribe(audio, **options)
        latency = round(time.time() - t_start, 2)

        detected_lang     = result.get("language", "unknown")
        detected_lang_name = SUPPORTED_LANGUAGES.get(detected_lang, detected_lang.upper())
        text               = result.get("text", "").strip()

        return {
            "text":      text,
            "language":  detected_lang,
            "lang_name": detected_lang_name,
            "duration":  round(duration, 2),
            "latency":   latency,
        }

    def detect_language_only(self, audio: np.ndarray) -> tuple:
        """
        Quickly detect language without full transcription.
        Returns (language_code, confidence) tuple.
        Faster than full transcribe when you only need the language.
        """
        audio  = whisper.pad_or_trim(audio)
        mel    = whisper.log_mel_spectrogram(audio).to(self.model.device)
        _, probs = self.model.detect_language(mel)
        best_lang = max(probs, key=probs.get)
        return best_lang, round(probs[best_lang], 3)


# ─── QUICK TEST ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("STT Test — recording 5 seconds of audio from mic")
    print("Speak clearly in any language")
    print("=" * 55)

    import sounddevice as sd

    RECORD_SECONDS = 5
    print(f"\nRecording for {RECORD_SECONDS} seconds... SPEAK NOW")

    audio_data = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )
    sd.wait()
    audio_flat = audio_data[:, 0]

    print("Recording done. Transcribing...\n")

    stt    = STT(model_size="base")
    result = stt.transcribe(audio_flat)

    print("─" * 55)
    print(f"  Text detected : {result['text']}")
    print(f"  Language      : {result['lang_name']} ({result['language']})")
    print(f"  Audio duration: {result['duration']}s")
    print(f"  STT latency   : {result['latency']}s")
    print("─" * 55)

    if result["text"]:
        print("\nSTT is working correctly. Move to Phase 4.")
    else:
        print("\nNo text detected. Try speaking louder or check mic.")