import pyttsx3
import threading
import tempfile
import os
import time
import numpy as np
# import sounddevice as sd
# import soundfile as sf

try:
    import sounddevice as sd
    import soundfile as sf
except OSError:
    sd = None
    sf = None
    #This prevents a crash on servers without audio hardware
from gtts import gTTS

# ─── CONFIGURATION ───────────────────────────────────────────
SAMPLE_RATE = 22050   # standard TTS output sample rate

# pyttsx3 voice language hints
# These map target language codes to voice keywords
# pyttsx3 uses whatever voices are installed on Windows
LANG_VOICE_HINTS = {
    "en": "english",
    "hi": "hindi",
    "ta": "tamil",
    "te": "telugu",
    "bn": "bengali",
    "mr": "marathi",
    "gu": "gujarati",
    "ur": "urdu",
    "fr": "french",
    "de": "german",
    "es": "spanish",
    "ja": "japanese",
    "zh": "chinese",
    "ar": "arabic",
}

# Speech rate — words per minute
# 150 = natural, 120 = slow/clear, 180 = fast
SPEECH_RATE = 150


class TTS:
    """
    Text-to-Speech using pyttsx3 (fully offline, Python 3.13 compatible).
    Falls back gracefully if a language voice is not installed on Windows.
    For Indic languages without installed voices, uses English voice
    to read the translated text — user can still read the translation on screen.
    """

    def __init__(self):
        self.lock   = threading.Lock()
        self.engine = None
        self.voices = []
        self.voice_map = {}

        try:
            print("[TTS] Initializing pyttsx3 engine...")
            self.engine = pyttsx3.init()
            self._setup_engine()
            self._list_voices()
            print("[TTS] pyttsx3 ready (local speaker fallback available).")
        except Exception as e:
            print(f"[TTS] pyttsx3 unavailable ({e}) — using gTTS only (fine for cloud deployment).")

    def _setup_engine(self):
        """Configure engine speed and volume."""
        self.engine.setProperty("rate",   SPEECH_RATE)
        self.engine.setProperty("volume", 1.0)

    def _list_voices(self):
        """List and cache all available voices on this system."""
        voices        = self.engine.getProperty("voices")
        self.voices   = voices
        self.voice_map = {}   # lang_code -> voice object

        print(f"[TTS] Found {len(voices)} voice(s) on this system:")
        for v in voices:
            name = v.name.lower()
            print(f"       - {v.name} | ID: {v.id}")
            # map language codes to available voices
            for lang_code, keyword in LANG_VOICE_HINTS.items():
                if keyword in name and lang_code not in self.voice_map:
                    self.voice_map[lang_code] = v
                    print(f"         ^ mapped to language: {lang_code}")

        # always ensure English is mapped
        if "en" not in self.voice_map and voices:
            self.voice_map["en"] = voices[0]

    def _get_voice_for_lang(self, lang_code: str):
        """
        Return the best available voice for a language.
        Falls back to English voice if no matching voice found.
        """
        if lang_code in self.voice_map:
            return self.voice_map[lang_code]
        # fallback to English
        return self.voice_map.get("en", self.voices[0] if self.voices else None)

    def speak(self, text: str, lang_code: str = "en") -> dict:
        """
        Speak text aloud using pyttsx3.
        Thread-safe — uses a lock to prevent concurrent speech.

        Args:
            text:      text to speak
            lang_code: ISO 639-1 language code

        Returns:
            {
                "success":   True,
                "lang_code": "hi",
                "latency":   0.45,
                "voice_used": "Microsoft Heera"
            }
        """
        if self.engine is None:
            return {"success": False, "lang_code": lang_code, "latency": 0.0, "voice_used": "unavailable"}
        ...
        if not text or not text.strip():
            return {"success": False, "lang_code": lang_code,
                    "latency": 0.0, "voice_used": None}

        t_start    = time.time()
        voice      = self._get_voice_for_lang(lang_code)
        voice_name = voice.name if voice else "default"

        with self.lock:
            if voice:
                self.engine.setProperty("voice", voice.id)
            self.engine.say(text)
            self.engine.runAndWait()

        latency = round(time.time() - t_start, 2)
        return {
            "success":    True,
            "lang_code":  lang_code,
            "latency":    latency,
            "voice_used": voice_name
        }

    def speak_async(self, text: str, lang_code: str = "en"):
        """
        Speak text in a background thread — non-blocking.
        Use this in the pipeline so TTS doesn't block the main thread.
        """
        t = threading.Thread(
            target=self.speak,
            args=(text, lang_code),
            daemon=True
        )
        t.start()
        return t

    # def save_to_file(self, text: str, lang_code: str = "en",
    #                  filepath: str = "output.wav") -> dict:
    #     """
    #     Save TTS audio to a WAV file instead of playing it.
    #     Used by Gradio UI to return audio for playback in browser.
    #     """
    #     t_start = time.time()
    #     voice   = self._get_voice_for_lang(lang_code)

    #     with self.lock:
    #         if voice:
    #             self.engine.setProperty("voice", voice.id)
    #         self.engine.save_to_file(text, filepath)
    #         self.engine.runAndWait()

    #     latency = round(time.time() - t_start, 2)
    #     return {
    #         "success":    os.path.exists(filepath),
    #         "filepath":   filepath,
    #         "lang_code":  lang_code,
    #         "latency":    latency,
    #         "voice_used": voice.name if voice else "default"
    #     }

    def save_to_file(self, text: str, lang_code: str = "en",
                 filepath: str = "output.wav") -> dict:
        """
        Save TTS audio to a file using gTTS (Google TTS).
        pyttsx3's save_to_file produces empty files on Windows (SAPI5 bug),
        so gTTS is used for file output. Falls back to pyttsx3 if offline.
        """
        t_start = time.time()

        # gTTS uses different code for Chinese
        gtts_lang = "zh-CN" if lang_code == "zh" else lang_code

        mp3_path = filepath.replace(".wav", ".mp3")

        try:
            tts_obj = gTTS(text=text, lang=gtts_lang)
            tts_obj.save(mp3_path)

            success = os.path.exists(mp3_path) and os.path.getsize(mp3_path) > 100
            latency = round(time.time() - t_start, 2)

            return {
                "success":    success,
                "filepath":   mp3_path,
                "lang_code":  lang_code,
                "latency":    latency,
                "voice_used": "gTTS"
            }

        except Exception as e:
            print(f"[TTS] gTTS failed ({e}), falling back to pyttsx3 speak (audio plays on server, not browser)")
            # fallback — at least speak it on the server
            self.speak(text, lang_code)
            return {
                "success":    False,
                "filepath":   None,
                "lang_code":  lang_code,
                "latency":    round(time.time() - t_start, 2),
                "voice_used": "pyttsx3 (server speaker only)"
            }

    def get_available_languages(self) -> list:
        """Return list of language codes that have a mapped voice."""
        return list(self.voice_map.keys())


# ─── QUICK TEST ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("TTS Module Test")
    print("=" * 55)

    tts = TTS()

    print(f"\n[TEST] Languages with voices: {tts.get_available_languages()}")

    test_cases = [
        ("Hello, my name is Akash Kumar.",        "en"),
        ("नमस्ते, मेरा नाम आकाश कुमार है।",         "hi"),
        ("Bonjour, je m'appelle Akash.",           "fr"),
        ("Hallo, mein Name ist Akash.",            "de"),
    ]

    for text, lang in test_cases:
        print(f"\n{'─'*55}")
        print(f"  Speaking ({lang}): {text}")
        result = tts.speak(text, lang)
        print(f"  Voice used : {result['voice_used']}")
        print(f"  Latency    : {result['latency']}s")
        time.sleep(0.5)

    # test save to file
    print(f"\n{'─'*55}")
    print("  Testing save_to_file...")
    result = tts.save_to_file(
        "This audio was saved to a file.",
        "en",
        "outputs/test_output.wav"
    )
    print(f"  File saved : {result['filepath']}")
    print(f"  Success    : {result['success']}")
    print(f"  Latency    : {result['latency']}s")

    print(f"\n{'='*55}")
    print("TTS test complete.")
    print("You should have heard speech for each test case.")
    print("If you heard audio — TTS is working. Move to Phase 6.")