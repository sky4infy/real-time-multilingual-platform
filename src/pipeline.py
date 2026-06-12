import time
import numpy as np

from src.vad import VAD, SAMPLE_RATE
from src.stt import STT
from src.translate import Translator
from src.tts import TTS


class TranslationPipeline:
    """
    End-to-end real-time translation pipeline.

    Flow:
        Microphone -> VAD -> Whisper STT -> Translator -> TTS -> Speaker

    Usage:
        pipeline = TranslationPipeline(target_lang="hi")
        pipeline.start()
        # speak into mic...
        pipeline.stop()
    """

    def __init__(self, target_lang: str = "hi",
                 whisper_model: str = "base",
                 source_lang: str = None):
        """
        Args:
            target_lang:   language to translate INTO (e.g. 'hi' for Hindi)
            whisper_model: whisper model size ('base' recommended for CPU)
            source_lang:   force source language (None = auto-detect)
        """
        self.target_lang = target_lang
        self.source_lang = source_lang

        print("[Pipeline] Initializing all modules...")
        print("[Pipeline] This may take a minute on first run (downloading models)\n")

        # load modules
        self.vad        = VAD()
        self.stt        = STT(model_size=whisper_model)
        self.translator = Translator()
        self.tts        = TTS()

        # preload translation models for instant first translation
        print("\n[Pipeline] Preloading translation models...")
        common_pairs = [
            ("en", target_lang), (target_lang, "en")
        ]
        self.translator.preload([p for p in common_pairs if p[0] != p[1]])

        print("\n[Pipeline] All modules ready.")

    def process_segment(self, audio_segment: np.ndarray) -> dict:
        """
        Process a single speech segment through the full pipeline.

        Args:
            audio_segment: numpy float32 array at 16kHz (from VAD)

        Returns:
            {
                "original_text":   "...",
                "translated_text": "...",
                "source_lang":     "en",
                "target_lang":     "hi",
                "stt_latency":     1.7,
                "translation_latency": 0.15,
                "tts_latency":     0.8,
                "total_latency":   2.65
            }
        """
        t_pipeline_start = time.time()

        # ── Step 1: Speech to Text ──
        # stt_result = self.stt.transcribe(audio_segment, language=self.source_lang)
        # original_text  = stt_result["text"]
        # detected_lang  = stt_result["language"]

        # ── Step 1: Detect language (constrained to en/hi for this demo) ──
        allowed_langs = ["en", self.target_lang]
        detected_lang = self.stt.detect_language_constrained(audio_segment, allowed_langs)

        # ── Step 2: Transcribe using the detected (constrained) language ──
        stt_result = self.stt.transcribe(audio_segment, language=detected_lang)
        original_text = stt_result["text"]

        if not original_text.strip():
            return None  # nothing detected, skip

        # ── Step 2: Determine translation direction ──
        # if speaker's language == target_lang, translate TO English instead
        # (assumes 2-person bidirectional conversation)
        if detected_lang == self.target_lang:
            translate_to = "en"
        else:
            translate_to = self.target_lang

        # ── Step 3: Translate ──
        translation_result = self.translator.translate(
            original_text,
            source_lang=detected_lang,
            target_lang=translate_to
        )
        translated_text = translation_result["translated_text"]

        # ── Step 4: Text to Speech (async — non-blocking) ──
        t_tts_start = time.time()
        self.tts.speak_async(translated_text, lang_code=translate_to)
        tts_latency = round(time.time() - t_tts_start, 2)

        total_latency = round(time.time() - t_pipeline_start, 2)

        return {
            "original_text":      original_text,
            "translated_text":    translated_text,
            "source_lang":        detected_lang,
            "source_lang_name":   stt_result["lang_name"],
            "target_lang":        translate_to,
            "stt_latency":        stt_result["latency"],
            "translation_latency": translation_result["latency"],
            "tts_latency":        tts_latency,
            "total_latency":      total_latency,
            "audio_duration":     stt_result["duration"],
        }

    def run_live(self, duration_seconds: int = 60):
        """
        Run the pipeline live — listens to mic continuously,
        processes each detected speech segment, prints results.

        Args:
            duration_seconds: how long to run before stopping
        """
        print("\n" + "=" * 60)
        print("LIVE TRANSLATION PIPELINE STARTED")
        print(f"Target language: {self.target_lang}")
        print(f"Running for {duration_seconds} seconds. Speak into your mic.")
        print("=" * 60 + "\n")

        self.vad.start()

        start_time = time.time()
        segment_count = 0

        try:
            while time.time() - start_time < duration_seconds:
                segment = self.vad.get_next_speech(timeout=2.0)

                if segment is None:
                    continue

                segment_count += 1
                print(f"\n--- Segment #{segment_count} ---")

                result = self.process_segment(segment)

                if result is None:
                    print("  (no speech detected in segment, skipped)")
                    continue

                print(f"  Original   ({result['source_lang_name']}): {result['original_text']}")
                print(f"  Translated ({result['target_lang']}): {result['translated_text']}")
                print(f"  Latencies  -> STT: {result['stt_latency']}s | "
                      f"Translation: {result['translation_latency']}s | "
                      f"TTS: {result['tts_latency']}s | "
                      f"Total: {result['total_latency']}s")

        except KeyboardInterrupt:
            print("\n[Pipeline] Interrupted by user.")

        finally:
            self.vad.stop()
            print(f"\n[Pipeline] Stopped. Processed {segment_count} segment(s).")


# ─── QUICK TEST ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("FULL PIPELINE TEST")
    print("Target language: Hindi (hi)")
    print("=" * 60)
    print("\nSpeak in English -> hear Hindi translation")
    print("Speak in Hindi    -> hear English translation\n")

    pipeline = TranslationPipeline(target_lang="hi", whisper_model="small")
    pipeline.run_live(duration_seconds=30)

    print("\nIf you saw transcriptions and translations above,")
    print("the full pipeline is working. Move to Phase 7 — Gradio UI.")