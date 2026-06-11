from transformers import MarianMTModel, MarianTokenizer
import torch
import time

# ─── LANGUAGE ROUTING TABLE ──────────────────────────────────
LANGUAGE_PAIRS = {
    # English <-> Indian languages
    ("en", "hi"): "Helsinki-NLP/opus-mt-en-hi",
    ("hi", "en"): "Helsinki-NLP/opus-mt-hi-en",
    ("en", "ta"): "Helsinki-NLP/opus-mt-en-dra",
    ("ta", "en"): "Helsinki-NLP/opus-mt-dra-en",
    ("en", "te"): "Helsinki-NLP/opus-mt-en-dra",
    ("te", "en"): "Helsinki-NLP/opus-mt-dra-en",
    ("en", "bn"): "Helsinki-NLP/opus-mt-en-bn",
    ("bn", "en"): "Helsinki-NLP/opus-mt-bn-en",
    ("en", "mr"): "Helsinki-NLP/opus-mt-en-mr",
    ("mr", "en"): "Helsinki-NLP/opus-mt-mr-en",
    ("en", "gu"): "Helsinki-NLP/opus-mt-en-gu",
    ("gu", "en"): "Helsinki-NLP/opus-mt-gu-en",
    ("en", "ur"): "Helsinki-NLP/opus-mt-en-ur",
    ("ur", "en"): "Helsinki-NLP/opus-mt-ur-en",

    # English <-> Global languages
    ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
    ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
    ("en", "de"): "Helsinki-NLP/opus-mt-en-de",
    ("de", "en"): "Helsinki-NLP/opus-mt-de-en",
    ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
    ("es", "en"): "Helsinki-NLP/opus-mt-es-en",
    ("en", "ja"): "Helsinki-NLP/opus-mt-en-jap",
    ("ja", "en"): "Helsinki-NLP/opus-mt-jap-en",
    ("en", "zh"): "Helsinki-NLP/opus-mt-en-zh",
    ("zh", "en"): "Helsinki-NLP/opus-mt-zh-en",
    ("en", "ar"): "Helsinki-NLP/opus-mt-en-ar",
    ("ar", "en"): "Helsinki-NLP/opus-mt-ar-en",

    # Same language — no translation needed
    ("en", "en"): None,
    ("hi", "hi"): None,
    ("ta", "ta"): None,
    ("te", "te"): None,
    ("bn", "bn"): None,
    ("fr", "fr"): None,
    ("de", "de"): None,
}

# Indic languages that need English as pivot
PIVOT_LANGUAGES = {"ta", "te", "kn", "ml", "gu", "mr"}


class Translator:
    """
    Text translation using Helsinki-NLP MarianMT models.
    Optimized for CPU with:
    - Greedy decoding (num_beams=1) — 3-4x faster than beam search
    - INT8 dynamic quantization — 2x faster with minimal quality loss
    - Model caching — each model loaded only once per session
    - English pivot for Indic-to-Indic pairs
    """

    def __init__(self):
        self._model_cache = {}
        print("[Translator] Ready. Models load and quantize on first use.")

    def _load_model(self, model_name: str):
        """
        Load, quantize, and cache a MarianMT model.
        Quantization converts weights to INT8 — ~2x faster on CPU.
        """
        if model_name in self._model_cache:
            return self._model_cache[model_name]

        print(f"[Translator] Loading: {model_name} ...")
        t0        = time.time()
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model     = MarianMTModel.from_pretrained(model_name)
        model.eval()

        # ── INT8 quantization for CPU speed ──
        model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )

        elapsed = round(time.time() - t0, 1)
        print(f"[Translator] Loaded + quantized: {model_name} ({elapsed}s)")

        self._model_cache[model_name] = (tokenizer, model)
        return tokenizer, model

    def _run_translation(self, text: str, model_name: str) -> str:
        """
        Run translation with greedy decoding (fastest on CPU).
        num_beams=1 = greedy — 3-4x faster than beam search,
        minimal quality loss for most language pairs.
        """
        tokenizer, model = self._load_model(model_name)

        inputs = tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        with torch.no_grad():   # disable gradient tracking — faster inference
            translated = model.generate(
                **inputs,
                max_length=512,
                num_beams=1,        # greedy decoding
                do_sample=False     # deterministic output
            )

        return tokenizer.decode(translated[0], skip_special_tokens=True)

    def translate(self, text: str, source_lang: str, target_lang: str) -> dict:
        """
        Translate text from source_lang to target_lang.

        Args:
            text:        input text string
            source_lang: ISO 639-1 code e.g. 'hi', 'en', 'ta'
            target_lang: ISO 639-1 code

        Returns:
            {
                "translated_text": "...",
                "source_lang":     "hi",
                "target_lang":     "en",
                "latency":         1.2,
                "method":          "direct" | "pivot_via_english" | "same_language"
            }
        """

        # ── same language shortcut ──
        if source_lang == target_lang:
            return {
                "translated_text": text,
                "source_lang":     source_lang,
                "target_lang":     target_lang,
                "latency":         0.0,
                "method":          "same_language"
            }

        # ── empty input shortcut ──
        if not text.strip():
            return {
                "translated_text": "",
                "source_lang":     source_lang,
                "target_lang":     target_lang,
                "latency":         0.0,
                "method":          "empty_input"
            }

        t_start = time.time()
        method  = "direct"

        pair       = (source_lang, target_lang)
        model_name = LANGUAGE_PAIRS.get(pair)

        # ── direct translation ──
        if model_name:
            translated_text = self._run_translation(text, model_name)

        # ── pivot via English for Indic-to-Indic ──
        elif source_lang in PIVOT_LANGUAGES or target_lang in PIVOT_LANGUAGES:
            method = "pivot_via_english"
            print(f"[Translator] Pivot: {source_lang} → en → {target_lang}")

            # step 1: source → English
            src_en_model = LANGUAGE_PAIRS.get((source_lang, "en"))
            if src_en_model:
                english_text = self._run_translation(text, src_en_model)
            else:
                english_text = text

            # step 2: English → target
            en_tgt_model = LANGUAGE_PAIRS.get(("en", target_lang))
            if en_tgt_model:
                translated_text = self._run_translation(english_text, en_tgt_model)
            else:
                translated_text = english_text

        # ── generic fallback — try constructing model name ──
        else:
            method     = "fallback"
            model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
            print(f"[Translator] Trying fallback: {model_name}")
            try:
                translated_text = self._run_translation(text, model_name)
            except Exception as e:
                print(f"[Translator] No model found for {source_lang}→{target_lang}: {e}")
                translated_text = text
                method          = "no_model_found"

        latency = round(time.time() - t_start, 2)

        return {
            "translated_text": translated_text,
            "source_lang":     source_lang,
            "target_lang":     target_lang,
            "latency":         latency,
            "method":          method
        }

    def preload(self, language_pairs: list):
        """
        Preload and quantize models for given pairs before demo starts.
        Call this at startup to avoid delay during live translation.

        Example:
            translator.preload([("en","hi"), ("hi","en")])
        """
        print(f"[Translator] Preloading {len(language_pairs)} language pair(s)...")
        for src, tgt in language_pairs:
            model_name = LANGUAGE_PAIRS.get((src, tgt))
            if model_name:
                self._load_model(model_name)
        print("[Translator] Preload complete.")


# ─── QUICK TEST ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Translation Module Test — Optimized CPU Version")
    print("=" * 60)

    translator = Translator()

    test_cases = [
        ("Hello, how are you?",           "en", "hi"),
        ("मेरा नाम आकाश कुमार है।",         "hi", "en"),
        ("Hello, how are you?",           "en", "fr"),
        ("Hello, how are you?",           "en", "de"),
        ("Bonjour, comment vas-tu?",      "fr", "en"),
    ]

    print("\nNote: First run downloads models. Second run uses cache.\n")

    results = []
    for text, src, tgt in test_cases:
        print(f"{'─'*60}")
        print(f"  Input     : {text}")
        print(f"  Direction : {src} → {tgt}")

        result = translator.translate(text, src, tgt)

        print(f"  Output    : {result['translated_text']}")
        print(f"  Method    : {result['method']}")
        print(f"  Latency   : {result['latency']}s")
        results.append(result['latency'])

    print(f"\n{'='*60}")
    avg = round(sum(results) / len(results), 2)
    print(f"  Average latency : {avg}s")
    print(f"  Min latency     : {min(results)}s")
    print(f"  Max latency     : {max(results)}s")
    print(f"{'='*60}")
    print("\nIf latency < 3s on second run — Translation module is ready.")
    print("Move to Phase 5 — TTS module.")