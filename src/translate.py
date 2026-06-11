from transformers import MarianMTModel, MarianTokenizer
import time

# ─── LANGUAGE ROUTING TABLE ──────────────────────────────────
# Maps (source_lang, target_lang) to Helsinki-NLP model name
# Add more pairs as needed from https://huggingface.co/Helsinki-NLP

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

    # Hindi <-> Other Indian languages (via pivot if needed)
    ("hi", "ta"): "Helsinki-NLP/opus-mt-hi-en",   # hi->en->ta pivot
    ("hi", "bn"): "Helsinki-NLP/opus-mt-hi-en",

    # Same language — no translation needed
    ("en", "en"): None,
    ("hi", "hi"): None,
}

# Languages that need English as pivot (source -> en -> target)
PIVOT_LANGUAGES = {"ta", "te", "kn", "ml"}


class Translator:
    """
    Text translation using Helsinki-NLP MarianMT models.
    Supports 1000+ language pairs via HuggingFace.
    Models are cached locally after first download.
    Uses English as pivot language for Indic-to-Indic pairs.
    """

    def __init__(self):
        self._model_cache = {}   # { model_name: (tokenizer, model) }
        print("[Translator] Ready. Models will download on first use.")

    def _load_model(self, model_name: str):
        """Load and cache a MarianMT model."""
        if model_name in self._model_cache:
            return self._model_cache[model_name]

        print(f"[Translator] Loading model: {model_name} ...")
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model     = MarianMTModel.from_pretrained(model_name)
        model.eval()
        
        # adding extra for 2x faster on cpu with minimal quality loss.
        # optimize for CPU inference
        import torch
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )

        self._model_cache[model_name] = (tokenizer, model)
        print(f"[Translator] Model loaded: {model_name}")
        return tokenizer, model

    def _translate_with_model(self, text: str, model_name: str) -> str:
        """Translate text using a specific model."""
        tokenizer, model = self._load_model(model_name)

        inputs = tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        # translated = model.generate(
        #     **inputs,
        #     max_length=512,
        #     num_beams=4,
        #     early_stopping=True
        # )
        translated = model.generate(
            **inputs,
            max_length=512,
            num_beams=1,          # greedy decoding — 3-4x faster on CPU
            do_sample=False
        )

        return tokenizer.decode(translated[0], skip_special_tokens=True)

    def translate(self, text: str, source_lang: str, target_lang: str) -> dict:
        """
        Translate text from source_lang to target_lang.

        Args:
            text:        text to translate
            source_lang: ISO 639-1 code (e.g. 'hi', 'en', 'ta')
            target_lang: ISO 639-1 code

        Returns dict:
            {
                "translated_text": "...",
                "source_lang":     "hi",
                "target_lang":     "en",
                "latency":         0.45,
                "method":          "direct" | "pivot" | "same_language"
            }
        """

        # ── same language — no translation needed ──
        if source_lang == target_lang:
            return {
                "translated_text": text,
                "source_lang":     source_lang,
                "target_lang":     target_lang,
                "latency":         0.0,
                "method":          "same_language"
            }

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

        # ── check direct model pair ──
        pair       = (source_lang, target_lang)
        model_name = LANGUAGE_PAIRS.get(pair)

        if model_name:
            translated_text = self._translate_with_model(text, model_name)

        # ── pivot via English (for Indic-to-Indic or unsupported pairs) ──
        elif source_lang in PIVOT_LANGUAGES or target_lang in PIVOT_LANGUAGES:
            method = "pivot_via_english"
            print(f"[Translator] Using English pivot: {source_lang} -> en -> {target_lang}")

            # step 1: source -> English
            src_to_en = LANGUAGE_PAIRS.get((source_lang, "en"))
            if src_to_en:
                english_text = self._translate_with_model(text, src_to_en)
            else:
                english_text = text  # assume English if no model found

            # step 2: English -> target
            en_to_tgt = LANGUAGE_PAIRS.get(("en", target_lang))
            if en_to_tgt:
                translated_text = self._translate_with_model(english_text, en_to_tgt)
            else:
                translated_text = english_text

        # ── fallback: try generic multilingual model ──
        else:
            method     = "fallback_generic"
            model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
            print(f"[Translator] Trying generic model: {model_name}")
            try:
                translated_text = self._translate_with_model(text, model_name)
            except Exception as e:
                print(f"[Translator] Model not found: {e}")
                translated_text = text  # return original if no model found
                method          = "no_model_found"

        latency = round(time.time() - t_start, 2)

        return {
            "translated_text": translated_text,
            "source_lang":     source_lang,
            "target_lang":     target_lang,
            "latency":         latency,
            "method":          method
        }


# ─── QUICK TEST ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("Translation Module Test")
    print("=" * 55)

    translator = Translator()

    test_cases = [
        ("Hello, how are you?",         "en", "hi"),
        # ("Mera naam Akash Kumar hai.",   "hi", "en"),
        ("मेरा नाम आकाश कुमार है।", "hi", "en"),
        ("Hello, how are you?",         "en", "fr"),
        ("Hello, how are you?",         "en", "de"),
        ("Bonjour, comment vas-tu?",    "fr", "en"),
    ]

    for text, src, tgt in test_cases:
        print(f"\n{'─'*55}")
        print(f"  Input    : {text}")
        print(f"  Direction: {src} → {tgt}")

        result = translator.translate(text, src, tgt)

        print(f"  Output   : {result['translated_text']}")
        print(f"  Method   : {result['method']}")
        print(f"  Latency  : {result['latency']}s")

    print(f"\n{'='*55}")
    print("Translation test complete. Move to Phase 5.")