import gradio as gr
import numpy as np
import tempfile
import os
from scipy.signal import resample

from src.stt import STT, SUPPORTED_LANGUAGES
from src.translate import Translator
from src.tts import TTS

# ─── LOAD MODELS ONCE AT STARTUP ─────────────────────────────
print("=" * 60)
print("Loading models — this takes 30-60 seconds...")
print("=" * 60)

stt        = STT(model_size="small")
translator = Translator()
tts        = TTS()

# preload common pairs for instant first translation
translator.preload([("en", "hi"), ("hi", "en")])

print("\nAll models loaded. Launching Gradio UI...\n")

WHISPER_SAMPLE_RATE = 16000

LANGUAGE_CHOICES = [
    ("English", "en"),
    ("Hindi", "hi"),
    ("Tamil", "ta"),
    ("Telugu", "te"),
    ("Bengali", "bn"),
    ("Marathi", "mr"),
    ("Gujarati", "gu"),
    ("French", "fr"),
    ("German", "de"),
    ("Spanish", "es"),
]


def resample_audio(audio_data: np.ndarray, original_sr: int, target_sr: int = WHISPER_SAMPLE_RATE) -> np.ndarray:
    """Resample audio from Gradio's mic sample rate to 16kHz for Whisper."""
    if original_sr == target_sr:
        return audio_data

    num_samples = int(len(audio_data) * target_sr / original_sr)
    resampled   = resample(audio_data, num_samples)
    return resampled.astype(np.float32)


def process_audio(audio, speaker_lang_code: str, listener_lang_code: str):
    """
    Full pipeline for one recorded clip:
    1. Resample to 16kHz
    2. Transcribe (forced to speaker's selected language for accuracy)
    3. Translate to listener's language
    4. Synthesize speech for listener
    5. Return transcript, translation, and audio file path

    Args:
        audio: tuple (sample_rate, numpy_array) from gr.Audio
        speaker_lang_code: language the speaker is speaking in
        listener_lang_code: language to translate INTO

    Returns:
        (original_text, translated_text, audio_filepath, status_message)
    """
    if audio is None:
        return "", "", None, "No audio recorded. Please record something first."

    sample_rate, audio_data = audio

    # # Gradio gives int16 or float32 depending on browser — normalize to float32
    # if audio_data.dtype != np.float32:
    #     audio_data = audio_data.astype(np.float32) / 32768.0

#  improvement
    # Normalize based on actual value range, not just dtype
    audio_data = audio_data.astype(np.float32)
    max_val = np.abs(audio_data).max()
    if max_val > 1.0:
        audio_data = audio_data / 32768.0
        max_val = np.abs(audio_data).max()

    # Gain boost — laptop mics are often too quiet for Whisper's
    # silence detector, causing "no speech detected" false negatives
    if max_val > 0:
        audio_data = audio_data / max_val * 0.9

    # convert stereo to mono if needed
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    # resample to 16kHz for Whisper
    audio_16k = resample_audio(audio_data, sample_rate, WHISPER_SAMPLE_RATE)

    duration_sec = len(audio_16k) / WHISPER_SAMPLE_RATE
    if len(audio_16k) < WHISPER_SAMPLE_RATE * 0.3:
        return "", "", None, f"Recording too short ({duration_sec:.2f}s). Speak for at least 1 second."

    # ── Step 1: Transcribe (forced language = speaker's selection) ──
    stt_result = stt.transcribe(audio_16k, language=speaker_lang_code)
    original_text = stt_result["text"].strip()

    if not original_text:
        return "", "", None, "No speech detected. Please try again, speak clearly."

    # ── Step 2: Translate ──
    translation_result = translator.translate(
        original_text,
        source_lang=speaker_lang_code,
        target_lang=listener_lang_code
    )
    translated_text = translation_result["translated_text"]

    # ── Step 3: Text to Speech — save to file for browser playback ──
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir="outputs") as f:
        output_path = f.name

    tts_result = tts.save_to_file(translated_text, listener_lang_code, output_path)
    output_path = tts_result["filepath"]  # now points to .mp3

    file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
    status = (
        f"STT: {stt_result['latency']}s | "
        f"Translation: {translation_result['latency']}s | "
        f"TTS: {tts_result['latency']}s | "
        f"Total: {round(stt_result['latency'] + translation_result['latency'] + tts_result['latency'], 2)}s | "
        f"TTS file: {file_size} bytes | success={tts_result['success']}"
    )

    return original_text, translated_text, output_path, status


# ─── BUILD GRADIO UI ──────────────────────────────────────────
with gr.Blocks(title="Lensara Real-Time Multilingual Translator", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🌐 Real-Time Multilingual Communication Platform
    ### Lensara Technologies — Zero Cost Demo (Architecture 1)

    Two participants speaking different languages can communicate naturally.
    Each person records their message, selects their language and the other
    person's language, then clicks Translate.
    """)

    with gr.Row():
        # ── USER A PANEL ──
        with gr.Column():
            gr.Markdown("## 🧑 Participant A")
            a_lang = gr.Dropdown(
                choices=LANGUAGE_CHOICES, value="en",
                label="A speaks in:"
            )
            b_lang_for_a = gr.Dropdown(
                choices=LANGUAGE_CHOICES, value="hi",
                label="Translate to (B's language):"
            )
            a_audio = gr.Audio(
                sources=["microphone"], type="numpy",
                label="Record your message"
            )
            a_translate_btn = gr.Button("Translate A → B", variant="primary")

            gr.Markdown("**A said (original):**")
            a_original = gr.Textbox(label="", interactive=False)

            gr.Markdown("**Translated for B:**")
            a_translated = gr.Textbox(label="", interactive=False)

            a_audio_out = gr.Audio(label="Play translation for B", autoplay=True)
            a_status    = gr.Textbox(label="Latency", interactive=False)

        # ── USER B PANEL ──
        with gr.Column():
            gr.Markdown("## 🧑 Participant B")
            b_lang = gr.Dropdown(
                choices=LANGUAGE_CHOICES, value="hi",
                label="B speaks in:"
            )
            a_lang_for_b = gr.Dropdown(
                choices=LANGUAGE_CHOICES, value="en",
                label="Translate to (A's language):"
            )
            b_audio = gr.Audio(
                sources=["microphone"], type="numpy",
                label="Record your message"
            )
            b_translate_btn = gr.Button("Translate B → A", variant="primary")

            gr.Markdown("**B said (original):**")
            b_original = gr.Textbox(label="", interactive=False)

            gr.Markdown("**Translated for A:**")
            b_translated = gr.Textbox(label="", interactive=False)

            b_audio_out = gr.Audio(label="Play translation for A", autoplay=True)
            b_status    = gr.Textbox(label="Latency", interactive=False)

    gr.Markdown("""
    ---
    **How to use:**
    1. Select each participant's spoken language and the language to translate into
    2. Click the microphone to record a short message (2-5 seconds works best)
    3. Click Translate — transcription, translation, and audio output will appear
    4. The translated audio plays automatically for the listener

    **Architecture:** Whisper (STT) → Helsinki-NLP / IndicTrans2 (Translation) → pyttsx3 (TTS)
    **Cost:** Rs. 0 — fully open source, runs locally
    """)

    # ── WIRE UP BUTTONS ──
    a_translate_btn.click(
        fn=process_audio,
        inputs=[a_audio, a_lang, b_lang_for_a],
        outputs=[a_original, a_translated, a_audio_out, a_status]
    )

    b_translate_btn.click(
        fn=process_audio,
        inputs=[b_audio, b_lang, a_lang_for_b],
        outputs=[b_original, b_translated, b_audio_out, b_status]
    )


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)