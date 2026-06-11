import torch
import numpy as np
import sounddevice as sd
import queue
import threading

# ─── CONFIGURATION ───────────────────────────────────────────
SAMPLE_RATE     = 16000   # Whisper expects 16kHz
CHUNK_SAMPLES   = 512     # SileroVAD REQUIRES exactly 512 samples at 16kHz
SILENCE_THRESH  = 6       # consecutive silent chunks before closing segment
VAD_THRESHOLD   = 0.4     # silero confidence threshold (0.0–1.0)
MIN_SPEECH_CHUNKS = 4     # ignore segments shorter than this (filters clicks/noise)


class VAD:
    """
    Voice Activity Detection using SileroVAD.
    Captures microphone audio in 512-sample chunks at 16kHz,
    filters out silence, and pushes complete speech segments
    into a queue for the STT module.
    """

    def __init__(self):
        print("[VAD] Loading SileroVAD model...")
        self.model, self.utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True
        )
        (self.get_speech_timestamps,
         self.save_audio,
         self.read_audio,
         self.VADIterator,
         self.collect_chunks) = self.utils

        self.audio_queue  = queue.Queue()   # raw 512-sample chunks from mic
        self.speech_queue = queue.Queue()   # complete speech segments for STT
        self.is_running   = False
        self.stream       = None
        print("[VAD] SileroVAD loaded successfully.")

    def _mic_callback(self, indata, frames, time, status):
        """Called by sounddevice for every 512-sample chunk from mic."""
        if status:
            print(f"[VAD] Mic warning: {status}")
        audio_chunk = indata[:, 0].astype(np.float32)
        self.audio_queue.put(audio_chunk)

    def _is_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        Run SileroVAD on exactly 512 samples.
        Returns True if speech confidence >= VAD_THRESHOLD.
        """
        tensor = torch.from_numpy(audio_chunk)
        confidence = self.model(tensor, SAMPLE_RATE).item()
        return confidence >= VAD_THRESHOLD

    def _processing_loop(self):
        """
        Background thread:
        - Reads 512-sample chunks from audio_queue
        - Runs VAD on each chunk
        - Accumulates speech chunks into a buffer
        - When silence detected after speech, pushes full segment to speech_queue
        """
        buffer       = []   # accumulates speech chunks
        silent_count = 0    # counts consecutive silent chunks
        speaking     = False

        while self.is_running:
            try:
                chunk = self.audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if self._is_speech(chunk):
                speaking     = True
                silent_count = 0
                buffer.append(chunk)

            else:
                if speaking:
                    silent_count += 1
                    buffer.append(chunk)  # include trailing silence for naturalness

                    if silent_count >= SILENCE_THRESH:
                        # only send if segment is long enough to be real speech
                        if len(buffer) >= MIN_SPEECH_CHUNKS:
                            speech_segment = np.concatenate(buffer)
                            self.speech_queue.put(speech_segment)
                            duration = len(speech_segment) / SAMPLE_RATE
                            print(f"[VAD] Speech segment ready: {duration:.2f}s")
                        else:
                            print("[VAD] Segment too short, ignoring (noise/click)")

                        buffer       = []
                        silent_count = 0
                        speaking     = False

        # flush remaining buffer on stop
        if buffer and len(buffer) >= MIN_SPEECH_CHUNKS:
            speech_segment = np.concatenate(buffer)
            self.speech_queue.put(speech_segment)

    def start(self):
        """Start microphone capture and VAD processing thread."""
        self.is_running = True

        self.proc_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True
        )
        self.proc_thread.start()

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,   # exactly 512 — matches SileroVAD requirement
            callback=self._mic_callback
        )
        self.stream.start()
        print("[VAD] Microphone started. Listening...")

    def stop(self):
        """Stop microphone and processing thread."""
        self.is_running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        print("[VAD] Stopped.")

    def get_next_speech(self, timeout=10.0):
        """
        Block until a speech segment is available.
        Returns numpy float32 array at 16kHz, or None on timeout.
        """
        try:
            return self.speech_queue.get(timeout=timeout)
        except queue.Empty:
            return None


# ─── QUICK TEST ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("VAD Test — speak into your mic for 10 seconds")
    print("Say something, pause, say something again")
    print("=" * 50)

    vad = VAD()
    vad.start()

    import time
    start          = time.time()
    segments_found = 0

    while time.time() - start < 10:
        segment = vad.get_next_speech(timeout=2.0)
        if segment is not None:
            segments_found += 1
            duration = len(segment) / SAMPLE_RATE
            print(f"[TEST] Segment #{segments_found} received — {duration:.2f}s")

    vad.stop()
    print(f"\n[TEST] Done. Detected {segments_found} speech segment(s).")
    if segments_found > 0:
        print("VAD is working correctly. Move to Phase 3.")
    else:
        print("No speech detected. Check your microphone.")