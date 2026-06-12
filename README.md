---
title: Real-Time Multilingual Communication Platform
emoji: 🌐
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.17.3
app_file: app.py
pinned: false
---
# Real-Time Multilingual Communication Platform

A zero-cost, open-source proof-of-concept for real-time voice-to-voice 
translation, built for Lensara Technologies (Architecture 1 — Demo).

## Architecture
Microphone → VAD (SileroVAD) → Speech-to-Text (Whisper) → 
Translation (Helsinki-NLP) → Text-to-Speech (gTTS) → Audio Output

## Setup

\`\`\`bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
\`\`\`

## Run the demo

\`\`\`bash
python app.py
\`\`\`

Open http://127.0.0.1:7860 in your browser.

## How it works
- Two participant panels (A and B)
- Each selects their spoken language and the language to translate into
- Record a short message (2-5 seconds)
- Click Translate — see transcription, translation, and hear audio output

## Tech Stack
- **VAD**: SileroVAD
- **STT**: OpenAI Whisper (small)
- **Translation**: Helsinki-NLP OPUS-MT (quantized, CPU-optimized)
- **TTS**: gTTS
- **UI**: Gradio

## Cost
Rs. 0 — fully open source (gTTS requires internet for synthesis)

## Author
Akash Kumar | NIAMT Ranchi