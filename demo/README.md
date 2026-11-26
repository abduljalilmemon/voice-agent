# Demo GIF

`docs/demo.gif` is generated from a real run of the app, not mocked up:

1. **`capture.mjs`** drives the real frontend in Chrome (Puppeteer) against a running server:
   two typed questions, then a booking by voice. Headless Chrome has no microphone, so the mic
   is replaced by pre-recorded caller clips (`assets/`, made with Edge TTS); the app's recorder,
   WebSocket, Deepgram, the LangGraph agent and TTS all run unmodified. It saves the screen
   recording, every server event with its time, and the agent's spoken replies.
2. **`captions.mjs`** runs **Whisper** on the caller clips and agent replies for word timings, so
   the silent GIF can show the voice as captions. The words come from the known text (the caller
   script and the agent's reply text); Whisper supplies the timing.
3. **`render.mjs`** renders the intro card with **HyperFrames** (`intro/index.html`), composes the
   video with **Remotion** (`src/Demo.tsx`: recording plus a live panel driven by the event log),
   and converts it to a GIF with FFmpeg.

## Regenerate

Requires Node 22+, FFmpeg, Chrome, [Whisper](https://github.com/openai/whisper)
(`pip install openai-whisper`) and the HyperFrames CLI (`npm i -g hyperframes`).

```bash
# 1. from the repo root: run the server with throwaway storage and no email
STORAGE_DIR=/tmp/demo-storage GMAIL_USER= uvicorn app.main:app --port 8765

# 2. in another terminal
cd demo
npm install
node capture.mjs          # needs the server; set CHROME_PATH if Chrome is elsewhere
node captions.mjs         # set WHISPER_BIN if whisper isn't at the default path
node render.mjs           # writes ../docs/demo.gif
```

The capture uses real API calls (Groq, Deepgram, Edge TTS), so each run differs slightly.
