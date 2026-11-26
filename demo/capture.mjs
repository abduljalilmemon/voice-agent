// Records the REAL app: drives frontend/index.html in Chrome against a running server and saves
//   out/capture.webm   screen recording of the page
//   out/events.json    every server event with its time (ms since the recording started)
//   out/agent-N.mp3    the agent's spoken replies (for Whisper captions)
//
// Headless Chrome has no microphone, so getUserMedia is replaced with a stream that plays a
// pre-recorded caller clip. Everything after the mic -- the app's MediaRecorder, the WebSocket,
// Deepgram, the agent and text-to-speech -- runs unmodified.
//
// Usage: start the server (see README), then  node capture.mjs [http://127.0.0.1:8765]
import fs from 'node:fs';
import path from 'node:path';
import puppeteer from 'puppeteer-core';

const BASE = process.argv[2] || 'http://127.0.0.1:8765';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const OUT = path.resolve('out');
const VIEWPORT = {width: 520, height: 760};
fs.mkdirSync(OUT, {recursive: true});
// remove the previous capture's files: stale agent audio would give stale captions
for (const f of fs.readdirSync(OUT)) {
  if (/^(capture.(webm|mp4)|events.json|captions.json|agent-d+.(mp3|json)|caller-d+.json)$/.test(f)) fs.rmSync(path.join(OUT, f));
}

const clip = (file) => 'data:audio/mpeg;base64,' + fs.readFileSync(path.resolve('assets', file)).toString('base64');
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: true,
  args: ['--autoplay-policy=no-user-gesture-required', `--window-size=${VIEWPORT.width},${VIEWPORT.height}`],
});
const page = await browser.newPage();
await page.setViewport(VIEWPORT);

const events = [];
let recordingStartedAt = null;
let audioCount = 0;
await page.exposeFunction('__demoEvent', (raw) => {
  const data = JSON.parse(raw);
  const t = recordingStartedAt === null ? 0 : Date.now() - recordingStartedAt;
  if (data.type === 'audio') {  // keep the mp3 for Whisper, not the base64 in the log
    audioCount += 1;
    const file = `agent-${audioCount}.mp3`;
    fs.writeFileSync(path.join(OUT, file), Buffer.from(data.data, 'base64'));
    events.push({t, type: 'audio', file});
  } else {
    events.push({t, ...data});
  }
});

await page.evaluateOnNewDocument(() => {
  // 1) log every message the server sends on any WebSocket
  const NativeWS = window.WebSocket;
  window.WebSocket = class extends NativeWS {
    constructor(...args) {
      super(...args);
      this.addEventListener('message', (e) => window.__demoEvent(e.data));
    }
  };
  // 2) report when the agent's spoken reply finishes playing
  const NativeAudio = window.Audio;
  window.Audio = function (...args) {
    const audio = new NativeAudio(...args);
    audio.addEventListener('ended', () => window.__demoEvent(JSON.stringify({type: 'audio_ended'})));
    return audio;
  };
  // 3) the "microphone" plays window.__nextClip (a data: URL) instead of a real device
  navigator.mediaDevices.getUserMedia = async () => {
    const ctx = new AudioContext();
    const buffer = await ctx.decodeAudioData(await (await fetch(window.__nextClip)).arrayBuffer());
    const source = ctx.createBufferSource();
    const dest = ctx.createMediaStreamDestination();
    source.buffer = buffer;
    source.connect(dest);
    source.start();
    return dest.stream;
  };
});

const agentMessages = () => page.$$eval('#messages .message.agent', (els) => els.length);

async function askByText(question) {
  const before = await agentMessages();
  await page.type('#input', question, {delay: 45});
  await sleep(300);
  await page.keyboard.press('Enter');
  await page.waitForFunction((n) => document.querySelectorAll('#messages .message.agent').length > n,
    {timeout: 60000}, before);
  await sleep(1800);
}

async function askByVoice(clipFile, seconds) {
  const endedBefore = events.filter((e) => e.type === 'audio_ended' || e.type === 'error').length;
  await page.evaluate((url) => { window.__nextClip = url; }, clip(clipFile));
  await page.click('#mic-btn');                  // start recording
  events.push({t: Date.now() - recordingStartedAt, type: 'caller_speaking', clip: clipFile});
  await sleep(seconds * 1000 + 500);
  await page.click('#mic-btn');                  // stop -> audio goes to the server
  const deadline = Date.now() + 90000;            // wait until the spoken reply has finished playing
  while (events.filter((e) => e.type === 'audio_ended' || e.type === 'error').length === endedBefore) {
    if (Date.now() > deadline) throw new Error('reply audio did not finish within 90s');
    await sleep(100);
  }
  await sleep(1200);
}

await page.goto(`${BASE}/static/index.html`, {waitUntil: 'networkidle0'});
await page.waitForFunction(() => document.querySelector('#status-text').textContent === 'Connected');

const recorder = await page.screencast({path: path.join(OUT, 'capture.webm')});
recordingStartedAt = Date.now();
await sleep(1500);

await askByText('What are your opening hours?');
await askByText('How much is teeth whitening?');

await page.click('.tab:nth-child(2)');           // Voice Chat tab
await page.waitForFunction(() => document.querySelector('#status-voice').textContent === 'Connected');
await sleep(1200);
await askByVoice('caller-1.mp3', 6.0);
await askByVoice('caller-2.mp3', 5.1);
await sleep(1000);

await recorder.stop();
const durationMs = Date.now() - recordingStartedAt;
fs.writeFileSync(path.join(OUT, 'events.json'), JSON.stringify({durationMs, viewport: VIEWPORT, events}, null, 2));
await browser.close();

console.log(`captured ${(durationMs / 1000).toFixed(1)}s, ${events.length} events, ${audioCount} agent replies`);
for (const e of events) {
  const text = e.reply || e.text || e.message || e.file || e.clip || '';
  console.log(`  ${(e.t / 1000).toFixed(1).padStart(5)}s  ${e.type.padEnd(16)} ${e.intent ? `[${e.intent}] ` : ''}${String(text).slice(0, 80)}${e.booking_confirmed ? '  <- BOOKING SAVED' : ''}`);
}
