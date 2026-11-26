// Captions for the silent GIF: Whisper supplies WHEN each word is spoken, the known text supplies
// WHAT is said (the caller's script, and the agent's reply text, which is exactly what TTS spoke).
// Whisper's small model mishears names ("Ali Khan" -> "Ollie Kahn"), so known words are mapped onto
// Whisper's timeline by their relative position in the sentence instead of trusting its spelling.
//   in:  out/events.json, assets/clips.json, assets/caller-*.mp3, out/agent-*.mp3
//   out: out/captions.json   [{speaker, start, words: [{text, start, end}]}]  (seconds in the recording)
//
// Usage: node captions.mjs   (WHISPER_BIN overrides the whisper executable)
import {execFileSync} from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const WHISPER = process.env.WHISPER_BIN || 'C:/Users/Abdul Jalil/Tools/whisper/Scripts/whisper.exe';
const OUT = path.resolve('out');
const {events} = JSON.parse(fs.readFileSync(path.join(OUT, 'events.json'), 'utf8'));

function transcribe(file) {
  const json = path.join(OUT, path.basename(file).replace(/\.mp3$/, '.json'));
  if (!fs.existsSync(json)) {
    execFileSync(WHISPER, [file, '--model', 'base.en', '--language', 'en', '--word_timestamps', 'True',
      '--output_format', 'json', '--output_dir', OUT, '--fp16', 'False'], {stdio: 'ignore'});
  }
  return JSON.parse(fs.readFileSync(json, 'utf8')).segments.flatMap((s) => s.words)
    .map((w) => ({text: w.word.trim(), start: w.start, end: w.end}));
}

// Map each known word to the time Whisper heard the same relative position of the sentence.
function align(knownText, heard) {
  const timeline = [];  // [charOffset, seconds] points along what Whisper heard
  let offset = 0;
  for (const w of heard) {
    timeline.push([offset, w.start]);
    offset += w.text.length + 1;
    timeline.push([offset - 1, w.end]);
  }
  const timeAt = (fraction) => {
    const target = fraction * offset;
    for (let i = 1; i < timeline.length; i++) {
      const [c0, t0] = timeline[i - 1];
      const [c1, t1] = timeline[i];
      if (target <= c1) return t0 + (c1 > c0 ? (target - c0) / (c1 - c0) : 0) * (t1 - t0);
    }
    return timeline.at(-1)[1];
  };
  const words = knownText.split(/\s+/).filter(Boolean);
  const total = words.reduce((n, w) => n + w.length + 1, 0);
  let pos = 0;
  return words.map((text) => {
    const word = {text, start: timeAt(pos / total), end: timeAt((pos + text.length) / total)};
    pos += text.length + 1;
    return word;
  });
}

const clipText = JSON.parse(fs.readFileSync(path.resolve('assets', 'clips.json'), 'utf8'));
const captions = [];
let lastReply = '';
for (const e of events) {
  if (e.type === 'reply') lastReply = e.text;
  if (e.type === 'caller_speaking') {
    // the fake mic starts playing the clip when recording starts
    const heard = transcribe(path.resolve('assets', e.clip));
    captions.push({speaker: 'caller', start: e.t / 1000, words: align(clipText[e.clip], heard)});
  } else if (e.type === 'audio') {
    // the frontend plays the reply as soon as it arrives; TTS spoke exactly the reply text
    const heard = transcribe(path.join(OUT, e.file));
    captions.push({speaker: 'agent', start: e.t / 1000, words: align(lastReply, heard)});
  }
}
fs.writeFileSync(path.join(OUT, 'captions.json'), JSON.stringify(captions, null, 2));
for (const c of captions) {
  const last = c.words.at(-1);
  console.log(`${c.start.toFixed(1).padStart(5)}s ${c.speaker.padEnd(6)} ${last.end.toFixed(1)}s long: ${c.words.map((w) => w.text).join(' ')}`);
}
