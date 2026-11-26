// Turns a capture into docs/demo.gif:
//   1. out/frames.json (timestamped frames) -> out/capture.mp4 at a constant 25 fps for Remotion
//   2. HyperFrames renders the intro card -> out/intro.mp4
//   3. Remotion renders the composition (src/Demo.tsx) -> out/demo.mp4
//   4. FFmpeg two-pass palette GIF -> ../docs/demo.gif
// Usage: node render.mjs   (needs ffmpeg on PATH; run capture.mjs and captions.mjs first)
import {execSync} from 'node:child_process';
import fs from 'node:fs';

const run = (cmd, opts = {}) => {
  console.log(`> ${cmd}`);
  execSync(cmd, {stdio: 'inherit', ...opts});
};
const GIF_FPS = 12;
const GIF_WIDTH = 800;

// Resample the timestamped frames to a constant 25 fps: slot k (at k * 40 ms) shows the frame that
// was on screen at that moment. Each slot becomes one numbered file (a hard link, so no copying),
// and FFmpeg reads a plain image sequence -- duration is exactly slots / 25. (Feeding FFmpeg's concat
// demuxer per-frame durations instead made the video ~4% longer than real time.)
const FPS = 25;
const realMs = JSON.parse(fs.readFileSync('out/events.json', 'utf8')).durationMs;
const frameTimes = JSON.parse(fs.readFileSync('out/frames.json', 'utf8'));
fs.rmSync('out/slots', {recursive: true, force: true});
fs.mkdirSync('out/slots');
const slots = Math.round((realMs / 1000) * FPS);
for (let slot = 0, i = 0; slot < slots; slot++) {
  while (i + 1 < frameTimes.length && frameTimes[i + 1].t <= (slot * 1000) / FPS) i++;
  fs.linkSync(`out/${frameTimes[i].file}`, `out/slots/${String(slot).padStart(5, '0')}.jpg`);
}
run(`ffmpeg -v error -y -framerate ${FPS} -i out/slots/%05d.jpg -c:v libx264 -pix_fmt yuv420p -crf 18 out/capture.mp4`);

// guard: the video must last as long as the capture did, or the event-driven panel drifts out of sync
const videoSeconds = Number(execSync('ffprobe -v error -show_entries format=duration -of csv=p=0 out/capture.mp4').toString());
const realSeconds = realMs / 1000;
if (Math.abs(videoSeconds - realSeconds) / realSeconds > 0.01) {
  throw new Error(`recording is ${videoSeconds.toFixed(1)}s but the capture ran ${realSeconds.toFixed(1)}s -- out of sync`);
}
console.log(`sync ok: video ${videoSeconds.toFixed(2)}s, capture ${realSeconds.toFixed(2)}s`);
run('hyperframes render -o ../out/intro.mp4 --quiet', {
  cwd: 'intro', env: {...process.env, HYPERFRAMES_NO_TELEMETRY: '1', HYPERFRAMES_SKIP_SKILLS: '1'},
});
run('npx remotion render src/index.ts Demo out/demo.mp4 --public-dir=out --log=error');

fs.mkdirSync('../docs', {recursive: true});
const filters = `fps=${GIF_FPS},scale=${GIF_WIDTH}:-1:flags=lanczos`;
run(`ffmpeg -v error -y -i out/demo.mp4 -vf "${filters},palettegen=stats_mode=diff" out/palette.png`);
run(`ffmpeg -v error -y -i out/demo.mp4 -i out/palette.png -lavfi "${filters} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle" ../docs/demo.gif`);
console.log(`docs/demo.gif: ${(fs.statSync('../docs/demo.gif').size / 1024 / 1024).toFixed(1)} MB`);
