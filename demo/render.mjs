// Turns a capture into docs/demo.gif:
//   1. out/capture.webm -> out/capture.mp4 (constant frame rate, seekable for Remotion)
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

run('ffmpeg -v error -y -i out/capture.webm -r 25 -c:v libx264 -pix_fmt yuv420p -crf 18 out/capture.mp4');
run('hyperframes render -o ../out/intro.mp4 --quiet', {
  cwd: 'intro', env: {...process.env, HYPERFRAMES_NO_TELEMETRY: '1', HYPERFRAMES_SKIP_SKILLS: '1'},
});
run('npx remotion render src/index.ts Demo out/demo.mp4 --public-dir=out --log=error');

fs.mkdirSync('../docs', {recursive: true});
const filters = `fps=${GIF_FPS},scale=${GIF_WIDTH}:-1:flags=lanczos`;
run(`ffmpeg -v error -y -i out/demo.mp4 -vf "${filters},palettegen=stats_mode=diff" out/palette.png`);
run(`ffmpeg -v error -y -i out/demo.mp4 -i out/palette.png -lavfi "${filters} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle" ../docs/demo.gif`);
console.log(`docs/demo.gif: ${(fs.statSync('../docs/demo.gif').size / 1024 / 1024).toFixed(1)} MB`);
