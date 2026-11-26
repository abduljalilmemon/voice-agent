// The demo video: HyperFrames intro -> real screen recording with a live panel -> end card.
// Everything on the panel comes from the real run: out/events.json (server events with times)
// and out/captions.json (Whisper word timings). Nothing is scripted by hand.
import React from 'react';
import {
  AbsoluteFill, Easing, interpolate, OffthreadVideo, Sequence, staticFile, useCurrentFrame,
} from 'remotion';
import capture from '../out/events.json';
import captions from '../out/captions.json';

export const FPS = 25;
export const SPEED = 1.5;          // the recording plays faster; the panel says so
export const INTRO_SECONDS = 3;
export const OUTRO_SECONDS = 3.5;

type Event = {t: number; type: string; [key: string]: unknown};
type Word = {text: string; start: number; end: number};
type Caption = {speaker: 'caller' | 'agent'; start: number; words: Word[]};

const events = capture.events as Event[];
const sec = (e: Event) => e.t / 1000;
const COLORS = {bg: '#0f2a44', panel: '#13355a', line: '#2c5a85', text: '#ffffff', dim: '#9cc3e6', accent: '#4ade80'};
const FONT = '"Segoe UI", Arial, sans-serif';

// ── timeline derived from the real events ──────────────────────
type Stage = 'stt' | 'agent' | 'tts' | 'speaking' | null;

function stageAt(t: number): Stage {
  let stage: Stage = null;
  for (const e of events) {
    if (sec(e) > t) break;
    if (e.type === 'typing') stage = 'agent';
    else if (e.type === 'message') stage = null;
    else if (e.type === 'status' && e.message === 'Transcribing...') stage = 'stt';
    else if (e.type === 'status' && e.message === 'Thinking...') stage = 'agent';
    else if (e.type === 'status' && e.message === 'Generating voice...') stage = 'tts';
    else if (e.type === 'audio') stage = 'speaking';
    else if (e.type === 'audio_ended') stage = null;
  }
  return stage;
}

type FeedItem = {t: number; text: string; good?: boolean};

function buildFeed(): FeedItem[] {
  const feed: FeedItem[] = [];
  let typingAt = 0;
  let transcribingAt = 0;
  let thinkingAt = 0;
  for (const e of events) {
    if (e.type === 'typing') typingAt = sec(e);
    if (e.type === 'message') {
      feed.push({t: sec(e), text: `Intent ${e.intent} · answered from the knowledge base in ${(sec(e) - typingAt).toFixed(1)}s`});
    }
    if (e.type === 'status' && e.message === 'Transcribing...') transcribingAt = sec(e);
    if (e.type === 'transcript') feed.push({t: sec(e), text: `Deepgram transcribed the caller in ${(sec(e) - transcribingAt).toFixed(1)}s`});
    if (e.type === 'status' && e.message === 'Thinking...') thinkingAt = sec(e);
    if (e.type === 'reply') {
      feed.push({t: sec(e), text: `Intent ${e.intent} · agent replied in ${(sec(e) - thinkingAt).toFixed(1)}s`});
      if (e.booking_confirmed) feed.push({t: sec(e), text: `Booking #${e.booking_id} saved to the database`, good: true});
    }
  }
  return feed;
}

const FEED = buildFeed();
const BOOKING = events.find((e) => e.booking_confirmed) as Event | undefined;

// ── pieces ─────────────────────────────────────────────────────
const Chip: React.FC<{label: string; active: boolean}> = ({label, active}) => (
  <div style={{
    flex: 1, textAlign: 'center', padding: '9px 4px', borderRadius: 10, fontSize: 15, fontWeight: 600,
    color: active ? '#0f2a44' : COLORS.dim, background: active ? COLORS.accent : 'transparent',
    border: `1px solid ${active ? COLORS.accent : COLORS.line}`,
  }}>{label}</div>
);

const CaptionCard: React.FC<{t: number}> = ({t}) => {
  const current = (captions as Caption[]).find((c) => t >= c.start && t <= c.start + c.words.at(-1)!.end + 0.8);
  if (!current) {
    return <div style={{color: COLORS.dim, fontSize: 18, fontStyle: 'italic'}}>Voice captions appear here (the GIF has no sound)</div>;
  }
  const local = t - current.start;
  return (
    <div>
      <div style={{fontSize: 14, letterSpacing: '0.12em', textTransform: 'uppercase', color: COLORS.dim, marginBottom: 8}}>
        {current.speaker === 'caller' ? 'Caller speaking' : 'Agent speaking'}
      </div>
      <div style={{fontSize: 22, lineHeight: 1.45, fontWeight: 500}}>
        {current.words.map((w, i) => (
          <span key={i} style={{color: local >= w.start ? COLORS.text : '#4f7aa3'}}>{w.text} </span>
        ))}
      </div>
    </div>
  );
};

const Main: React.FC = () => {
  const frame = useCurrentFrame();
  const t = (frame / FPS) * SPEED;   // seconds into the real recording
  const stage = stageAt(t);
  const shown = FEED.filter((f) => f.t <= t).slice(-4);
  const {width, height} = capture.viewport;
  const scale = 560 / height;

  return (
    <AbsoluteFill style={{background: COLORS.bg, fontFamily: FONT, color: COLORS.text}}>
      <div style={{
        position: 'absolute', left: 28, top: 20, width: width * scale, height: 560,
        borderRadius: 14, overflow: 'hidden', boxShadow: '0 12px 40px rgba(0,0,0,0.45)',
      }}>
        <OffthreadVideo src={staticFile('capture.mp4')} playbackRate={SPEED} muted
          style={{width: width * scale, height: 560}} />
      </div>

      <div style={{position: 'absolute', left: 450, top: 28, width: 482, display: 'flex', flexDirection: 'column', gap: 18}}>
        <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
          <div style={{fontSize: 15, letterSpacing: '0.14em', textTransform: 'uppercase', color: COLORS.dim}}>What's happening</div>
          <div style={{fontSize: 13, color: COLORS.dim, border: `1px solid ${COLORS.line}`, borderRadius: 999, padding: '3px 10px'}}>
            real run · {SPEED}× speed
          </div>
        </div>

        <div style={{display: 'flex', gap: 8}}>
          <Chip label="Speech-to-text" active={stage === 'stt'} />
          <Chip label="Agent" active={stage === 'agent'} />
          <Chip label="Text-to-speech" active={stage === 'tts' || stage === 'speaking'} />
        </div>

        <div style={{background: COLORS.panel, borderRadius: 14, padding: '18px 20px', minHeight: 160}}>
          <CaptionCard t={t} />
        </div>

        <div style={{display: 'flex', flexDirection: 'column', gap: 9}}>
          {shown.map((f, i) => (
            <div key={`${f.t}-${i}`} style={{
              fontSize: 16, padding: '9px 12px', borderRadius: 10,
              background: f.good ? 'rgba(74,222,128,0.16)' : 'rgba(255,255,255,0.05)',
              color: f.good ? COLORS.accent : '#d5e5f5', fontWeight: f.good ? 700 : 400,
              opacity: interpolate(t - f.t, [0, 0.4], [0, 1], {extrapolateRight: 'clamp'}),
            }}>{f.good ? '✓ ' : ''}{f.text}</div>
          ))}
        </div>
      </div>
    </AbsoluteFill>
  );
};

const Outro: React.FC = () => {
  const frame = useCurrentFrame();
  const appear = (delay: number) => ({
    opacity: interpolate(frame, [delay, delay + 10], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}),
    transform: `translateY(${interpolate(frame, [delay, delay + 12], [14, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.out(Easing.cubic)})}px)`,
  });
  return (
    <AbsoluteFill style={{
      background: 'radial-gradient(circle at 50% 40%, #1f4e79 0%, #0f2a44 70%)', fontFamily: FONT, color: COLORS.text,
      justifyContent: 'center', alignItems: 'center', textAlign: 'center',
    }}>
      <div style={{fontSize: 44, fontWeight: 700, color: COLORS.accent, ...appear(0)}}>
        ✓ Booking #{String(BOOKING?.booking_id ?? '')} saved
      </div>
      <div style={{fontSize: 22, color: '#d5e5f5', marginTop: 16, ...appear(8)}}>
        Two questions by text, then a booking made by voice
      </div>
      <div style={{fontSize: 18, color: COLORS.dim, marginTop: 34, ...appear(16)}}>
        github.com/abduljalilmemon/voice-agent
      </div>
    </AbsoluteFill>
  );
};

export const Demo: React.FC = () => {
  const introFrames = INTRO_SECONDS * FPS;
  const mainFrames = Math.ceil((capture.durationMs / 1000 / SPEED) * FPS);
  return (
    <AbsoluteFill style={{background: COLORS.bg}}>
      <Sequence durationInFrames={introFrames}>
        <OffthreadVideo src={staticFile('intro.mp4')} muted />
      </Sequence>
      <Sequence from={introFrames} durationInFrames={mainFrames}>
        <Main />
      </Sequence>
      <Sequence from={introFrames + mainFrames}>
        <Outro />
      </Sequence>
    </AbsoluteFill>
  );
};
