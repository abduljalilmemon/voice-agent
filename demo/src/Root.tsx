import React from 'react';
import {Composition} from 'remotion';
import capture from '../out/events.json';
import {Demo, FPS, INTRO_SECONDS, OUTRO_SECONDS, SPEED} from './Demo';

const mainSeconds = capture.durationMs / 1000 / SPEED;

export const Root: React.FC = () => (
  <Composition
    id="Demo"
    component={Demo}
    fps={FPS}
    width={960}
    height={600}
    durationInFrames={Math.ceil((INTRO_SECONDS + mainSeconds + OUTRO_SECONDS) * FPS)}
  />
);
