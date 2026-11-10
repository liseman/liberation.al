// The live feed's own state, outside the query cache: whether the stream is up, and a clock for ages and staleness.
import { create } from 'zustand';

export interface Connection {
  state: '' | 'on' | 'bad';
  text: 'connecting' | 'live' | 'polling' | 'reconnecting';
}

export interface LiveStatus {
  conn: Connection;
  now: number; // seconds; ticks every few seconds while the feed runs
}

export const useLiveStatus = create<LiveStatus>(() => ({
  conn: { state: '', text: 'connecting' },
  now: Date.now() / 1000,
}));
