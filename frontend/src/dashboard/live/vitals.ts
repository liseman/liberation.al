// Live's heart rates: the newest confident reading for each person a close camera measured (GET /vitals,
// dashboard/vitals.py, from what see-actions reads off the colour of skin). The server has already dropped every
// reading it could not trust and every one older than twenty seconds, so a person in this answer has a rate worth
// showing; one who is absent simply gets nothing on the list.
import { request } from '../../shared/http';

export const VITALS_MS = 15_000; // as often as the here-now list itself

export interface VitalsReading {
  bpm: number; // beats a minute, rounded: it is an estimate
  confidence: number; // 0 to 1, the measurement's own confidence
  face_px: number; // the face's side in pixels, the window's median
  at: number; // when the reading's window ended, epoch seconds
}

export interface Vitals {
  at: number; // when the server answered
  people: Record<string, VitalsReading>; // keyed by the normalized name the People list matches by
}

export const VITALS_KEY = ['vitals'] as const;

export const getVitals = (signal?: AbortSignal) => request<Vitals>('/vitals', undefined, signal);

/** The normalized name the server keys a reading by: whitespace collapsed and case folded, as registry.norm has it. */
export const normName = (name: string | null | undefined): string =>
  (name ?? '').trim().replace(/\s+/g, ' ').toLocaleLowerCase();
