// Readable text for the loosely shaped JSON the Ask server returns. Every field is untrusted and may be missing.

export type Rec = Record<string, unknown>;

export const isRec = (value: unknown): value is Rec => !!value && typeof value === 'object' && !Array.isArray(value);

/** The objects in a list; anything else (null, numbers, nested lists) is dropped. */
export const records = (value: unknown): Rec[] => (Array.isArray(value) ? value.filter(isRec) : []);

export const list = (value: unknown): unknown[] => (Array.isArray(value) ? value : []);

/** A string, number, or boolean as text; anything else is empty. */
export const plain = (value: unknown): string =>
  typeof value === 'string' ? value : typeof value === 'number' || typeof value === 'boolean' ? String(value) : '';

const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);

/** A Date from epoch seconds, epoch milliseconds, or a date string; null when it is none of those. */
export function dateOf(value: unknown): Date | null {
  if (value === null || value === undefined || value === '') return null;
  if (typeof value !== 'number' && typeof value !== 'string') return null;
  const d = new Date(typeof value === 'number' ? (value < 1e12 ? value * 1000 : value) : value);
  return Number.isNaN(d.getTime()) ? null : d;
}

export const duration = (value: unknown): string =>
  finite(value) && value >= 0 ? `${value.toFixed(value < 10 ? 2 : 1)}s` : '';

export const nameOf = (value: Rec | null | undefined): string =>
  plain(value?.model) || plain(value?.name) || 'Unknown model';

export const pct = (v: unknown): string => (finite(v) ? `${Math.round(v * 100)}%` : 'unknown');
export const minutes = (v: unknown): string => (finite(v) ? `${Math.round(v / 60)} min` : 'unknown');
export const clock = (v: unknown): string =>
  dateOf(v)?.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) ?? '';

/** A field name as a label: first_seen is "First seen". */
export const label = (key: string): string => key.replace(/_/g, ' ').replace(/^./, x => x.toUpperCase());

const LABELS: Record<string, string> = {
  ok: 'Answered',
  partial: 'Partial answer',
  error: 'Failed',
  timeout: 'Timed out',
  unavailable: 'Unavailable',
  provider_unavailable: 'Unavailable',
  synthesis_failed: 'Summary unavailable',
  busy: 'Busy',
  blocked_sensitive: 'Not sent',
  invalid: 'Invalid request',
  input_too_long: 'Request too long',
};

export function statusLabel(value: unknown): string {
  const key = plain(value);
  return (Object.hasOwn(LABELS, key) && LABELS[key]) || key.replace(/_/g, ' ') || 'No answer';
}

const REASONS: Record<string, string> = {
  timeout: 'The model timed out before returning an answer.',
  provider_unavailable: 'This model is currently unavailable.',
  no_final_answer_within_token_budget: 'The model did not return a final answer within its token budget.',
  FileNotFoundError: 'A required local program could not be found.',
  backend_error: 'The model service reported an error.',
};

/** A readable sentence for an error that may be a string, a code, or a nested object. */
export function reason(value: unknown): string {
  for (let depth = 0; depth < 3 && isRec(value); depth += 1) {
    value = value.message ?? value.reason ?? value.error ?? value.code ?? value.name ?? '';
  }
  const code = plain(value);
  return (
    (Object.hasOwn(REASONS, code) && REASONS[code]) || code.replace(/_/g, ' ') || 'No further details were returned.'
  );
}

/** One line of a notes list: a plain string, or "model - reason" for a structured note. */
export function noteText(item: unknown): string {
  if (typeof item === 'string') return item;
  if (isRec(item))
    return [plain(item.model), reason(item.reason || item.error || item.message)].filter(Boolean).join(' - ');
  return plain(item);
}

/** One round of the search-or-answer loop (memory/recall.py), as a line. */
export function roundText(r: Rec): string {
  if (r.decision === 'answer') return 'Enough evidence; answered.';
  if (r.decision === 'error') return `Skipped: ${plain(r.error)}`;
  const args = isRec(r.args)
    ? ` (${Object.entries(r.args)
        .map(([k, v]) => `${k} ${plain(v)}`)
        .join(', ')})`
    : '';
  const fresh = Array.isArray(r.new) && r.new.length ? `new ${r.new.map(plain).join(' ')}` : 'nothing new';
  return `${plain(r.tool)}${args}: ${plain(r.found ?? 0)} found, ${fresh}`;
}
