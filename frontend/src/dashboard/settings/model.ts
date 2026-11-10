// The settings API describes every editable field; drafts keep numbers as text until the server validates them.
export type Values = Record<string, string | boolean>;
export interface Setting {
  key: string;
  type: 'str' | 'int' | 'float' | 'bool' | 'list';
  value: string | number | boolean | (string | number)[] | null;
  default: string | number | boolean | (string | number)[] | null;
  secret: boolean;
  set?: boolean;
  description: string;
  env?: string | null;
  feature?: string;
  label?: string; // its plain words in Settings (config/labels.toml), "" for one shown by its key
  help?: string; // one line under the label
  section?: string; // which menu section shows it (catalog.ts), "" for Advanced only
  order?: number; // its place in labels.toml: how its section lists it
  group?: string; // the titled card it shares in its section (labels.toml); "" for none (the Other card)
  live?: boolean; // its services follow it without a restart (labels.toml "live")
  restarts?: string[]; // what saving it restarts: the services that read it, [] for a live one
}
export interface Section {
  name: string;
  description: string;
  settings: Setting[];
}
export interface ConfigView {
  ok: boolean;
  control: boolean;
  needs_setup: boolean;
  path: string;
  sections: Section[];
  restarts?: string[]; // what a save restarts (the save bar says so)
}
export interface ConfigReply {
  ok: boolean;
  error?: string;
  field?: string;
  overridden?: string[];
  url?: string;
  restarted?: string[]; // the services this save restarts
  restarting?: boolean; // whether the dashboard restarts too; false: stay on this page
}

/** What saving these settings restarts, in the order the view lists every candidate: each row's own list, or every
 * candidate for a row from a server that doesn't say. null when the server can't say what restarts. */
export function restartsFor(view: ConfigView, keys: string[]): string[] | null {
  const rows = new Map(view.sections.flatMap(s => s.settings).map(r => [r.key, r]));
  if (!view.restarts?.length && keys.some(k => !rows.get(k)?.restarts)) return null;
  const hit = new Set(keys.flatMap(k => rows.get(k)?.restarts ?? view.restarts ?? []));
  const order = view.restarts ?? [];
  return [...order.filter(n => hit.has(n)), ...[...hit].filter(n => !order.includes(n))];
}
export const CONFIG_KEY = ['config'] as const;
// The voice reference's query key. Here, not in VoiceRef.tsx, so Getting started can read it without pulling the
// recorder into the entry bundle.
export const VOICE_REF_KEY = ['config', 'voice_ref'] as const;
const acronyms: Record<string, string> = { asr: 'ASR', tts: 'TTS', vlm: 'VLM', url: 'URL', fov: 'FOV', llm: 'LLM' };
export const nice = (s: string): string =>
  s
    .replace(/[._]/g, ' ')
    .replace(/\b\w+/g, w => acronyms[w.toLowerCase()] ?? w)
    .replace(/^\w/, c => c.toUpperCase());
export const isRequired = (row: Setting): boolean =>
  !row.secret &&
  row.type !== 'bool' &&
  row.type !== 'list' &&
  (row.key === 'owner.name' || row.key === 'owner.phone' || (row.default !== '' && row.default != null));
export const blank = (value: Values[string] | undefined): boolean => value == null || String(value).trim() === '';
/** A list setting as the draft's text: its items joined with ", ", or as JSON when an item holds a comma (an ALSA
 *  device name like "USB Audio: - (hw:1,0)"), which the server reads back whole. */
export const listText = (list: readonly (string | number)[]): string =>
  list.some(x => String(x).includes(',')) ? JSON.stringify(list) : list.join(', ');

/** A list setting's items from the draft's text (listText's inverse). */
export function listItems(text: string | boolean | undefined): string[] {
  const s = String(text ?? '').trim();
  if (s.startsWith('[')) {
    try {
      const list: unknown = JSON.parse(s);
      if (Array.isArray(list)) return list.map(String).filter(Boolean);
    } catch {
      // typed by hand and not JSON after all: split it like any other
    }
  }
  return s
    .split(',')
    .map(x => x.trim())
    .filter(Boolean);
}

export function initialValues(view: ConfigView): Values {
  return Object.fromEntries(
    view.sections.flatMap(section =>
      section.settings.map(row => [
        row.key,
        row.secret
          ? ''
          : row.type === 'bool'
            ? Boolean(row.value)
            : Array.isArray(row.value)
              ? listText(row.value)
              : String(row.value ?? ''),
      ]),
    ),
  );
}
/** Where to go after a save that restarts the dashboard (maybe on another port): the same page there, Settings
 *  included, on the section it was on. */
export function restartDestination(replyUrl: string | undefined, currentHref: string = location.href): string {
  const current = new URL(currentHref);
  const target = new URL(replyUrl ?? currentHref, currentHref);
  target.pathname = current.pathname;
  target.search = current.search;
  target.hash = current.hash;
  return target.href;
}

/** What Settings calls a setting: its plain label, else a new feature's name, else its key's last part in words. */
export const labelOf = (row: Setting): string =>
  row.label || (row.feature && row.type === 'bool' ? row.feature : '') || nice(row.key.split('.').at(-1) ?? row.key);

/** The line under a setting's label: its help, else its comment in defaults.toml, and for a new feature whether it
 *  is on by default. */
export function helpOf(row: Setting): string {
  // A feature's default isn't said on every line any more: the row marks one that differs from it (SettingRow)
  return row.help || row.description;
}

/** Whether, after a restart, the page to show is this one again (only its #hash may differ): then it must reload,
 *  since assigning a URL that differs only by its hash doesn't. */
export const sameDocument = (url: string, current: string = location.href): boolean =>
  new URL(url, current).href.split('#')[0] === new URL(current).href.split('#')[0];
