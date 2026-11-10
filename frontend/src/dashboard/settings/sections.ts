// Which Settings sections are open, remembered per browser. Storage can be missing or throw (a private window,
// blocked site data), so every read and write is guarded and the page works without it.
import { blank, isRequired, type Section, type Values } from './model';

export const OPEN_KEY = 'liberational.settings.open';
/** The accordion item that holds the new-feature switches. */
export const FEATURES = 'new-features';

export function loadOpen(): string[] {
  try {
    const parsed: unknown = JSON.parse(localStorage.getItem(OPEN_KEY) ?? '[]');
    return Array.isArray(parsed) ? parsed.filter((x): x is string => typeof x === 'string') : [];
  } catch {
    return [];
  }
}

export function saveOpen(open: readonly string[]): void {
  try {
    localStorage.setItem(OPEN_KEY, JSON.stringify(open));
  } catch {
    // Not remembered this time; the sections still open and close.
  }
}

/** What a section's header says about it: an unsaved change, or something to fix (a required field left blank, or
 * the field the last save refused). */
export interface SectionState {
  changed: boolean;
  trouble: boolean;
}

export function sectionState(section: Section, values: Values, saved: Values, failedField?: string): SectionState {
  return {
    changed: section.settings.some(row => values[row.key] !== saved[row.key]),
    trouble: section.settings.some(
      row => (isRequired(row) && blank(values[row.key])) || (failedField !== undefined && row.key === failedField),
    ),
  };
}

/** The accordion item a field id (set-<key>) lives in: New features for a feature switch, else its section. */
export function itemOf(id: string, sections: readonly Section[]): string | null {
  const key = id.replace(/^set-/, '');
  for (const section of sections)
    for (const row of section.settings)
      if (row.key === key) return row.feature && row.type === 'bool' ? FEATURES : section.name;
  return null;
}
