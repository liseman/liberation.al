// Shared display names for walk-bys and see-actions' model actions. Other labels pass through.

const NAMES: Record<string, string> = {
  crossing: 'Walked past',
  sit_down: 'Sat down',
  stand_up: 'Stood up',
  pick_up: 'Picked something up',
  put_down: 'Put something down',
  fall: 'Fell',
  drink: 'Drinking',
  eat: 'Eating',
  phone_call: 'On a phone call',
  phone_use: 'Using a phone',
};

const HINTS: Record<string, string> = {
  crossing: "A person crossed the camera's counting line.",
};

// Words an activity's name keeps in capitals ("watch tv" is "Watch TV").
const ACRONYMS = new Set(['tv', 'pc', 'dvd', 'vr', 'bbq', 'atm', 'gps', 'usb', 'id', 'ok']);

/** A kind's display name: "crossing" is "Walked past". Any other kind reads as words in sentence case, its
 *  underscores as spaces and known acronyms in capitals ("watch_tv" is "Watch TV", "read" is "Read"), so it sits
 *  beside "Eating" as a name, not a raw label. A name already given its capitals keeps them. */
export function activityName(kind: string): string {
  if (Object.hasOwn(NAMES, kind)) return NAMES[kind] ?? kind;
  const words = kind
    .replaceAll('_', ' ')
    .split(/\s+/)
    .filter(Boolean)
    .map(w => (ACRONYMS.has(w.toLowerCase()) ? w.toUpperCase() : w));
  const text = words.join(' ');
  return text ? text.charAt(0).toLocaleUpperCase() + text.slice(1) : kind;
}

/** A one-line explanation of a kind for its tooltip, or null when its name says enough. */
export function activityHint(kind: string | null | undefined): string | null {
  return kind && Object.hasOwn(HINTS, kind) ? (HINTS[kind] ?? null) : null;
}

/** An event series' name as Reports shows it: a kind's plain name when the name is a raw kind (an event the owner
 *  never named), else the name as given. */
export function eventName(name: string): string {
  return Object.hasOwn(NAMES, name) ? (NAMES[name] ?? name) : name;
}
