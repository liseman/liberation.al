// Review keeps its place in the URL's hash (#day=YYYY-MM-DD&t=<unix s>&person=<name>), so a link or a reload lands
// on the same moment. Other tabs link into Review with reviewURL().
import { href } from '../router';

export const DAY_RE = /^\d{4}-\d{2}-\d{2}$/;

export interface ReviewPlace {
  day?: string;
  t?: number;
  person?: string;
  /** Confirmed people in selection order; the last name is the person tab. */
  people?: string[];
  cam?: string;
  cameras?: string[];
  play?: boolean;
}

/** The hash for a place in Review: "#person=Dana&day=2026-10-28&t=1793212345.50". Empty fields are left out, and a
 *  whole-second t has no decimals ("t=1793212345"). */
export function reviewHash({ person, people, day, t, cam, cameras, play }: ReviewPlace): string {
  const q = new URLSearchParams();
  const names = [...new Set((people ?? (person ? [person] : [])).filter(Boolean))];
  if (names.length) q.set('person', '');
  if (cam) q.set('cam', cam);
  for (const camera of new Set(cameras)) q.append('cameras', camera);
  if (play) q.set('play', '1');
  if (day) q.set('day', day);
  if (t != null && Number.isFinite(t)) q.set('t', Number.isInteger(t) ? String(t) : t.toFixed(2));
  // Commas separate encoded names. A comma inside a name stays %2C, including in old single-person links.
  const encodedNames = names.map(name => new URLSearchParams({ person: name }).toString().slice(7));
  return `#${q.toString().replace(/^person=/, `person=${encodedNames.join(',')}`)}`;
}

/** A link to a place in Review. */
export const reviewURL = (place: ReviewPlace): string => href('/review') + reviewHash(place);

/** The place in the current hash. A day that isn't YYYY-MM-DD, or a t that isn't a number, is left out. */
export function readHash(hash = location.hash): ReviewPlace {
  const raw = hash.replace(/^#/, '');
  const q = new URLSearchParams(raw);
  const personField = raw.split('&').find(field => new URLSearchParams(field).has('person'));
  const names = personField?.includes('=') ? personField.slice(personField.indexOf('=') + 1).split(',') : [];
  const people = [
    ...new Set(names.map(name => new URLSearchParams(`person=${name}`).get('person') ?? '').filter(Boolean)),
  ];
  const day = q.get('day') ?? '';
  const t = Number.parseFloat(q.get('t') ?? '');
  return {
    person: people.at(-1) ?? '',
    people,
    ...(q.has('cameras') ? { cameras: [...new Set(q.getAll('cameras').filter(Boolean))] } : {}),
    ...(q.get('cam') ? { cam: q.get('cam') ?? undefined } : {}),
    ...(q.get('play') === '1' ? { play: true } : {}),
    day: DAY_RE.test(day) ? day : undefined,
    t: Number.isFinite(t) ? t : undefined,
  };
}
