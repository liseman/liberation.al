// Which cameras' and views' points and Gaussians the 3D view draws (#409): the owner switches cameras off one at a
// time, or one camera's earlier views, to find which makes the map look wrong. What's hidden is a set of view ids
// (kept frames' ids), kept per place for the browser tab's session, so a newer map of the same place keeps them
// hidden and a view it adds is shown. Nothing here touches the GPU.
import type { Sources, SourceView } from '../api';
import { UNTAGGED } from './load';

const KEY = 'places.cloud.hidden.';

type Store = Pick<Storage, 'getItem' | 'setItem' | 'removeItem'>;

const session = (): Store | null => {
  try {
    return sessionStorage;
  } catch {
    return null; // storage refused (a private window, or blocked site data): nothing is kept
  }
};

/** The view ids hidden for `place` this session. */
export function readHidden(place: string, store: Store | null = session()): Set<string> {
  try {
    const got: unknown = JSON.parse(store?.getItem(KEY + place) ?? '[]');
    return new Set(Array.isArray(got) ? got.filter((v): v is string => typeof v === 'string') : []);
  } catch {
    return new Set();
  }
}

/** Keep `hidden` for `place` this session (nothing kept when it's empty). */
export function writeHidden(place: string, hidden: ReadonlySet<string>, store: Store | null = session()): void {
  try {
    if (hidden.size) store?.setItem(KEY + place, JSON.stringify([...hidden]));
    else store?.removeItem(KEY + place);
  } catch {
    // storage full or refused: the choice holds until the page closes
  }
}

/** The indices into `sources.views` to draw, or null when every view is (so nothing needs masking). */
export function visibleViews(sources: Sources, hidden: ReadonlySet<string>): Set<number> | null {
  if (!sources.views.some(v => hidden.has(v.id))) return null;
  return new Set(sources.views.flatMap((v, i) => (hidden.has(v.id) ? [] : [i])));
}

/** A camera's box: on when all its views are, off when none is, and mixed between. */
export function cameraState(views: readonly SourceView[], hidden: ReadonlySet<string>): 'on' | 'off' | 'mixed' {
  const off = views.filter(v => hidden.has(v.id)).length;
  return off === 0 ? 'on' : off === views.length ? 'off' : 'mixed';
}

/** `hidden` with `ids` shown (`show`) or hidden. */
export function toggled(hidden: ReadonlySet<string>, ids: readonly string[], show: boolean): Set<string> {
  const out = new Set(hidden);
  for (const id of ids) {
    if (show) out.delete(id);
    else out.add(id);
  }
  return out;
}

/** How many of a camera's or a view's points (Points) or Gaussians (Splat) there are, as a short count. */
export function countText(n: number | null, noun: 'points' | 'splats'): string {
  if (n == null) return '';
  const short =
    n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 10_000 ? `${Math.round(n / 1000)}k` : n.toLocaleString('en-US');
  return `${short} ${n === 1 ? noun.slice(0, -1) : noun}`;
}

/** The view mask texture's width: a row of this many Gaussians. */
export const MASK_WIDTH = 2048;

/** Which of `views` to draw, as one byte each (255 drawn, 0 hidden), `width` to a row and padded to whole rows. */
export function viewMask(views: Uint16Array, visible: ReadonlySet<number> | null, width = MASK_WIDTH): Uint8Array {
  const out = new Uint8Array(Math.max(1, Math.ceil(views.length / width)) * width);
  for (let i = 0; i < views.length; i++) {
    const v = views[i] ?? UNTAGGED;
    out[i] = !visible || v === UNTAGGED || visible.has(v) ? 255 : 0;
  }
  return out;
}

/** The squares (two triangles, four corners) of the points of `views` to draw: all of them with `visible` null. */
export function pointIndices(
  count: number,
  views: Uint16Array | null,
  visible: ReadonlySet<number> | null,
): Uint32Array {
  let n = 0;
  const drawn = (i: number) => {
    const v = views?.[i] ?? UNTAGGED;
    return !visible || v === UNTAGGED || visible.has(v);
  };
  for (let i = 0; i < count; i++) if (drawn(i)) n++;
  const out = new Uint32Array(n * 6);
  for (let i = 0, at = 0; i < count; i++)
    if (drawn(i)) {
      out.set([i * 4, i * 4 + 1, i * 4 + 2, i * 4, i * 4 + 2, i * 4 + 3], at);
      at += 6;
    }
  return out;
}
