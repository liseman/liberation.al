// The 2D plan's grid fades out a little way past what is drawn, instead of running to the edges of the view, so the
// walls, doors, and room labels read first. The editor draws its grid over the whole view; plan-editor.css masks it
// with the mask made here, in the plan's own coordinates (metres, the same space the editor draws its walls in), so
// zooming and panning move the mask with the plan. The box is what the editor has drawn (its walls, openings, and
// rooms, and the camera spots), measured again whenever the drawing changes. A plan with nothing drawn yet keeps
// the whole grid, to draw on.

/** How far past what is drawn the grid stays at full strength, and how far it then takes to fade out (metres). */
export const MARGIN_M = 0.75;
export const FADE_M = 1.25;

export interface Box {
  x: number;
  y: number;
  width: number;
  height: number;
}

const NS = 'http://www.w3.org/2000/svg';
export const MASK_ID = 'plan-grid-fade';

/** The gradient's ends and its stops (0 to 1) along one axis: full from `MARGIN_M` past each edge, none from
 *  `MARGIN_M + FADE_M` past it. */
export function ramp(start: number, size: number): { from: number; to: number; stops: [number, number] } {
  const from = start - MARGIN_M - FADE_M;
  const to = start + size + MARGIN_M + FADE_M;
  const inner = FADE_M / (to - from);
  return { from, to, stops: [inner, 1 - inner] };
}

/** The box around everything in `boxes` with an area, or null when nothing is drawn. */
export function union(boxes: Box[]): Box | null {
  const drawn = boxes.filter(b => b.width > 0 || b.height > 0);
  if (!drawn.length) return null;
  const x = Math.min(...drawn.map(b => b.x));
  const y = Math.min(...drawn.map(b => b.y));
  const right = Math.max(...drawn.map(b => b.x + b.width));
  const bottom = Math.max(...drawn.map(b => b.y + b.height));
  return { x, y, width: right - x, height: bottom - y };
}

function gradient(id: string, axis: 'x' | 'y', box: Box): string {
  const { from, to, stops } = ramp(axis === 'x' ? box.x : box.y, axis === 'x' ? box.width : box.height);
  const ends = axis === 'x' ? `x1="${from}" x2="${to}" y1="0" y2="0"` : `x1="0" x2="0" y1="${from}" y2="${to}"`;
  // mask luminance: white keeps the grid, black hides it (these are the mask's values, not colours anyone sees)
  return (
    `<linearGradient id="${id}" gradientUnits="userSpaceOnUse" ${ends}>` +
    `<stop offset="0" stop-color="#000"/><stop offset="${stops[0]}" stop-color="#fff"/>` +
    `<stop offset="${stops[1]}" stop-color="#fff"/><stop offset="1" stop-color="#000"/></linearGradient>`
  );
}

/** The mask's markup for a box: a rectangle faded along x, itself masked by one faded along y, so the corners fade
 *  both ways. */
export function maskMarkup(box: Box): string {
  const all = 'x="-100000" y="-100000" width="200000" height="200000"';
  const region = `maskUnits="userSpaceOnUse" maskContentUnits="userSpaceOnUse" ${all}`;
  return (
    gradient(`${MASK_ID}-x`, 'x', box) +
    gradient(`${MASK_ID}-y`, 'y', box) +
    `<mask id="${MASK_ID}-v" ${region}><rect ${all} fill="url(#${MASK_ID}-y)"/></mask>` +
    `<mask id="${MASK_ID}" ${region}><rect ${all} fill="url(#${MASK_ID}-x)" mask="url(#${MASK_ID}-v)"/></mask>`
  );
}

/** What the editor has drawn on the plan, in the plan's coordinates, or null with nothing drawn (or no plan shown). */
function drawnBox(): Box | null {
  const parts = document.querySelectorAll<SVGGraphicsElement>(
    'g[data-floorplan-scene] .floorplan-registry-layer, g[data-floorplan-scene] [data-camera-spots]',
  );
  return union([...parts].map(part => part.getBBox()));
}

/** Keep the grid's mask on what the editor draws, from now on. */
export function installGridFade(): void {
  const host = document.createElementNS(NS, 'svg');
  host.setAttribute('aria-hidden', 'true');
  host.setAttribute('width', '0');
  host.setAttribute('height', '0');
  host.style.position = 'absolute';
  const defs = document.createElementNS(NS, 'defs');
  host.append(defs);
  document.body.append(host);

  let shown = '';
  let queued = false;
  const update = () => {
    queued = false;
    const box = drawnBox();
    const markup = box ? maskMarkup(box) : '';
    if (markup === shown) return;
    shown = markup;
    defs.innerHTML = markup;
    document.documentElement.toggleAttribute('data-grid-fade', box !== null);
  };
  // one measure a frame at most, however many changes the editor makes in it
  new MutationObserver(records => {
    if (queued || records.every(r => host.contains(r.target))) return;
    queued = true;
    requestAnimationFrame(update);
  }).observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['points', 'd'] });
}
