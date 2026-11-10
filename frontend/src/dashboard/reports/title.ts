// How a report's saved title reads on the page.
import { capitalize } from '../lib/format';
import type { ChosenReport } from './api';

/** A report's title as the page shows it: in sentence case, and the walk-by events by their plain name. Saved titles
 *  come from the server as they were made ("crossing over time", now "crossing, day by day"), so this applies to old
 *  ones too. The event reads "Walked past", as the stream names it, so it isn't taken for a camera's "Walk-bys on …". */
export function reportTitle(r: Pick<ChosenReport, 'arg' | 'title'>): string {
  if (r.arg === 'event:crossing' && /^crossing( over time|, day by day)$/i.test(r.title))
    return 'Walked past, day by day';
  return capitalize(r.title);
}
