// Whether skeletons are shown: one choice for Live and Review in this browser, kept with the other prefs.
import { setPref, usePrefs } from '../lib/prefs';

/** Whether this browser shows skeletons (on until switched off). */
export const useSkeletons = (): boolean => usePrefs(s => s.skeletons !== false);

/** Show skeletons in this browser, or not. */
export const setSkeletons = (on: boolean): void => setPref('skeletons', on);
