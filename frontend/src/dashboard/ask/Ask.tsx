import { lazy } from 'react';
import { Deferred } from '../Deferred';
import { AskPanel } from './AskPanel';
import { closeAsk, useAsk } from './openAsk';

const PackAsk = lazy(() => import('./PackAsk').then(m => ({ default: m.PackAsk })));

/** The one Ask, in its floating window: opened from anywhere through openAsk (openAsk.ts), and the same each time. */
export function Ask({ hidden = false }: { hidden?: boolean }) {
  const open = useAsk(s => s.open);
  return (
    <AskPanel opened={open && !hidden} onClose={closeAsk} title="Ask">
      <Deferred name="Ask">
        <PackAsk />
      </Deferred>
    </AskPanel>
  );
}
