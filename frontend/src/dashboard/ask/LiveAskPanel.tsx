import { lazy } from 'react';
import { Deferred } from '../Deferred';
import { AskPanel } from './AskPanel';

const PackAsk = lazy(() => import('./PackAsk').then(m => ({ default: m.PackAsk })));

/** The Ask off Review (My recordings searches recordings; Friends and Everyone are general), in its floating window. */
export function LiveAskPanel({ opened, onClose }: { opened: boolean; onClose: () => void }) {
  return (
    <AskPanel opened={opened} onClose={onClose} title="Ask" kind="live">
      <Deferred name="Ask">
        <PackAsk showModes bare compact />
      </Deferred>
    </AskPanel>
  );
}
