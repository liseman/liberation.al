import { act, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import type { DetectedActivity } from './api';
import { EngineContext } from './context';
import { ReviewEngine } from './engine';
import { PersonMoments } from './PersonMoments';
import { PersonPlayback } from './personPlayback';
import type { Chip } from './types';

const T = new Date(2026, 9, 27, 10).getTime() / 1000;
const entries = [
  { ts_start: T, ts_end: T + 2, speaker: 'Alice', text: 'First line' },
  { ts_start: T + 10, ts_end: T + 12, speaker: 'Dana', text: 'Other speaker' },
  { ts_start: T + 100, ts_end: T + 103, speaker: 'Alice', text: 'Next line' },
];
const chips: Chip[] = [
  {
    id: 'person:alice',
    kind: 'person',
    label: 'Alice',
    spans: [],
    total_s: 5,
    count: 1,
    cams: { rear: [[T + 20, T + 25]] },
  },
];
function showPlayback(
  withMoments = true,
  summaryState: 'loading' | 'error' | 'partial' | 'ready' = 'ready',
  activities: DetectedActivity[] = [],
) {
  const engine = new ReviewEngine({ cams: ['front', 'rear'], fetchNotes: async () => ({ cams: {} }) });
  engine.dayStart = T - 36000;
  engine.ui.setState({ t: T });
  const host = { play: vi.fn(), stop: vi.fn() };
  const playback = new PersonPlayback(host);
  renderPage(
    <EngineContext value={engine}>
      <PersonMoments
        person="Alice"
        entries={withMoments ? entries : []}
        chips={withMoments ? chips : []}
        activities={activities}
        blocks={[]}
        summaryState={summaryState}
        playback={playback}
      />
    </EngineContext>,
  );
  return { engine, host, playback };
}
afterEach(() => vi.unstubAllGlobals());
beforeEach(() => {
  vi.stubGlobal('NAMES', { views: { rear: 'rear camera', front: 'front camera' } });
  server.use(http.get('/speakers', () => HttpResponse.json({ speakers: [], merged: {} })));
});

describe('person playback controls', () => {
  it('queues only their bounded lines and exposes progress and stop controls', async () => {
    const { host, playback } = showPlayback();
    const audio = screen.getByRole('button', { name: "Play Alice's audio" });
    await userEvent.click(audio);
    expect(host.play).toHaveBeenLastCalledWith({ start: T, end: T + 2, cam: null }, 'audio');
    expect(audio).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByRole('status')).toHaveTextContent("Playing Alice's audio, moment 1 of 2.");
    act(() => {
      playback.atEnd(T + 2);
    });
    expect(host.play).toHaveBeenLastCalledWith({ start: T + 100, end: T + 103, cam: null }, 'audio');
    expect(screen.getByRole('status')).toHaveTextContent('moment 2 of 2');
    await userEvent.click(screen.getByRole('button', { name: 'Stop person playback' }));
    expect(host.stop).toHaveBeenCalledOnce();
    expect(audio).toHaveAttribute('aria-pressed', 'false');
    expect(screen.queryByRole('status')).not.toBeInTheDocument();
  });

  it('queues video on the view that saw them, with its short lead-in', async () => {
    const { host } = showPlayback();
    await userEvent.click(screen.getByRole('button', { name: "Play Alice's video" }));
    expect(host.play).toHaveBeenCalledExactlyOnceWith({ start: T + 18, end: T + 25, cam: 'rear' }, 'video');
  });

  it('does not leave the last visit active beyond its end', () => {
    const { engine } = showPlayback();
    const visits = screen.getByRole('list', { name: "Alice's visits" });
    expect(
      within(visits)
        .getAllByRole('button')
        .some(b => b.getAttribute('aria-current') === 'true'),
    ).toBe(true);
    act(() => {
      engine.ui.setState({ t: T + 1000 });
    });
    expect(
      within(visits)
        .getAllByRole('button')
        .every(b => !b.hasAttribute('aria-current')),
    ).toBe(true);
  });

  it('keeps a point action active through its bounded video tail and clears it at the queue end', async () => {
    const { engine, host, playback } = showPlayback(true, 'ready', [
      {
        id: 'point',
        camera: 'front',
        start: T + 200,
        end: T + 200,
        person_name: 'Alice',
        person: null,
        name: 'Reach',
        kind: 'action',
        count: null,
        promoted: false,
      },
    ]);
    await userEvent.click(screen.getByRole('button', { name: "Play Alice's video" }));
    expect(host.play).toHaveBeenLastCalledWith({ start: T + 18, end: T + 25, cam: 'rear' }, 'video');
    act(() => playback.atEnd(T + 25));
    expect(host.play).toHaveBeenLastCalledWith({ start: T + 198, end: T + 202, cam: 'front' }, 'video');
    const visits = screen.getByRole('list', { name: "Alice's visits" });
    const pointVisit = within(visits).getByRole('button', { name: '10:03' });
    act(() => engine.ui.setState({ t: T + 201 }));
    expect(pointVisit).toHaveAttribute('aria-current', 'true');
    act(() => engine.ui.setState({ t: T + 202 }));
    expect(pointVisit).not.toHaveAttribute('aria-current');
  });

  it('keeps playback available while summary mentions cannot be checked', async () => {
    const { host } = showPlayback(true, 'error');
    expect(screen.getByText('Summary unavailable. Mentions could not be checked.')).toBeInTheDocument();
    expect(screen.queryByText("The summary doesn't mention Alice this day.")).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: "Play Alice's audio" }));
    expect(host.play).toHaveBeenLastCalledWith({ start: T, end: T + 2, cam: null }, 'audio');
  });

  it('has no playable queue for a person with no timed media', () => {
    const { host } = showPlayback(false);
    expect(screen.getByRole('button', { name: "Play Alice's audio" })).toBeDisabled();
    expect(screen.getByRole('button', { name: "Play Alice's video" })).toBeDisabled();
    expect(host.play).not.toHaveBeenCalled();
  });
});

it('makes every speech, sighting, and action moment a full clickable target on its own time and camera', async () => {
  const engine = new ReviewEngine({ cams: ['front', 'rear'], fetchNotes: async () => ({ cams: {} }) });
  const seek = vi.spyOn(engine, 'seekActivity').mockImplementation(() => {});
  renderPage(
    <EngineContext value={engine}>
      <PersonMoments
        person="Alice"
        entries={[{ ts_start: T, ts_end: T + 2, speaker: 'Alice', text: 'First line' }]}
        chips={[
          {
            id: 'person:alice',
            kind: 'person',
            label: 'Alice',
            spans: [],
            total_s: 10,
            count: 1,
            cams: { rear: [[T + 10, T + 20]] },
          },
        ]}
        activities={[
          {
            id: 'a1',
            camera: 'front',
            start: T + 15,
            end: T + 17,
            person_name: 'Alice',
            person: null,
            name: 'Reach',
            kind: 'action',
            count: null,
            promoted: false,
          },
        ]}
        summaryState="ready"
        blocks={[]}
      />
    </EngineContext>,
  );
  const visits = screen.getByRole('list', { name: "Alice's visits" });
  for (const [label, ts, cam] of [
    ['Show 10:00:00: First line', T, ''],
    ['Show 10:00:10: Seen · Rear camera', T + 10, 'rear'],
    ['Show 10:00:15: Reach · Front camera', T + 15, 'front'],
  ] as const) {
    const button = within(visits).getByRole('button', { name: label });
    await userEvent.click(button);
    expect(seek).toHaveBeenLastCalledWith(ts, cam);
  }
});

function showSummary(
  summaryState: 'loading' | 'error' | 'partial' | 'ready',
  blocks: { start: number; end: number; text: string; tags: string[] }[] = [],
) {
  const engine = new ReviewEngine({ cams: [], fetchNotes: async () => ({ cams: {} }) });
  const retry = vi.fn();
  renderPage(
    <EngineContext value={engine}>
      <PersonMoments
        person="Alice"
        entries={[]}
        chips={[]}
        activities={[]}
        blocks={blocks}
        summaryState={summaryState}
        onRetrySummary={retry}
      />
    </EngineContext>,
  );
  return retry;
}

describe('person summary evidence', () => {
  it.each([
    ['loading', 'Loading summary mentions'],
    ['error', 'Summary unavailable'],
    ['partial', 'The summary is still being written'],
  ] as const)('does not claim no mentions when the summary is %s', (state, message) => {
    showSummary(state);
    expect(screen.getByText(new RegExp(message))).toBeInTheDocument();
    expect(screen.queryByText("The summary doesn't mention Alice this day.")).not.toBeInTheDocument();
  });

  it('reserves no mentions for a completed summary', () => {
    showSummary('ready');
    expect(screen.getByText("The summary doesn't mention Alice this day.")).toBeInTheDocument();
  });

  it('keeps available mentions while explaining partial coverage', () => {
    showSummary('partial', [{ start: 100, end: 110, text: 'Alice made tea.', tags: [] }]);
    expect(screen.getByText('Alice')).toBeInTheDocument();
    expect(screen.getByText(/The summary is still being written/)).toBeInTheDocument();
  });

  it('offers an explicit retry after the summary fails', async () => {
    const retry = showSummary('error');
    await userEvent.click(screen.getByRole('button', { name: 'Retry summary' }));
    expect(retry).toHaveBeenCalledOnce();
  });
});
