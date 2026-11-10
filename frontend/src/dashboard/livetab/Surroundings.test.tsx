import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import { LearnSurroundings, type SurroundingsChange, type SurroundingsView, toggled } from './Surroundings';

let view: SurroundingsView;
let posts: SurroundingsChange[];

/** The server's side of a change, as dashboard/surroundings.py answers it. */
function apply(change: SurroundingsChange): SurroundingsView {
  const chosen = change.cameras ?? view.chosen;
  return {
    ...view,
    on: change.on ?? view.on,
    chosen,
    cameras: view.cameras.map(c => ({ ...c, sampled: chosen.length ? chosen.includes(c.name) : c.default })),
  };
}

beforeEach(() => {
  posts = [];
  view = {
    control: true,
    on: false,
    chosen: [],
    cameras: [
      { name: 'front', sampled: true, default: true },
      { name: 'hall', sampled: false, default: false },
    ],
  };
  server.use(
    http.get('/surroundings', () => HttpResponse.json(view)),
    http.post('/surroundings', async ({ request }) => {
      const change = (await request.json()) as SurroundingsChange;
      posts.push(change);
      view = apply(change);
      return HttpResponse.json(view);
    }),
  );
});

afterEach(() => server.resetHandlers());

const button = () => screen.findByRole('button', { name: 'Learn surroundings' });

describe("Live's Learn surroundings switch", () => {
  it('starts off, and turns learning on and off', async () => {
    const user = userEvent.setup();
    renderPage(<LearnSurroundings />);
    expect(await button()).toHaveAttribute('aria-pressed', 'false');
    await user.click(await button());
    await waitFor(async () => expect(await button()).toHaveAttribute('aria-pressed', 'true'));
    await user.click(await button());
    await waitFor(async () => expect(await button()).toHaveAttribute('aria-pressed', 'false'));
    expect(posts).toEqual([{ on: true }, { on: false }]);
  });

  it('lists the cameras with the sampled one checked and the default marked, and adds one', async () => {
    const user = userEvent.setup();
    renderPage(<LearnSurroundings />);
    await user.click(await screen.findByRole('button', { name: 'Cameras to learn surroundings from: front' }));
    const front = await screen.findByRole('checkbox', { name: 'front (default)' });
    const hall = screen.getByRole('checkbox', { name: 'hall' });
    expect(front).toBeChecked();
    expect(hall).not.toBeChecked();
    await user.click(hall);
    await waitFor(() => expect(hall).toBeChecked());
    expect(posts).toEqual([{ cameras: ['front', 'hall'] }]);
  });

  it('shows a viewer on another device the switch, but not as something to change', async () => {
    view = { ...view, control: false, on: true };
    renderPage(<LearnSurroundings />);
    const shown = await button();
    expect(shown).toHaveAttribute('aria-pressed', 'true');
    expect(shown).toBeDisabled();
    expect(shown).toHaveAttribute('title', 'Only the machine Liberational runs on can change this.');
  });

  it("says why a change didn't take, in the server's words", async () => {
    server.use(
      http.post('/surroundings', () =>
        HttpResponse.json({ ok: false, error: 'pack.toml is read only' }, { status: 400 }),
      ),
    );
    const user = userEvent.setup();
    renderPage(<LearnSurroundings />);
    await user.click(await button());
    expect(await screen.findByRole('alert')).toHaveTextContent("Couldn't change it: pack.toml is read only");
  });

  it('is not drawn when there is no camera to learn from', async () => {
    view = { ...view, cameras: [] };
    const { container } = renderPage(<LearnSurroundings />);
    await waitFor(() => expect(container.querySelector('.surroundings')).toBeNull());
    expect(screen.queryByRole('button', { name: 'Learn surroundings' })).toBeNull();
  });
});

describe('toggled', () => {
  it('takes the last camera away to leave none chosen, which is the default', () => {
    expect(toggled(view, 'front')).toEqual([]);
  });

  it('starts from the default camera when none was chosen', () => {
    expect(toggled(view, 'hall')).toEqual(['front', 'hall']);
  });
});
