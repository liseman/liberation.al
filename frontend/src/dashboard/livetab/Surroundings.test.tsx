import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { delay, HttpResponse, http } from 'msw';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import {
  applied,
  LearnSurroundings,
  type SurroundingsChange,
  type SurroundingsView,
  surroundingsHint,
  toggled,
} from './Surroundings';

let view: SurroundingsView;
let posts: SurroundingsChange[];

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
      view = applied(view, change);
      return HttpResponse.json(view);
    }),
  );
});

afterEach(() => server.resetHandlers());

const toggle = () => screen.findByRole('switch', { name: 'Learn surroundings' });
const menuButton = () => screen.findByRole('button', { name: 'Cameras to learn from' });

/** A POST that fails with the server's words, after `ms`. */
function failing(ms = 0) {
  server.use(
    http.post('/surroundings', async () => {
      await delay(ms);
      return HttpResponse.json({ ok: false, error: 'pack.toml is read only' }, { status: 400 });
    }),
  );
}

describe("Live's Learn surroundings switch", () => {
  it('starts off, and turns learning on and off at once', async () => {
    const user = userEvent.setup();
    renderPage(<LearnSurroundings />);
    expect(await toggle()).not.toBeChecked();
    await user.click(await toggle());
    await waitFor(async () => expect(await toggle()).toBeChecked());
    await user.click(await toggle());
    await waitFor(async () => expect(await toggle()).not.toBeChecked());
    expect(posts).toEqual([{ on: true }, { on: false }]);
  });

  it('turns on from the keyboard', async () => {
    const user = userEvent.setup();
    renderPage(<LearnSurroundings />);
    (await toggle()).focus();
    await user.keyboard(' ');
    await waitFor(async () => expect(await toggle()).toBeChecked());
    expect(posts).toEqual([{ on: true }]);
  });

  it('shows the change and a pending state while the request is out', async () => {
    let release: () => void = () => {};
    const held = new Promise<void>(resolve => {
      release = resolve;
    });
    server.use(
      http.post('/surroundings', async ({ request }) => {
        const change = (await request.json()) as SurroundingsChange;
        await held;
        view = applied(view, change);
        return HttpResponse.json(view);
      }),
    );
    const user = userEvent.setup();
    renderPage(<LearnSurroundings />);
    await user.click(await toggle());
    const shown = await toggle();
    await waitFor(() => expect(shown).toHaveAttribute('aria-busy', 'true'));
    expect(shown).toBeChecked();
    release();
    await waitFor(() => expect(shown).toHaveAttribute('aria-busy', 'false'));
    expect(shown).toBeChecked();
  });

  it('lists the cameras in a menu with the sampled one checked and the default marked, and adds one', async () => {
    const user = userEvent.setup();
    renderPage(<LearnSurroundings />);
    await user.click(await menuButton());
    const front = await screen.findByRole('menuitemcheckbox', { name: /front/ });
    const hall = screen.getByRole('menuitemcheckbox', { name: 'hall' });
    expect(front).toHaveTextContent('Used when none is chosen');
    expect(front).toHaveAttribute('aria-checked', 'true');
    expect(hall).toHaveAttribute('aria-checked', 'false');
    await user.click(hall);
    await waitFor(() => expect(hall).toHaveAttribute('aria-checked', 'true'));
    expect(posts).toEqual([{ cameras: ['front', 'hall'] }]);
    // The menu stays open while toggling, so a second camera is one more click.
    expect(screen.getByRole('menu')).toBeInTheDocument();
    await waitFor(() => expect(front).not.toHaveAttribute('data-disabled'));
    await user.click(front);
    await waitFor(() => expect(front).toHaveAttribute('aria-checked', 'false'));
    expect(posts).toEqual([{ cameras: ['front', 'hall'] }, { cameras: ['hall'] }]);
  });

  it("puts the switch back and says why when turning on fails, in the server's words", async () => {
    failing();
    const user = userEvent.setup();
    renderPage(<LearnSurroundings />);
    await user.click(await toggle());
    expect(await screen.findByRole('alert')).toHaveTextContent("Couldn't change it: pack.toml is read only");
    expect(await toggle()).not.toBeChecked();
  });

  it('puts the camera back when checking it fails', async () => {
    failing(20);
    const user = userEvent.setup();
    renderPage(<LearnSurroundings />);
    await user.click(await menuButton());
    const hall = await screen.findByRole('menuitemcheckbox', { name: 'hall' });
    await user.click(hall);
    expect(hall).toHaveAttribute('aria-checked', 'true'); // shown at once, while saving
    expect(await screen.findByRole('alert')).toHaveTextContent("Couldn't change it: pack.toml is read only");
    await waitFor(() => expect(hall).toHaveAttribute('aria-checked', 'false'));
  });

  it('says what the switch does beside it, off and on, with a Places link while learning', async () => {
    const user = userEvent.setup();
    renderPage(<LearnSurroundings />);
    await toggle();
    expect(screen.getByText(surroundingsHint(view).off)).toBeInTheDocument();
    expect(await menuButton()).toHaveTextContent('front');
    await user.click(await toggle());
    await waitFor(async () => expect(await toggle()).toBeChecked());
    expect(screen.getByText(/Collecting pictures from front\. See them in/)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Places' })).toHaveAttribute('href', '/places');
  });

  it('disables the switch and the menu on another device, and says why', async () => {
    view = { ...view, control: false, on: true };
    renderPage(<LearnSurroundings />);
    const shown = await toggle();
    expect(shown).toBeChecked();
    expect(shown).toBeDisabled();
    expect(await menuButton()).toBeDisabled();
    expect(screen.getByText('Only the machine Liberational runs on can change this.')).toBeInTheDocument();
  });

  it('is not drawn when there is no camera to learn from', async () => {
    view = { ...view, cameras: [] };
    const { container } = renderPage(<LearnSurroundings />);
    await waitFor(() => expect(container.querySelector('.surroundings')).toBeNull());
    expect(screen.queryByRole('switch')).toBeNull();
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
