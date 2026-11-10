import { screen } from '@testing-library/react';
import { HttpResponse, http } from 'msw';
import { expect, it } from 'vitest';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import { DevicesTab } from './DevicesTab';

const headings = () => screen.getAllByRole('heading').map(h => h.textContent);

it("leads with whose phones these are while no one's is linked, and leaves out the empty cards", async () => {
  server.use(http.get('/config', () => HttpResponse.json({ control: true })));
  renderPage(<DevicesTab />);
  await screen.findByRole('heading', { name: 'Whose phones are these?' });
  expect(headings()).not.toContain("Who's home");
  expect(headings()).not.toContain('Your things');
  // The caveat is said once, under the title.
  expect(screen.getAllByText(/a clue, not proof/)).toHaveLength(1);
});

it("shows who's home and the Wi-Fi card once a person is linked", async () => {
  server.use(
    http.get('/config', () => HttpResponse.json({ control: true })),
    http.get('/device-presence', () =>
      HttpResponse.json({
        at: 0,
        start: 0,
        gap_seconds: 600,
        caveat: '',
        people: [{ name: 'Dana', status: 'unknown', last_seen: null, source: null, place: null, links: 1 }],
      }),
    ),
  );
  renderPage(<DevicesTab />);
  await screen.findByRole('heading', { name: "Who's home" });
  expect(headings()).toContain('On the Wi-Fi today');
  expect(headings()).not.toContain('Whose phones are these?');
});

it('shows the cards without waiting for the device list, which fills in last', async () => {
  server.use(
    http.get('/config', () => HttpResponse.json({ control: true })),
    http.get('/device-links', () => new Promise<never>(() => {})), // still counting a day of sightings
  );
  renderPage(<DevicesTab />);
  await screen.findByRole('heading', { name: 'Whose phones are these?' });
  expect(screen.getByText('Looking for devices…')).toBeInTheDocument();
});
