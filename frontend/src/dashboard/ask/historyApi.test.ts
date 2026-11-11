import { HttpResponse, http } from 'msw';
import { expect, it } from 'vitest';
import { server } from '../../test/server';
import { deleteHistory } from './historyApi';

it('deletes through the shared layer: a DELETE with a JSON body, keeping the deleted count as the check', async () => {
  let seen: { method: string; body: unknown; type: string | null } | null = null;
  server.use(
    http.delete('/pack-ask/history/saved-1', async ({ request }) => {
      seen = { method: request.method, body: await request.json(), type: request.headers.get('Content-Type') };
      return HttpResponse.json({ deleted: 1 });
    }),
  );
  await deleteHistory('saved-1', null);
  expect(seen).toEqual({ method: 'DELETE', body: {}, type: 'application/json' });
});

it('says the status when a delete is answered with something that is not JSON', async () => {
  // A proxy between the page and the dashboard, or a dashboard restarting, answers with a page, not JSON: the
  // viewer should read the status, not the browser's own parsing words.
  server.use(
    http.delete('/pack-ask/history/saved-1', () => new HttpResponse('<html>Bad gateway</html>', { status: 502 })),
  );
  await expect(deleteHistory('saved-1', null)).rejects.toMatchObject({
    name: 'HttpError',
    status: 502,
    message: 'Request failed (502)',
  });
});

it('refuses a reply that says nothing was deleted', async () => {
  server.use(http.delete('/pack-ask/history/saved-1', () => HttpResponse.json({ ok: true })));
  await expect(deleteHistory('saved-1', null)).rejects.toThrow('Invalid deletion response');
});
