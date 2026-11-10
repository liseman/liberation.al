import { QueryClient } from '@tanstack/react-query';
import { HttpResponse, http } from 'msw';
import { beforeEach, describe, expect, it } from 'vitest';
import { server } from '../../../test/server';
import type { Clone } from '../api';
import { build, cloneOf, useClones } from './store';

const clone = (id: number, extra: Partial<Clone> = {}): Clone => ({
  id,
  label: `Speaker ${id}`,
  ready: false,
  built: false,
  ...extra,
});

describe('cloneOf', () => {
  it('takes a built voice first, then a ready one, then any', () => {
    const list = [clone(1), clone(2, { ready: true }), clone(3, { ready: true, built: true })];
    expect(cloneOf({ voices: [1, 2, 3] }, list)?.id).toBe(3);
    expect(cloneOf({ voices: [1, 2] }, list)?.id).toBe(2);
    expect(cloneOf({ voices: [1] }, list)?.id).toBe(1);
    expect(cloneOf({ voices: [9] }, list)).toBeNull();
  });
});

describe('build', () => {
  beforeEach(() => useClones.setState({ sel: null, busy: null }));

  it('builds one voice at a time, and frees the others after', async () => {
    let calls = 0;
    server.use(
      http.post('/clones/build', async () => {
        calls++;
        return HttpResponse.json({ ok: true, utterances: 4, seconds: 13 });
      }),
      http.get('/clones', () => HttpResponse.json({ clones: [] })),
    );
    const client = new QueryClient();
    const [first, second] = await Promise.all([build(client, 3), build(client, 3)]);
    expect(calls).toBe(1);
    expect(first).toEqual({ ok: true, utterances: 4, seconds: 13 });
    expect(second.ok).toBe(false);
    expect(useClones.getState().busy).toBeNull();
  });

  it("reports the server's reason", async () => {
    server.use(
      http.post('/clones/build', () => HttpResponse.json({ error: 'not enough clean speech' }, { status: 400 })),
    );
    expect(await build(new QueryClient(), 3)).toEqual({ ok: false, error: 'Building failed: not enough clean speech' });
  });
});
