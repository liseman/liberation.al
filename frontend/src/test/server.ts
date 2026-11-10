import { HttpResponse, http, ws } from 'msw';
import { setupServer } from 'msw/node';

/** Continuum's gateway (camera/gateway.ts): a page showing a live camera connects to it. Tests that look at the
 *  gateway hand gateway.ts a fake connection; for the rest it accepts the connection and sends nothing, so the
 *  cameras just show their snapshots. */
const gateway = ws.link('ws://*:8766/ws');

/** The fake network for tests: add handlers per test with server.use(http.get(...)). */
/** Review's Ask shows a Research button (#1336), which asks whether research can run: by default it can't, with
 *  nothing in the Inbox. Tests about research serve their own reply. */
const research = http.get('/research/jobs', () =>
  HttpResponse.json({ jobs: [], available: false, reason: 'Research is off in tests.', unread: 0 }),
);

export const server = setupServer(
  gateway.addEventListener('connection', () => {}),
  research,
  // Activity clip cards ask for their start-frame still (#2296): none is kept unless a test serves one.
  http.get('/people/video-thumb.jpg', () => new HttpResponse(null, { status: 404 })),
  http.get('/network-cameras', () => HttpResponse.json({ cameras: [], capture_ready: false })),
  // The header's privacy pauses (#2260): read only, nothing paused, unless a test serves its own.
  http.get('/devices/privacy', () => HttpResponse.json({ ok: true, control: false, ready: false, devices: [] })),
  // Settings' Phones, watches, and tags (#1363): nothing seen and nothing linked, unless a test serves its own.
  // Known › Devices' whereabouts and Wi-Fi today (#1326): nothing linked and nothing on the Wi-Fi, unless a test serves its own.
  http.get('/device-presence', () => HttpResponse.json({ at: 0, start: 0, gap_seconds: 600, people: [], caveat: '' })),
  http.get('/device-whereabouts', () => HttpResponse.json({ at: 0, days: 30, caveat: '', links: [] })),
  http.get('/device-comings', () => HttpResponse.json({ at: 0, start: 0, gap_seconds: 600, rows: 0, devices: [] })),
  http.get('/device-links', () =>
    HttpResponse.json({ devices: [], links: [], total: 0, linked: 0, unknown: 0, hours: 24 }),
  ),
  // People's Pets (#1332): no animal sightings, unless a test serves its own.
  http.get('/calls/live', () => HttpResponse.json({ live: null, count: 0 })), // Calls looks for a call on (#2702)
  http.get('/pets', () => HttpResponse.json({ pets: [], species: [] })), // Live's large view reads it for animal boxes
  http.post('/frontend/log', () => HttpResponse.json({ kept: 0 })), // the page's own log (shared/telemetry.ts)
  http.get('/pets/sightings', () => HttpResponse.json({ groups: [], recognition: [], names: {} })),
);
