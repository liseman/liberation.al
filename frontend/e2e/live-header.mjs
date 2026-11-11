/** Headless demo-only layout regression and review screenshots.
 * Run from frontend: ../tools/test-slot.sh node e2e/live-header.mjs DEMO_URL OUTPUT_DIR
 * Start pack-demo with this checkout's frontend build first. No real camera images are requested.
 */
import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import { chromium } from 'playwright';

const [base, output] = process.argv.slice(2);
assert(base && output, 'Pass the demo URL and screenshot directory');
const url = new URL(base);
assert(['localhost', '127.0.0.1'].includes(url.hostname), 'Use a local demo');
const html = await (await fetch(url)).text();
assert(html.includes('\\"gatewayPort\\": 9'), 'Only the isolated demo gateway is allowed');
assert(html.includes('\\"owner\\": \\"Alice\\"'), 'Use the made-up demo household');
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true });
const measurements = [];
try {
  for (const width of [1280, 390]) {
    for (const theme of ['light', 'dark']) {
      const context = await browser.newContext({ viewport: { width, height: 950 }, colorScheme: theme });
      await context.addInitScript(() => {
        localStorage.setItem('liberational.prefs', JSON.stringify({ gettingStartedDone: true, liveTiles: false }));
      });
      const page = await context.newPage();
      let release;
      const held = new Promise(resolve => {
        release = resolve;
      });
      let on = false;
      const view = () => ({
        control: true,
        on,
        chosen: [],
        cameras: [
          { name: 'cam0', sampled: true, default: true },
          { name: 'cam1', sampled: false, default: false },
        ],
      });
      await page.route('**/*', async route => {
        const request = route.request();
        const path = new URL(request.url()).pathname;
        if (request.isNavigationRequest()) {
          const shell = {
            assistant: 'Bob',
            owner: 'Alice',
            cams: ['cam0', 'cam1'],
            activeCams: ['cam0', 'cam1'],
            views: { cam0: 'Front camera', cam1: 'Desk camera' },
            hotkeys: false,
            gatewayPort: 9,
            needsSetup: false,
          };
          return route.fulfill({
            contentType: 'text/html',
            body: html.replace(/window\.NAMES=.*?;<\/script>/, `window.NAMES=${JSON.stringify(shell)};</script>`),
          });
        }
        if (path === '/surroundings') {
          if (request.method() === 'POST') on = request.postDataJSON().on ?? on;
          else await held;
          return route.fulfill({ json: view() });
        }
        if (path === '/events') {
          await held;
          const now = Date.now() / 1000;
          const events = {
            scene: [
              { cam: 'cam0', ts: now },
              { cam: 'cam1', ts: now },
            ],
            vision: { ts: now, cameras: { cam0: { state: 'on' }, cam1: { state: 'on' } } },
            ambient: {},
            transcript: { entries: [], reset: true },
            speakers: { speakers: [], merged: {} },
          };
          return route.fulfill({
            contentType: 'text/event-stream',
            body: Object.entries(events)
              .map(([kind, data]) => `event: ${kind}\ndata: ${JSON.stringify(data)}\n\n`)
              .join(''),
          });
        }
        if (/preview|snapshot/.test(path)) return route.abort();
        return route.continue();
      });
      await page.goto(url.href);
      const toggle = page.getByRole('switch', { name: 'Learn surroundings' });
      await toggle.waitFor();
      await page.evaluate(() => document.fonts.ready);
      assert(await toggle.isDisabled(), 'First-paint switch must be disabled');
      const bounds = () =>
        page.locator('.cams-head').evaluate(el => {
          const box = el.getBoundingClientRect();
          const row = el.querySelector('.surroundings').getBoundingClientRect();
          return {
            x: box.x,
            y: box.y,
            width: box.width,
            height: box.height,
            rowX: row.x,
            rowY: row.y,
            rowWidth: row.width,
            rowHeight: row.height,
          };
        });
      const before = await bounds();
      const shot = state =>
        page.screenshot({ animations: 'disabled', path: `${output}/${width}-${theme}-${state}.png` });
      await shot('loading');
      release();
      await page.waitForFunction(() => !document.querySelector('.surroundings input').disabled);
      await page.locator('.live-cams:not(.stage-wait)').waitFor();
      const after = await bounds();
      assert.deepEqual(after, before, 'Status arrival must not resize the header or learning row');
      await shot('off');
      await toggle.click();
      await page.getByText('Collecting pictures. See them in').waitFor();
      assert.deepEqual(await bounds(), before, 'Learning on must preserve dimensions');
      await shot('on');
      await page.getByRole('button', { name: 'Cameras to learn from' }).click();
      await page.getByRole('menu').waitFor();
      await shot('menu');
      measurements.push({ width, theme, before, after });
      await context.close();
    }
  }
  await writeFile(`${output}/measurements.json`, JSON.stringify(measurements, null, 2));
  console.log(`Passed ${measurements.length} viewport/theme cases; 16 screenshots saved to ${output}`);
} finally {
  await browser.close();
}
