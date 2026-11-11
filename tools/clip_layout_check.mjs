// Headless clip layout regression and design evidence, using only an isolated pack-demo and synthetic frames.
// Run after the frontend build, through tools/test-slot.sh:
// node tools/clip_layout_check.mjs <demo URL> <output directory> [baseline build directory]
// The optional baseline records the old measurements without asserting the new layout contract.
import assert from 'node:assert/strict';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import path from 'node:path';

const [base, output, baseline] = process.argv.slice(2);
assert(base && output, 'Provide an isolated demo URL and an output directory');
const html = await (await fetch(`${base}/activities`, { headers: { Accept: 'text/html' } })).text();
const names = JSON.parse(JSON.parse(html.match(/window.NAMES=JSON.parse\((.*?)\);/s)?.[1] ?? '"null"'));
assert.equal(names?.gatewayPort, 9, 'Refusing a dashboard that is not isolated from the real gateway');
const { chromium } = createRequire(path.resolve('frontend/package.json'))('playwright');
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true });
const measurements = [];
try {
  const fixture = await browser.newPage();
  const images = await fixture.evaluate(() => [
    [360, 640], [960, 540], [640, 640],
  ].map(([width, height]) => {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#d7e4dc';
    ctx.fillRect(0, 0, width, height);
    ctx.fillStyle = '#afc6ba';
    ctx.fillRect(0, height * 0.7, width, height * 0.3);
    ctx.fillStyle = '#4c7165';
    ctx.beginPath();
    ctx.arc(width / 2, height * 0.3, height * 0.07, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillRect(width / 2 - height * 0.06, height * 0.4, height * 0.12, height * 0.3);
    ctx.fillStyle = '#253e34';
    ctx.font = `${Math.round(height / 30)}px sans-serif`;
    ctx.fillText('Synthetic demo clip', 12, height - 18);
    return canvas.toDataURL('image/jpeg').split(',')[1];
  }));
  await fixture.close();
  const frames = images.map(image => Buffer.from(image, 'base64'));
  for (const width of [1440, 1220, 390, 320]) {
    for (const scheme of ['light', 'dark']) {
      const context = await browser.newContext({ viewport: { width, height: 1000 }, colorScheme: scheme, hasTouch: width < 720, isMobile: width < 720 });
      const page = await context.newPage();
      page.setDefaultTimeout(15000);
      console.log(`Checking ${width} ${scheme}`);
      const cameras = ['front-room', 'kitchen', 'study'];
      const microphone = 'Demo microphone with a very long device name';
      await context.route('**/ambient', route => route.fulfill({ json: {
        devices: { demo: { name: microphone, healthy: true } },
      } }));
      await context.route('**/events', route => route.fulfill({
        contentType: 'text/event-stream',
        body: `event: ambient\ndata: ${JSON.stringify({ devices: { demo: { name: microphone, healthy: true } } })}\n\nevent: ready\ndata: {}\n\n`,
      }));
      const start = Math.floor(Date.now() / 1000) - 3600;
      const clips = cameras.map((camera, i) => ({
        id: `demo-clip-${i}`, camera, start: start - i * 60, end: start - i * 60 + 10,
        count: null, name: null, kind: 'using_phone', person: null, promoted: false,
      }));
      let releaseListing = () => {};
      let holdListing = new Promise(resolve => { releaseListing = resolve; });
      await context.route('**/activities?*', route => {
        const query = new URL(route.request().url()).searchParams;
        const days = (query.get('days') ?? query.get('day') ?? '').split(',');
        const listing = { items: clips, promotion: [], truncated: false };
        return route.fulfill({ json: { days: Object.fromEntries(days.map((day, i) => [day, i ? { ...listing, items: [] } : listing])) } });
      });
      await context.route('**/activity/kinds', route => route.fulfill({ json: { kinds: [{
        kind: 'using_phone', name: 'Using a phone', description: '', rule: false,
        classes: { ntu: [], kinetics: [] }, learned: { ntu: [], kinetics: [] }, answers: 0, promotion: null,
      }], deleted: [] } }));
      await context.route('**/people/video-thumb.jpg?*', route => {
        const camera = new URL(route.request().url()).searchParams.get('cam');
        return route.fulfill({ contentType: 'image/jpeg', body: frames[cameras.indexOf(camera)] });
      });
      // All gateway requests are answered in-process. No socket reaches a real camera service.
      await context.routeWebSocket('**/ws', socket => {
        socket.send(JSON.stringify({ type: 'hello', transport: 'websocket', webtransport: null }));
        socket.onMessage(async message => {
          const request = JSON.parse(message);
          const send = reply => socket.send(JSON.stringify(reply));
          if (request.type === 'streams') {
            await holdListing;
            send({ type: 'streams', streams: cameras.map((camera, i) => ({
              id: i + 1, name: `${camera}.video`, format: 'jpeg', start_us: 0, end_us: (start + 100) * 1e6,
            })) });
          } else if (request.type === 'spans') {
            send({ ...request, spans: [[request.start_us, request.end_us]] });
          } else if (request.type === 'range') {
            const jpeg = frames[request.stream - 1];
            const header = Buffer.alloc(16);
            header.writeUInt32LE(request.stream, 0);
            header[4] = 4;
            header[5] = 1;
            header.writeBigUInt64LE(BigInt(request.start_us), 8);
            socket.send(Buffer.concat([header, jpeg]));
            send({ ...request, type: 'done', frames: 1, skipped: 0, next_us: null, codecs: ['jpeg'] });
          }
        });
      });
      if (baseline) {
        await context.route('**/static/app/**', route => route.fulfill({ path: path.join(baseline, new URL(route.request().url()).pathname.replace('/static/app/', '')) }));
        await context.route(`${base}/activities`, async route => route.fulfill({
          contentType: 'text/html', body: (await readFile(path.join(baseline, 'index.html'), 'utf8')).replace('{{names_json}}', JSON.stringify(names)),
        }));
      }
      await page.goto(`${base}/activities`);
      const cards = page.getByRole('listitem').filter({ has: page.getByText('Is this “Using a phone”?', { exact: true }) });
      await cards.first().waitFor();
      assert.equal(await cards.count(), 3);
      await page.waitForFunction(() => [...document.querySelectorAll('img')].filter(i => i.src.includes('video-thumb')).every(i => i.complete && i.naturalWidth));
      const measure = () => cards.evaluateAll(elements => elements.map(card => {
        const media = card.firstElementChild.firstElementChild;
        return { card: card.getBoundingClientRect().height, media: media.getBoundingClientRect().height,
          question: [...card.querySelectorAll('p')].find(p => p.textContent.startsWith('Is this'))?.getBoundingClientRect().top - card.getBoundingClientRect().top,
          band: card.querySelector('[data-controls]')?.getAttribute('data-controls') ?? null };
      }));
      await page.evaluate(() => document.fonts.ready);
      const states = { poster: await measure() };
      await page.screenshot({ path: path.join(output, `${width}-${scheme}-poster.png`), fullPage: true });
      for (let i = 0; i < 3; i++) await cards.nth(i).getByRole('button', { name: /^Play Using/ }).click();
      states.loading = await measure();
      releaseListing();
      holdListing = Promise.resolve();
      await page.waitForFunction(() => document.querySelectorAll('canvas[data-captured-at]').length >= 3);
      await cards.nth(2).getByRole('button', { name: 'Pause', exact: true }).click();
      states.playing = await measure();
      if (!baseline) {
        for (const card of await cards.all()) {
          const name = card.getByText(microphone, { exact: true });
          await name.waitFor();
          assert.match(await name.getAttribute('title'), new RegExp(microphone));
          const bounds = await card.getByRole('region', { name: 'Video' }).boundingBox();
          const box = await name.boundingBox();
          assert(box.x >= bounds.x && box.x + box.width <= bounds.x + bounds.width);
          for (const label of ['Yes', 'No', 'Not sure']) {
            const button = card.getByRole('button', { name: label, exact: true });
            assert(await button.locator('.mantine-Button-label').evaluate(el => el.scrollWidth <= el.clientWidth), `${label} must not truncate`);
          }
        }
        const name = cards.first().getByText(microphone, { exact: true });
        assert(await name.evaluate(el => getComputedStyle(el).textOverflow === 'ellipsis' && el.scrollWidth > el.clientWidth));
      }
      await page.mouse.move(0, 0);
      await page.evaluate(() => { document.activeElement?.blur(); window.scrollTo(0, 0); });
      await page.waitForTimeout(150);
      await page.screenshot({ path: path.join(output, `${width}-${scheme}-playing.png`), fullPage: true });
      for (let i = 0; i < 2; i++) await cards.nth(i).getByRole('button', { name: 'Pause', exact: true }).click();
      states.paused = await measure();
      console.log(`Paused ${width} ${scheme}`);
      for (let i = 0; i < 3; i++) await cards.nth(i).getByRole('button', { name: 'Play', exact: true }).click();
      for (let i = 0; i < 3; i++) await cards.nth(i).getByRole('button', { name: 'Play', exact: true }).waitFor({ timeout: 15000 });
      states.ended = await measure();
      if (!baseline) {
        assert.equal(states.playing[0].band, 'side');
        assert.equal(states.playing[1].band, 'bottom');
        for (const state of Object.values(states)) {
          for (let i = 0; i < 3; i++) {
            assert.equal(state[i].card, states.poster[i].card);
            assert.equal(state[i].media, 240);
            assert.equal(state[i].question, states.poster[i].question);
          }
        }
        // Tab through the actual browser controls, including the expand action, without clicking them.
        for (let i = 0; i < 3; i++) {
          const card = cards.nth(i);
          const bounds = await card.getByRole('region', { name: 'Video' }).boundingBox();
          for (const control of await card.getByRole('region', { name: 'Video' }).getByRole('button').all()) {
            const box = await control.boundingBox();
            assert(box.x >= bounds.x && box.y >= bounds.y);
            assert(box.x + box.width <= bounds.x + bounds.width + 1);
            assert(box.y + box.height <= bounds.y + bounds.height + 1);
          }
          const play = card.getByRole('button', { name: 'Play', exact: true });
          await play.focus();
          await page.keyboard.press('Tab');
          assert.match(await page.locator(':focus').getAttribute('aria-label'), /^Speed/);
          await page.keyboard.press('Tab');
          assert.match(await page.locator(':focus').getAttribute('aria-label'), /room microphone/);
          await page.keyboard.press('Tab');
          assert.equal(await page.locator(':focus').getAttribute('aria-label'), 'Open Using a phone bigger');
        }
      }
      measurements.push({ width, scheme, states });
      await writeFile(path.join(output, 'measurements.json'), JSON.stringify(measurements, null, 2));
      await context.close();
    }
  }
} finally {
  await browser.close();
}
console.log(`${measurements.length} viewport/scheme cases passed; evidence: ${output}`);
