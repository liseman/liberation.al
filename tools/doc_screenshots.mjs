// The documentation's screenshots, taken in headless Chromium from a running demo household.
// tools/doc_screenshots.py starts the demo and runs this; see its docstring. Run by hand as
//   node tools/doc_screenshots.mjs <demo address> <output directory> <frontend directory> [name ...]
//
// Every picture is one entry of SHOTS: its file name, the colour scheme, and the steps that bring the page to
// what the picture shows. One window size for all of them, so the pictures sit together.
// Only pages whose content is the demo's own are here. Add a page when the demo household fills it.
import { createRequire } from 'node:module';
import path from 'node:path';

const [base, out, frontend, ...only] = process.argv.slice(2);
if (!base || !out || !frontend) {
  console.error('usage: node tools/doc_screenshots.mjs <demo address> <output directory> <frontend directory> [name ...]');
  process.exit(2);
}
// Playwright is the dashboard's own dev dependency, so it is found from frontend/, not from this file.
const { chromium } = createRequire(path.resolve(frontend) + path.sep)('playwright');

const SIZE = { width: 1440, height: 900 };
const SETTLE_MS = 2500; // a page's reads after it opens
const STEP_MS = 1500; // what a click brings in

/** Open a page of the dashboard and let its reads finish. */
async function open(page, route) {
  await page.goto(base + route, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(SETTLE_MS);
}

/** The Journal, opened on the demo's one day: the day's row is the one that reads "12 h". */
async function journalDay(page) {
  await open(page, '/journal');
  const day = page.getByText('12 h').first();
  const weeks = page.getByRole('option').filter({ hasText: / to / });
  for (let i = 0; i < (await weeks.count()) && !(await day.isVisible().catch(() => false)); i++) {
    await weeks.nth(i).click();
    await page.waitForTimeout(STEP_MS);
  }
  await day.click();
  await page.waitForTimeout(STEP_MS);
}

async function click(page, locator, wait = STEP_MS) {
  await locator.click();
  await page.waitForTimeout(wait);
}

/** Rex's card, with Training images open and in view. */
async function petTraining(page) {
  await open(page, '/pets');
  await click(page, page.getByText('Rex', { exact: true }).first(), SETTLE_MS);
  await click(page, page.getByRole('button', { name: 'Show', exact: true }).first(), SETTLE_MS);
  await page.getByText('Training images', { exact: true }).scrollIntoViewIfNeeded();
  await page.mouse.wheel(0, 500);
  await page.waitForTimeout(STEP_MS);
}

/** The plan editor, which is a page of its own and takes a while to open. */
async function planEditor(page) {
  await open(page, '/places');
  await click(page, page.getByRole('button', { name: 'Edit plan' }), 9000);
}

/** Live, once its tiles and the people here now have arrived. */
async function live(page) {
  await open(page, '/');
  await page.waitForTimeout(SETTLE_MS);
}

const SHOTS = [
  { name: 'demo-live', steps: live },
  { name: 'demo-live-dark', scheme: 'dark', steps: live },
  { name: 'demo-journal-day', steps: journalDay },
  { name: 'demo-journal-day-dark', scheme: 'dark', steps: journalDay },
  {
    name: 'demo-journal-hour',
    steps: async page => {
      await journalDay(page);
      await click(page, page.getByText('18:00').first());
    },
  },
  {
    name: 'demo-whos-this',
    steps: async page => {
      await open(page, '/people');
      await page.waitForTimeout(SETTLE_MS); // the faces to name come after the list
    },
  },
  {
    name: 'demo-people-card',
    steps: async page => {
      await open(page, '/people');
      await click(page, page.getByText('Dana', { exact: true }).first(), SETTLE_MS);
    },
  },
  {
    name: 'demo-pet-card',
    steps: async page => {
      await open(page, '/pets');
      await click(page, page.getByText('Rex', { exact: true }).first(), SETTLE_MS);
    },
  },
  {
    name: 'demo-pet-remove-selected',
    steps: async page => {
      await petTraining(page);
      await click(page, page.getByRole('checkbox').first()); // the grey tile, listed first
    },
  },
  {
    name: 'demo-ask',
    steps: async page => {
      await live(page);
      await click(page, page.getByRole('button', { name: /Ask/ }).first(), SETTLE_MS);
      const box = page.getByPlaceholder('Ask about your recordings');
      await box.fill('Who came by yesterday?');
      await box.press('Enter');
      await page.waitForTimeout(9000); // the demo's prepared answer
    },
  },
  {
    name: 'demo-activities',
    steps: async page => {
      await open(page, '/activities');
      await page.waitForTimeout(SETTLE_MS); // the clips' pictures
    },
  },
  {
    name: 'demo-places-plan',
    steps: async page => {
      await open(page, '/places');
      await page.waitForTimeout(SETTLE_MS); // the paths are drawn after the plan
    },
  },
  {
    name: 'demo-3d-view',
    steps: async page => {
      await open(page, '/places');
      await click(page, page.getByText('3D view', { exact: true }), 8000); // the cloud's points
    },
  },
  {
    name: 'demo-3d-all-cameras',
    steps: async page => {
      await open(page, '/places');
      await click(page, page.getByText('3D view', { exact: true }), 6000);
      await click(page, page.getByText('All cameras', { exact: true }), 9000); // every camera's cloud, placed
    },
  },
  {
    name: 'demo-zones',
    steps: async page => {
      // A zone is drawn on the front room's rug and named, and not saved.
      await open(page, '/settings#devices');
      await click(page, page.getByRole('button', { name: 'Zones', exact: true }).first(), SETTLE_MS);
      const editor = page.getByRole('dialog').filter({ hasText: 'Zones on' });
      await click(page, editor.getByRole('button', { name: 'Add a zone' }));
      const box = await editor.locator('img, canvas, video').first().boundingBox();
      for (const [x, y] of [[0.47, 0.72], [0.8, 0.6], [0.99, 0.82], [0.6, 0.99]]) {
        await page.mouse.click(box.x + x * box.width, box.y + y * box.height);
        await page.waitForTimeout(300);
      }
      await click(page, editor.getByRole('button', { name: 'Done', exact: true }));
      await editor.getByRole('textbox').first().fill('Rug');
      await page.waitForTimeout(STEP_MS);
    },
  },
  { name: 'demo-plan-editor', steps: planEditor },
  { name: 'demo-plan-editor-dark', scheme: 'dark', steps: planEditor },
  {
    name: 'demo-statistics',
    steps: async page => {
      await open(page, '/statistics');
      await page.waitForTimeout(SETTLE_MS); // the charts draw after the day's numbers arrive
    },
  },
];

const unknown = only.filter(name => !SHOTS.some(s => s.name === name));
if (unknown.length) {
  console.error(`no such picture: ${unknown.join(', ')}. The pictures are: ${SHOTS.map(s => s.name).join(', ')}`);
  process.exit(2);
}

const browser = await chromium.launch();
let failed = 0;
for (const shot of SHOTS.filter(s => !only.length || only.includes(s.name))) {
  const context = await browser.newContext({ viewport: SIZE, colorScheme: shot.scheme ?? 'light' });
  const page = await context.newPage();
  try {
    await shot.steps(page);
    await page.screenshot({ path: path.join(out, `${shot.name}.png`) });
    console.log(`took ${shot.name}`);
  } catch (error) {
    failed += 1;
    console.error(`${shot.name} failed: ${String(error).split('\n')[0]}`);
  }
  await context.close();
}
await browser.close();
process.exit(failed ? 1 : 0);
