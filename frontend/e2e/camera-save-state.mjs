/** A camera's controls panel: its save state keeps its place.
 *
 * Headless demo-only layout regression and review screenshots. The demo has no camera hardware, so the panel's own
 * reads and writes are answered here with made-up controls: what is checked is the panel's layout, on demo pictures.
 *
 * Run from frontend: ../tools/test-slot.sh node e2e/camera-save-state.mjs DEMO_URL OUTPUT_DIR
 * Start pack-demo with this checkout's frontend build first. It measures the controls' boxes with the save state
 * empty, spinning, checked, and failed, and fails if any of them moves; then it saves a picture of the three states in
 * both colour schemes.
 */
import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import { chromium } from 'playwright';

const [base, output] = process.argv.slice(2);
assert(base && output, 'Pass the demo URL and screenshot directory');
const url = (() => {
  try {
    return new URL(base);
  } catch {
    throw new Error('The demo address is not a URL');
  }
})();
assert(['localhost', '127.0.0.1'].includes(url.hostname), 'Use a local demo');
const html = await (await fetch(url)).text();
assert(html.includes('window.__PACK_DEMO__=true'), 'Only the isolated demo dashboard is allowed');
assert(html.includes('\\"owner\\": \\"Alice\\"'), 'Use the made-up demo household');
await mkdir(output, { recursive: true });

// What the panel reads and writes: made-up controls, shaped like the ones a camera exposes.
const controls = {
  camera: 'cam0',
  available: true,
  can_control: true,
  auto_exposure: true,
  controls: [
    {
      id: 1,
      name: 'Brightness',
      type: 'integer',
      min: 0,
      max: 100,
      step: 1,
      default: 50,
      current: 50,
      disabled: false,
    },
    { id: 2, name: 'Contrast', type: 'integer', min: 0, max: 100, step: 1, default: 50, current: 40, disabled: false },
    {
      id: 3,
      name: 'Power line frequency',
      type: 'menu',
      min: 0,
      max: 2,
      step: 1,
      default: 0,
      current: 0,
      disabled: false,
      options: [
        { value: 0, label: 'Off' },
        { value: 2, label: '60 Hz' },
      ],
    },
  ],
};

const browser = await chromium.launch({ headless: true });
const measurements = [];
try {
  for (const theme of ['light', 'dark']) {
    const context = await browser.newContext({ viewport: { width: 1280, height: 950 }, colorScheme: theme });
    await context.addInitScript(() => {
      localStorage.setItem('liberational.prefs', JSON.stringify({ gettingStartedDone: true, liveTiles: true }));
    });
    // The write waits on this until the panel has been measured with the spinner showing. It then answers that the
    // camera is applying the change, and the first read after it still says so: that is the state the old message line
    // appeared in, and the one that pushed the controls down.
    let hold = null;
    let release = () => {};
    let wrote = () => {};
    const wroteOnce = new Promise(resolve => {
      wrote = resolve;
    });
    let refuse = false;
    let applying = false;
    let polls = 0;
    await context.route('**/devices/camera-controls*', async route => {
      const request = route.request();
      if (request.method() === 'POST') {
        wrote();
        if (hold) await hold;
        if (refuse) return route.fulfill({ status: 503, json: { error: 'The camera did not answer.' } });
        applying = true;
        polls = 0;
        return route.fulfill({ json: { ...controls, pending: true } });
      }
      if (applying && polls++ < 1) return route.fulfill({ json: { ...controls, pending: true } });
      applying = false;
      return route.fulfill({ json: controls });
    });
    const page = await context.newPage();
    page.setDefaultTimeout(20_000);
    await page.goto(url.href);
    await page
      .getByRole('button', { name: /^Controls for / })
      .first()
      .click();
    const panel = page.locator('.mantine-Popover-dropdown');
    const exposure = panel.getByText('Automatic exposure', { exact: true });
    await exposure.waitFor();
    await page.evaluate(() => document.fonts.ready);
    await page.waitForTimeout(2000); // Live's tiles and previews settling under the open panel
    const slot = panel.locator('[aria-live="polite"]');
    const reset = panel.getByRole('button', { name: 'Reset defaults' });
    /** Where the panel and the controls in it stand, to the pixel. */
    const boxes = async () => ({
      panel: await panel.boundingBox(),
      slot: (await slot.count()) ? await slot.boundingBox() : null,
      exposure: await exposure.boundingBox(),
      reset: await reset.boundingBox(),
    });
    const shot = async state => {
      await panel.screenshot({ animations: 'disabled', path: `${output}/camera-options-${state}-${theme}.png` });
    };

    const before = await boxes();

    hold = new Promise(resolve => {
      release = resolve;
    });
    await exposure.click();
    await wroteOnce; // the write is on its way and the panel is saving: the spinner is up
    const saving = await boxes();
    assert.deepEqual(saving, before, 'Saving must not move the panel or the controls in it');
    await shot('saving');

    release();
    hold = null;
    await page.waitForTimeout(500); // the reply says the camera is applying the change: this is where the line appeared
    const applyingBoxes = await boxes();
    assert.deepEqual(applyingBoxes, before, 'Applying must not move the panel or the controls in it');
    assert.ok(applyingBoxes.slot, 'The save state has a slot of its own, in the panel from the first render');
    assert.equal(await slot.getAttribute('data-phase'), 'saving');

    await page.locator('[data-phase="saved"]').waitFor();
    const applied = await boxes();
    assert.deepEqual(applied, before, 'The check must not move the panel or the controls in it');
    await shot('saved');

    refuse = true;
    await exposure.click();
    await page.getByTestId('save-error').waitFor();
    const failed = await boxes();
    assert.deepEqual(failed, before, 'A failed save must not move the panel or the controls in it');
    await shot('error');

    // The reason is on the mark itself, not in a line that would move the controls.
    await page.getByTestId('save-error').hover();
    assert.equal(await page.getByRole('tooltip').textContent(), 'The camera did not answer.');
    measurements.push({ theme, before, saving, applying: applyingBoxes, applied, failed });
    console.log(`${theme}: measured and pictured`);
    await context.close();
  }
  console.log(`Passed ${measurements.length} colour schemes; 6 screenshots in ${output}`);
} finally {
  await browser.close();
}
