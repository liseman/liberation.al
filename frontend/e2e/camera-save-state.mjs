/** A control that saves itself: its save state keeps its place, in its own row.
 *
 * Headless demo-only layout regression and review screenshots. The demo has no camera hardware, so the panel's own
 * reads and writes (and the Devices card's camera write) are answered here with made-up controls: what is checked is
 * the layout, on demo pictures.
 *
 * Run from frontend: ../tools/test-slot.sh node e2e/camera-save-state.mjs DEMO_URL OUTPUT_DIR
 * Start pack-demo with this checkout's frontend build first. It changes one control (automatic exposure) in Live's
 * controls panel and in Settings' Devices card, and measures the rows around it with the save state empty, spinning,
 * applying, checked, and failed, at desktop and phone widths: nothing may move. Then it saves pictures of the three
 * states in both colour schemes.
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

/** The demo's address for a path. */
const at = pathname => {
  try {
    return new URL(pathname, url);
  } catch {
    throw new Error('The demo address is not a URL');
  }
};

// The demo's own devices view, which the Devices card reads and writes: a write's reply is the same view.
const devices = await (await fetch(at('/devices'))).json();
assert(devices?.cameras?.length, 'The demo must list cameras for the Devices card check');

/** A locator's box, to the pixel. */
const box = locator => locator.boundingBox();
/** Holds a request until released, so the state can be measured with the spinner up. */
function held() {
  let release = () => {};
  let saw = () => {};
  const promise = new Promise(resolve => {
    release = resolve;
  });
  const seen = new Promise(resolve => {
    saw = resolve;
  });
  return { promise, seen, release, saw };
}

const browser = await chromium.launch({ headless: true });
const measurements = [];
try {
  for (const theme of ['light', 'dark']) {
    for (const width of [1280, 390]) {
      // ---- Live's camera controls panel: one change, beside automatic exposure ----
      const context = await browser.newContext({ viewport: { width, height: 950 }, colorScheme: theme });
      await context.addInitScript(() => {
        localStorage.setItem('liberational.prefs', JSON.stringify({ gettingStartedDone: true, liveTiles: true }));
      });
      let write = held();
      let refusing = false;
      let applying = false;
      let polls = 0;
      await context.route('**/devices/camera-controls*', async route => {
        if (route.request().method() === 'POST') {
          write.saw();
          await write.promise;
          if (refusing) return route.fulfill({ status: 503, json: { error: 'The camera did not answer.' } });
          applying = true;
          polls = 0;
          return route.fulfill({ json: { ...controls, pending: true } });
        }
        // The camera applies the change: the first read after it still says pending, the next says it is done.
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
      const slots = panel.locator('[aria-live="polite"]'); // automatic exposure's first, then each control's, then reset's
      const frequencySlot = slots.nth(3);
      /** Where the panel and the rows in it stand, to the pixel. */
      const boxes = async () => ({
        panel: await box(panel),
        exposure: await box(exposure),
        exposureSlot: await box(slots.first()),
        reset: await box(panel.getByRole('button', { name: 'Reset defaults' })),
      });
      const shot = state =>
        panel.screenshot({
          animations: 'disabled',
          path: `${output}/camera-options-${state}-${theme}-${width}.png`,
        });

      const before = await boxes();
      write = held();
      await exposure.click();
      await write.seen;
      const saving = await boxes();
      assert.deepEqual(saving, before, 'Saving must not move the panel or the row it is in');
      assert.equal(await slots.first().getAttribute('data-phase'), 'saving');
      assert.equal(await frequencySlot.getAttribute('data-phase'), 'idle', 'Only the control that changed shows it');
      await shot('saving');

      write.release();
      await page.waitForTimeout(500); // the reply says the camera is applying the change: this is where the line appeared
      const applyingBoxes = await boxes();
      assert.deepEqual(applyingBoxes, before, 'Applying must not move the panel or the row it is in');
      assert.equal(await frequencySlot.getAttribute('data-phase'), 'idle');

      await page.locator('[data-phase="saved"]').waitFor();
      const applied = await boxes();
      assert.deepEqual(applied, before, 'The check must not move the panel or the row it is in');
      await shot('saved');

      refusing = true;
      write = held();
      await exposure.click();
      write.release();
      await page.getByTestId('save-error').first().waitFor();
      const failed = await boxes();
      assert.deepEqual(failed, before, 'A failed save must not move the panel or the row it is in');
      await shot('error');
      // The reason is on the mark itself, not in a line that would move the controls.
      await page.getByTestId('save-error').first().hover();
      assert.equal(await page.getByRole('tooltip').textContent(), 'The camera did not answer.');
      measurements.push({ where: 'panel', theme, width, before, saving, applying: applyingBoxes, applied, failed });
      console.log(`${theme} ${width}: camera controls measured and pictured`);
      await context.close();

      // ---- Settings' Devices card: the same change, beside automatic exposure's own row ----
      const deviceContext = await browser.newContext({ viewport: { width, height: 950 }, colorScheme: theme });
      await deviceContext.addInitScript(() => {
        localStorage.setItem('liberational.prefs', JSON.stringify({ gettingStartedDone: true }));
      });
      let cameraWrite = held();
      await deviceContext.route('**/devices/camera', async route => {
        if (route.request().method() !== 'POST') return route.continue();
        cameraWrite.saw();
        await cameraWrite.promise;
        return route.fulfill({ json: devices });
      });
      const settings = await deviceContext.newPage();
      settings.setDefaultTimeout(20_000);
      await settings.goto(at('/settings#devices').href);
      const deviceSwitch = settings.getByRole('switch', { name: /^Automatic exposure for / }).first();
      await deviceSwitch.waitFor();
      await settings.evaluate(() => document.fonts.ready);
      await settings.waitForTimeout(1500);
      // The Live tab is still in the page under Settings: take the card from the switch's own section, and the row
      // from the switch itself (the one marked data-toggle), so nothing else named Cameras can be measured by mistake.
      const card = deviceSwitch.locator('xpath=ancestor::section[1]');
      const exposureRow = deviceSwitch.locator('xpath=ancestor::div[@data-toggle][1]');
      const deviceExposure = exposureRow.getByText('Automatic exposure', { exact: true });
      const deviceSlot = exposureRow.locator('[aria-live="polite"]');
      const cardBoxes = async () => ({
        card: await box(card),
        exposure: await box(deviceExposure),
        slot: await box(deviceSlot),
      });
      const beforeDevice = await cardBoxes();
      assert.equal(await deviceSlot.getAttribute('data-phase'), 'idle');
      cameraWrite = held();
      await deviceSwitch.click({ force: true }); // the track covers Mantine's switch input
      await cameraWrite.seen;
      assert.deepEqual(await cardBoxes(), beforeDevice, 'Saving must not move the camera card or its row');
      assert.equal(await deviceSlot.getAttribute('data-phase'), 'saving');
      cameraWrite.release();
      await settings.locator('[data-phase="saved"]').first().waitFor();
      const afterDevice = await cardBoxes();
      assert.deepEqual(afterDevice, beforeDevice, 'The check must not move the camera card or its row');
      // The camera's own card, not the whole section: a section taller than the window screenshots as blank.
      const cameraCard = deviceSwitch.locator('xpath=ancestor::div[contains(@class,"_source_")][1]');
      await cameraCard.screenshot({
        animations: 'disabled',
        path: `${output}/devices-exposure-saved-${theme}-${width}.png`,
      });
      measurements.push({ where: 'devices', theme, width, before: beforeDevice, after: afterDevice });
      console.log(`${theme} ${width}: Devices card measured and pictured`);
      await deviceContext.close();
    }
  }
  console.log(`Passed ${measurements.length} measurements; screenshots in ${output}`);
} finally {
  await browser.close();
}
