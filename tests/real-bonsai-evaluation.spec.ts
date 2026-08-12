import { expect, test } from '@playwright/test';

test.skip(process.env.RUN_REAL_BONSAI !== '1', 'Opt-in evaluation downloads and runs the actual Bonsai 27B model.');

test('actual Bonsai 27B plans and executes a reviewed task', async ({ page }) => {
  test.setTimeout(30 * 60 * 1000);
  await page.goto('/');
  await page.evaluate(() => localStorage.clear());
  await page.reload();

  const gpu = await page.evaluate(async () => {
    if (!navigator.gpu) return null;
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return null;
    return {
      architecture: adapter.info.architecture,
      device: adapter.info.device,
      vendor: adapter.info.vendor,
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize
    };
  });
  console.log('WebGPU adapter:', gpu);
  expect(gpu, 'A hardware WebGPU adapter is required.').not.toBeNull();

  page.once('dialog', (dialog) => dialog.accept());
  await page.getByRole('button', { name: /Bonsai custom.*27B/ }).click();

  const deadline = Date.now() + 25 * 60 * 1000;
  let previousNote = '';
  while (Date.now() < deadline && await page.locator('.state').textContent() !== 'bonsai') {
    const note = await page.locator('.runtime-note').textContent() ?? '';
    if (note !== previousNote) {
      console.log(note);
      previousNote = note;
    }
    const failure = await page.getByText('Bonsai 27B could not start').count();
    if (failure) throw new Error(await page.locator('[role="status"]').textContent() ?? 'Bonsai failed to load.');
    await page.waitForTimeout(2_000);
  }
  await expect(page.locator('.state')).toHaveText('bonsai');

  await page.getByLabel('What should we get done?').fill('Add a medium-priority task titled Verify Bonsai WebGPU integration');
  await page.locator('.prompt-box').getByRole('button', { name: 'Plan' }).click();

  const proposal = page.getByRole('region', { name: 'Proposed actions' });
  await expect(proposal).toBeVisible({ timeout: 5 * 60 * 1000 });
  await expect(proposal.getByText('add_task', { exact: true })).toBeVisible();
  await expect(proposal.getByRole('textbox', { name: 'title' })).toHaveValue(/Verify Bonsai WebGPU integration/i);
  await proposal.getByRole('button', { name: 'Approve and execute' }).click();

  await expect(page.getByTestId('today-total-count')).toHaveText('1 total · 0 done');
  await expect(page.locator('.activity-panel')).toContainText(/Added “Verify Bonsai WebGPU integration”/i);
});
