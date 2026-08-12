import { expect, test } from '@playwright/test';

test.skip(process.env.RUN_REAL_MODEL !== '1', 'Opt-in evaluation requires the actual Chrome built-in model.');

test('actual Chrome model handles the realistic event-trip scenario', async ({ page }) => {
  test.setTimeout(5 * 60 * 1000);
  await page.goto('/');
  await page.evaluate(() => localStorage.clear());
  await page.reload();

  if (await page.locator('.state').textContent() !== 'chrome') {
    const chromeButton = page.getByRole('button', { name: /Chrome built-in/ });
    await expect(chromeButton, 'Chrome Prompt API must be supported for this opt-in evaluation.').toBeEnabled();
    await chromeButton.click();
  }
  await expect(page.locator('.state')).toHaveText('chrome', { timeout: 2 * 60 * 1000 });

  await page.getByRole('button', { name: /Plan tomorrow’s event trip/ }).click();
  await page.getByRole('button', { name: 'Load scenario' }).click();
  await page.getByRole('button', { name: 'Run workflow' }).click();

  await expect(page.locator('.scenario-state')).toHaveText('passed', { timeout: 2 * 60 * 1000 });
  await expect(page.getByTestId('today-total-count')).toHaveText('6 total · 0 done');

  await page.getByRole('button', { name: /Record newly finished work/ }).click();
  page.once('dialog', (dialog) => dialog.accept());
  await page.getByRole('button', { name: 'Load scenario' }).click();
  await page.getByRole('button', { name: 'Run workflow' }).click();

  await expect(page.locator('.scenario-state')).toHaveText('passed', { timeout: 2 * 60 * 1000 });
  await expect(page.getByTestId('today-total-count')).toHaveText('1 total · 1 done');
});
