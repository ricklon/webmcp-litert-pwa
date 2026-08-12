import { expect, test } from '@playwright/test';

test('model decomposes the Boston trip paragraph into distinct tasks', async ({ page }) => {
  await page.addInitScript(() => {
    const calls = [
      ['pack for Boston trip', 'high'],
      ['bring train ticket', 'high'],
      ['pack clothes', 'medium'],
      ['pack project materials', 'medium'],
      ['bring project power supplies', 'high'],
      ['bring project tools', 'high']
    ].map(([title, priority]) => ({ name: 'add_task', arguments: { title, priority } }));
    Object.defineProperty(window, 'LanguageModel', {
      configurable: true,
      value: {
        availability: async () => 'available',
        create: async () => ({
          prompt: async () => JSON.stringify({ outcome: 'act', calls, message: 'I separated the trip preparation into six tasks.' }),
          measureContextUsage: async () => 80,
          contextUsage: 750,
          contextWindow: 4096,
          destroy: () => undefined
        })
      }
    });
  });
  await page.goto('/');
  await page.evaluate(() => localStorage.clear());
  await page.reload();

  await expect(page.locator('.state')).toHaveText('chrome');
  await page.getByRole('button', { name: /Break down a Boston trip/ }).click();
  await page.getByRole('button', { name: 'Load scenario' }).click();
  await page.getByRole('button', { name: 'Run workflow' }).click();

  await expect(page.locator('.scenario-state')).toHaveText('passed');
  await expect(page.getByTestId('today-total-count')).toHaveText('6 total · 0 done');
  await expect(page.getByTestId('today-open-count')).toHaveText('6 open');
  for (const title of ['pack for Boston trip', 'bring train ticket', 'pack clothes', 'pack project materials', 'bring project power supplies', 'bring project tools']) {
    await expect(page.getByTestId('today-panel').getByText(title, { exact: true })).toBeVisible();
  }
  await expect(page.locator('.run-feedback .called-tools code')).toHaveCount(6);
  await expect(page.locator('.planner-metrics')).toContainText('750 / 4,096');
  await expect(page.locator('.planner-metrics')).toContainText('tok/s');
});
