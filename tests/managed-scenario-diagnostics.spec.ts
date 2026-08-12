import { expect, test } from '@playwright/test';

test('reports when a model answers a managed scenario without tool calls', async ({ page }) => {
  await page.addInitScript(() => {
    let promptCount = 0;
    Object.defineProperty(window, 'LanguageModel', {
      configurable: true,
      value: {
        availability: async () => 'available',
        create: async () => ({
          prompt: async () => {
            promptCount += 1;
            return promptCount === 1
              ? JSON.stringify({
                outcome: 'answer',
                calls: [],
                message: 'I would pack the equipment and arrange transportation.'
              })
              : JSON.stringify({
                outcome: 'act',
                calls: [{ name: 'add_task', arguments: { title: 'Recovery task', priority: 'medium' } }],
                message: 'I proposed the recovery task.'
              });
          },
          destroy: () => undefined
        })
      }
    });
  });
  await page.goto('/');
  await page.evaluate(() => localStorage.clear());
  await page.reload();

  await page.getByRole('button', { name: /Plan tomorrow’s event trip/ }).click();
  await page.getByRole('button', { name: 'Load scenario' }).click();
  await page.getByRole('button', { name: 'Run workflow' }).click();

  await expect(page.locator('.scenario-state')).toHaveText('failed');
  await expect(page.locator('.scenario-result')).toContainText('answered without proposing any tool calls');
  await expect(page.locator('.scenario-result')).toContainText('I would pack the equipment and arrange transportation.');
  await expect(page.getByTestId('today-total-count')).toHaveText('0 total · 0 done');

  await page.getByLabel('What should we get done?').fill('Add a recovery task');
  await page.locator('.prompt-box').getByRole('button', { name: 'Plan' }).click();
  await page.getByRole('region', { name: 'Proposed actions' }).getByRole('button', { name: 'Approve and execute' }).click();

  await expect(page.locator('.scenario-state')).toHaveText('stale');
  await expect(page.locator('.scenario-result')).toContainText('previous result is stale');
  await expect(page.getByTestId('today-total-count')).toHaveText('1 total · 0 done');
});
