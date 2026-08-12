import { expect, test } from '@playwright/test';

test('executes an ordered add-then-complete proposal inferred by the model', async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(window, 'LanguageModel', {
      configurable: true,
      value: {
        availability: async () => 'available',
        create: async () => ({
          prompt: async () => JSON.stringify({
            outcome: 'act',
            calls: [
              { name: 'add_task', arguments: { title: 'pack soldering iron', priority: 'medium' } },
              { name: 'complete_task', arguments: { task: 'pack soldering iron' } }
            ],
            message: 'I interpreted this as newly reported completed work.'
          }),
          contextUsage: 500,
          contextWindow: 4096,
          measureContextUsage: async () => 50,
          destroy: () => undefined
        })
      }
    });
  });
  await page.goto('/');
  await page.evaluate(() => localStorage.clear());
  await page.reload();

  await page.getByLabel('What should we get done?').fill('I packed a soldering iron as well');
  await page.locator('.prompt-box').getByRole('button', { name: 'Plan' }).click();

  const proposal = page.getByRole('region', { name: 'Proposed actions' });
  await expect(proposal.getByText('add_task', { exact: true })).toBeVisible();
  await expect(proposal.getByText('complete_task', { exact: true })).toBeVisible();
  await expect(page.getByTestId('today-total-count')).toHaveText('0 total · 0 done');

  await proposal.getByRole('button', { name: 'Approve and execute' }).click();
  await expect(page.getByTestId('today-total-count')).toHaveText('1 total · 1 done');
  await expect(page.getByTestId('today-panel').getByText('pack soldering iron', { exact: true }).locator('..').locator('..')).toHaveClass(/completed/);
});
