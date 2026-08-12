import { expect, test } from '@playwright/test';

test('replans a write proposal from user feedback before execution', async ({ page }) => {
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
                outcome: 'act',
                calls: [{ name: 'add_task', arguments: { title: 'pack electronics', priority: 'medium' } }],
                message: 'I proposed one packing task.'
              })
              : promptCount === 2 ? JSON.stringify({
                outcome: 'act',
                calls: [
                  { name: 'add_task', arguments: { title: 'pack robot', priority: 'medium' } },
                  { name: 'add_task', arguments: { title: 'pack voice agent', priority: 'medium' } }
                ],
                message: 'I split the packing work into two tasks.'
              }) : JSON.stringify({
                outcome: 'act',
                calls: [{ name: 'add_task', arguments: { title: 'pack USB-C cables', priority: 'medium' } }],
                message: 'I added the missing cable task.'
              });
          },
          contextUsage: 600,
          contextWindow: 4096,
          measureContextUsage: async () => 40,
          destroy: () => undefined
        })
      }
    });
  });
  await page.goto('/');
  await page.evaluate(() => localStorage.clear());
  await page.reload();

  await page.getByLabel('What should we get done?').fill('Pack my robot and voice agent');
  await page.locator('.prompt-box').getByRole('button', { name: 'Plan' }).click();
  await expect(page.getByRole('region', { name: 'Proposed actions' }).getByLabel('title')).toHaveValue('pack electronics');
  await expect(page.getByTestId('today-total-count')).toHaveText('0 total · 0 done');

  await page.getByLabel('Refine this proposal').fill('Split that into separate robot and voice agent tasks');
  await page.locator('.prompt-box').getByRole('button', { name: 'Refine' }).click();

  const revised = page.getByRole('region', { name: 'Proposed actions' });
  await expect(revised.getByLabel('title').nth(0)).toHaveValue('pack robot');
  await expect(revised.getByLabel('title').nth(1)).toHaveValue('pack voice agent');
  await expect(page.getByTestId('today-total-count')).toHaveText('0 total · 0 done');

  await revised.getByRole('button', { name: 'Approve and execute' }).click();
  await expect(page.getByTestId('today-total-count')).toHaveText('2 total · 0 done');
  await expect(page.getByRole('region', { name: 'Executed plan' })).toBeVisible();

  await page.getByRole('button', { name: 'Refine result' }).click();
  await page.getByLabel('Refine the completed result').fill('Also add the USB-C cables I forgot');
  await page.locator('.prompt-box').getByRole('button', { name: 'Refine' }).click();
  const followUp = page.getByRole('region', { name: 'Proposed actions' });
  await expect(followUp.getByLabel('title')).toHaveValue('pack USB-C cables');
  await expect(page.getByTestId('today-total-count')).toHaveText('2 total · 0 done');
  await followUp.getByRole('button', { name: 'Approve and execute' }).click();
  await expect(page.getByTestId('today-total-count')).toHaveText('3 total · 0 done');
  await expect(page.locator('.activity-panel')).not.toContainText('"ok":true');
});
