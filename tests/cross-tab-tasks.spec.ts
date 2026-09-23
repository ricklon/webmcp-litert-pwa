import { expect, test } from '@playwright/test';

test('a task added in one tab survives an edit made in another tab', async ({ context }) => {
  const first = await context.newPage();
  const second = await context.newPage();
  for (const page of [first, second]) {
    await page.addInitScript(() => {
      Object.defineProperty(window, 'LanguageModel', { configurable: true, value: undefined });
    });
  }
  await first.goto('/');
  await first.evaluate(() => localStorage.clear());
  await first.reload();
  await second.goto('/');

  async function addTask(page: typeof first, title: string) {
    await page.getByLabel('What should we get done?').fill(`Add ${title}`);
    await page.locator('.prompt-box').getByRole('button', { name: /Plan/ }).click();
    await page.getByRole('button', { name: 'Approve and execute' }).click();
    await expect(page.getByTestId('today-panel').getByText(title, { exact: true })).toBeVisible();
  }

  await addTask(first, 'buy coffee filters');
  await expect(second.getByTestId('today-panel').getByText('buy coffee filters', { exact: true })).toBeVisible();

  await addTask(second, 'plan trip route');
  for (const page of [first, second]) {
    await expect(page.getByText('2 total · 0 done')).toBeVisible();
  }
  await first.reload();
  await expect(first.getByText('2 total · 0 done')).toBeVisible();
});
