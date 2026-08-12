import { expect, test } from '@playwright/test';

async function approveProposal(page: import('@playwright/test').Page) {
  await page.getByRole('button', { name: 'Approve and execute' }).click();
}

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(window, 'LanguageModel', { configurable: true, value: undefined });
  });
});

test('Today reflects every task lifecycle transition', async ({ page }) => {
  await page.goto('/');
  await page.evaluate(() => localStorage.clear());
  await page.reload();
  await page.getByRole('button', { name: /Demo rules/ }).click();

  const panel = page.getByTestId('today-panel');
  const open = page.getByTestId('today-open-count');
  const totals = page.getByTestId('today-total-count');

  await expect(open).toHaveText('0 open');
  await expect(totals).toHaveText('0 total · 0 done');
  await panel.screenshot({ path: 'artifacts/today/01-empty.png' });

  await page.getByLabel('What should we get done?').fill('Add buy coffee filters');
  await page.locator('.prompt-box').getByRole('button', { name: /Plan/ }).click();
  await approveProposal(page);
  await expect(open).toHaveText('1 open');
  await expect(totals).toHaveText('1 total · 0 done');
  await expect(panel.getByText('buy coffee filters', { exact: true })).toBeVisible();
  await panel.screenshot({ path: 'artifacts/today/02-added.png' });

  await panel.getByRole('button', { name: 'Mark buy coffee filters complete' }).click();
  await expect(open).toHaveText('0 open');
  await expect(totals).toHaveText('1 total · 1 done');
  await panel.screenshot({ path: 'artifacts/today/03-completed.png' });

  await panel.getByRole('button', { name: 'Mark buy coffee filters open' }).click();
  await expect(open).toHaveText('1 open');
  await expect(totals).toHaveText('1 total · 0 done');
  await panel.screenshot({ path: 'artifacts/today/04-reopened.png' });

  await panel.getByRole('button', { name: 'Mark buy coffee filters complete' }).click();
  await page.getByLabel('What should we get done?').fill('Clear completed tasks');
  await page.locator('.prompt-box').getByRole('button', { name: /Plan/ }).click();
  page.once('dialog', (dialog) => dialog.accept());
  await approveProposal(page);
  await expect(open).toHaveText('0 open');
  await expect(totals).toHaveText('0 total · 0 done');
  await expect(panel.getByText('No tasks yet.')).toBeVisible();
  await panel.screenshot({ path: 'artifacts/today/05-cleared.png' });
});

test('Reset local data clears recoverable app state', async ({ page }) => {
  await page.goto('/');
  await page.evaluate(() => localStorage.clear());
  await page.reload();
  await page.getByRole('button', { name: /Demo rules/ }).click();
  await page.getByLabel('What should we get done?').fill('Add bad imported task data');
  await page.locator('.prompt-box').getByRole('button', { name: /Plan/ }).click();
  await approveProposal(page);
  await expect(page.getByTestId('today-total-count')).toHaveText('1 total · 0 done');

  page.once('dialog', (dialog) => dialog.accept());
  await page.getByRole('button', { name: 'Reset local data' }).click();

  await expect(page.getByTestId('today-open-count')).toHaveText('0 open');
  await expect(page.getByTestId('today-total-count')).toHaveText('0 total · 0 done');
  await expect(page.getByText('No tasks yet.')).toBeVisible();
  await expect(page.locator('.run-feedback')).toContainText('Local app data reset');
  await expect(page.getByText('Tool calls will appear here with their source.')).toBeVisible();
  await expect.poll(() => page.evaluate(() => localStorage.getItem('local-tools-lab.tasks.v1'))).toBe('[]');
});
