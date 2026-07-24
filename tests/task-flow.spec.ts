import { expect, test } from '@playwright/test';

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await page.evaluate(() => localStorage.clear());
  await page.reload();
});

test('adds, completes, and clears a task with visible feedback', async ({ page }) => {
  await expect(page.getByRole('heading', { name: /Today/ })).toContainText('0 open');
  await expect(page.getByText('0 total · 0 done')).toBeVisible();

  await page.getByLabel('What should we get done?').fill('Add buy coffee filters');
  await page.locator('.prompt-box').getByRole('button', { name: /Run/ }).click();

  await expect(page.getByText('buy coffee filters', { exact: true })).toBeVisible();
  await expect(page.getByRole('heading', { name: /Today/ })).toContainText('1 open');
  await expect(page.getByText('1 total · 0 done')).toBeVisible();
  await expect(page.locator('.run-feedback')).toContainText('Request completed');
  await expect(page.locator('.run-feedback')).toContainText('add_task');

  await page.getByLabel('What should we get done?').fill("Let's mark buy coffee filters complete");
  await page.locator('.prompt-box').getByRole('button', { name: /Run/ }).click();
  await expect(page.getByRole('heading', { name: /Today/ })).toContainText('0 open');
  await expect(page.getByText('1 total · 1 done')).toBeVisible();
  await expect(page.locator('.run-feedback')).toContainText('complete_task');

  page.once('dialog', (dialog) => dialog.accept());
  await page.getByLabel('What should we get done?').fill('Clear completed tasks');
  await page.locator('.prompt-box').getByRole('button', { name: /Run/ }).click();

  await expect(page.getByText('No tasks yet.')).toBeVisible();
  await expect(page.getByText('0 total · 0 done')).toBeVisible();
  await expect(page.locator('.run-feedback')).toContainText('clear_completed');
});

test('persists task state across a reload', async ({ page }) => {
  await page.getByLabel('What should we get done?').fill('Add submit expense report');
  await page.locator('.prompt-box').getByRole('button', { name: /Run/ }).click();
  await expect(page.getByText('submit expense report', { exact: true })).toBeVisible();

  await page.reload();

  await expect(page.getByText('submit expense report', { exact: true })).toBeVisible();
  await expect(page.getByText('1 total · 0 done')).toBeVisible();
});

test('completes a task when the request contains a minor spelling error', async ({ page }) => {
  await page.getByLabel('What should we get done?').fill('Add buy coffee filters');
  await page.locator('.prompt-box').getByRole('button', { name: /Run/ }).click();
  await expect(page.getByText('1 total · 0 done')).toBeVisible();

  await page.getByLabel('What should we get done?').fill('Mark "buy cofee filters as complete"');
  await page.locator('.prompt-box').getByRole('button', { name: /Run/ }).click();

  await expect(page.getByText('0 open', { exact: false })).toBeVisible();
  await expect(page.getByText('1 total · 1 done')).toBeVisible();
  await expect(page.locator('.run-feedback')).toContainText('Request completed');
});

test('loads and runs the typo recovery scenario from the interface', async ({ page }) => {
  await page.getByRole('button', { name: /Demo rules/ }).click();
  await page.getByRole('button', { name: /Recover from a typo/ }).click();
  await page.getByRole('button', { name: 'Load scenario' }).click();

  await expect(page.getByText('2 total · 0 done')).toBeVisible();
  await page.getByRole('button', { name: 'Run workflow' }).click();

  await expect(page.locator('.scenario-state')).toHaveText('passed');
  await expect(page.locator('.scenario-result')).toContainText('Passed all 1 step');
  await expect(page.getByText('2 total · 1 done')).toBeVisible();
});

test('clear-finished scenario preserves open work', async ({ page }) => {
  await page.getByRole('button', { name: /Demo rules/ }).click();
  await page.getByRole('button', { name: /Clear finished work/ }).click();
  await page.getByRole('button', { name: 'Load scenario' }).click();
  await expect(page.getByText('3 total · 2 done')).toBeVisible();

  page.once('dialog', (dialog) => dialog.accept());
  await page.getByRole('button', { name: 'Run workflow' }).click();

  await expect(page.locator('.scenario-state')).toHaveText('passed');
  await expect(page.getByText('1 total · 0 done')).toBeVisible();
  await expect(page.getByText('plan trip route', { exact: true })).toBeVisible();
});

test('natural update completes multiple matching tasks', async ({ page }) => {
  await page.getByRole('button', { name: /Demo rules/ }).click();
  await page.getByRole('button', { name: /Understand completed work/ }).click();
  await page.getByRole('button', { name: 'Load scenario' }).click();
  await expect(page.getByText('3 total · 0 done')).toBeVisible();

  await page.getByRole('button', { name: 'Run workflow' }).click();

  await expect(page.locator('.scenario-state')).toHaveText('passed');
  await expect(page.getByText('3 total · 2 done')).toBeVisible();
  await expect(page.locator('.run-feedback')).toContainText('complete_task');
});

test('Enter submits while Shift+Enter inserts a new line', async ({ page }) => {
  await page.getByRole('button', { name: /Demo rules/ }).click();
  const prompt = page.getByLabel('What should we get done?');
  await prompt.fill('Add buy pears');
  await prompt.press('Shift+Enter');
  await expect(prompt).toHaveValue('Add buy pears\n');
  await expect(page.getByText('buy pears', { exact: true })).toHaveCount(0);

  await prompt.press('Enter');
  await expect(page.getByText('buy pears', { exact: true })).toBeVisible();
  await expect(page.getByText('1 total · 0 done')).toBeVisible();
});
