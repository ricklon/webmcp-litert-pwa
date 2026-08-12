import { expect, test } from '@playwright/test';

test('turns an explicit all-done statement into reviewed calls for every open task', async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(window, 'LanguageModel', {
      configurable: true,
      value: {
        availability: async () => 'available',
        create: async () => ({
          // Deliberately reproduce the observed model failure. The application
          // guardrail must recover the explicit universal intent.
          prompt: async () => JSON.stringify({ outcome: 'answer', calls: [], message: 'Great, everything is done.' }),
          contextUsage: 1630,
          contextWindow: 9216,
          measureContextUsage: async () => 49,
          destroy: () => undefined
        })
      }
    });
  });
  await page.goto('/');
  await page.evaluate(() => localStorage.setItem('local-tools-lab.tasks.v1', JSON.stringify([
    { id: 'robot', title: 'Pack robot', priority: 'medium', completed: false, createdAt: new Date().toISOString() },
    { id: 'lyft', title: 'Take Lyft to Airbnb', priority: 'medium', completed: false, createdAt: new Date().toISOString() }
  ])));
  await page.reload();

  await page.getByLabel('What should we get done?').fill('OK. I completed all the tasks');
  await page.locator('.prompt-box').getByRole('button', { name: 'Plan' }).click();

  const proposal = page.getByRole('region', { name: 'Proposed actions' });
  await expect(proposal.getByText('complete_task', { exact: true })).toHaveCount(2);
  const taskInputs = proposal.getByRole('textbox', { name: 'task' });
  await expect(taskInputs.nth(0)).toHaveValue('robot');
  await expect(taskInputs.nth(1)).toHaveValue('lyft');
  await expect(page.getByTestId('today-total-count')).toHaveText('2 total · 0 done');

  await proposal.getByRole('button', { name: 'Approve and execute' }).click();
  await expect(page.getByTestId('today-total-count')).toHaveText('2 total · 2 done');
  await expect(page.locator('.activity-panel')).toContainText('Completed “Pack robot”.');
  await expect(page.locator('.activity-panel')).toContainText('Completed “Take Lyft to Airbnb”.');
});
