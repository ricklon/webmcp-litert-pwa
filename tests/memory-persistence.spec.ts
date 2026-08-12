import { expect, test } from '@playwright/test';

test('restores conversation context and an unapproved proposal after reload', async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(window, 'LanguageModel', {
      configurable: true,
      value: {
        availability: async () => 'available',
        create: async () => ({
          prompt: async (message: string) => {
            if (message.includes('User: What did I ask to pack?')) {
              (window as typeof window & { restoredContextPrompt?: string }).restoredContextPrompt = message;
              return JSON.stringify({ outcome: 'answer', calls: [], message: 'You asked to pack the robot.' });
            }
            return JSON.stringify({
              outcome: 'act',
              calls: [{ name: 'add_task', arguments: { title: 'Pack robot', priority: 'medium' } }],
              message: 'I proposed the robot packing task.'
            });
          },
          contextUsage: 500,
          contextWindow: 4096,
          measureContextUsage: async () => 50,
          destroy: () => undefined
        })
      }
    });
  });
  await page.goto('/');

  await page.getByLabel('What should we get done?').fill('Remember that I need to pack the robot.');
  await page.locator('.prompt-box').getByRole('button', { name: 'Plan' }).click();
  await expect(page.getByRole('region', { name: 'Proposed actions' })).toBeVisible();
  await expect(page.getByTestId('today-total-count')).toHaveText('0 total · 0 done');

  await expect.poll(() => page.evaluate(async () => {
    const database = await new Promise<IDBDatabase>((resolve, reject) => {
      const request = indexedDB.open('local-tools-lab.memory.v1');
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
    const transaction = database.transaction('sessions', 'readonly');
    const sessions = await new Promise<Array<{ planReview?: { status?: string } }>>((resolve, reject) => {
      const request = transaction.objectStore('sessions').getAll();
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
    database.close();
    return sessions.some((session) => session.planReview?.status === 'proposed');
  })).toBe(true);

  await page.reload();
  const restored = page.getByRole('region', { name: 'Proposed actions' });
  await expect(restored).toBeVisible();
  await expect(restored.getByLabel('title')).toHaveValue('Pack robot');
  await expect(page.locator('.run-feedback')).toContainText('Proposal restored');
  await expect(page.locator('.activity-panel')).toContainText('Remember that I need to pack the robot.');
  await expect(page.getByTestId('today-total-count')).toHaveText('0 total · 0 done');

  await restored.getByRole('button', { name: 'Approve and execute' }).click();
  await expect(page.getByTestId('today-total-count')).toHaveText('1 total · 0 done');

  await page.getByLabel('What should we get done?').fill('What did I ask to pack?');
  await page.locator('.prompt-box').getByRole('button', { name: 'Plan' }).click();
  await expect(page.locator('.run-feedback')).toContainText('You asked to pack the robot.');
  const prompt = await page.evaluate(() => (window as typeof window & { restoredContextPrompt?: string }).restoredContextPrompt);
  expect(prompt).toContain('Remember that I need to pack the robot.');
});

test('switches conversations without changing shared task state', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByText('saved locally')).toBeVisible();
  const firstConversation = await page.getByLabel('Conversation').inputValue();

  await page.getByRole('button', { name: 'New conversation' }).click();
  await expect(page.getByLabel('Conversation').locator('option')).toHaveCount(2);
  await expect(page.getByLabel('Conversation')).not.toHaveValue(firstConversation);
  const secondConversation = await page.getByLabel('Conversation').inputValue();
  expect(secondConversation).not.toBe(firstConversation);

  await page.getByLabel('Conversation').selectOption(firstConversation);
  await expect(page.getByLabel('Conversation')).toHaveValue(firstConversation);
});
