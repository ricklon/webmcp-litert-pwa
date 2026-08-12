import { expect, test } from '@playwright/test';

const ORIGINAL_REQUEST = 'I need to pack for my event tomorrow. I have a robot, a voice agent, usb cables, and power. All of which need packing. Then I need get to southbend station and take a lyft to the airbnb.';

test('uses earlier user details when a later request refers to those items', async ({ page }) => {
  await page.addInitScript(() => {
    let promptCount = 0;
    Object.defineProperty(window, 'LanguageModel', {
      configurable: true,
      value: {
        availability: async () => 'available',
        create: async () => ({
          prompt: async (message: string) => {
            promptCount += 1;
            if (promptCount === 3) {
              (window as typeof window & { followUpPlannerPrompt?: string }).followUpPlannerPrompt = message;
            }
            if (promptCount === 1) return JSON.stringify({
              outcome: 'act',
              calls: [
                { name: 'add_task', arguments: { title: 'Pack for event', priority: 'medium' } },
                { name: 'add_task', arguments: { title: 'Get to South Bend Station', priority: 'medium' } },
                { name: 'add_task', arguments: { title: 'Take Lyft to Airbnb', priority: 'medium' } }
              ],
              message: 'I proposed a packing task and two travel tasks.'
            });
            if (promptCount === 2) return JSON.stringify({
              outcome: 'answer',
              calls: [],
              message: 'You mentioned a robot, voice agent, USB cables, and power.'
            });
            return JSON.stringify({
              outcome: 'act',
              calls: [
                { name: 'add_task', arguments: { title: 'Pack robot', priority: 'medium' } },
                { name: 'add_task', arguments: { title: 'Pack voice agent', priority: 'medium' } },
                { name: 'add_task', arguments: { title: 'Pack USB cables', priority: 'medium' } },
                { name: 'add_task', arguments: { title: 'Pack power supply', priority: 'medium' } }
              ],
              message: 'I proposed each previously mentioned packing item separately.'
            });
          },
          contextUsage: 900,
          contextWindow: 4096,
          measureContextUsage: async () => 80,
          destroy: () => undefined
        })
      }
    });
  });
  await page.goto('/');
  await page.evaluate(() => localStorage.clear());
  await page.reload();

  await page.getByLabel('What should we get done?').fill(ORIGINAL_REQUEST);
  await page.locator('.prompt-box').getByRole('button', { name: 'Plan' }).click();
  await page.getByRole('region', { name: 'Proposed actions' }).getByRole('button', { name: 'Approve and execute' }).click();

  await page.getByLabel('What should we get done?').fill('What items do I need to pack?');
  await page.locator('.prompt-box').getByRole('button', { name: 'Plan' }).click();
  await expect(page.locator('.run-feedback')).toContainText('robot, voice agent, USB cables, and power');

  await page.getByLabel('What should we get done?').fill("Let's add tasks for those.");
  await page.locator('.prompt-box').getByRole('button', { name: 'Plan' }).click();

  const proposal = page.getByRole('region', { name: 'Proposed actions' });
  await expect(proposal.getByLabel('title')).toHaveCount(4);
  const capturedPrompt = await page.evaluate(() => (window as typeof window & { followUpPlannerPrompt?: string }).followUpPlannerPrompt);
  expect(capturedPrompt).toContain(ORIGINAL_REQUEST);
  expect(capturedPrompt).toContain('What items do I need to pack?');
  expect(capturedPrompt).toContain('You mentioned a robot, voice agent, USB cables, and power.');
  expect(capturedPrompt).not.toContain('Performance ·');

  await proposal.getByRole('button', { name: 'Approve and execute' }).click();
  await expect(page.getByTestId('today-total-count')).toHaveText('7 total · 0 done');
  await expect(page.locator('.activity-panel')).toContainText(/Performance · .*context 900 \/ 4,096 tokens \(22%\).*output ~80 tokens.*tok\/s/);
});
