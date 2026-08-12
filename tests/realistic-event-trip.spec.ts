import { expect, test } from '@playwright/test';

const REQUEST = 'I need to pack for my event tomorrow. I have a robot, a voice agent, usb cables, and power. All of which need packing. Then I need get to southbend statuion and take a lyft to the airbnb.';

test('reviews and executes a realistic event packing and travel plan', async ({ page }) => {
  await page.addInitScript(() => {
    const calls = [
      { name: 'add_task', arguments: { title: 'Pack robot for tomorrow’s event', priority: 'medium' } },
      { name: 'add_task', arguments: { title: 'Pack voice agent for tomorrow’s event', priority: 'medium' } },
      { name: 'add_task', arguments: { title: 'Pack USB cables', priority: 'medium' } },
      { name: 'add_task', arguments: { title: 'Pack power supplies', priority: 'medium' } },
      { name: 'add_task', arguments: { title: 'Get to South Bend Station', priority: 'medium' } },
      { name: 'add_task', arguments: { title: 'Take Lyft from South Bend Station to Airbnb', priority: 'medium' } }
    ];
    Object.defineProperty(window, 'LanguageModel', {
      configurable: true,
      value: {
        availability: async () => 'available',
        create: async () => ({
          prompt: async (message: string) => {
            (window as typeof window & { capturedPlannerPrompt?: string }).capturedPlannerPrompt = message;
            return JSON.stringify({ outcome: 'act', calls, message: 'I proposed four packing tasks and two travel legs.' });
          },
          contextUsage: 900,
          contextWindow: 4096,
          measureContextUsage: async () => 100,
          destroy: () => undefined
        })
      }
    });
  });
  await page.goto('/');
  await page.evaluate(() => localStorage.clear());
  await page.reload();

  await page.getByLabel('What should we get done?').fill(REQUEST);
  await page.locator('.prompt-box').getByRole('button', { name: 'Plan' }).click();

  const proposal = page.getByRole('region', { name: 'Proposed actions' });
  await expect(proposal.getByLabel('title')).toHaveCount(6);
  await expect(page.getByTestId('today-total-count')).toHaveText('0 total · 0 done');
  await expect(proposal).toContainText('four packing tasks and two travel legs');
  const capturedPrompt = await page.evaluate(() => (window as typeof window & { capturedPlannerPrompt?: string }).capturedPlannerPrompt);
  expect(capturedPrompt).toContain(REQUEST);
  expect(capturedPrompt).toContain('Runtime clock:');
  const browserTimeZone = await page.evaluate(() => Intl.DateTimeFormat().resolvedOptions().timeZone);
  expect(capturedPrompt).toContain(`IANA time zone: ${browserTimeZone}`);
  expect(capturedPrompt).toContain('UTC timestamp:');

  await proposal.getByRole('button', { name: 'Approve and execute' }).click();
  await expect(page.getByTestId('today-total-count')).toHaveText('6 total · 0 done');
  for (const title of ['Pack robot for tomorrow’s event', 'Pack voice agent for tomorrow’s event', 'Pack USB cables', 'Pack power supplies', 'Get to South Bend Station', 'Take Lyft from South Bend Station to Airbnb']) {
    await expect(page.getByTestId('today-panel').getByText(title, { exact: true })).toBeVisible();
  }
});
