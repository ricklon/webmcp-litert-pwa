import { expect, test } from '@playwright/test';
import type { PlannerTraceEntry } from '../src/types';

test('records clarification and follow-up tool decisions for benchmark scoring', async ({ page }) => {
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
              ? JSON.stringify({ outcome: 'clarify', calls: [], message: 'Which task: submit report or review report?' })
              : JSON.stringify({
                outcome: 'act',
                calls: [{ name: 'complete_task', arguments: { task: 'submit report' } }],
                message: 'I will complete submit report.'
              });
          },
          contextUsage: 500,
          contextWindow: 4096,
          measureContextUsage: async () => 30,
          destroy: () => undefined
        })
      }
    });
  });
  await page.goto('/');
  await page.evaluate(() => localStorage.clear());
  await page.reload();

  await page.getByRole('button', { name: /Clarify an ambiguous completion/ }).click();
  await page.getByRole('button', { name: 'Load scenario' }).click();
  await page.getByRole('button', { name: 'Run workflow' }).click();

  await expect(page.locator('.scenario-state')).toHaveText('passed');
  await expect(page.getByTestId('today-total-count')).toHaveText('2 total · 1 done');
  const raw = await page.getByTestId('scenario-trace').getAttribute('data-trace');
  const trace = JSON.parse(raw ?? '[]') as PlannerTraceEntry[];
  expect(trace).toHaveLength(2);
  expect(trace[0]).toMatchObject({
    request: 'Complete the report',
    originalRequest: 'Complete the report',
    outcome: 'clarify',
    calls: [],
    status: 'clarification'
  });
  expect(trace[1]).toMatchObject({
    request: 'submit report',
    originalRequest: 'Complete the report',
    outcome: 'act',
    calls: [{ name: 'complete_task', arguments: { task: 'submit report' } }],
    status: 'executed'
  });
});
