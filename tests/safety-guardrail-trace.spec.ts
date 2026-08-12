import { expect, test } from '@playwright/test';
import type { PlannerTraceEntry } from '../src/types';

test('records an unsafe raw model completion separately from the guarded decision', async ({ page }) => {
  await page.addInitScript(() => {
    const response = JSON.stringify({
      outcome: 'act',
      calls: [
        { name: 'complete_task', arguments: { task: 'book dentist appointment' } },
        { name: 'add_task', arguments: { title: 'renew passport' } }
      ],
      message: 'I completed one task and added another.'
    });
    const requestSession = () => ({ prompt: async () => response, destroy: () => undefined });
    Object.defineProperty(window, 'LanguageModel', {
      configurable: true,
      value: {
        availability: async () => 'available',
        create: async () => ({
          ...requestSession(),
          clone: async () => requestSession()
        })
      }
    });
  });
  await page.goto('/');
  await page.evaluate(() => localStorage.clear());
  await page.reload();

  await page.getByRole('button', { name: /Do not guess a missing task/ }).click();
  await page.getByRole('button', { name: 'Load scenario' }).click();
  await page.getByRole('button', { name: 'Run workflow' }).click();

  await expect(page.locator('.scenario-state')).toHaveText('passed');
  await expect(page.getByTestId('today-total-count')).toHaveText('1 total · 0 done');
  const raw = await page.getByTestId('scenario-trace').getAttribute('data-trace');
  const trace = JSON.parse(raw ?? '[]') as PlannerTraceEntry[];
  expect(trace[0]).toMatchObject({
    modelOutcome: 'act',
    modelCalls: [
      { name: 'complete_task' },
      { name: 'add_task' }
    ],
    outcome: 'answer',
    calls: [],
    status: 'answered',
    guardrailInterventions: ['missing-completion-target']
  });
});
