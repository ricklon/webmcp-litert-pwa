import { expect, test } from '@playwright/test';

test('routes a native WebMCP write through user review before changing tasks', async ({ page }) => {
  await page.addInitScript(() => {
    const registered = new Map<string, { execute: (args: Record<string, unknown>) => Promise<unknown> }>();
    Object.defineProperty(document, 'modelContext', {
      configurable: true,
      value: {
        registerTool: async (tool: { name: string; execute: (args: Record<string, unknown>) => Promise<unknown> }) => {
          registered.set(tool.name, tool);
        },
        getTools: async () => [...registered.keys()].map((name) => ({ name })),
        executeTool: async () => undefined
      }
    });
    (window as typeof window & { callRegisteredTool?: (name: string, args: Record<string, unknown>) => Promise<unknown> }).callRegisteredTool =
      (name, args) => registered.get(name)!.execute(args);
  });
  await page.goto('/');
  await expect(page.locator('.mcp-badge')).toHaveText('native');

  const result = await page.evaluate(() => (window as typeof window & { callRegisteredTool: (name: string, args: Record<string, unknown>) => Promise<unknown> })
    .callRegisteredTool('add_task', { title: 'Pack USB cables', priority: 'medium' }));
  expect(result).toMatchObject({ ok: true, pendingReview: true, tool: 'add_task' });

  await expect(page.getByTestId('today-total-count')).toHaveText('0 total · 0 done');
  const proposal = page.getByRole('region', { name: 'Proposed actions' });
  await expect(proposal.getByLabel('title')).toHaveValue('Pack USB cables');
  await expect(page.locator('.activity-panel')).toContainText('Proposed add_task for review.');

  await proposal.getByRole('button', { name: 'Approve and execute' }).click();
  await expect(page.getByTestId('today-total-count')).toHaveText('1 total · 0 done');
  await expect(page.getByTestId('today-panel').getByText('Pack USB cables', { exact: true })).toBeVisible();
});

test('executes a native WebMCP read immediately without opening review', async ({ page }) => {
  await page.addInitScript(() => {
    const registered = new Map<string, { execute: (args: Record<string, unknown>) => Promise<unknown> }>();
    Object.defineProperty(document, 'modelContext', {
      configurable: true,
      value: {
        registerTool: async (tool: { name: string; execute: (args: Record<string, unknown>) => Promise<unknown> }) => { registered.set(tool.name, tool); },
        getTools: async () => [...registered.keys()].map((name) => ({ name })),
        executeTool: async () => undefined
      }
    });
    (window as typeof window & { callRegisteredTool?: (name: string, args: Record<string, unknown>) => Promise<unknown> }).callRegisteredTool =
      (name, args) => registered.get(name)!.execute(args);
  });
  await page.goto('/');
  await page.evaluate(() => localStorage.setItem('local-tools-lab.tasks.v1', JSON.stringify([
    { id: 'open', title: 'submit report', priority: 'high', completed: false, createdAt: new Date().toISOString() },
    { id: 'done', title: 'buy filters', priority: 'medium', completed: true, createdAt: new Date().toISOString() }
  ])));
  await page.reload();
  await expect(page.locator('.mcp-badge')).toHaveText('native');

  const result = await page.evaluate(() => (window as typeof window & { callRegisteredTool: (name: string, args: Record<string, unknown>) => Promise<unknown> })
    .callRegisteredTool('list_tasks', { status: 'open' }));
  expect(result).toMatchObject({ count: 1, tasks: [{ title: 'submit report', completed: false }] });
  await expect(page.getByRole('region', { name: 'Proposed actions' })).toHaveCount(0);
  await expect(page.getByTestId('today-total-count')).toHaveText('2 total · 1 done');
});

test('preserves completed tasks when a destructive WebMCP proposal is cancelled', async ({ page }) => {
  await page.addInitScript(() => {
    const registered = new Map<string, { execute: (args: Record<string, unknown>) => Promise<unknown> }>();
    Object.defineProperty(document, 'modelContext', {
      configurable: true,
      value: {
        registerTool: async (tool: { name: string; execute: (args: Record<string, unknown>) => Promise<unknown> }) => { registered.set(tool.name, tool); },
        getTools: async () => [...registered.keys()].map((name) => ({ name })),
        executeTool: async () => undefined
      }
    });
    (window as typeof window & { callRegisteredTool?: (name: string, args: Record<string, unknown>) => Promise<unknown> }).callRegisteredTool =
      (name, args) => registered.get(name)!.execute(args);
  });
  await page.goto('/');
  await page.evaluate(() => localStorage.setItem('local-tools-lab.tasks.v1', JSON.stringify([
    { id: 'done', title: 'submit report', priority: 'high', completed: true, createdAt: new Date().toISOString() }
  ])));
  await page.reload();
  await expect(page.locator('.mcp-badge')).toHaveText('native');
  await page.evaluate(() => (window as typeof window & { callRegisteredTool: (name: string, args: Record<string, unknown>) => Promise<unknown> })
    .callRegisteredTool('clear_completed', {}));

  await page.getByRole('region', { name: 'Proposed actions' }).getByRole('button', { name: 'Cancel' }).click();
  await expect(page.getByTestId('today-total-count')).toHaveText('1 total · 1 done');
  await expect(page.locator('.run-feedback')).toContainText('Proposal cancelled');
  await expect(page.getByRole('region', { name: 'Proposed actions' })).toHaveCount(0);
});
