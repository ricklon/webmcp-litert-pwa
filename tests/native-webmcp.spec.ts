import { expect, test } from '@playwright/test';

test.skip(process.env.RUN_NATIVE_WEBMCP !== '1', 'Requires a Chrome build/profile with native WebMCP enabled.');

test('native Chrome discovers the registered task tools', async ({ page }) => {
  await page.goto('/');
  const snapshot = await page.evaluate(async () => {
    if (!document.modelContext) return null;
    const tools = await document.modelContext.getTools();
    const listTasks = tools.find((tool) => tool.name === 'list_tasks');
    if (!listTasks) return { tools, readResult: null };
    const readResult = await document.modelContext.executeTool(listTasks, JSON.stringify({ status: 'open' }));
    return { tools, readResult };
  });
  expect(snapshot, 'document.modelContext must be enabled for native WebMCP conformance').not.toBeNull();
  expect(snapshot!.tools.map((tool) => tool.name).sort()).toEqual([
    'add_task', 'clear_completed', 'complete_task', 'list_tasks'
  ]);
  expect(snapshot!.readResult).not.toBeNull();
  await expect(page.getByRole('region', { name: 'Proposed actions' })).toHaveCount(0);
});
