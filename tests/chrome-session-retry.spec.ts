import { expect, test } from '@playwright/test';

test('retries a transient Chrome model error in a fresh request session', async ({ page }) => {
  await page.addInitScript(() => {
    let createCount = 0;
    const successfulResponse = JSON.stringify({
      outcome: 'act',
      calls: [{ name: 'add_task', arguments: { title: 'pack USB cables', priority: 'medium' } }],
      message: 'I added the missing packing item.'
    });
    Object.defineProperty(window, 'LanguageModel', {
      configurable: true,
      value: {
        availability: async () => 'available',
        create: async () => {
          createCount += 1;
          (window as typeof window & { chromeCreateCount?: number }).chromeCreateCount = createCount;
          if (createCount === 1) {
            return {
              prompt: async () => successfulResponse,
              clone: async () => ({
                prompt: async () => { throw new Error('An unknown error occurred: kErrorUnknown'); },
                destroy: () => undefined
              }),
              destroy: () => undefined
            };
          }
          return {
            prompt: async () => successfulResponse,
            contextUsage: 400,
            contextWindow: 4096,
            measureContextUsage: async () => 30,
            destroy: () => undefined
          };
        }
      }
    });
  });
  await page.goto('/');
  await page.evaluate(() => localStorage.clear());
  await page.reload();

  await page.getByLabel('What should we get done?').fill('Add the items that need to be packed as well');
  await page.locator('.prompt-box').getByRole('button', { name: 'Plan' }).click();

  const proposal = page.getByRole('region', { name: 'Proposed actions' });
  await expect(proposal.getByLabel('title')).toHaveValue('pack USB cables');
  await expect(page.getByTestId('today-total-count')).toHaveText('0 total · 0 done');
  await expect.poll(() => page.evaluate(() => (window as typeof window & { chromeCreateCount?: number }).chromeCreateCount)).toBe(2);
});

test('rebuilds the Chrome base session when its first clone was destroyed', async ({ page }) => {
  await page.addInitScript(() => {
    let createCount = 0;
    const response = JSON.stringify({
      outcome: 'act',
      calls: [{ name: 'add_task', arguments: { title: 'submit expense report', priority: 'high' } }],
      message: 'I proposed the expense report task.'
    });
    Object.defineProperty(window, 'LanguageModel', {
      configurable: true,
      value: {
        availability: async () => 'available',
        create: async () => {
          createCount += 1;
          (window as typeof window & { chromeCreateCount?: number }).chromeCreateCount = createCount;
          return createCount === 1
            ? {
              prompt: async () => response,
              clone: async () => { throw new Error("Failed to execute 'clone' on 'LanguageModel': The model execution session has been destroyed."); },
              destroy: () => undefined
            }
            : {
              prompt: async () => response,
              clone: async () => ({ prompt: async () => response, destroy: () => undefined }),
              destroy: () => undefined
            };
        }
      }
    });
  });
  await page.goto('/');
  await page.evaluate(() => localStorage.clear());
  await page.reload();

  await page.getByLabel('What should we get done?').fill('Add submit expense report as high priority');
  await page.locator('.prompt-box').getByRole('button', { name: 'Plan' }).click();

  const proposal = page.getByRole('region', { name: 'Proposed actions' });
  await expect(proposal.getByLabel('title')).toHaveValue('submit expense report');
  await expect.poll(() => page.evaluate(() => (window as typeof window & { chromeCreateCount?: number }).chromeCreateCount)).toBe(2);
});

test('a late automatic Chrome load cannot override a newer runtime selection', async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(window, 'LanguageModel', {
      configurable: true,
      value: {
        availability: async () => 'available',
        create: async () => new Promise((resolve) => {
          (window as typeof window & { resolveChromeLoad?: () => void }).resolveChromeLoad = () => resolve({
            prompt: async () => '{"outcome":"answer","calls":[],"message":"ready"}',
            destroy: () => {
              const state = window as typeof window & { destroyedChromeSessions?: number };
              state.destroyedChromeSessions = (state.destroyedChromeSessions ?? 0) + 1;
            }
          });
        })
      }
    });
  });
  await page.goto('/');
  await expect.poll(() => page.evaluate(() => typeof (window as typeof window & { resolveChromeLoad?: unknown }).resolveChromeLoad)).toBe('function');

  await page.getByRole('button', { name: /Demo rules/ }).click();
  await page.evaluate(() => (window as typeof window & { resolveChromeLoad?: () => void }).resolveChromeLoad?.());

  await expect(page.locator('.state')).toHaveText('demo');
  await expect(page.locator('.runtime-note')).toContainText('Demo agent active');
  await expect.poll(() => page.evaluate(() => (window as typeof window & { destroyedChromeSessions?: number }).destroyedChromeSessions)).toBe(1);
});

test('runtime cards are locked while an explicit model load is in progress', async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(window, 'LanguageModel', {
      configurable: true,
      value: {
        availability: async () => 'downloadable',
        create: async () => new Promise((resolve) => {
          (window as typeof window & { resolveChromeLoad?: () => void }).resolveChromeLoad = () => resolve({
            prompt: async () => '{"outcome":"answer","calls":[],"message":"ready"}',
            destroy: () => undefined
          });
        })
      }
    });
  });
  await page.goto('/');
  const chrome = page.getByRole('button', { name: /Chrome built-in/ });
  await expect(chrome).toBeEnabled();
  await chrome.click();

  await expect(page.getByRole('button', { name: /LiteRT-LM recommended/ })).toBeDisabled();
  await expect(page.getByRole('button', { name: /LiteRT-LM lighter/ })).toBeDisabled();
  await expect(page.getByRole('button', { name: /Bonsai custom/ })).toBeDisabled();
  await expect(page.getByRole('button', { name: /Demo rules/ })).toBeDisabled();

  await page.evaluate(() => (window as typeof window & { resolveChromeLoad?: () => void }).resolveChromeLoad?.());
  await expect(page.locator('.state')).toHaveText('chrome');
});
