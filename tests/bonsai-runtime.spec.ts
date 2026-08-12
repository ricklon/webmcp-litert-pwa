import { expect, test } from '@playwright/test';

test('offers Bonsai 27B as an explicitly confirmed WebGPU runtime', async ({ page }) => {
  await page.goto('/');

  const bonsai = page.getByRole('button', { name: /Bonsai custom.*27B.*1-bit GGUF.*WebGPU/ });
  await expect(bonsai).toBeVisible();

  page.once('dialog', async (dialog) => {
    expect(dialog.message()).toContain('~3.8 GB Bonsai 27B');
    expect(dialog.message()).toContain('16 GB');
    await dialog.dismiss();
  });
  await bonsai.click();

  await expect(bonsai).toHaveAttribute('aria-pressed', 'false');
  await expect(page.getByText(/Bonsai is ~3.8 GB/)).toBeVisible();
});
