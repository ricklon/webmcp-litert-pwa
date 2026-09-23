import { defineConfig, devices } from '@playwright/test';

const nativeWebMcp = process.env.RUN_NATIVE_WEBMCP === '1';
// Chrome on Linux keeps WebGPU behind flags; opt in to benchmark WebGPU models there.
const linuxWebGpu = process.env.ENABLE_LINUX_WEBGPU === '1';
// Chrome honors only the last --enable-features flag, so features are merged.
const enabledFeatures = [...(nativeWebMcp ? ['WebMCP'] : []), ...(linuxWebGpu ? ['Vulkan'] : [])];
const chromeArgs = [
  ...(enabledFeatures.length ? [`--enable-features=${enabledFeatures.join(',')}`] : []),
  ...(nativeWebMcp ? ['--enable-blink-features=WebMCPTesting'] : []),
  ...(linuxWebGpu ? ['--enable-unsafe-webgpu', '--use-angle=vulkan'] : [])
];

export default defineConfig({
  testDir: './tests',
  fullyParallel: true,
  retries: process.env.CI ? 2 : 0,
  reporter: 'list',
  use: {
    baseURL: 'http://127.0.0.1:4175',
    trace: 'on-first-retry'
  },
  projects: [
    {
      name: 'chrome',
      use: {
        ...devices['Desktop Chrome'],
        channel: 'chrome',
        launchOptions: chromeArgs.length ? { args: chromeArgs } : undefined
      }
    }
  ],
  webServer: {
    command: 'npm run dev -- --host 127.0.0.1 --port 4175',
    url: 'http://127.0.0.1:4175',
    reuseExistingServer: !process.env.CI
  }
});
