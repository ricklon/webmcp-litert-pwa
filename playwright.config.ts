import { defineConfig, devices } from '@playwright/test';

const nativeWebMcp = process.env.RUN_NATIVE_WEBMCP === '1';

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
        launchOptions: nativeWebMcp ? {
          args: ['--enable-features=WebMCP', '--enable-blink-features=WebMCPTesting']
        } : undefined
      }
    }
  ],
  webServer: {
    command: 'npm run dev -- --host 127.0.0.1 --port 4175',
    url: 'http://127.0.0.1:4175',
    reuseExistingServer: !process.env.CI
  }
});
