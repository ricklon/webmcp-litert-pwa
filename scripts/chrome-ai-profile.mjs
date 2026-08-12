import { mkdir } from 'node:fs/promises';
import { homedir } from 'node:os';
import { resolve, sep } from 'node:path';
import { chromium } from '@playwright/test';

const normalChromeData = resolve(homedir(), 'Library/Application Support/Google/Chrome');
const profile = resolve(process.env.CHROME_AI_PROFILE ?? '.playwright/chrome-ai-profile');
const playwrightDisabledFeatures = `--disable-features=${[
  'AvoidUnnecessaryBeforeUnloadCheckSync', 'BoundaryEventDispatchTracksNodeRemoval', 'DestroyProfileOnBrowserClose',
  'DialMediaRouteProvider', 'GlobalMediaControls', 'HttpsUpgrades', 'LensOverlay', 'MediaRouter', 'PaintHolding',
  'ThirdPartyStoragePartitioning', 'BlockOriginHeaderModificationOnRedirect', 'Translate', 'AutoDeElevate',
  'OptimizationHints', 'msForceBrowserSignIn', 'msEdgeUpdateLaunchServicesPreferredVersion'
].join(',')}`;

if (profile === normalChromeData || profile.startsWith(`${normalChromeData}${sep}`)) {
  throw new Error('Refusing to automate a normal Chrome profile. Choose a dedicated CHROME_AI_PROFILE directory.');
}

await mkdir(profile, { recursive: true });
const context = await chromium.launchPersistentContext(profile, {
  channel: 'chrome',
  headless: false,
  // Preserve Playwright's dedicated user-data directory and control pipe while
  // omitting only the defaults that block Chrome's managed model services.
  ignoreDefaultArgs: ['--disable-background-networking', '--disable-component-update', playwrightDisabledFeatures]
});

const pages = context.pages();
const flags = pages[0] ?? await context.newPage();
await flags.goto('chrome://flags/#optimization-guide-on-device-model');
const promptFlag = await context.newPage();
await promptFlag.goto('chrome://flags/#prompt-api-for-gemini-nano');
const internals = await context.newPage();
await internals.goto('chrome://on-device-internals');

console.log(`Dedicated Chrome AI profile: ${profile}`);
console.log('1. Enable “Enables optimization guide on device”.');
console.log('2. Enable “Prompt API for Gemini Nano” (or its multilingual option).');
console.log('3. Relaunch Chrome when prompted.');
console.log('4. Run this setup command again to inspect chrome://on-device-internals, or run npm run benchmark:chrome.');
console.log('Close the dedicated Chrome window when setup is complete.');

await new Promise((done) => context.once('close', done));
