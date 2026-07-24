# Local Tools Lab

Live demo: <https://ricklon.github.io/webmcp-litert-pwa/>

A Vite/React PWA that tests how Chrome's built-in model, a custom LiteRT-LM model, and WebMCP page tools can work together. The useful demo is a private task planner:

1. Natural language is planned by Chrome's built-in Gemini Nano when available, with Gemma 4 E2B through LiteRT-LM as an optional custom local model.
2. The plan calls the same typed task functions registered with `document.modelContext`.
3. Task data and the audit log remain in the browser.
4. A deterministic demo planner remains the universal fallback when neither model API is available.

The in-app **Managed scenarios** lab provides repeatable workflows with controlled seed data, scripted user requests, expected final state, and visible pass/fail results. The Playwright suite drives these same controls so manual exploration and automation share one workflow.

The Today panel's **Reset local data** action provides a confirmed recovery path that clears tasks, activity, and scenario progress. It does not remove Chrome-managed AI models.

## Run

```bash
npm install
npm run dev
```

Run unit and browser smoke tests with:

```bash
npm test
npm run test:smoke
```

The public demo is deployed from the committed `docs/` production build on the
`main` branch. Rebuild and verify locally before updating that directory.

Open the local HTTPS/localhost page in a WebGPU-capable Chrome build. To test native WebMCP during local development, enable:

```text
chrome://flags/#enable-webmcp-testing
```

Then relaunch Chrome. The app reports `native` when its four tools register successfully. Chrome's Model Context Tool Inspector extension can discover and invoke them.

Playwright's bundled Chromium is separate from your normal Chrome profile. For an automated native-WebMCP check, launch Chromium with the feature enabled:

```ts
// playwright.config.ts
export default defineConfig({
  use: {
    launchOptions: {
      args: [
        '--enable-features=WebMCP',
        '--enable-blink-features=WebMCPTesting'
      ]
    }
  }
});
```

Keep native WebMCP tests Chromium-only and feature-detect `document.modelContext`, since the browser API and its testing switches are still experimental.

## What is real versus simulated

- **Real:** Chrome `LanguageModel`, structured output, LiteRT-LM `Engine.create()`, the web-compatible `.litertlm` Gemma model, WebGPU inference, WebMCP `document.modelContext.registerTool()`, local persistence, and the PWA service worker.
- **Fallback:** Demo mode uses small deterministic rules to produce the same structured calls. It does not claim to be model inference.
- **Bridge:** LiteRT-LM currently provides text generation in its Web API. This prototype prompts the model for constrained JSON, validates the basic shape, then dispatches only to an allow-listed tool registry. WebMCP exposes those identical functions to external browser agents.

## Constraints

- LiteRT-LM's browser API is early preview and currently supports only designated web `.litertlm` models.
- The model download is large; the app never starts it without an explicit click.
- WebMCP is an origin-trial/proposed API and needs a compatible Chrome version or local flag.
- The production host must send `Origin-Agent-Cluster: ?1`; the Vite dev and preview servers are configured to do so.
- The service worker caches the app shell, not the multi-gigabyte model.

## Sources

- [LiteRT-LM Web API](https://developers.google.com/edge/litert-lm/js)
- [WebMCP overview](https://developer.chrome.com/docs/ai/webmcp)
- [WebMCP imperative API](https://developer.chrome.com/docs/ai/webmcp/imperative-api)
- [WebMCP draft](https://github.com/webmachinelearning/webmcp)
