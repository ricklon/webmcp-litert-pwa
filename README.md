# Local Tools Lab

Live demo: <https://ricklon.github.io/webmcp-litert-pwa/>

A Vite/React PWA that tests how Chrome's built-in model, custom WebGPU models, and WebMCP page tools can work together. The useful demo is a private task planner:

1. Natural language is planned by Chrome's built-in Gemini Nano when available, with Gemma 4 E4B recommended through LiteRT-LM, lighter/faster E2B, and 1-bit Bonsai 27B through `bitgpu` as optional custom local models.
2. Read-only plans can run immediately; write plans become editable proposals and call tools only after user approval.
3. Task data, conversation memory, proposals, and the audit log remain in the browser.
4. A deterministic demo planner remains the universal fallback when neither model API is available.

Planner decisions distinguish actions, direct answers, and clarification questions. When one completion request matches several tasks, the app preserves the original request, asks the user to choose, and executes no matching task until the follow-up resolves the ambiguity.

Proposals preserve the original request and the agent's note. Users can edit arguments directly, add or remove proposed tasks, refine the whole proposal conversationally, approve it, or cancel without changing app data. Executed plans remain available for an explicit follow-up refinement. Managed scenarios use a deliberate auto-approval path so repeatable workflow tests remain automated.

Conversation history and review state are stored locally in IndexedDB. A conversation can be continued after reload, an unfinished proposal can be resumed, and a new conversation can be started without clearing the shared task state. The model receives only a bounded chronological projection of recent messages; performance telemetry is retained in the audit log but excluded from inference context. The task list remains authoritative rather than being reconstructed from model memory.

Native WebMCP reads execute immediately. Native WebMCP writes enter the same editable proposal card as local-model writes and return a `pendingReview` result; they cannot mutate tasks until the person approves the proposal in the page.

Every model request receives a fresh runtime clock containing local date/time, the browser's IANA timezone, and a UTC timestamp. This keeps relative language such as “today” and “tomorrow” grounded even when a model session remains open for a long time. Tasks do not yet have a structured due-date field, so relevant timing is preserved in titles for now.

Planning also uses a clean per-request model context: Chrome clones its system-prefaced base session (or creates a fresh compatible session as a fallback), LiteRT-LM creates a fresh conversation from the loaded engine, and Bonsai resets its chat cache before each plan. The app supplies proposal and refinement history explicitly, avoiding duplicated hidden conversation context. Chrome's transient `kErrorUnknown` is retried once in a fresh session; a repeated failure leaves the proposal and user refinement recoverable.

Bonsai support uses the public `Bonsai-27B-Q1_0.gguf` through `bitgpu`, with a 4,096-token q8 KV cache and schema-constrained JSON planning. Loading is always opt-in because the weights are approximately 3.8 GB; a GPU with at least 16 GB of available memory is recommended. The current adapter uses the text trunk only and does not expose Bonsai's vision path.

The result panel reports end-to-end planner time for every runtime. Chrome sessions also report their live context usage and context-window limit, plus clearly labeled estimated output throughput when Chrome's context measurement API is available.

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

Run the realistic managed scenario against the actual installed Chrome model with one worker:

```bash
npm run test:model
```

Run the opt-in Bonsai 27B WebGPU evaluation in headed Chrome (downloads approximately 3.8 GB):

```bash
npm run test:bonsai
```

Run the opt-in Gemma 4 E2B LiteRT-LM WebGPU evaluation:

```bash
npm run test:litert
```

Compare Chrome's built-in LLM and both external models on the same seven managed scenarios:

```bash
npm run benchmark:models
```

Benchmark v2 uses the ignored `.playwright/chrome-ai-profile` persistent automation profile, compares `chrome,litert,bonsai` with three runs of twelve scenarios, and writes `benchmark-results/latest.json`. It scores outcome choice, tool selection, exact ordering, unexpected calls, clarification, raw structured-output validity, recovered output, and final state separately. It also reports activation plus median/p95 workflow and planner latency. If the dedicated profile cannot access Chrome's Prompt API/model, the report records that entry as unavailable and continues with the external models. Override the model set, scenario ids, repeat count, or dedicated profile location when needed:

```bash
BENCHMARK_MODELS=litert,bonsai BENCHMARK_RUNS=3 npm run benchmark:models
CHROME_AI_PROFILE=/path/to/dedicated-profile npm run benchmark:chrome
```

For a shorter diagnostic run:

```bash
BENCHMARK_MODELS=litert BENCHMARK_RUNS=1 BENCHMARK_SCENARIOS=typo-completion,ambiguous-completion npm run benchmark:models
```

Focused model variants are explicit benchmark modes:

```bash
npm run benchmark:litert-e4b
npm run benchmark:bonsai-thinking
```

The app applies deterministic safety checks to explicit completion and
unsupported communication requests after planning. Benchmark traces retain the
raw model calls and the guarded calls separately, including the intervention
name, so runtime safety improvements do not hide model errors.

See [BENCHMARK.md](./BENCHMARK.md) for the remembered v1 results, the v2 measurement contract, interpretation rules, and remaining gaps. V2 explicitly measures the app's local planner/tool path; native WebMCP needs a separate real external-agent benchmark.

### Set up Chrome's built-in model for Playwright

Playwright must not automate a normal browsing profile. Prepare the dedicated benchmark profile once:

```bash
npm run setup:chrome-ai-profile
```

In the opened Chrome window, enable `chrome://flags/#optimization-guide-on-device-model` and `chrome://flags/#prompt-api-for-gemini-nano`, then relaunch that Chrome profile. Inspect `chrome://on-device-internals` for download or eligibility errors. Run the setup command again if the relaunch closed Playwright's setup process, then test only the built-in model with:

```bash
npm run benchmark:chrome
```

The benchmark report records whether `LanguageModel` exists, its exact `availability()` result, user activation, Chrome version, and whether the run used the dedicated persistent profile. If availability is `downloadable`, the benchmark's real button click supplies user activation and starts Chrome's managed download. Close the setup window before benchmarking because Chrome cannot open the same profile twice.

The dedicated-profile launcher omits Playwright's usual browser-isolation defaults because they disable background networking, Chrome component updates, and `OptimizationHints`, all of which participate in Gemini Nano eligibility and initial model registration/download. It retains only first-run suppression and disabled sync. No normal Chrome profile is modified. Only the on-device-model and Prompt API flags are required for this text benchmark; multimodal input, sampling mode, Writer API, and Chrome benchmarking flags are not required.

This is an evaluation rather than a deterministic CI test. Chrome's result requires the Prompt API in Playwright's temporary profile; otherwise that entry is reported as unavailable. Individual scenarios can still fail when a model interprets a request incompletely. The regular smoke suite uses controlled model responses to verify application behavior repeatably. The same scenarios can be run from **Managed scenarios** in a personal Chrome profile that has Gemini Nano available.

The opt-in evaluation currently stresses two deliberately human requests: a typo-filled event packing/travel story that should become six distinct tasks, and “I packed a sldering iron as well,” which should infer an ordered add-then-complete transaction. Demo rules intentionally remain a limited baseline rather than accumulating phrase-specific grammar for these cases.

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

- **Real:** Chrome `LanguageModel`, structured output, LiteRT-LM `Engine.create()`, the web-compatible `.litertlm` Gemma model, `bitgpu` with the 1-bit Bonsai 27B GGUF, WebGPU inference, WebMCP `document.modelContext.registerTool()`, local persistence, and the PWA service worker.
- **Fallback:** Demo mode uses small deterministic rules to produce the same structured calls. It does not claim to be model inference.
- **Bridge:** LiteRT-LM currently provides text generation in its Web API. This prototype asks for the planning envelope, preserves the raw response, applies narrowly bounded syntax and outcome-alias recovery, validates it against the allow-listed tool contracts, and retries validation once before failing closed. Raw and recovered validity are reported separately. WebMCP exposes the same contracts to external browser agents, with write calls wrapped by the page's approval gateway.

## Constraints

- LiteRT-LM's browser API is early preview and currently supports only designated web `.litertlm` models.
- Custom-model downloads are large; the app never starts them without an explicit click. Bonsai requires WebGPU and may fail or run slowly on memory-constrained GPUs.
- WebMCP is an origin-trial/proposed API and needs a compatible Chrome version or local flag.
- The production host must send `Origin-Agent-Cluster: ?1`; the Vite dev and preview servers are configured to do so.
- The service worker caches the app shell, not the multi-gigabyte model.

## Finding compatible models

| Runtime | Where to browse | What this web app can load |
| --- | --- | --- |
| LiteRT-LM Web | [Hugging Face models tagged `litert-lm`](https://huggingface.co/models?other=litert-lm) and the [official JavaScript support list](https://github.com/google-ai-edge/LiteRT-LM/tree/main/js/packages/core) | The current JavaScript API explicitly supports the Gemma 4 E2B and E4B `*-web.litertlm` files. A generic `.litertlm` listing is not enough; confirm it appears in the JS support list. |
| `bitgpu` WebGPU | [PrismML Bonsai 1-bit collection](https://huggingface.co/collections/prism-ml/bonsai) and [Bonsai 27B collection](https://huggingface.co/collections/prism-ml/bonsai-27b) | PrismML-style 1-bit `Q1_0` GGUF models in bitgpu's supported Qwen3 envelope, plus its explicit Qwen3.5 hybrid Bonsai-27B path. The 27B tokenizer files come from the matching unpacked repository. |
| Chrome built-in | Chrome's Prompt API | Browser-managed only; users cannot substitute a Hugging Face model. |

Do not advertise arbitrary GGUF, MLX, safetensors, ONNX, or every model carrying the broad `litert-lm` tag as drop-in compatible. Each custom model still needs an entry in the app describing its model URL, tokenizer source when required, context limit, expected download size, and runtime-specific settings.

## Sources

- [LiteRT-LM Web API](https://developers.google.com/edge/litert-lm/js)
- [Bonsai 27B WebGPU Space](https://huggingface.co/spaces/webml-community/bonsai-webgpu-kernels)
- [`bitgpu` WebGPU runtime](https://github.com/stfurkan/bitgpu)
- [Bonsai model repository](https://huggingface.co/prism-ml/Bonsai-27B-gguf)
- [WebMCP overview](https://developer.chrome.com/docs/ai/webmcp)
- [WebMCP imperative API](https://developer.chrome.com/docs/ai/webmcp/imperative-api)
- [WebMCP draft](https://github.com/webmachinelearning/webmcp)
