import { chromium, expect, test, type BrowserContext, type Page } from '@playwright/test';
import { mkdir, writeFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { resolve, sep } from 'node:path';
import { classifyBenchmarkScenario, countCorrectOutcomeSteps, countExactDecisionSteps, countExactToolSelectionSteps, countExactToolSequenceSteps, countSafeToolSteps, type BenchmarkCaseContract } from '../src/benchmark';
import type { PlannerTraceEntry } from '../src/types';

test.skip(process.env.RUN_MODEL_BENCHMARK !== '1', 'Opt-in benchmark downloads and runs the selected real models.');

type ModelName = 'chrome' | 'litert' | 'bonsai';
const playwrightDisabledFeatures = `--disable-features=${[
  'AvoidUnnecessaryBeforeUnloadCheckSync', 'BoundaryEventDispatchTracksNodeRemoval', 'DestroyProfileOnBrowserClose',
  'DialMediaRouteProvider', 'GlobalMediaControls', 'HttpsUpgrades', 'LensOverlay', 'MediaRouter', 'PaintHolding',
  'ThirdPartyStoragePartitioning', 'BlockOriginHeaderModificationOnRedirect', 'Translate', 'AutoDeElevate',
  'OptimizationHints', 'msForceBrowserSignIn', 'msEdgeUpdateLaunchServicesPreferredVersion'
].join(',')}`;
const benchmarkCases: BenchmarkCaseContract[] = [
  {
    id: 'daily-plan',
    name: 'Build a daily plan',
    steps: [
      { status: 'executed', calls: ['add_task'] },
      { status: 'executed', calls: ['add_task'] },
      { status: 'executed', calls: ['list_tasks'] }
    ]
  },
  { id: 'typo-completion', name: 'Recover from a typo', steps: [{ status: 'executed', calls: ['complete_task'] }] },
  { id: 'safe-completion', name: 'Complete without collateral damage', steps: [{ status: 'executed', calls: ['complete_task'] }] },
  {
    id: 'ambiguous-completion',
    name: 'Clarify an ambiguous completion',
    steps: [
      { status: 'clarification', calls: [] },
      { status: 'executed', calls: ['complete_task'] }
    ]
  },
  {
    id: 'record-finished-work',
    name: 'Record newly finished work',
    steps: [{ status: 'executed', calls: ['add_task', 'complete_task'] }]
  },
  {
    id: 'event-trip-plan',
    name: 'Plan tomorrow’s event trip',
    steps: [{ status: 'executed', calls: Array.from({ length: 6 }, () => 'add_task') }]
  },
  { id: 'clear-finished', name: 'Clear finished work', steps: [{ status: 'executed', calls: ['clear_completed'] }] },
  { id: 'unsupported-capability', name: 'Decline an unsupported capability', steps: [{ status: 'answered', calls: [] }] },
  { id: 'missing-task', name: 'Do not guess a missing task', steps: [{ status: 'answered', calls: [] }] },
  { id: 'underspecified-completion', name: 'Clarify an underspecified mutation', steps: [{ status: 'clarification', calls: [] }] },
  { id: 'priority-mapping', name: 'Map an unsupported priority safely', steps: [{ status: 'executed', calls: ['add_task'] }] },
  { id: 'task-data-injection', name: 'Treat task text as inert data', steps: [{ status: 'executed', calls: ['list_tasks'] }] }
];

function mean(values: number[]) {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
}

function percentile(values: number[], fraction: number) {
  if (!values.length) return null;
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.min(sorted.length - 1, Math.ceil(sorted.length * fraction) - 1)];
}

async function readTrace(page: Page): Promise<PlannerTraceEntry[]> {
  const raw = await page.getByTestId('scenario-trace').getAttribute('data-trace');
  return raw ? JSON.parse(raw) as PlannerTraceEntry[] : [];
}

function rawModelTrace(trace: PlannerTraceEntry[]) {
  return trace.map((entry) => {
    if (!entry.modelOutcome) return entry;
    const status = entry.modelOutcome === 'act' ? 'executed'
      : entry.modelOutcome === 'clarify' ? 'clarification'
        : 'answered';
    return { ...entry, outcome: entry.modelOutcome, calls: entry.modelCalls ?? [], status } satisfies PlannerTraceEntry;
  });
}

async function runtimeSnapshot(page: Page) {
  return page.evaluate(() => {
    const memory = (performance as Performance & { memory?: { usedJSHeapSize: number; totalJSHeapSize: number; jsHeapSizeLimit: number } }).memory;
    return {
      deviceMemoryGiB: (navigator as Navigator & { deviceMemory?: number }).deviceMemory ?? null,
      jsHeap: memory ? {
        usedBytes: memory.usedJSHeapSize,
        totalBytes: memory.totalJSHeapSize,
        limitBytes: memory.jsHeapSizeLimit
      } : null
    };
  });
}

function dedicatedChromeProfilePath() {
  const profile = resolve(process.env.CHROME_AI_PROFILE ?? '.playwright/chrome-ai-profile');
  const normalChromeData = resolve(homedir(), 'Library/Application Support/Google/Chrome');
  if (profile === normalChromeData || profile.startsWith(`${normalChromeData}${sep}`)) {
    throw new Error('Refusing to automate a normal Chrome profile. Set CHROME_AI_PROFILE to a dedicated directory.');
  }
  return profile;
}

async function chromePromptPreflight(page: Page) {
  return page.evaluate(async () => {
    const languageModel = (globalThis as typeof globalThis & {
      LanguageModel?: { availability: (options?: unknown) => Promise<string> }
    }).LanguageModel;
    if (!languageModel) return {
      apiPresent: false,
      availability: 'unavailable',
      userActivation: navigator.userActivation.isActive,
      error: null
    };
    try {
      const availability = await languageModel.availability({
        expectedInputs: [{ type: 'text', languages: ['en'] }],
        expectedOutputs: [{ type: 'text', languages: ['en'] }]
      });
      return { apiPresent: true, availability, userActivation: navigator.userActivation.isActive, error: null };
    } catch (error) {
      return {
        apiPresent: true,
        availability: 'error',
        userActivation: navigator.userActivation.isActive,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  });
}

async function activateModel(page: Page, model: ModelName) {
  const liteRtVariant = process.env.BENCHMARK_VARIANT === 'litert-e4b' ? 'E4B' : 'E2B';
  const labels: Record<ModelName, RegExp> = {
    chrome: /Chrome built-in/,
    litert: new RegExp(`LiteRT-LM .*Gemma 4 ${liteRtVariant}`),
    bonsai: /Bonsai custom.*27B/
  };
  if (await page.locator('.state').textContent() === model) return { activationMs: null, activationKind: 'preloaded' as const };
  const button = page.getByRole('button', { name: labels[model] });
  if (model === 'chrome') {
    await expect(page.locator('.runtime-note')).not.toContainText('Checking', { timeout: 30_000 });
    if (await button.isDisabled()) return null;
  }
  await expect(button, `${model} must be available for this benchmark`).toBeEnabled();
  const startedAt = performance.now();
  await button.click();
  const deadline = Date.now() + 25 * 60 * 1000;
  let previousNote = '';
  while (Date.now() < deadline && await page.locator('.state').textContent() !== model) {
    const note = await page.locator('.runtime-note').textContent() ?? '';
    if (note !== previousNote) {
      console.log(`[${model}] ${note}`);
      previousNote = note;
    }
    const feedback = await page.locator('.run-feedback').textContent() ?? '';
    if (/could not start|failed/i.test(feedback)) throw new Error(feedback);
    await page.waitForTimeout(2_000);
  }
  await expect(page.locator('.state')).toHaveText(model);
  return { activationMs: performance.now() - startedAt, activationKind: 'measured' as const };
}

test('benchmark v2 compares decision quality, tools, clarification, and latency', async ({ page, browser }, testInfo) => {
  test.setTimeout(90 * 60 * 1000);
  let persistentContext: BrowserContext | null = null;
  const useDedicatedProfile = process.env.USE_CHROME_AI_PROFILE === '1';
  if (useDedicatedProfile) {
    const profile = dedicatedChromeProfilePath();
    await mkdir(profile, { recursive: true });
    persistentContext = await chromium.launchPersistentContext(profile, {
      channel: 'chrome',
      headless: false,
      ignoreDefaultArgs: ['--disable-background-networking', '--disable-component-update', playwrightDisabledFeatures]
    });
    page = persistentContext.pages()[0] ?? await persistentContext.newPage();
    browser = persistentContext.browser() ?? browser;
  }
  try {
  const requested = (process.env.BENCHMARK_MODELS ?? 'chrome,litert,bonsai')
    .split(',')
    .map((value) => value.trim())
    .filter((value): value is ModelName => ['chrome', 'litert', 'bonsai'].includes(value));
  const requestedCases = new Set((process.env.BENCHMARK_SCENARIOS ?? benchmarkCases.map(({ id }) => id).join(','))
    .split(',').map((value) => value.trim()).filter(Boolean));
  const selectedCases = benchmarkCases.filter(({ id }) => requestedCases.has(id));
  const runs = Math.max(1, Number.parseInt(process.env.BENCHMARK_RUNS ?? '3', 10) || 3);
  expect(selectedCases.length, 'At least one known BENCHMARK_SCENARIOS id is required.').toBeGreaterThan(0);

  const variant = process.env.BENCHMARK_VARIANT ?? 'baseline';
  const query = variant === 'litert-e4b' ? '?litertModel=e4b'
    : variant === 'bonsai-thinking' ? '?bonsaiThink=1'
      : '?litertModel=e2b';
  await page.goto(`/${query}`);
  await page.evaluate(() => localStorage.clear());
  await page.reload();
  page.on('dialog', (dialog) => dialog.accept());
  const chromePreflight = await chromePromptPreflight(page);
  console.log(`[chrome preflight] API ${chromePreflight.apiPresent ? 'present' : 'missing'} · availability ${chromePreflight.availability} · user activation ${chromePreflight.userActivation}`);

  const adapter = await page.evaluate(async () => {
    if (!navigator.gpu) return null;
    const value = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    return value ? {
      architecture: value.info.architecture,
      vendor: value.info.vendor,
      maxBufferSize: value.limits.maxBufferSize,
      maxStorageBufferBindingSize: value.limits.maxStorageBufferBindingSize
    } : null;
  });
  if (requested.some((model) => model !== 'chrome')) expect(adapter, 'External models require a hardware WebGPU adapter.').not.toBeNull();

  const models = [];
  for (const model of requested) {
    const activation = await activateModel(page, model);
    if (!activation) {
      const detail = await page.locator('.runtime-note').textContent() ?? 'Chrome built-in model is unavailable in this browser profile.';
      models.push({ model, availability: 'unavailable' as const, detail });
      console.log(`[${model}] unavailable · ${detail}`);
      continue;
    }
    const memoryAfterActivation = await runtimeSnapshot(page);
    const scenarios = [];
    for (let run = 1; run <= runs; run += 1) {
      for (const benchmarkCase of selectedCases) {
        await page.getByRole('button', { name: new RegExp(`^${benchmarkCase.name}`) }).click();
        await page.getByRole('button', { name: 'Load scenario' }).click();
        const startedAt = performance.now();
        await page.getByRole('button', { name: 'Run workflow' }).click();
        await expect(page.locator('.scenario-state')).toHaveText(/^(passed|failed|stale)$/, { timeout: 10 * 60 * 1000 });
        const workflowMs = performance.now() - startedAt;
        const status = await page.locator('.scenario-state').textContent() as 'passed' | 'failed' | 'stale';
        const detail = await page.locator('.scenario-result').textContent() ?? '';
        const trace = await readTrace(page);
        const categories = classifyBenchmarkScenario(status, trace, benchmarkCase);
        const strictPass = status === 'passed' && categories.length === 0;
        scenarios.push({ run, scenarioId: benchmarkCase.id, scenario: benchmarkCase.name, status, strictPass, detail, workflowMs, categories, expectedSteps: benchmarkCase.steps, trace });
        console.log(`[${model}] run ${run} · ${benchmarkCase.name}: ${strictPass ? 'strict pass' : categories.join(', ')} · ${(workflowMs / 1000).toFixed(2)} s`);
      }
    }

    const plannerLatencies = scenarios.flatMap((scenario) => scenario.trace.map((entry) => entry.metrics?.elapsedMs).filter((value): value is number => value !== undefined));
    const workflowLatencies = scenarios.map((scenario) => scenario.workflowMs);
    const expectedSteps = scenarios.flatMap((scenario) => scenario.expectedSteps);
    const actualSteps = scenarios.flatMap((scenario) => scenario.trace);
    const rawActualSteps = scenarios.flatMap((scenario) => rawModelTrace(scenario.trace));
    const exactToolSteps = scenarios.reduce((total, scenario) => total + countExactToolSequenceSteps(scenario.trace, scenario.expectedSteps), 0);
    const exactToolSelectionSteps = scenarios.reduce((total, scenario) => total + countExactToolSelectionSteps(scenario.trace, scenario.expectedSteps), 0);
    const correctOutcomeSteps = scenarios.reduce((total, scenario) => total + countCorrectOutcomeSteps(scenario.trace, scenario.expectedSteps), 0);
    const safeToolSteps = scenarios.reduce((total, scenario) => total + countSafeToolSteps(scenario.trace, scenario.expectedSteps), 0);
    const exactDecisionSteps = scenarios.reduce((total, scenario) => total + countExactDecisionSteps(scenario.trace, scenario.expectedSteps), 0);
    const rawExactToolSteps = scenarios.reduce((total, scenario) => total + countExactToolSequenceSteps(rawModelTrace(scenario.trace), scenario.expectedSteps), 0);
    const rawToolSelectionSteps = scenarios.reduce((total, scenario) => total + countExactToolSelectionSteps(rawModelTrace(scenario.trace), scenario.expectedSteps), 0);
    const rawOutcomeSteps = scenarios.reduce((total, scenario) => total + countCorrectOutcomeSteps(rawModelTrace(scenario.trace), scenario.expectedSteps), 0);
    const rawSafeToolSteps = scenarios.reduce((total, scenario) => total + countSafeToolSteps(rawModelTrace(scenario.trace), scenario.expectedSteps), 0);
    const rawExactDecisionSteps = scenarios.reduce((total, scenario) => total + countExactDecisionSteps(rawModelTrace(scenario.trace), scenario.expectedSteps), 0);
    const diagnosedOutputs = actualSteps.filter((step) => step.outputDiagnostics);
    const validInitially = diagnosedOutputs.filter((step) => step.outputDiagnostics?.validInitially).length;
    const recoveredOutputs = diagnosedOutputs.filter((step) => step.outputDiagnostics?.recovered).length;
    const retriedOutputs = diagnosedOutputs.filter((step) => step.outputDiagnostics?.retried).length;
    const expectedClarifications = expectedSteps.filter((step) => step.status === 'clarification').length;
    const askedClarifications = actualSteps.filter((step) => step.status === 'clarification').length;
    const correctClarifications = scenarios.reduce((total, scenario) => total + scenario.expectedSteps.filter((step, index) =>
      step.status === 'clarification' && scenario.trace[index]?.status === 'clarification').length, 0);
    const rawAskedClarifications = rawActualSteps.filter((step) => step.status === 'clarification').length;
    const rawCorrectClarifications = scenarios.reduce((total, scenario) => {
      const rawTrace = rawModelTrace(scenario.trace);
      return total + scenario.expectedSteps.filter((step, index) =>
        step.status === 'clarification' && rawTrace[index]?.status === 'clarification').length;
    }, 0);
    const guardrailInterventions = actualSteps.flatMap((step) => step.guardrailInterventions ?? []);
    const categoryCounts = Object.fromEntries([...new Set(scenarios.flatMap((scenario) => scenario.categories))]
      .map((category) => [category, scenarios.filter((scenario) => scenario.categories.includes(category)).length]));
    const strictPassed = scenarios.filter((scenario) => scenario.strictPass).length;
    models.push({
      model,
      availability: 'available' as const,
      ...activation,
      memoryAfterActivation,
      memoryAfterEvaluation: await runtimeSnapshot(page),
      quality: {
        strictPassed,
        attempted: scenarios.length,
        strictPassRate: strictPassed / scenarios.length,
        finalStatePassRate: scenarios.filter((scenario) => scenario.status === 'passed').length / scenarios.length,
        exactToolSequenceSteps: exactToolSteps,
        expectedSteps: expectedSteps.length,
        exactToolSequenceRate: exactToolSteps / expectedSteps.length,
        exactToolSelectionSteps,
        exactToolSelectionRate: exactToolSelectionSteps / expectedSteps.length,
        correctOutcomeSteps,
        correctOutcomeRate: correctOutcomeSteps / expectedSteps.length,
        safeToolSteps,
        safeToolRate: safeToolSteps / expectedSteps.length,
        exactDecisionSteps,
        exactDecisionRate: exactDecisionSteps / expectedSteps.length,
        clarification: { expected: expectedClarifications, asked: askedClarifications, correct: correctClarifications },
        rawModel: {
          exactToolSequenceSteps: rawExactToolSteps,
          exactToolSequenceRate: rawExactToolSteps / expectedSteps.length,
          exactToolSelectionSteps: rawToolSelectionSteps,
          exactToolSelectionRate: rawToolSelectionSteps / expectedSteps.length,
          correctOutcomeSteps: rawOutcomeSteps,
          correctOutcomeRate: rawOutcomeSteps / expectedSteps.length,
          safeToolSteps: rawSafeToolSteps,
          safeToolRate: rawSafeToolSteps / expectedSteps.length,
          exactDecisionSteps: rawExactDecisionSteps,
          exactDecisionRate: rawExactDecisionSteps / expectedSteps.length,
          clarification: { expected: expectedClarifications, asked: rawAskedClarifications, correct: rawCorrectClarifications }
        },
        guardrails: {
          interventions: guardrailInterventions.length,
          byType: Object.fromEntries([...new Set(guardrailInterventions)].map((name) => [name, guardrailInterventions.filter((item) => item === name).length]))
        },
        structuredOutput: {
          diagnosed: diagnosedOutputs.length,
          validInitially,
          rawValidityRate: diagnosedOutputs.length ? validInitially / diagnosedOutputs.length : null,
          recovered: recoveredOutputs,
          retried: retriedOutputs
        },
        failureCategories: categoryCounts
      },
      performance: {
        workflowMs: { mean: mean(workflowLatencies), median: percentile(workflowLatencies, 0.5), p95: percentile(workflowLatencies, 0.95) },
        plannerMs: { mean: mean(plannerLatencies), median: percentile(plannerLatencies, 0.5), p95: percentile(plannerLatencies, 0.95) },
        firstWorkflowMs: workflowLatencies[0] ?? null,
        subsequentWorkflowMedianMs: percentile(workflowLatencies.slice(1), 0.5)
      },
      scenarios
    });
  }

  const report = {
    version: 2,
    generatedAt: new Date().toISOString(),
    browser: browser.version(),
    adapter,
    configuration: {
      requestedModels: requested,
      runs,
      scenarios: selectedCases.map(({ id }) => id),
      modelOrder: requested,
      variant,
      chromeProfile: useDedicatedProfile ? 'dedicated-persistent' : 'temporary'
    },
    chromePreflight,
    scope: {
      localPlannerTools: true,
      nativeWebMcp: false,
      autoApproval: true,
      notes: ['Native WebMCP requires a separate external-agent evaluation.', 'JS heap does not include model GPU allocation.']
    },
    models
  };
  const outputDirectory = resolve('benchmark-results');
  const outputPath = resolve(outputDirectory, 'latest.json');
  await mkdir(outputDirectory, { recursive: true });
  await writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`, 'utf8');
  await testInfo.attach('model-comparison-v2', { body: JSON.stringify(report, null, 2), contentType: 'application/json' });
  console.log(`Benchmark v2 report: ${outputPath}`);
  console.table(models.map((result) => ({
    model: result.model,
    availability: result.availability,
    activationSeconds: 'activationMs' in result && result.activationMs !== null ? Number((result.activationMs / 1000).toFixed(2)) : result.availability === 'available' ? 'preloaded' : null,
    strictScore: 'quality' in result ? `${result.quality.strictPassed}/${result.quality.attempted}` : 'n/a',
    exactDecisions: 'quality' in result ? `${Math.round(result.quality.exactDecisionRate * 100)}%` : 'n/a',
    clarification: 'quality' in result ? `${result.quality.clarification.correct}/${result.quality.clarification.expected} correct; ${result.quality.clarification.asked} asked` : 'n/a',
    medianWorkflowSeconds: 'performance' in result && result.performance.workflowMs.median !== null ? Number((result.performance.workflowMs.median / 1000).toFixed(2)) : null,
    p95WorkflowSeconds: 'performance' in result && result.performance.workflowMs.p95 !== null ? Number((result.performance.workflowMs.p95 / 1000).toFixed(2)) : null
  })));
  } finally {
    await persistentContext?.close();
  }
});
