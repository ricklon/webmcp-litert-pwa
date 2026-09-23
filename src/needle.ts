import { PlannerOutputError } from './agent';
import type { ToolDefinition } from './tools';
import type { AgentPlan, ToolCall } from './types';
import type { NeedleWorkerRequest, NeedleWorkerResponse } from './needle.worker';

const NEEDLE_REVISION = 'b274efcb211a9eef48c9a88da4b43bd569696a39';
export const NEEDLE_MODEL_URL = `https://huggingface.co/Cactus-Compute/needle3/resolve/${NEEDLE_REVISION}/needle3.cact`;
const NEEDLE_MODEL_SHA256 = 'c9d915eca282ed42d1a09b143b592adb4cc6744ffe2d294adf5cfc5548170c38';
const MAX_NEW_TOKENS = 256;

type NeedleOutput = {
  success?: boolean;
  error?: string | null;
  function_calls?: unknown;
  confidence?: number;
  decode_tps?: number;
};

let worker: Worker | null = null;
let nextRequestId = 0;
const pending = new Map<number, { resolve: (value: unknown) => void; reject: (error: Error) => void; onProgress?: (message: string) => void }>();

function rejectAll(error: Error) {
  for (const request of pending.values()) request.reject(error);
  pending.clear();
}

function ensureWorker() {
  if (worker) return worker;
  worker = new Worker(new URL('./needle.worker.ts', import.meta.url), { type: 'module' });
  worker.onmessage = (event: MessageEvent<NeedleWorkerResponse>) => {
    const message = event.data;
    const request = pending.get(message.id);
    if (!request) return;
    if (message.type === 'progress') {
      request.onProgress?.(message.message);
      return;
    }
    pending.delete(message.id);
    if (message.type === 'result') request.resolve(message.value);
    else request.reject(new Error(message.message));
  };
  worker.onerror = (event) => {
    rejectAll(new Error(event.message || 'The Needle 3 worker stopped unexpectedly.'));
    unloadNeedle();
  };
  return worker;
}

type WithoutId<T> = T extends unknown ? Omit<T, 'id'> : never;

function send<T>(request: WithoutId<NeedleWorkerRequest>, onProgress?: (message: string) => void) {
  const target = ensureWorker();
  const id = ++nextRequestId;
  return new Promise<T>((resolve, reject) => {
    pending.set(id, { resolve: resolve as (value: unknown) => void, reject, onProgress });
    target.postMessage({ ...request, id });
  });
}

/** Needle expects OpenAI-style function declarations. */
export function buildNeedleToolCatalog(tools: ToolDefinition[]) {
  return tools.map(({ name, description, inputSchema }) => ({ name, description, parameters: inputSchema }));
}

export async function loadNeedle(tools: ToolDefinition[], onProgress?: (message: string) => void) {
  unloadNeedle();
  await send({
    type: 'load',
    modelUrl: NEEDLE_MODEL_URL,
    modelSha256: NEEDLE_MODEL_SHA256,
    systemPrompt: '',
    toolsJson: JSON.stringify(buildNeedleToolCatalog(tools))
  }, onProgress);
  return true;
}

export function unloadNeedle() {
  worker?.terminate();
  worker = null;
  rejectAll(new Error('Needle 3 was unloaded.'));
}

/**
 * Needle only extracts calls from the request text. It cannot ask questions or
 * answer, so an empty call list becomes an answer and the app's guardrails
 * handle grounding against the current task list.
 */
export function parseNeedleOutput(raw: string): { plan: AgentPlan; confidence?: number; decodeTokensPerSecond?: number } {
  let parsed: NeedleOutput;
  try {
    parsed = JSON.parse(raw) as NeedleOutput;
  } catch (error) {
    throw new PlannerOutputError('Needle 3 returned invalid JSON.', raw.slice(0, 4_000), { cause: error });
  }
  if (parsed.success === false) throw new PlannerOutputError(`Needle 3 could not plan that request: ${parsed.error ?? 'unknown error'}.`, raw.slice(0, 4_000));
  const calls = parsed.function_calls;
  const validCalls = Array.isArray(calls) && calls.every((call) => call
    && typeof call === 'object'
    && typeof (call as ToolCall).name === 'string'
    && (call as ToolCall).arguments
    && typeof (call as ToolCall).arguments === 'object'
    && !Array.isArray((call as ToolCall).arguments));
  if (!validCalls) throw new PlannerOutputError('Needle 3 returned an invalid call list.', raw.slice(0, 4_000));
  const decodeTokensPerSecond = typeof parsed.decode_tps === 'number' && parsed.decode_tps > 0 ? parsed.decode_tps : undefined;
  const confidence = typeof parsed.confidence === 'number' ? parsed.confidence : undefined;
  if (calls.length === 0) {
    return {
      plan: { outcome: 'answer', calls: [], message: 'I can add, list, complete, or clear tasks, but that request does not match any of those actions.' },
      confidence,
      decodeTokensPerSecond
    };
  }
  const confidenceNote = confidence === undefined ? '' : ` (confidence ${confidence.toFixed(2)})`;
  return {
    plan: { outcome: 'act', calls: calls as ToolCall[], message: `Needle 3 proposed ${calls.length} action${calls.length === 1 ? '' : 's'}${confidenceNote}.` },
    confidence,
    decodeTokensPerSecond
  };
}

/** The tool catalog is fixed at load time; Needle receives only the request text. */
export async function planWithNeedle(input: string): Promise<AgentPlan & { confidence?: number }> {
  if (!worker) throw new Error('Needle 3 is not loaded.');
  const startedAt = performance.now();
  const { output } = await send<{ output: string }>({ type: 'plan', input, maxNewTokens: MAX_NEW_TOKENS });
  const { plan, confidence, decodeTokensPerSecond } = parseNeedleOutput(output);
  return {
    ...plan,
    confidence,
    outputDiagnostics: { rawOutput: output.slice(0, 4_000), validInitially: true, recovered: false, retried: false, attempts: 1, recoverySteps: [] },
    metrics: { elapsedMs: performance.now() - startedAt, estimatedTokensPerSecond: decodeTokensPerSecond }
  };
}
