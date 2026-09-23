import { PlannerOutputError } from './agent';
import { findTask, type ToolDefinition } from './tools';
import type { AgentPlan, Task, ToolCall } from './types';
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
let defaultToolsJson = '';
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

const RECORD_FINISHED_TASK = 'record_finished_task';

/**
 * Needle picks exactly one tool per intent, so it gets its own view of the app's
 * tools (OpenAI-style declarations): add_task keeps the user's wording, and
 * record_finished_task covers work the user already did. Its calls are
 * translated back to the app's tools before planning continues.
 */
export function buildNeedleToolCatalog(tools: ToolDefinition[]) {
  const catalog = tools.map(({ name, description, inputSchema }) => ({
    name,
    description: name === 'add_task'
      ? 'Add one new task the user still needs to do in the future, such as \'I need to wash the car\' or \'remind me to pay rent\'. Use the user\'s own words for the title, including the action verb. Never use it for something the user already did. Call once for each distinct task.'
      : name === 'complete_task' ? 'Mark an existing task on the list complete by its title.' : description,
    parameters: inputSchema
  }));
  const record = {
    name: RECORD_FINISHED_TASK,
    description: 'Record something the user says they already did, described in the past tense, such as \'I packed the charger\', \'I washed the car\', \'I already bought the tickets\', or \'paid the rent\'. Saves it as a completed task.',
    parameters: { type: 'object', properties: { title: { type: 'string', description: 'What the user did, as a short task title.' } }, required: ['title'] }
  };
  // Order matters to Needle; this is the order used in the tool-catalog trial.
  const order = ['add_task', RECORD_FINISHED_TASK, 'complete_task'];
  const rank = (name: string) => (order.includes(name) ? order.indexOf(name) : order.length);
  return [...catalog, record].sort((left, right) => rank(left.name) - rank(right.name));
}

/** While the user is answering "which task?", Needle may only pick one of the offered tasks. */
export function buildNeedleChoiceCatalog(choices: string[]) {
  const listed = choices.map((choice, index) => `${index + 1}) ${choice}`).join(' ');
  return [{
    name: 'complete_task',
    description: `The user is choosing which task to complete. Options: ${listed}. Return the chosen option's exact title.`,
    parameters: { type: 'object', properties: { task: { type: 'string', enum: choices } }, required: ['task'] }
  }];
}

/** Titles offered in a "Which task should I complete: “A” or “B”?" question that are still open tasks. */
export function clarificationChoices(question: string, tasks: Task[]) {
  const titles = [...question.matchAll(/“([^”]+)”/g)].map((match) => match[1]);
  const open = new Set(tasks.filter((task) => !task.completed).map((task) => task.title));
  return titles.length >= 2 && titles.every((title) => open.has(title)) ? titles : undefined;
}

/** Rewrites record_finished_task into the app's tools: complete a matching open task, or add it and complete it. */
export function translateNeedleCalls(calls: ToolCall[], tasks: Task[]): ToolCall[] {
  const openTasks = tasks.filter((task) => !task.completed);
  // Work already recorded as finished in this plan, as pseudo-tasks for matching.
  const recorded: Task[] = [];
  return calls.flatMap((call): ToolCall[] => {
    if (call.name === 'complete_task' && recorded.length) {
      // Needle sometimes repeats a recorded item as a completion; drop the duplicate.
      if (findTask(recorded, call.arguments.task ?? call.arguments.title).task) return [];
    }
    if (call.name !== RECORD_FINISHED_TASK) return [call];
    const title = String(call.arguments.title ?? '').trim();
    recorded.push({ id: `recorded-${recorded.length}`, title, priority: 'medium', completed: true, createdAt: '' });
    const existing = findTask(openTasks, title).task;
    return existing
      ? [{ name: 'complete_task', arguments: { task: existing.id } }]
      : [{ name: 'add_task', arguments: { title } }, { name: 'complete_task', arguments: { task: title } }];
  });
}

export async function loadNeedle(tools: ToolDefinition[], onProgress?: (message: string) => void) {
  unloadNeedle();
  defaultToolsJson = JSON.stringify(buildNeedleToolCatalog(tools));
  await send({
    type: 'load',
    modelUrl: NEEDLE_MODEL_URL,
    modelSha256: NEEDLE_MODEL_SHA256,
    toolsJson: defaultToolsJson
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
    plan: { outcome: 'act', calls: calls as ToolCall[], message: `Needle 3 proposed a plan${confidenceNote}.` },
    confidence,
    decodeTokensPerSecond
  };
}

/**
 * Needle receives only the request text. With `choices`, it may only pick one
 * of the tasks offered in a pending clarification question.
 */
export async function planWithNeedle(input: string, tasks: Task[], choices?: string[]): Promise<AgentPlan & { confidence?: number }> {
  if (!worker) throw new Error('Needle 3 is not loaded.');
  const startedAt = performance.now();
  const toolsJson = choices ? JSON.stringify(buildNeedleChoiceCatalog(choices)) : defaultToolsJson;
  const { output } = await send<{ output: string }>({ type: 'plan', input, maxNewTokens: MAX_NEW_TOKENS, toolsJson });
  const parsed = parseNeedleOutput(output);
  const { confidence, decodeTokensPerSecond } = parsed;
  const plan = { ...parsed.plan, calls: translateNeedleCalls(parsed.plan.calls, tasks) };
  return {
    ...plan,
    confidence,
    outputDiagnostics: { rawOutput: output.slice(0, 4_000), validInitially: true, recovered: false, retried: false, attempts: 1, recoverySteps: [] },
    metrics: { elapsedMs: performance.now() - startedAt, estimatedTokensPerSecond: decodeTokensPerSecond }
  };
}
