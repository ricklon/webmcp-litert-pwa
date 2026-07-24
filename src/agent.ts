import type { AgentPlan, Task } from './types';
import type { ToolDefinition } from './tools';

const MODEL_URL = 'https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it-web.litertlm';
export const AGENT_SYSTEM_PROMPT = 'You are a careful local task-planning agent. Return only valid JSON tool plans and never invent tools or task ids.';

type Conversation = {
  sendMessage: (message: string) => Promise<{ content: Array<{ type?: string; text?: string }> }>;
};

type Engine = {
  createConversation: (config?: unknown) => Promise<Conversation>;
  delete: () => Promise<void>;
};

let engine: Engine | null = null;
let conversation: Conversation | null = null;

type ChromeAvailability = 'unavailable' | 'downloadable' | 'downloading' | 'available';
type ChromeSession = {
  prompt: (message: string, options?: { responseConstraint?: Record<string, unknown> }) => Promise<string>;
  destroy?: () => void;
};
type ChromeLanguageModel = {
  availability: (options?: unknown) => Promise<ChromeAvailability>;
  create: (options?: unknown) => Promise<ChromeSession>;
};

declare global { interface Window { LanguageModel?: ChromeLanguageModel } }

let chromeSession: ChromeSession | null = null;

const chromeOptions = {
  expectedInputs: [{ type: 'text', languages: ['en'] }],
  expectedOutputs: [{ type: 'text', languages: ['en'] }]
};

export async function getChromeModelAvailability(): Promise<ChromeAvailability> {
  if (!window.LanguageModel) return 'unavailable';
  return window.LanguageModel.availability(chromeOptions);
}

export async function loadChromeModel(onProgress?: (message: string) => void) {
  if (!window.LanguageModel) throw new Error('Chrome’s Prompt API is not available.');
  chromeSession?.destroy?.();
  chromeSession = await window.LanguageModel.create({
    ...chromeOptions,
    initialPrompts: [{ role: 'system', content: AGENT_SYSTEM_PROMPT }],
    monitor(monitor: EventTarget) {
      monitor.addEventListener('downloadprogress', (event) => {
        const loaded = (event as Event & { loaded: number }).loaded;
        onProgress?.(`Downloading Chrome’s model… ${Math.round(loaded * 100)}%`);
      });
    }
  });
}

export function unloadChromeModel() {
  chromeSession?.destroy?.();
  chromeSession = null;
}

export async function loadLiteRt(model = MODEL_URL, onProgress?: (message: string) => void) {
  if (!('gpu' in navigator)) throw new Error('WebGPU is not available in this browser.');
  onProgress?.('Importing LiteRT-LM Web API…');
  const { Engine } = await import('@litert-lm/core');
  onProgress?.('Downloading and compiling the on-device model…');
  engine = await Engine.create({ model, mainExecutorSettings: { maxNumTokens: 4096 } }) as Engine;
  conversation = await engine.createConversation({
    preface: { messages: [{ role: 'system', content: AGENT_SYSTEM_PROMPT }] }
  });
  return true;
}

export async function unloadLiteRt() {
  await engine?.delete();
  engine = null;
  conversation = null;
}

function extractJson(text: string): AgentPlan {
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/i)?.[1];
  const candidate = fenced ?? text.slice(text.indexOf('{'), text.lastIndexOf('}') + 1);
  const parsed = JSON.parse(candidate) as AgentPlan;
  if (!Array.isArray(parsed.calls) || typeof parsed.reply !== 'string') throw new Error('The model returned an invalid plan.');
  return parsed;
}

function buildAgentPrompt(prompt: string, tools: ToolDefinition[], tasks: Task[]) {
  const catalog = tools.map(({ name, description, inputSchema }) => ({ name, description, inputSchema }));
  return [
    'Plan tool calls for the user request.',
    'Return exactly JSON: {"calls":[{"name":"tool_name","arguments":{}}],"reply":"short confirmation"}.',
    'Use only listed tools. Do not invent task ids. Call the minimum number of tools needed.',
    'Decompose compound requests into every independently actionable outcome before choosing tools.',
    'When a user asks for several new tasks, call add_task once for each distinct task; do not collapse the whole request into one item.',
    'Before responding, silently check that every requested item is represented exactly once.',
    'Never call clear_completed unless the user explicitly asks to clear, remove, or delete completed tasks.',
    'For a direct add, complete, list, or clear request, call only that one corresponding tool.',
    'Use an empty calls array if no tool is appropriate.',
    `Tools: ${JSON.stringify(catalog)}`,
    `Current tasks: ${JSON.stringify(tasks)}`,
    `User: ${prompt}`
  ].join('\n');
}

export async function planWithChrome(prompt: string, tools: ToolDefinition[], tasks: Task[]): Promise<AgentPlan> {
  if (!chromeSession) throw new Error('Chrome’s built-in model is not ready.');
  const responseConstraint = {
    type: 'object',
    properties: {
      calls: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            name: { type: 'string', enum: tools.map((tool) => tool.name) },
            arguments: { type: 'object' }
          },
          required: ['name', 'arguments'],
          additionalProperties: false
        }
      },
      reply: { type: 'string' }
    },
    required: ['calls', 'reply'],
    additionalProperties: false
  };
  return extractJson(await chromeSession.prompt(buildAgentPrompt(prompt, tools, tasks), { responseConstraint }));
}

function validateArguments(schema: Record<string, unknown>, args: Record<string, unknown>) {
  const required = Array.isArray(schema.required) ? schema.required as string[] : [];
  const properties = (schema.properties ?? {}) as Record<string, { type?: string; enum?: unknown[] }>;
  for (const name of required) if (!(name in args)) throw new Error(`Missing required argument “${name}”.`);
  for (const [name, value] of Object.entries(args)) {
    const property = properties[name];
    if (!property) continue;
    if (property.type === 'string' && typeof value !== 'string') throw new Error(`Argument “${name}” must be a string.`);
    if (property.enum && !property.enum.includes(value)) throw new Error(`Argument “${name}” has an unsupported value.`);
  }
}

/** The model decides intent; the application validates capability contracts. */
export function authorizeToolPlan(plan: AgentPlan, tools: ToolDefinition[]): AgentPlan {
  if (plan.calls.length > 10) throw new Error('The model proposed too many tool calls.');
  const calls = plan.calls.map((call) => {
    const tool = tools.find((candidate) => candidate.name === call.name);
    if (!tool) throw new Error(`The model proposed unknown tool “${call.name}”.`);
    if (!call.arguments || typeof call.arguments !== 'object' || Array.isArray(call.arguments)) throw new Error(`Tool “${call.name}” requires an argument object.`);
    const completionAlias = call.arguments.title ?? call.arguments.id;
    const normalizedArguments = call.name === 'complete_task' && !('task' in call.arguments) && completionAlias !== undefined
      ? { ...call.arguments, task: completionAlias }
      : call.arguments;
    validateArguments(tool.inputSchema, normalizedArguments);
    return { ...call, arguments: normalizedArguments };
  });
  return { ...plan, calls };
}

const LEADING_TASK_VERBS = new Set(['buy', 'wash', 'submit', 'send', 'pay', 'call', 'book', 'fix', 'clean', 'get', 'make', 'plan', 'pack']);

function findReferencedTasks(request: string, tasks: Task[]) {
  const requestTokens = new Set(request.toLowerCase().match(/[a-z0-9]+/g) ?? []);
  return tasks.filter((task) => {
    const words = task.title.toLowerCase().match(/[a-z0-9]+/g) ?? [];
    const firstWord = words[0];
    const looksLikeLeadingVerb = firstWord && [...LEADING_TASK_VERBS].some((verb) => verb === firstWord || verb.startsWith(firstWord) || firstWord.startsWith(verb));
    const subject = looksLikeLeadingVerb ? words.slice(1) : words;
    return subject.length > 0 && subject.every((word) => requestTokens.has(word));
  });
}

export async function planWithLiteRt(prompt: string, tools: ToolDefinition[], tasks: Task[]): Promise<AgentPlan> {
  if (!conversation) throw new Error('LiteRT-LM is not loaded.');
  const response = await conversation.sendMessage(buildAgentPrompt(prompt, tools, tasks));
  return extractJson(response.content.map((item) => item.text ?? '').join(''));
}

export function planDeterministically(prompt: string, tasks: Task[]): AgentPlan {
  const text = prompt.trim();
  const lower = text.toLowerCase();
  if (/\b(clear|remove|delete)\b.*\b(completed|done)\b/.test(lower)) {
    return { calls: [{ name: 'clear_completed', arguments: {} }], reply: 'I cleared the completed tasks.' };
  }
  if (/\b(show|list|what|review)\b/.test(lower) && /\b(tasks?|todos?|open|done|completed)\b/.test(lower)) {
    const status = /\b(done|completed)\b/.test(lower) ? 'completed' : /\b(open|pending)\b/.test(lower) ? 'open' : 'all';
    return { calls: [{ name: 'list_tasks', arguments: { status } }], reply: `I checked your ${status} tasks.` };
  }
  if (/\b(complete|finish|finished|completed|done|check off|mark|bought|purchased|washed|submitted|sent|paid|called|booked|fixed|cleaned)\b/.test(lower)) {
    const referenced = findReferencedTasks(prompt, tasks.filter((task) => !task.completed));
    if (referenced.length > 1) {
      return {
        calls: referenced.map((task) => ({ name: 'complete_task', arguments: { task: task.id } })),
        reply: `I marked ${referenced.length} matching tasks complete.`
      };
    }
    const known = tasks.find((task) => lower.includes(task.title.toLowerCase()));
    const target = known?.id ?? referenced[0]?.id ?? text
      .replace(/^.*?\b(complete|finish|done|check off|mark)\b/i, '')
      .replace(/\bas complete\b/gi, '')
      .replace(/["“”]/g, '')
      .trim();
    return { calls: [{ name: 'complete_task', arguments: { task: target } }], reply: 'I marked the matching task complete.' };
  }
  const title = text.replace(/^(please\s+)?(add|remember to|remind me to|create a task to|i need to)\s+/i, '').trim();
  if (title && /^(please\s+)?(add|remember|remind|create|i need)/i.test(text)) {
    const priority = /\b(urgent|important|high priority)\b/i.test(text) ? 'high' : /\b(low priority|someday)\b/i.test(text) ? 'low' : 'medium';
    return { calls: [{ name: 'add_task', arguments: { title, priority } }], reply: `I added “${title}”.` };
  }
  return { calls: [], reply: 'Try “add buy printer paper”, “show open tasks”, or “complete buy printer paper”.' };
}

export { MODEL_URL };
