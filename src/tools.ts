import type { Task, ToolCall } from './types';

export type ToolRuntime = {
  getTasks: () => Task[];
  setTasks: (updater: (tasks: Task[]) => Task[]) => void;
  log: (message: string, source?: 'person' | 'local agent' | 'browser agent') => void;
};

export type ToolDefinition = {
  name: string;
  description: string;
  inputSchema: Record<string, unknown>;
  annotations?: { readOnlyHint?: boolean; destructiveHint?: boolean; idempotentHint?: boolean };
  execute: (args: Record<string, unknown>, source?: 'local agent' | 'browser agent') => Promise<unknown>;
};

const cleanTitle = (value: unknown) => String(value ?? '').trim().slice(0, 120);
const normalize = (value: string) => value.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();

function editDistance(left: string, right: string) {
  const previous = Array.from({ length: right.length + 1 }, (_, index) => index);
  for (let leftIndex = 1; leftIndex <= left.length; leftIndex += 1) {
    let diagonal = previous[0];
    previous[0] = leftIndex;
    for (let rightIndex = 1; rightIndex <= right.length; rightIndex += 1) {
      const above = previous[rightIndex];
      previous[rightIndex] = left[leftIndex - 1] === right[rightIndex - 1]
        ? diagonal
        : Math.min(diagonal, above, previous[rightIndex - 1]) + 1;
      diagonal = above;
    }
  }
  return previous[right.length];
}

export function findTask(tasks: Task[], requested: unknown) {
  const rawQuery = cleanTitle(requested);
  const byId = tasks.find((item) => item.id === rawQuery);
  if (byId) return { task: byId, candidates: [] as Task[] };
  const query = normalize(rawQuery);
  if (!query) return { task: undefined, candidates: [] as Task[] };
  const direct = tasks.filter((item) => normalize(item.title).includes(query) || query.includes(normalize(item.title)));
  if (direct.length === 1) return { task: direct[0], candidates: [] as Task[] };
  if (direct.length > 1) return { task: undefined, candidates: direct };
  const ranked = tasks
    .map((item) => ({ item, distance: editDistance(normalize(item.title), query) }))
    .sort((left, right) => left.distance - right.distance);
  const best = ranked[0];
  if (!best || best.distance > Math.max(2, Math.floor(query.length * 0.2))) return { task: undefined, candidates: [] as Task[] };
  const equallyClose = ranked.filter((item) => item.distance === best.distance).map((item) => item.item);
  return equallyClose.length === 1
    ? { task: best.item, candidates: [] as Task[] }
    : { task: undefined, candidates: equallyClose };
}

export function createTools(runtime: ToolRuntime): ToolDefinition[] {
  return [
    {
      name: 'add_task',
      description: 'Add a useful task to the local task list. Use when the user asks to remember, schedule, or add something to do.',
      inputSchema: {
        type: 'object',
        properties: {
          title: { type: 'string', description: 'Concise action-oriented task title.' },
          priority: { type: 'string', enum: ['low', 'medium', 'high'], description: 'Task importance.' }
        },
        required: ['title']
      },
      annotations: { readOnlyHint: false, destructiveHint: false },
      execute: async ({ title, priority = 'medium' }, source = 'browser agent') => {
        const cleaned = cleanTitle(title);
        if (!cleaned) throw new Error('A non-empty title is required.');
        const task: Task = {
          id: crypto.randomUUID(),
          title: cleaned,
          priority: ['low', 'medium', 'high'].includes(String(priority)) ? priority as Task['priority'] : 'medium',
          completed: false,
          createdAt: new Date().toISOString()
        };
        runtime.setTasks((tasks) => [task, ...tasks]);
        runtime.log(`Added “${task.title}” (${task.priority}).`, source);
        return { ok: true, task };
      }
    },
    {
      name: 'list_tasks',
      description: 'List current local tasks, optionally filtered by completion state. Use to answer what is pending or already done.',
      inputSchema: {
        type: 'object',
        properties: {
          status: { type: 'string', enum: ['all', 'open', 'completed'], description: 'Which tasks to return.' }
        }
      },
      annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true },
      execute: async ({ status = 'all' }) => {
        const tasks = runtime.getTasks().filter((task) => status === 'all' || (status === 'completed' ? task.completed : !task.completed));
        return { count: tasks.length, tasks };
      }
    },
    {
      name: 'complete_task',
      description: 'Mark one task complete by its exact id or by a distinctive part of its title.',
      inputSchema: {
        type: 'object',
        properties: {
          task: { type: 'string', description: 'Task id or distinctive title text.' }
        },
        required: ['task']
      },
      annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: true },
      execute: async ({ task, title, id }, source = 'browser agent') => {
        // Models commonly choose a semantically equivalent key despite the schema.
        // Accept those aliases at the trust boundary, then normalize once.
        const requested = task ?? title ?? id;
        const query = cleanTitle(requested);
        const { task: found, candidates } = findTask(runtime.getTasks(), requested);
        if (candidates.length > 1) {
          return {
            ok: false,
            ambiguous: true,
            error: `Several tasks matched “${query}”.`,
            candidates: candidates.map(({ id, title }) => ({ id, title }))
          };
        }
        if (!found) return { ok: false, error: query ? `No task closely matched “${query}”.` : 'A task id or title is required.' };
        runtime.setTasks((tasks) => tasks.map((item) => item.id === found.id ? { ...item, completed: true } : item));
        runtime.log(`Completed “${found.title}”.`, source);
        return { ok: true, task: { ...found, completed: true } };
      }
    },
    {
      name: 'clear_completed',
      description: 'Remove all completed tasks. Only use when the user explicitly asks to clear or delete completed work.',
      inputSchema: { type: 'object', properties: {} },
      annotations: { readOnlyHint: false, destructiveHint: true, idempotentHint: true },
      execute: async (_, source = 'browser agent') => {
        const count = runtime.getTasks().filter((task) => task.completed).length;
        if (count > 0 && !globalThis.confirm(`Remove ${count} completed task${count === 1 ? '' : 's'}?`)) {
          return { ok: false, cancelled: true };
        }
        runtime.setTasks((tasks) => tasks.filter((task) => !task.completed));
        runtime.log(`Cleared ${count} completed task${count === 1 ? '' : 's'}.`, source);
        return { ok: true, removed: count };
      }
    }
  ];
}

export async function executeLocalTool(tools: ToolDefinition[], call: ToolCall, source: 'local agent' | 'browser agent') {
  const tool = tools.find((candidate) => candidate.name === call.name);
  if (!tool) throw new Error(`Unknown tool: ${call.name}`);
  return tool.execute(call.arguments, source);
}
