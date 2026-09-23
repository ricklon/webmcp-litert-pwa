import { afterEach, describe, expect, it, vi } from 'vitest';
import { createTools, findTask } from './tools';
import type { Task } from './types';

function harness() {
  let tasks: Task[] = [{
    id: 'coffee', title: 'buy coffee filters', priority: 'medium', completed: false, createdAt: ''
  }];
  const tools = createTools({
    getTasks: () => tasks,
    setTasks: (update) => { tasks = update(tasks); },
    log: () => undefined
  });
  return { tools, getTasks: () => tasks };
}

function ambiguousHarness() {
  let tasks: Task[] = [
    { id: 'submit', title: 'submit report', priority: 'medium', completed: false, createdAt: '' },
    { id: 'review', title: 'review report', priority: 'medium', completed: false, createdAt: '' }
  ];
  const tools = createTools({
    getTasks: () => tasks,
    setTasks: (update) => { tasks = update(tasks); },
    log: () => undefined
  });
  return { tools, getTasks: () => tasks };
}

afterEach(() => vi.unstubAllGlobals());

describe('complete_task', () => {
  it('tolerates a small spelling error in a task title', async () => {
    const app = harness();
    await app.tools.find((tool) => tool.name === 'complete_task')!.execute({ task: 'buy cofee filters' });
    expect(app.getTasks()[0].completed).toBe(true);
  });

  it('accepts the title alias from a model-generated call', async () => {
    const app = harness();
    await app.tools.find((tool) => tool.name === 'complete_task')!.execute({ title: 'buy coffee filters' });
    expect(app.getTasks()[0].completed).toBe(true);
  });

  it('matches the exact UUID supplied from the current task context', async () => {
    const app = harness();
    await app.tools.find((tool) => tool.name === 'complete_task')!.execute({ task: 'coffee' });
    expect(app.getTasks()[0].completed).toBe(true);
  });

  it('returns candidates without changing tasks when a title is ambiguous', async () => {
    const app = ambiguousHarness();
    const result = await app.tools.find((tool) => tool.name === 'complete_task')!.execute({ task: 'report' }) as {
      ok: boolean; ambiguous: boolean; candidates: Array<{ id: string; title: string }>;
    };
    expect(result).toMatchObject({ ok: false, ambiguous: true });
    expect(result.candidates).toHaveLength(2);
    expect(app.getTasks().every((task) => !task.completed)).toBe(true);
  });

  it('returns a no-match error without changing another task', async () => {
    const app = harness();
    const result = await app.tools.find((tool) => tool.name === 'complete_task')!.execute({ task: 'renew passport' });
    expect(result).toMatchObject({ ok: false, error: expect.stringContaining('No task closely matched') });
    expect(app.getTasks()).toHaveLength(1);
    expect(app.getTasks()[0].completed).toBe(false);
  });
});

describe('clear_completed', () => {
  it('removes completed tasks without a second browser confirmation', async () => {
    const app = harness();
    app.getTasks()[0].completed = true;
    const confirm = vi.fn(() => false);
    vi.stubGlobal('confirm', confirm);
    const result = await app.tools.find((tool) => tool.name === 'clear_completed')!.execute({});
    expect(result).toEqual({ ok: true, removed: 1 });
    expect(app.getTasks()).toHaveLength(0);
    expect(confirm).not.toHaveBeenCalled();
  });
});

describe('findTask', () => {
  const task = (id: string, title: string): Task => ({ id, title, priority: 'medium', completed: false, createdAt: '' });

  it('does not match a title that only appears inside another word', () => {
    expect(findTask([task('call', 'Call')], 'recall notes').task).toBeUndefined();
  });

  it('still matches a title contained as whole words', () => {
    expect(findTask([task('call', 'Call')], 'call mom').task?.id).toBe('call');
    expect(findTask([task('filters', 'buy coffee filters')], 'coffee filters').task?.id).toBe('filters');
  });

  it('never matches a title with no words', () => {
    expect(findTask([task('blank', '!!!'), task('call', 'call mom')], 'call mom').task?.id).toBe('call');
    expect(findTask([task('blank', '!!!')], 'ab').task).toBeUndefined();
  });
});
