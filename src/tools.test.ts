import { describe, expect, it } from 'vitest';
import { createTools } from './tools';
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
});
