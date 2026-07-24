import { describe, expect, it } from 'vitest';
import { authorizeToolPlan, planDeterministically } from './agent';
import type { ToolDefinition } from './tools';

const contractTools: ToolDefinition[] = [
  {
    name: 'complete_task', description: 'Complete a task',
    inputSchema: { type: 'object', properties: { task: { type: 'string' } }, required: ['task'] },
    execute: async () => ({ ok: true })
  },
  {
    name: 'clear_completed', description: 'Clear completed tasks',
    inputSchema: { type: 'object', properties: {} },
    execute: async () => ({ ok: true })
  }
];

describe('demo planner', () => {
  it('plans an add task call', () => {
    expect(planDeterministically('Add buy coffee filters', []).calls[0]).toEqual({
      name: 'add_task', arguments: { title: 'buy coffee filters', priority: 'medium' }
    });
  });

  it('uses a known task id when completing', () => {
    const tasks = [{ id: 'abc', title: 'Buy coffee filters', priority: 'medium' as const, completed: false, createdAt: '' }];
    expect(planDeterministically('Complete buy coffee filters', tasks).calls[0]).toEqual({
      name: 'complete_task', arguments: { task: 'abc' }
    });
  });
});

describe('tool contracts', () => {
  it('lets the model determine intent and preserves valid calls', () => {
    const plan = authorizeToolPlan({
      reply: 'Done',
      calls: [
        { name: 'complete_task', arguments: { task: 'abc' } },
        { name: 'clear_completed', arguments: {} }
      ]
    }, contractTools);
    expect(plan.calls).toHaveLength(2);
  });

  it('rejects unknown capabilities', () => {
    expect(() => authorizeToolPlan({ reply: 'Done', calls: [{ name: 'delete_everything', arguments: {} }] }, contractTools))
      .toThrow('unknown tool');
  });

  it('normalizes a semantically equivalent completion argument', () => {
    const plan = authorizeToolPlan({ reply: 'Done', calls: [{ name: 'complete_task', arguments: { title: 'buy apples' } }] }, contractTools);
    expect(plan.calls[0].arguments.task).toBe('buy apples');
  });

  it('rejects calls missing required arguments', () => {
    expect(() => authorizeToolPlan({ reply: 'Done', calls: [{ name: 'complete_task', arguments: {} }] }, contractTools))
      .toThrow('Missing required argument');
  });

  it('caps model-generated call volume', () => {
    const calls = Array.from({ length: 11 }, () => ({ name: 'clear_completed', arguments: {} }));
    expect(() => authorizeToolPlan({ reply: 'Done', calls }, contractTools)).toThrow('too many tool calls');
  });
});
