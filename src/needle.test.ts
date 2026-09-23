import { describe, expect, it } from 'vitest';
import { PlannerOutputError } from './agent';
import { buildNeedleChoiceCatalog, buildNeedleToolCatalog, clarificationChoices, parseNeedleOutput, translateNeedleCalls } from './needle';
import type { Task } from './types';
import type { ToolDefinition } from './tools';

const output = (fields: Record<string, unknown>) => JSON.stringify({
  type: 'call', success: true, error: null, function_calls: [], confidence: 0.99, decode_tps: 60, ...fields
});

describe('Needle 3 adapter', () => {
  it('declares OpenAI-style tools plus a Needle-only record_finished_task', () => {
    const tools: ToolDefinition[] = ['add_task', 'complete_task'].map((name) => ({
      name, description: `${name} description`,
      inputSchema: { type: 'object', properties: {} },
      execute: async () => ({ ok: true })
    }));
    const catalog = buildNeedleToolCatalog(tools);
    expect(catalog.map((tool) => tool.name)).toEqual(['add_task', 'record_finished_task', 'complete_task']);
    expect(catalog[0].description).toMatch(/own words/);
    expect(catalog[0].parameters).toEqual({ type: 'object', properties: {} });
  });

  it('limits a clarification answer to the offered open tasks', () => {
    const tasks: Task[] = ['submit report', 'review report'].map((title, index) => ({ id: `t${index}`, title, priority: 'medium', completed: false, createdAt: '' }));
    expect(clarificationChoices('Which task should I complete: “submit report” or “review report”?', tasks)).toEqual(['submit report', 'review report']);
    expect(clarificationChoices('Which report do you mean?', tasks)).toBeUndefined();
    expect(clarificationChoices('Which task should I complete: “submit report” or “pay rent”?', tasks)).toBeUndefined();
    expect(buildNeedleChoiceCatalog(['submit report', 'review report'])[0].parameters.properties.task.enum).toEqual(['submit report', 'review report']);
  });

  it('drops a completion that repeats work recorded as finished', () => {
    expect(translateNeedleCalls([
      { name: 'record_finished_task', arguments: { title: 'Sldering iron' } },
      { name: 'complete_task', arguments: { task: 'packed a sldering iron' } }
    ], [])).toEqual([
      { name: 'add_task', arguments: { title: 'Sldering iron' } },
      { name: 'complete_task', arguments: { task: 'Sldering iron' } }
    ]);
  });

  it('translates record_finished_task into the app tools', () => {
    const tasks: Task[] = [{ id: 'slides', title: 'finish the slides', priority: 'medium', completed: false, createdAt: '' }];
    expect(translateNeedleCalls([
      { name: 'record_finished_task', arguments: { title: 'Slides' } },
      { name: 'record_finished_task', arguments: { title: 'Handouts' } },
      { name: 'list_tasks', arguments: {} }
    ], tasks)).toEqual([
      { name: 'complete_task', arguments: { task: 'slides' } },
      { name: 'add_task', arguments: { title: 'Handouts' } },
      { name: 'complete_task', arguments: { task: 'Handouts' } },
      { name: 'list_tasks', arguments: {} }
    ]);
  });

  it('maps function calls to an action plan', () => {
    const calls = [{ name: 'add_task', arguments: { title: 'Buy milk' } }, { name: 'add_task', arguments: { title: 'Call the dentist' } }];
    const { plan, decodeTokensPerSecond } = parseNeedleOutput(output({ function_calls: calls, confidence: 0.52 }));
    expect(plan).toEqual({ outcome: 'act', calls, message: 'Needle 3 proposed a plan (confidence 0.52).' });
    expect(decodeTokensPerSecond).toBe(60);
  });

  it('maps an empty call list to an answer', () => {
    const { plan } = parseNeedleOutput(output({ function_calls: [] }));
    expect(plan.outcome).toBe('answer');
    expect(plan.calls).toEqual([]);
  });

  it('fails closed on engine errors and malformed calls', () => {
    expect(() => parseNeedleOutput('not json')).toThrow(PlannerOutputError);
    expect(() => parseNeedleOutput(output({ success: false, error: 'context overflow' }))).toThrow(/context overflow/);
    expect(() => parseNeedleOutput(output({ function_calls: [{ name: 'add_task', arguments: 'title' }] }))).toThrow(/invalid call list/);
  });
});
