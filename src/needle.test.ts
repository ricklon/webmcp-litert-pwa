import { describe, expect, it } from 'vitest';
import { PlannerOutputError } from './agent';
import { buildNeedleToolCatalog, parseNeedleOutput } from './needle';
import type { ToolDefinition } from './tools';

const output = (fields: Record<string, unknown>) => JSON.stringify({
  type: 'call', success: true, error: null, function_calls: [], confidence: 0.99, decode_tps: 60, ...fields
});

describe('Needle 3 adapter', () => {
  it('declares tools with OpenAI-style parameters', () => {
    const tools: ToolDefinition[] = [{
      name: 'add_task', description: 'Add a task',
      inputSchema: { type: 'object', properties: { title: { type: 'string' } }, required: ['title'] },
      execute: async () => ({ ok: true })
    }];
    expect(buildNeedleToolCatalog(tools)).toEqual([{
      name: 'add_task', description: 'Add a task',
      parameters: { type: 'object', properties: { title: { type: 'string' } }, required: ['title'] }
    }]);
  });

  it('maps function calls to an action plan', () => {
    const calls = [{ name: 'add_task', arguments: { title: 'Buy milk' } }, { name: 'add_task', arguments: { title: 'Call the dentist' } }];
    const { plan, decodeTokensPerSecond } = parseNeedleOutput(output({ function_calls: calls, confidence: 0.52 }));
    expect(plan).toEqual({ outcome: 'act', calls, message: 'Needle 3 proposed 2 actions (confidence 0.52).' });
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
