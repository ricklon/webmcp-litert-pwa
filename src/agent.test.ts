import { describe, expect, it } from 'vitest';
import { authorizeToolPlan, buildBonsaiPlanSchema, buildConversationContext, buildTemporalContext, enforceExplicitBulkCompletion, enforceSafetyGuardrails, parsePlannerOutput, planDeterministically } from './agent';
import { validateJsonSchema } from 'bitgpu/chat';
import type { Activity } from './types';
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

const addTaskTool: ToolDefinition = {
  name: 'add_task', description: 'Add a task',
  inputSchema: {
    type: 'object',
    properties: {
      title: { type: 'string' },
      priority: { type: 'string', enum: ['low', 'medium', 'high'] }
    },
    required: ['title']
  },
  execute: async () => ({ ok: true })
};

describe('runtime clock', () => {
  it('provides local, timezone, and UTC context for relative dates', () => {
    const context = buildTemporalContext(new Date('2026-07-25T18:30:00.000Z'), 'America/New_York');
    expect(context).toContain('2026-07-25, 14:30:00 GMT-04:00');
    expect(context).toContain('IANA time zone: America/New_York');
    expect(context).toContain('UTC timestamp: 2026-07-25T18:30:00.000Z');
  });
});

describe('conversation context', () => {
  it('keeps prior intent in chronological order and excludes telemetry', () => {
    const history: Activity[] = [
      { id: '3', source: 'person', message: 'Add tasks for those.', at: '3:15 PM' },
      { id: '2', source: 'local agent', message: 'Performance · 1.2 s · context 800 / 4,096 tokens', at: '3:14 PM' },
      { id: '1', source: 'person', message: 'Pack the robot, voice agent, USB cables, and power.', at: '3:13 PM' }
    ];
    const context = buildConversationContext(history);
    expect(context).toBe('person: Pack the robot, voice agent, USB cables, and power.\nperson: Add tasks for those.');
  });
});

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

  it('asks which task to complete when one phrase matches several tasks', () => {
    const tasks = [
      { id: 'submit', title: 'submit report', priority: 'medium' as const, completed: false, createdAt: '' },
      { id: 'review', title: 'review report', priority: 'medium' as const, completed: false, createdAt: '' }
    ];
    const plan = planDeterministically('Complete the report', tasks);
    expect(plan.outcome).toBe('clarify');
    expect(plan.calls).toEqual([]);
    expect(plan.message).toContain('submit report');
    expect(plan.message).toContain('review report');
  });

  it('uses a clarification to resolve an ambiguous task', () => {
    const tasks = [
      { id: 'submit', title: 'submit report', priority: 'medium' as const, completed: false, createdAt: '' },
      { id: 'review', title: 'review report', priority: 'medium' as const, completed: false, createdAt: '' }
    ];
    const plan = planDeterministically('Original request: Complete the report\nQuestion asked: Which task?\nUser clarification: submit report', tasks);
    expect(plan.outcome).toBe('act');
    expect(plan.calls[0].arguments.task).toBe('submit');
  });

  it('plans the explicit all-tasks completion capability', () => {
    const tasks = [
      { id: 'one', title: 'Pack robot', priority: 'medium' as const, completed: false, createdAt: '' },
      { id: 'two', title: 'Take Lyft', priority: 'medium' as const, completed: false, createdAt: '' }
    ];
    expect(planDeterministically('OK. I completed all the tasks', tasks).calls).toEqual([
      { name: 'complete_task', arguments: { task: 'one' } },
      { name: 'complete_task', arguments: { task: 'two' } }
    ]);
  });

});

describe('explicit bulk completion guardrail', () => {
  it('repairs a model acknowledgement that omitted the required write proposal', () => {
    const tasks = [
      { id: 'one', title: 'Pack robot', priority: 'medium' as const, completed: false, createdAt: '' },
      { id: 'two', title: 'Take Lyft', priority: 'medium' as const, completed: false, createdAt: '' }
    ];
    const plan = enforceExplicitBulkCompletion(
      { outcome: 'answer', calls: [], message: 'Great!' },
      'OK. I completed all the tasks',
      tasks
    );
    expect(plan.outcome).toBe('act');
    expect(plan.calls).toEqual([
      { name: 'complete_task', arguments: { task: 'one' } },
      { name: 'complete_task', arguments: { task: 'two' } }
    ]);
  });
});

describe('tool contracts', () => {
  it('builds a discriminated constrained-output schema for Bonsai', () => {
    const schema = buildBonsaiPlanSchema(contractTools);
    const calls = schema.properties?.calls;
    expect(calls?.type).toBe('array');
    expect(calls?.items?.properties?.name.enum).toEqual(['complete_task', 'clear_completed']);
    expect(() => validateJsonSchema(schema)).not.toThrow();
  });

  it('lets the model determine intent and preserves valid calls', () => {
    const plan = authorizeToolPlan({
      outcome: 'act', message: 'Done',
      calls: [
        { name: 'complete_task', arguments: { task: 'abc' } },
        { name: 'clear_completed', arguments: {} }
      ]
    }, contractTools);
    expect(plan.calls).toHaveLength(2);
  });

  it('rejects unknown capabilities', () => {
    expect(() => authorizeToolPlan({ outcome: 'act', message: 'Done', calls: [{ name: 'delete_everything', arguments: {} }] }, contractTools))
      .toThrow('unknown tool');
  });

  it('normalizes a semantically equivalent completion argument', () => {
    const plan = authorizeToolPlan({ outcome: 'act', message: 'Done', calls: [{ name: 'complete_task', arguments: { title: 'buy apples' } }] }, contractTools);
    expect(plan.calls[0].arguments.task).toBe('buy apples');
  });

  it('normalizes common priority labels case-insensitively', () => {
    const plan = authorizeToolPlan({
      outcome: 'act', message: 'Done', calls: [{ name: 'add_task', arguments: { title: 'file report', priority: 'Urgent' } }]
    }, [addTaskTool], 'Add file report as urgent');
    expect(plan.calls[0].arguments.priority).toBe('high');
  });

  it('uses the requested priority when the model emits a non-string value', () => {
    const plan = authorizeToolPlan({
      outcome: 'act', message: 'Done', calls: [{ name: 'add_task', arguments: { title: 'file report', priority: { value: 'high' } } }]
    }, [addTaskTool], 'Add file report as high priority');
    expect(plan.calls[0].arguments.priority).toBe('high');
  });

  it('rejects calls missing required arguments', () => {
    expect(() => authorizeToolPlan({ outcome: 'act', message: 'Done', calls: [{ name: 'complete_task', arguments: {} }] }, contractTools))
      .toThrow('Missing required argument');
  });

  it('caps model-generated call volume', () => {
    const calls = Array.from({ length: 11 }, () => ({ name: 'clear_completed', arguments: {} }));
    expect(() => authorizeToolPlan({ outcome: 'act', message: 'Done', calls }, contractTools)).toThrow('too many tool calls');
  });

  it('rejects tool calls attached to a clarification', () => {
    expect(() => authorizeToolPlan({
      outcome: 'clarify', message: 'Which task?', calls: [{ name: 'clear_completed', arguments: {} }]
    }, contractTools)).toThrow('cannot execute tool calls');
  });
});

describe('planner output recovery', () => {
  it('preserves a raw valid plan without marking it recovered', () => {
    const result = parsePlannerOutput('{"outcome":"answer","calls":[],"message":"No action needed."}', true);
    expect(result.plan.outcome).toBe('answer');
    expect(result.diagnostics).toMatchObject({ validInitially: true, recovered: false, attempts: 1 });
  });

  it('closes truncated containers without changing semantic fields', () => {
    const result = parsePlannerOutput('{"outcome":"act","calls":[{"name":"complete_task","arguments":{"task":"abc"}},"message":"Done"}', true);
    expect(result.plan.calls).toEqual([{ name: 'complete_task', arguments: { task: 'abc' } }]);
    expect(result.diagnostics.recoverySteps).toContain('repaired-json-syntax');
  });

  it('normalizes only an allowlisted action outcome when calls exist', () => {
    const result = parsePlannerOutput('{"outcome":"complete_task","calls":[{"name":"complete_task","arguments":{"task":"abc"}}],"message":"Done"}', true);
    expect(result.plan.outcome).toBe('act');
    expect(result.diagnostics.recoverySteps).toContain('normalized-outcome-alias');
  });

  it('recovers a tool argument object emitted as a JSON string', () => {
    const result = parsePlannerOutput('{"outcome":"act","calls":[{"name":"add_task","arguments":"{\\"title\\":\\"buy filters\\",\\"priority\\":\\"medium\\"}"}],"message":"Done"}', true);
    expect(result.plan.calls[0].arguments).toEqual({ title: 'buy filters', priority: 'medium' });
    expect(result.diagnostics.recoverySteps).toContain('parsed-stringified-arguments');
  });

  it('does not turn an alias into an action without a tool call', () => {
    expect(() => parsePlannerOutput('{"outcome":"complete","calls":[],"message":"Done"}', true)).toThrow('invalid plan');
  });
});

describe('deterministic safety guardrails', () => {
  const tasks = [
    { id: 'submit', title: 'submit report', priority: 'medium' as const, completed: false, createdAt: '' },
    { id: 'review', title: 'review report', priority: 'medium' as const, completed: false, createdAt: '' }
  ];
  const unsafe = { outcome: 'act' as const, calls: [{ name: 'complete_task', arguments: { task: 'submit' } }], message: 'Done' };

  it('clarifies an ambiguous explicit completion regardless of the model choice', () => {
    const result = enforceSafetyGuardrails(unsafe, 'Complete the report', tasks);
    expect(result.plan).toMatchObject({ outcome: 'clarify', calls: [] });
    expect(result.interventions).toEqual(['ambiguous-completion']);
  });

  it('fails closed when an explicit completion target is absent', () => {
    const result = enforceSafetyGuardrails(unsafe, 'Complete renew passport', tasks);
    expect(result.plan).toMatchObject({ outcome: 'answer', calls: [] });
    expect(result.interventions).toEqual(['missing-completion-target']);
  });

  it('asks for the target of an underspecified completion', () => {
    expect(enforceSafetyGuardrails(unsafe, 'Complete a task', tasks).plan.outcome).toBe('clarify');
  });

  it('blocks unsupported imperative communication tools', () => {
    const result = enforceSafetyGuardrails(unsafe, 'Email the submit report task to Alex', tasks);
    expect(result.plan).toMatchObject({ outcome: 'answer', calls: [] });
    expect(result.interventions).toEqual(['unsupported-capability']);
  });

  it('constrains an explicit completion to the user-referenced task', () => {
    const result = enforceSafetyGuardrails(unsafe, 'Complete review report', tasks);
    expect(result.plan.calls).toEqual([{ name: 'complete_task', arguments: { task: 'review' } }]);
    expect(result.interventions).toEqual(['constrained-completion-target']);
  });
});
