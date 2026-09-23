import { afterEach, describe, expect, it, vi } from 'vitest';
import type { ToolDefinition } from './tools';

const deleteConversation = vi.fn(async () => undefined);
const sendMessage = vi.fn();

vi.mock('@litert-lm/core', () => ({
  Engine: {
    create: async () => ({
      createConversation: async () => ({ sendMessage, delete: deleteConversation }),
      delete: async () => undefined
    })
  }
}));

const tools: ToolDefinition[] = [{
  name: 'list_tasks', description: 'List tasks',
  inputSchema: { type: 'object', properties: {} },
  annotations: { readOnlyHint: true },
  execute: async () => ({ count: 0, tasks: [] })
}];

const reply = (text: string) => ({ content: [{ type: 'text', text }] });

describe('planWithLiteRt', () => {
  afterEach(async () => {
    const { unloadLiteRt } = await import('./agent');
    await unloadLiteRt();
    vi.unstubAllGlobals();
    deleteConversation.mockClear();
    sendMessage.mockReset();
  });

  async function loadStubEngine() {
    vi.stubGlobal('navigator', { gpu: {} });
    const agent = await import('./agent');
    await agent.loadLiteRt();
    return agent;
  }

  it('deletes the per-request conversation after a valid plan', async () => {
    const { planWithLiteRt } = await loadStubEngine();
    sendMessage.mockResolvedValueOnce(reply('{"outcome":"act","calls":[{"name":"list_tasks","arguments":{}}],"message":"ok"}'));
    const plan = await planWithLiteRt('show tasks', tools, []);
    expect(plan.calls).toHaveLength(1);
    expect(deleteConversation).toHaveBeenCalledTimes(1);
  });

  it('deletes the per-request conversation when both attempts fail', async () => {
    const { planWithLiteRt } = await loadStubEngine();
    sendMessage.mockResolvedValue(reply('not json'));
    await expect(planWithLiteRt('show tasks', tools, [])).rejects.toThrow();
    expect(sendMessage).toHaveBeenCalledTimes(2);
    expect(deleteConversation).toHaveBeenCalledTimes(1);
  });
});
