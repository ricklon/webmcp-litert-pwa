import { describe, expect, it, vi } from 'vitest';
import type { ToolDefinition } from './tools';
import { registerWebMcpTools, type ModelContext } from './webmcp';

const tools = ['add_task', 'list_tasks', 'complete_task'].map((name) => ({
  name,
  description: name,
  inputSchema: { type: 'object', properties: {} },
  execute: async () => ({ ok: true })
})) satisfies ToolDefinition[];

describe('WebMCP registration lifecycle', () => {
  it('reports unavailable without attempting registration', async () => {
    const controller = new AbortController();
    await expect(registerWebMcpTools(tools, controller.signal, undefined)).resolves.toBe('unavailable');
  });

  it('registers every tool with its schema and browser-agent source', async () => {
    const controller = new AbortController();
    const execute = vi.fn(async () => ({ ok: true }));
    const describedTools: ToolDefinition[] = [{
      name: 'list_tasks',
      description: 'List tasks',
      inputSchema: { type: 'object', properties: { status: { type: 'string' } } },
      annotations: { readOnlyHint: true, idempotentHint: true },
      execute
    }];
    const registered: Array<Record<string, unknown>> = [];
    const context = {
      registerTool: async (tool: unknown) => { registered.push(tool as Record<string, unknown>); },
      getTools: async () => [],
      executeTool: async () => ({})
    } satisfies ModelContext;

    await expect(registerWebMcpTools(describedTools, controller.signal, context)).resolves.toBe('registered');
    expect(registered[0]).toMatchObject({
      name: 'list_tasks',
      description: 'List tasks',
      inputSchema: describedTools[0].inputSchema,
      annotations: { readOnlyHint: true, idempotentHint: true }
    });
    await (registered[0].execute as (args: Record<string, unknown>) => Promise<unknown>)({ status: 'open' });
    expect(execute).toHaveBeenCalledWith({ status: 'open' }, 'browser agent');
  });

  it('reports a registration error without registering later tools', async () => {
    const controller = new AbortController();
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    const registerTool = vi.fn(async () => { throw new Error('bad schema'); });
    const context = {
      registerTool,
      getTools: async () => [],
      executeTool: async () => ({})
    } satisfies ModelContext;
    await expect(registerWebMcpTools(tools, controller.signal, context)).resolves.toBe('error');
    expect(registerTool).toHaveBeenCalledTimes(1);
    expect(consoleError).toHaveBeenCalledWith('WebMCP registration failed', expect.any(Error));
    consoleError.mockRestore();
  });

  it('stops registering tools as soon as its effect signal is aborted', async () => {
    const controller = new AbortController();
    const registerTool = vi.fn(async () => controller.abort());
    const context = {
      registerTool,
      getTools: async () => [],
      executeTool: async () => ({})
    } satisfies ModelContext;

    await expect(registerWebMcpTools(tools, controller.signal, context)).resolves.toBe('available');
    expect(registerTool).toHaveBeenCalledTimes(1);
  });

  it('does no work when cleanup happened before registration began', async () => {
    const controller = new AbortController();
    controller.abort();
    const registerTool = vi.fn();
    const context = {
      registerTool,
      getTools: async () => [],
      executeTool: async () => ({})
    } satisfies ModelContext;

    await expect(registerWebMcpTools(tools, controller.signal, context)).resolves.toBe('available');
    expect(registerTool).not.toHaveBeenCalled();
  });
});
