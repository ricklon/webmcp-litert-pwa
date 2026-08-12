import type { ToolDefinition } from './tools';

export type ModelContext = {
  registerTool: (tool: unknown, options?: { signal?: AbortSignal }) => Promise<void> | void;
  getTools: () => Promise<Array<{ name: string }>>;
  executeTool: (tool: unknown, args: string) => Promise<unknown>;
};

declare global {
  interface Document { modelContext?: ModelContext }
}

export type WebMcpStatus = 'available' | 'unavailable' | 'registered' | 'error';

export async function registerWebMcpTools(
  tools: ToolDefinition[],
  signal: AbortSignal,
  modelContext: ModelContext | undefined = typeof document === 'undefined' ? undefined : document.modelContext
): Promise<WebMcpStatus> {
  if (!modelContext) return 'unavailable';
  if (signal.aborted) return 'available';
  try {
    for (const tool of tools) {
      if (signal.aborted) return 'available';
      await modelContext.registerTool({
        ...tool,
        execute: (args: Record<string, unknown>) => tool.execute(args, 'browser agent')
      }, { signal });
    }
    if (signal.aborted) return 'available';
    return 'registered';
  } catch (error) {
    if (signal.aborted || (error instanceof DOMException && error.name === 'AbortError')) return 'available';
    console.error('WebMCP registration failed', error);
    return 'error';
  }
}
