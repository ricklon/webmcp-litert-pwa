import type { ToolDefinition } from './tools';

type ModelContext = {
  registerTool: (tool: unknown, options?: { signal?: AbortSignal }) => Promise<void> | void;
  getTools: () => Promise<Array<{ name: string }>>;
  executeTool: (tool: unknown, args: string) => Promise<unknown>;
};

declare global {
  interface Document { modelContext?: ModelContext }
}

export type WebMcpStatus = 'available' | 'unavailable' | 'registered' | 'error';

export async function registerWebMcpTools(tools: ToolDefinition[], signal: AbortSignal): Promise<WebMcpStatus> {
  if (!document.modelContext) return 'unavailable';
  try {
    for (const tool of tools) {
      await document.modelContext.registerTool({
        ...tool,
        execute: (args: Record<string, unknown>) => tool.execute(args, 'browser agent')
      }, { signal });
    }
    return 'registered';
  } catch (error) {
    console.error('WebMCP registration failed', error);
    return 'error';
  }
}
