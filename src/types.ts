export type Task = {
  id: string;
  title: string;
  priority: 'low' | 'medium' | 'high';
  completed: boolean;
  createdAt: string;
};

export type ToolCall = { name: string; arguments: Record<string, unknown> };

export type AgentPlan = {
  calls: ToolCall[];
  reply: string;
};

export type Activity = {
  id: string;
  source: 'person' | 'local agent' | 'browser agent';
  message: string;
  at: string;
};
