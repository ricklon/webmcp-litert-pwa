export type Task = {
  id: string;
  title: string;
  priority: 'low' | 'medium' | 'high';
  completed: boolean;
  createdAt: string;
};

export type ToolCall = { name: string; arguments: Record<string, unknown> };

export type PlannerMetrics = {
  elapsedMs: number;
  contextUsage?: number;
  contextWindow?: number;
  estimatedOutputTokens?: number;
  estimatedTokensPerSecond?: number;
  /** Set when Needle-first routing chose which model produced the plan. */
  plannedBy?: string;
};

export type PlannerOutputDiagnostics = {
  rawOutput: string;
  validInitially: boolean;
  recovered: boolean;
  retried: boolean;
  attempts: number;
  recoverySteps: string[];
};

export type AgentPlan = {
  outcome: 'act' | 'clarify' | 'answer';
  calls: ToolCall[];
  message: string;
  metrics?: PlannerMetrics;
  outputDiagnostics?: PlannerOutputDiagnostics;
};

export type PlannerTraceEntry = {
  request: string;
  originalRequest: string;
  planner: 'demo' | 'chrome' | 'litert' | 'bonsai' | 'needle';
  outcome: AgentPlan['outcome'] | 'error';
  calls: ToolCall[];
  message: string;
  status: 'executed' | 'answered' | 'proposed' | 'clarification' | 'failed';
  metrics?: PlannerMetrics;
  rawOutput?: string;
  outputDiagnostics?: PlannerOutputDiagnostics;
  modelOutcome?: AgentPlan['outcome'];
  modelCalls?: ToolCall[];
  guardrailInterventions?: string[];
  /** Needle-first routing: which model decided, and why Needle was bypassed. */
  route?: { decidedBy: 'needle' | 'chrome' | 'litert' | 'bonsai'; escalation: string | null; needleCalls?: ToolCall[] };
};

export type Activity = {
  id: string;
  source: 'person' | 'local agent' | 'browser agent';
  message: string;
  at: string;
  createdAt?: string;
  conversationId?: string;
  order?: number;
};

export type PendingClarification = { request: string; question: string };

export type PlanReview = {
  originalRequest: string;
  plan: AgentPlan;
  status: 'proposed' | 'executed';
};

export type Conversation = {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
};

export type ConversationSession = {
  conversationId: string;
  planReview: PlanReview | null;
  pendingClarification: PendingClarification | null;
  refiningExecutedPlan: boolean;
};
