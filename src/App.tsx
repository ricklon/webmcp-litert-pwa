import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AGENT_SYSTEM_PROMPT, authorizeToolPlan, enforceExplicitBulkCompletion, enforceSafetyGuardrails, getChromeModelAvailability, loadBonsai, loadChromeModel, loadLiteRt, MODEL_E4B_URL, MODEL_URL, PlannerOutputError, planDeterministically, planWithBonsai, planWithChrome, planWithLiteRt, unloadBonsai, unloadChromeModel, unloadLiteRt } from './agent';
import { appendMemoryEvent, clearMemory, createMemoryConversation, loadMemory, saveConversationSession, selectMemoryConversation } from './memory';
import { createTools, executeLocalTool } from './tools';
import type { Activity, AgentPlan, Conversation, PendingClarification, PlannerMetrics, PlannerTraceEntry, PlanReview, Task } from './types';
import { registerWebMcpTools, type WebMcpStatus } from './webmcp';
import { evaluateScenario, SCENARIOS, seedScenario } from './scenarios';

const TASK_KEY = 'local-tools-lab.tasks.v1';

function readTasks(): Task[] {
  try { return JSON.parse(localStorage.getItem(TASK_KEY) ?? '[]') as Task[]; }
  catch { return []; }
}

const time = () => new Intl.DateTimeFormat(undefined, { hour: 'numeric', minute: '2-digit' }).format(new Date());
type RunFeedback = { tone: 'idle' | 'working' | 'success' | 'proposal' | 'clarify' | 'error'; title: string; detail: string; tools?: string[] };
type RequestResult = { status: 'executed' | 'answered' | 'proposed' | 'clarification' | 'failed'; message: string };

function formatDuration(elapsedMs: number) {
  return elapsedMs < 1000 ? `${Math.max(1, Math.round(elapsedMs))} ms` : `${(elapsedMs / 1000).toFixed(2)} s`;
}

function formatPlannerActivity(metrics: PlannerMetrics) {
  const parts = [`Performance · ${formatDuration(metrics.elapsedMs)}`];
  if (metrics.contextWindow) {
    const usage = metrics.contextUsage;
    const percentage = usage === undefined ? '' : ` (${Math.round((usage / metrics.contextWindow) * 100)}%)`;
    parts.push(`context ${usage?.toLocaleString() ?? '—'} / ${metrics.contextWindow.toLocaleString()} tokens${percentage}`);
  }
  if (metrics.estimatedOutputTokens !== undefined) parts.push(`output ~${Math.round(metrics.estimatedOutputTokens)} tokens`);
  if (metrics.estimatedTokensPerSecond !== undefined) parts.push(`~${metrics.estimatedTokensPerSecond.toFixed(1)} tok/s`);
  return parts.join(' · ');
}

function describePlannerError(error: unknown) {
  const message = error instanceof Error ? error.message : String(error);
  return message.includes('kErrorUnknown')
    ? 'Chrome’s local model hit a transient error. Nothing was changed; retry this request or refinement.'
    : message || 'The agent could not finish that request.';
}

function describeToolResult(name: string, value: unknown) {
  const result = value as { task?: Task; count?: number; removed?: number };
  if (name === 'add_task' && result.task) return `Added “${result.task.title}”.`;
  if (name === 'complete_task' && result.task) return `Completed “${result.task.title}”.`;
  if (name === 'list_tasks') return `Found ${result.count ?? 0} matching task${result.count === 1 ? '' : 's'}.`;
  if (name === 'clear_completed') return `Removed ${result.removed ?? 0} completed task${result.removed === 1 ? '' : 's'}.`;
  return `${name} completed.`;
}

export default function App() {
  const requestedLiteRt = new URLSearchParams(location.search).get('litertModel');
  const [liteRtVariant, setLiteRtVariant] = useState<'e2b' | 'e4b'>(requestedLiteRt === 'e2b' ? 'e2b' : 'e4b');
  const liteRtModelName = liteRtVariant === 'e4b' ? 'Gemma 4 E4B' : 'Gemma 4 E2B';
  const [tasks, setTasks] = useState<Task[]>(readTasks);
  const tasksRef = useRef(tasks);
  const [activity, setActivity] = useState<Activity[]>([]);
  const activityRef = useRef(activity);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [conversationId, setConversationId] = useState('');
  const conversationIdRef = useRef('');
  const [memoryReady, setMemoryReady] = useState(false);
  const [webMcp, setWebMcp] = useState<WebMcpStatus>('available');
  const [planner, setPlanner] = useState<'demo' | 'chrome' | 'litert' | 'bonsai'>('demo');
  const [loadingPlanner, setLoadingPlanner] = useState<'chrome' | 'litert' | 'bonsai' | null>(null);
  const runtimeRequestRef = useRef(0);
  const runtimeActivationBusyRef = useRef(false);
  const [chromeAvailability, setChromeAvailability] = useState<'checking' | 'unavailable' | 'downloadable' | 'downloading' | 'available'>('checking');
  const [engineNote, setEngineNote] = useState('Checking for Chrome’s built-in model…');
  const [pwaReady, setPwaReady] = useState(false);
  const [prompt, setPrompt] = useState('');
  const [pendingClarification, setPendingClarification] = useState<PendingClarification | null>(null);
  const pendingClarificationRef = useRef<PendingClarification | null>(null);
  const setPendingClarificationNow = useCallback((value: PendingClarification | null) => {
    pendingClarificationRef.current = value;
    setPendingClarification(value);
  }, []);
  const [planReview, setPlanReview] = useState<PlanReview | null>(null);
  const [refiningExecutedPlan, setRefiningExecutedPlan] = useState(false);
  const [plannerMetrics, setPlannerMetrics] = useState<PlannerMetrics | null>(null);
  const [busy, setBusy] = useState(false);
  const [feedback, setFeedback] = useState<RunFeedback>({
    tone: 'idle',
    title: 'Ready for a request',
    detail: 'Choose an example or type a request, then press Run.'
  });
  const [scenarioId, setScenarioId] = useState(SCENARIOS[0].id);
  const [scenarioState, setScenarioState] = useState<'idle' | 'loaded' | 'running' | 'passed' | 'failed' | 'stale'>('idle');
  const scenarioSnapshotRef = useRef('');
  const [scenarioStep, setScenarioStep] = useState(0);
  const [scenarioResult, setScenarioResult] = useState('Choose Load scenario to replace the current task list with controlled test data.');
  const [scenarioTrace, setScenarioTrace] = useState<PlannerTraceEntry[]>([]);
  const scenarioTraceRef = useRef<PlannerTraceEntry[]>([]);

  useEffect(() => {
    let active = true;
    loadMemory().then((snapshot) => {
      if (!active) return;
      setConversations(snapshot.conversations);
      setConversationId(snapshot.activeConversation.id);
      conversationIdRef.current = snapshot.activeConversation.id;
      setActivity(snapshot.activity);
      activityRef.current = snapshot.activity;
      setPlanReview(snapshot.session.planReview);
      setPendingClarificationNow(snapshot.session.pendingClarification);
      setRefiningExecutedPlan(snapshot.session.refiningExecutedPlan);
      if (snapshot.session.planReview?.status === 'proposed') {
        setFeedback({ tone: 'proposal', title: 'Proposal restored', detail: 'Review, refine, or approve the proposal saved in this conversation.', tools: snapshot.session.planReview.plan.calls.map((call) => call.name) });
      }
      setMemoryReady(true);
    }).catch((error) => {
      if (!active) return;
      console.error('Conversation memory could not be loaded', error);
      setMemoryReady(true);
    });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    tasksRef.current = tasks;
    localStorage.setItem(TASK_KEY, JSON.stringify(tasks));
  }, [tasks]);

  useEffect(() => {
    // A managed result describes one exact task snapshot. Any later task edit
    // invalidates that result, including edits made through WebMCP.
    if ((scenarioState === 'passed' || scenarioState === 'failed')
      && JSON.stringify(tasks) !== scenarioSnapshotRef.current) {
      setScenarioState('stale');
      setScenarioResult('Task state changed after that managed run. Its previous result is stale; load the scenario again for a controlled evaluation.');
    }
    // Intentionally react only to task changes. A run may set its final state
    // after its last task update, and that new result is still current.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tasks]);

  useEffect(() => {
    if (!memoryReady || !conversationId) return;
    saveConversationSession({ conversationId, planReview, pendingClarification, refiningExecutedPlan })
      .catch((error) => console.error('Conversation state could not be saved', error));
  }, [conversationId, memoryReady, pendingClarification, planReview, refiningExecutedPlan]);

  useEffect(() => {
    if (!('serviceWorker' in navigator)) return;
    navigator.serviceWorker.ready.then(() => setPwaReady(true));
  }, []);

  useEffect(() => {
    let active = true;
    const requestId = ++runtimeRequestRef.current;
    getChromeModelAvailability().then(async (availability) => {
      if (!active) return;
      setChromeAvailability(availability);
      if (availability === 'available') {
        try {
          await loadChromeModel(setEngineNote);
          if (!active || requestId !== runtimeRequestRef.current) {
            unloadChromeModel();
            return;
          }
          setPlanner('chrome');
          setEngineNote('Chrome built-in model active · on device');
          setFeedback({ tone: 'success', title: 'Chrome’s model is ready', detail: 'Your next request will be planned locally with Gemini Nano.' });
        } catch (error) {
          if (!active || requestId !== runtimeRequestRef.current) return;
          setEngineNote(error instanceof Error ? error.message : 'Chrome model could not start.');
        }
      } else if (availability === 'downloadable' || availability === 'downloading') {
        setEngineNote('Chrome’s model is supported and can be enabled below. Demo rules are active until then.');
      } else {
        setEngineNote('Chrome Prompt API unavailable. Demo rules are active; LiteRT-LM is optional.');
      }
    });
    return () => { active = false; };
  }, []);

  const log = useCallback((message: string, source: Activity['source'] = 'person') => {
    const createdAt = new Date().toISOString();
    const conversation = conversationIdRef.current;
    const entry: Activity = {
      id: crypto.randomUUID(), source, message, at: time(), createdAt,
      conversationId: conversation || undefined,
      order: Math.round((performance.timeOrigin + performance.now()) * 1000)
    };
    const next = [entry, ...activityRef.current].slice(0, 30);
    activityRef.current = next;
    setActivity(next);
    if (conversation) appendMemoryEvent(conversation, entry).catch((error) => console.error('Activity could not be saved', error));
    if (conversation) {
      setConversations((items) => items
        .map((item) => item.id === conversation ? {
          ...item,
          title: source === 'person' && item.title === 'New conversation' ? message.replace(/\s+/g, ' ').trim().slice(0, 54) : item.title,
          updatedAt: createdAt
        } : item)
        .sort((left, right) => right.updatedAt.localeCompare(left.updatedAt)));
    }
  }, []);

  const tools = useMemo(() => createTools({
    getTasks: () => tasksRef.current,
    setTasks: (updater) => {
      // Tool plans and managed evaluations are imperative. Compute from the
      // synchronous reference so a completed tool call cannot be evaluated
      // before React has had a chance to run a queued state updater.
      const next = updater(tasksRef.current);
      tasksRef.current = next;
      setTasks(next);
    },
    log
  }), [log]);

  const webMcpTools = useMemo(() => tools.map((tool) => tool.annotations?.readOnlyHint ? tool : {
    ...tool,
    description: `Propose this write for user review. Nothing changes until the user approves it. ${tool.description}`,
    execute: async (arguments_: Record<string, unknown>) => {
      const call = { name: tool.name, arguments: arguments_ };
      setPendingClarificationNow(null);
      setRefiningExecutedPlan(false);
      setPlanReview((current) => current?.status === 'proposed'
        ? {
          ...current,
          plan: {
            ...current.plan,
            calls: [...current.plan.calls, call],
            message: 'A browser agent added an action to this proposal. Review every action before approval.'
          }
        }
        : {
          originalRequest: `Browser agent requested ${tool.name}`,
          status: 'proposed',
          plan: {
            outcome: 'act',
            calls: [call],
            message: 'A browser agent proposed this action. Nothing has changed yet.'
          }
        });
      log(`Proposed ${tool.name} for review.`, 'browser agent');
      setFeedback({ tone: 'proposal', title: 'Browser-agent proposal', detail: 'Nothing has changed yet. Review, edit, refine, or approve the proposed action.', tools: [tool.name] });
      return { ok: true, pendingReview: true, tool: tool.name };
    }
  }), [log, tools]);

  useEffect(() => {
    const controller = new AbortController();
    setWebMcp(document.modelContext ? 'available' : 'unavailable');
    if (!memoryReady) return () => controller.abort();
    registerWebMcpTools(webMcpTools, controller.signal).then((status) => {
      if (!controller.signal.aborted) setWebMcp(status);
    });
    return () => controller.abort();
  }, [memoryReady, webMcpTools]);

  async function activateModel(variant: 'e2b' | 'e4b') {
    if (runtimeActivationBusyRef.current) return;
    const modelName = variant === 'e4b' ? 'Gemma 4 E4B' : 'Gemma 4 E2B';
    const size = variant === 'e4b' ? '~4 GB' : '~2 GB';
    if (!globalThis.confirm(`Download the ${size} ${modelName} model and run it locally with WebGPU?`)) return;
    runtimeActivationBusyRef.current = true;
    const requestId = ++runtimeRequestRef.current;
    setLiteRtVariant(variant);
    setLoadingPlanner('litert');
    setFeedback({ tone: 'working', title: 'Preparing LiteRT-LM', detail: `Downloading and compiling ${modelName}. This can take a while.` });
    try {
      await unloadLiteRt();
      unloadChromeModel();
      unloadBonsai();
      setPlanner('demo');
      await loadLiteRt(variant === 'e4b' ? MODEL_E4B_URL : MODEL_URL, setEngineNote);
      if (requestId !== runtimeRequestRef.current) return;
      setPlanner('litert');
      setPendingClarificationNow(null);
      setPlanReview(null);
      setRefiningExecutedPlan(false);
      setEngineNote(`${modelName} · WebGPU · on device`);
      log('LiteRT-LM model is ready.', 'local agent');
      setFeedback({ tone: 'success', title: 'LiteRT-LM is ready', detail: `Your next request will use ${modelName} locally through WebGPU.` });
    } catch (error) {
      if (requestId !== runtimeRequestRef.current) return;
      setEngineNote(error instanceof Error ? error.message : 'Model failed to load.');
      setFeedback({ tone: 'error', title: 'LiteRT-LM could not start', detail: error instanceof Error ? error.message : 'Model failed to load.' });
    } finally {
      runtimeActivationBusyRef.current = false;
      setLoadingPlanner(null);
    }
  }

  async function activateBonsai() {
    if (runtimeActivationBusyRef.current) return;
    if (!globalThis.confirm('Download the ~3.8 GB Bonsai 27B model and run it locally with WebGPU? A GPU with at least 16 GB of available memory is recommended.')) return;
    runtimeActivationBusyRef.current = true;
    const requestId = ++runtimeRequestRef.current;
    setLoadingPlanner('bonsai');
    setFeedback({ tone: 'working', title: 'Preparing Bonsai 27B', detail: 'Downloading the 1-bit GGUF weights and compiling custom WebGPU kernels. This can take a while.' });
    try {
      await unloadLiteRt();
      unloadChromeModel();
      await loadBonsai(setEngineNote);
      if (requestId !== runtimeRequestRef.current) return;
      setPlanner('bonsai');
      setPendingClarificationNow(null);
      setPlanReview(null);
      setRefiningExecutedPlan(false);
      setEngineNote('Bonsai 27B · 1-bit GGUF · WebGPU · on device');
      log('Bonsai 27B is ready.', 'local agent');
      setFeedback({ tone: 'success', title: 'Bonsai 27B is ready', detail: 'Your next request will use Bonsai locally through custom WebGPU kernels.' });
    } catch (error) {
      unloadBonsai();
      if (requestId !== runtimeRequestRef.current) return;
      setEngineNote(error instanceof Error ? error.message : 'Bonsai 27B failed to load.');
      setFeedback({ tone: 'error', title: 'Bonsai 27B could not start', detail: error instanceof Error ? error.message : 'Model failed to load.' });
    } finally {
      runtimeActivationBusyRef.current = false;
      setLoadingPlanner(null);
    }
  }

  async function activateChromeModel() {
    if (runtimeActivationBusyRef.current) return;
    runtimeActivationBusyRef.current = true;
    const requestId = ++runtimeRequestRef.current;
    setLoadingPlanner('chrome');
    setFeedback({ tone: 'working', title: 'Preparing Chrome’s model', detail: 'Chrome may download Gemini Nano before creating the local session.' });
    try {
      await unloadLiteRt();
      unloadBonsai();
      await loadChromeModel(setEngineNote);
      if (requestId !== runtimeRequestRef.current) return;
      setPlanner('chrome');
      setPendingClarificationNow(null);
      setPlanReview(null);
      setRefiningExecutedPlan(false);
      setChromeAvailability('available');
      setEngineNote('Chrome built-in model active · on device');
      log('Chrome’s built-in model is ready.', 'local agent');
      setFeedback({ tone: 'success', title: 'Chrome’s model is ready', detail: 'Your next request will be planned locally with Gemini Nano.' });
    } catch (error) {
      if (requestId !== runtimeRequestRef.current) return;
      setEngineNote(error instanceof Error ? error.message : 'Chrome’s model failed to load.');
      setFeedback({ tone: 'error', title: 'Chrome’s model could not start', detail: error instanceof Error ? error.message : 'The built-in model failed to load.' });
    } finally {
      runtimeActivationBusyRef.current = false;
      setLoadingPlanner(null);
    }
  }

  async function useDemoMode() {
    if (runtimeActivationBusyRef.current) return;
    runtimeRequestRef.current += 1;
    await unloadLiteRt();
    unloadBonsai();
    unloadChromeModel();
    setPlanner('demo');
    setPendingClarificationNow(null);
    setPlanReview(null);
    setRefiningExecutedPlan(false);
    setEngineNote('Demo agent active. Enter a request under “Local agent” and press Run.');
    log('Demo agent selected. It is ready for a request.', 'local agent');
    setFeedback({ tone: 'success', title: 'Demo rules selected', detail: 'Type a supported task request and press Run.' });
  }

  function isReadOnlyPlan(plan: AgentPlan) {
    return plan.calls.every((call) => tools.find((tool) => tool.name === call.name)?.annotations?.readOnlyHint === true);
  }

  function hasRequiredArguments(plan: AgentPlan) {
    return plan.calls.length > 0 && plan.calls.every((call) => {
      const tool = tools.find((candidate) => candidate.name === call.name);
      const required = Array.isArray(tool?.inputSchema.required) ? tool.inputSchema.required as string[] : [];
      return required.every((name) => call.arguments[name] !== undefined
        && (typeof call.arguments[name] !== 'string' || String(call.arguments[name]).trim().length > 0));
    });
  }

  async function executePlan(plan: AgentPlan, originalRequest: string) {
    const results: Array<{ name: string; value: unknown }> = [];
    for (const call of plan.calls) {
      setFeedback({ tone: 'working', title: `Calling ${call.name}`, detail: `Using ${JSON.stringify(call.arguments)}`, tools: plan.calls.map((item) => item.name) });
      const result = await executeLocalTool(tools, call, 'local agent');
      if (result && typeof result === 'object' && 'ok' in result && result.ok === false) {
        const failure = result as { error?: string; cancelled?: boolean; ambiguous?: boolean; candidates?: Array<{ title: string }> };
        if (failure.ambiguous && failure.candidates?.length) {
          const question = `Which task should I complete: ${failure.candidates.map((candidate) => `“${candidate.title}”`).join(' or ')}?`;
          setPendingClarificationNow({ request: originalRequest, question });
          setPlanReview(null);
          log(question, 'local agent');
          setFeedback({ tone: 'clarify', title: 'More information needed', detail: question });
          return { status: 'clarification', message: question } satisfies RequestResult;
        }
        throw new Error(failure.cancelled ? 'Action cancelled.' : failure.error ?? `${call.name} failed.`);
      }
      results.push({ name: call.name, value: result });
    }
    const reply = results.length
      ? results.map((result) => describeToolResult(result.name, result.value)).join(' ')
      : plan.message;
    if (plan.calls.length === 0 || isReadOnlyPlan(plan)) log(reply, 'local agent');
    setPlanReview(plan.calls.length ? { originalRequest, plan, status: 'executed' } : null);
    setRefiningExecutedPlan(false);
    setFeedback({
      tone: 'success',
      title: plan.calls.length ? 'Request completed' : 'Answer',
      detail: reply,
      tools: plan.calls.map((call) => call.name)
    });
    return {
      status: plan.calls.length ? 'executed' : 'answered',
      message: plan.calls.length ? reply : plan.message
    } satisfies RequestResult;
  }

  async function planRequest(request: string, options: { autoApprove?: boolean; includeHistory?: boolean; includeTasks?: boolean } = {}) {
    const conversationHistory = options.includeHistory === false ? [] : activityRef.current;
    log(request, 'person');
    const currentClarification = pendingClarificationRef.current;
    const refiningProposal = planReview?.status === 'proposed';
    const refiningCompleted = refiningExecutedPlan && planReview?.status === 'executed';
    const originalRequest = currentClarification?.request
      ?? ((refiningProposal || refiningCompleted) ? planReview?.originalRequest : undefined)
      ?? request;
    if (!currentClarification && !refiningProposal && !refiningCompleted) setPlanReview(null);
    const planningRequest = currentClarification
      ? `Original request: ${currentClarification.request}\nQuestion asked: ${currentClarification.question}\nUser clarification: ${request}`
      : refiningProposal
        ? `Original request: ${planReview.originalRequest}\nCurrent proposed plan: ${JSON.stringify(planReview.plan.calls)}\nUser refinement: ${request}\nReturn a complete replacement plan incorporating the refinement. Do not execute anything yet.`
        : refiningCompleted
          ? `Original request: ${planReview.originalRequest}\nAlready executed calls: ${JSON.stringify(planReview.plan.calls)}\nUser refinement after execution: ${request}\nReturn only additional or corrective calls. Do not repeat completed work.`
          : request;
    setPlannerMetrics(null);
    setFeedback({ tone: 'working', title: `${planner === 'chrome' ? 'Chrome’s model' : planner === 'litert' ? 'LiteRT-LM' : planner === 'bonsai' ? 'Bonsai 27B' : 'Demo rules'} is planning`, detail: `Reading: “${request}”` });
    let tracePlan: AgentPlan | null = null;
    let traceModelPlan: AgentPlan | null = null;
    let traceGuardrails: string[] = [];
    let traceRawOutput: string | undefined;
    const recordTrace = (status: PlannerTraceEntry['status'], message: string) => {
      const entry: PlannerTraceEntry = {
        request,
        originalRequest,
        planner,
        outcome: tracePlan?.outcome ?? 'error',
        calls: tracePlan?.calls ?? [],
        message,
        status,
        metrics: tracePlan?.metrics,
        rawOutput: traceRawOutput ?? tracePlan?.outputDiagnostics?.rawOutput,
        outputDiagnostics: tracePlan?.outputDiagnostics,
        modelOutcome: traceModelPlan?.outcome,
        modelCalls: traceModelPlan?.calls,
        guardrailInterventions: traceGuardrails
      };
      scenarioTraceRef.current = [...scenarioTraceRef.current, entry];
      setScenarioTrace(scenarioTraceRef.current);
    };
    try {
      const startedAt = performance.now();
      const planningTasks = options.includeTasks === false ? [] : tasksRef.current;
      const proposedPlan = planner === 'chrome'
        ? await planWithChrome(planningRequest, tools, planningTasks, conversationHistory)
        : planner === 'litert'
          ? await planWithLiteRt(planningRequest, tools, planningTasks, conversationHistory)
          : planner === 'bonsai'
            ? await planWithBonsai(planningRequest, tools, planningTasks, conversationHistory)
          : planDeterministically(planningRequest, planningTasks);
      const metrics = proposedPlan.metrics ?? { elapsedMs: performance.now() - startedAt };
      traceModelPlan = proposedPlan;
      tracePlan = proposedPlan;
      setPlannerMetrics(metrics);
      log(formatPlannerActivity(metrics), 'local agent');
      const guarded = enforceSafetyGuardrails(proposedPlan, request, tasksRef.current);
      traceGuardrails = guarded.interventions;
      const plan = authorizeToolPlan(enforceExplicitBulkCompletion(guarded.plan, request, tasksRef.current), tools, request);
      tracePlan = { ...plan, metrics };
      if (plan.outcome === 'clarify') {
        setPendingClarificationNow({ request: originalRequest, question: plan.message });
        setPlanReview(null);
        setRefiningExecutedPlan(false);
        log(plan.message, 'local agent');
        setFeedback({ tone: 'clarify', title: 'More information needed', detail: plan.message });
        recordTrace('clarification', plan.message);
        return { status: 'clarification', message: plan.message } satisfies RequestResult;
      }
      setPendingClarificationNow(null);
      if (plan.outcome === 'act' && !isReadOnlyPlan(plan) && !options.autoApprove) {
        setPlanReview({ originalRequest, plan, status: 'proposed' });
        setRefiningExecutedPlan(false);
        setFeedback({ tone: 'proposal', title: 'Review proposed actions', detail: 'Nothing has changed yet. Edit, refine, or approve this plan.', tools: plan.calls.map((call) => call.name) });
        recordTrace('proposed', plan.message);
        return { status: 'proposed', message: plan.message } satisfies RequestResult;
      }
      const result = await executePlan(plan, originalRequest);
      recordTrace(result.status, result.message);
      return result;
    } catch (error) {
      if (error instanceof PlannerOutputError) traceRawOutput = error.rawOutput;
      const detail = describePlannerError(error);
      setPrompt(request);
      log(detail, 'local agent');
      setFeedback({ tone: 'error', title: 'Request failed', detail });
      recordTrace('failed', detail);
      return { status: 'failed', message: detail } satisfies RequestResult;
    }
  }

  async function approvePlan() {
    if (!planReview || planReview.status !== 'proposed' || busy || !hasRequiredArguments(planReview.plan)) return;
    setBusy(true);
    try { await executePlan(authorizeToolPlan(planReview.plan, tools, planReview.originalRequest), planReview.originalRequest); }
    catch (error) {
      log(error instanceof Error ? error.message : 'The agent could not execute that plan.', 'local agent');
      setFeedback({ tone: 'error', title: 'Execution failed', detail: error instanceof Error ? error.message : 'The agent could not execute that plan.' });
    } finally { setBusy(false); }
  }

  function updateProposalArgument(callIndex: number, name: string, value: string) {
    setPlanReview((current) => current?.status === 'proposed' ? {
      ...current,
      plan: {
        ...current.plan,
        calls: current.plan.calls.map((call, index) => index === callIndex
          ? { ...call, arguments: { ...call.arguments, [name]: value } }
          : call)
      }
    } : current);
  }

  function removeProposedCall(callIndex: number) {
    setPlanReview((current) => current?.status === 'proposed' ? {
      ...current,
      plan: { ...current.plan, calls: current.plan.calls.filter((_, index) => index !== callIndex) }
    } : current);
  }

  function addProposedTask() {
    setPlanReview((current) => current?.status === 'proposed' ? {
      ...current,
      plan: { ...current.plan, calls: [...current.plan.calls, { name: 'add_task', arguments: { title: '', priority: 'medium' } }] }
    } : current);
  }

  async function submitPrompt(event: FormEvent) {
    event.preventDefault();
    const request = prompt.trim();
    if (!request || busy) return;
    setPrompt('');
    setBusy(true);
    await planRequest(request);
    setBusy(false);
  }

  const activeScenario = SCENARIOS.find((scenario) => scenario.id === scenarioId)!;

  function loadScenario() {
    if (tasksRef.current.length > 0 && !globalThis.confirm('Replace the current task list with this scenario’s controlled starting state?')) return;
    const seeded = seedScenario(activeScenario);
    tasksRef.current = seeded;
    setTasks(seeded);
    setScenarioState('loaded');
    scenarioSnapshotRef.current = '';
    scenarioTraceRef.current = [];
    setScenarioTrace([]);
    setScenarioStep(0);
    setPendingClarificationNow(null);
    setPlanReview(null);
    setRefiningExecutedPlan(false);
    setScenarioResult('Starting state loaded. Run the workflow when ready.');
    setFeedback({ tone: 'success', title: `Loaded “${activeScenario.name}”`, detail: `${seeded.length} seed task${seeded.length === 1 ? '' : 's'} ready.` });
  }

  async function runScenario() {
    if (scenarioState !== 'loaded' && scenarioState !== 'failed' && scenarioState !== 'passed') return;
    setBusy(true);
    setScenarioState('running');
    setScenarioResult(`Running with ${planner === 'chrome' ? 'Chrome built-in AI' : planner === 'litert' ? 'LiteRT-LM' : planner === 'bonsai' ? 'Bonsai 27B' : 'demo rules'}…`);
    let workflowFailure = '';
    for (let index = 0; index < activeScenario.requests.length; index += 1) {
      setScenarioStep(index + 1);
      const result = await planRequest(activeScenario.requests[index], {
        autoApprove: true,
        includeHistory: false,
        includeTasks: !activeScenario.hiddenTaskContextRequests?.includes(index)
      });
      const expectedStatus = activeScenario.expectedRequestStatuses?.[index] ?? 'executed';
      if (result.status !== expectedStatus) {
        workflowFailure = result.status === 'answered'
          ? `Expected ${expectedStatus}, but the model answered without proposing any tool calls: ${result.message}`
          : result.status === 'clarification'
            ? `Expected ${expectedStatus}, but the model requested clarification: ${result.message}`
            : result.status === 'proposed'
              ? `Expected ${expectedStatus}, but the managed run paused on an unexecuted proposal: ${result.message}`
              : `Expected ${expectedStatus}, but the workflow request failed: ${result.message}`;
        break;
      }
    }
    const failures = workflowFailure ? [workflowFailure] : evaluateScenario(activeScenario, tasksRef.current);
    scenarioSnapshotRef.current = JSON.stringify(tasksRef.current);
    setScenarioState(failures.length ? 'failed' : 'passed');
    setScenarioResult(failures.length ? failures.join(' ') : `Passed all ${activeScenario.requests.length} step${activeScenario.requests.length === 1 ? '' : 's'} and state checks.`);
    setBusy(false);
  }

  async function switchConversation(nextConversationId: string) {
    if (!nextConversationId || nextConversationId === conversationId || busy) return;
    setBusy(true);
    try {
      if (conversationId) {
        await saveConversationSession({ conversationId, planReview, pendingClarification, refiningExecutedPlan });
      }
      const selected = await selectMemoryConversation(nextConversationId);
      conversationIdRef.current = nextConversationId;
      setConversationId(nextConversationId);
      setActivity(selected.activity);
      activityRef.current = selected.activity;
      setPlanReview(selected.session.planReview);
      setPendingClarificationNow(selected.session.pendingClarification);
      setRefiningExecutedPlan(selected.session.refiningExecutedPlan);
      setPlannerMetrics(null);
      setPrompt('');
      setFeedback(selected.session.planReview?.status === 'proposed'
        ? { tone: 'proposal', title: 'Proposal restored', detail: 'Review, refine, or approve the saved proposal.', tools: selected.session.planReview.plan.calls.map((call) => call.name) }
        : { tone: 'idle', title: 'Conversation restored', detail: 'Continue naturally; current tasks are shared across conversations.' });
    } catch (error) {
      setFeedback({ tone: 'error', title: 'Conversation could not be opened', detail: error instanceof Error ? error.message : 'Conversation memory failed.' });
    } finally {
      setBusy(false);
    }
  }

  async function startNewConversation() {
    if (busy) return;
    setBusy(true);
    try {
      if (conversationId) {
        await saveConversationSession({ conversationId, planReview, pendingClarification, refiningExecutedPlan });
      }
      const conversation = await createMemoryConversation();
      conversationIdRef.current = conversation.id;
      setConversationId(conversation.id);
      setConversations((items) => [conversation, ...items]);
      setActivity([]);
      activityRef.current = [];
      setPlanReview(null);
      setPendingClarificationNow(null);
      setRefiningExecutedPlan(false);
      setPlannerMetrics(null);
      setPrompt('');
      setFeedback({ tone: 'idle', title: 'New conversation', detail: 'Describe what you want to get done. Existing tasks remain available as current state.' });
    } catch (error) {
      setFeedback({ tone: 'error', title: 'Conversation could not be created', detail: error instanceof Error ? error.message : 'Conversation memory failed.' });
    } finally {
      setBusy(false);
    }
  }

  async function resetLocalData() {
    if (!globalThis.confirm('Reset all Local Tools Lab tasks, activity, and scenario progress on this device? This cannot be undone.')) return;
    const conversation = await clearMemory();
    tasksRef.current = [];
    setTasks([]);
    setActivity([]);
    activityRef.current = [];
    conversationIdRef.current = conversation.id;
    setConversationId(conversation.id);
    setConversations([conversation]);
    setPendingClarificationNow(null);
    setPlanReview(null);
    setRefiningExecutedPlan(false);
    setScenarioState('idle');
    setScenarioStep(0);
    setScenarioResult('Local scenario progress was reset. Load a scenario to begin again.');
    localStorage.removeItem(TASK_KEY);
    setFeedback({ tone: 'success', title: 'Local app data reset', detail: 'Tasks, activity, and scenario progress were cleared. Browser-managed models were not removed.' });
  }

  const openCount = tasks.filter((task) => !task.completed).length;
  const completedCount = tasks.length - openCount;

  return (
    <main>
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Local Tools Lab home">
          <span className="brand-mark">LT</span><span>Local Tools Lab</span>
        </a>
        <div className="top-status"><span className="live-dot" /> local-first experiment</div>
      </header>

      <section className="hero" id="top">
        <p className="eyebrow">WebMCP × on-device AI</p>
        <h1>A tiny agent that can<br /><em>actually use the page.</em></h1>
        <p className="lede">Plan your day in natural language. Review and refine the model’s proposal before any write reaches the page.</p>
        <div className="architecture" aria-label="Application architecture">
          <span>You</span><b>→</b><span className={planner !== 'demo' ? 'active' : ''}>{planner === 'chrome' ? 'Chrome model' : planner === 'litert' ? 'LiteRT model' : planner === 'bonsai' ? 'Bonsai 27B' : 'Demo rules'}</span><b>→</b><span className={planReview?.status === 'proposed' ? 'active' : ''}>Review</span><b>→</b><span className={webMcp === 'registered' ? 'active' : ''}>Page tools</span><b>→</b><span>Tasks</span>
        </div>
      </section>

      <section className="setup-panel" aria-labelledby="setup-title">
        <div className="setup-copy">
          <p className="kicker">Setup check</p>
          <h2 id="setup-title">What works in this browser?</h2>
          <p>Chrome’s built-in model is preferred when supported. Demo rules work everywhere, while LiteRT-LM provides an optional custom local model. Native WebMCP needs a compatible Chrome build.</p>
        </div>
        <ul className="checks">
          <li className="pass"><span>✓</span><div><b>Demo workflow</b><small>Ready now—no browser setting or model required.</small></div></li>
          <li className={chromeAvailability === 'available' ? 'pass' : 'wait'}><span>{chromeAvailability === 'available' ? '✓' : '2'}</span><div><b>Chrome built-in AI</b><small>{chromeAvailability === 'available' ? 'Gemini Nano is ready.' : chromeAvailability === 'downloadable' ? 'Supported; select it below to download.' : chromeAvailability === 'downloading' ? 'Chrome is downloading its model.' : chromeAvailability === 'checking' ? 'Checking availability…' : 'Prompt API not available on this browser/device.'}</small></div></li>
          <li className={'gpu' in navigator ? 'pass' : 'wait'}><span>{'gpu' in navigator ? '✓' : '!'}</span><div><b>WebGPU</b><small>{'gpu' in navigator ? 'Available for LiteRT-LM.' : 'Not detected; use current Chrome with supported hardware.'}</small></div></li>
          <li className={webMcp === 'registered' ? 'pass' : 'wait'}><span>{webMcp === 'registered' ? '✓' : '3'}</span><div><b>Native WebMCP</b><small>{webMcp === 'registered' ? 'Four page tools registered.' : 'Open chrome://flags/#enable-webmcp-testing, enable it, relaunch Chrome, then reload.'}</small></div></li>
          <li className={pwaReady ? 'pass' : 'wait'}><span>{pwaReady ? '✓' : '4'}</span><div><b>Offline app shell</b><small>{pwaReady ? 'Service worker ready.' : 'Available after the production app is installed or revisited.'}</small></div></li>
        </ul>
        <details>
          <summary>Testing with Playwright</summary>
          <p>Playwright uses its own Chromium by default; changing your personal Chrome flags does not affect it. Launch Chromium with <code>--enable-features=WebMCP</code> and <code>--enable-blink-features=WebMCPTesting</code>, or configure Playwright to use your flagged Chrome channel/profile. Native calls should still be feature-detected because this API is experimental.</p>
        </details>
      </section>

      <section className="workspace">
        <div className="agent-panel">
          <div className="section-heading">
            <div><p className="kicker">01 / Ask</p><h2>Local agent</h2></div>
            <span className="state state-ready">{planner}</span>
          </div>
          <div className="conversation-controls">
            <label><span>Conversation</span><select aria-label="Conversation" value={conversationId} onChange={(event) => switchConversation(event.target.value)} disabled={!memoryReady || busy}>
              {conversations.map((conversation) => <option key={conversation.id} value={conversation.id}>{conversation.title}</option>)}
            </select></label>
            <button type="button" onClick={startNewConversation} disabled={!memoryReady || busy}>New conversation</button>
          </div>
          <form className="prompt-box" onSubmit={submitPrompt}>
            <label htmlFor="agent-prompt">{pendingClarification
              ? 'Your clarification'
              : planReview?.status === 'proposed'
                ? 'Refine this proposal'
                : refiningExecutedPlan
                  ? 'Refine the completed result'
                  : 'What should we get done?'}</label>
            <textarea id="agent-prompt" value={prompt} onChange={(event) => setPrompt(event.target.value)} onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
                event.preventDefault();
                event.currentTarget.form?.requestSubmit();
              }
            }} placeholder={pendingClarification
              ? 'Type your answer…'
              : planReview?.status === 'proposed' || refiningExecutedPlan
                ? 'Describe what the agent should change…'
                : 'Add submit the expense report as high priority…'} rows={4} />
            <div className="prompt-footer">
              <span>{busy ? 'Planning…' : `Enter to submit · Shift+Enter for a new line · ${planner === 'chrome' ? 'Chrome AI' : planner === 'litert' ? 'LiteRT-LM' : planner === 'bonsai' ? 'Bonsai 27B' : 'Demo rules'}`}</span>
              <button type="submit" disabled={!memoryReady || busy || !prompt.trim()}>{pendingClarification ? 'Answer' : planReview?.status === 'proposed' || refiningExecutedPlan ? 'Refine' : 'Plan'} <span>↗</span></button>
            </div>
          </form>
          {planReview && <section className={`plan-review ${planReview.status}`} aria-label={planReview.status === 'proposed' ? 'Proposed actions' : 'Executed plan'}>
            <div className="plan-review-heading">
              <div><small>{planReview.status === 'proposed' ? 'Awaiting your approval' : 'Executed'}</small><b>{planReview.status === 'proposed' ? 'Agent proposal' : 'Completed plan'}</b></div>
              <span>{planReview.plan.calls.length} action{planReview.plan.calls.length === 1 ? '' : 's'}</span>
            </div>
            <p className="original-request"><small>Original request</small>{planReview.originalRequest}</p>
            <p className="agent-note"><small>Agent note</small>{planReview.plan.message}</p>
            <ol className="proposed-calls">
              {planReview.plan.calls.map((call, callIndex) => <li key={`${call.name}-${callIndex}`}>
                <div className="proposed-call-heading"><code>{call.name}</code>{planReview.status === 'proposed' && <button type="button" onClick={() => removeProposedCall(callIndex)} aria-label={`Remove proposed ${call.name} action`}>Remove</button>}</div>
                {Object.keys(call.arguments).length === 0
                  ? <span className="no-arguments">No arguments</span>
                  : Object.entries(call.arguments).map(([name, value]) => <label key={name}>
                    <span>{name}</span>
                    {planReview.status === 'proposed' && name === 'priority'
                      ? <select value={String(value)} onChange={(event) => updateProposalArgument(callIndex, name, event.target.value)}><option value="low">low</option><option value="medium">medium</option><option value="high">high</option></select>
                      : planReview.status === 'proposed' && typeof value === 'string'
                        ? <input value={value} onChange={(event) => updateProposalArgument(callIndex, name, event.target.value)} />
                        : <output>{String(value)}</output>}
                  </label>)}
              </li>)}
            </ol>
            {planReview.status === 'proposed' ? <div className="plan-review-actions">
              <button type="button" onClick={addProposedTask}>Add task</button>
              <button type="button" onClick={() => {
                setPrompt(planReview.originalRequest);
                setPlanReview(null);
                setFeedback({ tone: 'idle', title: 'Proposal cancelled', detail: 'Nothing was changed. Edit the request and submit it when ready.' });
              }}>Cancel</button>
              <button type="button" className="approve" onClick={approvePlan} disabled={busy || !hasRequiredArguments(planReview.plan)}>Approve and execute</button>
            </div> : <div className="plan-review-actions">
              <button type="button" onClick={() => {
                setRefiningExecutedPlan(true);
                document.getElementById('agent-prompt')?.focus();
              }}>Refine result</button>
              <button type="button" onClick={() => {
                setPlanReview(null);
                setRefiningExecutedPlan(false);
                setPrompt('');
                document.getElementById('agent-prompt')?.focus();
              }}>Start new request</button>
            </div>}
          </section>}
          {!pendingClarification && !planReview && <div className="examples">
            {['Add buy coffee filters', 'Show my open tasks', 'Complete buy coffee filters'].map((example) => (
              <button key={example} onClick={() => setPrompt(example)}>{example}</button>
            ))}
          </div>}
          <div className={`run-feedback ${feedback.tone}`} role="status" aria-live="polite" aria-atomic="true">
            <span className="feedback-signal">{feedback.tone === 'working' ? '···' : feedback.tone === 'success' ? '✓' : feedback.tone === 'proposal' ? '≡' : feedback.tone === 'clarify' ? '?' : feedback.tone === 'error' ? '!' : '→'}</span>
            <div><b>{feedback.title}</b><p>{feedback.detail}</p>
              {plannerMetrics && <div className="planner-metrics" aria-label="Planner performance">
                <span><small>Context</small>{plannerMetrics.contextWindow
                  ? `${plannerMetrics.contextUsage?.toLocaleString() ?? '—'} / ${plannerMetrics.contextWindow.toLocaleString()}`
                  : 'unavailable'}</span>
                <span><small>Time</small>{formatDuration(plannerMetrics.elapsedMs)}</span>
                <span title="Estimated from the returned text and total planning time; this is not raw decoder throughput."><small>Speed</small>{plannerMetrics.estimatedTokensPerSecond
                  ? `~${plannerMetrics.estimatedTokensPerSecond.toFixed(1)} tok/s`
                  : 'unavailable'}</span>
              </div>}
              {feedback.tools && feedback.tools.length > 0 && <div className="called-tools"><span>Tools called</span>{feedback.tools.map((tool, index) => <code key={`${tool}-${index}`}>{tool}</code>)}</div>}
            </div>
          </div>
        </div>

        <aside className="runtime-panel">
          <p className="kicker">02 / Runtime</p>
          <h2>Planner source</h2>
          <button className={`runtime-card ${planner === 'chrome' ? 'selected' : ''}`} onClick={activateChromeModel} disabled={loadingPlanner !== null || chromeAvailability === 'unavailable' || chromeAvailability === 'checking'} aria-pressed={planner === 'chrome'}>
            <span className="runtime-icon chrome">◉</span><span><b>Chrome built-in</b><small>Gemini Nano · managed by Chrome</small></span><i>{loadingPlanner === 'chrome' ? 'Loading' : chromeAvailability === 'downloadable' ? 'Enable' : 'Use'}</i>
          </button>
          <button className={`runtime-card ${planner === 'litert' && liteRtVariant === 'e4b' ? 'selected' : ''}`} onClick={() => activateModel('e4b')} disabled={loadingPlanner !== null} aria-pressed={planner === 'litert' && liteRtVariant === 'e4b'}>
            <span className="runtime-icon lime">◆</span><span><b>LiteRT-LM recommended</b><small>Gemma 4 E4B · stronger JSON · WebGPU</small></span><i>{loadingPlanner === 'litert' && liteRtVariant === 'e4b' ? 'Loading' : 'Load'}</i>
          </button>
          <button className={`runtime-card ${planner === 'litert' && liteRtVariant === 'e2b' ? 'selected' : ''}`} onClick={() => activateModel('e2b')} disabled={loadingPlanner !== null} aria-pressed={planner === 'litert' && liteRtVariant === 'e2b'}>
            <span className="runtime-icon lime">◇</span><span><b>LiteRT-LM lighter</b><small>Gemma 4 E2B · faster · WebGPU</small></span><i>{loadingPlanner === 'litert' && liteRtVariant === 'e2b' ? 'Loading' : 'Load'}</i>
          </button>
          <button className={`runtime-card ${planner === 'bonsai' ? 'selected' : ''}`} onClick={activateBonsai} disabled={loadingPlanner !== null} aria-pressed={planner === 'bonsai'}>
            <span className="runtime-icon lime">🌳</span><span><b>Bonsai custom</b><small>27B · 1-bit GGUF · WebGPU</small></span><i>{loadingPlanner === 'bonsai' ? 'Loading' : 'Load'}</i>
          </button>
          <button className={`runtime-card ${planner === 'demo' ? 'selected' : ''}`} onClick={useDemoMode} disabled={loadingPlanner !== null} aria-pressed={planner === 'demo'}>
            <span className="runtime-icon">⚡</span><span><b>Demo rules</b><small>Universal fallback · zero download</small></span><i>Use</i>
          </button>
          <p className="runtime-note">{engineNote}</p>
          {planner === 'demo' && <button className="try-demo" onClick={() => {
            setPrompt('Add buy coffee filters');
            document.getElementById('agent-prompt')?.focus();
          }}>Try a demo request →</button>}
          <p className="download-warning">Custom models are fetched from Hugging Face only after confirmation. E4B is recommended for reliability; E2B is lighter and faster. Bonsai is ~3.8 GB and works best with at least 16 GB of GPU memory.</p>
          <details className="prompt-details"><summary>View agent system prompt</summary><code>{AGENT_SYSTEM_PROMPT}</code></details>
        </aside>
      </section>

      <section className="lower-grid">
        <div className="tasks-panel" data-testid="today-panel">
          <div className="section-heading">
            <div><p className="kicker">03 / Result</p><h2>Today <sup data-testid="today-open-count" aria-label={`${openCount} open tasks`}>{openCount} open</sup></h2></div>
            <div className="today-actions">
              <span className="task-totals" data-testid="today-total-count" aria-live="polite">{tasks.length} total · {completedCount} done</span>
              <button className="reset-data" onClick={resetLocalData} disabled={tasks.length === 0 && activity.length === 0 && scenarioState === 'idle'}>Reset local data</button>
            </div>
          </div>
          {tasks.length === 0 ? (
            <div className="empty"><span>＋</span><p>No tasks yet.</p><small>Ask the local agent or a WebMCP-aware browser agent to add one.</small></div>
          ) : (
            <ul className="task-list">
              {tasks.map((task) => (
                <li key={task.id} className={task.completed ? 'completed' : ''}>
                  <button aria-label={`Mark ${task.title} ${task.completed ? 'open' : 'complete'}`} onClick={() => {
                    setTasks((items) => {
                      const next = items.map((item) => item.id === task.id ? { ...item, completed: !item.completed } : item);
                      tasksRef.current = next;
                      return next;
                    });
                    log(`${task.completed ? 'Reopened' : 'Completed'} “${task.title}”.`);
                  }}>{task.completed ? '✓' : ''}</button>
                  <span><b>{task.title}</b><small>{task.priority} priority</small></span>
                  <i className={`priority ${task.priority}`} />
                </li>
              ))}
            </ul>
          )}
        </div>

        <aside className="tools-panel">
          <div className="section-heading">
            <div><p className="kicker">04 / Interface</p><h2>WebMCP tools</h2></div>
            <span className={`mcp-badge ${webMcp}`}>{webMcp === 'registered' ? 'native' : webMcp}</span>
          </div>
          <p className="tool-intro">These functions are shared by this app’s agent and exposed to compatible browser agents. Native write calls become proposals and require approval here.</p>
          <div className="tool-list">
            {tools.map((tool) => <div className="tool" key={tool.name}><code>{tool.name}</code><span>{tool.annotations?.readOnlyHint ? 'read' : 'write'}</span></div>)}
          </div>
          {webMcp === 'unavailable' && <p className="hint">Enable <code>chrome://flags/#enable-webmcp-testing</code> in a compatible Chrome build to expose them natively.</p>}
        </aside>
      </section>

      <section className="scenario-panel" aria-labelledby="scenario-title">
        <output hidden data-testid="scenario-trace" data-trace={JSON.stringify(scenarioTrace)} />
        <div className="scenario-heading">
          <div><p className="kicker">05 / Workflow lab</p><h2 id="scenario-title">Managed scenarios</h2></div>
          <span className={`scenario-state ${scenarioState}`}>{scenarioState}</span>
        </div>
        <div className="scenario-layout">
          <div className="scenario-picker" role="list" aria-label="Task workflow scenarios">
            {SCENARIOS.map((scenario) => (
              <button key={scenario.id} className={scenario.id === scenarioId ? 'selected' : ''} onClick={() => {
                setScenarioId(scenario.id);
                setScenarioState('idle');
                setScenarioStep(0);
                setScenarioResult('Choose Load scenario to prepare its controlled starting state.');
              }} aria-pressed={scenario.id === scenarioId}>
                <b>{scenario.name}</b><small>{scenario.purpose}</small>
              </button>
            ))}
          </div>
          <div className="scenario-detail">
            <p className="scenario-purpose">{activeScenario.purpose}</p>
            <ol>
              {activeScenario.requests.map((request, index) => (
                <li key={request} className={scenarioState === 'running' && scenarioStep === index + 1 ? 'running' : scenarioStep > index || scenarioState === 'passed' ? 'done' : ''}>
                  <span>{index + 1}</span><code>{request}</code>
                </li>
              ))}
            </ol>
            <div className="expected"><b>Expected state</b><span>{activeScenario.expected.total} total · {activeScenario.expected.open} open</span></div>
            <div className="scenario-actions">
              <button onClick={loadScenario} disabled={busy}>Load scenario</button>
              <button className="primary" onClick={runScenario} disabled={busy || scenarioState === 'idle' || scenarioState === 'running'}>Run workflow</button>
            </div>
            <p className={`scenario-result ${scenarioState}`} role="status" aria-live="polite">{scenarioResult}</p>
          </div>
        </div>
      </section>

      <section className="activity-panel">
        <div className="section-heading"><div><p className="kicker">06 / Audit</p><h2>Activity</h2></div><span className="memory-state">{memoryReady ? 'saved locally' : 'loading memory'}</span></div>
        {activity.length === 0 ? <p className="quiet">Tool calls will appear here with their source.</p> : (
          <ol>{activity.map((item) => <li key={item.id}><time>{item.at}</time><span>{item.source}</span><p>{item.message}</p></li>)}</ol>
        )}
      </section>

      <footer><span>Everything except the optional model download stays in this browser.</span><a href="https://huggingface.co/models?other=litert-lm">LiteRT models ↗</a><a href="https://huggingface.co/collections/prism-ml/bonsai">Bonsai models ↗</a><a href="https://github.com/webmachinelearning/webmcp">WebMCP draft ↗</a><a href="https://developers.google.com/edge/litert-lm/js">LiteRT-LM Web API ↗</a></footer>
    </main>
  );
}
