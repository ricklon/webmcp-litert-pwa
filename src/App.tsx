import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AGENT_SYSTEM_PROMPT, authorizeToolPlan, getChromeModelAvailability, loadChromeModel, loadLiteRt, MODEL_URL, planDeterministically, planWithChrome, planWithLiteRt, unloadChromeModel, unloadLiteRt } from './agent';
import { createTools, executeLocalTool } from './tools';
import type { Activity, Task } from './types';
import { registerWebMcpTools, type WebMcpStatus } from './webmcp';
import { evaluateScenario, SCENARIOS, seedScenario } from './scenarios';

const TASK_KEY = 'local-tools-lab.tasks.v1';

function readTasks(): Task[] {
  try { return JSON.parse(localStorage.getItem(TASK_KEY) ?? '[]') as Task[]; }
  catch { return []; }
}

const time = () => new Intl.DateTimeFormat(undefined, { hour: 'numeric', minute: '2-digit' }).format(new Date());
type RunFeedback = { tone: 'idle' | 'working' | 'success' | 'error'; title: string; detail: string; tools?: string[] };

function describeToolResult(name: string, value: unknown) {
  const result = value as { task?: Task; count?: number; removed?: number };
  if (name === 'add_task' && result.task) return `Added “${result.task.title}”.`;
  if (name === 'complete_task' && result.task) return `Completed “${result.task.title}”.`;
  if (name === 'list_tasks') return `Found ${result.count ?? 0} matching task${result.count === 1 ? '' : 's'}.`;
  if (name === 'clear_completed') return `Removed ${result.removed ?? 0} completed task${result.removed === 1 ? '' : 's'}.`;
  return `${name} completed.`;
}

export default function App() {
  const [tasks, setTasks] = useState<Task[]>(readTasks);
  const tasksRef = useRef(tasks);
  const [activity, setActivity] = useState<Activity[]>([]);
  const [webMcp, setWebMcp] = useState<WebMcpStatus>('available');
  const [planner, setPlanner] = useState<'demo' | 'chrome' | 'litert'>('demo');
  const [loadingPlanner, setLoadingPlanner] = useState<'chrome' | 'litert' | null>(null);
  const [chromeAvailability, setChromeAvailability] = useState<'checking' | 'unavailable' | 'downloadable' | 'downloading' | 'available'>('checking');
  const [engineNote, setEngineNote] = useState('Checking for Chrome’s built-in model…');
  const [pwaReady, setPwaReady] = useState(false);
  const [prompt, setPrompt] = useState('');
  const [busy, setBusy] = useState(false);
  const [feedback, setFeedback] = useState<RunFeedback>({
    tone: 'idle',
    title: 'Ready for a request',
    detail: 'Choose an example or type a request, then press Run.'
  });
  const [scenarioId, setScenarioId] = useState(SCENARIOS[0].id);
  const [scenarioState, setScenarioState] = useState<'idle' | 'loaded' | 'running' | 'passed' | 'failed'>('idle');
  const [scenarioStep, setScenarioStep] = useState(0);
  const [scenarioResult, setScenarioResult] = useState('Choose Load scenario to replace the current task list with controlled test data.');

  useEffect(() => {
    tasksRef.current = tasks;
    localStorage.setItem(TASK_KEY, JSON.stringify(tasks));
  }, [tasks]);

  useEffect(() => {
    if (!('serviceWorker' in navigator)) return;
    navigator.serviceWorker.ready.then(() => setPwaReady(true));
  }, []);

  useEffect(() => {
    let active = true;
    getChromeModelAvailability().then(async (availability) => {
      if (!active) return;
      setChromeAvailability(availability);
      if (availability === 'available') {
        try {
          await loadChromeModel(setEngineNote);
          if (!active) return;
          setPlanner('chrome');
          setEngineNote('Chrome built-in model active · on device');
          setFeedback({ tone: 'success', title: 'Chrome’s model is ready', detail: 'Your next request will be planned locally with Gemini Nano.' });
        } catch (error) {
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
    setActivity((items) => [{ id: crypto.randomUUID(), source, message, at: time() }, ...items].slice(0, 30));
  }, []);

  const tools = useMemo(() => createTools({
    getTasks: () => tasksRef.current,
    setTasks: (updater) => setTasks((current) => {
      const next = updater(current);
      // Tool plans may execute more than one call before React runs effects.
      // Keep the imperative tool view and rendered state in lockstep.
      tasksRef.current = next;
      return next;
    }),
    log
  }), [log]);

  useEffect(() => {
    const controller = new AbortController();
    setWebMcp(document.modelContext ? 'available' : 'unavailable');
    registerWebMcpTools(tools, controller.signal).then(setWebMcp);
    return () => controller.abort();
  }, [tools]);

  async function activateModel() {
    if (!globalThis.confirm('Download the large Gemma 4 E2B model and run it locally with WebGPU?')) return;
    setLoadingPlanner('litert');
    setFeedback({ tone: 'working', title: 'Preparing LiteRT-LM', detail: 'Downloading and compiling Gemma 4 E2B. This can take a while.' });
    try {
      unloadChromeModel();
      await loadLiteRt(MODEL_URL, setEngineNote);
      setPlanner('litert');
      setEngineNote('Gemma 4 E2B · WebGPU · on device');
      log('LiteRT-LM model is ready.', 'local agent');
      setFeedback({ tone: 'success', title: 'LiteRT-LM is ready', detail: 'Your next request will use Gemma 4 E2B locally through WebGPU.' });
    } catch (error) {
      setEngineNote(error instanceof Error ? error.message : 'Model failed to load.');
      setFeedback({ tone: 'error', title: 'LiteRT-LM could not start', detail: error instanceof Error ? error.message : 'Model failed to load.' });
    } finally {
      setLoadingPlanner(null);
    }
  }

  async function activateChromeModel() {
    setLoadingPlanner('chrome');
    setFeedback({ tone: 'working', title: 'Preparing Chrome’s model', detail: 'Chrome may download Gemini Nano before creating the local session.' });
    try {
      await unloadLiteRt();
      await loadChromeModel(setEngineNote);
      setPlanner('chrome');
      setChromeAvailability('available');
      setEngineNote('Chrome built-in model active · on device');
      log('Chrome’s built-in model is ready.', 'local agent');
      setFeedback({ tone: 'success', title: 'Chrome’s model is ready', detail: 'Your next request will be planned locally with Gemini Nano.' });
    } catch (error) {
      setEngineNote(error instanceof Error ? error.message : 'Chrome’s model failed to load.');
      setFeedback({ tone: 'error', title: 'Chrome’s model could not start', detail: error instanceof Error ? error.message : 'The built-in model failed to load.' });
    } finally {
      setLoadingPlanner(null);
    }
  }

  async function useDemoMode() {
    await unloadLiteRt();
    unloadChromeModel();
    setPlanner('demo');
    setEngineNote('Demo agent active. Enter a request under “Local agent” and press Run.');
    log('Demo agent selected. It is ready for a request.', 'local agent');
    setFeedback({ tone: 'success', title: 'Demo rules selected', detail: 'Type a supported task request and press Run.' });
  }

  async function executeRequest(request: string) {
    log(request, 'person');
    setFeedback({ tone: 'working', title: `${planner === 'chrome' ? 'Chrome’s model' : planner === 'litert' ? 'LiteRT-LM' : 'Demo rules'} is planning`, detail: `Reading: “${request}”` });
    try {
      const proposedPlan = planner === 'chrome'
        ? await planWithChrome(request, tools, tasksRef.current)
        : planner === 'litert'
          ? await planWithLiteRt(request, tools, tasksRef.current)
          : planDeterministically(request, tasksRef.current);
      const plan = authorizeToolPlan(proposedPlan, tools);
      const results: Array<{ name: string; value: unknown }> = [];
      for (const call of plan.calls) {
        setFeedback({ tone: 'working', title: `Calling ${call.name}`, detail: `Using ${JSON.stringify(call.arguments)}`, tools: plan.calls.map((item) => item.name) });
        const result = await executeLocalTool(tools, call, 'local agent');
        if (result && typeof result === 'object' && 'ok' in result && result.ok === false) {
          const failure = result as { error?: string; cancelled?: boolean };
          throw new Error(failure.cancelled ? 'Action cancelled.' : failure.error ?? `${call.name} failed.`);
        }
        results.push({ name: call.name, value: result });
      }
      const reply = results.length
        ? results.map((result) => describeToolResult(result.name, result.value)).join(' ')
        : `No app tool was executed. Model response: ${plan.reply}`;
      const detail = results.length ? ` ${results.map((result) => JSON.stringify(result.value)).join(' ')}` : '';
      log(`${reply}${detail}`, 'local agent');
      setFeedback({
        tone: 'success',
        title: plan.calls.length ? 'Request completed' : 'No action taken',
        detail: reply,
        tools: plan.calls.map((call) => call.name)
      });
      return true;
    } catch (error) {
      log(error instanceof Error ? error.message : 'The agent could not finish that request.', 'local agent');
      setFeedback({ tone: 'error', title: 'Request failed', detail: error instanceof Error ? error.message : 'The agent could not finish that request.' });
      return false;
    }
  }

  async function submitPrompt(event: FormEvent) {
    event.preventDefault();
    const request = prompt.trim();
    if (!request || busy) return;
    setPrompt('');
    setBusy(true);
    await executeRequest(request);
    setBusy(false);
  }

  const activeScenario = SCENARIOS.find((scenario) => scenario.id === scenarioId)!;

  function loadScenario() {
    if (tasksRef.current.length > 0 && !globalThis.confirm('Replace the current task list with this scenario’s controlled starting state?')) return;
    const seeded = seedScenario(activeScenario);
    tasksRef.current = seeded;
    setTasks(seeded);
    setScenarioState('loaded');
    setScenarioStep(0);
    setScenarioResult('Starting state loaded. Run the workflow when ready.');
    setFeedback({ tone: 'success', title: `Loaded “${activeScenario.name}”`, detail: `${seeded.length} seed task${seeded.length === 1 ? '' : 's'} ready.` });
  }

  async function runScenario() {
    if (scenarioState !== 'loaded' && scenarioState !== 'failed' && scenarioState !== 'passed') return;
    setBusy(true);
    setScenarioState('running');
    setScenarioResult(`Running with ${planner === 'chrome' ? 'Chrome built-in AI' : planner === 'litert' ? 'LiteRT-LM' : 'demo rules'}…`);
    let requestFailed = false;
    for (let index = 0; index < activeScenario.requests.length; index += 1) {
      setScenarioStep(index + 1);
      if (!await executeRequest(activeScenario.requests[index])) {
        requestFailed = true;
        break;
      }
    }
    const failures = requestFailed ? ['A workflow request failed.'] : evaluateScenario(activeScenario, tasksRef.current);
    setScenarioState(failures.length ? 'failed' : 'passed');
    setScenarioResult(failures.length ? failures.join(' ') : `Passed all ${activeScenario.requests.length} step${activeScenario.requests.length === 1 ? '' : 's'} and state checks.`);
    setBusy(false);
  }

  function resetLocalData() {
    if (!globalThis.confirm('Reset all Local Tools Lab tasks, activity, and scenario progress on this device? This cannot be undone.')) return;
    tasksRef.current = [];
    setTasks([]);
    setActivity([]);
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
        <p className="lede">Plan your day in natural language. The model runs on your device; the actions are explicit, inspectable WebMCP tools.</p>
        <div className="architecture" aria-label="Application architecture">
          <span>You</span><b>→</b><span className={planner !== 'demo' ? 'active' : ''}>{planner === 'chrome' ? 'Chrome model' : planner === 'litert' ? 'LiteRT model' : 'Demo rules'}</span><b>→</b><span className={webMcp === 'registered' ? 'active' : ''}>Page tools</span><b>→</b><span>Tasks</span>
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
          <form className="prompt-box" onSubmit={submitPrompt}>
            <label htmlFor="agent-prompt">What should we get done?</label>
            <textarea id="agent-prompt" value={prompt} onChange={(event) => setPrompt(event.target.value)} onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
                event.preventDefault();
                event.currentTarget.form?.requestSubmit();
              }
            }} placeholder="Add submit the expense report as high priority…" rows={4} />
            <div className="prompt-footer">
              <span>{busy ? 'Planning and calling tools…' : `Enter to run · Shift+Enter for a new line · ${planner === 'chrome' ? 'Chrome AI' : planner === 'litert' ? 'LiteRT-LM' : 'Demo rules'}`}</span>
              <button type="submit" disabled={busy || !prompt.trim()}>Run <span>↗</span></button>
            </div>
          </form>
          <div className="examples">
            {['Add buy coffee filters', 'Show my open tasks', 'Complete buy coffee filters'].map((example) => (
              <button key={example} onClick={() => setPrompt(example)}>{example}</button>
            ))}
          </div>
          <div className={`run-feedback ${feedback.tone}`} role="status" aria-live="polite" aria-atomic="true">
            <span className="feedback-signal">{feedback.tone === 'working' ? '···' : feedback.tone === 'success' ? '✓' : feedback.tone === 'error' ? '!' : '→'}</span>
            <div><b>{feedback.title}</b><p>{feedback.detail}</p>
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
          <button className={`runtime-card ${planner === 'litert' ? 'selected' : ''}`} onClick={activateModel} disabled={loadingPlanner !== null} aria-pressed={planner === 'litert'}>
            <span className="runtime-icon lime">◆</span><span><b>LiteRT-LM custom</b><small>Gemma 4 E2B · WebGPU</small></span><i>{loadingPlanner === 'litert' ? 'Loading' : 'Load'}</i>
          </button>
          <button className={`runtime-card ${planner === 'demo' ? 'selected' : ''}`} onClick={useDemoMode} aria-pressed={planner === 'demo'}>
            <span className="runtime-icon">⚡</span><span><b>Demo rules</b><small>Universal fallback · zero download</small></span><i>Use</i>
          </button>
          <p className="runtime-note">{engineNote}</p>
          {planner === 'demo' && <button className="try-demo" onClick={() => {
            setPrompt('Add buy coffee filters');
            document.getElementById('agent-prompt')?.focus();
          }}>Try a demo request →</button>}
          <p className="download-warning">The preview model is large and fetched from Hugging Face. Load it only on a suitable connection.</p>
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
          <p className="tool-intro">These functions are shared by this app’s agent and exposed to compatible browser agents.</p>
          <div className="tool-list">
            {tools.map((tool) => <div className="tool" key={tool.name}><code>{tool.name}</code><span>{tool.annotations?.readOnlyHint ? 'read' : 'write'}</span></div>)}
          </div>
          {webMcp === 'unavailable' && <p className="hint">Enable <code>chrome://flags/#enable-webmcp-testing</code> in a compatible Chrome build to expose them natively.</p>}
        </aside>
      </section>

      <section className="scenario-panel" aria-labelledby="scenario-title">
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
        <div className="section-heading"><div><p className="kicker">06 / Audit</p><h2>Activity</h2></div></div>
        {activity.length === 0 ? <p className="quiet">Tool calls will appear here with their source.</p> : (
          <ol>{activity.map((item) => <li key={item.id}><time>{item.at}</time><span>{item.source}</span><p>{item.message}</p></li>)}</ol>
        )}
      </section>

      <footer><span>Everything except the optional model download stays in this browser.</span><a href="https://github.com/webmachinelearning/webmcp">WebMCP draft ↗</a><a href="https://developers.google.com/edge/litert-lm/js">LiteRT-LM Web API ↗</a></footer>
    </main>
  );
}
