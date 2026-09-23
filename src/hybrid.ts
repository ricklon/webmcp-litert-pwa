import { enforceExplicitBulkCompletion, enforceSafetyGuardrails } from './agent';
import { findTask } from './tools';
import type { AgentPlan, Task } from './types';

// Needle 3 is fast and reliable on short, direct commands. These limits come
// from the Needle benchmark: long stories, low-confidence plans, follow-ups, and
// completions of tasks that do not exist are where it fails.
export const NEEDLE_MAX_WORDS = 25;
export const NEEDLE_MIN_CONFIDENCE = 0.7;

export type EscalationReason = 'follow-up' | 'long-request' | 'no-calls' | 'low-confidence' | 'unknown-completion-target';

/** Reasons to skip Needle before calling it. */
export function preNeedleEscalation(request: string, isFollowUp: boolean): EscalationReason | null {
  if (isFollowUp) return 'follow-up';
  const words = request.trim().split(/\s+/).filter(Boolean).length;
  return words > NEEDLE_MAX_WORDS ? 'long-request' : null;
}

/**
 * True when the app's deterministic guardrails will replace the model's plan
 * for this request anyway, so asking a larger model would only add latency.
 */
export function guardrailsDecide(plan: AgentPlan, request: string, tasks: Task[]) {
  return enforceSafetyGuardrails(plan, request, tasks).interventions.length > 0
    || enforceExplicitBulkCompletion(plan, request, tasks) !== plan;
}

/** Reasons to hand Needle's plan to the larger model instead of using it. */
export function postNeedleEscalation(plan: AgentPlan, confidence: number | undefined, tasks: Task[], request: string): EscalationReason | null {
  if (guardrailsDecide(plan, request, tasks)) return null;
  if (plan.calls.length === 0) return 'no-calls';
  if (confidence !== undefined && confidence < NEEDLE_MIN_CONFIDENCE) return 'low-confidence';
  const openTasks = tasks.filter((task) => !task.completed);
  const unknownTarget = plan.calls.some((call) => {
    if (call.name !== 'complete_task') return false;
    const target = findTask(openTasks, call.arguments.task ?? call.arguments.title ?? call.arguments.id);
    return !target.task && target.candidates.length === 0;
  });
  return unknownTarget ? 'unknown-completion-target' : null;
}
