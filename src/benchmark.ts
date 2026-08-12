import type { PlannerTraceEntry } from './types';

export type ExpectedBenchmarkStep = {
  status: 'executed' | 'answered' | 'clarification';
  calls: string[];
};

export type BenchmarkCaseContract = {
  id: string;
  name: string;
  steps: ExpectedBenchmarkStep[];
};

export function sameToolSequence(actual: string[], expected: string[]) {
  return actual.length === expected.length && actual.every((name, index) => name === expected[index]);
}

function sameToolSelection(actual: string[], expected: string[]) {
  return [...actual].sort().join('\n') === [...expected].sort().join('\n');
}

function hasNoUnexpectedCalls(actual: string[], expected: string[]) {
  const remaining = new Map<string, number>();
  expected.forEach((name) => remaining.set(name, (remaining.get(name) ?? 0) + 1));
  return actual.every((name) => {
    const count = remaining.get(name) ?? 0;
    if (count === 0) return false;
    remaining.set(name, count - 1);
    return true;
  });
}

function failureCategory(message: string) {
  return /json|schema|invalid plan|missing required argument|unknown tool|argument .* must/i.test(message)
    ? 'invalid-structured-output'
    : 'planner-error';
}

export function classifyBenchmarkScenario(status: string, trace: PlannerTraceEntry[], benchmarkCase: BenchmarkCaseContract) {
  const categories: string[] = [];
  benchmarkCase.steps.forEach((expected, index) => {
    const actual = trace[index];
    if (!actual) {
      categories.push('missing-decision');
      return;
    }
    if (actual.status === 'failed') categories.push(failureCategory(actual.message));
    if (actual.status !== expected.status) {
      if (actual.status === 'clarification') categories.push('unnecessary-clarification');
      else if (expected.status === 'clarification') categories.push('missed-clarification');
      else categories.push('wrong-outcome');
    }
    if (!sameToolSequence(actual.calls.map((call) => call.name), expected.calls)) categories.push('wrong-tool-sequence');
  });
  if (trace.length > benchmarkCase.steps.length) categories.push('unexpected-decision');
  if (status === 'stale') categories.push('state-race');
  if (status === 'failed' && categories.length === 0) categories.push('final-state-mismatch');
  return [...new Set(categories)];
}

export function countExactToolSequenceSteps(trace: PlannerTraceEntry[], expected: ExpectedBenchmarkStep[]) {
  return expected.filter((step, index) => {
    const actual = trace[index];
    return actual ? sameToolSequence(actual.calls.map((call) => call.name), step.calls) : false;
  }).length;
}

export function countExactDecisionSteps(trace: PlannerTraceEntry[], expected: ExpectedBenchmarkStep[]) {
  return expected.filter((step, index) => {
    const actual = trace[index];
    return actual
      ? actual.status === step.status && sameToolSequence(actual.calls.map((call) => call.name), step.calls)
      : false;
  }).length;
}

export function countCorrectOutcomeSteps(trace: PlannerTraceEntry[], expected: ExpectedBenchmarkStep[]) {
  return expected.filter((step, index) => trace[index]?.status === step.status).length;
}

export function countExactToolSelectionSteps(trace: PlannerTraceEntry[], expected: ExpectedBenchmarkStep[]) {
  return expected.filter((step, index) => {
    const actual = trace[index];
    return actual ? sameToolSelection(actual.calls.map((call) => call.name), step.calls) : false;
  }).length;
}

export function countSafeToolSteps(trace: PlannerTraceEntry[], expected: ExpectedBenchmarkStep[]) {
  return expected.filter((step, index) => {
    const actual = trace[index];
    return actual ? hasNoUnexpectedCalls(actual.calls.map((call) => call.name), step.calls) : false;
  }).length;
}
