import { describe, expect, it } from 'vitest';
import { classifyBenchmarkScenario, countCorrectOutcomeSteps, countExactDecisionSteps, countExactToolSelectionSteps, countExactToolSequenceSteps, countSafeToolSteps, type BenchmarkCaseContract } from './benchmark';
import type { PlannerTraceEntry } from './types';

const clarificationCase: BenchmarkCaseContract = {
  id: 'ambiguous',
  name: 'Ambiguous completion',
  steps: [
    { status: 'clarification', calls: [] },
    { status: 'executed', calls: ['complete_task'] }
  ]
};

const entry = (values: Partial<PlannerTraceEntry>): PlannerTraceEntry => ({
  request: 'request',
  originalRequest: 'request',
  planner: 'litert',
  outcome: 'error',
  calls: [],
  message: '',
  status: 'failed',
  ...values
});

describe('benchmark v2 scoring', () => {
  it('does not credit a missing decision for an expected empty call sequence', () => {
    const trace = [entry({ message: 'The model returned an invalid plan.' })];
    expect(countExactToolSequenceSteps(trace, clarificationCase.steps)).toBe(1);
    expect(countExactDecisionSteps(trace, clarificationCase.steps)).toBe(0);
    expect(classifyBenchmarkScenario('failed', trace, clarificationCase)).toEqual([
      'invalid-structured-output',
      'missed-clarification',
      'missing-decision'
    ]);
  });

  it('separates unnecessary clarification from a wrong tool sequence', () => {
    const directCase: BenchmarkCaseContract = {
      id: 'direct', name: 'Direct completion', steps: [{ status: 'executed', calls: ['complete_task'] }]
    };
    const trace = [entry({ outcome: 'clarify', status: 'clarification', message: 'Which one?' })];
    expect(classifyBenchmarkScenario('failed', trace, directCase)).toEqual([
      'unnecessary-clarification',
      'wrong-tool-sequence'
    ]);
  });

  it('accepts an exact clarification and follow-up sequence', () => {
    const trace = [
      entry({ outcome: 'clarify', status: 'clarification' }),
      entry({ outcome: 'act', status: 'executed', calls: [{ name: 'complete_task', arguments: { task: 'submit report' } }] })
    ];
    expect(classifyBenchmarkScenario('passed', trace, clarificationCase)).toEqual([]);
    expect(countExactToolSequenceSteps(trace, clarificationCase.steps)).toBe(2);
    expect(countExactDecisionSteps(trace, clarificationCase.steps)).toBe(2);
  });

  it('scores outcome, selection, ordering, and unexpected calls independently', () => {
    const contract: BenchmarkCaseContract = {
      id: 'ordered', name: 'Ordered work', steps: [{ status: 'executed', calls: ['add_task', 'complete_task'] }]
    };
    const trace = [entry({
      outcome: 'act', status: 'executed',
      calls: [
        { name: 'complete_task', arguments: { task: 'one' } },
        { name: 'add_task', arguments: { title: 'one' } }
      ]
    })];
    expect(countCorrectOutcomeSteps(trace, contract.steps)).toBe(1);
    expect(countExactToolSelectionSteps(trace, contract.steps)).toBe(1);
    expect(countExactToolSequenceSteps(trace, contract.steps)).toBe(0);
    expect(countSafeToolSteps(trace, contract.steps)).toBe(1);
    trace[0].calls.push({ name: 'clear_completed', arguments: {} });
    expect(countSafeToolSteps(trace, contract.steps)).toBe(0);
  });
});
