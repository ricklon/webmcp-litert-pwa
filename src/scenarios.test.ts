import { describe, expect, it } from 'vitest';
import { evaluateScenario, SCENARIOS, seedScenario } from './scenarios';

describe('scenario definitions', () => {
  it('have unique ids and at least one request', () => {
    expect(new Set(SCENARIOS.map((scenario) => scenario.id)).size).toBe(SCENARIOS.length);
    expect(SCENARIOS.every((scenario) => scenario.requests.length > 0)).toBe(true);
  });

  it('reports expectation failures clearly', () => {
    const scenario = SCENARIOS.find((item) => item.id === 'typo-completion')!;
    expect(evaluateScenario(scenario, seedScenario(scenario))).toContain('Expected 1 open; found 2.');
  });
});
