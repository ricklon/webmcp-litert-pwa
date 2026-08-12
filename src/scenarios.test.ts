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

  it('requires realistic compound outcomes to be represented by distinct tasks', () => {
    const scenario = SCENARIOS.find((item) => item.id === 'event-trip-plan')!;
    const combined = [{
      id: 'one', title: 'Pack robot voice agent USB cables and power, then go to station and take Lyft to Airbnb',
      priority: 'medium' as const, completed: false, createdAt: ''
    }];
    const failures = evaluateScenario(scenario, combined);
    expect(failures).toContainEqual(expect.stringContaining('Expected 6 total'));
    expect(failures).toContainEqual(expect.stringContaining('Expected distinct tasks'));
  });

  it('checks expected priority mappings', () => {
    const scenario = SCENARIOS.find((item) => item.id === 'priority-mapping')!;
    const tasks = [{ id: 'one', title: 'book a dentist appointment', priority: 'medium' as const, completed: false, createdAt: '' }];
    expect(evaluateScenario(scenario, tasks)).toContain('Expected “book a dentist appointment” to have high priority.');
  });
});
