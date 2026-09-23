import { describe, expect, it } from 'vitest';
import { postNeedleEscalation, preNeedleEscalation } from './hybrid';
import type { AgentPlan, Task } from './types';

const tasks: Task[] = [
  { id: 'coffee', title: 'buy coffee filters', priority: 'medium', completed: false, createdAt: '' },
  { id: 'submit', title: 'submit report', priority: 'medium', completed: false, createdAt: '' },
  { id: 'review', title: 'review report', priority: 'medium', completed: false, createdAt: '' },
  { id: 'done', title: 'pay rent', priority: 'medium', completed: true, createdAt: '' }
];
const act = (calls: AgentPlan['calls']): AgentPlan => ({ outcome: 'act', calls, message: '' });

describe('Needle-first routing', () => {
  it('sends follow-ups and long requests straight to the larger model', () => {
    expect(preNeedleEscalation('submit report', true)).toBe('follow-up');
    expect(preNeedleEscalation(Array.from({ length: 26 }, () => 'word').join(' '), false)).toBe('long-request');
    expect(preNeedleEscalation('Add buy milk', false)).toBeNull();
  });

  it('keeps confident plans for known or new tasks', () => {
    expect(postNeedleEscalation(act([{ name: 'add_task', arguments: { title: 'Buy milk' } }]), 0.99, tasks, 'Please do this')).toBeNull();
    expect(postNeedleEscalation(act([{ name: 'complete_task', arguments: { task: 'buy coffee filters' } }]), 1, tasks, 'Please do this')).toBeNull();
  });

  it('keeps an ambiguous completion so the guardrail can ask which task', () => {
    expect(postNeedleEscalation(act([{ name: 'complete_task', arguments: { task: 'report' } }]), 1, tasks, 'Please do this')).toBeNull();
  });

  it('escalates empty, uncertain, and ungrounded plans', () => {
    expect(postNeedleEscalation({ outcome: 'answer', calls: [], message: '' }, 1, tasks, 'Please do this')).toBe('no-calls');
    expect(postNeedleEscalation(act([{ name: 'add_task', arguments: { title: 'Trip' } }]), 0.59, tasks, 'Please do this')).toBe('low-confidence');
    expect(postNeedleEscalation(act([{ name: 'complete_task', arguments: { task: 'packed sldering iron' } }]), 0.99, tasks, 'I packed a sldering iron as well')).toBe('unknown-completion-target');
    expect(postNeedleEscalation(act([{ name: 'complete_task', arguments: { task: 'pay rent' } }]), 0.99, tasks, 'Please do this')).toBe('unknown-completion-target');
  });

  it('keeps Needle when the guardrails will decide the outcome anyway', () => {
    const typo = act([{ name: 'complete_task', arguments: { task: 'buy cofee filters as complete' } }]);
    expect(postNeedleEscalation(typo, 0.58, tasks, 'Mark "buy cofee filters as complete"')).toBeNull();
    expect(postNeedleEscalation({ outcome: 'answer', calls: [], message: '' }, 1, tasks, 'Email the submit report task to Alex')).toBeNull();
    expect(postNeedleEscalation(act([{ name: 'complete_task', arguments: { task: 'renew passport' } }]), 1, tasks, 'Complete renew passport')).toBeNull();
    expect(postNeedleEscalation(act([{ name: 'list_tasks', arguments: {} }]), 0.99, tasks, 'I finished all my tasks')).toBeNull();
  });
});
