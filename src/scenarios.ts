import type { Task } from './types';

export type Scenario = {
  id: string;
  name: string;
  purpose: string;
  seed: Array<Pick<Task, 'title' | 'priority' | 'completed'>>;
  requests: string[];
  expected: {
    total: number;
    open: number;
    completedTitles?: string[];
    absentTitles?: string[];
    presentTitles?: string[];
  };
};

export const SCENARIOS: Scenario[] = [
  {
    id: 'daily-plan',
    name: 'Build a daily plan',
    purpose: 'Add prioritized work and inspect the resulting open list.',
    seed: [],
    requests: [
      'Add submit the expense report as high priority',
      'Add buy coffee filters',
      'Show my open tasks'
    ],
    expected: { total: 2, open: 2 }
  },
  {
    id: 'typo-completion',
    name: 'Recover from a typo',
    purpose: 'Complete the intended task despite a small spelling error.',
    seed: [
      { title: 'buy coffee filters', priority: 'medium', completed: false },
      { title: 'plan trip route', priority: 'medium', completed: false }
    ],
    requests: ['Mark "buy cofee filters as complete"'],
    expected: { total: 2, open: 1, completedTitles: ['buy coffee filters'] }
  },
  {
    id: 'safe-completion',
    name: 'Complete without collateral damage',
    purpose: 'Verify a completion request cannot trigger extra list cleanup.',
    seed: [
      { title: 'buy coffee filters', priority: 'medium', completed: false },
      { title: 'pack for trip', priority: 'high', completed: false },
      { title: 'get out of town', priority: 'high', completed: false }
    ],
    requests: ["Let's mark buy coffee filters complete"],
    expected: { total: 3, open: 2, completedTitles: ['buy coffee filters'] }
  },
  {
    id: 'clear-finished',
    name: 'Clear finished work',
    purpose: 'Remove completed items while preserving every open task.',
    seed: [
      { title: 'submit expense report', priority: 'high', completed: true },
      { title: 'buy coffee filters', priority: 'medium', completed: true },
      { title: 'plan trip route', priority: 'medium', completed: false }
    ],
    requests: ['Clear completed tasks'],
    expected: { total: 1, open: 1, absentTitles: ['submit expense report', 'buy coffee filters'] }
  },
  {
    id: 'natural-batch-completion',
    name: 'Understand completed work',
    purpose: 'Turn a natural past-tense update into multiple precise completions.',
    seed: [
      { title: 'buy apples', priority: 'medium', completed: false },
      { title: 'wash clothes', priority: 'medium', completed: false },
      { title: 'submit expense report', priority: 'high', completed: false }
    ],
    requests: ['When I was shopping I purchased apples, and later I washed clothes'],
    expected: { total: 3, open: 1, completedTitles: ['buy apples', 'wash clothes'] }
  },
  {
    id: 'boston-trip-breakdown',
    name: 'Break down a Boston trip',
    purpose: 'Turn one detailed paragraph into separate, useful preparation tasks.',
    seed: [],
    requests: ['Tomorrow I need to pack for my trip to boston, make sure I have my ticket for the train, pack clothes, and all my projects. Plus I need to have power and tools to support my project.'],
    expected: {
      total: 6,
      open: 6,
      presentTitles: [
        'pack for Boston trip',
        'bring train ticket',
        'pack clothes',
        'pack project materials',
        'bring project power supplies',
        'bring project tools'
      ]
    }
  }
];

export function seedScenario(scenario: Scenario): Task[] {
  return scenario.seed.map((task, index) => ({
    ...task,
    id: `scenario-${scenario.id}-${index}`,
    createdAt: new Date().toISOString()
  }));
}

export function evaluateScenario(scenario: Scenario, tasks: Task[]) {
  const failures: string[] = [];
  const open = tasks.filter((task) => !task.completed).length;
  if (tasks.length !== scenario.expected.total) failures.push(`Expected ${scenario.expected.total} total; found ${tasks.length}.`);
  if (open !== scenario.expected.open) failures.push(`Expected ${scenario.expected.open} open; found ${open}.`);
  for (const title of scenario.expected.completedTitles ?? []) {
    if (!tasks.some((task) => task.title === title && task.completed)) failures.push(`Expected “${title}” to be completed.`);
  }
  for (const title of scenario.expected.absentTitles ?? []) {
    if (tasks.some((task) => task.title === title)) failures.push(`Expected “${title}” to be removed.`);
  }
  for (const title of scenario.expected.presentTitles ?? []) {
    if (!tasks.some((task) => task.title === title)) failures.push(`Expected “${title}” to be present.`);
  }
  return failures;
}
