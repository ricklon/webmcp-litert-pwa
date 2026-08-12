import { describe, expect, it } from 'vitest';
import { deduplicateActivity } from './memory';
import type { Activity } from './types';

describe('conversation memory', () => {
  it('removes only exact duplicate events from earlier Strict Mode logging', () => {
    const event: Activity = {
      id: 'first', source: 'person', message: 'What should I pack?', at: '3:55 PM', createdAt: '2026-07-25T19:55:00.000Z'
    };
    const sameEvent = { ...event, id: 'duplicate' };
    const intentionalRepeat = { ...event, id: 'later', createdAt: '2026-07-25T19:56:00.000Z' };

    expect(deduplicateActivity([event, sameEvent, intentionalRepeat]).map((item) => item.id))
      .toEqual(['first', 'later']);
  });
});
